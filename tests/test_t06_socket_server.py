"""
Test T06 — Socket Protocol and Server Resilience for SepsisSignal

Protocol:
  Input:  H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>\\n
  Output: R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>\\n
  Error:  E|<reason>\\n

Required Test Cases:
1. Valid complete observation produces a correctly formatted response.
2. Empty fields, including trailing empty fields, are parsed as missing rather than omitted or shifted.
3. LF and CRLF line endings are handled correctly.
4. Multiple valid requests sent in one TCP write are processed as separate messages.
5. A request split across several TCP writes is handled correctly.
6. Malformed field counts and invalid field values produce the documented error behavior.
7. An invalid request does not terminate the server or poison the following valid request.
8. Out-of-order observations preserve the established causal feature-extraction behavior.
9. Repeated requests do not create duplicate prediction rows or rewrite existing prediction records.
10. A disconnect or incomplete message does not crash the server or leave persistent resources improperly managed.
11. Unseen patient IDs work seamlessly without any dataset lookup.
12. Concurrent client connections are handled safely without cross-talk or server degradation.

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import asyncio
import os
import socket
import sys
import tempfile
import threading
import time
import unittest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import handle_client, parse_and_process_line
from src.store import get_connection, init_db


class ManagedTCPServer:
    """
    Spins up the actual SepsisSignal asyncio TCP line server on an ephemeral OS-assigned port
    in a dedicated daemon thread. Ensures clean lifecycle management and teardown.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.loop = asyncio.new_event_loop()
        self.server = None
        self.port = None
        self.started_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        asyncio.set_event_loop(self.loop)

        async def _main():
            self.server = await asyncio.start_server(
                lambda r, w: handle_client(r, w, db_path=self.db_path),
                "127.0.0.1",
                0,
            )
            self.port = self.server.sockets[0].getsockname()[1]
            self.started_event.set()
            async with self.server:
                await self.server.serve_forever()

        try:
            self.loop.run_until_complete(_main())
        except asyncio.CancelledError:
            pass
        finally:
            self.loop.close()

    def start(self):
        self.thread.start()
        if not self.started_event.wait(timeout=5.0):
            raise RuntimeError("TCPServer failed to start within 5.0 seconds")

    def stop(self):
        if self.server:
            self.loop.call_soon_threadsafe(self.server.close)

        def _cancel():
            for task in asyncio.all_tasks(self.loop):
                task.cancel()

        self.loop.call_soon_threadsafe(_cancel)
        self.thread.join(timeout=3.0)


def read_exact_lines(sock: socket.socket, count: int = 1, timeout: float = 3.0) -> list[str]:
    """Read exactly `count` newline-terminated lines from a socket."""
    sock.settimeout(timeout)
    lines = []
    buf = ""
    while len(lines) < count:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            lines.append(line + "\n")
            if len(lines) == count:
                break
    return lines


class TestT06SocketProtocolAndResilience(unittest.TestCase):
    """
    Exhaustive suite testing the real TCP line server, request parser,
    immutability, and resilience across 12 distinct test cases.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "t06_test.db")
        init_db(self.db_path)
        self.server = ManagedTCPServer(self.db_path)
        self.server.start()
        self.host = "127.0.0.1"
        self.port = self.server.port

    def tearDown(self):
        self.server.stop()
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _connect(self, timeout: float = 3.0) -> socket.socket:
        """Helper to create a configured client socket connected to the test server."""
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.settimeout(timeout)
        return s

    def test_case_1_valid_complete_observation(self):
        """
        1. Valid complete observation produces a correctly formatted response:
        R|<patient id>|<hour>|<probability 4 decimals>|<alert 0 or 1>|<count of empty fields>\\n
        """
        with self._connect() as s:
            s.sendall(b"H|P101|1|76.0|115.0|16.0|36.8|7.2\n")
            lines = read_exact_lines(s, 1)
            self.assertEqual(len(lines), 1)
            reply = lines[0]
            self.assertTrue(reply.startswith("R|P101|1|"))
            tokens = reply.strip().split("|")
            self.assertEqual(len(tokens), 6, f"Expected 6 tokens in reply, got {tokens}")
            self.assertEqual(tokens[0], "R")
            self.assertEqual(tokens[1], "P101")
            self.assertEqual(tokens[2], "1")

            # Check probability formatting (4 decimals)
            prob_token = tokens[3]
            self.assertIn(".", prob_token)
            self.assertEqual(len(prob_token.split(".")[1]), 4)
            prob = float(prob_token)
            self.assertTrue(0.0 <= prob <= 1.0)

            # Check alert is 0 or 1
            alert_token = tokens[4]
            self.assertIn(alert_token, ["0", "1"])

            # Check empty count is 0
            self.assertEqual(tokens[5], "0")

    def test_case_2_empty_and_trailing_fields(self):
        """
        2. Empty fields, including trailing empty fields, are parsed as missing rather than
        omitted or shifted into the wrong signal.
        """
        with self._connect() as s:
            # Subcase 2A: Trailing empty fields (Resp, Temp, WBC empty) -> 3 empty fields
            s.sendall(b"H|P102|1|88|112|||\n")
            lines = read_exact_lines(s, 1)
            self.assertEqual(len(lines), 1)
            reply_a = lines[0]
            tokens_a = reply_a.strip().split("|")
            self.assertEqual(tokens_a[5], "3", "Trailing empty fields must yield empty_count 3")

            # Subcase 2B: Middle empty fields (HR, Resp, WBC empty) -> 3 empty fields
            s.sendall(b"H|P102|2||110||36.9|\n")
            lines = read_exact_lines(s, 1)
            self.assertEqual(len(lines), 1)
            reply_b = lines[0]
            tokens_b = reply_b.strip().split("|")
            self.assertEqual(tokens_b[5], "3", "Middle empty fields must yield empty_count 3")

            # Subcase 2C: All 5 signal fields empty -> 5 empty fields
            s.sendall(b"H|P102|3|||||\n")
            lines = read_exact_lines(s, 1)
            self.assertEqual(len(lines), 1)
            reply_c = lines[0]
            tokens_c = reply_c.strip().split("|")
            self.assertEqual(tokens_c[5], "5", "All missing signals must yield empty_count 5")

        # Verify underlying database observations preserved correct signal assignments
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT hr, sbp, resp, temp, wbc FROM observations WHERE patient_id = 'P102' AND hour = 1")
            row1 = cur.fetchone()
            self.assertEqual(row1["hr"], 88.0)
            self.assertEqual(row1["sbp"], 112.0)
            self.assertIsNone(row1["resp"])
            self.assertIsNone(row1["temp"])
            self.assertIsNone(row1["wbc"])

            cur.execute("SELECT hr, sbp, resp, temp, wbc FROM observations WHERE patient_id = 'P102' AND hour = 2")
            row2 = cur.fetchone()
            self.assertIsNone(row2["hr"])
            self.assertEqual(row2["sbp"], 110.0)
            self.assertIsNone(row2["resp"])
            self.assertEqual(row2["temp"], 36.9)
            self.assertIsNone(row2["wbc"])

    def test_case_3_line_endings_lf_and_crlf(self):
        """
        3. LF and CRLF line endings are handled correctly without retaining \\r in tokens
        or corrupting downstream parsing.
        """
        with self._connect() as s:
            # CRLF
            s.sendall(b"H|P103|1|80|120|18|36.6|8.0\r\n")
            line_crlf = read_exact_lines(s, 1)[0]
            self.assertTrue(line_crlf.endswith("\n"))
            self.assertFalse(line_crlf.endswith("\r\n"))
            tokens_crlf = line_crlf.strip().split("|")
            self.assertEqual(tokens_crlf[1], "P103")
            self.assertEqual(tokens_crlf[5], "0")

            # LF
            s.sendall(b"H|P103|2|82|118|17|36.7|7.8\n")
            line_lf = read_exact_lines(s, 1)[0]
            self.assertTrue(line_lf.endswith("\n"))
            self.assertFalse(line_lf.endswith("\r\n"))
            tokens_lf = line_lf.strip().split("|")
            self.assertEqual(tokens_lf[1], "P103")
            self.assertEqual(tokens_lf[5], "0")

    def test_case_4_multiple_requests_single_tcp_write(self):
        """
        4. Multiple valid requests sent in one TCP write are processed as separate messages.
        """
        payload = (
            b"H|P104|1|80|120|18|36.5|8.0\n"
            b"H|P104|2|85|118|19|36.8|8.5\n"
            b"H|P104|3|90|115|20|37.0|9.0\n"
        )
        with self._connect() as s:
            s.sendall(payload)
            lines = read_exact_lines(s, 3)
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("R|P104|1|"))
            self.assertTrue(lines[1].startswith("R|P104|2|"))
            self.assertTrue(lines[2].startswith("R|P104|3|"))

    def test_case_5_request_split_across_multiple_tcp_writes(self):
        """
        5. A request split across several TCP writes (chunked arrival) is handled correctly.
        """
        with self._connect() as s:
            s.sendall(b"H|P105|1|80|12")
            time.sleep(0.04)
            s.sendall(b"0|18|")
            time.sleep(0.04)
            s.sendall(b"36.5|8.0\n")

            lines = read_exact_lines(s, 1)
            self.assertEqual(len(lines), 1)
            reply = lines[0]
            self.assertTrue(reply.startswith("R|P105|1|"))
            tokens = reply.strip().split("|")
            self.assertEqual(tokens[5], "0")

    def test_case_6_malformed_field_counts_and_invalid_values(self):
        """
        6. Malformed field counts and invalid field values produce the documented E|<reason>\\n.
        """
        test_inputs = [
            ("H|P106|1|80|120\n", "too few fields (< 8)"),
            ("H|P106|1|80|120|18|36.5|8.0|EXTRA\n", "too many fields (> 8)"),
            ("BADHEADER|P106|1|80|120|18|36.5|8.0\n", "unrecognized header"),
            ("H||1|80|120|18|36.5|8.0\n", "empty patient ID"),
            ("H|P106|not_an_int|80|120|18|36.5|8.0\n", "non-integer hour"),
            ("H|P106|-1|80|120|18|36.5|8.0\n", "negative hour"),
            ("H|P106|1|not_a_float|120|18|36.5|8.0\n", "non-numeric vital value"),
            ("\n", "empty line LF"),
            ("\r\n", "empty line CRLF"),
        ]
        with self._connect() as s:
            for raw_cmd, desc in test_inputs:
                s.sendall(raw_cmd.encode("utf-8"))
                lines = read_exact_lines(s, 1)
                self.assertEqual(len(lines), 1, f"Failed to get reply for: {desc}")
                reply = lines[0]
                self.assertTrue(
                    reply.startswith("E|"),
                    f"Expected 'E|' for {desc} ('{raw_cmd.strip()}'), got: '{reply.strip()}'",
                )

    def test_case_7_error_does_not_poison_subsequent_requests(self):
        """
        7. An invalid request does not terminate the server or poison the following valid request.
        """
        with self._connect() as s:
            # Send invalid request
            s.sendall(b"H|P107|invalid_hour|80|120|18|36.5|8.0\n")
            line_err = read_exact_lines(s, 1)[0]
            self.assertTrue(line_err.startswith("E|"))

            # Send valid request immediately on same socket connection
            s.sendall(b"H|P107|1|80|120|18|36.5|8.0\n")
            line_ok = read_exact_lines(s, 1)[0]
            self.assertTrue(line_ok.startswith("R|P107|1|"))

        # Verify a new client connection can also talk to server normally
        with self._connect() as s2:
            s2.sendall(b"H|P107|2|82|118|17|36.7|7.8\n")
            line_ok2 = read_exact_lines(s2, 1)[0]
            self.assertTrue(line_ok2.startswith("R|P107|2|"))

    def test_case_8_out_of_order_causal_preservation(self):
        """
        8. Out-of-order observations preserve the established causal feature-extraction behavior.
        Hour 5 extreme vitals do not leak into Hour 2 prediction when Hour 2 arrives later.
        """
        with self._connect() as s:
            # Hour 1 arrives (baseline)
            s.sendall(b"H|P108|1|75|120|16|36.6|7.0\n")
            line_h1 = read_exact_lines(s, 1)[0]
            self.assertTrue(line_h1.startswith("R|P108|1|"))

            # Hour 5 arrives with acute septic shock vitals
            s.sendall(b"H|P108|5|140|70|35|40.0|25.0\n")
            line_h5 = read_exact_lines(s, 1)[0]
            tokens_h5 = line_h5.strip().split("|")
            self.assertEqual(tokens_h5[4], "1", "Hour 5 acute vitals should trigger alert=1")

            # Hour 2 arrives out-of-order with normal vitals
            s.sendall(b"H|P108|2|78|118|16|36.7|7.2\n")
            line_h2 = read_exact_lines(s, 1)[0]
            tokens_h2 = line_h2.strip().split("|")

            prob_h2 = float(tokens_h2[3])
            alert_h2 = int(tokens_h2[4])
            # Strict causal verification: Hour 5 must not leak into Hour 2
            self.assertTrue(
                prob_h2 < 0.8,
                f"Hour 5 vitals leaked into Hour 2! Probability is {prob_h2}",
            )
            self.assertEqual(alert_h2, 0, "Hour 2 should not alert based on future vitals")

    def test_case_9_repeated_requests_immutability(self):
        """
        9. Repeated requests do not create duplicate prediction rows or rewrite existing
        prediction records.
        """
        with self._connect() as s:
            # First request
            s.sendall(b"H|P109|1|80|120|18|36.8|8.0\n")
            line_1 = read_exact_lines(s, 1)[0]
            self.assertTrue(line_1.startswith("R|P109|1|"))

            # Allow time passage so any re-creation would produce different created_at
            time.sleep(0.02)

            # Repeated request with conflicting vital values
            s.sendall(b"H|P109|1|150|60|35|40.5|30.0\n")
            line_2 = read_exact_lines(s, 1)[0]

            # Responses must be byte-for-byte identical
            self.assertEqual(line_1, line_2)

        # Database record check
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM predictions WHERE patient_id = 'P109' AND hour = 1")
            cnt = cur.fetchone()["cnt"]
            self.assertEqual(cnt, 1, "Must contain exactly one prediction record")

            cur.execute("SELECT COUNT(*) as cnt FROM observations WHERE patient_id = 'P109' AND hour = 1")
            obs_cnt = cur.fetchone()["cnt"]
            self.assertEqual(obs_cnt, 1, "Must contain exactly one observation record")

    def test_case_10_disconnect_and_incomplete_messages_resilience(self):
        """
        10. A disconnect or incomplete message does not crash the server or leave persistent
        resources improperly managed.
        """
        # Scenario A: Connect, send partial incomplete line, abruptly close socket
        s_partial = self._connect()
        s_partial.sendall(b"H|P110|1|80|120")
        s_partial.close()

        # Scenario B: Connect and immediately close without sending anything
        s_empty = self._connect()
        s_empty.close()

        # Scenario C: Connect normally right after and verify server is fully operational
        with self._connect() as s_normal:
            s_normal.sendall(b"H|P110|1|80|120|18|36.8|8.0\n")
            line = read_exact_lines(s_normal, 1)[0]
            self.assertTrue(line.startswith("R|P110|1|"))

    def test_case_11_unseen_patient_seamless_processing(self):
        """
        11. Unseen patient IDs work seamlessly over socket without external dataset lookup.
        """
        with self._connect() as s:
            s.sendall(b"H|PATIENT_UNSEEN_ZERO_DAY|1|84|118|18|36.8|7.5\n")
            line = read_exact_lines(s, 1)[0]
            self.assertTrue(line.startswith("R|PATIENT_UNSEEN_ZERO_DAY|1|"))
            tokens = line.strip().split("|")
            self.assertEqual(len(tokens), 6)
            prob = float(tokens[3])
            self.assertTrue(0.0 <= prob <= 1.0)

    def test_case_12_concurrent_client_connections(self):
        """
        12. Multiple concurrent client connections are handled safely without cross-talk.
        """
        results = {}

        def client_worker(client_id: int):
            with self._connect() as s:
                pid = f"CONCUR_P{client_id}"
                line = f"H|{pid}|1|80|120|18|36.8|8.0\n"
                s.sendall(line.encode("utf-8"))
                resp = read_exact_lines(s, 1)[0]
                results[client_id] = resp

        threads = [threading.Thread(target=client_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        self.assertEqual(len(results), 5)
        for i in range(5):
            self.assertTrue(results[i].startswith(f"R|CONCUR_P{i}|1|"))


if __name__ == "__main__":
    unittest.main()
