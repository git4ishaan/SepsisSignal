"""
TCP Line Server for SepsisSignal (Jury Protocol).
One line in, one line out.
Strict adherence to DATA_CONTRACT.md:
  Input:  H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>
  Output: R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>
  Error:  E|<reason>
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional, Tuple

from src.predict import predict
from src.store import init_db, save_observation

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sepsis_server")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


def parse_and_process_line(raw_line: str, db_path: Optional[str] = None) -> str:
    """
    Parse a single line from the TCP stream and process it.
    Guarantees:
    - Never strips trailing pipes (split on '|' directly).
    - Accepts \\r\\n and \\n.
    - Counts empty signal fields on the incoming record before imputation.
    - Handles malformed lines by returning 'E|<reason>\\n' without crashing.
    - Causal persistence & prediction: stores observation, queries predict(), replies 'R|...\\n'.
    """
    # Strip only line-endings (CR, LF), preserving trailing pipes
    line = raw_line.rstrip("\r\n")
    if not line:
        return "E|empty line received\n"

    tokens = line.split("|")
    if len(tokens) != 8:
        return f"E|malformed line: expected 8 fields separated by |, got {len(tokens)}\n"

    header = tokens[0]
    if header != "H":
        return f"E|malformed header: expected 'H', got '{header}'\n"

    patient_id = tokens[1].strip()
    if not patient_id:
        return "E|invalid patient id: cannot be empty\n"

    hour_str = tokens[2].strip()
    try:
        hour = int(hour_str)
        if hour < 0:
            return f"E|invalid hour: {hour} (must be non-negative)\n"
    except ValueError:
        return f"E|invalid hour: '{hour_str}' is not an integer\n"

    # Count empty fields on the raw incoming signals (tokens[3:8])
    signal_tokens = tokens[3:8]
    empty_count = sum(1 for tok in signal_tokens if tok.strip() == "")

    # Parse numeric signals
    signal_names = ["HR", "SBP", "Resp", "Temp", "WBC"]
    parsed_vals = []
    for name, tok in zip(signal_names, signal_tokens):
        s = tok.strip()
        if not s:
            parsed_vals.append(None)
        else:
            try:
                parsed_vals.append(float(s))
            except ValueError:
                return f"E|invalid numeric value for {name}: '{s}'\n"

    hr, sbp, resp, temp, wbc = parsed_vals

    try:
        # 1. Store observation in SQLite (WAL mode)
        kwargs = {"db_path": db_path} if db_path else {}
        save_observation(
            patient_id=patient_id,
            hour=hour,
            hr=hr,
            sbp=sbp,
            resp=resp,
            temp=temp,
            wbc=wbc,
            **kwargs,
        )

        # 2. Call single prediction point-of-truth
        res = predict(patient_id=patient_id, hour=hour, **kwargs)

        prob = res["probability"]
        alert = res["alert"]
        # Format probability to 4 decimals as mandated in DATA_CONTRACT.md
        prob_str = f"{prob:.4f}"

        return f"R|{patient_id}|{hour}|{prob_str}|{alert}|{empty_count}\n"

    except Exception as exc:
        logger.exception("Error processing record for patient %s hour %d: %s", patient_id, hour, exc)
        return f"E|internal prediction error: {str(exc)}\n"


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, db_path: Optional[str] = None):
    """Handle an incoming client TCP connection."""
    peer = writer.get_extra_info("peername")
    logger.info("Client connected from %s", peer)
    try:
        while True:
            line_bytes = await reader.readline()
            if not line_bytes:
                # Connection closed by client
                break
            try:
                raw_line = line_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                writer.write(b"E|invalid utf-8 encoding\n")
                await writer.drain()
                continue

            response = parse_and_process_line(raw_line, db_path=db_path)
            writer.write(response.encode("utf-8"))
            await writer.drain()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("Connection exception for %s: %s", peer, exc)
    finally:
        logger.info("Client disconnected: %s", peer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def start_server_async(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, db_path: Optional[str] = None):
    """Start the TCP line server asynchronously."""
    init_db(db_path) if db_path else init_db()
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, db_path=db_path),
        host,
        port,
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    logger.info("SepsisSignal TCP Line Server listening on %s", addrs)
    async with server:
        await server.serve_forever()


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="SepsisSignal TCP Line Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host address to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--db", default=None, help="Path to SQLite database")
    args = parser.parse_args()

    try:
        asyncio.run(start_server_async(host=args.host, port=args.port, db_path=args.db))
    except KeyboardInterrupt:
        logger.info("Server terminated by user (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
