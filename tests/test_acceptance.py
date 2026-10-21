"""
Acceptance test suite for SepsisSignal.
Verifies all 6 mandatory jury checks defined in AGENTS.md and TASKS.md:
1. Byte-identical immutable predictions: Hour 2 reply identical before and after Hour 3 arrives.
2. Trailing pipes & empty counts: 'H|P001|2|88|112|||' returns empty count 3 (> 0).
3. Streaming parity check: streaming matches single point-in-time calculation.
4. Unified prediction: predict() matches server response.
5. Unseen patient IDs work seamlessly without any external dataset lookup.
6. Malformed lines return E|<reason> and do not crash or hang the server (\\r\\n and \\n accepted).
"""

import os
import sys
import tempfile
import pytest

# Ensure repository root is on sys.path for direct pytest invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict
from src.server import parse_and_process_line
from src.store import init_db


@pytest.fixture
def clean_db():
    """Create a temporary isolated SQLite database for a test."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_sepsis.db")
    init_db(db_path)
    yield db_path
    # Cleanup
    for suffix in ["", "-wal", "-shm"]:
        p = db_path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def test_acceptance_check_2_empty_fields_count(clean_db):
    """
    Check 2: H|P001|2|88|112||| returns an empty-field count of 3 (greater than zero).
    Never strips trailing pipes.
    """
    line = "H|P001|2|88|112|||\n"
    reply = parse_and_process_line(line, db_path=clean_db)
    assert reply.startswith("R|P001|2|"), f"Unexpected reply: {reply}"
    tokens = reply.strip().split("|")
    assert len(tokens) == 6, f"Expected 6 tokens in reply, got {len(tokens)}: {tokens}"
    empty_count = int(tokens[5])
    assert empty_count == 3, f"Expected empty_count 3, got {empty_count}"


def test_acceptance_check_1_immutability(clean_db):
    """
    Check 1: Send hours 1, 2, 3 for a test patient.
    Hour 2's reply is byte-identical before and after hour 3 arrives.
    """
    h1 = "H|P002|1|80|120|18|36.5|8.0\n"
    h2 = "H|P002|2|85|118|19|36.8|8.5\n"
    
    reply1 = parse_and_process_line(h1, db_path=clean_db)
    reply2_before = parse_and_process_line(h2, db_path=clean_db)
    
    # Now send Hour 3 with extreme sepsis-like vital signs
    h3 = "H|P002|3|135|85|32|39.4|18.5\n"
    reply3 = parse_and_process_line(h3, db_path=clean_db)
    assert reply3.startswith("R|P002|3|")
    
    # Query Hour 2 again by re-sending Hour 2 or predicting Hour 2
    reply2_after = parse_and_process_line(h2, db_path=clean_db)
    
    assert reply2_before == reply2_after, (
        f"Immutability violated!\nBefore: {reply2_before}\nAfter: {reply2_after}"
    )


def test_acceptance_check_3_streaming_parity(clean_db):
    """
    Check 3: Streaming patient through the protocol produces identical results to single predict().
    """
    hours = [
        (1, "H|P100|1|78|115|16|36.7|7.2\n"),
        (2, "H|P100|2|82|118|17|36.8|7.5\n"),
        (3, "H|P100|3|90|112|20|37.1|8.9\n"),
    ]
    for hr, line in hours:
        sock_reply = parse_and_process_line(line, db_path=clean_db)
        tokens = sock_reply.strip().split("|")
        sock_prob = float(tokens[3])
        sock_alert = int(tokens[4])

        pred = predict("P100", hr, db_path=clean_db)
        assert abs(pred["probability"] - sock_prob) < 1e-4, f"Probability mismatch at hour {hr}"
        assert pred["alert"] == sock_alert, f"Alert mismatch at hour {hr}"


def test_acceptance_check_5_unseen_patient(clean_db):
    """
    Check 5: A patient ID never seen before works with no dataset lookup.
    First hour with no history uses training sentinel values.
    """
    new_patient_line = "H|UNKNOWN_PATIENT_999|1|92|124|16|37.0|7.5\r\n"
    reply = parse_and_process_line(new_patient_line, db_path=clean_db)
    assert reply.startswith("R|UNKNOWN_PATIENT_999|1|"), f"Failed for unseen patient: {reply}"
    tokens = reply.strip().split("|")
    assert len(tokens) == 6
    prob = float(tokens[3])
    assert 0.0 <= prob <= 1.0


def test_acceptance_check_6_malformed_inputs(clean_db):
    """
    Check 6: Malformed lines return E|<reason> and do not crash the server.
    Accepts \\r\\n and \\n.
    """
    bad_lines = [
        "INVALID_LINE\n",
        "H|P003|1|80|120\n",  # Not enough fields
        "H|P003|not_an_int|80|120|18|36.5|8.0\n",  # Non-integer hour
        "H|P003|1|abc|120|18|36.5|8.0\r\n",  # Non-numeric HR
        "X|P003|1|80|120|18|36.5|8.0\n",  # Bad header
        "\n",  # Empty line
    ]
    for bad in bad_lines:
        reply = parse_and_process_line(bad, db_path=clean_db)
        assert reply.startswith("E|"), f"Expected error for '{bad}', got '{reply}'"


def test_acceptance_check_4_predict_api_parity(clean_db):
    """
    Check 4: The predict() function returns the exact same probability and alert
    as the socket line reply for that hour.
    """
    line = "H|P004|1|102|95|24|38.2|14.0\n"
    reply = parse_and_process_line(line, db_path=clean_db)
    tokens = reply.strip().split("|")
    socket_prob = float(tokens[3])
    socket_alert = int(tokens[4])
    
    pred = predict("P004", 1, db_path=clean_db)
    assert abs(pred["probability"] - socket_prob) < 1e-6, "Probability mismatch"
    assert pred["alert"] == socket_alert, "Alert mismatch"


def test_out_of_order_causality(clean_db):
    """
    Causality check: If hour 5 arrives, predicting hour 2 only considers rows <= hour 2.
    """
    parse_and_process_line("H|P005|1|80|120|18|36.5|8.0\n", db_path=clean_db)
    parse_and_process_line("H|P005|5|130|80|30|39.5|20.0\n", db_path=clean_db)
    
    # Now hour 2 arrives
    reply_h2 = parse_and_process_line("H|P005|2|82|119|18|36.6|8.1\n", db_path=clean_db)
    assert reply_h2.startswith("R|P005|2|")
    
    # Predict hour 2 directly
    p2 = predict("P005", 2, db_path=clean_db)
    # The extreme vitals at hour 5 must NOT affect hour 2
    assert p2["probability"] < 0.8, f"Hour 5 vitals leaked into Hour 2: {p2}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
