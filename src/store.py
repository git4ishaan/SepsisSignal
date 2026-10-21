"""
SQLite storage layer for SepsisSignal.
WAL mode enabled, immutable predictions, and strict causal observation querying.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from contextlib import contextmanager

DB_PATH = os.environ.get("SEPSIS_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "sepsis_signal.db"))


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Create and configure a SQLite connection in WAL mode, ensuring it is cleanly closed."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
    finally:
        conn.close()



def init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite database tables as specified in DATA_CONTRACT.md."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                patient_id  TEXT    NOT NULL,
                hour        INTEGER NOT NULL,
                hr          REAL,
                sbp         REAL,
                resp        REAL,
                temp        REAL,
                wbc         REAL,
                received_at TEXT    NOT NULL,
                PRIMARY KEY (patient_id, hour)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                patient_id   TEXT    NOT NULL,
                hour         INTEGER NOT NULL,
                probability  REAL    NOT NULL,
                alert        INTEGER NOT NULL,
                empty_count  INTEGER NOT NULL,
                model_name   TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                PRIMARY KEY (patient_id, hour)
            );
        """)
        conn.commit()


def save_observation(
    patient_id: str,
    hour: int,
    hr: Optional[float],
    sbp: Optional[float],
    resp: Optional[float],
    temp: Optional[float],
    wbc: Optional[float],
    db_path: str = DB_PATH,
) -> bool:
    """
    Insert an observation for a patient-hour.
    If the (patient_id, hour) already exists, it is NOT overwritten.
    Returns True if inserted, False if already existed.
    """
    received_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO observations (patient_id, hour, hr, sbp, resp, temp, wbc, received_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (patient_id, hour, hr, sbp, resp, temp, wbc, received_at),
        )
        conn.commit()
        return cur.rowcount > 0


def get_observations_up_to(
    patient_id: str,
    hour: int,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve all observations for patient_id with hour <= requested hour, ordered by hour ASC.
    Guarantees strict causal query: observations from future hours are never returned.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT patient_id, hour, hr, sbp, resp, temp, wbc, received_at
            FROM observations
            WHERE patient_id = ? AND hour <= ?
            ORDER BY hour ASC;
            """,
            (patient_id, hour),
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def get_prediction(
    patient_id: str,
    hour: int,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve stored prediction for (patient_id, hour).
    Returns None if no prediction has been recorded yet.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT patient_id, hour, probability, alert, empty_count, model_name, created_at
            FROM predictions
            WHERE patient_id = ? AND hour = ?;
            """,
            (patient_id, hour),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def save_prediction(
    patient_id: str,
    hour: int,
    probability: float,
    alert: int,
    empty_count: int,
    model_name: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """
    Save prediction once. If prediction already exists for (patient_id, hour),
    it is NOT overwritten (immutability rule); the existing prediction is returned.
    """
    existing = get_prediction(patient_id, hour, db_path=db_path)
    if existing is not None:
        return existing

    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO predictions (patient_id, hour, probability, alert, empty_count, model_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (patient_id, hour, probability, alert, empty_count, model_name, created_at),
        )
        conn.commit()

    return get_prediction(patient_id, hour, db_path=db_path) or {
        "patient_id": patient_id,
        "hour": hour,
        "probability": probability,
        "alert": alert,
        "empty_count": empty_count,
        "model_name": model_name,
        "created_at": created_at,
    }


def list_patient_ids(db_path: str = DB_PATH) -> List[str]:
    """Return sorted list of all distinct patient IDs in observations."""
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT patient_id FROM observations ORDER BY patient_id ASC;")
        rows = cur.fetchall()
        return [row["patient_id"] for row in rows]


def get_timeline(
    patient_id: str,
    upto: Optional[int] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Return time series of observations merged with predictions for patient_id up to hour 'upto'.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        if upto is not None:
            cur.execute(
                """
                SELECT o.hour, o.hr, o.sbp, o.resp, o.temp, o.wbc,
                       p.probability, p.alert, p.empty_count, p.model_name
                FROM observations o
                LEFT JOIN predictions p ON o.patient_id = p.patient_id AND o.hour = p.hour
                WHERE o.patient_id = ? AND o.hour <= ?
                ORDER BY o.hour ASC;
                """,
                (patient_id, upto),
            )
        else:
            cur.execute(
                """
                SELECT o.hour, o.hr, o.sbp, o.resp, o.temp, o.wbc,
                       p.probability, p.alert, p.empty_count, p.model_name
                FROM observations o
                LEFT JOIN predictions p ON o.patient_id = p.patient_id AND o.hour = p.hour
                WHERE o.patient_id = ?
                ORDER BY o.hour ASC;
                """,
                (patient_id,),
            )
        rows = cur.fetchall()
        result = []
        for r in rows:
            values = {
                "hr": r["hr"],
                "sbp": r["sbp"],
                "resp": r["resp"],
                "temp": r["temp"],
                "wbc": r["wbc"],
            }
            missing = [k for k, v in values.items() if v is None]
            result.append({
                "hour": r["hour"],
                "values": values,
                "missing": missing,
                "probability": r["probability"],
                "alert": r["alert"],
            })
        return result
