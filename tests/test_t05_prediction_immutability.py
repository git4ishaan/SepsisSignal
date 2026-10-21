"""
Test T05 — Prediction Immutability for SepsisSignal
Target: SQLite predictions storage, predict() in src/predict.py, and line server ingestion in src/server.py

Objective:
Test whether stored predictions remain strictly immutable after repeated prediction requests,
future observation ingestion, late historical data, out-of-order data, and cross-patient isolation.

Schema under test:
predictions(patient_id, hour, probability, alert, empty_count, model_name, created_at)
PRIMARY KEY (patient_id, hour)

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import os
import sys
import tempfile
import time
import unittest
import sqlite3

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict
from src.server import parse_and_process_line
from src.store import get_connection, get_prediction, init_db, save_observation, save_prediction


class TestT05PredictionImmutability(unittest.TestCase):
    """
    Exhaustive test suite for T05 Prediction Immutability covering all 8 specified scenarios.
    """

    def setUp(self):
        """Create an isolated temporary SQLite database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "t05_test.db")
        init_db(self.db_path)

    def tearDown(self):
        """Clean up isolated database files."""
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _query_prediction_row(self, patient_id: str, hour: int):
        """Direct SQL query to fetch raw row from predictions table."""
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT patient_id, hour, probability, alert, empty_count, model_name, created_at
                FROM predictions
                WHERE patient_id = ? AND hour = ?;
                """,
                (patient_id, hour),
            )
            return cur.fetchone()

    def test_case_1_repeated_prediction(self):
        """
        1. Repeated prediction:
        Create a prediction for a patient and hour, request the same prediction again,
        and verify the existing stored record remains byte-for-byte unchanged (including created_at).
        """
        save_observation("P001", 1, 80.0, 120.0, 18.0, 36.8, 8.0, db_path=self.db_path)
        pred_initial = predict("P001", 1, db_path=self.db_path)

        # Allow time to pass so created_at would differ if re-generated
        time.sleep(0.01)

        pred_repeated = predict("P001", 1, db_path=self.db_path)

        # Verify application-level return equality
        self.assertEqual(pred_initial["probability"], pred_repeated["probability"])
        self.assertEqual(pred_initial["alert"], pred_repeated["alert"])
        self.assertEqual(pred_initial["empty_count"], pred_repeated["empty_count"])
        self.assertEqual(pred_initial["model_name"], pred_repeated["model_name"])
        self.assertEqual(pred_initial["created_at"], pred_repeated["created_at"])

        # Direct database record verification
        row = self._query_prediction_row("P001", 1)
        self.assertIsNotNone(row)
        self.assertEqual(row["created_at"], pred_initial["created_at"])
        self.assertEqual(row["probability"], pred_initial["probability"])

    def test_case_2_future_observations(self):
        """
        2. Future observations:
        Store a prediction, ingest observations at later hours with alarming vitals,
        then request original prediction again. Verify original database record is unchanged.
        """
        # Hour 1 and Hour 2 normal vitals
        save_observation("P002", 1, 75.0, 120.0, 16.0, 36.7, 7.0, db_path=self.db_path)
        save_observation("P002", 2, 78.0, 118.0, 16.0, 36.8, 7.2, db_path=self.db_path)
        pred_h2_before = predict("P002", 2, db_path=self.db_path)
        raw_row_before = dict(self._query_prediction_row("P002", 2))

        # Later hours arrive with extreme sepsis-like vitals
        save_observation("P002", 3, 135.0, 80.0, 32.0, 39.5, 22.0, db_path=self.db_path)
        save_observation("P002", 4, 150.0, 70.0, 36.0, 40.1, 28.0, db_path=self.db_path)
        pred_h4 = predict("P002", 4, db_path=self.db_path)
        self.assertEqual(pred_h4["alert"], 1, "Extreme sepsis vitals at hour 4 should trigger an alert")

        # Query Hour 2 again
        pred_h2_after = predict("P002", 2, db_path=self.db_path)
        raw_row_after = dict(self._query_prediction_row("P002", 2))

        # Must be strictly identical
        self.assertEqual(pred_h2_before, pred_h2_after)
        self.assertEqual(raw_row_before, raw_row_after)

    def test_case_3_late_arriving_historical_data(self):
        """
        3. Late-arriving historical data:
        Store a prediction for Hour 2 before Hour 1 exists.
        Then ingest Hour 1 afterwards. Verify Hour 2's stored prediction is not silently rewritten.
        """
        # Hour 2 arrives first
        save_observation("P003", 2, 85.0, 115.0, 18.0, 37.0, 9.0, db_path=self.db_path)
        pred_h2_initial = predict("P003", 2, db_path=self.db_path)
        row_h2_initial = dict(self._query_prediction_row("P003", 2))

        # Now late-arriving Hour 1 is ingested
        save_observation("P003", 1, 140.0, 70.0, 35.0, 40.0, 25.0, db_path=self.db_path)

        # Query Hour 2 again
        pred_h2_subsequent = predict("P003", 2, db_path=self.db_path)
        row_h2_subsequent = dict(self._query_prediction_row("P003", 2))

        self.assertEqual(pred_h2_initial, pred_h2_subsequent)
        self.assertEqual(row_h2_initial, row_h2_subsequent)

    def test_case_4_late_arriving_target_hour_data(self):
        """
        4. Late-arriving target-hour data:
        Store a prediction for Hour 1 (e.g. empty or partial).
        Ingest late/duplicate observation for Hour 1.
        Verify historical prediction remains unchanged.
        """
        # Store prediction for Hour 1 with empty vitals
        save_observation("P004", 1, None, None, None, None, None, db_path=self.db_path)
        pred_h1_initial = predict("P004", 1, db_path=self.db_path)
        row_h1_initial = dict(self._query_prediction_row("P004", 1))

        # Ingestion contract: late update for Hour 1 arrives via socket
        # Note: save_observation uses INSERT OR IGNORE, preserving original observation
        inserted = save_observation("P004", 1, 130.0, 80.0, 30.0, 39.5, 20.0, db_path=self.db_path)
        self.assertFalse(inserted, "Duplicate observation for (patient_id, hour) must not overwrite")

        # Query Hour 1 prediction again
        pred_h1_after = predict("P004", 1, db_path=self.db_path)
        row_h1_after = dict(self._query_prediction_row("P004", 1))

        self.assertEqual(pred_h1_initial, pred_h1_after)
        self.assertEqual(row_h1_initial, row_h1_after)

    def test_case_5_out_of_order_ingestion(self):
        """
        5. Out-of-order ingestion:
        Ingest observations out of chronological order (Hour 3, then Hour 1, then Hour 2).
        Check that existing prediction records remain unchanged.
        """
        # Hour 3 arrives first and is predicted
        parse_and_process_line("H|P005|3|90|115|20|37.1|8.5\n", db_path=self.db_path)
        pred_h3_first = dict(self._query_prediction_row("P005", 3))

        # Hour 1 arrives next and is predicted
        parse_and_process_line("H|P005|1|75|120|16|36.5|7.0\n", db_path=self.db_path)
        pred_h1_first = dict(self._query_prediction_row("P005", 1))

        # Hour 2 arrives last and is predicted
        parse_and_process_line("H|P005|2|80|118|18|36.7|7.8\n", db_path=self.db_path)

        # Verify Hour 3 and Hour 1 predictions did not change
        pred_h3_after = dict(self._query_prediction_row("P005", 3))
        pred_h1_after = dict(self._query_prediction_row("P005", 1))

        self.assertEqual(pred_h3_first, pred_h3_after)
        self.assertEqual(pred_h1_first, pred_h1_after)

    def test_case_6_patient_isolation(self):
        """
        6. Patient isolation:
        Verify that observations and predictions for Patient A never alter or leak into
        Patient B's stored predictions.
        """
        # Patient A (healthy)
        save_observation("PATIENT_A", 1, 70.0, 120.0, 16.0, 36.8, 6.5, db_path=self.db_path)
        pred_a_initial = predict("PATIENT_A", 1, db_path=self.db_path)
        row_a_initial = dict(self._query_prediction_row("PATIENT_A", 1))

        # Patient B (acute septic shock)
        save_observation("PATIENT_B", 1, 145.0, 65.0, 35.0, 40.2, 29.0, db_path=self.db_path)
        pred_b = predict("PATIENT_B", 1, db_path=self.db_path)
        self.assertEqual(pred_b["alert"], 1)

        # Repeated operations on Patient B
        save_observation("PATIENT_B", 2, 150.0, 60.0, 38.0, 40.5, 32.0, db_path=self.db_path)
        predict("PATIENT_B", 2, db_path=self.db_path)

        # Verify Patient A is completely unaffected
        pred_a_after = predict("PATIENT_A", 1, db_path=self.db_path)
        row_a_after = dict(self._query_prediction_row("PATIENT_A", 1))

        self.assertEqual(pred_a_initial, pred_a_after)
        self.assertEqual(row_a_initial, row_a_after)

    def test_case_7_database_invariants(self):
        """
        7. Database invariants:
        Verify there is at most one prediction row per (patient_id, hour) and that repeated
        requests do not change probability, alert, empty_count, model_name, or created_at.
        Also verifies unique constraint enforcement against direct duplicate writes.
        """
        save_observation("P007", 1, 82.0, 118.0, 17.0, 36.9, 8.2, db_path=self.db_path)
        pred1 = predict("P007", 1, db_path=self.db_path)

        # Multiple duplicate calls to predict
        for _ in range(5):
            predict("P007", 1, db_path=self.db_path)

        # Attempt explicit duplicate write via save_prediction with different values
        saved_return = save_prediction(
            patient_id="P007",
            hour=1,
            probability=0.9999,
            alert=1,
            empty_count=0,
            model_name="malicious_overwrite_attempt",
            db_path=self.db_path,
        )
        self.assertEqual(saved_return["probability"], pred1["probability"])
        self.assertEqual(saved_return["created_at"], pred1["created_at"])

        # Direct SQL uniqueness assertion
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE patient_id = ? AND hour = ?;",
                ("P007", 1),
            )
            count = cur.fetchone()["cnt"]
            self.assertEqual(count, 1, "Exactly one prediction row must exist for (patient_id, hour)")

            # Attempt direct SQL insert violation
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO predictions (patient_id, hour, probability, alert, empty_count, model_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    ("P007", 1, 0.9999, 1, 0, "direct_sql_insert", "2099-01-01T00:00:00Z"),
                )

    def test_case_8_new_hour_control(self):
        """
        8. New-hour control:
        Confirm that a prediction for a genuinely new hour can still be created normally
        without altering previously stored records.
        """
        # Hour 1 created
        save_observation("P008", 1, 80.0, 120.0, 16.0, 36.8, 7.5, db_path=self.db_path)
        pred_h1 = predict("P008", 1, db_path=self.db_path)
        row_h1_initial = dict(self._query_prediction_row("P008", 1))

        # Hour 2 created
        save_observation("P008", 2, 85.0, 118.0, 18.0, 37.0, 8.0, db_path=self.db_path)
        pred_h2 = predict("P008", 2, db_path=self.db_path)
        row_h2_initial = dict(self._query_prediction_row("P008", 2))

        # Hour 3 created (new hour)
        save_observation("P008", 3, 110.0, 95.0, 24.0, 38.5, 14.0, db_path=self.db_path)
        pred_h3 = predict("P008", 3, db_path=self.db_path)
        row_h3 = dict(self._query_prediction_row("P008", 3))

        # Hour 3 prediction exists and is valid
        self.assertIsNotNone(row_h3)
        self.assertEqual(row_h3["hour"], 3)
        self.assertEqual(pred_h3["hour"], 3)

        # Hour 1 and Hour 2 records are strictly unchanged
        row_h1_after = dict(self._query_prediction_row("P008", 1))
        row_h2_after = dict(self._query_prediction_row("P008", 2))

        self.assertEqual(row_h1_initial, row_h1_after)
        self.assertEqual(row_h2_initial, row_h2_after)


if __name__ == "__main__":
    unittest.main()
