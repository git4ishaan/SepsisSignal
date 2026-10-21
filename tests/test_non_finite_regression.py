"""
Regression test suite for non-finite measurement handling in SepsisSignal.

Contract verified:
1. +inf and -inf at target hour (with and without prior finite reading) are treated as missing:
   missing_flag == 1.0, forward-filled from previous finite measurement or imputer default.
2. +inf and -inf in historical observations are ignored and never forward-filled.
3. Deltas and time-since are computed strictly over finite measurements.
4. Feature vectors and predict() outputs are strictly finite.
5. None and NaN handling remain intact.
"""

import math
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_features_point_in_time, load_imputer_values, FEATURE_NAMES
from src.predict import predict
from src.store import init_db, save_observation
import tempfile


class TestNonFiniteHandling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imputer = load_imputer_values()

    def test_target_hour_inf_without_prior_reading(self):
        """1a. +inf and -inf at target hour without earlier reading -> treated as missing, uses imputer."""
        obs = [
            {"hour": 1, "hr": float("inf"), "sbp": float("-inf"), "resp": 18.0, "temp": 37.0, "wbc": 10.0}
        ]
        vec, dict_f = extract_features_point_in_time(obs, target_hour=1, imputer=self.imputer)

        # hr had +inf -> must be missing, value from imputer
        self.assertEqual(dict_f["hr_missing"], 1.0, "Expected +inf to set hr_missing=1.0")
        self.assertEqual(dict_f["hr_val"], self.imputer["hr"])
        self.assertEqual(dict_f["hr_time_since"], 99.0)
        self.assertEqual(dict_f["hr_delta"], 0.0)

        # sbp had -inf -> must be missing, value from imputer
        self.assertEqual(dict_f["sbp_missing"], 1.0, "Expected -inf to set sbp_missing=1.0")
        self.assertEqual(dict_f["sbp_val"], self.imputer["sbp"])
        self.assertEqual(dict_f["sbp_time_since"], 99.0)
        self.assertEqual(dict_f["sbp_delta"], 0.0)

        # empty count must reflect the 2 missing signals
        self.assertEqual(dict_f["current_empty_count"], 2.0)
        self.assertTrue(np.all(np.isfinite(vec)), "Feature vector must be strictly finite")

    def test_target_hour_inf_with_prior_finite_reading(self):
        """1b. +inf and -inf at target hour with earlier finite reading -> forward-fills finite reading."""
        obs = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": float("inf"), "sbp": float("-inf"), "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        vec, dict_f = extract_features_point_in_time(obs, target_hour=2, imputer=self.imputer)

        # hr had +inf at hour 2 -> forward-fill from hour 1 (80.0), time_since=1.0
        self.assertEqual(dict_f["hr_missing"], 1.0, "Expected +inf to set hr_missing=1.0")
        self.assertEqual(dict_f["hr_val"], 80.0, "Expected +inf to forward-fill 80.0 from hour 1")
        self.assertEqual(dict_f["hr_time_since"], 1.0, "Time since last finite reading should be 1.0 hour")
        self.assertEqual(dict_f["hr_delta"], 0.0, "Only 1 finite measurement in history, delta must be 0.0")

        # sbp had -inf at hour 2 -> forward-fill from hour 1 (120.0), time_since=1.0
        self.assertEqual(dict_f["sbp_missing"], 1.0, "Expected -inf to set sbp_missing=1.0")
        self.assertEqual(dict_f["sbp_val"], 120.0, "Expected -inf to forward-fill 120.0 from hour 1")
        self.assertEqual(dict_f["sbp_time_since"], 1.0)
        self.assertEqual(dict_f["sbp_delta"], 0.0)

        self.assertEqual(dict_f["current_empty_count"], 2.0)
        self.assertTrue(np.all(np.isfinite(vec)), "Feature vector must be strictly finite")

    def test_historical_inf_never_forward_filled(self):
        """2. Historical +inf / -inf must never be forward-filled into subsequent hours."""
        obs = [
            {"hour": 1, "hr": 75.0, "sbp": 115.0, "resp": 16.0, "temp": 36.5, "wbc": 7.0},
            {"hour": 2, "hr": float("inf"), "sbp": float("-inf"), "resp": None, "temp": None, "wbc": None},
            {"hour": 3, "hr": None, "sbp": None, "resp": 18.0, "temp": 36.8, "wbc": 8.0},
        ]
        vec, dict_f = extract_features_point_in_time(obs, target_hour=3, imputer=self.imputer)

        # At hour 3, hr and sbp are None; hour 2 had inf/-inf.
        # They MUST forward-fill from hour 1 (finite 75.0 / 115.0), NOT from hour 2!
        self.assertEqual(dict_f["hr_val"], 75.0, "Hour 3 must forward-fill finite reading from hour 1, not hour 2 inf")
        self.assertEqual(dict_f["sbp_val"], 115.0, "Hour 3 must forward-fill finite reading from hour 1, not hour 2 -inf")
        self.assertEqual(dict_f["hr_time_since"], 2.0, "Time since hour 1 finite measurement is 3 - 1 = 2.0 hours")
        self.assertEqual(dict_f["sbp_time_since"], 2.0)
        self.assertEqual(dict_f["hr_missing"], 1.0)
        self.assertEqual(dict_f["sbp_missing"], 1.0)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_finite_downstream_predictions(self):
        """4. Downstream model predictions remain finite and valid when inputs contain inf."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_inf_pred.db")
        init_db(db_path)
        try:
            save_observation(
                patient_id="P_INF",
                hour=1,
                hr=80.0,
                sbp=120.0,
                resp=18.0,
                temp=37.0,
                wbc=10.0,
                db_path=db_path,
            )
            save_observation(
                patient_id="P_INF",
                hour=2,
                hr=float("inf"),
                sbp=float("-inf"),
                resp=float("inf"),
                temp=float("-inf"),
                wbc=float("inf"),
                db_path=db_path,
            )
            pred = predict("P_INF", 2, db_path=db_path)
            self.assertTrue(math.isfinite(pred["probability"]), "Prediction probability must be finite")
            self.assertTrue(0.0 <= pred["probability"] <= 1.0, "Probability must be in [0, 1]")
            self.assertIn(pred["alert"], (0, 1), "Alert must be 0 or 1")
            self.assertEqual(pred["empty_count"], 5, "All 5 inf signals should be counted as empty")
        finally:
            for s in ["", "-wal", "-shm"]:
                p = db_path + s
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def test_none_and_nan_behavior_preserved(self):
        """6. None and NaN behavior remains correct."""
        obs = [
            {"hour": 1, "hr": None, "sbp": float("nan"), "resp": np.nan, "temp": 37.0, "wbc": 10.0}
        ]
        vec, dict_f = extract_features_point_in_time(obs, target_hour=1, imputer=self.imputer)
        self.assertEqual(dict_f["hr_missing"], 1.0)
        self.assertEqual(dict_f["sbp_missing"], 1.0)
        self.assertEqual(dict_f["resp_missing"], 1.0)
        self.assertEqual(dict_f["hr_val"], self.imputer["hr"])
        self.assertEqual(dict_f["sbp_val"], self.imputer["sbp"])
        self.assertEqual(dict_f["resp_val"], self.imputer["resp"])
        self.assertEqual(dict_f["current_empty_count"], 3.0)
        self.assertTrue(np.all(np.isfinite(vec)))


if __name__ == "__main__":
    unittest.main()
