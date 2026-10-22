"""
Automated Test Suite for SepsisSignal Soft-Voting Ensemble Experiment

Research prototype, not for clinical decision-making.

Verifies:
1. Correct equal-weight probability averaging: (p_lr + p_xgb + p_gru) / 3.0
2. Correct model loading and input contracts
3. Finite and strictly bounded output probabilities [0.0, 1.0]
4. Missing or corrupt artifact error handling (no silent concealment or fallback invention)
5. Feature-order and preprocessing consistency across all 22 features
6. Patient and prediction-hour alignment in batch stay evaluation
7. Strict causal feature extraction (no future-data leakage)
8. Repeated prediction stability (deterministic inference)
9. Existing stored-prediction immutability (zero side-effects on SQLite database)
10. Model failure handling on malformed or empty inputs
11. Regression compatibility with existing live inference path (src/predict.py)
"""

import math
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from src.ensemble import SepsisEnsemble
from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values
from src.predict import predict as live_predict
from src.store import init_db, save_observation, save_prediction, get_prediction


class TestSepsisEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        cls.ensemble = SepsisEnsemble(models_dir=cls.models_dir)

    def setUp(self):
        self.sample_obs = [
            {"patient_id": "TEST_P01", "hour": 1, "hr": 82.0, "sbp": 120.0, "resp": 18.0, "temp": 36.8, "wbc": 7.5, "label": 0},
            {"patient_id": "TEST_P01", "hour": 2, "hr": 85.0, "sbp": 118.0, "resp": 19.0, "temp": 36.9, "wbc": 8.0, "label": 0},
            {"patient_id": "TEST_P01", "hour": 3, "hr": 92.0, "sbp": 110.0, "resp": 22.0, "temp": 37.4, "wbc": 9.5, "label": 0},
        ]

    def test_01_equal_weight_averaging(self):
        """Verify ensemble probability is strictly the equal-weighted average (p_lr + p_xgb + p_gru) / 3.0."""
        res = self.ensemble.predict_point_in_time(self.sample_obs, target_hour=3)
        p_lr = res["lr_probability"]
        p_xgb = res["xgb_probability"]
        p_gru = res["gru_probability"]
        p_ens = res["ensemble_probability"]

        expected = round((p_lr + p_xgb + p_gru) / 3.0, 4)
        self.assertAlmostEqual(p_ens, expected, places=4)

    def test_02_model_loading_and_input_contracts(self):
        """Verify models are correctly loaded with 22-dimensional feature contracts and proper types."""
        self.assertIsNotNone(self.ensemble.lr_model)
        self.assertIsNotNone(self.ensemble.xgb_model)
        self.assertIsNotNone(self.ensemble.gru_model)
        self.assertEqual(len(self.ensemble.train_mean), 22)
        self.assertEqual(len(self.ensemble.train_std), 22)

        # Verify LR expects 22 features
        dummy_vec = np.zeros((1, 22), dtype=np.float32)
        p_lr = self.ensemble.lr_model.predict_proba(dummy_vec)
        self.assertEqual(p_lr.shape, (1, 2))

        # Verify XGBoost expects 22 features
        p_xgb = self.ensemble.xgb_model.predict_proba(dummy_vec)
        self.assertEqual(p_xgb.shape, (1, 2))

        # Verify GRU accepts (1, seq_len, 22) and outputs (1, seq_len)
        dummy_seq = torch.zeros((1, 5, 22), dtype=torch.float32)
        with torch.no_grad():
            logits = self.ensemble.gru_model(dummy_seq)
        self.assertEqual(logits.shape, (1, 5))

    def test_03_finite_and_bounded_probabilities(self):
        """Verify output probabilities are finite and strictly bounded within [0.0, 1.0]."""
        res = self.ensemble.predict_point_in_time(self.sample_obs, target_hour=2)
        for key in ["lr_probability", "xgb_probability", "gru_probability", "ensemble_probability"]:
            prob = res[key]
            self.assertTrue(math.isfinite(prob), f"{key} is non-finite: {prob}")
            self.assertGreaterEqual(prob, 0.0, f"{key} is < 0: {prob}")
            self.assertLessEqual(prob, 1.0, f"{key} is > 1: {prob}")

    def test_04_missing_or_corrupt_artifacts(self):
        """Verify clear exception raising when model artifacts are missing (no silent fallback)."""
        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                SepsisEnsemble(models_dir=temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_05_feature_order_and_preprocessing_consistency(self):
        """Verify exact 22 features match canonical FEATURE_NAMES ordering and imputer is applied."""
        self.assertEqual(len(FEATURE_NAMES), 22)
        vec, feat_dict = extract_features_point_in_time(self.sample_obs, target_hour=2, imputer=self.ensemble.imputer)
        self.assertEqual(vec.shape, (1, 22))
        self.assertTrue(np.all(np.isfinite(vec)))

        # Verify key feature names present in metadata
        self.assertIn("hr_val", FEATURE_NAMES)
        self.assertIn("sbp_val", FEATURE_NAMES)
        self.assertIn("current_empty_count", feat_dict)

    def test_06_patient_stay_alignment(self):
        """Verify batch stay evaluation produces perfectly aligned arrays for hours, labels, and model probabilities."""
        res = self.ensemble.predict_patient_stay_aligned(self.sample_obs)
        n = len(self.sample_obs)

        self.assertEqual(len(res["hours"]), n)
        self.assertEqual(len(res["labels"]), n)
        self.assertEqual(len(res["p_lr"]), n)
        self.assertEqual(len(res["p_xgb"]), n)
        self.assertEqual(len(res["p_gru"]), n)
        self.assertEqual(len(res["p_ensemble"]), n)

        # Check alignment of individual hour calculations
        for i in range(n):
            expected_ens = (res["p_lr"][i] + res["p_xgb"][i] + res["p_gru"][i]) / 3.0
            self.assertAlmostEqual(res["p_ensemble"][i], expected_ens, places=4)

    def test_07_no_future_data_leakage(self):
        """Verify causal invariance: adding future hours does not alter Hour 2 predictions."""
        # 1. Predict Hour 2 with only observations up to Hour 2
        obs_past = self.sample_obs[:2]
        res_before = self.ensemble.predict_point_in_time(obs_past, target_hour=2)

        # 2. Add future Hour 3 with severe shock vitals
        obs_future = list(self.sample_obs)
        obs_future.append({
            "patient_id": "TEST_P01",
            "hour": 4,
            "hr": 160.0,
            "sbp": 60.0,
            "resp": 38.0,
            "temp": 40.2,
            "wbc": 28.0,
            "label": 1,
        })
        res_after = self.ensemble.predict_point_in_time(obs_future, target_hour=2)

        self.assertEqual(res_before["lr_probability"], res_after["lr_probability"])
        self.assertEqual(res_before["xgb_probability"], res_after["xgb_probability"])
        self.assertEqual(res_before["gru_probability"], res_after["gru_probability"])
        self.assertEqual(res_before["ensemble_probability"], res_after["ensemble_probability"])

    def test_08_repeated_prediction_stability(self):
        """Verify deterministic outputs on repeated evaluations."""
        run1 = self.ensemble.predict_point_in_time(self.sample_obs, target_hour=3)
        run2 = self.ensemble.predict_point_in_time(self.sample_obs, target_hour=3)
        self.assertEqual(run1, run2)

        stay1 = self.ensemble.predict_patient_stay_aligned(self.sample_obs)
        stay2 = self.ensemble.predict_patient_stay_aligned(self.sample_obs)
        np.testing.assert_array_equal(stay1["p_ensemble"], stay2["p_ensemble"])

    def test_09_database_stored_prediction_immutability(self):
        """Verify ensemble inference does not mutate existing SQLite stored predictions."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "immutability_test.db")
        try:
            init_db(db_path)
            # Insert initial observation and prediction
            save_observation("P_IMMUTABLE", 1, 80.0, 120.0, 18.0, 36.5, 8.0, db_path=db_path)
            save_prediction("P_IMMUTABLE", 1, 0.4200, 1, 0, "logistic_regression", db_path=db_path)

            pred_before = get_prediction("P_IMMUTABLE", 1, db_path=db_path)
            self.assertIsNotNone(pred_before)

            # Run ensemble evaluation independently
            _ = self.ensemble.predict_point_in_time(self.sample_obs, target_hour=2)

            pred_after = get_prediction("P_IMMUTABLE", 1, db_path=db_path)
            self.assertIsNotNone(pred_after)
            self.assertEqual(pred_before["probability"], pred_after["probability"])
            self.assertEqual(pred_before["alert"], pred_after["alert"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_10_model_failure_handling(self):
        """Verify graceful, explicit failure with ValueError on invalid or empty inputs."""
        with self.assertRaises(ValueError):
            self.ensemble.predict_point_in_time([], target_hour=1)

        with self.assertRaises(ValueError):
            self.ensemble.predict_point_in_time(self.sample_obs, target_hour=0)

        # Verify explicit exception on non-finite probability validation
        with self.assertRaises(ValueError):
            self.ensemble._validate_probability(float("nan"), "TestModel")
        with self.assertRaises(ValueError):
            self.ensemble._validate_probability(1.5, "TestModel")
        with self.assertRaises(ValueError):
            self.ensemble._validate_probability(-0.1, "TestModel")

    def test_11_live_inference_regression_compatibility(self):
        """Verify existing production live inference path (src/predict.py) remains completely unaltered."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "live_test.db")
        try:
            init_db(db_path)
            save_observation("P_LIVE", 1, 80.0, 120.0, 18.0, 36.5, 8.0, db_path=db_path)

            pred = live_predict("P_LIVE", 1, db_path=db_path)
            self.assertIn("probability", pred)
            self.assertIn("alert", pred)
            self.assertIn("threshold", pred)
            self.assertEqual(pred["threshold"], 0.42)
            self.assertTrue(0.0 <= pred["probability"] <= 1.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
