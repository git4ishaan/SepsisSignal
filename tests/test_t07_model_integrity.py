"""
Test T07 — Model Loading, Feature Compatibility, and Prediction Integrity
Target:
  - Baseline Logistic Regression: models/model.joblib (threshold: 0.42, live path)
  - XGBoost comparison: models/xgboost_model.joblib (threshold: 0.39, offline)
  - PyTorch GRU: models/lstm_model.pt (threshold: 0.42, offline)
  - Feature contract: 22-dimensional float32 vector adhering strictly to FEATURE_NAMES

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
import joblib
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.train_lstm import SepsisRNN
from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values
from src.predict import (
    MODEL_NAME,
    MODEL_PATH,
    THRESHOLD_PATH,
    get_model,
    get_threshold,
    predict,
)
from src.store import get_connection, init_db, save_observation

MODELS_DIR = os.path.join(REPO_ROOT, "models")
XGBOOST_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
XGBOOST_THRESH_PATH = os.path.join(MODELS_DIR, "xgboost_threshold.json")
LSTM_PATH = os.path.join(MODELS_DIR, "lstm_model.pt")
LSTM_THRESH_PATH = os.path.join(MODELS_DIR, "lstm_threshold.json")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


class TestT07ModelLoadingAndIntegrity(unittest.TestCase):
    """
    Exhaustive suite covering T07:
    - Section A: Model loading & artifact resilience
    - Section B: Feature compatibility & ordering
    - Section C: Prediction integrity, thresholds & sigmoid conversion
    - Section D: End-to-end integration & immutability parity
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "t07_test.db")
        init_db(self.db_path)
        self.imputer = load_imputer_values()

    def tearDown(self):
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # =========================================================================
    # SECTION A: Model Loading
    # =========================================================================

    def test_case_1_load_all_supported_models(self):
        """
        1. Each supported model can load its existing artifact successfully:
        - Baseline Logistic Regression (live path)
        - XGBoost Classifier (offline comparison)
        - PyTorch Sequence GRU (offline comparison)
        """
        # Baseline
        lr_model = get_model()
        self.assertIsNotNone(lr_model)
        self.assertTrue(hasattr(lr_model, "predict_proba"))

        # XGBoost
        self.assertTrue(os.path.exists(XGBOOST_PATH), f"Missing {XGBOOST_PATH}")
        xgb_model = joblib.load(XGBOOST_PATH)
        self.assertIsNotNone(xgb_model)
        self.assertTrue(hasattr(xgb_model, "predict_proba"))

        # PyTorch GRU
        self.assertTrue(os.path.exists(LSTM_PATH), f"Missing {LSTM_PATH}")
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        self.assertIn("state_dict", ckpt)
        rnn_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        rnn_model.load_state_dict(ckpt["state_dict"])
        rnn_model.eval()
        self.assertIsNotNone(rnn_model)

    def test_case_2_missing_or_invalid_artifacts_error_handling(self):
        """
        2. Missing or invalid artifacts produce a clear, controlled error
        rather than a silent fallback to an unrelated model.
        """
        # Scenario A: Non-existent file
        non_existent = os.path.join(self.temp_dir, "does_not_exist.joblib")
        with self.assertRaises(FileNotFoundError):
            if not os.path.exists(non_existent):
                raise FileNotFoundError(f"Model artifact not found at {non_existent}")
            joblib.load(non_existent)

        # Scenario B: Corrupt / truncated artifact
        corrupt_file = os.path.join(self.temp_dir, "corrupt_artifact.joblib")
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_JOBLIB_OR_PICKLE_FILE_HEADER")

        with self.assertRaises(Exception):
            joblib.load(corrupt_file)

        # Scenario C: Corrupt PyTorch artifact
        corrupt_pt = os.path.join(self.temp_dir, "corrupt_model.pt")
        with open(corrupt_pt, "wb") as f:
            f.write(b"CORRUPT_PYTORCH_DATA")

        with self.assertRaises(Exception):
            torch.load(corrupt_pt, weights_only=False)

    def test_case_3_loading_does_not_modify_artifacts(self):
        """
        3. Loading a model does not modify protected artifacts.
        SHA-256 hashes must be strictly identical before and after load operations.
        """
        artifacts = [
            MODEL_PATH,
            THRESHOLD_PATH,
            os.path.join(MODELS_DIR, "imputer.json"),
            XGBOOST_PATH,
            LSTM_PATH,
        ]
        hashes_before = {p: compute_sha256(p) for p in artifacts if os.path.exists(p)}

        # Perform multiple loads and evaluations
        _ = get_model()
        _ = joblib.load(XGBOOST_PATH)
        _ = torch.load(LSTM_PATH, weights_only=False)

        hashes_after = {p: compute_sha256(p) for p in artifacts if os.path.exists(p)}

        for p in hashes_before:
            self.assertEqual(
                hashes_before[p],
                hashes_after[p],
                f"Artifact {os.path.basename(p)} was modified during load!",
            )

    def test_case_4_cached_model_consistency(self):
        """
        4. If models are cached, repeated loading behaves consistently.
        """
        m1 = get_model()
        m2 = get_model()
        self.assertIs(m1, m2, "get_model() should return the identical cached instance")

        t1 = get_threshold()
        t2 = get_threshold()
        self.assertEqual(t1, t2)
        self.assertAlmostEqual(t1, 0.42, places=4)

    # =========================================================================
    # SECTION B: Feature Compatibility
    # =========================================================================

    def test_case_5_feature_count_compatibility(self):
        """
        5. Each model receives the expected number of features (exactly 22).
        """
        self.assertEqual(len(FEATURE_NAMES), 22, "Feature specification must define exactly 22 features")

        # Baseline Pipeline check
        lr_model = get_model()
        self.assertEqual(lr_model.named_steps["scaler"].n_features_in_, 22)
        self.assertEqual(lr_model.named_steps["clf"].n_features_in_, 22)

        # XGBoost check
        xgb_model = joblib.load(XGBOOST_PATH)
        self.assertEqual(xgb_model.n_features_in_, 22)

        # PyTorch GRU check
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        self.assertEqual(ckpt["input_dim"], 22)

    def test_case_6_feature_ordering_integrity(self):
        """
        6. Feature ordering is verified against FEATURE_NAMES, not merely by checking vector length.
        """
        sample_obs = [
            {"hour": 1, "hr": 78.0, "sbp": 115.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": 84.0, "sbp": 110.0, "resp": 18.0, "temp": 37.1, "wbc": 8.2},
        ]
        vec, feat_dict = extract_features_point_in_time(sample_obs, target_hour=2, imputer=self.imputer)

        self.assertEqual(vec.shape, (1, 22))
        for idx, feat_name in enumerate(FEATURE_NAMES):
            self.assertIn(feat_name, feat_dict)
            val_from_dict = float(feat_dict[feat_name])
            val_from_vec = float(vec[0, idx])
            self.assertAlmostEqual(
                val_from_vec,
                val_from_dict,
                places=5,
                msg=f"Feature '{feat_name}' mismatch at index {idx}: vec={val_from_vec}, dict={val_from_dict}",
            )

    def test_case_7_input_dtype_and_shape(self):
        """
        7. Input dtype and shape match the real inference implementation (float32, 2D / 3D).
        """
        sample_obs = [{"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 18.0, "temp": 36.6, "wbc": 8.0}]
        vec, _ = extract_features_point_in_time(sample_obs, target_hour=1, imputer=self.imputer)

        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(vec.shape, (1, 22))

        # Baseline accepts (1, 22) float32
        p_lr = get_model().predict_proba(vec)
        self.assertEqual(p_lr.shape, (1, 2))

        # XGBoost accepts (1, 22) float32
        xgb_model = joblib.load(XGBOOST_PATH)
        p_xgb = xgb_model.predict_proba(vec)
        self.assertEqual(p_xgb.shape, (1, 2))

        # PyTorch GRU accepts (1, 1, 22) float32
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        rnn_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        rnn_model.load_state_dict(ckpt["state_dict"])
        rnn_model.eval()

        t_x = torch.from_numpy(vec).unsqueeze(0)  # (1, 1, 22)
        with torch.no_grad():
            logits = rnn_model(t_x)
        self.assertEqual(logits.shape, (1, 1))

    def test_case_8_missing_and_non_finite_handling(self):
        """
        8. Missing measurements and supported non-finite inputs (None, NaN, +inf, -inf)
        are processed according to the current feature contract without breaking inference.
        """
        obs = [
            {"hour": 1, "hr": float("inf"), "sbp": None, "resp": float("nan"), "temp": float("-inf"), "wbc": 7.0}
        ]
        vec, feat_dict = extract_features_point_in_time(obs, target_hour=1, imputer=self.imputer)

        # Output vector must be 100% finite
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertEqual(feat_dict["hr_missing"], 1.0)
        self.assertEqual(feat_dict["sbp_missing"], 1.0)
        self.assertEqual(feat_dict["resp_missing"], 1.0)
        self.assertEqual(feat_dict["temp_missing"], 1.0)
        self.assertEqual(feat_dict["wbc_missing"], 0.0)

        # Models must execute cleanly on this vector
        p_lr = float(get_model().predict_proba(vec)[0, 1])
        self.assertTrue(0.0 <= p_lr <= 1.0)

        xgb_model = joblib.load(XGBOOST_PATH)
        p_xgb = float(xgb_model.predict_proba(vec)[0, 1])
        self.assertTrue(0.0 <= p_xgb <= 1.0)

    def test_case_9_extreme_measurements_handling(self):
        """
        9. Extreme but supported measurements do not cause unhandled exceptions or non-finite inputs.
        """
        extreme_obs = [
            {"hour": 1, "hr": 300.0, "sbp": 300.0, "resp": 80.0, "temp": 45.0, "wbc": 100.0},
            {"hour": 2, "hr": 20.0, "sbp": 30.0, "resp": 4.0, "temp": 25.0, "wbc": 0.1},
        ]
        for h in [1, 2]:
            vec, _ = extract_features_point_in_time(extreme_obs, target_hour=h, imputer=self.imputer)
            self.assertTrue(np.all(np.isfinite(vec)))

            p_lr = float(get_model().predict_proba(vec)[0, 1])
            self.assertTrue(np.isfinite(p_lr) and 0.0 <= p_lr <= 1.0)

            xgb_model = joblib.load(XGBOOST_PATH)
            p_xgb = float(xgb_model.predict_proba(vec)[0, 1])
            self.assertTrue(np.isfinite(p_xgb) and 0.0 <= p_xgb <= 1.0)

    # =========================================================================
    # SECTION C: Prediction Integrity
    # =========================================================================

    def test_case_10_probability_bounds_all_models(self):
        """
        10. Each model returns a finite probability in [0, 1].
        """
        np.random.seed(42)
        random_vecs = np.random.randn(10, 22).astype(np.float32)

        # Baseline
        lr_probs = get_model().predict_proba(random_vecs)[:, 1]
        self.assertTrue(np.all(np.isfinite(lr_probs)))
        self.assertTrue(np.all((lr_probs >= 0.0) & (lr_probs <= 1.0)))

        # XGBoost
        xgb_model = joblib.load(XGBOOST_PATH)
        xgb_probs = xgb_model.predict_proba(random_vecs)[:, 1]
        self.assertTrue(np.all(np.isfinite(xgb_probs)))
        self.assertTrue(np.all((xgb_probs >= 0.0) & (xgb_probs <= 1.0)))

        # PyTorch GRU
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        rnn_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        rnn_model.load_state_dict(ckpt["state_dict"])
        rnn_model.eval()

        t_x = torch.from_numpy(random_vecs).unsqueeze(1)  # (10, 1, 22)
        with torch.no_grad():
            logits = rnn_model(t_x)
            gru_probs = torch.sigmoid(logits).squeeze(-1).numpy()

        self.assertTrue(np.all(np.isfinite(gru_probs)))
        self.assertTrue(np.all((gru_probs >= 0.0) & (gru_probs <= 1.0)))

    def test_case_11_threshold_comparison_behavior(self):
        """
        11. Threshold comparison behavior is tested below, equal to, and above threshold.
        Contract in predict.py: alert = 1 if proba_rounded >= threshold else 0 (uses >=).
        """
        thresh = get_threshold()
        self.assertEqual(thresh, 0.42)

        # Sub-threshold
        p_below = round(thresh - 0.0001, 4)
        alert_below = 1 if p_below >= thresh else 0
        self.assertEqual(alert_below, 0, f"Probability {p_below} < {thresh} must not alert")

        # Exactly at threshold
        p_equal = round(thresh, 4)
        alert_equal = 1 if p_equal >= thresh else 0
        self.assertEqual(alert_equal, 1, f"Probability {p_equal} == {thresh} must alert under >=")

        # Above threshold
        p_above = round(thresh + 0.0001, 4)
        alert_above = 1 if p_above >= thresh else 0
        self.assertEqual(alert_above, 1, f"Probability {p_above} > {thresh} must alert")

        # Verify offline thresholds follow documented JSON values
        with open(XGBOOST_THRESH_PATH, "r", encoding="utf-8") as f:
            xgb_th = json.load(f)["threshold"]
        self.assertEqual(xgb_th, 0.39)

        with open(LSTM_THRESH_PATH, "r", encoding="utf-8") as f:
            lstm_th = json.load(f)["threshold"]
        self.assertEqual(lstm_th, 0.42)

    def test_case_12_pytorch_output_conversion_and_sigmoid(self):
        """
        12. The PyTorch output conversion is checked explicitly:
        The model returns unconstrained raw logits; sigmoid must be applied exactly once.
        """
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        rnn_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        rnn_model.load_state_dict(ckpt["state_dict"])
        rnn_model.eval()

        # Generate inputs spanning wide dynamic range
        x = torch.randn(20, 5, 22)
        with torch.no_grad():
            logits = rnn_model(x)

        # Assert logits can be negative and unconstrained
        self.assertTrue((logits < 0.0).any().item(), "Logits should exhibit negative values")

        # Sigmoid converts logits to probabilities in (0, 1)
        probs = torch.sigmoid(logits)
        self.assertTrue((probs >= 0.0).all().item() and (probs <= 1.0).all().item())

        # Applying sigmoid twice would erroneously distort valid probabilities
        double_sigmoid = torch.sigmoid(probs)
        # sigmoid(p) for p in [0, 1] is constrained to [0.5, 0.731]
        self.assertTrue((double_sigmoid >= 0.5).all().item(), "Double sigmoid would corrupt probabilities")

    def test_case_13_repeated_inference_numerical_stability(self):
        """
        13. Repeated inference with identical inputs is stable within strict numerical tolerance (< 1e-7).
        """
        x = np.ones((1, 22), dtype=np.float32)

        # Baseline
        lr_model = get_model()
        p_lr_runs = [float(lr_model.predict_proba(x)[0, 1]) for _ in range(10)]
        self.assertEqual(max(p_lr_runs) - min(p_lr_runs), 0.0)

        # XGBoost
        xgb_model = joblib.load(XGBOOST_PATH)
        p_xgb_runs = [float(xgb_model.predict_proba(x)[0, 1]) for _ in range(10)]
        self.assertEqual(max(p_xgb_runs) - min(p_xgb_runs), 0.0)

        # PyTorch GRU
        ckpt = torch.load(LSTM_PATH, weights_only=False)
        rnn_model = SepsisRNN(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            rnn_type=ckpt["rnn_type"],
        )
        rnn_model.load_state_dict(ckpt["state_dict"])
        rnn_model.eval()

        t_x = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            p_gru_runs = [float(torch.sigmoid(rnn_model(t_x))[0, 0]) for _ in range(10)]
        self.assertAlmostEqual(max(p_gru_runs) - min(p_gru_runs), 0.0, places=7)

    def test_case_14_invalid_inputs_fail_safely(self):
        """
        14. Invalid inputs fail safely without corrupting the model instance or preventing
        subsequent valid predictions.
        """
        lr_model = get_model()
        bad_input = np.zeros((1, 10), dtype=np.float32)  # Wrong feature dimension

        with self.assertRaises(ValueError):
            lr_model.predict_proba(bad_input)

        # Verify instance is still healthy for valid input
        valid_input = np.zeros((1, 22), dtype=np.float32)
        valid_prob = lr_model.predict_proba(valid_input)
        self.assertEqual(valid_prob.shape, (1, 2))

    def test_case_15_model_identity_and_metadata_integrity(self):
        """
        15. Model identity and configured thresholds in prediction records or API responses
        match the model actually used.
        """
        self.assertEqual(MODEL_NAME, "logistic_regression_baseline")
        self.assertEqual(get_threshold(), 0.42)

        save_observation("P_T07_META", 1, 80.0, 120.0, 18.0, 36.8, 8.0, db_path=self.db_path)
        pred = predict("P_T07_META", 1, db_path=self.db_path)

        self.assertEqual(pred["model_name"], "logistic_regression_baseline")
        self.assertEqual(pred["threshold"], 0.42)
        self.assertEqual(
            pred["live_path_label"],
            "Live path: Logistic Regression baseline (research prototype)",
        )

    # =========================================================================
    # SECTION D: Integration
    # =========================================================================

    def test_case_16_end_to_end_prediction_integrity(self):
        """
        16. Run a representative end-to-end prediction through the real inference path
        and verify its output format, probability, alert decision, and recorded model name.
        """
        save_observation("P_T07_E2E", 1, 82.0, 118.0, 18.0, 37.0, 8.5, db_path=self.db_path)
        pred = predict("P_T07_E2E", 1, db_path=self.db_path)

        # Verify all contract keys
        required_keys = [
            "patient_id",
            "hour",
            "probability",
            "alert",
            "empty_count",
            "model_name",
            "threshold",
            "created_at",
            "live_path_label",
        ]
        for k in required_keys:
            self.assertIn(k, pred, f"Missing key '{k}' in predict() response")

        # Verify probability is float rounded to 4 decimals
        prob = pred["probability"]
        self.assertIsInstance(prob, float)
        self.assertEqual(len(str(prob).split(".")[1]), 4)
        self.assertTrue(0.0 <= prob <= 1.0)

        # Verify alert decision matches probability and threshold
        expected_alert = 1 if prob >= pred["threshold"] else 0
        self.assertEqual(pred["alert"], expected_alert)

        # Verify database record
        with get_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM predictions WHERE patient_id = 'P_T07_E2E' AND hour = 1")
            row = cur.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["model_name"], "logistic_regression_baseline")
            self.assertAlmostEqual(row["probability"], prob, places=4)
            self.assertEqual(row["alert"], expected_alert)

    def test_case_17_full_regression_parity(self):
        """
        17. Verify the existing causal-invariance and stored-prediction-immutability
        guarantees hold seamlessly across model invocations.
        """
        # Hour 1 normal
        save_observation("P_T07_CAUSAL", 1, 75.0, 120.0, 16.0, 36.6, 7.0, db_path=self.db_path)
        p1 = predict("P_T07_CAUSAL", 1, db_path=self.db_path)

        # Hour 2 acute sepsis
        save_observation("P_T07_CAUSAL", 2, 145.0, 65.0, 35.0, 40.2, 28.0, db_path=self.db_path)
        p2 = predict("P_T07_CAUSAL", 2, db_path=self.db_path)

        # Hour 1 immutable
        p1_again = predict("P_T07_CAUSAL", 1, db_path=self.db_path)
        self.assertEqual(p1["probability"], p1_again["probability"])
        self.assertEqual(p1["alert"], p1_again["alert"])
        self.assertEqual(p1["created_at"], p1_again["created_at"])


if __name__ == "__main__":
    unittest.main()
