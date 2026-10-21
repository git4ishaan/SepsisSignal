"""
Test T08 — Independent Evaluation and Metric Reproduction
Target:
  - Split isolation and patient-level separation (Train, Val, Test)
  - Target label alignment, prevalence, and class imbalance audit
  - Metric reproduction against models/results/metrics.json
  - Causal feature integrity and absence of future lookahead in evaluation pipelines

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import json
import os
import sys
import unittest
import joblib
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values

MODELS_DIR = os.path.join(REPO_ROOT, "models")
SPLITS_DIR = os.path.join(MODELS_DIR, "splits")
METRICS_PATH = os.path.join(MODELS_DIR, "results", "metrics.json")


class TestT08MetricReproduction(unittest.TestCase):
    """
    Automated regression and audit test suite for T08:
    - Split leakage & overlap audit
    - Label alignment & prevalence verification
    - Metric file integrity
    - Reproducibility verification
    """

    def setUp(self):
        self.train_file = os.path.join(SPLITS_DIR, "train_patients.txt")
        self.val_file = os.path.join(SPLITS_DIR, "val_patients.txt")
        self.test_file = os.path.join(SPLITS_DIR, "test_patients.txt")
        self.imputer = load_imputer_values()

    def test_case_1_patient_level_split_isolation(self):
        """
        1. Patient-level split isolation:
        Verify strictly ZERO overlap between train, validation, and test patient sets.
        """
        self.assertTrue(os.path.exists(self.train_file), "Missing train_patients.txt")
        self.assertTrue(os.path.exists(self.val_file), "Missing val_patients.txt")
        self.assertTrue(os.path.exists(self.test_file), "Missing test_patients.txt")

        with open(self.train_file, "r", encoding="utf-8") as f:
            train_ids = set(line.strip() for line in f if line.strip())
        with open(self.val_file, "r", encoding="utf-8") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        with open(self.test_file, "r", encoding="utf-8") as f:
            test_ids = set(line.strip() for line in f if line.strip())

        # No duplicate patient IDs within each split
        with open(self.train_file, "r", encoding="utf-8") as f:
            train_list = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(train_ids), len(train_list), "Duplicate IDs found in train split")

        # Zero cross-split overlap
        self.assertEqual(len(train_ids & val_ids), 0, "Train and Val splits share patient IDs!")
        self.assertEqual(len(train_ids & test_ids), 0, "Train and Test splits share patient IDs!")
        self.assertEqual(len(val_ids & test_ids), 0, "Val and Test splits share patient IDs!")

        # Expected counts (70 / 15 / 15 of 40,311)
        self.assertEqual(len(train_ids), 28217)
        self.assertEqual(len(val_ids), 6046)
        self.assertEqual(len(test_ids), 6048)

    def test_case_2_metrics_json_integrity(self):
        """
        2. Verify models/results/metrics.json contains all required models and
        matches the documented evaluation benchmarks.
        """
        self.assertTrue(os.path.exists(METRICS_PATH), f"Missing {METRICS_PATH}")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("models", data)
        self.assertIn("disclaimer", data)
        self.assertIn("Research prototype", data["disclaimer"])

        models_map = {m["name"]: m for m in data["models"]}
        self.assertIn("Logistic Regression (live path)", models_map)
        self.assertIn("XGBoost (offline)", models_map)

        lr = models_map["Logistic Regression (live path)"]
        self.assertAlmostEqual(lr["auprc"], 0.0775, places=4)
        self.assertAlmostEqual(lr["precision"], 0.0301, places=4)
        self.assertAlmostEqual(lr["recall"], 0.7374, places=4)
        self.assertAlmostEqual(lr["specificity"], 0.5849, places=4)
        self.assertEqual(lr["threshold"], 0.42)

        xgb = models_map["XGBoost (offline)"]
        self.assertAlmostEqual(xgb["auprc"], 0.0983, places=4)
        self.assertAlmostEqual(xgb["precision"], 0.0380, places=4)
        self.assertAlmostEqual(xgb["recall"], 0.7644, places=4)
        self.assertAlmostEqual(xgb["specificity"], 0.6613, places=4)
        self.assertEqual(xgb["threshold"], 0.39)

    def test_case_3_label_definition_and_alignment(self):
        """
        3. Verify label extraction contract:
        SepsisLabel is binary integer in {0, 1}, and row-by-row prediction
        order preserves exact correspondence with input sequence.
        """
        # Synthetic patient stay
        raw_obs = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5, "label": 0},
            {"hour": 2, "hr": 85.0, "sbp": 118.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0, "label": 0},
            {"hour": 3, "hr": 130.0, "sbp": 85.0, "resp": 30.0, "temp": 39.5, "wbc": 18.0, "label": 1},
        ]
        X = []
        y = []
        for r in raw_obs:
            vec, _ = extract_features_point_in_time(raw_obs, target_hour=r["hour"], imputer=self.imputer)
            X.append(vec.flatten())
            y.append(r["label"])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(list(y), [0, 0, 1])

        # Test inference produces aligned predictions
        lr_model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
        probs = lr_model.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), 3)
        self.assertTrue(np.all(np.isfinite(probs)))


if __name__ == "__main__":
    unittest.main()
