"""
Test T09 — Temporal Evaluation and Robustness for SepsisSignal

Objectives:
1. Temporal partitioning and stability across ICU stay duration:
   - Early ICU Stay: 1–12 hours
   - Intermediate ICU Stay: 13–24 hours
   - Prolonged ICU Stay: >24 hours
2. Robustness to naturally occurring measurement missingness:
   - Evaluates handling of sparse monitoring (e.g. WBC drawn once daily, Temp every 4h).
3. Controlled perturbation stress tests:
   - Sensor noise (Gaussian jitter)
   - Measurement dropouts (simulated missing observations)
   - Extreme physiological spikes
4. Causal isolation under perturbation:
   - Perturbing future hours must strictly never mutate historical predictions.

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.features import FEATURE_NAMES, extract_features_point_in_time, load_imputer_values
from src.predict import get_model, get_threshold, predict
from src.store import init_db, save_observation


class TestT09TemporalRobustness(unittest.TestCase):
    """
    Exhaustive suite testing temporal subset partitioning, missingness resilience,
    controlled perturbation stress testing, and causal invariance.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "t09_test.db")
        init_db(self.db_path)
        self.imputer = load_imputer_values()
        self.model = get_model()
        self.threshold = get_threshold()

    def tearDown(self):
        for suffix in ["", "-wal", "-shm"]:
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_case_1_temporal_partitioning_disjoint_and_exhaustive(self):
        """
        1. Temporal partitioning:
        Verifies that ICU stay hours partition into Early (1-12h), Mid (13-24h),
        and Late (>24h) without overlaps or gaps.
        """
        stay_hours = list(range(1, 73))  # 72-hour ICU stay
        early = [h for h in stay_hours if h <= 12]
        mid = [h for h in stay_hours if 12 < h <= 24]
        late = [h for h in stay_hours if h > 24]

        # Exhaustive and disjoint
        self.assertEqual(len(early) + len(mid) + len(late), len(stay_hours))
        self.assertEqual(len(set(early) & set(mid)), 0)
        self.assertEqual(len(set(early) & set(late)), 0)
        self.assertEqual(len(set(mid) & set(late)), 0)

        self.assertEqual(early, list(range(1, 13)))
        self.assertEqual(mid, list(range(13, 25)))
        self.assertEqual(late, list(range(25, 73)))

    def test_case_2_sparse_missingness_robustness(self):
        """
        2. Sparse missingness robustness:
        Simulates realistic sparse ICU monitoring (e.g. only HR continuous,
        Temp intermittent, WBC only measured once at hour 1).
        Verifies causal forward fill and time-since feature behavior.
        """
        trajectory = []
        for h in range(1, 25):
            # HR every hour, SBP every hour, Temp every 6h, WBC only at hour 1
            hr = 75.0 + (h * 0.5)
            sbp = 120.0 - (h * 0.3)
            resp = 16.0 if h % 2 == 0 else None
            temp = 36.8 + (h * 0.05) if h % 6 == 1 else None
            wbc = 7.5 if h == 1 else None

            trajectory.append({
                "patient_id": "P_SPARSE_01",
                "hour": h,
                "hr": hr,
                "sbp": sbp,
                "resp": resp,
                "temp": temp,
                "wbc": wbc,
            })

            # Check point-in-time extraction at each hour
            vec, feat_dict = extract_features_point_in_time(trajectory, target_hour=h, imputer=self.imputer)

            # Vector must be strictly finite
            self.assertTrue(np.all(np.isfinite(vec)))

            # WBC time_since should equal (h - 1)
            self.assertEqual(feat_dict["wbc_time_since"], float(h - 1))
            self.assertEqual(feat_dict["wbc_val"], 7.5)

            # Temp forward filled from most recent measurement
            if h < 7:
                self.assertEqual(feat_dict["temp_time_since"], float(h - 1))
            else:
                expected_last_temp_h = (h - 1) // 6 * 6 + 1
                self.assertEqual(feat_dict["temp_time_since"], float(h - expected_last_temp_h))

            # Model prediction must remain bounded
            prob = float(self.model.predict_proba(vec)[0, 1])
            self.assertTrue(0.0 <= prob <= 1.0)

    def test_case_3_stress_test_measurement_dropout_perturbation(self):
        """
        3. Controlled Stress Test — Random Measurement Dropout:
        Artificially drops 40% of measured values across a 24-hour trajectory.
        Verifies model maintains finite predictions and graceful degradation.
        """
        np.random.seed(101)
        base_trajectory = [
            {"hour": h, "hr": 80.0 + h, "sbp": 120.0 - h, "resp": 16.0 + 0.5 * h, "temp": 36.8 + 0.05 * h, "wbc": 8.0}
            for h in range(1, 25)
        ]

        # Apply 40% dropout to signals
        dropped_trajectory = []
        for obs in base_trajectory:
            c = obs.copy()
            for s in ["hr", "sbp", "resp", "temp", "wbc"]:
                if np.random.rand() < 0.40:
                    c[s] = None
            dropped_trajectory.append(c)

        # Evaluate inference on every hour under dropout
        for h in range(1, 25):
            vec, _ = extract_features_point_in_time(dropped_trajectory, target_hour=h, imputer=self.imputer)
            self.assertTrue(np.all(np.isfinite(vec)))
            prob = float(self.model.predict_proba(vec)[0, 1])
            self.assertTrue(0.0 <= prob <= 1.0)

    def test_case_4_stress_test_physiological_noise_perturbation(self):
        """
        4. Controlled Stress Test — Physiological Sensor Noise (Gaussian Jitter):
        Adds +/-10% zero-mean Gaussian noise to physiological measurements.
        Verifies that prediction probabilities remain stable without numeric overflow.
        """
        np.random.seed(202)
        clean_trajectory = [
            {"hour": h, "hr": 85.0, "sbp": 115.0, "resp": 18.0, "temp": 37.0, "wbc": 8.5}
            for h in range(1, 13)
        ]

        noisy_trajectory = []
        for obs in clean_trajectory:
            c = obs.copy()
            c["hr"] += float(np.random.normal(0, 8.5))     # ~10% HR noise
            c["sbp"] += float(np.random.normal(0, 11.5))   # ~10% SBP noise
            c["resp"] += float(np.random.normal(0, 1.8))   # ~10% Resp noise
            c["temp"] += float(np.random.normal(0, 0.37))  # ~1% Temp noise
            c["wbc"] += float(np.random.normal(0, 0.85))   # ~10% WBC noise
            noisy_trajectory.append(c)

        for h in range(1, 13):
            vec_clean, _ = extract_features_point_in_time(clean_trajectory, target_hour=h, imputer=self.imputer)
            vec_noisy, _ = extract_features_point_in_time(noisy_trajectory, target_hour=h, imputer=self.imputer)

            self.assertTrue(np.all(np.isfinite(vec_noisy)))

            p_clean = float(self.model.predict_proba(vec_clean)[0, 1])
            p_noisy = float(self.model.predict_proba(vec_noisy)[0, 1])

            # Output remains valid probability and does not wildly diverge under moderate noise
            self.assertTrue(0.0 <= p_noisy <= 1.0)
            self.assertLess(abs(p_clean - p_noisy), 0.35, "Moderate noise caused excessive probability divergence")

    def test_case_5_extreme_sensor_spikes_safety(self):
        """
        5. Extreme sensor spikes safety:
        Injects sudden massive artifacts (e.g. SBP=350, HR=280, Temp=43.0).
        Verifies model does not crash or produce NaN/Inf.
        """
        spike_trajectory = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": 280.0, "sbp": 350.0, "resp": 75.0, "temp": 43.5, "wbc": 85.0},
        ]
        vec, _ = extract_features_point_in_time(spike_trajectory, target_hour=2, imputer=self.imputer)
        self.assertTrue(np.all(np.isfinite(vec)))

        prob = float(self.model.predict_proba(vec)[0, 1])
        self.assertTrue(np.isfinite(prob))
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertGreaterEqual(prob, self.threshold, "Severe physiological crisis must trigger alert")

    def test_case_6_causal_invariance_under_future_perturbation(self):
        """
        6. Causal invariance under future perturbation:
        Drastically altering hour 10 vitals must have EXACTLY ZERO effect on hour 5 prediction.
        """
        trajectory_a = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 5, "hr": 82.0, "sbp": 118.0, "resp": 17.0, "temp": 36.9, "wbc": 7.8},
            {"hour": 10, "hr": 85.0, "sbp": 115.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]

        # Trajectory B: hour 10 has extreme septic shock
        trajectory_b = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 5, "hr": 82.0, "sbp": 118.0, "resp": 17.0, "temp": 36.9, "wbc": 7.8},
            {"hour": 10, "hr": 160.0, "sbp": 60.0, "resp": 40.0, "temp": 41.0, "wbc": 35.0},
        ]

        vec_a, _ = extract_features_point_in_time(trajectory_a, target_hour=5, imputer=self.imputer)
        vec_b, _ = extract_features_point_in_time(trajectory_b, target_hour=5, imputer=self.imputer)

        np.testing.assert_array_equal(vec_a, vec_b)
        prob_a = float(self.model.predict_proba(vec_a)[0, 1])
        prob_b = float(self.model.predict_proba(vec_b)[0, 1])
        self.assertEqual(prob_a, prob_b)


if __name__ == "__main__":
    unittest.main()
