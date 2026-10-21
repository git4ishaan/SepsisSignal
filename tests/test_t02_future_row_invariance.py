"""
Test T02 — Future-Row Invariance & Causal Feature Extraction Invariants
Target: extract_features_point_in_time(observations, target_hour, imputer=None) in src/features.py

Objective:
Verify that features computed for a target hour T never depend on observations from later hours (hour > T),
preserve input immutability, ensure deterministic feature ordering, and evaluate non-finite value handling.

Safety Disclaimer:
Research prototype, not for clinical decision-making.
"""

import copy
import hashlib
import os
import random
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path for pytest and direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    FEATURE_NAMES,
    SIGNALS,
    extract_features_point_in_time,
    load_imputer_values,
)


def _compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file for strict content integrity checks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestFutureRowInvariance(unittest.TestCase):
    """
    Test suite verifying strict point-in-time causality, feature-order integrity,
    input immutability, and boundary invariants in extract_features_point_in_time().
    """

    @classmethod
    def setUpClass(cls):
        cls.imputer = load_imputer_values()
        cls.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        # Record SHA-256 content hashes of model artifacts to assert zero mutation
        cls.artifact_paths = [
            os.path.join(cls.models_dir, "model.joblib"),
            os.path.join(cls.models_dir, "threshold.json"),
            os.path.join(cls.models_dir, "imputer.json"),
        ]
        cls.initial_hashes = {
            p: _compute_file_sha256(p) for p in cls.artifact_paths if os.path.exists(p)
        }

    @classmethod
    def tearDownClass(cls):
        # Assert model artifacts were never modified, overwritten, or retrained
        for p, init_hash in cls.initial_hashes.items():
            current_hash = _compute_file_sha256(p)
            assert current_hash == init_hash, (
                f"Model artifact {p} was mutated during test execution! "
                f"Initial SHA-256: {init_hash}, Current SHA-256: {current_hash}"
            )

    def _get_base_history(self):
        """Create deterministic synthetic patient history containing observations at hours 1 and 2."""
        return [
            {
                "patient_id": "TEST_P001",
                "hour": 1,
                "hr": 80.0,
                "sbp": 120.0,
                "resp": 16.0,
                "temp": 36.8,
                "wbc": 7.5,
            },
            {
                "patient_id": "TEST_P001",
                "hour": 2,
                "hr": 84.0,
                "sbp": 118.0,
                "resp": 18.0,
                "temp": 37.0,
                "wbc": 8.0,
            },
        ]

    def _get_future_observations(self):
        """Observations for hours 3, 4, 5 with deliberately different vital values."""
        return [
            {
                "patient_id": "TEST_P001",
                "hour": 3,
                "hr": 110.0,
                "sbp": 95.0,
                "resp": 24.0,
                "temp": 38.5,
                "wbc": 14.0,
            },
            {
                "patient_id": "TEST_P001",
                "hour": 4,
                "hr": 130.0,
                "sbp": 85.0,
                "resp": 28.0,
                "temp": 39.2,
                "wbc": 18.5,
            },
            {
                "patient_id": "TEST_P001",
                "hour": 5,
                "hr": 145.0,
                "sbp": 75.0,
                "resp": 34.0,
                "temp": 39.8,
                "wbc": 22.0,
            },
        ]

    # -------------------------------------------------------------------------
    # Existing Tests (Preserved)
    # -------------------------------------------------------------------------

    def test_base_future_row_invariance(self):
        """
        Primary Test T02:
        1. Create deterministic history at hours 1 and 2.
        2. Compute vector for target_hour=2.
        3. Append hours 3, 4, 5 with different vitals.
        4. Recompute vector for target_hour=2.
        5. Verify shape, dtype, and exact equality across all 22 features.
        """
        base_history = self._get_base_history()
        vec_base, dict_base = extract_features_point_in_time(
            base_history, target_hour=2, imputer=self.imputer
        )

        # Assert shape and dtype
        self.assertEqual(vec_base.shape, (1, 22), "Vector must have shape (1, 22)")
        self.assertEqual(vec_base.dtype, np.float32, "Vector must be float32")
        self.assertEqual(len(dict_base), 22, "Feature dict must contain exactly 22 keys")

        # Extend history with hours 3, 4, 5
        extended_history = copy.deepcopy(base_history) + self._get_future_observations()
        vec_extended, dict_extended = extract_features_point_in_time(
            extended_history, target_hour=2, imputer=self.imputer
        )

        # Assert shape and dtype on extended vector
        self.assertEqual(vec_extended.shape, (1, 22), "Extended vector must have shape (1, 22)")
        self.assertEqual(vec_extended.dtype, np.float32, "Extended vector must be float32")

        # Required Assertions: Vectors must be strictly identical
        np.testing.assert_array_equal(
            vec_base,
            vec_extended,
            err_msg="Future observations at hours 3, 4, 5 altered features at hour 2!",
        )

        # Check every individual feature name
        for fname in FEATURE_NAMES:
            self.assertIn(fname, dict_base)
            self.assertIn(fname, dict_extended)
            self.assertEqual(
                dict_base[fname],
                dict_extended[fname],
                f"Feature '{fname}' differed between base and extended history!",
            )

        # Specific feature value assertions for hour 2
        self.assertEqual(dict_base["hour"], 2.0)
        self.assertEqual(dict_base["current_empty_count"], 0.0)
        self.assertEqual(dict_base["hr_val"], 84.0)
        self.assertEqual(dict_base["hr_delta"], 4.0)
        self.assertEqual(dict_base["hr_time_since"], 0.0)
        self.assertEqual(dict_base["hr_missing"], 0.0)

    def test_future_observations_with_missing_values(self):
        """Additional Case 1: Future observations contain NaN / None for all signals."""
        base_history = self._get_base_history()
        vec_base, dict_base = extract_features_point_in_time(
            base_history, target_hour=2, imputer=self.imputer
        )

        future_missing = [
            {
                "patient_id": "TEST_P001",
                "hour": 3,
                "hr": None,
                "sbp": np.nan,
                "resp": None,
                "temp": np.nan,
                "wbc": None,
            },
            {
                "patient_id": "TEST_P001",
                "hour": 4,
                "hr": np.nan,
                "sbp": None,
                "resp": np.nan,
                "temp": None,
                "wbc": np.nan,
            },
        ]

        history_with_missing = copy.deepcopy(base_history) + future_missing
        vec_missing, dict_missing = extract_features_point_in_time(
            history_with_missing, target_hour=2, imputer=self.imputer
        )

        np.testing.assert_array_equal(
            vec_base,
            vec_missing,
            err_msg="Future missing observations altered hour 2 features!",
        )
        self.assertEqual(dict_base, dict_missing)

    def test_future_observations_with_extreme_finite_values(self):
        """Additional Case 2: Future observations contain extreme physiological values."""
        base_history = self._get_base_history()
        vec_base, dict_base = extract_features_point_in_time(
            base_history, target_hour=2, imputer=self.imputer
        )

        future_extreme = [
            {
                "patient_id": "TEST_P001",
                "hour": 3,
                "hr": 300.0,  # extreme tachycardia
                "sbp": 10.0,   # extreme hypotension
                "resp": 90.0,  # extreme tachypnea
                "temp": 45.0,  # fatal hyperthermia
                "wbc": 250.0,  # extreme leukocytosis
            },
            {
                "patient_id": "TEST_P001",
                "hour": 4,
                "hr": 15.0,   # extreme bradycardia
                "sbp": 250.0,  # extreme hypertensive crisis
                "resp": 4.0,   # agonal respirations
                "temp": 28.0,  # deep hypothermia
                "wbc": 0.1,    # severe leukopenia
            },
        ]

        history_with_extreme = copy.deepcopy(base_history) + future_extreme
        vec_extreme, dict_extreme = extract_features_point_in_time(
            history_with_extreme, target_hour=2, imputer=self.imputer
        )

        np.testing.assert_array_equal(
            vec_base,
            vec_extreme,
            err_msg="Future extreme observations altered hour 2 features!",
        )
        self.assertEqual(dict_base, dict_extreme)

    def test_shuffled_input_observations(self):
        """
        Additional Case 3: Observations are supplied in random / non-chronological order.
        Causal filtering and sorting must ensure invariance regardless of list order.
        """
        base_history = self._get_base_history()
        vec_base, dict_base = extract_features_point_in_time(
            base_history, target_hour=2, imputer=self.imputer
        )

        full_history = copy.deepcopy(base_history) + self._get_future_observations()

        # Test reverse order: [hour 5, hour 4, hour 3, hour 2, hour 1]
        reverse_history = list(reversed(full_history))
        vec_reverse, dict_reverse = extract_features_point_in_time(
            reverse_history, target_hour=2, imputer=self.imputer
        )
        np.testing.assert_array_equal(
            vec_base,
            vec_reverse,
            err_msg="Reversed observation order altered hour 2 features!",
        )

        # Test multiple random permutations
        rng = random.Random(42)
        for trial in range(5):
            shuffled_history = copy.deepcopy(full_history)
            rng.shuffle(shuffled_history)
            vec_shuffled, dict_shuffled = extract_features_point_in_time(
                shuffled_history, target_hour=2, imputer=self.imputer
            )
            np.testing.assert_array_equal(
                vec_base,
                vec_shuffled,
                err_msg=f"Shuffled order (trial {trial}) altered hour 2 features!",
            )
            self.assertEqual(dict_base, dict_shuffled)

    def test_no_observation_at_target_hour(self):
        """
        Additional Case 4: No observation exists exactly at target_hour (e.g. target_hour=2, but only hour 1 exists).
        Future observations (hours 3, 4, 5) must not leak into hour 2 features.
        """
        history_h1_only = [
            {
                "patient_id": "TEST_P002",
                "hour": 1,
                "hr": 80.0,
                "sbp": 120.0,
                "resp": 16.0,
                "temp": 36.8,
                "wbc": 7.5,
            }
        ]

        vec_h2_base, dict_h2_base = extract_features_point_in_time(
            history_h1_only, target_hour=2, imputer=self.imputer
        )

        self.assertEqual(dict_h2_base["hour"], 2.0)
        self.assertEqual(dict_h2_base["current_empty_count"], 5.0)  # all 5 signals unmeasured at hour 2
        for sig in SIGNALS:
            self.assertEqual(dict_h2_base[f"{sig}_missing"], 1.0)
            self.assertEqual(dict_h2_base[f"{sig}_time_since"], 1.0)
            self.assertEqual(dict_h2_base[f"{sig}_delta"], 0.0)

        # Values should be causally forward-filled from hour 1
        self.assertEqual(dict_h2_base["hr_val"], 80.0)
        self.assertEqual(dict_h2_base["sbp_val"], 120.0)
        self.assertEqual(dict_h2_base["resp_val"], 16.0)
        self.assertEqual(dict_h2_base["temp_val"], 36.8)
        self.assertEqual(dict_h2_base["wbc_val"], 7.5)

        # Append observations at hours 3 and 5
        future_obs = [
            {
                "patient_id": "TEST_P002",
                "hour": 3,
                "hr": 125.0,
                "sbp": 80.0,
                "resp": 30.0,
                "temp": 39.5,
                "wbc": 20.0,
            },
            {
                "patient_id": "TEST_P002",
                "hour": 5,
                "hr": 140.0,
                "sbp": 70.0,
                "resp": 35.0,
                "temp": 40.1,
                "wbc": 25.0,
            },
        ]
        history_h1_and_future = history_h1_only + future_obs

        vec_h2_extended, dict_h2_extended = extract_features_point_in_time(
            history_h1_and_future, target_hour=2, imputer=self.imputer
        )

        np.testing.assert_array_equal(
            vec_h2_base,
            vec_h2_extended,
            err_msg="Future observations leaked into hour 2 when hour 2 was unmeasured!",
        )
        self.assertEqual(dict_h2_base, dict_h2_extended)

    # -------------------------------------------------------------------------
    # Refinement Tests: T02.6, T02.7, T02.8, T02.9
    # -------------------------------------------------------------------------

    def test_t02_6_future_value_mutation(self):
        """
        T02.6 — Future-value mutation:
        - Compute features for fixed target_hour=2.
        - Deep-copy observations and mutate ONLY the values in rows strictly after target_hour (hours 3, 4, 5).
        - Assert that the feature vector remains strictly identical.
        """
        full_history = self._get_base_history() + self._get_future_observations()
        vec_original, dict_original = extract_features_point_in_time(
            full_history, target_hour=2, imputer=self.imputer
        )

        # Deep-copy and mutate ONLY rows with hour > 2
        mutated_history = copy.deepcopy(full_history)
        for row in mutated_history:
            if int(row["hour"]) > 2:
                # Mutate every vital and inject arbitrary fields
                row["hr"] = 999.0
                row["sbp"] = 1.0
                row["resp"] = 88.0
                row["temp"] = 43.0
                row["wbc"] = 99.0
                row["unrelated_sensor_data"] = 12345.0

        vec_mutated, dict_mutated = extract_features_point_in_time(
            mutated_history, target_hour=2, imputer=self.imputer
        )

        np.testing.assert_array_equal(
            vec_original,
            vec_mutated,
            err_msg="T02.6 Failed: Mutating values in rows after target_hour changed target_hour features!",
        )
        self.assertEqual(dict_original, dict_mutated)

    def test_t02_7_non_finite_values_at_or_before_target(self):
        """
        T02.7 — Non-finite values at or before the target:
        Tests None, NaN, positive infinity (+inf), and negative infinity (-inf)
        at or before the target hour.

        Documented Contract Behavior:
        1. None and NaN:
           - Filter condition: `if v is not None and not (isinstance(v, float) and np.isnan(v))`
           - Evaluates to False -> Recognized as missing (`sig_missing == 1.0`).
           - Falls back to causal forward-fill or training-set imputer value.
           - `current_empty_count` increments.
        2. Positive Infinity (+inf) and Negative Infinity (-inf):
           - In IEEE 754, `isinstance(inf, float)` is True and `np.isnan(inf)` is False.
           - As implemented, `inf` and `-inf` are treated as valid float measurements (`sig_missing == 0.0`),
             causing `inf` to enter the feature dictionary and numpy array as `np.float32(inf)`.
           - This test verifies this actual implementation behavior and asserts the contract boundary.
        """
        # Case A: Missing values via None at target_hour
        history_none = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": None, "sbp": 118.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        vec_none, dict_none = extract_features_point_in_time(
            history_none, target_hour=2, imputer=self.imputer
        )
        self.assertEqual(dict_none["hr_missing"], 1.0, "None should set hr_missing to 1.0")
        self.assertEqual(dict_none["hr_val"], 80.0, "None at hour 2 should forward-fill from hour 1 (80.0)")
        self.assertEqual(dict_none["hr_time_since"], 1.0, "Time since last measurement should be 1.0 hour")
        self.assertEqual(dict_none["current_empty_count"], 1.0, "current_empty_count should be 1.0")

        # Case B: Missing values via float('nan') / np.nan at target_hour
        history_nan = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": float("nan"), "sbp": np.nan, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        vec_nan, dict_nan = extract_features_point_in_time(
            history_nan, target_hour=2, imputer=self.imputer
        )
        self.assertEqual(dict_nan["hr_missing"], 1.0, "NaN should set hr_missing to 1.0")
        self.assertEqual(dict_nan["sbp_missing"], 1.0, "NaN should set sbp_missing to 1.0")
        self.assertEqual(dict_nan["hr_val"], 80.0, "NaN should forward-fill hr from hour 1")
        self.assertEqual(dict_nan["sbp_val"], 120.0, "NaN should forward-fill sbp from hour 1")
        self.assertEqual(dict_nan["current_empty_count"], 2.0, "current_empty_count should be 2.0")

        # Case C: Positive infinity (+inf) in current observation
        history_inf = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": float("inf"), "sbp": 118.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        vec_inf, dict_inf = extract_features_point_in_time(
            history_inf, target_hour=2, imputer=self.imputer
        )
        # +inf is non-finite, must be treated as missing and forward-filled
        self.assertEqual(dict_inf["hr_missing"], 1.0, "+inf must set hr_missing to 1.0")
        self.assertEqual(dict_inf["hr_val"], 80.0, "+inf must forward-fill from hour 1 (80.0)")
        self.assertEqual(dict_inf["hr_time_since"], 1.0, "Time since last finite reading is 1.0 hour")
        self.assertEqual(dict_inf["current_empty_count"], 1.0, "+inf counts as empty")
        self.assertTrue(np.all(np.isfinite(vec_inf)), "Output vector must be strictly finite")

        # Case D: Negative infinity (-inf) in current observation
        history_ninf = [
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 2, "hr": float("-inf"), "sbp": 118.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        vec_ninf, dict_ninf = extract_features_point_in_time(
            history_ninf, target_hour=2, imputer=self.imputer
        )
        # -inf is non-finite, must be treated as missing and forward-filled
        self.assertEqual(dict_ninf["hr_missing"], 1.0, "-inf must set hr_missing to 1.0")
        self.assertEqual(dict_ninf["hr_val"], 80.0, "-inf must forward-fill from hour 1 (80.0)")
        self.assertEqual(dict_ninf["hr_time_since"], 1.0, "Time since last finite reading is 1.0 hour")
        self.assertEqual(dict_ninf["current_empty_count"], 1.0, "-inf counts as empty")
        self.assertTrue(np.all(np.isfinite(vec_ninf)), "Output vector must be strictly finite")

    def test_t02_8_feature_order_integrity(self):
        """
        T02.8 — Feature-order integrity:
        - Verify that the vector's 22 positions correspond exactly to FEATURE_NAMES.
        - Where the dictionary and vector are both returned, assert that each vector element
          matches the corresponding named feature, using the implementation's documented ordering.
        """
        base_history = self._get_base_history()
        vec, dict_features = extract_features_point_in_time(
            base_history, target_hour=2, imputer=self.imputer
        )

        # Documented feature order: 5 signals * 4 features + hour + current_empty_count = 22
        expected_feature_names = [
            "hr_val", "hr_missing", "hr_time_since", "hr_delta",
            "sbp_val", "sbp_missing", "sbp_time_since", "sbp_delta",
            "resp_val", "resp_missing", "resp_time_since", "resp_delta",
            "temp_val", "temp_missing", "temp_time_since", "temp_delta",
            "wbc_val", "wbc_missing", "wbc_time_since", "wbc_delta",
            "hour", "current_empty_count",
        ]

        # 1. Assert FEATURE_NAMES contract
        self.assertEqual(FEATURE_NAMES, expected_feature_names)
        self.assertEqual(len(FEATURE_NAMES), 22)
        self.assertEqual(vec.shape, (1, 22))

        # 2. Assert exact element-by-element correspondence between vector and dict
        for idx, feat_name in enumerate(FEATURE_NAMES):
            expected_val = dict_features[feat_name]
            actual_vector_val = vec[0, idx]
            self.assertEqual(
                actual_vector_val,
                np.float32(expected_val),
                f"Feature vector mismatch at index {idx} ({feat_name}): "
                f"Vector={actual_vector_val}, Dict={expected_val}",
            )

    def test_t02_9_input_immutability(self):
        """
        T02.9 — Input immutability:
        - Deep-copy original observations.
        - Run extract_features_point_in_time.
        - Verify that the input records, dictionary keys, and ordering remain completely unchanged.
        """
        original_observations = [
            {"hour": 3, "hr": 95.0, "sbp": 115.0, "resp": 20.0, "temp": 37.2, "wbc": 9.0},
            {"hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 16.0, "temp": 36.8, "wbc": 7.5},
            {"hour": 5, "hr": 110.0, "sbp": 100.0, "resp": 22.0, "temp": 38.0, "wbc": 12.0},
            {"hour": 2, "hr": 85.0, "sbp": 118.0, "resp": 18.0, "temp": 37.0, "wbc": 8.0},
        ]
        # Make a deep copy to compare against
        observations_snapshot = copy.deepcopy(original_observations)

        # Run feature extraction for target_hour=2
        vec, dict_feats = extract_features_point_in_time(
            original_observations, target_hour=2, imputer=self.imputer
        )

        # Assert input list length is unchanged
        self.assertEqual(len(original_observations), len(observations_snapshot))

        # Assert order of elements in input list is strictly preserved (sort must not mutate input)
        for i in range(len(original_observations)):
            self.assertEqual(
                original_observations[i],
                observations_snapshot[i],
                f"Observation at index {i} was mutated by feature extraction!",
            )

        # Exact deep equality
        self.assertEqual(original_observations, observations_snapshot)


if __name__ == "__main__":
    unittest.main()
