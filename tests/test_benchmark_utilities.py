"""
Automated Test Suite for Benchmark Utilities in SepsisSignal

Research prototype, not for clinical decision-making.

Verifies:
1. measure_latency_statistics correctness (warmup discard, percentiles monotonic p50 <= p95 <= p99)
2. profile_memory_consumption returns valid finite memory measurements
3. Isolated prediction helpers (predict_lr_isolated, predict_xgb_isolated, predict_gru_isolated)
4. verify_output_determinism_and_causality utility behavior
"""

import os
import unittest
import numpy as np
import torch

from scripts.benchmark_models import (
    measure_latency_statistics,
    profile_memory_consumption,
    predict_lr_isolated,
    predict_xgb_isolated,
    predict_gru_isolated,
    verify_output_determinism_and_causality,
)
from src.ensemble import SepsisEnsemble


class TestBenchmarkUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        cls.ensemble = SepsisEnsemble(models_dir=cls.models_dir)

    def setUp(self):
        self.sample_obs = [
            {"patient_id": "BENCH_P", "hour": 1, "hr": 80.0, "sbp": 120.0, "resp": 18.0, "temp": 36.5, "wbc": 7.0, "label": 0},
            {"patient_id": "BENCH_P", "hour": 2, "hr": 84.0, "sbp": 118.0, "resp": 19.0, "temp": 36.8, "wbc": 7.5, "label": 0},
            {"patient_id": "BENCH_P", "hour": 3, "hr": 90.0, "sbp": 112.0, "resp": 22.0, "temp": 37.2, "wbc": 8.5, "label": 0},
        ]

    def test_01_latency_measurement_statistics(self):
        """Verify latency statistics calculator returns valid monotonic percentiles."""
        dummy_fn = lambda: sum(i * i for i in range(100))
        stats = measure_latency_statistics(dummy_fn, n_warmup=5, n_reps=20)

        self.assertIn("median_ms", stats)
        self.assertIn("p95_ms", stats)
        self.assertIn("p99_ms", stats)
        self.assertGreater(stats["n_reps"], 0)
        self.assertGreater(stats["median_ms"], 0.0)
        # Monotonicity of percentiles
        self.assertLessEqual(stats["min_ms"], stats["median_ms"])
        self.assertLessEqual(stats["median_ms"], stats["p95_ms"])
        self.assertLessEqual(stats["p95_ms"], stats["p99_ms"])
        self.assertLessEqual(stats["p99_ms"], stats["max_ms"])

    def test_02_memory_consumption_profiling(self):
        """Verify memory profiler returns positive tracemalloc and RSS metrics."""
        dummy_fn = lambda: [x * 2 for x in range(1000)]
        mem = profile_memory_consumption(dummy_fn, n_iters=10)

        self.assertIn("tracemalloc_peak_kb", mem)
        self.assertIn("process_rss_after_mb", mem)
        self.assertGreater(mem["tracemalloc_peak_kb"], 0.0)
        self.assertGreater(mem["process_rss_after_mb"], 0.0)

    def test_03_isolated_prediction_contracts(self):
        """Verify isolated prediction helpers produce bounded probabilities in [0.0, 1.0]."""
        p_lr = predict_lr_isolated(self.sample_obs, 3, self.ensemble.lr_model, self.ensemble.imputer)
        p_xgb = predict_xgb_isolated(self.sample_obs, 3, self.ensemble.xgb_model, self.ensemble.imputer)
        p_gru = predict_gru_isolated(
            self.sample_obs, 3, self.ensemble.gru_model,
            self.ensemble.train_mean, self.ensemble.train_std, self.ensemble.imputer
        )

        for name, prob in [("LR", p_lr), ("XGB", p_xgb), ("GRU", p_gru)]:
            self.assertTrue(0.0 <= prob <= 1.0, f"{name} probability out of bounds: {prob}")
            self.assertTrue(np.isfinite(prob), f"{name} probability non-finite: {prob}")

    def test_04_determinism_and_causality_verification(self):
        """Verify verification helper correctly tests output determinism and future invariance."""
        res = verify_output_determinism_and_causality(self.ensemble, self.sample_obs)
        self.assertTrue(res["is_deterministic"])
        self.assertTrue(res["is_causally_invariant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
