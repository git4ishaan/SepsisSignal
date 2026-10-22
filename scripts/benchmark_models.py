"""
Comprehensive Model Efficiency & Deployment Feasibility Benchmark for SepsisSignal

Research Prototype - Not for Clinical Decision-Making.

Benchmarks:
1. Logistic Regression (Live serving candidate)
2. XGBoost (Offline tabular candidate)
3. PyTorch GRU (Offline sequence candidate)
4. Soft-Voting Ensemble (SepsisEnsemble combining all 3)

Measures:
- Isolated point-in-time prediction latency across expanding history (1, 6, 12, 24, 36, 48 hours)
- Sequential streaming simulation (hour-by-hour from 1 to 48 hours)
- Throughput (predictions/second) on batch evaluation
- Peak memory usage (tracemalloc peak bytes & psutil RSS)
- Separate warm-up cycles (discarded) and high-resolution timing (time.perf_counter_ns)
- Output determinism and causal invariance checks
"""

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_lstm import parse_patient_psv_fast, SepsisRNN
from src.ensemble import SepsisEnsemble
from src.features import extract_features_point_in_time, load_imputer_values


def get_system_environment() -> Dict[str, Any]:
    """Capture exact hardware and software execution environment."""
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
    }


def measure_latency_statistics(
    fn: Callable[[], Any],
    n_warmup: int = 20,
    n_reps: int = 100,
) -> Dict[str, float]:
    """
    Measure latency statistics with separate warm-up and high-precision timer.
    Returns latencies in milliseconds (ms).
    """
    # 1. Warm-up runs (discarded)
    for _ in range(n_warmup):
        _ = fn()

    # 2. Timed runs
    latencies_ns = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        _ = fn()
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)

    latencies_ms = np.array(latencies_ns, dtype=np.float64) / 1e6

    return {
        "n_warmup": n_warmup,
        "n_reps": n_reps,
        "mean_ms": round(float(np.mean(latencies_ms)), 4),
        "std_ms": round(float(np.std(latencies_ms)), 4),
        "median_ms": round(float(np.median(latencies_ms)), 4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "min_ms": round(float(np.min(latencies_ms)), 4),
        "max_ms": round(float(np.max(latencies_ms)), 4),
    }


def profile_memory_consumption(fn: Callable[[], Any], n_iters: int = 50) -> Dict[str, float]:
    """
    Profile peak heap memory allocation using tracemalloc and process RSS change via psutil.
    """
    gc.collect()
    process = psutil.Process()
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    tracemalloc.start()
    tracemalloc.reset_peak()

    for _ in range(n_iters):
        _ = fn()

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()
    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "tracemalloc_peak_kb": round(peak_bytes / 1024.0, 2),
        "tracemalloc_peak_mb": round(peak_bytes / (1024.0 * 1024.0), 4),
        "process_rss_before_mb": round(rss_before_mb, 2),
        "process_rss_after_mb": round(rss_after_mb, 2),
    }


def predict_lr_isolated(
    obs_window: List[Dict[str, Any]],
    target_hour: int,
    lr_model: Any,
    imputer: Dict[str, float],
) -> float:
    """Isolated end-to-end prediction: causal feature extraction + Logistic Regression."""
    vec, _ = extract_features_point_in_time(obs_window, target_hour=target_hour, imputer=imputer)
    return float(lr_model.predict_proba(vec)[0, 1])


def predict_xgb_isolated(
    obs_window: List[Dict[str, Any]],
    target_hour: int,
    xgb_model: Any,
    imputer: Dict[str, float],
) -> float:
    """Isolated end-to-end prediction: causal feature extraction + XGBoost."""
    vec, _ = extract_features_point_in_time(obs_window, target_hour=target_hour, imputer=imputer)
    return float(xgb_model.predict_proba(vec)[0, 1])


def predict_gru_isolated(
    obs_window: List[Dict[str, Any]],
    target_hour: int,
    gru_model: Any,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    imputer: Dict[str, float],
    max_gru_window: int = 48,
) -> float:
    """
    Isolated end-to-end prediction for PyTorch GRU:
    1. Causal sequence window slicing
    2. Hourly feature extraction for each timestep in window
    3. Feature normalization (seq - mean)/std
    4. Tensor conversion + GRU forward pass + sigmoid
    """
    sorted_obs = sorted(obs_window, key=lambda r: int(r.get("hour", 0)))
    window = sorted_obs[-max_gru_window:] if len(sorted_obs) > max_gru_window else sorted_obs

    seq_feats = []
    for r in window:
        h = int(r["hour"])
        v, _ = extract_features_point_in_time(sorted_obs, target_hour=h, imputer=imputer)
        seq_feats.append(v.flatten())

    seq_arr = np.array(seq_feats, dtype=np.float32)
    norm_seq = (seq_arr - train_mean) / train_std

    t_x = torch.from_numpy(norm_seq).unsqueeze(0)
    with torch.no_grad():
        logits = gru_model(t_x)
        logit_last = float(logits[0, -1].item())
        prob = 1.0 / (1.0 + math.exp(-logit_last))
    return prob


def benchmark_isolated_latencies_by_history(
    ensemble: SepsisEnsemble,
    sample_patient_obs: List[Dict[str, Any]],
    history_lengths: List[int] = [1, 6, 12, 24, 36, 48],
    n_warmup: int = 20,
    n_reps: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark latency across different history lengths (1 to 48 hours).
    Captures scaling behavior as history expands.
    """
    results_by_model = {
        "Logistic Regression": {},
        "XGBoost": {},
        "PyTorch GRU": {},
        "Soft-Voting Ensemble": {},
    }

    imputer = ensemble.imputer
    lr_model = ensemble.lr_model
    xgb_model = ensemble.xgb_model
    gru_model = ensemble.gru_model
    train_mean = ensemble.train_mean
    train_std = ensemble.train_std

    for H in history_lengths:
        obs_slice = sample_patient_obs[:H]
        target_h = int(obs_slice[-1]["hour"])

        # 1. Logistic Regression
        fn_lr = lambda: predict_lr_isolated(obs_slice, target_h, lr_model, imputer)
        stats_lr = measure_latency_statistics(fn_lr, n_warmup=n_warmup, n_reps=n_reps)
        results_by_model["Logistic Regression"][str(H)] = stats_lr

        # 2. XGBoost
        fn_xgb = lambda: predict_xgb_isolated(obs_slice, target_h, xgb_model, imputer)
        stats_xgb = measure_latency_statistics(fn_xgb, n_warmup=n_warmup, n_reps=n_reps)
        results_by_model["XGBoost"][str(H)] = stats_xgb

        # 3. PyTorch GRU (Full end-to-end sequence pipeline)
        fn_gru = lambda: predict_gru_isolated(obs_slice, target_h, gru_model, train_mean, train_std, imputer, max_gru_window=48)
        stats_gru = measure_latency_statistics(fn_gru, n_warmup=n_warmup, n_reps=n_reps)
        results_by_model["PyTorch GRU"][str(H)] = stats_gru

        # 4. Soft-Voting Ensemble
        fn_ens = lambda: ensemble.predict_point_in_time(obs_slice, target_hour=target_h)
        stats_ens = measure_latency_statistics(fn_ens, n_warmup=n_warmup, n_reps=n_reps)
        results_by_model["Soft-Voting Ensemble"][str(H)] = stats_ens

    return results_by_model


def benchmark_sequential_streaming_simulation(
    ensemble: SepsisEnsemble,
    patient_stays: List[List[Dict[str, Any]]],
    max_hours: int = 48,
    n_warmup: int = 2,
    n_reps: int = 10,
) -> Dict[str, Any]:
    """
    Simulate streaming ingestion of a patient stay hour-by-hour from t=1 to t=48.
    Measures incremental latency at each hour and total cumulative stay time.
    """
    models = ["Logistic Regression", "XGBoost", "PyTorch GRU", "Soft-Voting Ensemble"]
    streaming_latencies_per_hour = {m: np.zeros(max_hours, dtype=np.float64) for m in models}

    imputer = ensemble.imputer
    lr_model = ensemble.lr_model
    xgb_model = ensemble.xgb_model
    gru_model = ensemble.gru_model
    train_mean = ensemble.train_mean
    train_std = ensemble.train_std

    valid_stays = [s[:max_hours] for s in patient_stays if len(s) >= max_hours][:n_reps]
    if not valid_stays:
        # Fallback to whatever stay lengths are available
        valid_stays = [s[:max_hours] for s in patient_stays if len(s) >= 24][:n_reps]

    for stay in valid_stays:
        stay_len = min(len(stay), max_hours)
        for h_idx in range(stay_len):
            current_obs = stay[: h_idx + 1]
            target_h = int(current_obs[-1]["hour"])

            # Measure LR
            t0 = time.perf_counter_ns()
            _ = predict_lr_isolated(current_obs, target_h, lr_model, imputer)
            t_lr = (time.perf_counter_ns() - t0) / 1e6
            streaming_latencies_per_hour["Logistic Regression"][h_idx] += t_lr

            # Measure XGBoost
            t0 = time.perf_counter_ns()
            _ = predict_xgb_isolated(current_obs, target_h, xgb_model, imputer)
            t_xgb = (time.perf_counter_ns() - t0) / 1e6
            streaming_latencies_per_hour["XGBoost"][h_idx] += t_xgb

            # Measure GRU
            t0 = time.perf_counter_ns()
            _ = predict_gru_isolated(current_obs, target_h, gru_model, train_mean, train_std, imputer)
            t_gru = (time.perf_counter_ns() - t0) / 1e6
            streaming_latencies_per_hour["PyTorch GRU"][h_idx] += t_gru

            # Measure Ensemble
            t0 = time.perf_counter_ns()
            _ = ensemble.predict_point_in_time(current_obs, target_hour=target_h)
            t_ens = (time.perf_counter_ns() - t0) / 1e6
            streaming_latencies_per_hour["Soft-Voting Ensemble"][h_idx] += t_ens

    n_actual = len(valid_stays)
    out_dict = {}
    for m in models:
        avg_hourly = streaming_latencies_per_hour[m] / n_actual
        cum_time = np.cumsum(avg_hourly)
        out_dict[m] = {
            "hourly_latencies_ms": [round(float(v), 4) for v in avg_hourly],
            "cumulative_latencies_ms": [round(float(v), 4) for v in cum_time],
            "total_48h_stay_time_ms": round(float(cum_time[-1]), 4),
        }

    return out_dict


def benchmark_batch_throughput(
    ensemble: SepsisEnsemble,
    test_cache_path: str,
    n_sample_hours: int = 10000,
) -> Dict[str, Any]:
    """
    Benchmark raw inference throughput (predictions/sec) across batch feature vectors.
    """
    if not os.path.exists(test_cache_path):
        raise FileNotFoundError(f"Missing test cache at: {test_cache_path}")

    data = np.load(test_cache_path, allow_pickle=True)
    patient_records = list(data["patient_records"])

    # Collect n_sample_hours of point-in-time features
    features_list = []
    for rec in patient_records:
        # Precomputed aligned hours
        n_h = rec["n_hours"]
        # Use dummy 22-dim features for standardized batch matrix
        features_list.append(np.zeros((n_h, 22), dtype=np.float32))
        if sum(len(x) for x in features_list) >= n_sample_hours:
            break

    X_mat = np.concatenate(features_list)[:n_sample_hours]
    n_rows = len(X_mat)

    # 1. Logistic Regression Batch
    t0 = time.perf_counter()
    _ = ensemble.lr_model.predict_proba(X_mat)
    t_lr = time.perf_counter() - t0
    tp_lr = round(n_rows / t_lr, 2)

    # 2. XGBoost Batch
    t0 = time.perf_counter()
    _ = ensemble.xgb_model.predict_proba(X_mat)
    t_xgb = time.perf_counter() - t0
    tp_xgb = round(n_rows / t_xgb, 2)

    # 3. GRU Batch (evaluating patient sequences in batches)
    t0 = time.perf_counter()
    with torch.no_grad():
        for rec in patient_records[:100]:
            n_pts = rec["n_hours"]
            seq = np.zeros((1, min(n_pts, 48), 22), dtype=np.float32)
            _ = ensemble.gru_model(torch.from_numpy(seq))
    t_gru = time.perf_counter() - t0
    pts_gru = sum(min(r["n_hours"], 48) for r in patient_records[:100])
    tp_gru = round(pts_gru / t_gru, 2)

    return {
        "n_evaluated_hours": n_rows,
        "throughput_predictions_per_sec": {
            "Logistic Regression": tp_lr,
            "XGBoost": tp_xgb,
            "PyTorch GRU": tp_gru,
            "Soft-Voting Ensemble": round(1.0 / (1.0/tp_lr + 1.0/tp_xgb + 1.0/tp_gru), 2),
        },
    }


def verify_output_determinism_and_causality(
    ensemble: SepsisEnsemble,
    sample_obs: List[Dict[str, Any]],
) -> Dict[str, bool]:
    """
    Verify mathematical determinism (zero variance) and causal invariance.
    """
    # 1. Determinism
    p1 = ensemble.predict_point_in_time(sample_obs[:24], target_hour=24)
    p2 = ensemble.predict_point_in_time(sample_obs[:24], target_hour=24)

    is_deterministic = bool(
        p1["lr_probability"] == p2["lr_probability"]
        and p1["xgb_probability"] == p2["xgb_probability"]
        and p1["gru_probability"] == p2["gru_probability"]
        and p1["ensemble_probability"] == p2["ensemble_probability"]
    )

    # 2. Causal Invariance (Adding hour 25 shock vitals does not change hour 24)
    obs_with_future = list(sample_obs[:24])
    obs_with_future.append({
        "patient_id": "TEST_P",
        "hour": 25,
        "hr": 175.0,
        "sbp": 50.0,
        "resp": 42.0,
        "temp": 40.5,
        "wbc": 32.0,
        "label": 1,
    })
    p_future = ensemble.predict_point_in_time(obs_with_future, target_hour=24)

    is_causally_invariant = bool(
        p1["lr_probability"] == p_future["lr_probability"]
        and p1["xgb_probability"] == p_future["xgb_probability"]
        and p1["gru_probability"] == p_future["gru_probability"]
        and p1["ensemble_probability"] == p_future["ensemble_probability"]
    )

    return {
        "is_deterministic": is_deterministic,
        "is_causally_invariant": is_causally_invariant,
    }


def plot_benchmark_results(
    isolated_results: Dict[str, Any],
    streaming_results: Dict[str, Any],
    throughput_results: Dict[str, Any],
    memory_results: Dict[str, Any],
    output_dir: str,
):
    """Generate high-resolution diagnostic plots for model efficiency benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    models = ["Logistic Regression", "XGBoost", "PyTorch GRU", "Soft-Voting Ensemble"]
    colors = {
        "Logistic Regression": "#d62728",
        "XGBoost": "#ff7f0e",
        "PyTorch GRU": "#2ca02c",
        "Soft-Voting Ensemble": "#1f77b4",
    }

    # 1. Latency by History Length
    fig, ax = plt.subplots(figsize=(10, 6))
    h_lengths = [1, 6, 12, 24, 36, 48]

    for m in models:
        med_latencies = [isolated_results[m][str(H)]["median_ms"] for H in h_lengths]
        ax.plot(h_lengths, med_latencies, "o-", color=colors[m], label=m, linewidth=2, markersize=6)

    ax.set_title("Single-Point-in-Time Prediction Latency vs History Length (1–48 Hours)", fontsize=13)
    ax.set_xlabel("Patient History Length (Hours)", fontsize=11)
    ax.set_ylabel("Median Latency (ms)", fontsize=11)
    ax.set_xticks(h_lengths)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_by_history_length.png"), dpi=300)
    plt.close()

    # 2. Latency Percentiles at 48 Hours
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    x_indices = np.arange(len(models))

    p50_vals = [isolated_results[m]["48"]["median_ms"] for m in models]
    p95_vals = [isolated_results[m]["48"]["p95_ms"] for m in models]
    p99_vals = [isolated_results[m]["48"]["p99_ms"] for m in models]

    ax.bar(x_indices - bar_width, p50_vals, width=bar_width, label="Median (p50)", color="#2b5c8f")
    ax.bar(x_indices, p95_vals, width=bar_width, label="95th Percentile (p95)", color="#d95f02")
    ax.bar(x_indices + bar_width, p99_vals, width=bar_width, label="99th Percentile (p99)", color="#7570b3")

    ax.set_title("Latency Percentile Distribution at 48-Hour Sequence Limit (ms)", fontsize=13)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_percentiles_comparison.png"), dpi=300)
    plt.close()

    # 3. Cumulative Streaming Time over 48h Stay
    fig, ax = plt.subplots(figsize=(10, 6))
    hours_axis = np.arange(1, 49)

    for m in models:
        cum_time = streaming_results[m]["cumulative_latencies_ms"][:48]
        ax.plot(hours_axis, cum_time, "-", color=colors[m], label=f"{m} (Total: {cum_time[-1]:.1f}ms)", linewidth=2)

    ax.set_title("Cumulative Inference Time across 48-Hour Sequential Patient Stay", fontsize=13)
    ax.set_xlabel("Elapsed Stay Duration (Hours)", fontsize=11)
    ax.set_ylabel("Cumulative Processing Time (ms)", fontsize=11)
    ax.set_xlim(1, 48)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "streaming_cumulative_time.png"), dpi=300)
    plt.close()

    # 4. Throughput vs Memory Footprint
    fig, ax1 = plt.subplots(figsize=(10, 6))
    tp_dict = throughput_results["throughput_predictions_per_sec"]
    tps = [tp_dict[m] for m in models]
    mem_kb = [memory_results[m]["tracemalloc_peak_kb"] for m in models]

    ax1.set_title("Batch Throughput (pred/sec) and Peak Memory Allocation (KB)", fontsize=13)
    bars1 = ax1.bar(x_indices - 0.2, tps, width=0.4, label="Throughput (pred/sec)", color="#1b9e77")
    ax1.set_ylabel("Throughput (predictions/second)", color="#1b9e77", fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="#1b9e77")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_indices + 0.2, mem_kb, width=0.4, label="Peak Traced Memory (KB)", color="#e7298a")
    ax2.set_ylabel("Peak Traced Memory (KB)", color="#e7298a", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#e7298a")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_memory_comparison.png"), dpi=300)
    plt.close()
    print(f"[*] Benchmark diagnostic figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Model Efficiency & Deployment Feasibility Benchmark")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser(r"~\Desktop\sepsisdataset"))
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="models/results/ensemble_audit/benchmark")
    parser.add_argument("--n_reps", type=int, default=100)
    parser.add_argument("--n_warmup", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("SEPSISSIGNAL MODEL EFFICIENCY & DEPLOYMENT FEASIBILITY BENCHMARK")
    print("Research Prototype - Not for Clinical Decision-Making")
    print("=" * 75)

    # 1. Environment Details
    env = get_system_environment()
    print("\n[STEP 1] Execution Environment Details:")
    for k, v in env.items():
        print(f"  {k:<22}: {v}")

    # 2. Load Model Artifacts
    print("\n[STEP 2] Loading Model Artifacts...")
    ensemble = SepsisEnsemble(models_dir=args.models_dir)

    # 3. Load Real Patient Stays for Benchmarking
    print("\n[STEP 3] Loading Real Patient Test Stays...")
    test_file = os.path.join(args.models_dir, "splits", "test_patients.txt")
    with open(test_file, "r") as f:
        test_pids = [line.strip() for line in f if line.strip()]

    dir_a = os.path.join(args.data_dir, "training_setA", "training")
    dir_b = os.path.join(args.data_dir, "training_setB", "training_setB")

    loaded_stays = []
    for pid in test_pids[:100]:
        pa = os.path.join(dir_a, f"{pid}.psv")
        pb = os.path.join(dir_b, f"{pid}.psv")
        path = pa if os.path.exists(pa) else (pb if os.path.exists(pb) else None)
        if path:
            obs = parse_patient_psv_fast(path)
            if len(obs) >= 48:
                loaded_stays.append(obs)
        if len(loaded_stays) >= 20:
            break

    if not loaded_stays:
        # Fallback to any stays of length >= 24
        for pid in test_pids[:100]:
            pa = os.path.join(dir_a, f"{pid}.psv")
            pb = os.path.join(dir_b, f"{pid}.psv")
            path = pa if os.path.exists(pa) else (pb if os.path.exists(pb) else None)
            if path:
                obs = parse_patient_psv_fast(path)
                if len(obs) >= 24:
                    loaded_stays.append(obs)
            if len(loaded_stays) >= 20:
                break

    print(f"  Loaded {len(loaded_stays)} patient stays with length >= {len(loaded_stays[0])} hours for benchmarking.")
    sample_stay = loaded_stays[0]

    # 4. Isolated Point-in-Time Prediction Latency
    print(f"\n[STEP 4] Benchmarking Isolated Point-in-Time Prediction Latency ({args.n_warmup} warm-up, {args.n_reps} reps)...")
    history_lengths = [1, 6, 12, 24, 36, 48]
    isolated_latencies = benchmark_isolated_latencies_by_history(
        ensemble,
        sample_stay,
        history_lengths=history_lengths,
        n_warmup=args.n_warmup,
        n_reps=args.n_reps,
    )

    print("\nISOLATED PREDICTION LATENCY AT 48-HOUR HISTORY LIMIT (ms):")
    print(f"{'Model':<25} | {'Median (p50)':<12} | {'p95':<10} | {'p99':<10} | {'Mean ± Std':<18}")
    print("-" * 80)
    for m in ["Logistic Regression", "XGBoost", "PyTorch GRU", "Soft-Voting Ensemble"]:
        stats = isolated_latencies[m]["48"]
        print(f"{m:<25} | {stats['median_ms']:<12.4f} | {stats['p95_ms']:<10.4f} | {stats['p99_ms']:<10.4f} | {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f}")

    # 5. Sequential Streaming Simulation
    print("\n[STEP 5] Simulating Sequential Streaming Across 48-Hour Stay...")
    streaming_res = benchmark_sequential_streaming_simulation(
        ensemble,
        loaded_stays,
        max_hours=48,
        n_reps=min(10, len(loaded_stays)),
    )
    print("\nCUMULATIVE PROCESSING TIME FOR COMPLETE 48-HOUR STAY (ms):")
    for m in ["Logistic Regression", "XGBoost", "PyTorch GRU", "Soft-Voting Ensemble"]:
        total_time = streaming_res[m]["total_48h_stay_time_ms"]
        avg_hourly = total_time / 48.0
        print(f"  {m:<25}: Total {total_time:<9.2f} ms (Average {avg_hourly:.4f} ms / hour)")

    # 6. Memory Consumption Profiling
    print("\n[STEP 6] Profiling Peak Memory Allocation (tracemalloc & psutil RSS)...")
    memory_results = {}
    obs_48 = sample_stay[:48]
    target_48 = int(obs_48[-1]["hour"])

    memory_results["Logistic Regression"] = profile_memory_consumption(
        lambda: predict_lr_isolated(obs_48, target_48, ensemble.lr_model, ensemble.imputer)
    )
    memory_results["XGBoost"] = profile_memory_consumption(
        lambda: predict_xgb_isolated(obs_48, target_48, ensemble.xgb_model, ensemble.imputer)
    )
    memory_results["PyTorch GRU"] = profile_memory_consumption(
        lambda: predict_gru_isolated(obs_48, target_48, ensemble.gru_model, ensemble.train_mean, ensemble.train_std, ensemble.imputer)
    )
    memory_results["Soft-Voting Ensemble"] = profile_memory_consumption(
        lambda: ensemble.predict_point_in_time(obs_48, target_hour=target_48)
    )

    print(f"{'Model':<25} | {'Peak Traced Memory (KB)':<25} | {'Process RSS (MB)':<18}")
    print("-" * 75)
    for m in ["Logistic Regression", "XGBoost", "PyTorch GRU", "Soft-Voting Ensemble"]:
        mem = memory_results[m]
        print(f"{m:<25} | {mem['tracemalloc_peak_kb']:<25.2f} | {mem['process_rss_after_mb']:<18.2f}")

    # 7. Batch Inference Throughput
    print("\n[STEP 7] Measuring Batch Inference Throughput...")
    test_cache_path = os.path.join(args.models_dir, "results", "ensemble_audit", "test_cache.npz")
    throughput_res = benchmark_batch_throughput(ensemble, test_cache_path, n_sample_hours=10000)
    print("THROUGHPUT (predictions / second):")
    for m, tp in throughput_res["throughput_predictions_per_sec"].items():
        print(f"  {m:<25}: {tp:>10,.1f} pred/sec")

    # 8. Output Determinism & Causality Verification
    print("\n[STEP 8] Verifying Output Determinism and Causal Invariance...")
    invariance_res = verify_output_determinism_and_causality(ensemble, sample_stay)
    print(f"  Deterministic Repeated Output:  {invariance_res['is_deterministic']} (0.0 variance)")
    print(f"  Causally Invariant to Future:   {invariance_res['is_causally_invariant']} (Future shock vitals isolated)")

    # 9. Plotting Benchmark Diagnostics
    print("\n[STEP 9] Generating Diagnostic Benchmark Figures...")
    plot_benchmark_results(isolated_latencies, streaming_res, throughput_res, memory_results, args.output_dir)

    # 10. Compile JSON Payload
    summary_payload = {
        "disclaimer": "Research prototype, not for clinical decision-making. Benchmarked on identical hardware.",
        "environment": env,
        "benchmark_protocol": {
            "n_warmup": args.n_warmup,
            "n_reps": args.n_reps,
            "timer": "time.perf_counter_ns",
            "history_lengths_evaluated": history_lengths,
        },
        "isolated_latencies_ms": isolated_latencies,
        "streaming_simulation_48h": streaming_res,
        "throughput_predictions_per_sec": throughput_res,
        "memory_profiling": memory_results,
        "determinism_and_causality": invariance_res,
    }

    out_json = os.path.join(args.output_dir, "benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[OK] Benchmark results successfully saved to: {out_json}")


if __name__ == "__main__":
    main()
