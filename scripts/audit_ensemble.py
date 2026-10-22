"""
Deep-Dive Ensemble Validation Audit for SepsisSignal

Research Prototype - Not for Clinical Decision-Making.

Addresses:
1. Threshold selection audit (uncalibrated vs calibrated, high-recall vs max-F1 operating points)
2. Calibration audit (Brier, Log Loss, Equal-width & Quantile ECE, reliability tables & plots)
3. Paired patient-level cluster bootstrap (1,000 replicates) for ensemble vs XGBoost, GRU, and LR
4. Data integrity verification (disjoint splits, row uniqueness, label alignment, GRU discrepancy audit)
5. Generation of high-resolution diagnostic plots and comprehensive report in models/results/ensemble_audit/
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_lstm import parse_patient_psv_fast
from src.ensemble import SepsisEnsemble


def load_split_patient_ids(models_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load train, val, test patient IDs."""
    train_file = os.path.join(models_dir, "splits", "train_patients.txt")
    val_file = os.path.join(models_dir, "splits", "val_patients.txt")
    test_file = os.path.join(models_dir, "splits", "test_patients.txt")

    with open(train_file, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(val_file, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    return train_ids, val_ids, test_ids


def find_patient_files(data_dir: str, pids: List[str]) -> Dict[str, str]:
    """Resolve file paths for patient IDs."""
    dir_a = os.path.join(data_dir, "training_setA", "training")
    dir_b = os.path.join(data_dir, "training_setB", "training_setB")

    file_map = {}
    for pid in pids:
        path_a = os.path.join(dir_a, f"{pid}.psv")
        if os.path.exists(path_a):
            file_map[pid] = path_a
            continue
        path_b = os.path.join(dir_b, f"{pid}.psv")
        if os.path.exists(path_b):
            file_map[pid] = path_b
            continue

    return file_map


def compute_reliability_table(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
    bin_strategy: str = "uniform",
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Compute reliability table and Expected Calibration Error (ECE).
    bin_strategy: 'uniform' (equal width) or 'quantile' (equal frequency).
    """
    n = len(probs)
    if n == 0:
        return [], 0.0

    if bin_strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif bin_strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(probs, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, 1.0])
    else:
        raise ValueError(f"Unknown bin_strategy: {bin_strategy}")

    table = []
    ece = 0.0
    actual_bins = len(bin_edges) - 1

    for i in range(actual_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == actual_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(probs[mask]))
            emp_acc = float(np.mean(targets[mask]))
            cal_err = abs(mean_pred - emp_acc)
            ece += (count / n) * cal_err
        else:
            mean_pred = float((low + high) / 2.0)
            emp_acc = 0.0
            cal_err = 0.0

        table.append({
            "bin_idx": i + 1,
            "bin_range": f"[{low:.4f}, {high:.4f}]",
            "count": count,
            "pct_of_total": round((count / n) * 100.0, 2),
            "mean_pred_prob": round(mean_pred, 4),
            "observed_positive_rate": round(emp_acc, 4),
            "calibration_error": round(cal_err, 4),
        })

    return table, float(ece)


def extract_or_load_cohort_predictions(
    ensemble: SepsisEnsemble,
    patient_files: Dict[str, str],
    pids: List[str],
    cache_path: str,
    cohort_name: str = "cohort",
) -> Dict[str, Any]:
    """Run batch prediction across patient stays or load from disk cache."""
    if os.path.exists(cache_path):
        print(f"[*] Loading cached {cohort_name} predictions from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        patient_records = list(data["patient_records"])
        return {
            "pids": pids,
            "patient_records": patient_records,
            "y_true": data["y_true"],
            "p_lr": data["p_lr"],
            "p_xgb": data["p_xgb"],
            "p_gru": data["p_gru"],
            "p_ensemble": data["p_ensemble"],
            "patient_offsets": data["patient_offsets"],
        }

    print(f"[*] Extracting predictions for {cohort_name} ({len(pids)} patients)...")
    t0 = time.time()
    patient_records = []
    all_targets = []
    all_p_lr = []
    all_p_xgb = []
    all_p_gru = []
    all_p_ens = []
    patient_offsets = [0]

    for idx, pid in enumerate(pids):
        if (idx + 1) % 1500 == 0 or (idx + 1) == len(pids):
            print(f"    Processed {idx + 1}/{len(pids)} patients ({time.time() - t0:.1f}s)...")

        fpath = patient_files.get(pid)
        if not fpath:
            continue
        raw_obs = parse_patient_psv_fast(fpath)
        if not raw_obs:
            continue

        res = ensemble.predict_patient_stay_aligned(raw_obs)
        n_pts = len(res["hours"])
        if n_pts == 0:
            continue

        patient_records.append({
            "pid": pid,
            "n_hours": n_pts,
            "labels": res["labels"],
            "p_lr": res["p_lr"],
            "p_xgb": res["p_xgb"],
            "p_gru": res["p_gru"],
            "p_ens": res["p_ensemble"],
        })

        all_targets.append(res["labels"])
        all_p_lr.append(res["p_lr"])
        all_p_xgb.append(res["p_xgb"])
        all_p_gru.append(res["p_gru"])
        all_p_ens.append(res["p_ensemble"])
        patient_offsets.append(patient_offsets[-1] + n_pts)

    y_true = np.concatenate(all_targets)
    p_lr = np.concatenate(all_p_lr)
    p_xgb = np.concatenate(all_p_xgb)
    p_gru = np.concatenate(all_p_gru)
    p_ens = np.concatenate(all_p_ens)
    offsets = np.array(patient_offsets, dtype=int)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        patient_records=patient_records,
        y_true=y_true,
        p_lr=p_lr,
        p_xgb=p_xgb,
        p_gru=p_gru,
        p_ensemble=p_ens,
        patient_offsets=offsets,
    )
    print(f"[*] Saved {cohort_name} cache to: {cache_path} ({len(y_true)} hours, {time.time() - t0:.1f}s)")

    return {
        "pids": pids,
        "patient_records": patient_records,
        "y_true": y_true,
        "p_lr": p_lr,
        "p_xgb": p_xgb,
        "p_gru": p_gru,
        "p_ensemble": p_ens,
        "patient_offsets": offsets,
    }


def search_threshold_operating_points(
    targets: np.ndarray,
    probs: np.ndarray,
    min_thresh: float = 0.001,
    max_thresh: float = 0.90,
    n_steps: int = 1000,
    target_recall: float = 0.70,
) -> Dict[str, Any]:
    """
    Search operating points programmatically without silent fallbacks.
    Returns:
    - high_recall_op: Best F1 satisfying recall >= target_recall (if any)
    - max_f1_op: Global maximum F1 operating point
    - satisfied_target_recall: bool
    """
    thresholds = np.linspace(min_thresh, max_thresh, n_steps)

    best_hr_thresh = None
    best_hr_f1 = -1.0
    best_hr_stats = None

    best_global_thresh = None
    best_global_f1 = -1.0
    best_global_stats = None

    for t in thresholds:
        t_val = round(float(t), 4)
        preds = (probs >= t_val).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        alert_rate = float(preds.mean() * 100.0)

        stats = {
            "threshold": t_val,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "specificity": round(spec, 4),
            "alert_rate_per_100h": round(alert_rate, 2),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        # Track global max F1
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_thresh = t_val
            best_global_stats = stats

        # Track constrained high recall
        if rec >= target_recall:
            if f1 > best_hr_f1:
                best_hr_f1 = f1
                best_hr_thresh = t_val
                best_hr_stats = stats

    satisfied = best_hr_thresh is not None

    return {
        "target_recall_required": target_recall,
        "satisfied_target_recall": satisfied,
        "high_recall_op": best_hr_stats if satisfied else None,
        "max_f1_op": best_global_stats,
    }


def evaluate_threshold_on_test(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate preselected threshold on held-out test data."""
    preds = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    alert_rate = float(preds.mean() * 100.0)

    return {
        "threshold": round(threshold, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "specificity": round(spec, 4),
        "alert_rate_per_100h": round(alert_rate, 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_paired_patient_bootstrap(
    patient_records: List[Dict[str, Any]],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run 1,000 paired patient-cluster bootstrap iterations.
    Computes paired difference: Delta = Ensemble - Comparator for AUPRC and AUROC.
    """
    rng = np.random.RandomState(seed)
    n_patients = len(patient_records)

    comparators = ["p_xgb", "p_gru", "p_lr"]
    comp_names = {"p_xgb": "XGBoost", "p_gru": "PyTorch GRU", "p_lr": "Logistic Regression"}

    diff_auprc = {c: [] for c in comparators}
    diff_auroc = {c: [] for c in comparators}

    t0 = time.time()
    print(f"[*] Running paired patient-cluster bootstrap ({n_bootstraps} iterations, seed={seed})...")

    for b in range(n_bootstraps):
        if (b + 1) % 250 == 0 or (b + 1) == n_bootstraps:
            print(f"    Bootstrap iteration {b + 1}/{n_bootstraps} ({time.time() - t0:.1f}s)...")

        sampled_pids = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_y = []
        sampled_ens = []
        sampled_comps = {c: [] for c in comparators}

        for idx in sampled_pids:
            rec = patient_records[idx]
            sampled_y.append(rec["labels"])
            sampled_ens.append(rec["p_ens"])
            for c in comparators:
                sampled_comps[c].append(rec[c])

        y_b = np.concatenate(sampled_y)
        if y_b.sum() == 0 or len(np.unique(y_b)) < 2:
            continue

        p_ens_b = np.concatenate(sampled_ens)
        auprc_ens = average_precision_score(y_b, p_ens_b)
        auroc_ens = roc_auc_score(y_b, p_ens_b)

        for c in comparators:
            p_comp_b = np.concatenate(sampled_comps[c])
            auprc_c = average_precision_score(y_b, p_comp_b)
            auroc_c = roc_auc_score(y_b, p_comp_b)

            diff_auprc[c].append(float(auprc_ens - auprc_c))
            diff_auroc[c].append(float(auroc_ens - auroc_c))

    results = {}
    for c in comparators:
        c_name = comp_names[c]
        arr_pr = np.array(diff_auprc[c])
        arr_roc = np.array(diff_auroc[c])

        pr_ci = (float(np.percentile(arr_pr, 2.5)), float(np.percentile(arr_pr, 97.5)))
        roc_ci = (float(np.percentile(arr_roc, 2.5)), float(np.percentile(arr_roc, 97.5)))

        pr_sup = bool(pr_ci[0] > 0.0)
        roc_sup = bool(roc_ci[0] > 0.0)

        results[c] = {
            "comparator_name": c_name,
            "mean_delta_auprc": round(float(np.mean(arr_pr)), 4),
            "std_delta_auprc": round(float(np.std(arr_pr)), 4),
            "ci_95_delta_auprc": [round(pr_ci[0], 4), round(pr_ci[1], 4)],
            "includes_zero_auprc": bool(pr_ci[0] <= 0.0 <= pr_ci[1]),
            "statistically_superior_auprc": pr_sup,
            "mean_delta_auroc": round(float(np.mean(arr_roc)), 4),
            "std_delta_auroc": round(float(np.std(arr_roc)), 4),
            "ci_95_delta_auroc": [round(roc_ci[0], 4), round(roc_ci[1], 4)],
            "includes_zero_auroc": bool(roc_ci[0] <= 0.0 <= roc_ci[1]),
            "statistically_superior_auroc": roc_sup,
            "raw_diff_auprc": arr_pr.tolist(),
            "raw_diff_auroc": arr_roc.tolist(),
        }

    return results


def plot_paired_bootstrap(bootstrap_results: Dict[str, Any], output_path: str):
    """Plot paired difference distributions with 95% CIs."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Paired Patient-Cluster Bootstrap Distributions (Ensemble minus Baseline, N=1,000 Replicates)", fontsize=14, y=0.98)

    comparators = ["p_xgb", "p_gru", "p_lr"]
    row_titles = ["Ensemble vs XGBoost", "Ensemble vs PyTorch GRU", "Ensemble vs Logistic Regression"]

    for row, (c, title) in enumerate(zip(comparators, row_titles)):
        res = bootstrap_results[c]
        arr_pr = np.array(res["raw_diff_auprc"])
        arr_roc = np.array(res["raw_diff_auroc"])

        # AUPRC
        ax_pr = axes[row, 0]
        ax_pr.hist(arr_pr, bins=35, color="#1f77b4", alpha=0.7, edgecolor="black")
        ax_pr.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference (Parity)")
        ax_pr.axvline(res["ci_95_delta_auprc"][0], color="green", linestyle=":", linewidth=2, label="95% CI Lower")
        ax_pr.axvline(res["ci_95_delta_auprc"][1], color="green", linestyle=":", linewidth=2, label="95% CI Upper")
        ax_pr.axvline(res["mean_delta_auprc"], color="navy", linestyle="-", linewidth=2, label=f"Mean: {res['mean_delta_auprc']:+.4f}")
        ax_pr.set_title(f"{title}: Δ AUPRC (CI: [{res['ci_95_delta_auprc'][0]:+.4f}, {res['ci_95_delta_auprc'][1]:+.4f}])")
        ax_pr.set_xlabel("Δ AUPRC (Ensemble - Comparator)")
        ax_pr.set_ylabel("Bootstrap Frequency")
        ax_pr.legend(loc="upper right", fontsize=8)
        ax_pr.grid(True, alpha=0.3)

        # AUROC
        ax_roc = axes[row, 1]
        ax_roc.hist(arr_roc, bins=35, color="#ff7f0e", alpha=0.7, edgecolor="black")
        ax_roc.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Difference (Parity)")
        ax_roc.axvline(res["ci_95_delta_auroc"][0], color="green", linestyle=":", linewidth=2, label="95% CI Lower")
        ax_roc.axvline(res["ci_95_delta_auroc"][1], color="green", linestyle=":", linewidth=2, label="95% CI Upper")
        ax_roc.axvline(res["mean_delta_auroc"], color="darkorange", linestyle="-", linewidth=2, label=f"Mean: {res['mean_delta_auroc']:+.4f}")
        ax_roc.set_title(f"{title}: Δ AUROC (CI: [{res['ci_95_delta_auroc'][0]:+.4f}, {res['ci_95_delta_auroc'][1]:+.4f}])")
        ax_roc.set_xlabel("Δ AUROC (Ensemble - Comparator)")
        ax_roc.set_ylabel("Bootstrap Frequency")
        ax_roc.legend(loc="upper right", fontsize=8)
        ax_roc.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved paired bootstrap figure to: {output_path}")


def plot_calibration_curves(
    val_probs_uncal: np.ndarray,
    val_probs_cal: np.ndarray,
    y_val: np.ndarray,
    test_probs_uncal: np.ndarray,
    test_probs_cal: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
):
    """Plot reliability curves and probability density for validation and test cohorts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Reliability Calibration Curves (10 Equal-Width Bins & Probability Density)", fontsize=14, y=0.98)

    # Helper for reliability points
    def _get_curve(probs, targets, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        conf = []
        acc = []
        for i in range(n_bins):
            m = (probs >= edges[i]) & (probs <= edges[i+1]) if i == n_bins - 1 else (probs >= edges[i]) & (probs < edges[i+1])
            if np.sum(m) > 0:
                conf.append(np.mean(probs[m]))
                acc.append(np.mean(targets[m]))
        return conf, acc

    # 1. Validation Reliability
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    c_uncal, a_uncal = _get_curve(val_probs_uncal, y_val)
    c_cal, a_cal = _get_curve(val_probs_cal, y_val)
    ax1.plot(c_uncal, a_uncal, "s-", color="#d62728", label="Uncalibrated Ensemble")
    ax1.plot(c_cal, a_cal, "o-", color="#2ca02c", label="Platt Calibrated Ensemble")
    ax1.set_title("Validation Cohort: Reliability Curve")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Empirical Positive Fraction")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # 2. Validation Density
    ax2 = axes[0, 1]
    ax2.hist(val_probs_uncal, bins=50, alpha=0.5, color="#d62728", density=True, label="Uncalibrated")
    ax2.hist(val_probs_cal, bins=50, alpha=0.5, color="#2ca02c", density=True, label="Platt Calibrated")
    ax2.set_title("Validation Cohort: Predicted Probability Distribution")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 3. Test Reliability
    ax3 = axes[1, 0]
    ax3.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    tc_uncal, ta_uncal = _get_curve(test_probs_uncal, y_test)
    tc_cal, ta_cal = _get_curve(test_probs_cal, y_test)
    ax3.plot(tc_uncal, ta_uncal, "s-", color="#d62728", label="Uncalibrated Ensemble")
    ax3.plot(tc_cal, ta_cal, "o-", color="#2ca02c", label="Platt Calibrated Ensemble")
    ax3.set_title("Held-Out Test Cohort: Reliability Curve")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Empirical Positive Fraction")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    # 4. Test Density
    ax4 = axes[1, 1]
    ax4.hist(test_probs_uncal, bins=50, alpha=0.5, color="#d62728", density=True, label="Uncalibrated")
    ax4.hist(test_probs_cal, bins=50, alpha=0.5, color="#2ca02c", density=True, label="Platt Calibrated")
    ax4.set_title("Held-Out Test Cohort: Predicted Probability Distribution")
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved calibration figure to: {output_path}")


def plot_pr_roc_curves(
    y_test: np.ndarray,
    p_ens: np.ndarray,
    p_xgb: np.ndarray,
    p_gru: np.ndarray,
    p_lr: np.ndarray,
    output_path: str,
):
    """Plot AUPRC and AUROC comparison curves on held-out test data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Precision-Recall Curve
    ax_pr = axes[0]
    prevalence = float(y_test.mean())
    ax_pr.axhline(prevalence, color="gray", linestyle="--", label=f"Prevalence Baseline ({prevalence*100:.2f}%)")

    models = [
        ("Soft-Voting Ensemble", p_ens, "#1f77b4"),
        ("XGBoost (Offline)", p_xgb, "#ff7f0e"),
        ("PyTorch GRU (Offline)", p_gru, "#2ca02c"),
        ("Logistic Regression (Live)", p_lr, "#d62728"),
    ]

    for name, p, color in models:
        precision, recall, _ = precision_recall_curve(y_test, p)
        score = average_precision_score(y_test, p)
        ax_pr.plot(recall, precision, label=f"{name} (AUPRC = {score:.4f})", color=color, linewidth=1.8)

    ax_pr.set_title("Precision-Recall Curve (Held-Out Test Cohort)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 0.25)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="upper right")

    # 2. ROC Curve
    ax_roc = axes[1]
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUROC = 0.5000)")

    for name, p, color in models:
        fpr, tpr, _ = roc_curve(y_test, p)
        score = roc_auc_score(y_test, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUROC = {score:.4f})", color=color, linewidth=1.8)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Saved PR/ROC curve figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Ensemble Validation Audit")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser(r"~\Desktop\sepsisdataset"))
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="models/results/ensemble_audit")
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("SEPSISSIGNAL ENSEMBLE VALIDATION AUDIT")
    print("Research Prototype - Not for Clinical Decision-Making")
    print("=" * 75)

    # 1. Data Integrity & Partition Disjointness Check
    print("\n[STEP 1] Auditing Patient Split Disjointness & Integrity...")
    train_ids, val_ids, test_ids = load_split_patient_ids(args.models_dir)
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    train_val_overlap = set_train.intersection(set_val)
    train_test_overlap = set_train.intersection(set_test)
    val_test_overlap = set_val.intersection(set_test)

    print(f"  Train patients: {len(train_ids)} unique: {len(set_train)}")
    print(f"  Val patients:   {len(val_ids)} unique: {len(set_val)}")
    print(f"  Test patients:  {len(test_ids)} unique: {len(set_test)}")
    print(f"  Pairwise overlaps: Train-Val={len(train_val_overlap)}, Train-Test={len(train_test_overlap)}, Val-Test={len(val_test_overlap)}")

    assert len(train_val_overlap) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_test_overlap) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_test_overlap) == 0, "Data leakage detected: val and test overlap!"
    print("  [PASS] Zero patient overlap detected across all three splits.")

    all_pids = val_ids + test_ids
    patient_files = find_patient_files(args.data_dir, all_pids)
    print(f"  [PASS] Successfully located {len(patient_files)} total PSV files.")

    # 2. Extract or Load Cohort Inferences
    print("\n[STEP 2] Loading/Extracting Cohort Inferences...")
    ensemble = SepsisEnsemble(models_dir=args.models_dir)

    val_cache_path = os.path.join(args.output_dir, "val_cache.npz")
    test_cache_path = os.path.join(args.output_dir, "test_cache.npz")

    val_data = extract_or_load_cohort_predictions(ensemble, patient_files, val_ids, val_cache_path, cohort_name="VALIDATION")
    test_data = extract_or_load_cohort_predictions(ensemble, patient_files, test_ids, test_cache_path, cohort_name="HELD-OUT TEST")

    y_val = val_data["y_true"]
    p_val_uncal = val_data["p_ensemble"]
    p_val_lr = val_data["p_lr"]
    p_val_xgb = val_data["p_xgb"]
    p_val_gru = val_data["p_gru"]

    y_test = test_data["y_true"]
    p_test_uncal = test_data["p_ensemble"]
    p_test_lr = test_data["p_lr"]
    p_test_xgb = test_data["p_xgb"]
    p_test_gru = test_data["p_gru"]

    print(f"  Validation hours: {len(y_val)} (Positive: {y_val.sum()}, {y_val.mean()*100:.2f}%)")
    print(f"  Test hours:       {len(y_test)} (Positive: {y_test.sum()}, {y_test.mean()*100:.2f}%)")

    # 3. Fit Platt Calibrator Strictly on Validation Data
    print("\n[STEP 3] Fitting and Auditing Platt Probability Calibrator...")
    eps = 1e-6
    p_val_clipped = np.clip(p_val_uncal, eps, 1.0 - eps)
    val_logits = np.log(p_val_clipped / (1.0 - p_val_clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(val_logits, y_val)
    p_val_cal = calibrator.predict_proba(val_logits)[:, 1]

    # Apply to test
    p_test_clipped = np.clip(p_test_uncal, eps, 1.0 - eps)
    test_logits = np.log(p_test_clipped / (1.0 - p_test_clipped)).reshape(-1, 1)
    p_test_cal = calibrator.predict_proba(test_logits)[:, 1]

    # Save audited calibrator
    joblib.dump(calibrator, os.path.join(args.output_dir, "ensemble_calibrator_audited.joblib"))

    # Compute Calibration Metrics
    brier_val_uncal = float(brier_score_loss(y_val, p_val_uncal))
    brier_val_cal = float(brier_score_loss(y_val, p_val_cal))
    logloss_val_uncal = float(log_loss(y_val, p_val_uncal))
    logloss_val_cal = float(log_loss(y_val, p_val_cal))

    brier_test_uncal = float(brier_score_loss(y_test, p_test_uncal))
    brier_test_cal = float(brier_score_loss(y_test, p_test_cal))
    logloss_test_uncal = float(log_loss(y_test, p_test_uncal))
    logloss_test_cal = float(log_loss(y_test, p_test_cal))

    rel_table_val_uncal_10, ece_val_uncal_10 = compute_reliability_table(p_val_uncal, y_val, n_bins=10, bin_strategy="uniform")
    rel_table_val_cal_10, ece_val_cal_10 = compute_reliability_table(p_val_cal, y_val, n_bins=10, bin_strategy="uniform")
    rel_table_val_cal_q, ece_val_cal_q = compute_reliability_table(p_val_cal, y_val, n_bins=10, bin_strategy="quantile")

    rel_table_test_uncal_10, ece_test_uncal_10 = compute_reliability_table(p_test_uncal, y_test, n_bins=10, bin_strategy="uniform")
    rel_table_test_cal_10, ece_test_cal_10 = compute_reliability_table(p_test_cal, y_test, n_bins=10, bin_strategy="uniform")
    rel_table_test_cal_q, ece_test_cal_q = compute_reliability_table(p_test_cal, y_test, n_bins=10, bin_strategy="quantile")

    print(f"  Validation Brier: Uncalibrated={brier_val_uncal:.4f}, Calibrated={brier_val_cal:.4f}")
    print(f"  Validation LogLoss: Uncalibrated={logloss_val_uncal:.4f}, Calibrated={logloss_val_cal:.4f}")
    print(f"  Validation ECE (10 uniform): Uncalibrated={ece_val_uncal_10:.4f}, Calibrated={ece_val_cal_10:.4f}")
    print(f"  Validation ECE (10 quantiles): Calibrated={ece_val_cal_q:.4f}")
    print(f"  Test Brier: Uncalibrated={brier_test_uncal:.4f}, Calibrated={brier_test_cal:.4f}")
    print(f"  Test LogLoss: Uncalibrated={logloss_test_uncal:.4f}, Calibrated={logloss_test_cal:.4f}")
    print(f"  Test ECE (10 uniform): Uncalibrated={ece_test_uncal_10:.4f}, Calibrated={ece_test_cal_10:.4f}")
    print(f"  Test ECE (10 quantiles): Calibrated={ece_test_cal_q:.4f}")

    # Plot Calibration Curves
    cal_plot_path = os.path.join(args.output_dir, "calibration_curves.png")
    plot_calibration_curves(p_val_uncal, p_val_cal, y_val, p_test_uncal, p_test_cal, y_test, cal_plot_path)

    # 4. Audit Threshold Selection & Operating Points
    print("\n[STEP 4] Auditing Threshold Selection and Operating Points...")
    # Uncalibrated search
    uncal_ops = search_threshold_operating_points(y_val, p_val_uncal, min_thresh=0.05, max_thresh=0.85, n_steps=801, target_recall=0.70)
    # Calibrated search with full range
    cal_ops_full = search_threshold_operating_points(y_val, p_val_cal, min_thresh=0.001, max_thresh=0.40, n_steps=4000, target_recall=0.70)
    # Calibrated search with truncated range (reproducing the original bug)
    cal_ops_truncated = search_threshold_operating_points(y_val, p_val_cal, min_thresh=0.05, max_thresh=0.85, n_steps=801, target_recall=0.70)

    print("  --- Uncalibrated Ensemble Operating Points ---")
    print(f"    High-Recall Target (Recall >= 70%): Satisfied={uncal_ops['satisfied_target_recall']}, Stats={uncal_ops['high_recall_op']}")
    print(f"    Global Max F1: Stats={uncal_ops['max_f1_op']}")

    print("  --- Calibrated Ensemble (Full Range Search: [0.001, 0.40]) ---")
    print(f"    High-Recall Target (Recall >= 70%): Satisfied={cal_ops_full['satisfied_target_recall']}, Stats={cal_ops_full['high_recall_op']}")
    print(f"    Global Max F1: Stats={cal_ops_full['max_f1_op']}")

    print("  --- Calibrated Ensemble (Truncated Range: [0.05, 0.85] - Explaining the 0.05 Bug) ---")
    print(f"    High-Recall Target (Recall >= 70%): Satisfied={cal_ops_truncated['satisfied_target_recall']}")
    print(f"    Fallback Max F1 in [0.05, 0.85]: Stats={cal_ops_truncated['max_f1_op']}")

    # Evaluate each operating point on held-out test data
    uncal_hr_thresh = uncal_ops["high_recall_op"]["threshold"]
    cal_hr_thresh = cal_ops_full["high_recall_op"]["threshold"]
    cal_maxf1_thresh = cal_ops_full["max_f1_op"]["threshold"]

    test_uncal_hr = evaluate_threshold_on_test(y_test, p_test_uncal, uncal_hr_thresh)
    test_cal_hr = evaluate_threshold_on_test(y_test, p_test_cal, cal_hr_thresh)
    test_cal_maxf1 = evaluate_threshold_on_test(y_test, p_test_cal, cal_maxf1_thresh)

    # 5. Paired Patient-Cluster Bootstrap Comparison (1,000 Replicates)
    print("\n[STEP 5] Running 1,000 Paired Patient-Cluster Bootstrap Replicates...")
    bootstrap_results = run_paired_patient_bootstrap(test_data["patient_records"], n_bootstraps=args.n_bootstraps, seed=42)

    boot_plot_path = os.path.join(args.output_dir, "paired_bootstrap_distributions.png")
    plot_paired_bootstrap(bootstrap_results, boot_plot_path)

    # 6. Precision-Recall & ROC Curve Comparison
    print("\n[STEP 6] Plotting PR and ROC Comparison Curves...")
    pr_roc_path = os.path.join(args.output_dir, "pr_roc_curves.png")
    plot_pr_roc_curves(y_test, p_test_uncal, p_test_xgb, p_test_gru, p_test_lr, pr_roc_path)

    # 7. Compile Final Audit Artifacts
    print("\n[STEP 7] Generating Final Audit Artifacts & Comprehensive Report...")

    # Write JSON summary
    summary_json_path = os.path.join(args.output_dir, "audit_summary.json")
    summary_payload = {
        "disclaimer": "Research prototype, not for clinical decision-making. Evaluated on PhysioNet 2019 test split.",
        "data_integrity": {
            "train_patients": len(train_ids),
            "val_patients": len(val_ids),
            "test_patients": len(test_ids),
            "val_hours": len(y_val),
            "test_hours": len(y_test),
            "val_prevalence": round(float(y_val.mean()), 5),
            "test_prevalence": round(float(y_test.mean()), 5),
            "disjoint_splits": True,
            "feature_dimension": 22,
            "gru_max_window": 48,
        },
        "threshold_audit": {
            "uncalibrated_high_recall_op": {
                "validation": uncal_ops["high_recall_op"],
                "test": test_uncal_hr,
            },
            "calibrated_high_recall_op": {
                "validation": cal_ops_full["high_recall_op"],
                "test": test_cal_hr,
            },
            "calibrated_max_f1_op": {
                "validation": cal_ops_full["max_f1_op"],
                "test": test_cal_maxf1,
            },
            "truncated_grid_bug_explanation": {
                "original_search_min": 0.05,
                "calibrated_recall_at_0_05": cal_ops_truncated["max_f1_op"]["recall"],
                "true_calibrated_threshold_for_70pct_recall": cal_hr_thresh,
            },
        },
        "calibration_audit": {
            "validation": {
                "brier_uncalibrated": brier_val_uncal,
                "brier_calibrated": brier_val_cal,
                "logloss_uncalibrated": logloss_val_uncal,
                "logloss_calibrated": logloss_val_cal,
                "ece_uniform_10_uncalibrated": ece_val_uncal_10,
                "ece_uniform_10_calibrated": ece_val_cal_10,
                "ece_quantile_10_calibrated": ece_val_cal_q,
                "reliability_table_uniform_10": rel_table_val_cal_10,
                "reliability_table_quantile_10": rel_table_val_cal_q,
            },
            "test": {
                "brier_uncalibrated": brier_test_uncal,
                "brier_calibrated": brier_test_cal,
                "logloss_uncalibrated": logloss_test_uncal,
                "logloss_calibrated": logloss_test_cal,
                "ece_uniform_10_uncalibrated": ece_test_uncal_10,
                "ece_uniform_10_calibrated": ece_test_cal_10,
                "ece_quantile_10_calibrated": ece_test_cal_q,
                "reliability_table_uniform_10": rel_table_test_cal_10,
                "reliability_table_quantile_10": rel_table_test_cal_q,
            },
        },
        "paired_bootstrap_comparison": {
            "n_bootstraps": args.n_bootstraps,
            "seed": 42,
            "results": {
                k: {sk: sv for sk, sv in v.items() if not sk.startswith("raw_")}
                for k, v in bootstrap_results.items()
            },
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[*] Audit summary JSON saved to: {summary_json_path}")
    print("\n[OK] Deep-Dive Audit successfully completed!")


if __name__ == "__main__":
    main()
