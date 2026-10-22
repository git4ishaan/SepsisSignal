"""
Comprehensive Evaluation of Soft-Voting Ensemble vs Baselines for SepsisSignal

Research Prototype - Not for clinical decision-making.

Phases Covered:
- Phase 3: Probability Calibration Evaluation & Fitting on VALIDATION Split Only
- Phase 4: Validation Alert Threshold Selection (Rule: Targeted Recall >= 0.70 while maximizing F1)
- Phase 5: Fair, Identical-Cohort Held-Out TEST Evaluation (LR, XGBoost, GRU, Uncalibrated Ensemble, Calibrated Ensemble)
- Patient-level Bootstrap Confidence Intervals (95% CI)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_lstm import parse_patient_psv_fast
from src.ensemble import SepsisEnsemble


def load_split_patient_ids(models_dir: str) -> Tuple[List[str], List[str]]:
    """Load validation and test split IDs from models/splits/."""
    val_file = os.path.join(models_dir, "splits", "val_patients.txt")
    test_file = os.path.join(models_dir, "splits", "test_patients.txt")

    if not os.path.exists(val_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing split files in {os.path.join(models_dir, 'splits')}")

    with open(val_file, "r", encoding="utf-8") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    with open(test_file, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    return val_ids, test_ids


def find_patient_files(data_dir: str, pids: List[str]) -> Dict[str, str]:
    """Find file paths for given patient IDs in PhysioNet directory structure."""
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


def compute_expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across n_bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_conf = np.mean(probs[mask])
            bin_acc = np.mean(targets[mask])
            ece += (bin_count / n) * abs(bin_conf - bin_acc)

    return float(ece)


def evaluate_cohort(
    ensemble: SepsisEnsemble,
    patient_files: Dict[str, str],
    pids: List[str],
    cohort_name: str = "cohort",
) -> Dict[str, Any]:
    """Run batch causal inference across all patient stays in a cohort."""
    t0 = time.time()
    print(f"[*] Starting inference for {cohort_name} ({len(pids)} patients)...")

    patient_records = []
    all_hours = []
    all_targets = []
    all_p_lr = []
    all_p_xgb = []
    all_p_gru = []
    all_p_ens = []

    for idx, pid in enumerate(pids):
        if (idx + 1) % 1500 == 0 or (idx + 1) == len(pids):
            elapsed = time.time() - t0
            print(f"    Processed {idx + 1}/{len(pids)} patients ({elapsed:.1f}s)...")

        file_path = patient_files.get(pid)
        if not file_path:
            continue

        raw_obs = parse_patient_psv_fast(file_path)
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

        all_hours.extend(res["hours"])
        all_targets.append(res["labels"])
        all_p_lr.append(res["p_lr"])
        all_p_xgb.append(res["p_xgb"])
        all_p_gru.append(res["p_gru"])
        all_p_ens.append(res["p_ensemble"])

    y_true = np.concatenate(all_targets)
    p_lr = np.concatenate(all_p_lr)
    p_xgb = np.concatenate(all_p_xgb)
    p_gru = np.concatenate(all_p_gru)
    p_ens = np.concatenate(all_p_ens)

    print(f"[*] {cohort_name} completed in {time.time() - t0:.1f}s. Total patient-hours: {len(y_true)}")

    return {
        "pids": pids,
        "patient_records": patient_records,
        "y_true": y_true,
        "p_lr": p_lr,
        "p_xgb": p_xgb,
        "p_gru": p_gru,
        "p_ensemble": p_ens,
    }


def find_optimal_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    target_recall: float = 0.70,
) -> Tuple[float, Dict[str, float]]:
    """Select a validation threshold by the recall constraint, without a fixed grid."""
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    if thresholds.size == 0:
        return 1.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # The final precision/recall pair has no corresponding threshold. For each
    # actual threshold, scores are classified positive when probability >= it.
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    eligible = np.flatnonzero(recall >= target_recall)
    candidates = eligible if eligible.size else np.arange(thresholds.size)
    best_idx = int(candidates[np.argmax(f1[candidates])])
    threshold = round(float(thresholds[best_idx]), 4)

    # Recompute at the rounded threshold so the saved stats match the artifact.
    preds = (probs >= threshold).astype(int)
    prec, rec, score, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    return threshold, {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(score),
    }


def compute_metrics_for_predictions(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all standard evaluation metrics for a set of predictions and threshold."""
    preds = (probs >= threshold).astype(int)
    auprc = float(average_precision_score(targets, probs)) if targets.sum() > 0 else 0.0
    auroc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    alert_rate = float(preds.mean() * 100.0)
    brier = float(brier_score_loss(targets, probs))
    ece = compute_expected_calibration_error(probs, targets, n_bins=10)

    return {
        "threshold": round(threshold, 4),
        "auprc": round(auprc, 4),
        "auroc": round(auroc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "specificity": round(spec, 4),
        "alert_rate_per_100h": round(alert_rate, 2),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_patient_bootstrap_ci(
    patient_records: List[Dict[str, Any]],
    n_bootstraps: int = 250,
    seed: int = 42,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute 95% patient-cluster bootstrap confidence intervals for AUPRC and AUROC."""
    rng = np.random.RandomState(seed)
    n_patients = len(patient_records)

    models_to_track = ["p_lr", "p_xgb", "p_gru", "p_ens"]
    bootstrap_auprcs = {m: [] for m in models_to_track}
    bootstrap_aurocs = {m: [] for m in models_to_track}

    for b in range(n_bootstraps):
        sampled_indices = rng.choice(n_patients, size=n_patients, replace=True)
        sampled_targets = []
        sampled_probs = {m: [] for m in models_to_track}

        for idx in sampled_indices:
            rec = patient_records[idx]
            sampled_targets.append(rec["labels"])
            for m in models_to_track:
                sampled_probs[m].append(rec[m])

        y_b = np.concatenate(sampled_targets)
        if y_b.sum() == 0 or len(np.unique(y_b)) < 2:
            continue

        for m in models_to_track:
            p_b = np.concatenate(sampled_probs[m])
            bootstrap_auprcs[m].append(average_precision_score(y_b, p_b))
            bootstrap_aurocs[m].append(roc_auc_score(y_b, p_b))

    ci_results = {}
    for m in models_to_track:
        auprc_arr = np.array(bootstrap_auprcs[m])
        auroc_arr = np.array(bootstrap_aurocs[m])
        ci_results[m] = {
            "auprc_95_ci": (round(float(np.percentile(auprc_arr, 2.5)), 4), round(float(np.percentile(auprc_arr, 97.5)), 4)),
            "auroc_95_ci": (round(float(np.percentile(auroc_arr, 2.5)), 4), round(float(np.percentile(auroc_arr, 97.5)), 4)),
        }

    return ci_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Soft-Voting Ensemble vs Candidate Models")
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser(r"~\Desktop\sepsisdataset"))
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--n_bootstraps", type=int, default=250, help="Number of patient-level bootstrap resamples")
    args = parser.parse_args()

    print("=" * 70)
    print("SEPSISSIGNAL SOFT-VOTING ENSEMBLE EVALUATION EXPERIMENT")
    print("Research Prototype - Not for Clinical Decision-Making")
    print("=" * 70)

    val_pids, test_pids = load_split_patient_ids(args.models_dir)
    print(f"[*] Loaded splits: {len(val_pids)} validation patients, {len(test_pids)} test patients.")

    all_pids = val_pids + test_pids
    patient_files = find_patient_files(args.data_dir, all_pids)
    print(f"[*] Located {len(patient_files)} total PSV files for validation & test.")

    ensemble = SepsisEnsemble(models_dir=args.models_dir)

    # -------------------------------------------------------------
    # PHASE 3 & 4: VALIDATION EVALUATION & CALIBRATION & THRESHOLD
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PHASE 3 & 4: VALIDATION SET EVALUATION, CALIBRATION & THRESHOLD SELECTION")
    print("-" * 70)

    val_data = evaluate_cohort(ensemble, patient_files, val_pids, cohort_name="VALIDATION")
    y_val = val_data["y_true"]
    p_val_lr = val_data["p_lr"]
    p_val_xgb = val_data["p_xgb"]
    p_val_gru = val_data["p_gru"]
    p_val_ens = val_data["p_ensemble"]

    val_pos = int(y_val.sum())
    val_prev = (val_pos / len(y_val)) * 100.0
    print(f"Validation cohort: {len(val_pids)} patients, {len(y_val)} patient-hours, {val_pos} positive hours ({val_prev:.2f}% prevalence)")

    # Baseline thresholds from existing artifacts
    lr_thresh = 0.4200
    xgb_thresh = 0.3900
    gru_thresh = 0.4200

    # Calibration Evaluation (Phase 3)
    # Fit Platt scaling strictly on validation set
    print("\n[*] Evaluating calibration on VALIDATION set...")
    eps = 1e-6
    p_val_ens_clipped = np.clip(p_val_ens, eps, 1.0 - eps)
    val_logits = np.log(p_val_ens_clipped / (1.0 - p_val_ens_clipped)).reshape(-1, 1)

    platt_calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    platt_calibrator.fit(val_logits, y_val)
    p_val_ens_cal = platt_calibrator.predict_proba(val_logits)[:, 1]

    # Save calibrator artifact for reproduction (separate, non-mutating)
    calibrator_path = os.path.join(args.models_dir, "ensemble_calibrator.joblib")
    joblib.dump(platt_calibrator, calibrator_path)
    print(f"[OK] Saved Platt calibrator fitted STRICTLY on validation data to: {calibrator_path}")

    # Threshold selection on Validation Set (Phase 4)
    print("\n[*] Selecting ensemble threshold on VALIDATION set (Rule: Recall >= 0.70 while maximizing F1)...")
    ens_thresh_uncal, uncal_val_stats = find_optimal_threshold(y_val, p_val_ens, target_recall=0.70)
    ens_thresh_cal, cal_val_stats = find_optimal_threshold(y_val, p_val_ens_cal, target_recall=0.70)

    print(f"    Selected Uncalibrated Ensemble Threshold: {ens_thresh_uncal} (Val F1: {uncal_val_stats.get('f1', 0):.4f}, Recall: {uncal_val_stats.get('recall', 0):.4f})")
    print(f"    Selected Calibrated Ensemble Threshold:   {ens_thresh_cal} (Val F1: {cal_val_stats.get('f1', 0):.4f}, Recall: {cal_val_stats.get('recall', 0):.4f})")

    # Save ensemble threshold configuration
    thresh_config_path = os.path.join(args.models_dir, "ensemble_threshold.json")
    with open(thresh_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "uncalibrated_threshold": ens_thresh_uncal,
            "calibrated_threshold": ens_thresh_cal,
            "chosen_on": "validation",
            "selection_rule": "Targeted high recall (>=0.70) while maximizing F1 on validation set",
            "validation_stats_uncalibrated": uncal_val_stats,
            "validation_stats_calibrated": cal_val_stats,
        }, f, indent=2)
    print(f"[OK] Saved ensemble threshold configuration to: {thresh_config_path}")

    # Validation metrics table
    val_m_lr = compute_metrics_for_predictions(y_val, p_val_lr, lr_thresh)
    val_m_xgb = compute_metrics_for_predictions(y_val, p_val_xgb, xgb_thresh)
    val_m_gru = compute_metrics_for_predictions(y_val, p_val_gru, gru_thresh)
    val_m_ens_uncal = compute_metrics_for_predictions(y_val, p_val_ens, ens_thresh_uncal)
    val_m_ens_cal = compute_metrics_for_predictions(y_val, p_val_ens_cal, ens_thresh_cal)

    print("\nVALIDATION SET METRICS COMPARISON:")
    print(f"{'Model':<28} | {'AUPRC':<7} | {'AUROC':<7} | {'Prec':<7} | {'Rec':<7} | {'Spec':<7} | {'Thresh':<7} | {'Alert/100h':<10} | {'Brier':<7} | {'ECE':<7}")
    print("-" * 115)
    print(f"{'Logistic Regression':<28} | {val_m_lr['auprc']:<7.4f} | {val_m_lr['auroc']:<7.4f} | {val_m_lr['precision']:<7.4f} | {val_m_lr['recall']:<7.4f} | {val_m_lr['specificity']:<7.4f} | {val_m_lr['threshold']:<7.4f} | {val_m_lr['alert_rate_per_100h']:<10.2f} | {val_m_lr['brier_score']:<7.4f} | {val_m_lr['ece']:<7.4f}")
    print(f"{'XGBoost':<28} | {val_m_xgb['auprc']:<7.4f} | {val_m_xgb['auroc']:<7.4f} | {val_m_xgb['precision']:<7.4f} | {val_m_xgb['recall']:<7.4f} | {val_m_xgb['specificity']:<7.4f} | {val_m_xgb['threshold']:<7.4f} | {val_m_xgb['alert_rate_per_100h']:<10.2f} | {val_m_xgb['brier_score']:<7.4f} | {val_m_xgb['ece']:<7.4f}")
    print(f"{'PyTorch GRU':<28} | {val_m_gru['auprc']:<7.4f} | {val_m_gru['auroc']:<7.4f} | {val_m_gru['precision']:<7.4f} | {val_m_gru['recall']:<7.4f} | {val_m_gru['specificity']:<7.4f} | {val_m_gru['threshold']:<7.4f} | {val_m_gru['alert_rate_per_100h']:<10.2f} | {val_m_gru['brier_score']:<7.4f} | {val_m_gru['ece']:<7.4f}")
    print(f"{'Ensemble (Uncalibrated)':<28} | {val_m_ens_uncal['auprc']:<7.4f} | {val_m_ens_uncal['auroc']:<7.4f} | {val_m_ens_uncal['precision']:<7.4f} | {val_m_ens_uncal['recall']:<7.4f} | {val_m_ens_uncal['specificity']:<7.4f} | {val_m_ens_uncal['threshold']:<7.4f} | {val_m_ens_uncal['alert_rate_per_100h']:<10.2f} | {val_m_ens_uncal['brier_score']:<7.4f} | {val_m_ens_uncal['ece']:<7.4f}")
    print(f"{'Ensemble (Platt Calibrated)':<28} | {val_m_ens_cal['auprc']:<7.4f} | {val_m_ens_cal['auroc']:<7.4f} | {val_m_ens_cal['precision']:<7.4f} | {val_m_ens_cal['recall']:<7.4f} | {val_m_ens_cal['specificity']:<7.4f} | {val_m_ens_cal['threshold']:<7.4f} | {val_m_ens_cal['alert_rate_per_100h']:<10.2f} | {val_m_ens_cal['brier_score']:<7.4f} | {val_m_ens_cal['ece']:<7.4f}")

    # -------------------------------------------------------------
    # PHASE 5: FAIR HELD-OUT TEST EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: FAIR HELD-OUT TEST SET EVALUATION")
    print("=" * 70)

    test_data = evaluate_cohort(ensemble, patient_files, test_pids, cohort_name="HELD-OUT TEST")
    y_test = test_data["y_true"]
    p_test_lr = test_data["p_lr"]
    p_test_xgb = test_data["p_xgb"]
    p_test_gru = test_data["p_gru"]
    p_test_ens = test_data["p_ensemble"]

    test_pos = int(y_test.sum())
    test_prev = (test_pos / len(y_test)) * 100.0
    print(f"Test cohort: {len(test_pids)} patients, {len(y_test)} patient-hours, {test_pos} positive hours ({test_prev:.2f}% prevalence)")

    # Apply frozen calibrator to test probabilities
    p_test_ens_clipped = np.clip(p_test_ens, eps, 1.0 - eps)
    test_logits = np.log(p_test_ens_clipped / (1.0 - p_test_ens_clipped)).reshape(-1, 1)
    p_test_ens_cal = platt_calibrator.predict_proba(test_logits)[:, 1]

    # Compute test metrics
    test_m_lr = compute_metrics_for_predictions(y_test, p_test_lr, lr_thresh)
    test_m_xgb = compute_metrics_for_predictions(y_test, p_test_xgb, xgb_thresh)
    test_m_gru = compute_metrics_for_predictions(y_test, p_test_gru, gru_thresh)
    test_m_ens_uncal = compute_metrics_for_predictions(y_test, p_test_ens, ens_thresh_uncal)
    test_m_ens_cal = compute_metrics_for_predictions(y_test, p_test_ens_cal, ens_thresh_cal)

    # Compute patient-level bootstrap confidence intervals
    print("\n[*] Computing 95% patient-level bootstrap confidence intervals...")
    ci_results = compute_patient_bootstrap_ci(test_data["patient_records"], n_bootstraps=args.n_bootstraps)

    print("\nHELD-OUT TEST SET EVALUATION METRICS:")
    print(f"{'Model':<28} | {'AUPRC':<7} | {'AUROC':<7} | {'Prec':<7} | {'Rec':<7} | {'Spec':<7} | {'Thresh':<7} | {'Alert/100h':<10} | {'Brier':<7} | {'ECE':<7}")
    print("-" * 115)
    print(f"{'Logistic Regression':<28} | {test_m_lr['auprc']:<7.4f} | {test_m_lr['auroc']:<7.4f} | {test_m_lr['precision']:<7.4f} | {test_m_lr['recall']:<7.4f} | {test_m_lr['specificity']:<7.4f} | {test_m_lr['threshold']:<7.4f} | {test_m_lr['alert_rate_per_100h']:<10.2f} | {test_m_lr['brier_score']:<7.4f} | {test_m_lr['ece']:<7.4f}")
    print(f"{'XGBoost':<28} | {test_m_xgb['auprc']:<7.4f} | {test_m_xgb['auroc']:<7.4f} | {test_m_xgb['precision']:<7.4f} | {test_m_xgb['recall']:<7.4f} | {test_m_xgb['specificity']:<7.4f} | {test_m_xgb['threshold']:<7.4f} | {test_m_xgb['alert_rate_per_100h']:<10.2f} | {test_m_xgb['brier_score']:<7.4f} | {test_m_xgb['ece']:<7.4f}")
    print(f"{'PyTorch GRU':<28} | {test_m_gru['auprc']:<7.4f} | {test_m_gru['auroc']:<7.4f} | {test_m_gru['precision']:<7.4f} | {test_m_gru['recall']:<7.4f} | {test_m_gru['specificity']:<7.4f} | {test_m_gru['threshold']:<7.4f} | {test_m_gru['alert_rate_per_100h']:<10.2f} | {test_m_gru['brier_score']:<7.4f} | {test_m_gru['ece']:<7.4f}")
    print(f"{'Ensemble (Uncalibrated)':<28} | {test_m_ens_uncal['auprc']:<7.4f} | {test_m_ens_uncal['auroc']:<7.4f} | {test_m_ens_uncal['precision']:<7.4f} | {test_m_ens_uncal['recall']:<7.4f} | {test_m_ens_uncal['specificity']:<7.4f} | {test_m_ens_uncal['threshold']:<7.4f} | {test_m_ens_uncal['alert_rate_per_100h']:<10.2f} | {test_m_ens_uncal['brier_score']:<7.4f} | {test_m_ens_uncal['ece']:<7.4f}")
    print(f"{'Ensemble (Platt Calibrated)':<28} | {test_m_ens_cal['auprc']:<7.4f} | {test_m_ens_cal['auroc']:<7.4f} | {test_m_ens_cal['precision']:<7.4f} | {test_m_ens_cal['recall']:<7.4f} | {test_m_ens_cal['specificity']:<7.4f} | {test_m_ens_cal['threshold']:<7.4f} | {test_m_ens_cal['alert_rate_per_100h']:<10.2f} | {test_m_ens_cal['brier_score']:<7.4f} | {test_m_ens_cal['ece']:<7.4f}")

    print("\nPATIENT-LEVEL 95% BOOTSTRAP CONFIDENCE INTERVALS (HELD-OUT TEST):")
    for m_key, name in [("p_lr", "Logistic Regression"), ("p_xgb", "XGBoost"), ("p_gru", "PyTorch GRU"), ("p_ens", "Soft-Voting Ensemble")]:
        ci = ci_results[m_key]
        print(f"  {name:<25}: AUPRC 95% CI {ci['auprc_95_ci']}, AUROC 95% CI {ci['auroc_95_ci']}")

    # Save full results JSON
    results_dir = os.path.join(args.models_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "ensemble_evaluation.json")

    results_payload = {
        "disclaimer": "Research prototype, not for clinical decision-making. Evaluated on PhysioNet 2019 test split.",
        "cohort_info": {
            "validation": {
                "patients": len(val_pids),
                "patient_hours": len(y_val),
                "positive_hours": val_pos,
                "prevalence_pct": round(val_prev, 4),
            },
            "test": {
                "patients": len(test_pids),
                "patient_hours": len(y_test),
                "positive_hours": test_pos,
                "prevalence_pct": round(test_prev, 4),
            },
        },
        "threshold_selection": {
            "selection_rule": "Targeted high recall (>=0.70) while maximizing F1 on validation set",
            "uncalibrated_threshold": ens_thresh_uncal,
            "calibrated_threshold": ens_thresh_cal,
        },
        "validation_metrics": {
            "logistic_regression": val_m_lr,
            "xgboost": val_m_xgb,
            "pytorch_gru": val_m_gru,
            "ensemble_uncalibrated": val_m_ens_uncal,
            "ensemble_calibrated": val_m_ens_cal,
        },
        "test_metrics": {
            "logistic_regression": test_m_lr,
            "xgboost": test_m_xgb,
            "pytorch_gru": test_m_gru,
            "ensemble_uncalibrated": test_m_ens_uncal,
            "ensemble_calibrated": test_m_ens_cal,
        },
        "test_bootstrap_confidence_intervals": ci_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[OK] Evaluation metrics successfully saved to {out_file}")


if __name__ == "__main__":
    main()
