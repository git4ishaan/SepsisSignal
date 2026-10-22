# SepsisSignal: Ensemble Model Inference & ML Context Guide

> **Mandatory Disclaimer:** *Research prototype, not for clinical decision-making. Evaluated on the PhysioNet/CinC Challenge 2019 ICU cohort.*

This guide provides team members ([Soham Paranjape](file:///c:/Users/Mitansh/Desktop/SepsisSignal/README.md), [Ishaan Patil](file:///c:/Users/Mitansh/Desktop/SepsisSignal/README.md), [Mitansh Bheda](file:///c:/Users/Mitansh/Desktop/SepsisSignal/README.md)) and collaborators with complete architectural context, programmatic inference instructions, evaluation benchmarks, calibration analysis, and deployment feasibility findings for the **Soft-Voting Ensemble Model** and the candidate ML models in **SepsisSignal**.

---

## Table of Contents
1. [Executive Summary & Key Takeaways](#1-executive-summary--key-takeaways)
2. [Model Architecture & Ensemble Formulation](#2-model-architecture--ensemble-formulation)
3. [Feature Pipeline & Causal Invariants](#3-feature-pipeline--causal-invariants)
4. [Inference API & Code Usage Guide](#4-inference-api--code-usage-guide)
5. [Comparative Evaluation Benchmark (Held-Out Test Set)](#5-comparative-evaluation-benchmark-held-out-test-set)
6. [Threshold Selection & Operating Points Guide](#6-threshold-selection--operating-points-guide)
7. [Probability Calibration & ECE Analysis](#7-probability-calibration--ece-analysis)
8. [Statistical Significance (1,000 Paired Cluster Bootstraps)](#8-statistical-significance-1000-paired-cluster-bootstraps)
9. [Computational Efficiency, Latency & Resource Footprint](#9-computational-efficiency-latency--resource-footprint)
10. [Engineering Recommendations for Production & Future Work](#10-engineering-recommendations-for-production--future-work)
11. [Repository Artifacts & File Map](#11-repository-artifacts--file-map)

---

## 1. Executive Summary & Key Takeaways

| Metric / Attribute | Logistic Regression (Live Baseline) | XGBoost (Offline Tree) | PyTorch GRU (Offline RNN) | Soft-Voting Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| **Model Artifact** | [`models/model.joblib`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/model.joblib) | [`models/xgboost_model.joblib`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/xgboost_model.joblib) | [`models/lstm_model.pt`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/lstm_model.pt) | [`src/ensemble.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/ensemble.py) |
| **Test AUPRC** | 0.0775 | **0.0983** | 0.0823 | 0.0970 |
| **Test AUROC** | 0.7416 | 0.7977 | 0.7956 | **0.8045** |
| **Brier Score (Calibrated)**| 0.1981 (uncal) | 0.1603 (uncal) | 0.1785 (uncal) | **0.0162** (Platt) |
| **Latency at 48h (p50)** | **0.42 ms** | **0.56 ms** | 5.94 ms | 8.27 ms |
| **Batch Throughput** | **5,426,525 pred/s** | **1,687,308 pred/s** | 14,526 pred/s | 14,364 pred/s |
| **Peak Heap RAM** | **26.88 KB** | 255.60 KB | 34.13 KB | 284.10 KB |
| **Complexity Scaling** | **$O(1)$ constant** | **$O(1)$ constant** | $O(W^2)$ unrolled | $O(W^2)$ unrolled |
| **Live Serving Status** | **Active Production** | Offline Candidate | Offline Benchmark | Research Experiment |

### Core Findings Every Teammate Must Know:
1. **XGBoost is the Pareto Winner:** Standalone XGBoost achieves an AUPRC of **0.0983**, statistically indistinguishable from the Ensemble (**0.0970**, 95% bootstrap CI $[-0.0113, +0.0084]$ includes zero), but is **~14.8x faster** (0.56 ms vs 8.27 ms) and has **~117x higher throughput** (1.68M vs 14.3K pred/s) without requiring a PyTorch runtime.
2. **Historical GRU Metric Resolved:** The historical reported GRU AUPRC of 0.2495 was evaluated on artificially padded batch sequences with future tokens in [`scripts/train_lstm.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/scripts/train_lstm.py). In rigorous point-in-time causal evaluation on all 233,127 held-out test hours, the GRU achieves **0.0823**.
3. **Live Architecture Remains Untouched:** Logistic Regression remains the live serving model in [`src/predict.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/predict.py) for the TCP socket and WebSocket endpoints. It executes in **0.42 ms** with **26 KB** RAM.
4. **Calibrated Threshold Discrepancy Resolved:** The calibrated ensemble achieves the required $\ge 70\%$ validation recall at threshold $\tau = 0.0131$ (matching uncalibrated $\tau = 0.4110$). The earlier reported 32.79% recall at $\tau = 0.05$ was due to search grid truncation ($[0.05, 0.85]$), which actually represents an unconstrained max-$F_1$ operating point.

---

## 2. Model Architecture & Ensemble Formulation

The soft-voting ensemble combines three distinct learning paradigms:
1. **Linear Discriminative Baseline:** `StandardScaler` + L2-regularized `LogisticRegression` fit on tabular features.
2. **Non-linear Gradient Boosted Trees:** `XGBoostClassifier` fit on tabular features.
3. **Deep Recurrent Neural Network:** 2-layer `SepsisRNN` (`GRU`, hidden dimension 48, input dimension 22) processing normalized causal temporal sequence windows up to 48 hours.

```
                              Raw ICU Observations (Hour <= T)
                                            │
                                            ▼
                             Point-in-Time Feature Extractor
                                 (22 Causal Float32s)
                                            │
            ┌───────────────────────────────┼───────────────────────────────┐
            │                               │                               │
            ▼                               ▼                               ▼
    Logistic Regression                  XGBoost                     PyTorch GRU
     (StandardScaler)                 (Decision Trees)             (Window <= 48h)
            │                               │                               │
            ▼                               ▼                               ▼
       P(LR) in [0,1]                 P(XGB) in [0,1]                P(GRU) in [0,1]
            │                               │                               │
            └───────────────────────┬───────┴───────────────────────────────┘
                                    │
                                    ▼
                        Equal-Weight Soft Voting
                P_ensemble = (P(LR) + P(XGB) + P(GRU)) / 3.0
                                    │
                                    ▼
                      Platt Scaling / Calibrator
                  P_calibrated = Sigmoid(a * logit + b)
```

### Mathematical Formulation
For any patient at target hour $T$, using only observations $\mathcal{O}_{\le T}$:
$$p_{\text{ensemble}}(T) = \frac{p_{\text{lr}}(\mathbf{x}_T) + p_{\text{xgb}}(\mathbf{x}_T) + p_{\text{gru}}(\mathbf{X}_{T-W:T})}{3.0}$$

Where:
- $\mathbf{x}_T \in \mathbb{R}^{22}$ is the causal feature vector at target hour $T$.
- $\mathbf{X}_{T-W:T} \in \mathbb{R}^{W \times 22}$ is the sequence of normalized causal feature vectors for the preceding $W = \min(T, 48)$ hours.
- For Platt-calibrated output:
  $$p_{\text{calibrated}} = \sigma\left(a \cdot \ln\left(\frac{p_{\text{ensemble}}}{1 - p_{\text{ensemble}}}\right) + b\right)$$
  Fitted strictly on validation log-odds ($a = 0.7788, b = -2.9388$).

---

## 3. Feature Pipeline & Causal Invariants

All models share the canonical feature extraction contract implemented in [`src/features.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/features.py):
```python
extract_features_point_in_time(observations, target_hour, imputer=None)
```

### The 5 Clinical Signals
- `HR`: Heart Rate (bpm)
- `SBP`: Systolic Blood Pressure (mmHg)
- `Resp`: Respiration Rate (breaths/min)
- `Temp`: Core Body Temperature (**Celsius**)
- `WBC`: White Blood Cell Count ($10^3 / \mu\text{L}$)

### The 22-Dimensional Feature Vector (`FEATURE_NAMES`)
For each of the 5 signals, 4 features are extracted:
1. `{sig}_val`: Current measurement, causal forward-fill, or training median if never observed.
2. `{sig}_missing`: Binary indicator (`0.0` if measured at target hour, `1.0` if missing).
3. `{sig}_time_since`: Hours since last valid measurement (`0.0` at observation, `99.0` sentinel if unobserved).
4. `{sig}_delta`: Difference from immediately preceding measurement (`0.0` if $\le 1$ measurement in history).

Plus 2 contextual stay features:
5. `hour`: Integer hour of ICU stay ($1, 2, \dots$).
6. `current_empty_count`: Count of unmeasured clinical signals at hour $T$ ($0$ to $5$).

### Training Median Imputer (`models/imputer.json`)
Computed strictly on the 28,217 training cohort patients:
```json
{
  "HR": 85.0,
  "SBP": 120.0,
  "RESP": 18.0,
  "TEMP": 36.9,
  "WBC": 9.6
}
```

### Non-Negotiable Causal Invariants
1. **Strict Causality:** Rows with $\text{hour} > T$ must never be accessed when computing features or predictions for hour $T$.
2. **Prediction Immutability:** Once a risk prediction for $(\text{patient\_id}, T)$ is stored in SQLite, subsequent arrival of later hours never mutates earlier predictions.
3. **Disjoint Patient Splits:** Training (28,217 patients), Validation (6,046 patients), and Held-Out Test (6,048 patients) cohorts have zero patient overlap.

---

## 4. Inference API & Code Usage Guide

The ensemble engine is packaged in [`src/ensemble.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/ensemble.py) under the `SepsisEnsemble` class.

### 4.1 Initializing the Ensemble
```python
from src.ensemble import SepsisEnsemble

# Automatically loads LR, XGBoost, GRU, imputer, and normalizers from models/
ensemble = SepsisEnsemble()
```

### 4.2 Single Point-in-Time Inference
Used for streaming or evaluating risk at a specific hour $T$:
```python
raw_observations = [
    {"hour": 1, "HR": 88.0, "SBP": 112.0, "Resp": 16.0, "Temp": 37.0, "WBC": None},
    {"hour": 2, "HR": 94.0, "SBP": 108.0, "Resp": 20.0, "Temp": 37.8, "WBC": 11.2},
    {"hour": 3, "HR": 105.0, "SBP": 96.0, "Resp": 24.0, "Temp": 38.5, "WBC": None},
]

# Run causal inference at Hour 3 (only accesses hours <= 3)
result = ensemble.predict_point_in_time(raw_observations, target_hour=3)

print(result)
# Output:
# {
#   "target_hour": 3,
#   "lr_probability": 0.3842,
#   "xgb_probability": 0.4210,
#   "gru_probability": 0.4518,
#   "ensemble_probability": 0.4190,
#   "empty_count": 1
# }
```

### 4.3 Batch Patient Stay Evaluation
Used for offline evaluation, retrospective analysis, or auditing an entire patient stay:
```python
# Evaluates all hours of a patient stay with aligned arrays
stay_results = ensemble.predict_patient_stay_aligned(raw_observations, max_window=48)

print("Hours:       ", stay_results["hours"])         # array([1, 2, 3], dtype=int32)
print("LR Probs:    ", stay_results["p_lr"])          # array([...], dtype=float32)
print("XGB Probs:   ", stay_results["p_xgb"])         # array([...], dtype=float32)
print("GRU Probs:   ", stay_results["p_gru"])         # array([...], dtype=float32)
print("Ensemble:    ", stay_results["p_ensemble"])    # array([...], dtype=float32)
```

### 4.4 Applying Probability Calibration & Operating Thresholds
```python
import joblib
import numpy as np

# Load the audited validation Platt calibrator
calibrator = joblib.load("models/results/ensemble_audit/ensemble_calibrator_audited.joblib")

p_raw = result["ensemble_probability"]

# Convert raw probability to log-odds
p_clipped = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
logit = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

# Calibrated probability
p_calibrated = float(calibrator.predict_proba(logit)[0, 1])

# Check alert against Operating Point A (High Recall Bedside Screening, tau = 0.0131)
alert_screening = int(p_calibrated >= 0.0131)

# Check alert against Operating Point B (Targeted Max-F1, tau = 0.0526)
alert_targeted = int(p_calibrated >= 0.0526)

print(f"Raw: {p_raw:.4f} | Calibrated: {p_calibrated:.4f} | Screening Alert: {alert_screening} | Targeted Alert: {alert_targeted}")
```

---

## 5. Comparative Evaluation Benchmark (Held-Out Test Set)

Evaluated on the frozen, held-out PhysioNet 2019 test cohort:
- **Patients:** 6,048 (zero patient overlap with train or val)
- **Total Hours:** 233,127
- **Sepsis Positive Hours:** 4,006 (**1.72% positive prevalence**)

### Comprehensive Performance Comparison Table

| Model | Operating Threshold ($\tau$) | Optimization Target | Precision | Recall | Specificity | Alert Rate (/100h) | AUPRC | AUROC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Live)** | 0.4200 | Recall $\ge 70\%$ | 3.01% | 73.74% | 58.49% | 42.06 | 0.0775 | 0.7416 | 0.1981 |
| **XGBoost (Offline)** | 0.3900 | Recall $\ge 70\%$ | 3.80% | **76.44%** | 66.13% | 34.60 | **0.0983** | 0.7977 | 0.1603 |
| **PyTorch GRU (Offline)** | 0.4200 | Recall $\ge 70\%$ | 3.88% | 76.66% | 66.77% | 33.98 | 0.0823 | 0.7956 | 0.1785 |
| **Ensemble (Uncalibrated)** | 0.4110 | Recall $\ge 70\%$ | 4.15% | 75.29% | **69.59%** | **31.19** | 0.0970 | **0.8045** | 0.1685 |
| **Ensemble (Platt Calibrated)**| 0.0131 | Recall $\ge 70\%$ | 4.15% | 75.41% | 69.52% | 31.26 | 0.0970 | **0.8045** | **0.0162** |
| **Ensemble (Platt Max-$F_1$)** | 0.0526 | Unconstrained $F_1$ | **11.20%**| 34.32% | **95.24%** | **5.27** | 0.0970 | **0.8045** | **0.0162** |

---

## 6. Threshold Selection & Operating Points Guide

### 6.1 The "0.05 Calibrated Threshold" Audit Finding
In earlier exploratory scripts, a grid search was bounded between $[0.05, 0.85]$. Because Platt calibration adjusts probabilities to the real-world baseline prevalence (~1.7%), calibrated probabilities rarely exceed 0.15. 
- At $\tau = 0.05$, calibrated recall had already dropped to **31.41%**, failing the $\ge 70\%$ recall constraint.
- The script silently fell back to picking the maximum $F_1$ threshold inside $[0.05, 0.85]$ ($\tau \approx 0.05$).

### 6.2 The Two Valid Clinical Operating Points
When the search space is correctly searched across $[0.001, 0.400]$, two distinct operating points emerge:

#### Operating Point A: High-Recall Bedside Screening ($\text{Recall} \ge 70\%$)
- **Purpose:** Early bedside alert system where missing a deteriorating sepsis patient carries severe clinical consequence.
- **Uncalibrated Threshold:** $\tau = 0.4110$
- **Calibrated Threshold:** $\tau = 0.0131$
- **Test Performance:** Recall: **75.41%**, Precision: **4.15%**, Specificity: **69.52%**, Alert Rate: **31.26 alerts / 100 patient-hours**.

#### Operating Point B: High-Precision / Low-Alarm Fatigue ($\text{Max } F_1$)
- **Purpose:** Targeted alert system designed to prevent alarm fatigue in non-ICU step-down units.
- **Calibrated Threshold:** $\tau = 0.0526$
- **Test Performance:** Recall: **34.32%**, Precision: **11.20%**, Specificity: **95.24%**, Alert Rate: **5.27 alerts / 100 patient-hours**.
- **Clinical Trade-off:** Alert rate drops by **~83%** (from 31.3 to 5.3 alerts per 100 hours) and precision triples, but misses ~66% of sepsis hours.

---

## 7. Probability Calibration & ECE Analysis

### 7.1 Calibration Metric Summary (Held-Out Test Set)
- **Uncalibrated Ensemble:** Brier Score: `0.1685`, Log Loss: `0.5313`, Uniform ECE: `0.3572`
- **Platt Calibrated Ensemble:** Brier Score: **`0.0162`**, Log Loss: **`0.0753`**, Uniform ECE: **`0.0013`**, Quantile ECE: **`0.0029`**

### 7.2 Why Uniform ECE is ~0.0013 and Why Quantile Binning is Mandatory
Teammates analyzing calibration in severe class imbalance (~1.7% positive) must be aware of uniform binning distortion:
1. **Uniform Bin Masking:** In 10 uniform bins ($[0.0, 0.1), [0.1, 0.2), \dots$), **97.75% of test data (227,881 of 233,127 hours)** falls into Bin 1 ($[0.0, 0.1]$).
   - In Bin 1, predicted probability is 1.36% and observed rate is 1.45% (gap = 0.09 percentage points).
   - The sample-weighted contribution is $(227,881 / 233,127) \times 0.0009 \approx 0.00088$.
   - Although higher bins have larger gaps (e.g. Bin 4 $[0.3, 0.4]$ has an error of 18.3%), they contain only 61 hours, yielding negligible weight in uniform ECE.
2. **Quantile (Decile) ECE:** In 10 equal-frequency quantile bins (each containing exactly 23,313 hours), the calibrated test ECE is **`0.0029`** (0.29 percentage points).
   - Deciles 1–8 match observed rates within 0.1 percentage point.
   - Decile 10 (mean predicted 7.39%) exhibits an observed sepsis rate of 8.51% (gap = 1.12 percentage points).

---

## 8. Statistical Significance (1,000 Paired Cluster Bootstraps)

To determine whether the Soft-Voting Ensemble is genuinely superior to individual models or if differences are random sampling noise, we ran **1,000 paired cluster bootstrap iterations** over the 6,048 test patients (seed = 42).

```
                      95% Bootstrap Confidence Intervals (Δ = Ensemble - Comparator)
                                       Δ AUPRC
XGBoost:      [======|======]             (-0.0113 to +0.0084, contains 0 -> Parity)
GRU:                      [====|====]     (+0.0020 to +0.0276, excludes 0 -> Superior)
Logistic Reg:                [====|====]  (+0.0079 to +0.0297, excludes 0 -> Superior)
             ──────────────────┼────────────────────────────────
                             Δ = 0
```

### Paired Bootstrap Summary Table

| Comparator Model | Metric | Point Difference ($\Delta$) | Bootstrap Mean $\Delta$ | 95% Bootstrap CI | Contains Zero? | Statistically Superior? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Offline)** | **AUPRC** | **-0.0013** | -0.0009 | **[-0.0113, +0.0084]** | **YES** | **NO (Statistical Parity)** |
| **XGBoost (Offline)** | **AUROC** | **+0.0068** | +0.0068 | **[-0.0018, +0.0146]** | **YES** | **NO (Statistical Parity)** |
| **PyTorch GRU (Offline)** | **AUPRC** | **+0.0147** | +0.0148 | **[+0.0020, +0.0276]** | **NO** | **YES (Statistically Superior)** |
| **PyTorch GRU (Offline)** | **AUROC** | **+0.0089** | +0.0088 | **[+0.0014, +0.0158]** | **NO** | **YES (Statistically Superior)** |
| **Logistic Regression (Live)** | **AUPRC** | **+0.0195** | +0.0188 | **[+0.0079, +0.0297]** | **NO** | **YES (Statistically Superior)** |
| **Logistic Regression (Live)** | **AUROC** | **+0.0629** | +0.0633 | **[+0.0490, +0.0773]** | **NO** | **YES (Statistically Superior)** |

**Key Takeaway:** The Soft-Voting Ensemble is statistically superior to Logistic Regression and PyTorch GRU, but **fails to demonstrate statistical superiority over standalone XGBoost**.

---

## 9. Computational Efficiency, Latency & Resource Footprint

Benchmarked on Intel Core (16 physical / 22 logical cores), 32 GB RAM, Python 3.13.7, PyTorch 2.14.1+cpu. High-resolution monotonic timing (`time.perf_counter_ns`), 20 warm-up runs, 100 repeated trials.

### 9.1 Isolated Point-in-Time Latency by History Length (1 to 48 Hours)

| Model | 1 Hour (p50) | 6 Hours (p50) | 12 Hours (p50) | 24 Hours (p50) | 36 Hours (p50) | 48 Hours (p50) | Scaling Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.32 ms | 0.36 ms | 0.48 ms | 0.46 ms | 0.40 ms | **0.42 ms** | **$O(1)$ constant** |
| **XGBoost** | 0.51 ms | 0.47 ms | 0.55 ms | 0.52 ms | 0.53 ms | **0.56 ms** | **$O(1)$ constant** |
| **PyTorch GRU** | 0.32 ms | 0.73 ms | 1.04 ms | 2.43 ms | 3.08 ms | **5.94 ms** | **$O(W^2)$ sequence unrolling** |
| **Soft-Voting Ensemble** | 1.73 ms | 2.41 ms | 2.22 ms | 3.68 ms | 4.89 ms | **8.27 ms** | **$O(W^2)$ dominated by GRU** |

### 9.2 Streaming 48-Hour Stay Simulation & Throughput
- **Logistic Regression:** Cumulative 48h stay processing time = **33.57 ms** (0.70 ms/hour). Batch Throughput = **5,426,525 pred/sec**.
- **XGBoost:** Cumulative 48h stay processing time = **39.89 ms** (0.83 ms/hour). Batch Throughput = **1,687,308 pred/sec**.
- **PyTorch GRU:** Cumulative 48h stay processing time = **159.16 ms** (3.32 ms/hour). Batch Throughput = **14,526 pred/sec**.
- **Soft-Voting Ensemble:** Cumulative 48h stay processing time = **251.78 ms** (5.25 ms/hour). Batch Throughput = **14,364 pred/sec**.

### 9.3 Peak Traced Heap Memory
- **Logistic Regression:** **26.88 KB**
- **XGBoost:** **255.60 KB**
- **PyTorch GRU:** **34.13 KB**
- **Soft-Voting Ensemble:** **284.10 KB**
- *Process RSS:* Stable at ~516 MB with zero memory leaks over 10,000 iterations.

---

## 10. Engineering Recommendations for Production & Future Work

### 10.1 Production Recommendation: Do Not Deploy the Ensemble
1. **Live Socket & WebSocket Path:** Keep the **Logistic Regression baseline** (`src/predict.py`). It executes in **0.42 ms** with **26 KB** memory allocation, providing deterministic, instant response over the TCP socket protocol.
2. **Server-Side Offline Analytics Candidate:** If the team wishes to upgrade discriminative power over Logistic Regression, **select standalone XGBoost**, not the Ensemble.
   - **Performance:** XGBoost achieves 0.0983 AUPRC (matching the Ensemble's 0.0970).
   - **Latency:** XGBoost runs in 0.56 ms (vs 8.27 ms for the Ensemble, a **~14.8x speedup**).
   - **Throughput:** XGBoost delivers 1.68M pred/sec (vs 14.3K pred/sec, a **~117x speedup**).
   - **Simplicity:** Eliminates the PyTorch dependency, sequence window slicing, and multi-model synchronization overhead.

### 10.2 Future Optimization Guide for Recurrent / Sequence Models
If future research requires deploying deep sequential models (GRU or Ensemble), the current $O(W^2)$ overhead must be resolved:
1. **Feature Vector Caching ($O(1)$ lookup):**
   - Current implementation re-extracts the 22 features 48 separate times for an isolated hour-48 prediction.
   - Fix: Cache the 22-dimensional float32 vector in SQLite/Redis as each hour arrives.
2. **Recurrent Hidden State Caching ($O(1)$ forward step):**
   - Current implementation re-runs the full 48-step sequence through the GRU from $h_0 = 0$ on every hour.
   - Fix: Persist the GRU hidden state vector $h_{t-1} \in \mathbb{R}^{48}$ in the patient session. When hour $t$ arrives, execute a single `GRUCell(x_t, h_{t-1})` forward step, cutting inference time by **~85%**.

---

## 11. Repository Artifacts & File Map

### Code & Interfaces
- [`src/ensemble.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/ensemble.py) — Core ensemble inference class (`SepsisEnsemble`).
- [`src/predict.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/predict.py) — Production live serving pipeline (unaltered Logistic Regression).
- [`src/features.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/features.py) — 22-dimensional causal feature extraction engine.
- [`src/store.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/src/store.py) — SQLite persistent store with causal observation slicing.

### Model Artifacts
- [`models/model.joblib`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/model.joblib) — Frozen Logistic Regression pipeline (`StandardScaler` + `LogisticRegression`).
- [`models/xgboost_model.joblib`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/xgboost_model.joblib) — Frozen XGBoost classifier.
- [`models/lstm_model.pt`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/lstm_model.pt) — Frozen PyTorch 2-layer GRU checkpoint (`SepsisRNN`).
- [`models/imputer.json`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/imputer.json) — Training cohort median imputer values.
- [`models/threshold.json`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/threshold.json) — Production threshold ($\tau = 0.2500$ default / $0.4200$ high-recall).
- [`models/results/ensemble_audit/ensemble_calibrator_audited.joblib`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/ensemble_calibrator_audited.joblib) — Audited Platt scaling model.

### Reports & Benchmarks
- [`models/results/ensemble_audit/ENSEMBLE_AUDIT_REPORT.md`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/ENSEMBLE_AUDIT_REPORT.md) — Deep-dive audit report.
- [`models/results/ensemble_audit/benchmark/BENCHMARK_REPORT.md`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/BENCHMARK_REPORT.md) — Efficiency and deployment feasibility report.
- [`models/results/ensemble_audit/audit_summary.json`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/audit_summary.json) — Structured evaluation and bootstrap numbers.
- [`models/results/ensemble_audit/benchmark/benchmark_results.json`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/benchmark_results.json) — Structured latency and memory statistics.

### Diagnostic Visualizations
- [`models/results/ensemble_audit/pr_roc_curves.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/pr_roc_curves.png) — Precision-Recall & ROC curves.
- [`models/results/ensemble_audit/calibration_curves.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/calibration_curves.png) — Reliability diagrams (uniform & quantile).
- [`models/results/ensemble_audit/paired_bootstrap_distributions.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/paired_bootstrap_distributions.png) — 1,000 paired bootstrap difference distributions.
- [`models/results/ensemble_audit/benchmark/latency_by_history_length.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/latency_by_history_length.png) — Latency scaling vs history length.
- [`models/results/ensemble_audit/benchmark/latency_percentiles_comparison.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/latency_percentiles_comparison.png) — p50, p95, p99 isolated latency.
- [`models/results/ensemble_audit/benchmark/streaming_cumulative_time.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/streaming_cumulative_time.png) — 48-hour streaming cumulative runtime.
- [`models/results/ensemble_audit/benchmark/throughput_memory_comparison.png`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/benchmark/throughput_memory_comparison.png) — Throughput and memory allocation comparison.

### Automated Test Suite
- [`tests/test_ensemble.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/tests/test_ensemble.py) — 11 unit tests for ensemble mechanics and contracts.
- [`tests/test_ensemble_audit.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/tests/test_ensemble_audit.py) — 5 regression tests for audit and calibration contracts.
- [`tests/test_benchmark_utilities.py`](file:///c:/Users/Mitansh/Desktop/SepsisSignal/tests/test_benchmark_utilities.py) — 4 benchmark utility tests.
- Run complete test suite:
  ```bash
  python -m tests.run_tests
  ```
