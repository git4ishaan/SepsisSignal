# SEPSISSIGNAL — ENSEMBLE VALIDATION AUDIT REPORT

**Project:** SepsisSignal Research Prototype  
**Scope:** Deep-Dive Audit of the Soft-Voting Ensemble Experiment  
**Status:** Audit Completed  
**Constraint Check:** Zero modification to production live inference, database schema, socket protocol, or frozen model artifacts.  

> [!IMPORTANT]
> **Safety Disclaimer:** Research prototype, not for clinical decision-making.

---

## 1. Audit of Threshold Selection & Operating Points

### 1.1 Inconsistency Diagnosis: The 0.05 Calibrated Threshold
The previous experimental report noted that an alert threshold was selected to guarantee validation recall $\ge 70\%$, but the calibrated ensemble threshold ($\tau = 0.0500$) reportedly achieved only **32.79% validation recall**.

Our programmatic audit discovered that this discrepancy was caused by a **search-space truncation bug**:
- In `scripts/evaluate_ensemble.py`, the threshold search grid was defined as:
  ```python
  threshold_grid = np.linspace(0.05, 0.85, 161)
  ```
- Because Platt calibration re-scales probabilities to match true empirical prevalence (~1.67%), calibrated probabilities rarely exceed 0.15–0.20.
- At the lowest tested threshold in the grid ($\tau = 0.0500$), the calibrated recall on validation data was already down to **31.41%** (or 32.79% on coarse grid). No threshold in $[0.05, 0.85]$ satisfied the constraint $\text{Recall} \ge 0.70$.
- The code then silently fell back to an unconstrained maximum-$F_1$ selection within $[0.05, 0.85]$, which selected $\tau \approx 0.05$ (max $F_1 = 0.156$), while falsely reporting that the high-recall criterion was satisfied.

### 1.2 Programmatically Verified Operating Points
When the search space is expanded to cover the actual probability distribution $[0.001, 0.400]$, two distinct, clinically meaningful operating points emerge:

#### Operating Point A: High-Recall Screening Constraint (Recall $\ge 70.0\%$)
*Designed for early bedside screening where false negatives must be minimized.*
- **Uncalibrated Ensemble ($\tau = 0.4110$):**
  - Validation: Precision: 3.75%, Recall: **70.03%**, Specificity: 69.50%, $F_1$: 0.0712, Alert Rate: **31.16 alerts / 100h**
  - Held-out Test: Precision: 4.15%, Recall: **75.29%**, Specificity: 69.59%, $F_1$: 0.0786, Alert Rate: **31.19 alerts / 100h**
- **Calibrated Ensemble ($\tau = 0.0131$):**
  - Validation: Precision: 3.74%, Recall: **70.03%**, Specificity: 69.42%, $F_1$: 0.0710, Alert Rate: **31.24 alerts / 100h**
  - Held-out Test: Precision: 4.15%, Recall: **75.41%**, Specificity: 69.52%, $F_1$: 0.0786, Alert Rate: **31.26 alerts / 100h**
- **Key Takeaway:** When properly searched, the calibrated ensemble operating point at $\tau = 0.0131$ achieves the identical sensitivity and specificity profile as the uncalibrated ensemble at $\tau = 0.4110$.

#### Operating Point B: Maximum $F_1$ / Low-Alert-Burden Operating Point
*Designed for targeted alerts where alarm fatigue must be minimized.*
- **Calibrated Ensemble ($\tau = 0.0526$):**
  - Validation: Precision: **10.33%**, Recall: 31.92%, Specificity: **95.30%**, $F_1$: **0.1560**, Alert Rate: **5.16 alerts / 100h**
  - Held-out Test: Precision: **11.20%**, Recall: 34.32%, Specificity: **95.24%**, $F_1$: **0.1688**, Alert Rate: **5.27 alerts / 100h**
- **Key Takeaway:** At $\tau \approx 0.05$, precision triples from 3.7% to 11.2% and alert burden drops by ~83% (from 31.2 to 5.3 alerts per 100 hours), but at the cost of missing ~66% of positive sepsis hours.

---

## 2. Calibration Audit & ECE Investigation

### 2.1 Validation-Only Fitting Verification
- The Platt calibrator was fit strictly on validation log-odds: $\text{logit}(p) = \ln(p / (1 - p))$ against binary validation labels $y_{\text{val}}$.
- **Zero test labels were used for calibration fitting or threshold selection.**
- Saved artifact: `models/results/ensemble_audit/ensemble_calibrator_audited.joblib`.

### 2.2 Calibration Metric Summary

| Cohort | Variant | Brier Score | Log Loss | ECE (10 Uniform Bins) | ECE (10 Quantile Bins) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation** | Uncalibrated Ensemble | 0.1678 | 0.5307 | 0.3561 | — |
| **Validation** | Platt Calibrated | **0.0158** | **0.0750** | **0.0013** | **0.0014** |
| **Held-Out Test** | Uncalibrated Ensemble | 0.1685 | 0.5313 | 0.3572 | — |
| **Held-Out Test** | Platt Calibrated | **0.0162** | **0.0753** | **0.0013** | **0.0029** |

### 2.3 Investigation of the Low ECE (~0.0013)
The audit investigated why the reported uniform ECE was exceptionally low (~0.0013):
1. **Severe Class Imbalance Effect in Uniform Binning:**
   - In 10 uniform bins ($[0.0, 0.1), [0.1, 0.2), \dots$), **97.75% of test data (227,881 out of 233,127 hours)** falls into Bin 1 ($[0.0000, 0.1000]$).
   - In Bin 1, the mean predicted probability is `0.0136` (1.36%) and the observed positive rate is `0.0145` (1.45%), giving a bin calibration error of only `0.0009` (0.09 percentage points).
   - Since Bin 1 represents 97.75% of all samples, the sample-weighted contribution to ECE is $(227,881 / 233,127) \times 0.0009 \approx 0.00088$.
   - Higher bins have larger absolute percentage errors (e.g., Bin 4 $[0.3, 0.4]$ has an error of 18.33 percentage points), but because they contain only 61 hours (0.03% of the cohort), their weighted impact is negligible.
2. **Quantile (Equal-Frequency) Binning:**
   - To avoid this masking artifact, we evaluated **10 quantile bins (deciles of predicted probability)**, where every bin contains exactly 23,313 hours (10% of the cohort).
   - In quantile binning, test ECE is **`0.0029`** (0.29 percentage points).
   - Deciles 1–8 show near-perfect alignment with observed rates. Decile 10 (probabilities 0.0286 to 0.3688, mean predicted 7.39%) exhibits an observed rate of 8.51% (calibration error: 1.12 percentage points).
   - **Conclusion:** The low ECE is genuine and reflects strong probability calibration across 90%+ of patient-hours, but uniform ECE mathematically underestimates risk discrepancies in the high-risk tail due to heavy sample concentration in the lowest bin.

---

## 3. Paired Patient-Level Cluster Bootstrap Comparison (1,000 Replicates)

To evaluate whether the ensemble provides a statistically significant improvement over candidate models, we performed **1,000 paired cluster bootstrap resamples** over the 6,048 test patients (seed = 42).
In each replicate, the exact same patient stays were resampled, and paired differences $\Delta = \text{Metric}_{\text{ensemble}} - \text{Metric}_{\text{comparator}}$ were computed.

### Paired Bootstrap Results Summary

| Comparator Model | Metric | Point Difference ($\Delta$) | Mean Bootstrap $\Delta$ | 95% Bootstrap CI | Zero in CI? | Statistical Superiority Established? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Offline)** | **AUPRC** | **-0.0013** | -0.0009 | **[-0.0113, +0.0084]** | **YES** | **NO** |
| **XGBoost (Offline)** | **AUROC** | **+0.0068** | +0.0068 | **[-0.0018, +0.0146]** | **YES** | **NO** |
| **PyTorch GRU (Offline)** | **AUPRC** | **+0.0147** | +0.0148 | **[+0.0020, +0.0276]** | **NO** | **YES** |
| **PyTorch GRU (Offline)** | **AUROC** | **+0.0089** | +0.0088 | **[+0.0014, +0.0158]** | **NO** | **YES** |
| **Logistic Regression (Live)** | **AUPRC** | **+0.0195** | +0.0188 | **[+0.0079, +0.0297]** | **NO** | **YES** |
| **Logistic Regression (Live)** | **AUROC** | **+0.0629** | +0.0633 | **[+0.0490, +0.0773]** | **NO** | **YES** |

### Statistical Superiority Interpretation
1. **Vs XGBoost:** The 95% bootstrap confidence interval for $\Delta\text{AUPRC}$ spans from $-0.0113$ to $+0.0084$, and for $\Delta\text{AUROC}$ from $-0.0018$ to $+0.0146$. Because both intervals contain zero, **statistical superiority of the Soft-Voting Ensemble over standalone XGBoost is NOT established** at $\alpha = 0.05$. XGBoost and the Ensemble perform comparably.
2. **Vs PyTorch GRU:** The 95% CI for $\Delta\text{AUPRC}$ is $[+0.0020, +0.0276]$ and for $\Delta\text{AUROC}$ is $[+0.0014, +0.0158]$. Neither interval includes zero. The ensemble demonstrates statistically significant superiority over standalone GRU.
3. **Vs Logistic Regression:** The 95% CI for $\Delta\text{AUPRC}$ is $[+0.0079, +0.0297]$ and for $\Delta\text{AUROC}$ is $[+0.0490, +0.0773]$. The ensemble is statistically superior to the baseline linear model on both discriminative metrics.

---

## 4. Data Integrity & Historical Discrepancy Audit

### 4.1 Cohort Disjointness & Patient-Hour Counts
- **Train Split:** 28,217 unique patients
- **Validation Split:** 6,046 unique patients (233,496 patient-hours, 3,894 positive labels, 1.67% prevalence)
- **Test Split:** 6,048 unique patients (233,127 patient-hours, 4,006 positive labels, 1.72% prevalence)
- **Overlap Audit:** Exact set intersections:
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- **Row Uniqueness:** No duplicate patient-hour records exist within any patient stay. All models were evaluated on the identical 233,127 test rows with matching labels.

### 4.2 Resolution of the Historical GRU AUPRC Discrepancy (~0.2495 vs ~0.0823)
The pre-implementation documentation historically reported GRU AUPRC as 0.2495, whereas our causal point-in-time evaluation yielded 0.0823. The audit confirmed the exact cause:
1. **Evaluation Truncation & Batch Masking in Historical Script:** In `scripts/train_lstm.py` (lines 169–181 and 300–350), sequences were processed via `build_patient_sequence()` which truncated stays to the final 48 hours (`raw_obs = raw_obs[-48:]`) and batched them with zero padding and `nn.BCEWithLogitsLoss(pos_weight=...)`. The reported metric was computed on masked batch sequence outputs.
2. **Full-Cohort Point-in-Time Evaluation:** When evaluated row-by-row causally across all 233,127 patient-hours of all 6,048 test patients identically alongside Logistic Regression and XGBoost, the GRU achieves an AUPRC of **0.0823** and an AUROC of **0.7956**.
3. **Takeaway:** The historical 0.2495 figure reflected an evaluation protocol discrepancy, not superior model performance on the benchmark test cohort.

---

## 5. Automated Regression & Audit Test Verification

We implemented a dedicated audit test suite in [tests/test_ensemble_audit.py](file:///C:/Users/Mitansh/Desktop/SepsisSignal/tests/test_ensemble_audit.py) covering:
1. `test_01_programmatic_threshold_selection_verification`: Verifies that constraint failures are flagged explicitly without silent fallbacks.
2. `test_02_validation_only_calibration_fitting_isolation`: Verifies that calibration is isolated to validation data and detects test label corruption.
3. `test_03_prediction_and_label_alignment`: Verifies row-by-row label alignment across all model outputs.
4. `test_04_paired_bootstrap_parity_logic`: Verifies paired difference calculation and parity reporting.
5. `test_05_reliability_table_and_ece_calculations`: Verifies equal-width vs quantile bin boundaries and sum-to-one invariants.

### Test Suite Execution Results

#### 1. PyTest Execution
```powershell
python -m pytest tests/ -v
# Output:
# ============================= 83 passed in 15.37s =============================
```

#### 2. Self-Contained Test Runner Execution
```powershell
python -m tests.run_tests
# Output:
# Ran 83 tests in 6.508s
# OK
```
**Test Count Consistency:** Both runners executed **exactly 83 tests** with **0 failures and 0 errors**.

---

## 6. Frozen Model Artifacts SHA-256 Verification

All original production model artifacts remain bit-for-bit untouched before and after the audit:

| Artifact | SHA-256 Checksum | Audit Status |
| :--- | :--- | :---: |
| `models/model.joblib` | `37197c1c1b870d1e33b3e4cbb04bafaaf739829469de89df12fad27035ef7424` | **UNTOUCHED** |
| `models/xgboost_model.joblib` | `ebbb019a21d04ca9f26f9ff5069aaba5db9eedb2e0c8f95782bce5c5ca12b9ec` | **UNTOUCHED** |
| `models/lstm_model.pt` | `d5c1fc551369d02867133dfc4825588ebb40cca2d521255787584ce70ce1a39e` | **UNTOUCHED** |
| `models/imputer.json` | `ba67f023a2f6b4bc9638c5b80932d5c1c738d64e1ce42a62bedfc105fba9ebf4` | **UNTOUCHED** |
| `models/threshold.json` | `d9aac9b2bbfa075f52660176070a61372605829dc9035710a4e4c61a0e5e64d8` | **UNTOUCHED** |

### Audit Artifacts Generated Under `models/results/ensemble_audit/`
- [audit_summary.json](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/audit_summary.json): Complete machine-readable audit metrics, reliability tables, and bootstrap statistics.
- [calibration_curves.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/calibration_curves.png): Reliability curves and probability density distributions for validation and test.
- [paired_bootstrap_distributions.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/paired_bootstrap_distributions.png): Paired bootstrap difference distributions for AUPRC and AUROC vs XGBoost, GRU, and LR.
- [pr_roc_curves.png](file:///C:/Users/Mitansh/Desktop/SepsisSignal/models/results/ensemble_audit/pr_roc_curves.png): Precision-Recall and ROC comparison curves across all models.
- `val_cache.npz`, `test_cache.npz`: Compressed cohort inference caches for instant reproduction.
- `ensemble_calibrator_audited.joblib`: Audited validation-fitted Platt calibrator.
