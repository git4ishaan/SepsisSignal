# Evaluation Rules

These rules apply to every number shown in the dashboard, README and slides.
Research prototype, not for clinical decision-making.

## Dataset
- PhysioNet/Computing in Cardiology Challenge 2019 (public, CC BY 4.0)
- 40,336 patients, hourly records, `SepsisLabel` per hour
- Citation: Reyna et al., Crit Care Med 2020. DOI 10.13026/v64v-d857
- Raw files live OUTSIDE the repo folder. Never commit or upload them.

## Target
- Per patient-hour binary label: `SepsisLabel` from the benchmark.
- This is a benchmark label, not proof of clinical diagnosis. Say "benchmark label" in the UI.

## Split
- Patient-level only. A patient appears in exactly one of train, validation, test.
- Split ratio: 70 / 15 / 15 (Train: 28,217 / Val: 6,046 / Test: 6,048 patients). Random seed: 42
- Stratify by whether the patient ever has a positive label: no (patient-level random permutation)
- Save the patient ID lists for each split in `models/splits/` so the split is reproducible.
- Predictions for a patient must come from a model trained on OTHER patients.

## Preprocessing (training and serving share the same function)
- Causal forward fill only (use the last observed value up to hour T). No backfill.
- Missingness flag per signal, plus time since last measurement.
- Imputers and scalers are fit on TRAIN patients only.
- Rows with hour > T are never visible when predicting hour T.

## Threshold
- Chosen on VALIDATION data only. Selection rule: Targeted high recall (>=0.70) while maximizing F1 on validation set.
- Final threshold: 0.4200. Stored with the model artifacts in `models/threshold.json`.
- Never retuned on the test set.

## Metrics (held-out test patients)
Report all of these, with the threshold above:
- AUPRC (primary, because positive hours are rare)
- Precision, recall, specificity
- Confusion matrix
- Alert burden: [alerts per N patient-hours]
- Lead time: only if it can be computed honestly, otherwise write "not reported"
- Calibration (Brier score or plot): include if time allows, otherwise "not done"
- Accuracy alone is NOT reported as a headline number.

## Model comparison
- Same test patients and same target for every model.
- Baseline: Logistic Regression (this is the LIVE path in Round 2).
- Optional offline comparison: XGBoost/LightGBM and LSTM/GRU.
- If the baseline beats a more complex model, say so.

## Live path (Round 2 twist)
- The live path sees only 5 signals: HR, SBP, Resp, Temp, WBC.
- Features it cannot supply are treated as unmeasured (training-set fill + missingness flag).
- Expect different, often lower, probabilities than the full-feature offline results. Report this openly.
- Streaming parity check: stream a held-out patient through the socket hour by hour and compare to the offline
  batch features. Report the maximum difference. Do not hide mismatches.

## Results table (fill from saved result files only)
| Model | AUPRC | Precision | Recall | Specificity | Threshold | Alert Burden |
|---|---|---|---|---|---|---|
| Logistic Regression (live path) | 0.0775 | 0.0301 | 0.7374 | 0.5849 | 0.4200 | 42.1 / 100 patient-hours |
| XGBoost / LightGBM (offline) | 0.0983 | 0.0380 | 0.7644 | 0.6613 | 0.3900 | 34.6 / 100 patient-hours |
| LSTM / GRU (offline, stretch) | not yet computed | not yet computed | not yet computed | not yet computed | not yet computed | not yet computed |

## Limits to state every time
- Retrospective public benchmark, may not match any real hospital.
- Does not prove prediction before clinical symptoms.
- Hourly-row metrics are not the same as episode-level alert precision.
- No external or prospective validation.

## Rules for numbers
- Never hardcode or invent a metric. Every number comes from a saved result file.
- If a result is not computed yet, write "not yet computed", not an estimate.
