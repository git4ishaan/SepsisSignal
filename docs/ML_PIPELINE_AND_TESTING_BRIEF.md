# SepsisSignal: ML Pipeline Briefing & Test Generation Specification

> **Context for AI Test Generator:**  
> This specification documents the machine learning pipeline, feature engineering rules, models, and protocols built for **SepsisSignal** (an early sepsis risk estimation research prototype evaluated on the PhysioNet 2019 ICU dataset).  
> **Goal:** Use this briefing to generate automated test cases (pytest / unittest) that thoroughly validate all three models (**Logistic Regression**, **XGBoost**, and **PyTorch GRU/LSTM**), causal invariants, edge cases, and inference protocols.
>
> **Mandatory Disclaimer:** *"Research prototype, not for clinical decision-making."*

---

## 1. System Overview & Core Invariants

SepsisSignal predicts hourly sepsis risk for ICU patients using only **5 clinical signals**:
- `HR`: Heart Rate (bpm)
- `SBP`: Systolic Blood Pressure (mmHg)
- `Resp`: Respiration Rate (breaths/min)
- `Temp`: Core Body Temperature (**in Celsius**)
- `WBC`: White Blood Cell Count ($10^3/\mu\text{L}$)

### Non-Negotiable Hard Rules (Must be Tested):
1. **Strict Causality (Zero Future Look-Ahead):**
   - When predicting risk for hour $T$, the pipeline can **only** inspect records with $\text{hour} \le T$. Future rows ($\text{hour} > T$) must never leak into features or predictions.
2. **Prediction Immutability:**
   - Once a prediction for $(\text{patient\_id}, \text{hour})$ is computed and stored, subsequent arrival of later hours (or out-of-order rows) must **never** alter earlier stored predictions.
3. **Patient-Level Split (Zero Patient Leakage):**
   - Splitting is performed by patient, never by row. All rows of a patient belong strictly to one cohort (Train, Validation, or Test).
4. **Out-of-Order Ingestion:**
   - Arrival of observations out of temporal order must be handled causally without crashing or rewriting past states.
5. **Cold-Start / Unseen Patients:**
   - Patients never seen in training must execute seamlessly without external dataset lookup, initializing from baseline training medians.

---

## 2. Feature Pipeline (`src/features.py`)

All models and live serving rely on the single feature extraction contract:
`extract_features_point_in_time(observations, target_hour, imputer=None)`

### Feature Specification:
- Produces a deterministic **22-dimensional float32 vector** (`FEATURE_NAMES`):
  - For each of the 5 signals (`hr`, `sbp`, `resp`, `temp`, `wbc`):
    1. `{sig}_val`: Current measurement, causal forward-fill, or training median if never measured.
    2. `{sig}_missing`: Binary indicator (`0.0` if measured at hour $T$, `1.0` if missing).
    3. `{sig}_time_since`: Hours since last valid measurement (`0.0` if measured at $T$, sentinel `99.0` if never).
    4. `{sig}_delta`: Difference from immediately preceding measurement (`0.0` if $\le 1$ measurement in history).
  - Stay Context:
    5. `hour`: Integer hour of ICU stay ($1, 2, \dots$).
    6. `current_empty_count`: Number of unmeasured signals at hour $T$ ($0$ to $5$).

### Training Imputer Fill Values (`models/imputer.json`):
Computed strictly on the training cohort (28,217 patients):
```json
{
  "HR": 85.0,
  "SBP": 120.0,
  "RESP": 18.0,
  "TEMP": 36.9,
  "WBC": 9.6
}
```

---

## 3. The Three Models Under Test

| Property | 1. Live Path: Logistic Regression | 2. Offline Comparison: XGBoost | 3. Offline Benchmark: PyTorch GRU |
|---|---|---|---|
| **Role** | **Live serving path** (FastAPI, TCP line server, dashboard) | Offline tabular tree benchmark | Offline recurrent sequence benchmark |
| **Artifact Path** | `models/model.joblib` | `models/xgboost_model.joblib` | `models/lstm_model.pt` |
| **Threshold File** | `models/threshold.json` (`0.4200`) | `models/xgboost_threshold.json` (`0.3900`) | `models/lstm_threshold.json` (`0.4200`) |
| **Architecture** | `StandardScaler` + `LogisticRegression(class_weight="balanced")` | `xgb.XGBClassifier(tree_method="hist", max_depth=5, n_estimators=150)` | `nn.Linear(22, 48)` $\to$ `nn.GRU(48, 48)` $\to$ `nn.Linear(48, 1)` |
| **Training Input** | Flat 22-dim point-in-time vector | Flat 22-dim point-in-time vector | Sequence of 22-dim vectors (max window 48h, padded) |
| **Evaluation AUPRC** | `0.0775` | `0.0983` (+26.8%) | `0.2495` (+222%) |
| **Recall / Specificity** | `73.74%` / `58.49%` | `76.44%` / `66.13%` | `75.31%` / `76.03%` |
| **Alert Burden** | `42.1` alerts / 100h | `34.6` alerts / 100h | `24.9` alerts / 100h |

---

## 4. Live Serving & Protocol Contracts

### 4.1. Jury Line Protocol (`src/server.py`)
- **Incoming Format:** `H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>\n`
  - Empty field means unmeasured.
  - Example: `H|P001|2|88|112|||\n` has **3 empty fields** (`Resp`, `Temp`, `WBC`).
  - Splitting must preserve trailing empty pipes without stripping!
- **Outgoing Format:** `R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<empty count>\n`
  - Example: `R|P001|2|0.5214|1|3\n`
- **Error Format:** `E|<reason>\n`
  - Malformed lines (wrong prefix, non-numeric hour, insufficient fields) must reply with `E|<reason>` and **keep the server alive**.
  - Must accept both `\n` and `\r\n`.

### 4.2. Single Prediction Function (`src/predict.py`)
`predict(patient_id: str, hour: int) -> Dict[str, Any]`
- Serves as the single source of truth for the socket server, FastAPI backend (`api/main.py`), and Next.js clock scrubber.
- Queries local SQLite database (`src/store.py`).

### 4.3. SQLite Storage Contract (`src/store.py`)
- Tables in WAL mode:
  - `observations(patient_id, hour, hr, sbp, resp, temp, wbc, received_at)` with `UNIQUE(patient_id, hour)`.
  - `predictions(patient_id, hour, probability, alert, empty_count, model_name, created_at)` with `UNIQUE(patient_id, hour)`.

---

## 5. Scope & Test Categories Needed

The AI test generator should produce comprehensive test cases covering:

### Test Category A: Model Loading & Shape Parity
1. **Logistic Regression:** Load `models/model.joblib`. Verify it outputs probabilities in $[0, 1]$ for 22-dimensional input. Verify threshold is loaded from `models/threshold.json`.
2. **XGBoost:** Load `models/xgboost_model.joblib`. Verify predictions on 22-dimensional input match expected classes $\{0, 1\}$.
3. **PyTorch GRU:** Load checkpoint `models/lstm_model.pt`. Verify model architecture (`state_dict`), forward pass on shape `(batch, seq_len, 22)`, and sigmoid probability output.
4. **Model Protection:** Assert that running test suites or evaluations never mutates or overwrites `models/model.joblib`.

### Test Category B: Causal Integrity & Look-Ahead Prevention
1. **Zero Future Leakage:** For a patient with observations at hours 1, 2, 3, 4, 5, `extract_features_point_in_time(obs, target_hour=2)` must return identical feature vectors regardless of whether hours 3, 4, and 5 exist in the input list.
2. **Immutability of Predictions:** Calling `predict("P_TEST", 2)` before and after inserting hour 3 data must produce byte-identical probability and alert values.
3. **Out-of-Order Hours:** Ingesting hour 5 first, then hour 2, must compute hour 2 using strictly observations with $\text{hour} \le 2$.

### Test Category C: Cold Start, Missingness, and Physiological Edge Cases
1. **First-Hour Sentinel Values:** An unseen patient at hour 1 with no vitals measured (`H|P_NEW|1|||||`) must use training medians for values, `1.0` for missing flags, `99.0` for `time_since`, `0.0` for `delta`, and `5` for empty count.
2. **Trailing Pipes Preservation:** `H|P_TEST|2|80|120|||` must parse exactly 3 empty fields; trailing pipes must not be trimmed.
3. **Extreme Physiological Values:** Test model response to high fever (`Temp = 41.5`), hypothermia (`Temp = 34.0`), tachycardia (`HR = 180`), severe bradycardia (`HR = 30`), and septic shock hypotension (`SBP = 60`). Output probabilities must remain valid floats between $0.0$ and $1.0$.

### Test Category D: Protocol & Server Resilience
1. **Line Terminators:** Server must correctly handle both LF (`\n`) and CRLF (`\r\n`).
2. **Malformed Commands:**
   - Bad prefix: `X|P001|1|80|120|20|37|10`
   - Non-numeric hour: `H|P001|two|80|120|20|37|10`
   - Incomplete columns: `H|P001|1|80`
   - Extra pipe delimiters.
   - All must return `E|<reason>` without crashing.
3. **Streaming vs Single Parity:** Streaming hours 1 through 10 sequentially must match single batch `predict()` calls at each hour within $10^{-5}$ tolerance.

### Test Category E: Benchmark Metrics & Threshold Consistency
1. Verify `models/results/metrics.json` contains valid numeric entries for all three models (`Logistic Regression`, `XGBoost`, `LSTM / GRU`).
2. Assert `0.0 <= auprc <= 1.0`, `0.0 <= precision <= 1.0`, `0.0 <= recall <= 1.0`, `0.0 <= specificity <= 1.0`.
3. Verify `models/threshold.json` has `threshold` set to `0.42`.
4. Verify `models/xgboost_threshold.json` has `threshold` set to `0.39`.
5. Verify `models/lstm_threshold.json` has `threshold` set to `0.42`.

---

## 6. Execution Command for Acceptance Suite
The primary test harness in the repo is:
```bash
python -m tests.run_tests
```
Any newly generated test file (e.g. `tests/test_all_models.py`) can be run with:
```bash
python -m pytest tests/
# or
python -m unittest discover tests
```
