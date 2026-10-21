# Architecture

Research prototype, not for clinical decision-making.

## Principle
One source of truth (SQLite), one prediction function, three callers.
No caller computes risk on its own and no caller reads precomputed scores.

## Flow
```
Jury / replay client                 Dashboard (Next.js + React + TS)
       |                                      |
  H|... line in                         clock set to hour T
       v                                      v
 src/server.py (TCP or WS)            api/ (FastAPI: /predict /patients /timeline)
       \                                      /
        \________ predict(patient_id, hour) _/
                          |
        +-----------------+------------------+
        |                                    |
 src/store.py (SQLite, WAL)            src/features.py
 observations, predictions             point-in-time features, rows hour <= T only
        |                                    |
        +-----------------+------------------+
                          |
                   models/ (model.joblib, imputer.json, threshold.json)
```

## Components

| Component | File | Responsibility |
|---|---|---|
| Line server | `src/server.py` | Parse `H|...`, count empty fields, call predict, reply `R|...` or `E|...` |
| Predict | `src/predict.py` | `predict(patient_id, hour)`: read rows <= hour, build features, score, store once |
| Features | `src/features.py` | ONE point-in-time feature function, shared by training and serving |
| Store | `src/store.py` | SQLite access, WAL mode, unique (patient, hour) keys |
| API | `api/` | Thin FastAPI layer over predict and store |
| Dashboard | `dashboard/` | Timeline, clock, probability and alert, missing vs imputed legend |
| Artifacts | `models/` | Trained model, training-set fill values, validation-chosen threshold, splits, results |

## Request lifecycle (one `H|...` line)
1. Server splits on "|" and checks for exactly 8 fields.
2. Empty signal fields are counted before any imputation.
3. The observation is stored (duplicate hour returns the stored prediction).
4. `predict` loads that patient's rows with hour <= T and builds features.
5. Missing values use causal fill or the training-set fill value, plus a missingness flag.
6. The model scores the features; alert = probability >= saved threshold.
7. The prediction is stored once and never overwritten.
8. Server replies `R|<patient id>|<hour>|<probability>|<alert>|<empty count>`.

## Causality guarantees
- Query for hour T filters `hour <= T` for that patient only.
- Forward fill only, never backfill.
- Imputers and scalers are fit on training patients only.
- Past predictions are immutable. A later hour cannot change an earlier row.
- Out-of-order arrival: compute from rows that exist now and never rewrite earlier predictions.

## Offline versus live
| | Offline evaluation | Live path (Round 2) |
|---|---|---|
| Model | Logistic Regression, plus XGBoost/LightGBM, LSTM/GRU comparison | Logistic Regression baseline only |
| Inputs | Full benchmark feature set | HR, SBP, Resp, Temp, WBC |
| Patients | Held-out test split | Any ID; demo uses test-split patients |
| Purpose | Honest held-out comparison | Per-hour inference demo |

## Training and serving parity
Training calls the same `features.py` as serving. Parity test: stream a held-out patient through the
socket hour by hour and compare against offline batch features. Record the maximum difference.

## Failure handling
- Malformed line: `E|<reason>`, server stays up.
- Unseen patient ID: works, no dataset lookup. First hour uses training sentinel values.
- Concurrent writers: WAL mode, short transactions.
- Port blocked: switch to another local port; TCP is the fallback if WebSocket misbehaves.

## Out of scope
Clinical integration, authentication, multi-hospital deployment, real patient data, prospective validation.

## Decisions to record in MEMORY.md
- Reduced-feature model vs treating extras as unmeasured
- TCP vs WebSocket
- Probability decimal places and threshold value
