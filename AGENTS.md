# SepsisSignal: Agent Rules

Research prototype for studying early sepsis risk estimation on ICU time-series.
NOT a clinical tool. This disclaimer must appear in the README and on every dashboard page:
"Research prototype, not for clinical decision-making."

Reference docs (read when relevant, do not paste into this file):
- @docs/brief.md (full project spec)
- @docs/EVALUATION.md (split, target, threshold, metrics)
- @docs/DATA_CONTRACT.md (JSON schema between model, API and dashboard)
- @docs/TASKS.md (owners and definition of done)

## Stack
- Python: pandas, NumPy, scikit-learn, XGBoost or LightGBM, optional PyTorch (offline comparison only)
- Storage: local SQLite (WAL mode)
- Serving: one prediction function, exposed through a local TCP or WebSocket line server and FastAPI
- Dashboard: Next.js + React + TypeScript, Plotly or Recharts
- Everything runs locally. No patient data goes to any external service.

## Current task: Round 2 twist (causal, per-hour inference)

Keep the existing trained model. The live prediction path uses the Logistic Regression
baseline unless told otherwise, and the screen must say so:
"Live path: Logistic Regression baseline (research prototype)".

### 1. Storage and causality
- Store every patient-hour in SQLite as it arrives.
- A risk request for patient P at hour T may use ONLY rows for P with hour <= T.
- Predictions for a patient must come from a model trained on OTHER patients (patient-level split preserved).
- Past predictions are immutable. Once (patient, hour) has a stored prediction, a later hour must never change it.

### 2. Jury input (one line in, one line out)
Input:  `H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>`
Output: `R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>`

- An empty field means the signal was not measured. Example: `H|P001|2|88|112|||` has 3 empty fields.
- Parse with a plain split on "|". Never strip trailing pipes, or the empty count will be wrong.
- Count empty fields on the INCOMING record, before any imputation.
- Accept `\n` and `\r\n`. On malformed input, reply with one error line (`E|<reason>`) and keep the server running.
- Map the five signals to the features the model already uses. Features the five signals cannot supply
  are treated as unmeasured: training-set fill value plus the missingness flag set.
- Unseen patient IDs must work (no dataset lookup). The first hour of a patient has no history, so use the same sentinel values as training.
- Probability is formatted consistently (for example 4 decimals). The alert is 1 if probability >= the saved threshold, else 0.

### 3. Dashboard clock
- Add a clock (slider or scrubber) to the existing dashboard. Moving it to hour T shows the probability and alert for that hour.
- The clock must call the same `predict(patient, T)` as the socket server. It must never read a precomputed score file.
- Show the live-path label, the missing-vs-imputed legend, and the disclaimer.
- Demo patients for the held-out comparison come from the TEST split only.

## One prediction function, three callers
`predict(patient_id, hour)` is the only place a risk score is computed. The socket handler, the API and the
dashboard all call it. SQLite is the single source of truth. Do not duplicate feature logic.

Training and serving must share ONE feature function that takes only rows with hour <= T.
Do not write separate pandas "whole stay" features for training and a different path for serving.

## Hard rules
- Split by patient, never by row. Fit imputers and scalers on training patients only.
- Causal forward fill only. No backfill, no look-ahead, no rolling windows that include future rows.
- Choose the alert threshold on validation data only and store it with the model artifacts.
- Never hardcode or invent metrics. Every number shown comes from the saved model or a stored result file.
- Do not return one precomputed score for a whole patient stay.
- Keep missing, imputed and benchmark-label values visually distinct in the dashboard.
- Temperature unit is Celsius. State this assumption in the UI and README.
- Do not read, print, upload or commit raw patient files. The dataset lives OUTSIDE this repo folder. `data/` stays in .gitignore.
- Do not claim prediction before clinical symptoms or any clinical validity.
- Do not change the model, split or threshold unless the task says so. If a change seems necessary, stop and explain why first.
- Baseline first. A sequence model (LSTM or GRU) is offline comparison only and never goes into the live path in this round.

## SQLite
- Use WAL mode and short transactions.
- Suggested tables:
  - `observations(patient_id, hour, hr, sbp, resp, temp, wbc, received_at)` with unique (patient_id, hour)
  - `predictions(patient_id, hour, probability, alert, empty_count, model_name, created_at)` with unique (patient_id, hour)
- Duplicate hour for the same patient: return the stored prediction, do not recompute or overwrite.
- Out-of-order hours: compute using only rows with hour <= T that exist at that moment, and store that result. Do not rewrite earlier predictions.

## Acceptance tests (must pass before saying a task is done)
1. Send hours 1, 2, 3 for a test patient. Hour 2's reply is byte-identical before and after hour 3 arrives.
2. `H|P001|2|88|112|||` returns an empty-field count of 3 (greater than zero).
3. Stream a held-out patient through the socket hour by hour. Probabilities match the offline batch features within a tiny tolerance. Report any mismatch, do not hide it.
4. The dashboard clock at hour T shows the same probability and alert as the socket reply for that hour.
5. A patient ID never seen before works with no dataset lookup.
6. Malformed lines return an error line and do not crash the server.

## Working style
- Small, reviewable changes. Run the relevant test after each change.
- Prefer the simplest thing that satisfies the acceptance tests. No new dependencies without a reason.
- If something in the spec is ambiguous, ask before guessing.
- Report limits honestly: the live path sees only five signals, so expect different (often lower) probabilities than offline results. Say this in the UI notes, do not quietly retrain a different model.
- Leave a short note in `docs/TASKS.md` for anything left unfinished.

## Repo layout (suggested)
```
AGENTS.md
README.md
docs/            brief.md, EVALUATION.md, DATA_CONTRACT.md, TASKS.md
src/features.py  single point-in-time feature function
src/predict.py   predict(patient_id, hour)
src/store.py     SQLite access
src/server.py    TCP or WebSocket line server
api/             FastAPI app
dashboard/       Next.js app (with the clock)
models/          saved model, imputer values, threshold
tests/           acceptance tests above
```
