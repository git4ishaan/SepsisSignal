
# SepsisSignal

Explainable early-warning research for ICU time-series.

> **Research prototype, not for clinical decision-making.**

Team HackerSIM: Soham Paranjape, Ishaan Patil, Mitansh Bheda

## What it is
A leakage-resistant pipeline and dashboard for studying early sepsis risk estimation on public ICU data.
It shows missing data, alert trade-offs and honest limits, not just a score.

## Round 2: causal, per-hour inference
- Each patient-hour is stored in a local SQLite database as it arrives.
- A risk request for hour T uses only that patient's records with hour <= T.
- Past predictions are immutable: a later hour never changes an earlier one.
- Predictions for a patient come from a model trained on other patients (patient-level split).
- Live path: **Logistic Regression baseline (research prototype)**.
- The dashboard has a clock. Moving it shows the probability and alert for the selected hour.

## Data
PhysioNet/Computing in Cardiology Challenge 2019 (public, CC BY 4.0), 40,336 patients, hourly records.
Reyna et al., Crit Care Med 2020. DOI 10.13026/v64v-d857

The raw dataset is NOT in this repository. Download it from PhysioNet and keep it outside this folder.
Set its location in `[config file / env var]`.

## Quick start
```bash
# 1. Start the line server (TCP - default port 8765)
python -m src.server --port 8765

# 2. Start the FastAPI backend (default port 8000)
uvicorn api.main:app --port 8000 --reload

# 3. Start the Next.js dashboard (default port 3000)
cd dashboard && npm run dev

# 4. Run automated acceptance test suite
python -m tests.run_tests
```


## Line protocol
```
In : H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>
Out: R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>
```
Empty field = not measured. Example:
```bash
echo "H|P001|2|88|112|||" | nc localhost [PORT]
```
Expected shape: `R|P001|2|<probability>|<0 or 1>|3`

Full details: `docs/DATA_CONTRACT.md`

## Evaluation
Patient-level split, causal preprocessing, threshold chosen on validation data only, metrics on held-out patients.
Rules: `docs/EVALUATION.md`. Results are read from `models/results/` and never typed by hand.

| Model | AUPRC | Precision | Recall |
|---|---|---|---|
| Logistic Regression (live path) | [__] | [__] | [__] |
| XGBoost / LightGBM (offline) | [__] | [__] | [__] |
| LSTM / GRU (offline, stretch) | [__] | [__] | [__] |

## Tech stack
Python (pandas, NumPy, scikit-learn, XGBoost or LightGBM, optional PyTorch), SQLite, FastAPI,
Next.js + React + TypeScript, Plotly or Recharts. Everything runs locally.

## Limitations
- Retrospective public benchmark; may not match any real hospital.
- Does not prove prediction before clinical symptoms.
- The live path sees only five signals (HR, SBP, Resp, Temp, WBC), so probabilities can differ from the full-feature offline results.
- Hourly-row metrics are not the same as episode-level alert precision.
- No external or prospective validation.
- Next: external validation, silent-mode testing, clinician feedback, privacy, security and regulatory review.

## Tools used
Canva (deck), Claude (brainstorming, drafting, coding help), Google Antigravity (AI-assisted coding).
No patient data is uploaded to any AI tool; model training and evaluation run locally.

## Repository layout
```
AGENTS.md  README.md
docs/      brief.md  EVALUATION.md  DATA_CONTRACT.md  TASKS.md  MODEL_TRAINING_GUIDE.md
scripts/   train_baseline.py
src/       features.py  predict.py  store.py  server.py
api/       FastAPI app
dashboard/ Next.js app (with the clock)
models/    model, imputer values, threshold, splits, results
tests/     acceptance tests
```