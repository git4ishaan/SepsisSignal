# Data Contract

One schema shared by the model, the line server, the API and the dashboard.
All three callers use the same `predict(patient_id, hour)` function. SQLite is the source of truth.

## 1. Jury line protocol (TCP or WebSocket, one line in, one line out)

Input:
```
H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>
```
Output:
```
R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>
```
Error:
```
E|<reason>
```

Rules:
- Split on "|". Never strip trailing pipes.
- Empty field = not measured. `H|P001|2|88|112|||` has 3 empty signal fields (Resp, Temp, WBC).
- Count empty fields on the incoming record, before imputation.
- Accept `\n` and `\r\n`. Malformed input returns an `E|...` line and the server keeps running.
- Probability: fixed decimals ([4]). Alert: 1 if probability >= saved threshold, else 0.

Note on the example: `H|P001|2|88|112|||` splits into 8 fields: H, P001, 2, 88, 112, "", "", "".
HR=88 and SBP=112 are present, while Resp, Temp and WBC are empty, so the empty count is 3.
A valid line always has exactly 8 fields. Anything else returns an `E|...` line.

## 2. Signal mapping

| Jury field | Model feature(s) | Unit | If empty |
|---|---|---|---|
| HR | `hr_val`, `hr_missing`, `hr_time_since`, `hr_delta` | beats/min | causal forward fill (or 84.0), missing flag = 1, time_since = 99.0 |
| SBP | `sbp_val`, `sbp_missing`, `sbp_time_since`, `sbp_delta` | mmHg | causal forward fill (or 118.0), missing flag = 1, time_since = 99.0 |
| Resp | `resp_val`, `resp_missing`, `resp_time_since`, `resp_delta` | breaths/min | causal forward fill (or 18.0), missing flag = 1, time_since = 99.0 |
| Temp | `temp_val`, `temp_missing`, `temp_time_since`, `temp_delta` | Celsius | causal forward fill (or 36.8), missing flag = 1, time_since = 99.0 |
| WBC | `wbc_val`, `wbc_missing`, `wbc_time_since`, `wbc_delta` | 10^3/uL | causal forward fill (or 10.5), missing flag = 1, time_since = 99.0 |

Context features: `hour` (current ICU stay hour) and `current_empty_count` (0 to 5 count of unmeasured signals at hour T).
Features the five signals cannot supply: treated as unmeasured (training-set fill + missingness flag set).


## 3. SQLite schema (WAL mode)

```sql
CREATE TABLE IF NOT EXISTS observations (
  patient_id  TEXT    NOT NULL,
  hour        INTEGER NOT NULL,
  hr REAL, sbp REAL, resp REAL, temp REAL, wbc REAL,
  received_at TEXT    NOT NULL,
  PRIMARY KEY (patient_id, hour)
);

CREATE TABLE IF NOT EXISTS predictions (
  patient_id   TEXT    NOT NULL,
  hour         INTEGER NOT NULL,
  probability  REAL    NOT NULL,
  alert        INTEGER NOT NULL,
  empty_count  INTEGER NOT NULL,
  model_name   TEXT    NOT NULL,
  created_at   TEXT    NOT NULL,
  PRIMARY KEY (patient_id, hour)
);
```

Behavior:
- A prediction for (patient, T) uses only observations with hour <= T that exist at that moment.
- Predictions are immutable. A duplicate (patient, hour) returns the stored prediction.
- Out-of-order arrival: compute for that hour from rows that exist now. Never rewrite earlier predictions.

## 4. API (FastAPI)

`GET /predict?patient_id=P001&hour=2`
```json
{
  "patient_id": "P001",
  "hour": 2,
  "probability": 0.0000,
  "alert": 0,
  "empty_count": 3,
  "model_name": "logistic_regression_baseline",
  "threshold": 0.0000,
  "live_path_label": "Live path: Logistic Regression baseline (research prototype)"
}
```

`GET /patients` returns the patients available in SQLite, with a `split` field ("test" for held-out demo patients).

`GET /timeline?patient_id=P001&upto=5` returns stored observations and predictions with hour <= upto:
```json
{
  "patient_id": "P001",
  "points": [
    {"hour": 1, "values": {"hr": 90, "sbp": 118, "resp": null, "temp": null, "wbc": null},
     "missing": ["resp", "temp", "wbc"], "probability": 0.0000, "alert": 0}
  ]
}
```

The dashboard clock calls `/predict` (or the same function) for the selected hour. It never reads a precomputed score file.

## 5. Model artifacts (`models/`)
- `model.joblib`: the trained Logistic Regression
- `imputer.json`: training-set fill values per feature
- `threshold.json`: `{"threshold": [__], "chosen_on": "validation"}`
- `splits/`: patient ID lists for train, validation, test
- `results/`: metric files that the dashboard and slides read from
