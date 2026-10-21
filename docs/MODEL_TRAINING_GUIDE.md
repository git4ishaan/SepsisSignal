# SepsisSignal — Beginner's Guide to Model Training

> **Target Audience:** HackerSIM Teammate in charge of Model Training  
> **Skill Level:** Beginner-friendly  
> **Goal:** Train the baseline model and produce the 5 required files in `models/`  
> **Disclaimer:** *"Research prototype, not for clinical decision-making."*

---

## 1. What You Need to Know (The Big Picture)

Welcome! Your job is to train our **baseline machine learning model** that estimates the risk of sepsis for ICU patients hour-by-hour. 

Don't worry if you are new to ML. We have already built the feature extraction and server code. You have an automated script ready to run, but here are the **3 golden rules** you must follow to make sure our project passes jury scrutiny:

1. **Split by Patient, NEVER by Row:**
   - A single patient has dozens of hourly rows. If Patient `P001` is in the training set, **none** of `P001`'s rows can ever be in the test set. 
2. **Never Look into the Future (Causal Only):**
   - When predicting risk for Hour 3, the model can **only** see Hour 1, 2, and 3. It cannot see Hour 4 or 5. (Our provided script handles this automatically).
3. **Choose the Alert Threshold on VALIDATION Data Only:**
   - You never pick the cutoff threshold on the test set. You choose it on validation data, then evaluate on test data once.

---

## 2. The 5 Signals We Use

Our live server and the jury will send records with only **5 clinical signals**:

| Signal Name | What It Is | Normal ICU Range |
|---|---|---|
| `HR` | Heart Rate (beats per minute) | 60 – 100 bpm |
| `SBP` | Systolic Blood Pressure (mmHg) | 90 – 140 mmHg |
| `Resp` | Respiration Rate (breaths per minute) | 12 – 20 breaths/min |
| `Temp` | Core Body Temperature (Celsius) | 36.5 – 37.5 °C |
| `WBC` | White Blood Cell Count ($10^3/\mu\text{L}$) | 4.5 – 11.0 |
| `SepsisLabel` | **Target to Predict** (0 = No sepsis, 1 = Sepsis) | Binary (0 or 1) |

---

## 3. Step-by-Step Instructions

### Step 1: Download the Public Dataset
1. Visit the official PhysioNet 2019 challenge page:  
   👉 [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)
2. Download `training_setA.zip` (or `training_setB.zip`).
3. Extract it into a folder **outside** our git repository, or inside `d:\SepsisSignal\data\` (we already configured `.gitignore` so raw patient files will never accidentally be uploaded to GitHub).

### Step 2: Run the Training Script
Open your terminal (PowerShell / Command Prompt) in the repo directory `d:\SepsisSignal` and run:

```bash
# If you want a quick run on 300 patients (takes ~1-2 minutes):
python scripts/train_baseline.py --data_dir "path/to/extracted/training_setA" --max_patients 300

# Or to run on more patients (e.g., 1000 patients):
python scripts/train_baseline.py --data_dir "path/to/extracted/training_setA" --max_patients 1000
```

*Replace `"path/to/extracted/training_setA"` with the actual folder path where the `.psv` files are located.*

### Step 3: Verify the Output Files
When the script finishes, check that the following 5 files have been created in `models/`:

1. `models/model.joblib` — The trained scikit-learn Logistic Regression pipeline.
2. `models/imputer.json` — The median values computed strictly from the training cohort.
3. `models/threshold.json` — The optimal decision threshold chosen on the validation split.
4. `models/splits/` — Text files containing the patient IDs:
   - `train_patients.txt`
   - `val_patients.txt`
   - `test_patients.txt`
5. `models/results/metrics.json` — Real evaluation metrics on the held-out test cohort (AUPRC, Precision, Recall, Specificity).

### Step 4: Run the Acceptance Tests
To confirm that your newly trained model works with our live server and passes all jury checks, run:

```bash
python -m tests.run_tests
```

If it prints **`OK`** at the bottom, your job is complete! 🎉

---

## 4. Prompts You Can Use (If You Want to Ask AI for Help)

If you get stuck or want to understand things deeper, copy and paste these exact prompts into ChatGPT or Claude:

### Prompt 1: Understanding Patient-Level Splitting
```text
I am building an early warning sepsis prediction model for an ICU hackathon using the PhysioNet 2019 dataset. Each patient has multiple hourly rows in a PSV file. Can you explain why doing a standard train_test_split on rows is a severe data leakage mistake, and how a patient-level split prevents this?
```

### Prompt 2: Understanding AUPRC vs Accuracy
```text
In ICU sepsis prediction, less than 2% of the hourly rows are positive for sepsis. Why is standard classification accuracy a misleading metric here, and why are AUPRC (Area Under the Precision-Recall Curve) and alert burden preferred by clinical informatics researchers?
```

### Prompt 3: Debugging Training Errors
```text
I am running a Python script to train LogisticRegression on PhysioNet 2019 ICU data. I received this error:
[PASTE YOUR ERROR HERE]
Can you explain in simple terms what caused this error and how I can fix it?
```

---

## 5. Frequently Asked Questions (FAQ)

#### Q: Can I use XGBoost, LightGBM, or an LSTM?
> **Answer:** For the **Live Path** in Round 2, the jury rules state we must use the **Logistic Regression baseline** (and it is labeled on screen). If you have extra time, you can train XGBoost offline and record its metrics in `models/results/metrics.json` for the comparison slide, but Logistic Regression is our primary priority!

#### Q: Why are my predicted probabilities lower than expected?
> **Answer:** That is normal and expected! The live path only uses 5 vitals/labs (HR, SBP, Resp, Temp, WBC). In real hospitals, doctors also look at lab results like Lactate, Creatinine, and Platelets. We explain this honestly in our pitch: fewer signals means lower confidence, which is why transparent early warning is valuable.

#### Q: Can I change `src/features.py`?
> **Answer:** **No.** `src/features.py` is the single source of truth shared by both training and the live socket server. The training script automatically uses `src/features.py` so there is zero mismatch between offline and live performance.

---

## 6. Checklist Before Handing Off to the Team

- [ ] Ran `python scripts/train_baseline.py` without errors.
- [ ] Checked that `models/model.joblib` exists.
- [ ] Checked that `models/threshold.json` exists.
- [ ] Checked that `models/imputer.json` exists.
- [ ] Ran `python -m tests.run_tests` and all 7 tests passed.
- [ ] Sent the test set metrics (AUPRC, Recall, Precision) to the presentation teammate.
