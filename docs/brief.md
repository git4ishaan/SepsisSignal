# SepsisSignal — Explainable ICU Early-Warning Research Prototype

> **Project type:** Healthcare ML / clinical decision-support research prototype  
> **Hackathon constraints:** 3 advanced developers · 6-hour live build · PPT-first selection round  
> **Primary objective:** Build a demonstrable, reproducible ML pipeline that estimates sepsis risk from historical ICU time-series data and explains its uncertainty and limitations.

## 1. Executive summary

SepsisSignal is a research prototype for exploring early-warning signals of sepsis using longitudinal intensive-care data. It processes hourly vital signs, laboratory measurements, patient/context variables, and missingness indicators to estimate a model score at each time step. A dashboard visualizes how that score changes over time, shows missing observations, and lets the team inspect the trade-off between sensitivity and false alarms at different alert thresholds.

The project is intentionally **not** presented as a clinically validated sepsis detector, medical device, or replacement for clinician judgment. It is a retrospective ML demonstration using a public dataset. The purpose is to show that the team can build a leakage-resistant time-series pipeline, evaluate a model honestly, and explain the risks of translating an experimental model into clinical practice.

### One-line pitch

**SepsisSignal turns historical ICU time-series into an explainable early-warning research dashboard, making model performance, uncertainty, missing data, and alert trade-offs visible.**

### Problem statement

Sepsis can deteriorate rapidly, while ICU observations are incomplete and irregularly collected. Building a useful early-warning model requires more than fitting a classifier: data leakage, missing measurements, class imbalance, temporal validation, false alarms, and clinical workflow must all be handled carefully.

### Proposed solution

Build a pipeline that:

1. Loads publicly available patient time-series.
2. Preprocesses missing and irregularly measured variables without using future information.
3. Trains a transparent baseline model and, if time allows, an LSTM/GRU sequence model.
4. Evaluates predictions on held-out patients and a predefined temporal prediction target.
5. Displays a patient timeline, risk estimates, missingness, threshold behavior, and model explanations.

---

## 2. Who is it for, and what problem does it address?

### Intended audience

- Hackathon judges evaluating ML depth and implementation quality.
- Students and researchers exploring early-warning systems.
- Clinical informatics teams interested in how a research model might be evaluated.

### Not intended for

- Direct bedside use.
- Diagnosis or treatment decisions.
- Automated triage or denial of care.
- Deployment on live patient data without clinical, privacy, security, prospective-validation, and regulatory review.

### Why this is technically meaningful

A convincing project needs to demonstrate understanding of five central challenges:

1. **Time dependence:** risk can evolve, so the model must use the history available at a prediction time.
2. **Missingness:** many clinical variables are not measured every hour. A missing lab result is not equivalent to a normal value.
3. **Class imbalance:** positive sepsis labels are much less common than negative labels at the hourly-row level.
4. **Leakage risk:** future observations or information created after the prediction time can make evaluation unrealistically optimistic.
5. **Alert fatigue:** high sensitivity can create a large number of false alarms. A model score alone is not enough to demonstrate an operationally useful system.

---

## 3. Data source

### Recommended public benchmark

**PhysioNet / Computing in Cardiology Challenge 2019: Early Prediction of Sepsis from Clinical Data**

- Official dataset page: <https://physionet.org/content/challenge-2019/1.0.0/>
- Official paper DOI: <https://doi.org/10.1097/CCM.0000000000004145>
- Dataset DOI: <https://doi.org/10.13026/v64v-d857>
- License listed by PhysioNet: Creative Commons Attribution 4.0 International (CC BY 4.0); read the dataset license and attribution requirements before redistributing any data.

The challenge dataset contains records for **40,336 subjects** in two training subsets. Each subject has a separate time-series file with hourly rows. Variables include vital signs, laboratory measurements, demographic/context fields, hospital admission time, ICU length of stay, and a `SepsisLabel` field. A `NaN` indicates that a variable was not measured or was not available at that time step.

Refer to the official source for the exact column definitions, labeling protocol, download instructions, license, and citation requirements. Do not assume the public benchmark's population, equipment, measurement practices, or label definitions represent every hospital.

### Data-use checklist

- Preserve the official dataset citation and license notice.
- Keep source files unchanged; write processed data to a separate directory.
- Do not upload the dataset or patient records to third-party AI services.
- Do not display patient identifiers in the public-facing demo. Use randomized demo IDs.
- Document every filtering, feature-generation, split, and label-generation choice.

### Data scope for the hackathon

Start with a subset of patients only if runtime or laptop memory is a constraint. Keep patient-level separation between training, validation, and test sets. The subset and sampling procedure must be documented; do not select only easy examples.

---

## 4. Prediction target and label strategy

The project must specify precisely what “predict sepsis” means. Do not casually claim that the model predicts sepsis before any clinical symptoms or guarantees a fixed early-warning interval.

### Recommended prototype target

Use the challenge's `SepsisLabel` and published labeling protocol as the starting point. The challenge was designed around early prediction, with positive labels associated with a defined window before the reference onset. Confirm the precise semantics in the official challenge documentation and implement them exactly.

For a separate experiment, the team may define a prediction horizon such as “identify positive labels at least *h* hours before a chosen reference event.” If doing so:

1. Define the reference event and horizon before training.
2. Exclude information recorded after the prediction time.
3. Document which hours are eligible for scoring.
4. Avoid changing the target after looking at test results.

### Key distinction

Performance on a retrospective benchmark does **not** prove that the model will recognize sepsis before clinical symptoms appear in real practice. It only establishes performance against the dataset's label definition under the chosen evaluation design.

---

## 5. Machine-learning plan

### Model strategy: baseline first, deep learning second

Do not start with an LSTM as the only model. A strong baseline gives the team a working result and provides a fair comparison for sequence modeling.

#### Model A — engineered-feature baseline

Recommended model: **Logistic Regression** as a simple baseline, followed by **XGBoost or LightGBM** if the dependency and runtime are manageable.

Create time-aware features from data available at each prediction time, such as:

- Latest observed values.
- Recent rolling mean, minimum, maximum, and variability.
- Trend or change over a recent window.
- Time since the last measurement for each variable.
- Missingness indicator flags.
- Time since admission / ICU time, where appropriate and permitted by the benchmark definition.

The model predicts a score for each eligible time step. Avoid features that encode the target or use future rows.

#### Model B — LSTM / GRU sequence model (stretch goal)

Build a sequence window from hourly observations, for example the most recent 6–24 hours available at the prediction point. Choose the window based on feasible compute time and document it.

Possible input channels:

- Imputed or normalized measurements.
- A separate mask that indicates which values were measured.
- Time-since-last-observation features.

Use padding/masking consistently for varying sequence lengths. Train only on training patients; select architecture and hyperparameters on validation data. If the recurrent model is not better, report that honestly and use the baseline for the demo.

#### Optional Model C — Temporal Convolutional Network (TCN)

A TCN is an alternative to LSTM/GRU for sequence modeling. It is a stretch goal, not a required part of a six-hour build. Choose only one deep-learning path; implementing two sequence architectures is less valuable than correct validation and a reliable demo.

### Recommended technical stack

- Python 3.11+ (use a compatible environment based on selected libraries).
- pandas and NumPy for data handling.
- scikit-learn for preprocessing, baseline models, and metrics.
- XGBoost or LightGBM, optional.
- PyTorch for the optional LSTM/GRU.
- FastAPI for model inference API.
- Next.js + React + TypeScript for the dashboard.
- Plotly, Recharts, or another plotting library for time-series visualization.
- Supabase/PostgreSQL only if persistent user accounts or saved cases are needed; a local research demo can use files and seeded examples.

---

## 6. Data preprocessing and leakage controls

### Missing-data strategy

For the hackathon baseline, use a simple and auditable policy:

1. Sort each patient's observations by time.
2. Forward-fill only from measurements available at or before the current time step, within that patient.
3. Add a missingness indicator for each important variable.
4. Add “time since last observed” features where feasible.
5. Use fallback imputation values estimated from the training set only when no previous observation is available.
6. Fit scalers/imputers on training patients only, then apply the saved transformations to validation and test patients.

Do **not** use backward filling or interpolation that draws on future measurements for an online early-warning simulation. MICE or more sophisticated imputation can be studied later, but it is unnecessary for the first prototype and can introduce complexity or leakage if mishandled.

### Imbalanced labels

- Start with class weights or an appropriate weighted loss.
- For neural networks, weighted binary cross-entropy is a reasonable starting point; focal loss is an optional experiment.
- Do not rely on raw accuracy: a model can score high accuracy by mostly predicting the majority class.
- Any oversampling or augmentation must be performed within the training set only. Never oversample validation/test patients or create duplicates across splits.

### Leakage checks

- Split by **patient**, not by individual hourly rows.
- Ensure the same patient cannot appear in both train and test.
- Use only variables available by the prediction time.
- Do not use future observations to fill missing values.
- Do not tune thresholds on the test set.
- Fit feature transformations on training data only.
- Keep a time-ordered view of each patient when creating windows.
- Check whether any variables or labels encode the target or event timing indirectly.
- Keep a reproducible random seed and write down the split logic.

### Recommended split

Create a train/validation/test split at the patient level, such as 70% / 15% / 15%, stratifying by patient-level sepsis presence if feasible. The exact proportions are less important than patient independence and a clearly documented selection process. If the project attempts to mimic deployment over time, consider a temporal/site-based evaluation when the data permits it.

---

## 7. Evaluation plan

Evaluate both classification quality and the practical burden of alerts.

### Core metrics

- **AUPRC / average precision:** useful when positive labels are uncommon.
- **Precision:** proportion of alerts that correspond to positive labels under the evaluation setup.
- **Recall / sensitivity:** proportion of positive labels detected.
- **Specificity:** proportion of negative labels correctly not alerted on.
- **Confusion matrix:** inspect false positives and false negatives.
- **Calibration:** check whether risk scores correspond to observed frequencies; use a calibration plot or Brier score if feasible.

### Temporal and operational metrics

Where the label design allows it, report:

- Lead time relative to the benchmark's reference onset definition.
- Number of alert events per patient or patient-day.
- Number of repeated alerts for the same patient episode.
- Time from first qualifying alert to the reference event.
- Sensitivity at a selected alert burden / threshold.

Be explicit that hourly-row precision is not necessarily the same as episode-level alert precision. Define whether consecutive positive hours are consolidated into a single alert episode.

### Threshold selection

1. Train models using the training set.
2. Compare candidates on validation data.
3. Choose a threshold based on an explicitly stated trade-off between sensitivity and false-alert burden.
4. Freeze the threshold and report final metrics on held-out test patients.
5. Never present a threshold as clinically validated merely because it works on the benchmark.

Do not invent performance results. All displayed numbers in the PPT or dashboard must come from your actual run, with the split, target, threshold, and metrics stated clearly.

---

## 8. Product and dashboard design

### Main views

#### A. Patient timeline

- De-identified demonstration patient ID.
- Hour-by-hour vital and laboratory trends.
- Distinct visual markers for missing measurements.
- Model score over time.
- Reference label/onset marker for retrospective evaluation only; do not show it as available in a prospective setting.

#### B. Early-warning view

- Latest model score and its time point.
- Threshold line selected on validation data.
- Explanation that the value is a model estimate, not a diagnosis.
- Model inputs contributing to the score, where the chosen model supports a defensible explanation.
- Missing-data status and uncertainty caveat.

#### C. Evaluation page

- Precision-recall curve.
- Confusion matrix at the selected threshold.
- Per-class precision/recall.
- A summary of alert burden and lead time, if implemented correctly.
- Model comparison table: baseline vs. LSTM/GRU.

#### D. Timeline replay (best live demo)

Let the presenter advance through one held-out patient record hour by hour. The dashboard should show only information available through that hour and update the model score using the trained model. Add a clear retrospective marker for the benchmark label so judges can compare model behavior without confusing retrospective labels with live inputs.

### Visual design guidance

- Use a calm clinical dashboard layout with accessible typography.
- Avoid dramatic flashing warnings or alarm colors that could imply a validated clinical alarm.
- Distinguish “measured,” “missing,” “imputed,” and “model-estimated” values.
- Add a visible banner: **Research prototype — not for clinical decision-making.**
- Use synthetic data for any scripted demonstration that is not directly from the public benchmark, and label it as synthetic.

---

## 9. Architecture

```text
PhysioNet 2019 files
        |
        v
Python data loader and patient-level split
        |
        v
Leakage-safe preprocessing
  - causal forward fill
  - masks / missingness indicators
  - time-since-last-measurement
  - training-only transformations
        |
        +-------------------------+
        |                         |
        v                         v
Engineered-feature baseline    Optional sequence pipeline
(Logistic Regression /         (LSTM or GRU)
 XGBoost / LightGBM)                 |
        |                         |
        +------------+------------+
                     v
             Held-out evaluation
  - AUPRC, precision, recall, calibration
  - threshold analysis, alert burden, lead time
                     |
                     v
          FastAPI inference / evaluation API
                     |
                     v
         Next.js clinical research dashboard
    - patient timeline / risk score / missingness
    - threshold and model performance / limitations
```

### Example repository structure

```text
sepsissignal/
├── app/
│   ├── page.tsx
│   ├── components/
│   └── api-client.ts
├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── requirements.txt
├── ml/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── build_features.py
│   ├── train_baseline.py
│   ├── train_sequence_model.py
│   ├── evaluate.py
│   └── inference.py
├── models/
│   └── README.md
├── tests/
│   ├── test_splits.py
│   ├── test_preprocessing.py
│   └── test_inference.py
├── docs/
│   ├── problem-statement.md
│   ├── experiment-log.md
│   └── limitations.md
├── README.md
└── .gitignore
```

Do not commit the raw dataset, secrets, patient-specific files, or large model weights to the repository unless the storage and license terms allow it. Document how to obtain the dataset and reproduce the experiment.

---

## 10. Six-hour hackathon execution plan

### 00:00–00:30 — Scope and data inspection

- Confirm the target/label semantics from official documentation.
- Check Python environment and load a small number of files.
- Agree on schema, split strategy, and dashboard data format.
- Choose the baseline model and minimum demo workflow.

**Deliverable:** data loader runs; output schema and target are defined.

### 00:30–02:00 — Baseline pipeline

- Load patient files and create patient-level splits.
- Implement causal imputation and missingness features.
- Train Logistic Regression or XGBoost.
- Produce initial test/validation metrics using the planned split.

**Deliverable:** a reproducible baseline, saved model, and real metrics.

### 02:00–03:30 — Dashboard and optional sequence experiment

- Developer 1 builds the dashboard and timeline replay with agreed JSON schema.
- Developer 2 builds a small LSTM/GRU experiment only if data preprocessing works.
- Developer 3 builds model inference/API integration and starts end-to-end testing.

**Deliverable:** a dashboard connected to baseline predictions; deep model remains optional.

### 03:30–04:30 — Integration and model evaluation

- Connect saved model output to the timeline.
- Compare the baseline and sequence model on the same held-out patient split.
- Choose a threshold on validation data.
- Add a precision-recall chart and confusion matrix.

**Deliverable:** end-to-end demo with evaluation evidence.

### 04:30–05:15 — Reliability pass

- Test missing values, shorter sequences, and inference failures.
- Confirm no future rows are used in patient replay.
- Check the no-data and invalid-input states.
- Fix integration bugs; remove optional features if needed.

**Deliverable:** repeatable demo flow.

### 05:15–06:00 — Freeze and rehearse

- Freeze features and preserve model/config/metrics.
- Rehearse the full demonstration twice.
- Prepare screenshots and a short backup recording.
- Ensure all teammates can explain the problem, target, split, preprocessing, evaluation, and limitations.

**Deliverable:** a working demo and defensible technical presentation.

### Scope rule

If the baseline is not working by roughly the first two hours, drop the deep model. A reliable baseline plus an honest evaluation is stronger than a nonfunctional LSTM with impressive claims.

---

## 11. Team division (3 developers)

### Developer 1 — Data and baseline ML

- Inspect raw files and feature schema.
- Write patient-level split logic.
- Implement preprocessing, baseline features, training, and metrics.
- Save the trained pipeline and evaluation outputs.

**Acceptance test:** the training script runs from documented instructions and reproduces saved metrics for a fixed split/seed.

### Developer 2 — Sequence model and experiments

- Build fixed-length or padded sequence windows.
- Implement an LSTM/GRU only after agreeing on target and split.
- Handle masks and class weighting.
- Compare against the baseline on identical test patients.

**Acceptance test:** sequence model outputs one score for the specified prediction time and uses no future observations.

### Developer 3 — Dashboard, API, and integration

- Build timeline, score plot, missingness indicators, model-comparison view, and disclaimer.
- Create the API schema and connect inference results.
- Build patient timeline replay.
- Test end-to-end operation and deploy if time permits.

**Acceptance test:** the user can select a de-identified record, replay hours, and see predictions and charts derived from the saved model.

### Shared responsibilities

All team members should review the split strategy, understand the challenge's label definition, and be able to explain why the product is not clinically validated. Keep the shared interface stable; avoid three separate AI-generated codebases that are connected only at the end.

---

## 12. PPT-first pitch outline (6 slides)

### Slide 1 — SepsisSignal
**Title:** Explainable Early-Warning Research for ICU Time-Series  
**Subtitle:** A leakage-aware ML prototype for studying early sepsis risk estimation.

Visual: patient timeline with vital trends, missing observations, and a model score.

### Slide 2 — The problem
- ICU data is longitudinal and incomplete.
- Early-warning work must balance timely detection with false alerts.
- Missing-data handling and leakage can make offline results misleading.

Avoid unreferenced global mortality or “hours saved” claims.

### Slide 3 — Our approach
- Public ICU dataset.
- Leakage-safe preprocessing and missingness indicators.
- Baseline model plus optional LSTM/GRU.
- Validation-based threshold and held-out patient evaluation.

Visual: data → preprocessing → model → evaluation → timeline.

### Slide 4 — Product demo
- Replay a held-out patient record hour by hour.
- Show actual model scores and missing data.
- Display threshold, precision-recall chart, and confusion matrix.

Clearly distinguish demo data, benchmark labels, and model estimates.

### Slide 5 — Technical contribution
- Patient-level split.
- Time-aware feature engineering and causal imputation.
- Imbalanced-data strategy.
- Alert threshold and operational metrics.
- Explainable model comparison.

### Slide 6 — Results, limits, and next steps
- Report actual metrics from the held-out test set.
- State data limitations and external-validity concerns.
- Future work: external validation, prospective evaluation, workflow study, privacy/security review, and clinical oversight.

Do not claim a clinical outcome or production readiness from a six-hour prototype.

---

## 13. 45-second opening pitch

> Sepsis can deteriorate rapidly, but ICU measurements are incomplete, irregular, and difficult to interpret over time. A useful early-warning model must do more than output a risk score: it must handle missing data, avoid leakage, and balance missed cases against false alarms.
>
> We're building SepsisSignal, an explainable research dashboard trained on public ICU time-series data. It compares a strong ML baseline with an optional sequence model, replays patient timelines, and makes the evaluation trade-offs visible.
>
> Our aim is not to claim a clinically ready sepsis alarm. It's to demonstrate a rigorous, reproducible pathway from real-world clinical data to a transparent model evaluation and an interface that makes model behavior easier to inspect.

---

## 14. Questions judges may ask

### “Why use an LSTM if XGBoost performs better?”

The sequence model is an experiment, not a requirement. We compare models using the same patient-level splits and the same target. If a baseline performs better, we report that honestly. The model choice should follow evidence rather than assumption.

### “How do you prevent data leakage?”

We split at the patient level, fit transformations on the training set only, use only observations available at each prediction time, prohibit backward fill from future values, and select thresholds on validation data rather than the test set.

### “How do you handle missing labs?”

We use causal forward fill when a prior value exists, explicit missingness indicators, and time-since-last-measurement features. Training-set fallback values are used when needed. We do not assume an unmeasured lab is normal.

### “How do you handle class imbalance?”

We compare class-weighted training against an unweighted baseline and evaluate with AUPRC, precision, recall, and the confusion matrix rather than accuracy alone. For the sequence model, weighted binary cross-entropy is the first approach; focal loss is optional.

### “How do you avoid alert fatigue?”

We examine precision/recall trade-offs and alert burden on validation patients, consolidate consecutive flagged hours where appropriate, and report lead time. We do not claim any threshold is clinically validated.

### “Does this detect sepsis before symptoms?”

Not proven. We evaluate against a public benchmark's reference labels. That does not demonstrate prediction before symptoms in a real hospital or establish clinical utility.

### “Can the model be used in an ICU?”

No. The hackathon result is retrospective research only. Clinical use would require external and prospective validation, local calibration, workflow evaluation, safety monitoring, appropriate regulatory review, and clinician oversight.

---

## 15. Safety, privacy, and responsible claims

- Label the application and every demo page **“Research prototype — not for clinical decision-making.”**
- Do not use the model to diagnose or rule out sepsis, recommend treatment, or determine access to care.
- Do not display fake probabilities or invented performance results.
- Do not claim the model predicts before symptoms unless this has been specifically defined and validated against a suitable reference.
- Explain that the public dataset may not generalize to a new hospital, patient population, device, or clinical workflow.
- Do not submit patient data to external LLM or image services.
- Keep secrets server-side and avoid collecting identifiable information.
- Use de-identified IDs for the demo.
- Clearly distinguish retrospective ground-truth labels from information that would be available prospectively.
- Treat explanations as associations in model behavior, not clinical causation.

### Medical context

Sepsis is a clinical condition requiring professional assessment. The Surviving Sepsis Campaign guidelines emphasize clinical judgment and the limitations of relying on a single score, biomarker, or screening tool. This project should be positioned as a research experiment and not a clinical recommendation.

---

## 16. Longer-term roadmap

### Phase 1 — Hackathon prototype

- Benchmark dataset, baseline model, optional LSTM/GRU.
- Retrospective timeline replay.
- Honest metrics and visible limitations.

### Phase 2 — Reproducible evaluation

- Repeat experiments across multiple patient-level splits or temporal/site-based splits if possible.
- Add calibration assessment and event-level alert metrics.
- Record model/data versions and configuration.
- Perform subgroup error analysis when the dataset supports it responsibly.

### Phase 3 — Research-grade workflow

- Clinician and clinical-informatics feedback.
- External validation on an appropriately governed dataset.
- Prospective silent-mode evaluation before any clinical action.
- Formal privacy, security, regulatory, and risk-management review.

A real clinical system would also require dependable data integration, monitoring for data/model drift, workflow integration, and processes for investigating false alarms and missed cases. These are future work, not features delivered by the six-hour prototype.

---

## 17. Definition of done

Before presenting, confirm all of the following:

- [ ] Official dataset citation and license requirements are recorded.
- [ ] The prediction target and label semantics are documented.
- [ ] Train/validation/test splits are patient-independent.
- [ ] No future observation is used in preprocessing or replay.
- [ ] The trained baseline and metrics are saved reproducibly.
- [ ] The optional sequence model is compared on the same held-out patients or is omitted.
- [ ] Threshold selection uses validation data only.
- [ ] Dashboard values are produced by the saved model, not hardcoded.
- [ ] The dashboard visibly distinguishes missing and imputed values.
- [ ] The test metrics, threshold, and split procedure are stated in the PPT.
- [ ] A no-data/error path is tested.
- [ ] The “research only” disclaimer is visible.
- [ ] All three developers can explain the model, data pipeline, evaluation, and limitations.

---

## 18. Sources and references

1. **PhysioNet / Computing in Cardiology Challenge 2019 — official dataset page.** Data description, file format, label column, access/license details, and citation instructions.  
   <https://physionet.org/content/challenge-2019/1.0.0/>

2. **Reyna MA, Josef C, Jeter R, et al.** Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. *Critical Care Medicine*. 2020;48(2):210–217.  
   DOI: <https://doi.org/10.1097/CCM.0000000000004145>

3. **PhysioNet dataset DOI.** <https://doi.org/10.13026/v64v-d857>

4. **Surviving Sepsis Campaign — Guidelines.** Use the most recent official guidelines when contextualizing any potential clinical workflow, and do not treat this student prototype as a guideline or clinical decision rule.  
   <https://www.sccm.org/SurvivingSepsisCampaign/Guidelines>

### Suggested dataset citation

Reyna M, Josef C, Jeter R, Shashikumar S, Moody B, Westover MB, Sharma A, Nemati S, Clifford GD. *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019* (version 1.0.0). PhysioNet; 2019. DOI: <https://doi.org/10.13026/v64v-d857>.

---

## Final recommendation

Build **SepsisSignal** as a rigorously evaluated research prototype rather than a claimed clinical sepsis alarm. Start with a reproducible baseline, get the patient-level split and leakage controls right, display real evaluation results, then add a small sequence model if time allows. A transparent model with honest limitations is more defensible—and more technically impressive—than an unvalidated LSTM wrapped in a polished dashboard.
