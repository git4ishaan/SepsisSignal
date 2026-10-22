# SepsisSignal — Phased Feature Implementation Plan

> **Standard of Quality**: Zero regressions, strict causal isolation, robust null handling, and editorial design consistency.  
> **Target Audience**: Team HackerSIM (ICU Early-Warning Research Prototype).

This plan covers three optional demo features. The priority order below reflects hackathon jury value, implementation effort, and risk of unsupported clinical claims. Phase numbers below are section identifiers, not execution order. Ownership is split between Ishaan (features 1–2) and the project team (feature 3).

| Priority | Feature | Owner | Value | Recommendation |
|---|---|---|---|---|
| **P1** | Patient-stay playback (Phase 1) | Ishaan | High demo value; low implementation risk | Implement first. It replays stored hours; call it playback unless it actually inserts new observations. |
| **P2** | Model contribution display (Phase 3) | Project team | Medium–high; supports explainability | Implement after the jury path. Use exact model preprocessing and describe contributions as model behavior, not clinical causes. |
| **P3** | Research trajectory summary / PDF (Phase 2) | Ishaan | Medium–low; more effort and less jury alignment | Defer until higher-priority work is complete. Avoid presenting it as a clinical handover tool. |

## Ownership

| Phase | Feature | Owner |
|---|---|---|
| 1 | Patient-stay playback | Ishaan |
| 2 | Research trajectory summary / PDF | Ishaan |
| 3 | Model contributions | Project team |

Ownership is separate from priority. The assigned owner should follow the priority order and keep the jury evaluation path as the first acceptance gate.

## First Priority: Preserve the Jury Evaluation Path

These are acceptance requirements, not optional roadmap features. Keep them ahead of the three enhancements:

- Accept one `H|...` record and return one `R|...` result through the chosen WebSocket or raw TCP transport.
- Store each patient-hour in SQLite before inference; count empty signal fields before imputation.
- Ensure hour T uses only that patient's observations with hour `<= T`, and preserve the stored prediction for T after later hours arrive.
- Keep the existing trained model and patient-level split; show the live Logistic Regression baseline label and research-prototype notice in the dashboard.
- Keep the dashboard clock tied to the selected patient and hour, showing that hour's probability and alert.

Do not treat the demo features below as substitutes for these checks.

---

## Phase 1: Patient-Stay Playback

### 1. Objective & Value
Allow presenters and evaluators to press **Play (`▶`)** on the patient clock so the selected stay advances through its stored hours automatically (for example, one hour every 800ms). This makes deterioration and alert timing easy to follow without manual slider scrubbing. This is playback of stored data, not a telemetry stream simulator.

### 2. Architecture & Data Flow
* **File Affected**: [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx)
* **Scope**: Frontend playback state over the existing stored timeline. No backend or database changes required.
* **State Lifecycle**:
  ```
  [User clicks Play] -> setInterval(tick, 800ms) -> setCurrentHour(prev + 1)
  [Hour reaches maxHour OR User scrubs manually OR User clicks Pause] -> clearInterval() -> isPlaying = false
  ```

### 3. Step-by-Step Implementation

#### A. Add State Variables to `Dashboard.tsx`
```tsx
const [isPlaying, setIsPlaying] = useState<boolean>(false);
const playTimerRef = useRef<NodeJS.Timeout | null>(null);
```

#### B. Implement Autoplay Logic with Safety Guards
```tsx
const stopPlayback = () => {
  if (playTimerRef.current) {
    clearInterval(playTimerRef.current);
    playTimerRef.current = null;
  }
  setIsPlaying(false);
};

const togglePlayback = () => {
  if (isPlaying) {
    stopPlayback();
  } else {
    // If already at end of stay, restart from hour 1
    if (currentHour >= maxHour) {
      setCurrentHour(1);
    }
    setIsPlaying(true);
  }
};

useEffect(() => {
  if (!isPlaying) return;

  playTimerRef.current = setInterval(() => {
    setCurrentHour((prev) => {
      if (prev >= maxHour) {
        stopPlayback();
        return prev;
      }
      return prev + 1;
    });
  }, 850);

  return () => {
    if (playTimerRef.current) clearInterval(playTimerRef.current);
  };
}, [isPlaying, maxHour]);

// Automatically pause playback if the user switches patient
useEffect(() => {
  stopPlayback();
}, [selectedPatientId]);
```

#### C. Add Play/Pause Button to the Clock Scrubber
In `HourScrubber` or directly beside it in the `clock-heading` section:
```tsx
<button
  type="button"
  onClick={togglePlayback}
  className="inline-flex h-9 items-center gap-1.5 rounded-panel border border-rule px-3 text-xs font-medium text-ink hover:bg-surface focus-visible:outline-2 focus-visible:outline-rule"
  title={isPlaying ? "Pause playback" : "Autoplay patient trajectory"}
>
  {isPlaying ? (
    <>
      <span className="h-2 w-2 rounded-sm bg-alert animate-pulse" />
      Pause
    </>
  ) : (
    <>
      <span className="text-[10px]">▶</span>
      Play Stay
    </>
  )}
</button>
```

### 4. Edge Cases & Bug Prevention
* **Timer Leak**: Ensure `clearInterval` runs on component unmount and when changing patients.
* **Scrubber Conflict**: Calling `onChange` on the range slider must immediately invoke `stopPlayback()` so manual drag doesn't fight the timer.
* **Max Hour Clamping**: Ensure `prev >= maxHour` stops the timer cleanly rather than repeatedly requesting out-of-bounds hours.

### 5. Verification Checklist
- [ ] Clicking **Play** advances `currentHour` smoothly from 1 to `maxHour`.
- [ ] Reaching `maxHour` stops playback and switches button back to "Play Stay".
- [ ] Dragging the manual slider while playing immediately stops autoplay.
- [ ] Changing patient from `P001` to `P002` immediately resets playback.

---

## Phase 2: Research Trajectory Summary / PDF

### 1. Objective & Value
Provide a one-click **"Export trajectory summary"** view with print styling for the selected patient up to hour $T$. This is a research-demo summary, not a clinical handover or decision-support document.

### 2. Architecture & Data Flow
* **Files Affected**:
  - [`dashboard/app/components/TrajectorySummary.tsx`](file:///d:/SepsisSignal/dashboard/app/components/TrajectorySummary.tsx) (New component)
  - [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx) (Header button & modal toggle)
  - [`dashboard/app/globals.css`](file:///d:/SepsisSignal/dashboard/app/globals.css) (Print stylesheet rules)

### 3. Report Data Structure & Content
The report synthesizes data already present in `timeline` and `shownPrediction`:
1. **Header Block**: Patient ID, Cohort Split (Test vs. Live), Current ICU Hour $T$, Report Generated Timestamp, Research Prototype Notice.
2. **Alert & Trajectory Status**:
   - Current Risk Probability vs. Validation Threshold ($0.25$).
   - Earliest alert hour, if any, as a descriptive value; do not infer lead time to clinical deterioration.
   - Peak Risk Probability during stay.
3. **Vital Signs Anomaly & Delta Table**:
   - Vital names: HR, SBP, Resp, Temp, WBC.
   - Initial Admission Value (Hour 1).
   - Current Value (Hour $T$).
   - Delta ($\Delta = \text{Current} - \text{Initial}$).
   - Measurement Status: *Freshly Measured*, *Causally Forward-Filled*, or *Unmeasured*.
4. **Missingness Audit**: Total empty fields at hour $T$, hours since last laboratory draw (WBC).
5. **Model context**: Algorithm (`logistic_regression_baseline`), feature count ($22$), validation threshold source, and research-prototype notice. Do not include clinician sign-off fields.

### 4. Step-by-Step Implementation

#### A. Create `dashboard/app/components/TrajectorySummary.tsx`
```tsx
"use client";

import React from "react";
import { X, Printer, AlertTriangle, CheckCircle2 } from "lucide-react";

interface TrajectorySummaryProps {
  isOpen: boolean;
  onClose: () => void;
  patientId: string;
  currentHour: number;
  maxHour: number;
  split: string;
  prediction: {
    probability: number;
    alert: number;
    threshold: number;
    empty_count: number;
    model_name: string;
  } | null;
  timelinePoints: Array<{
    hour: number;
    values: {
      hr: number | null;
      sbp: number | null;
      resp: number | null;
      temp: number | null;
      wbc: number | null;
    };
    missing: string[];
    probability: number | null;
    alert: number | null;
  }>;
}

export default function TrajectorySummary({
  isOpen,
  onClose,
  patientId,
  currentHour,
  maxHour,
  split,
  prediction,
  timelinePoints,
}: TrajectorySummaryProps) {
  if (!isOpen) return null;

  // Filter timeline points up to current hour (strict causal integrity)
  const causalPoints = timelinePoints.filter((pt) => pt.hour <= currentHour);
  const firstPoint = causalPoints[0];
  const latestPoint = causalPoints[causalPoints.length - 1];

  // Compute trajectory statistics
  const alertPoints = causalPoints.filter((pt) => pt.alert === 1);
  const firstAlertHour = alertPoints.length > 0 ? alertPoints[0].hour : null;
  const peakProb = Math.max(...causalPoints.map((pt) => pt.probability ?? 0), 0);

  // Compute vitals delta
  const vitals = [
    { key: "hr", name: "Heart Rate", unit: "bpm", normal: "60-100" },
    { key: "sbp", name: "Systolic BP", unit: "mmHg", normal: "90-120" },
    { key: "resp", name: "Resp Rate", unit: "breaths/min", normal: "12-20" },
    { key: "temp", name: "Temperature", unit: "°C", normal: "36.5-37.5" },
    { key: "wbc", name: "White Blood Cells", unit: "10³/µL", normal: "4.0-11.0" },
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 print:p-0 print:bg-white">
      <div className="relative max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-panel border border-rule bg-surface p-8 shadow-xl print:max-h-none print:w-full print:border-none print:p-0 print:shadow-none">
        
        {/* Modal Action Bar (hidden when printing) */}
        <div className="mb-6 flex items-center justify-between border-b border-rule pb-4 print:hidden">
          <div className="flex items-center gap-2">
              <span className="font-label text-xs uppercase tracking-wider text-muted">Research summary</span>
            <span className="rounded bg-app px-2 py-0.5 text-xs text-muted">Hour {currentHour} of {maxHour}</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => window.print()}
              className="inline-flex items-center gap-1.5 rounded-panel border border-accent bg-accent px-3 py-1.5 text-xs font-medium text-surface hover:opacity-90"
            >
              <Printer size={14} /> Print / Save PDF
            </button>
            <button
              onClick={onClose}
              className="rounded-panel p-1.5 text-muted hover:bg-app hover:text-ink"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Printable Document Body */}
        <div className="space-y-6 text-ink">
          
          {/* Header */}
          <div className="flex items-start justify-between border-b border-rule pb-4">
            <div>
              <h1 className="text-xl font-medium tracking-tight">Patient Trajectory & Sepsis Risk Summary</h1>
              <p className="mt-1 text-xs text-muted">
                Patient: <strong className="text-ink">{patientId}</strong> ({split} cohort) · Stay Hour: <strong>{currentHour}</strong> · Generated: {new Date().toLocaleString()}
              </p>
            </div>
            <div className="text-right">
              <span className="inline-block rounded border border-rule bg-app px-2.5 py-1 text-[11px] font-medium text-muted">
                RESEARCH PROTOTYPE
              </span>
            </div>
          </div>

          {/* Sepsis Early-Warning Trajectory Overview */}
          <div className="grid grid-cols-3 gap-4 rounded-panel border border-rule bg-app/40 p-4">
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted font-label">Current Risk Score</p>
              <p className="mt-1 text-2xl font-medium text-accent">
                {prediction ? prediction.probability.toFixed(4) : "—"}
              </p>
              <p className="text-[11px] text-muted">Threshold: {prediction?.threshold.toFixed(2) ?? "0.25"}</p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted font-label">Alert Status</p>
              <div className="mt-1 flex items-center gap-1.5">
                {prediction?.alert === 1 ? (
                  <span className="inline-flex items-center gap-1 text-sm font-medium text-alert">
                    <AlertTriangle size={15} /> ACTIVE ALERT
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-sm font-medium text-muted">
                    <CheckCircle2 size={15} /> Normal Range
                  </span>
                )}
              </div>
              <p className="text-[11px] text-muted">
                {firstAlertHour ? `First alert at Hour ${firstAlertHour}` : "No alert triggered yet"}
              </p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-wider text-muted font-label">Peak Observed Risk</p>
              <p className="mt-1 text-2xl font-medium text-ink">{peakProb.toFixed(4)}</p>
              <p className="text-[11px] text-muted">Across hours 1–{currentHour}</p>
            </div>
          </div>

          {/* Vitals Anomaly & Delta Table */}
          <div>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-muted font-label">
              Physiological Trajectory & Completeness Audit
            </h3>
            <table className="w-full text-left text-xs border border-rule rounded-panel overflow-hidden">
              <thead className="bg-app text-muted font-medium border-b border-rule">
                <tr>
                  <th className="py-2 px-3">Vital Sign</th>
                  <th className="py-2 px-3">Admission (Hr 1)</th>
                  <th className="py-2 px-3">Current (Hr {currentHour})</th>
                  <th className="py-2 px-3">Net Delta</th>
                  <th className="py-2 px-3">Measurement Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule">
                {vitals.map((v) => {
                  const initVal = firstPoint?.values[v.key];
                  const currVal = latestPoint?.values[v.key];
                  const isMissing = latestPoint?.missing.includes(v.key);
                  const delta = initVal !== null && currVal !== null ? currVal - initVal : null;

                  return (
                    <tr key={v.key} className="hover:bg-app/20">
                      <td className="py-2 px-3 font-medium">
                        {v.name} <span className="text-muted font-normal">({v.unit})</span>
                      </td>
                      <td className="py-2 px-3 text-muted">{initVal !== null ? initVal : "—"}</td>
                      <td className="py-2 px-3 font-medium">{currVal !== null ? currVal : "—"}</td>
                      <td className="py-2 px-3">
                        {delta !== null ? (
                          <span className={delta > 0 ? "text-ink" : "text-muted"}>
                            {delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="py-2 px-3">
                        {isMissing ? (
                          <span className="text-missing italic">Causally forward-filled</span>
                        ) : (
                          <span className="text-accent font-medium">Measured at Hr {currentHour}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Missing Data & Model Context */}
          <div className="rounded-panel border border-rule bg-app/20 p-3.5 text-xs text-muted space-y-1.5">
            <p>
              <strong className="text-ink">Data Freshness Audit:</strong> {prediction?.empty_count ?? 0} of 5 vital signs unmeasured at Hour {currentHour}. Missing values are imputed using causal forward-fill and training-set median baselines without future lookahead.
            </p>
            <p>
              <strong className="text-ink">Inference Engine:</strong> {prediction?.model_name ?? "logistic_regression_baseline"} (22 point-in-time features). Predictions past hour {currentHour} are strictly isolated.
            </p>
          </div>

          {/* Regulatory Disclaimer */}
          <p className="text-[10px] text-center text-muted italic">
            Not for clinical decision-making. Research prototype evaluated on PhysioNet Computing in Cardiology Challenge 2019 data.
          </p>

        </div>
      </div>
    </div>
  );
}
```

#### B. Add Print Styling to `dashboard/app/globals.css`
```css
@media print {
  body * {
    visibility: hidden;
  }
  .print\:p-0,
  .print\:p-0 * {
    visibility: visible;
  }
  .print\:p-0 {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
  }
}
```

#### C. Connect "Export Report" Button in `Dashboard.tsx`
Add state:
```tsx
const [isReportOpen, setIsReportOpen] = useState(false);
```
Add button next to "Live input" in header:
```tsx
<button
  type="button"
  className="inline-flex min-h-[42px] cursor-pointer items-center gap-2 rounded-panel border border-rule bg-surface px-3.5 text-[13px] font-medium text-ink hover:bg-app"
  onClick={() => setIsReportOpen(true)}
>
  <FileText size={15} strokeWidth={2} />
  Export Report
</button>
```

### 5. Edge Cases & Bug Prevention
* **Empty Timeline Data**: If no observations exist, guard against `Math.max()` on empty arrays (defaults to 0).
* **Missing Initial Hour**: If Hour 1 has missing vitals, delta displays `—` gracefully rather than `NaN`.
* **Causal Leakage**: Filter `timelinePoints` strictly with `.filter(pt => pt.hour <= currentHour)` so future hours never leak into the shift report.

---

## Phase 3: Model Contributions ("What Influenced the Score?")

**Status: Implemented; focused backend verification passed.**

### 1. Objective & Value
Show the largest positive model contributions to the prediction at hour $T$. This helps explain how the Logistic Regression model produced its score; it does not establish clinical causation or identify validated physiological drivers.

### 2. Architecture & Data Flow
* **Files Affected**: [`src/predict.py`](file:///d:/SepsisSignal/src/predict.py), [`src/store.py`](file:///d:/SepsisSignal/src/store.py), and [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx).
* **Math Foundation**: For Logistic Regression, a feature's contribution to the model logit is:
  $$\text{contribution}_i = w_i \times x_i$$
  where $w_i$ is the trained model coefficient and $x_i$ is that feature after the model's exact serving-time preprocessing. Compute contributions in the same feature space used by `predict_proba`; do not independently refit or approximate preprocessing.

The current saved pipeline is `StandardScaler` followed by the `clf` Logistic Regression step. The implementation uses those exact steps and the canonical feature order from `src.features.FEATURE_NAMES`. For an already-cached prediction, it reconstructs the input from observations received by the time that prediction was created, so later hours cannot alter the explanation. Attribution is supplemental and must not block or change the risk estimate.

### 3. Step-by-Step Implementation

#### A. Compute Top Contributors in `src/predict.py`
```python
def explain_prediction(model, feature_vec, feature_names, top_k=3):
    """
    Computes top positive model contributions to the log-odds score.
    These are model explanations, not clinical causes.
    """
    # LogisticRegression pipeline: extract final estimator coefficients
    try:
        classifier = model.named_steps["clf"]
        scaler = model.named_steps.get("scaler", None) if hasattr(model, "named_steps") else None
        
        # If scaler exists, scale feature vector; otherwise use raw
        x = scaler.transform(feature_vec)[0] if scaler else feature_vec[0]
        coefs = classifier.coef_[0]
        
        contributions = x * coefs
        
        # Feature name mapping to neutral display labels
        label_map = {
            "hr_val": "Heart rate level",
            "hr_delta": "Heart rate acceleration",
            "sbp_val": "Systolic blood pressure",
            "sbp_delta": "Blood pressure drop",
            "resp_val": "Respiratory rate",
            "resp_delta": "Respiratory rate increase",
            "temp_val": "Temperature level",
            "temp_delta": "Temperature change",
            "wbc_val": "Leukocyte count (WBC)",
            "wbc_delta": "WBC trajectory",
            "current_empty_count": "Measurement sparsity",
        }
        
        indexed_contribs = []
        for name, contrib in zip(feature_names, contributions):
            if name in label_map and contrib > 0:
                indexed_contribs.append({
                    "feature": name,
                    "label": label_map[name],
                    "contribution": round(float(contrib), 3),
                })
        
        # Sort descending by contribution
        indexed_contribs.sort(key=lambda item: item["contribution"], reverse=True)
        return indexed_contribs[:top_k]
    except Exception:
        return []
```

#### B. Return in `/predict` and Show in UI
Add `contributors` to the `/predict` dictionary response:
```python
"contributors": get_model_contributions(feature_vec),
```
In `Dashboard.tsx`, render directly under the probability:
```tsx
{!!shownPrediction?.contributors?.length && (
  <div className="mt-4 border-t border-rule pt-3">
    <p className="font-label text-[10px] font-medium uppercase tracking-[0.07em] text-muted">
      Model contributions · log-odds
    </p>
    <ul className="mt-2 grid gap-1.5 text-xs">
      {shownPrediction.contributors.map((item) => (
        <li key={item.feature} className="flex items-center justify-between gap-4">
          <span>{item.label}</span>
          <span className="font-label tabular-nums text-muted">+{item.contribution.toFixed(3)}</span>
        </li>
      ))}
    </ul>
    <p className="mt-2 text-[10px] leading-relaxed text-muted">
      Positive values raise model log-odds; these are model contributions, not clinical causes.
    </p>
  </div>
)}
```

---

## Phase 4: qSOFA / SIRS Criteria (Dropped)

Do not include qSOFA or SIRS in the dashboard. They are not required by the jury path, the available patient-hour input cannot calculate either set completely, and presenting partial criteria adds clinical interpretation risk without explaining the model prediction. Revisit only if the team obtains complete inputs and a clear evaluation goal.

---

## Prioritized Execution & Quality Gates

| Priority | Feature | Verification focus | Main risk |
|---|---|---|---|
| **P1** | Patient-stay playback — Ishaan | Playback stops at the available hour; manual scrub and patient change stop playback. | Confusing replay with incoming live telemetry. |
| **P2** | Model contributions — Project team | Each displayed value matches its coefficient times the serving model's transformed feature value. | Mislabeling a model contribution as causation or clinical explanation. |
| **P3** | Research trajectory summary / PDF — Ishaan | Values are limited to hour T, missing values are clear, and print output retains the research disclaimer. | Clinical workflow implications and stale or misleading report data. |

Implement in priority order after the jury path is confirmed. qSOFA/SIRS criteria are dropped from the dashboard; do not add them without complete inputs and a clear, validated evaluation purpose.
