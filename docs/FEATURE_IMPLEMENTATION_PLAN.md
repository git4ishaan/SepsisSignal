# SepsisSignal — Phased Feature Implementation Plan

> **Standard of Quality**: Zero regressions, strict causal isolation, robust null handling, and editorial design consistency.  
> **Target Audience**: Team HackerSIM (ICU Early-Warning Research Prototype).

This implementation plan details the step-by-step rollout of 4 high-impact competitive features:
1. **Phase 1: Autoplay / Telemetry Stream Simulator** (UI playback control)
2. **Phase 2: Shift Handover & Trajectory Report Generator** (Interactive modal + printable PDF)
3. **Phase 3: Explainable Feature Attribution ("Why the Alert?")** (Log-odds coefficient weighting)
4. **Phase 4: Classical Clinical Baseline Benchmarking (qSOFA & SIRS)** (Standard-of-care delta)

---

## Phase 1: Autoplay / Telemetry Stream Simulator

### 1. Objective & Value
Allow presenters and evaluators to press **Play (`▶`)** on the patient clock so the ICU stay advances automatically hour-by-hour (e.g. 1 hour every 800ms). This showcases real-time deterioration and alert triggering dynamically without manual slider scrubbing.

### 2. Architecture & Data Flow
* **File Affected**: [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx)
* **Scope**: Pure frontend state management. No backend or database changes required.
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

## Phase 2: Shift Handover & Trajectory Report Generator

### 1. Objective & Value
Provide a one-click **"Export Shift Report"** modal with `@media print` styling that compiles an ICU handover summary for the selected patient up to hour $T$. Clinicians can review it on-screen or print/save it as a clean 1-page PDF.

### 2. Architecture & Data Flow
* **Files Affected**:
  - [`dashboard/app/components/ShiftReportModal.tsx`](file:///d:/SepsisSignal/dashboard/app/components/ShiftReportModal.tsx) (New component)
  - [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx) (Header button & modal toggle)
  - [`dashboard/app/globals.css`](file:///d:/SepsisSignal/dashboard/app/globals.css) (Print stylesheet rules)

### 3. Report Data Structure & Content
The report synthesizes data already present in `timeline` and `shownPrediction`:
1. **Header Block**: Patient ID, Cohort Split (Test vs. Live), Current ICU Hour $T$, Report Generated Timestamp, Research Prototype Notice.
2. **Alert & Trajectory Status**:
   - Current Risk Probability vs. Validation Threshold ($0.25$).
   - Earliest Alert Hour (if any) and Lead Time to peak deterioration.
   - Peak Risk Probability during stay.
3. **Vital Signs Anomaly & Delta Table**:
   - Vital names: HR, SBP, Resp, Temp, WBC.
   - Initial Admission Value (Hour 1).
   - Current Value (Hour $T$).
   - Delta ($\Delta = \text{Current} - \text{Initial}$).
   - Measurement Status: *Freshly Measured*, *Causally Forward-Filled*, or *Unmeasured*.
4. **Missingness Audit**: Total empty fields at hour $T$, hours since last laboratory draw (WBC).
5. **Model Card & Sign-Off**: Algorithm (`logistic_regression_baseline`), feature count ($22$), validation threshold source, blank clinician signature block for handover review.

### 4. Step-by-Step Implementation

#### A. Create `dashboard/app/components/ShiftReportModal.tsx`
```tsx
"use client";

import React from "react";
import { X, Printer, AlertTriangle, CheckCircle2 } from "lucide-react";

interface ShiftReportModalProps {
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

export default function ShiftReportModal({
  isOpen,
  onClose,
  patientId,
  currentHour,
  maxHour,
  split,
  prediction,
  timelinePoints,
}: ShiftReportModalProps) {
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
            <span className="font-label text-xs uppercase tracking-wider text-muted">Clinical Audit Tool</span>
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
              <h1 className="text-xl font-medium tracking-tight">ICU Shift Handover & Sepsis Risk Summary</h1>
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

          {/* Handover Sign-Off Block */}
          <div className="mt-8 pt-4 border-t border-rule grid grid-cols-2 gap-8 text-xs text-muted">
            <div>
              <p className="font-medium text-ink">Handing-Off Clinician / Fellow:</p>
              <div className="mt-6 border-b border-dashed border-rule" />
            </div>
            <div>
              <p className="font-medium text-ink">Receiving Clinician / Fellow:</p>
              <div className="mt-6 border-b border-dashed border-rule" />
            </div>
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

## Phase 3: Explainable Feature Attribution ("Why the Alert?")

### 1. Objective & Value
Directly break down the model's prediction at hour $T$ into its top 3 physiological drivers (e.g. `▲ HR acceleration`, `▼ SBP drop`, `▲ Temp elevation`), ending the "black-box" complaint common to Epic ESM.

### 2. Architecture & Data Flow
* **File Affected**: [`src/predict.py`](file:///d:/SepsisSignal/src/predict.py) & [`dashboard/app/components/Dashboard.tsx`](file:///d:/SepsisSignal/dashboard/app/components/Dashboard.tsx)
* **Math Foundation**: For Logistic Regression, log-odds contribution is:
  $$\text{contribution}_i = w_i \times x_i$$
  where $w_i$ is the trained model coefficient for feature $i$ and $x_i$ is the normalized feature value.

### 3. Step-by-Step Implementation

#### A. Compute Top Contributors in `src/predict.py`
```python
def explain_prediction(model, feature_vec, feature_names, top_k=3):
    """
    Computes top positive contributors to the log-odds score.
    Returns human-readable physiological explanations.
    """
    # LogisticRegression pipeline: extract final estimator coefficients
    try:
        classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
        scaler = model.named_steps.get("scaler", None) if hasattr(model, "named_steps") else None
        
        # If scaler exists, scale feature vector; otherwise use raw
        x = scaler.transform(feature_vec)[0] if scaler else feature_vec[0]
        coefs = classifier.coef_[0]
        
        contributions = x * coefs
        
        # Feature name mapping to clinical labels
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
        for name, val, contrib in zip(feature_names, feature_vec[0], contributions):
            if name in label_map and contrib > 0.05:  # Only positive risk drivers
                indexed_contribs.append({
                    "feature": name,
                    "label": label_map[name],
                    "contribution": round(float(contrib), 3),
                    "value": round(float(val), 2),
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
"contributors": explain_prediction(model, feature_vec, get_feature_names()),
```
In `Dashboard.tsx`, render directly under the probability:
```tsx
{shownPrediction?.contributors && shownPrediction.contributors.length > 0 && (
  <div className="mt-3.5 space-y-1.5 border-t border-rule pt-3">
    <p className="font-label text-[10px] font-medium uppercase tracking-wider text-muted">
      Key Contributing Drivers
    </p>
    <div className="flex flex-wrap gap-1.5">
      {shownPrediction.contributors.map((c) => (
        <span
          key={c.feature}
          className="inline-flex items-center gap-1 rounded bg-app px-2 py-0.5 text-xs text-ink"
        >
          <span className="text-alert font-bold">▲</span> {c.label} (+{c.contribution})
        </span>
      ))}
    </div>
  </div>
)}
```

---

## Phase 4: Classical Clinical Baseline Comparison (qSOFA & SIRS)

### 1. Objective & Value
Show a side-by-side indicator of classical standard-of-care criteria (**qSOFA** and **SIRS**) at hour $T$ to demonstrate the lead-time advantage of SepsisSignal (flagging risk 3–5 hours before classical bedside scores turn positive).

### 2. Clinical Rules Definition
1. **qSOFA** (Quick Sequential Organ Failure Assessment):
   - Resp Rate $\ge 22$ breaths/min (+1)
   - Systolic BP $\le 100$ mmHg (+1)
   - *(Score $\ge 2$ indicates high risk of poor ICU outcome)*.
2. **SIRS** (Systemic Inflammatory Response Syndrome):
   - Heart Rate $> 90$ bpm (+1)
   - Temperature $> 38.0^\circ\text{C}$ or $< 36.0^\circ\text{C}$ (+1)
   - Resp Rate $> 20$ breaths/min (+1)
   - WBC $> 12.0$ or $< 4.0\times 10^3/\mu\text{L}$ (+1)
   - *(Score $\ge 2$ meets SIRS criteria)*.

### 3. Implementation in `Dashboard.tsx`
```tsx
function computeClinicalBaselines(values: PointValues) {
  let qsofa = 0;
  let qsofaTested = 0;
  if (values.resp !== null) { qsofaTested++; if (values.resp >= 22) qsofa++; }
  if (values.sbp !== null) { qsofaTested++; if (values.sbp <= 100) qsofa++; }

  let sirs = 0;
  let sirsTested = 0;
  if (values.hr !== null) { sirsTested++; if (values.hr > 90) sirs++; }
  if (values.temp !== null) { sirsTested++; if (values.temp > 38.0 || values.temp < 36.0) sirs++; }
  if (values.resp !== null) { sirsTested++; if (values.resp > 20) sirs++; }
  if (values.wbc !== null) { sirsTested++; if (values.wbc > 12.0 || values.wbc < 4.0) sirs++; }

  return {
    qsofa: { score: qsofa, max: qsofaTested, positive: qsofa >= 2 },
    sirs: { score: sirs, max: sirsTested, positive: sirs >= 2 },
  };
}
```

Render alongside threshold:
```tsx
<div className="mt-3 flex items-center gap-3 text-xs">
  <span className="text-muted">Standard criteria:</span>
  <span className={`rounded px-1.5 py-0.5 text-[11px] ${qsofa.positive ? 'bg-alert/15 text-alert font-medium' : 'bg-app text-muted'}`}>
    qSOFA: {qsofa.score}/{qsofa.max} {qsofa.positive ? '(Pos)' : '(Neg)'}
  </span>
  <span className={`rounded px-1.5 py-0.5 text-[11px] ${sirs.positive ? 'bg-alert/15 text-alert font-medium' : 'bg-app text-muted'}`}>
    SIRS: {sirs.score}/{sirs.max} {sirs.positive ? '(Pos)' : '(Neg)'}
  </span>
</div>
```

---

## Phased Execution & Quality Gate Sequence

| Phase | Feature | Estimated Time | Regression Risk | Automated Test Command |
|---|---|---|---|---|
| **Phase 1** | Autoplay Stream Simulator | 5–10 min | Very Low | `bun test` or UI scrubber verify |
| **Phase 2** | Shift Handover Report Modal | 15–20 min | Very Low | Test print dialog & modal toggle |
| **Phase 3** | Explainable Attribution | 10–15 min | Low | `python -m tests.run_tests` |
| **Phase 4** | Classical Rules (qSOFA/SIRS) | 10 min | Zero | Verify scoring logic on P001/P002 |

Follow this document sequentially: execute Phase 1 and 2 first for immediate demo readiness, followed by Phase 3 and 4 to seal the clinical explainability advantage.
