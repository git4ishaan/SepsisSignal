"use client";

import React from "react";
import { X, Printer, AlertTriangle, CheckCircle2 } from "lucide-react";

export interface PointValues {
  hr: number | null;
  sbp: number | null;
  resp: number | null;
  temp: number | null;
  wbc: number | null;
}

export interface TimelinePoint {
  hour: number;
  values: PointValues;
  missing: string[];
  probability: number | null;
  alert: number | null;
}

export interface ShiftReportModalProps {
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
  timelinePoints: TimelinePoint[];
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

  // Strictly isolate data up to the selected current hour (causal guarantee)
  const causalPoints = timelinePoints.filter((pt) => pt.hour <= currentHour);
  const latestPoint = causalPoints[causalPoints.length - 1];

  // Trajectory analytics
  const alertPoints = causalPoints.filter((pt) => pt.alert === 1);
  const firstAlertHour = alertPoints.length > 0 ? alertPoints[0].hour : null;
  const peakProb = causalPoints.reduce(
    (max, pt) => (pt.probability !== null && pt.probability > max ? pt.probability : max),
    prediction ? prediction.probability : 0
  );

  const vitals = [
    { key: "hr" as const, name: "Heart Rate", unit: "bpm", normal: "60–100" },
    { key: "sbp" as const, name: "Systolic BP", unit: "mmHg", normal: "90–140" },
    { key: "resp" as const, name: "Respiration", unit: "br/min", normal: "12–20" },
    { key: "temp" as const, name: "Temperature", unit: "°C", normal: "36.5–37.5" },
    { key: "wbc" as const, name: "White Blood Cells", unit: "10³/µL", normal: "4.5–11.0" },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 print:p-0 print:bg-white"
      role="dialog"
      aria-modal="true"
      aria-labelledby="shift-report-title"
    >
      <div
        id="printable-shift-report"
        className="relative max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-panel border border-rule bg-surface p-7 shadow-xl"
      >
        {/* Header Action Bar (excluded from printed copy) */}
        <div className="mb-6 flex items-center justify-between border-b border-rule pb-4 print-hidden">
          <div className="flex items-center gap-2">
            <span className="font-label text-xs uppercase tracking-wider text-muted">Clinical Audit Tool</span>
            <span className="rounded bg-app px-2 py-0.5 text-xs font-medium text-ink">
              Hour {currentHour} of {maxHour}
            </span>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => window.print()}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-panel border border-accent bg-accent px-3 py-1.5 text-xs font-medium text-surface transition-opacity hover:opacity-90"
            >
              <Printer size={14} /> Print / Save PDF
            </button>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-panel text-muted transition-colors hover:bg-app hover:text-ink"
              aria-label="Close handover report"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Printable Document Body */}
        <div className="space-y-6 text-ink">
          {/* Document Header */}
          <div className="flex items-start justify-between border-b border-rule pb-4">
            <div>
              <h1 id="shift-report-title" className="text-xl font-semibold tracking-tight text-ink">
                ICU Shift Handover & Sepsis Risk Summary
              </h1>
              <p className="mt-1 text-xs text-muted">
                Patient: <strong className="text-ink">{patientId || "—"}</strong> ({split} cohort) · Stay Hour:{" "}
                <strong className="text-ink">{currentHour}</strong> · Generated: {new Date().toLocaleString()}
              </p>
            </div>
            <div className="text-right">
              <span className="inline-block rounded border border-rule bg-app px-2.5 py-1 font-label text-[10.5px] font-medium uppercase tracking-[0.05em] text-muted">
                Research Prototype
              </span>
            </div>
          </div>

          {/* Sepsis Early-Warning Trajectory Overview */}
          <div className="grid grid-cols-3 gap-4 rounded-panel border border-rule bg-app/50 p-4">
            <div>
              <p className="font-label text-[11px] uppercase tracking-wider text-muted">Current Risk Score</p>
              <p className="mt-1 font-label text-2xl font-medium text-accent">
                {prediction ? prediction.probability.toFixed(4) : "—"}
              </p>
              <p className="mt-0.5 text-[11px] text-muted">
                Threshold: {prediction?.threshold.toFixed(2) ?? "0.25"}
              </p>
            </div>

            <div>
              <p className="font-label text-[11px] uppercase tracking-wider text-muted">Alert Status</p>
              <div className="mt-1 flex items-center gap-1.5">
                {prediction?.alert === 1 ? (
                  <span className="inline-flex items-center gap-1 font-label text-xs font-semibold text-alert">
                    <AlertTriangle size={15} /> ACTIVE ALERT
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 font-label text-xs font-medium text-accent">
                    <CheckCircle2 size={15} /> Normal Range
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-[11px] text-muted">
                {firstAlertHour !== null
                  ? `First triggered at Hour ${firstAlertHour}`
                  : "No alert triggered yet"}
              </p>
            </div>

            <div>
              <p className="font-label text-[11px] uppercase tracking-wider text-muted">Peak Observed Risk</p>
              <p className="mt-1 font-label text-2xl font-medium text-ink">
                {peakProb.toFixed(4)}
              </p>
              <p className="mt-0.5 text-[11px] text-muted">Across hours 1–{currentHour}</p>
            </div>
          </div>

          {/* Vitals Anomaly & Delta Table */}
          <div>
            <h2 className="mb-2 font-label text-[11.5px] font-medium uppercase tracking-wider text-muted">
              Physiological Trajectory & Completeness Audit
            </h2>
            <div className="overflow-x-auto rounded-panel border border-rule">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-rule bg-app font-medium text-muted">
                  <tr>
                    <th className="py-2.5 px-3">Vital Sign</th>
                    <th className="py-2.5 px-3">Baseline</th>
                    <th className="py-2.5 px-3">Current (Hr {currentHour})</th>
                    <th className="py-2.5 px-3">Net Delta</th>
                    <th className="py-2.5 px-3">Measurement Provenance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-rule bg-surface">
                  {vitals.map((v) => {
                    const firstRecordedPoint = causalPoints.find((pt) => pt.values[v.key] !== null);
                    const baselineVal = firstRecordedPoint?.values[v.key] ?? null;
                    const baselineHour = firstRecordedPoint?.hour ?? null;

                    const currVal = latestPoint?.values[v.key] ?? null;
                    const isMissingNow = latestPoint?.missing.includes(v.key) ?? false;
                    const delta = baselineVal !== null && currVal !== null ? currVal - baselineVal : null;

                    return (
                      <tr key={v.key} className="hover:bg-app/20">
                        <td className="py-2.5 px-3 font-medium text-ink">
                          {v.name} <span className="font-normal text-muted">({v.unit})</span>
                        </td>
                        <td className="py-2.5 px-3 text-muted">
                          {baselineVal !== null
                            ? `${baselineVal}${baselineHour !== 1 ? ` (Hr ${baselineHour})` : ""}`
                            : "—"}
                        </td>
                        <td className="py-2.5 px-3 font-medium text-ink">
                          {currVal !== null ? currVal : "—"}
                        </td>
                        <td className="py-2.5 px-3">
                          {delta !== null ? (
                            <span
                              className={
                                Math.abs(delta) > 0.01
                                  ? delta > 0
                                    ? "font-medium text-ink"
                                    : "text-muted"
                                  : "text-muted"
                              }
                            >
                              {delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)}
                            </span>
                          ) : (
                            <span className="text-muted">—</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3">
                          {currVal === null ? (
                            <span className="text-missing italic">Unmeasured</span>
                          ) : isMissingNow ? (
                            <span className="text-imputed italic">Causally forward-filled</span>
                          ) : (
                            <span className="font-medium text-accent">Measured at Hr {currentHour}</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Data Freshness & Model Provenance */}
          <div className="rounded-panel border border-rule bg-app/30 p-3.5 text-xs text-muted space-y-1.5">
            <p>
              <strong className="text-ink">Data Freshness Audit:</strong>{" "}
              {prediction?.empty_count ?? 0} of 5 vital signs unmeasured at Hour {currentHour}. Missing
              values are causally imputed using forward-fill and training-set median baselines without
              lookahead bias.
            </p>
            <p>
              <strong className="text-ink">Inference Engine:</strong>{" "}
              {prediction?.model_name ?? "logistic_regression_baseline"} (22 point-in-time features). Predictions
              strictly isolated up to Hour {currentHour}.
            </p>
          </div>

          {/* Clinician Handover Sign-Off Block */}
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

          {/* Research Prototype Regulatory Notice */}
          <p className="text-[10.5px] text-center text-muted italic">
            Not for direct clinical decision-making. Research prototype evaluated on PhysioNet Computing in
            Cardiology Challenge 2019 data.
          </p>
        </div>
      </div>
    </div>
  );
}
