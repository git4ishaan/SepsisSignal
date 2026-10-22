"use client";

import React from "react";
import { X, Printer } from "lucide-react";
import { createPortal } from "react-dom";

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

export interface ShiftReportData {
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
    contributors?: { feature: string; label: string; contribution: number }[];
  } | null;
  timelinePoints: TimelinePoint[];
}

export interface ShiftReportModalProps extends ShiftReportData {
  isOpen: boolean;
  onClose: () => void;
  generatedAt?: string;
}

function ReportRiskChart({
  series,
  threshold,
  cutoff,
}: {
  series: { hour: number; probability: number; alert: number }[];
  threshold: number | null;
  cutoff: number;
}) {
  if (series.length === 0) {
    return <p className="py-5 text-xs text-muted">No prediction history is available through hour {cutoff}.</p>;
  }

  const width = 720;
  const height = 190;
  const pad = { top: 18, right: 18, bottom: 28, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = (hour: number) => pad.left + (cutoff <= 1 ? plotW / 2 : ((hour - 1) / (cutoff - 1)) * plotW);
  const y = (probability: number) => pad.top + (1 - probability) * plotH;
  const step = cutoff <= 10 ? 1 : Math.ceil(cutoff / 8);

  return (
    <svg
      className="mt-2 block h-auto w-full"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Prediction probability by hour, from hour 1 through hour ${cutoff}`}
    >
      {[0, 0.25, 0.5, 0.75, 1].map((value) => (
        <g key={value}>
          <line className="stroke-rule-soft stroke-1" x1={pad.left} x2={width - pad.right} y1={y(value)} y2={y(value)} />
          <text className="fill-muted font-label text-[10px]" x={pad.left - 8} y={y(value) + 3.5} textAnchor="end">
            {value.toFixed(2)}
          </text>
        </g>
      ))}

      {Array.from({ length: cutoff }, (_, index) => index + 1)
        .filter((hour) => hour === 1 || hour === cutoff || hour % step === 0)
        .map((hour) => (
          <text key={hour} className="fill-muted font-label text-[10px]" x={x(hour)} y={height - 7} textAnchor="middle">
            {hour}
          </text>
        ))}

      {threshold !== null && (
        <>
          <line className="stroke-alert stroke-1 [stroke-dasharray:5_4]" x1={pad.left} x2={width - pad.right} y1={y(threshold)} y2={y(threshold)} />
          <text className="fill-alert font-label text-[10px]" x={width - pad.right} y={Math.max(pad.top + 10, y(threshold) - 6)} textAnchor="end">
            threshold {threshold.toFixed(2)}
          </text>
        </>
      )}

      {series.length > 1 && (
        <polyline
          className="fill-none stroke-accent stroke-2 [stroke-linejoin:round] [stroke-linecap:round]"
          points={series.map((point) => `${x(point.hour)},${y(point.probability)}`).join(" ")}
        />
      )}

      {series.map((point) => (
        <circle
          key={point.hour}
          className={point.alert === 1 ? "fill-alert stroke-surface stroke-2" : "fill-accent stroke-surface stroke-2"}
          cx={x(point.hour)}
          cy={y(point.probability)}
          r={point.hour === cutoff ? 5 : 3.5}
        />
      ))}
    </svg>
  );
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
  generatedAt,
}: ShiftReportModalProps) {
  const [portalReady, setPortalReady] = React.useState(false);

  React.useEffect(() => {
    setPortalReady(true);
  }, []);

  if (!isOpen || !portalReady) return null;

  // Keep the report point-in-time: never include data after the selected hour.
  const causalPoints = timelinePoints.filter((pt) => pt.hour <= currentHour);
  const latestPoint = causalPoints[causalPoints.length - 1];

  const alertPoints = causalPoints.filter((pt) => pt.alert === 1);
  const byHour = new Map<number, { hour: number; probability: number; alert: number }>();
  for (const point of causalPoints) {
    if (point.probability !== null) {
      byHour.set(point.hour, { hour: point.hour, probability: point.probability, alert: point.alert ?? 0 });
    }
  }
  if (prediction) {
    byHour.set(currentHour, { hour: currentHour, probability: prediction.probability, alert: prediction.alert });
  }
  const riskSeries = [...byHour.values()].sort((a, b) => a.hour - b.hour);
  const firstAlertHour = alertPoints.length > 0
    ? alertPoints[0].hour
    : prediction?.alert === 1 ? currentHour : null;
  const observedProbabilities = causalPoints
    .map((pt) => pt.probability)
    .filter((probability): probability is number => probability !== null);
  if (prediction) observedProbabilities.push(prediction.probability);
  const peakProb = observedProbabilities.length > 0 ? Math.max(...observedProbabilities) : null;

  const vitals = [
    { key: "hr" as const, name: "Heart rate", unit: "bpm" },
    { key: "sbp" as const, name: "Systolic BP", unit: "mmHg" },
    { key: "resp" as const, name: "Respiratory rate", unit: "br/min" },
    { key: "temp" as const, name: "Temperature", unit: "°C" },
    { key: "wbc" as const, name: "White blood cells", unit: "10³/µL" },
  ];
  const readingRows = vitals.map((v) => {
    const measured = causalPoints.filter((point) => point.values[v.key] !== null);
    return {
      vital: v,
      first: measured[0] ?? null,
      latest: measured[measured.length - 1] ?? null,
    };
  });
  const contributions = prediction?.contributors ?? [];
  const modelName = prediction?.model_name === "logistic_regression_baseline"
    ? "Logistic Regression baseline"
    : prediction?.model_name ?? "Logistic Regression baseline";
  const splitLabel = split ? `${split[0].toUpperCase()}${split.slice(1)} split` : "Unknown split";
  const repeatedReadings = readingRows.filter(({ first, latest }) =>
    first && latest && first.hour !== latest.hour
  );
  const readingChanges = repeatedReadings.map(({ vital, first, latest }) => {
    const firstValue = first!.values[vital.key]!;
    const latestValue = latest!.values[vital.key]!;
    if (firstValue === latestValue) return `${vital.name.toLowerCase()} remained at ${latestValue} ${vital.unit}`;
    const direction = latestValue > firstValue ? "rose" : "fell";
    return `${vital.name.toLowerCase()} ${direction} from ${firstValue} to ${latestValue} ${vital.unit}`;
  });
  const readingTrend = readingChanges.length > 0
    ? `Across repeated readings, ${readingChanges.join("; ")}.`
    : "There are not enough repeated readings to describe changes over time.";

  const reportTableWrap = "mt-2 overflow-x-auto rounded-panel border border-rule bg-surface";
  const reportTable = "w-full border-collapse text-[13px]";
  const reportTableHeader =
    "whitespace-nowrap border-b border-rule bg-app px-4 py-2.5 text-left font-label text-[10.5px] font-medium uppercase tracking-[0.06em] text-muted";
  const reportTableRow =
    "transition-colors hover:bg-app [&:last-child>th]:border-b-0 [&:last-child>td]:border-b-0";
  const reportTableCell = "border-b border-rule-soft px-4 py-2.5 text-left align-middle";

  return createPortal((
    <div
      className="print-report-overlay fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 print:p-0 print:bg-white"
      role="dialog"
      aria-modal="true"
      aria-labelledby="shift-report-title"
    >
      <div
        id="printable-shift-report"
        className="relative max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-panel border border-rule bg-surface p-6 text-ink shadow-xl"
      >
        <div className="mb-6 flex items-center justify-between gap-4 print-hidden">
          <span className="text-xs text-muted">Hour {currentHour} of {maxHour}</span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => window.print()}
              className="inline-flex min-h-9 cursor-pointer items-center gap-1.5 rounded-panel bg-accent px-3 text-xs font-medium text-surface hover:opacity-90"
            >
              <Printer size={14} aria-hidden="true" /> Save as PDF
            </button>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-panel text-muted hover:bg-app hover:text-ink"
              aria-label="Close report"
            >
              <X size={18} aria-hidden="true" />
            </button>
          </div>
        </div>

        <header>
          <h1 id="shift-report-title" className="text-xl font-semibold tracking-tight">
            Patient-hour risk summary
          </h1>
          <p className="mt-1 text-xs text-muted">
            {patientId || "Unknown patient"} · {splitLabel} · Hour {currentHour} · {new Date(generatedAt ?? Date.now()).toLocaleString()}
          </p>
        </header>

        <section className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-3" aria-label="Risk summary">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted">Risk probability</p>
            <p className="mt-1 font-label text-2xl tabular-nums text-accent">
              {prediction ? prediction.probability.toFixed(4) : "—"}
            </p>
            <p className="mt-1 text-[11px] text-muted">Threshold {prediction?.threshold.toFixed(2) ?? "—"}</p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted">Alert</p>
            <p className={`mt-2 flex items-center gap-2 text-sm font-semibold ${prediction?.alert === 1 ? "text-alert" : "text-ink"}`}>
              <span
                className={`h-2.5 w-2.5 flex-none border-[1.5px] ${prediction?.alert === 1 ? "border-alert bg-alert" : "rounded-full border-accent"}`}
                aria-hidden="true"
              />
              <span>{prediction ? (prediction.alert === 1 ? "Alert raised" : "No alert") : "No estimate"}</span>
            </p>
            <p className="mt-1 text-[11px] text-muted">
              {firstAlertHour !== null ? `First at hour ${firstAlertHour}` : `None through hour ${currentHour}`}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-[0.06em] text-muted">Peak probability</p>
            <p className="mt-1 font-label text-2xl tabular-nums">{peakProb !== null ? peakProb.toFixed(4) : "—"}</p>
            <p className="mt-1 text-[11px] text-muted">Hours 1–{currentHour}</p>
          </div>
        </section>

        <section className="mt-6" aria-labelledby="report-risk-chart-heading">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 id="report-risk-chart-heading" className="text-xs font-medium">Prediction by hour</h2>
            <span className="text-[10px] text-muted">Hours 1–{currentHour}</span>
          </div>
          <ReportRiskChart series={riskSeries} threshold={prediction?.threshold ?? null} cutoff={Math.max(currentHour, 1)} />
        </section>

        <section className="mt-6" aria-labelledby="inference-summary-heading">
          <h2 id="inference-summary-heading" className="text-xs font-medium">Inference summary</h2>
          <div className="mt-2 space-y-1.5 text-xs leading-relaxed text-muted">
            <p>
              <strong className="font-semibold text-ink">Estimate.</strong>{" "}
              {prediction
                ? <>{modelName} returned <strong className="font-semibold text-ink">{prediction.probability.toFixed(4)}</strong> at hour {currentHour}, against a {prediction.threshold.toFixed(2)} threshold; <strong className="font-semibold text-ink">{prediction.alert === 1 ? "an alert was raised" : "no alert was raised"}</strong>.</>
                : <>No model estimate is available for hour {currentHour}.</>}
            </p>
            <p><strong className="font-semibold text-ink">Reading trend.</strong> {readingTrend}</p>
            {prediction && (
              <p>
                <strong className="font-semibold text-ink">Missing inputs.</strong>{" "}
                <strong className="font-semibold text-ink">{prediction.empty_count} of 5 signals</strong> were missing; imputation uses <em>past-only carry-forward</em> and training-set medians.
              </p>
            )}
          </div>
        </section>

        {contributions.length > 0 && (
          <section className="mt-6" aria-labelledby="report-contributions-heading">
            <h2 id="report-contributions-heading" className="text-xs font-medium">Top model contributions</h2>
            <div className={reportTableWrap}>
              <table className={reportTable}>
                <thead>
                  <tr>
                    <th scope="col" className={reportTableHeader}>Model feature</th>
                    <th scope="col" className={`${reportTableHeader} text-right`}>Contribution (log-odds)</th>
                  </tr>
                </thead>
                <tbody>
                  {contributions.map((item) => (
                    <tr key={item.feature} className={reportTableRow}>
                      <th scope="row" className={`${reportTableCell} font-medium`} title={item.label}>{item.label}</th>
                      <td className={`${reportTableCell} text-right font-label tabular-nums ${item.contribution >= 0 ? "text-alert" : "text-muted"}`}>
                        {item.contribution >= 0 ? "+" : "−"}{Math.abs(item.contribution).toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-1.5 text-[10px] text-muted">Log-odds contributions describe model behavior, not clinical causes.</p>
          </section>
        )}

        <section className="mt-7" aria-labelledby="report-vitals-heading">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 id="report-vitals-heading" className="text-xs font-medium">Signal readings</h2>
            <span className="text-[10px] text-muted">Latest recorded hour: {latestPoint?.hour ?? "—"}</span>
          </div>
          <div className={reportTableWrap}>
            <table className={reportTable}>
              <thead>
                <tr>
                  <th scope="col" className={reportTableHeader}>Signal</th>
                  <th scope="col" className={reportTableHeader}>First reading</th>
                  <th scope="col" className={reportTableHeader}>Latest reading</th>
                </tr>
              </thead>
              <tbody>
                {readingRows.map(({ vital, first, latest }) => {
                  const firstValue = first?.values[vital.key] ?? null;
                  const latestValue = latest?.values[vital.key] ?? null;
                  return (
                    <tr key={vital.key} className={reportTableRow}>
                      <th scope="row" className={`${reportTableCell} font-medium`}>
                        {vital.name} <span className="font-normal text-muted">({vital.unit})</span>
                      </th>
                      <td className={`${reportTableCell} tabular-nums text-muted`}>
                        {firstValue !== null ? `${firstValue} · hour ${first?.hour}` : "—"}
                      </td>
                      <td className={`${reportTableCell} tabular-nums`}>
                        {latestValue !== null ? `${latestValue} · hour ${latest?.hour}` : "Not measured"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <p className="mt-6 text-[10px] leading-relaxed text-muted">
          Research demonstration using PhysioNet Challenge 2019 data. Not for clinical decision-making.
        </p>
      </div>
    </div>
  ), document.body);
}
