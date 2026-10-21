"use client";

import React, { useEffect, useId, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { usePathname } from "next/navigation";
import Select from "./Select";
import { useDashboardState } from "./DashboardState";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Patient {
  patient_id: string;
  max_hour: number;
  total_observations: number;
  split: string;
}

interface PointValues {
  hr: number | null;
  sbp: number | null;
  resp: number | null;
  temp: number | null;
  wbc: number | null;
}

interface TimelinePoint {
  hour: number;
  values: PointValues;
  missing: string[];
  probability: number | null;
  alert: number | null;
}

interface TimelineResponse {
  patient_id: string;
  upto: number | null;
  threshold: number;
  live_path_label: string;
  points: TimelinePoint[];
}

interface PredictResponse {
  patient_id: string;
  hour: number;
  probability: number;
  alert: number;
  empty_count: number;
  model_name: string;
  threshold: number;
  live_path_label: string;
}

interface ModelMetric {
  name: string;
  auprc: number | string;
  precision: number | string;
  recall: number | string;
  specificity: number | string;
  threshold: number | string;
  alert_burden?: string;
  notes?: string;
}

type SignalKey = keyof PointValues;
type CellState = "measured" | "imputed" | "missing";
type SignalCell = { hour: number; value: number | null; state: CellState };

interface Signal {
  key: SignalKey;
  label: string;
  unit: string;
  normal: string;
}

const SIGNALS: Signal[] = [
  { key: "hr", label: "Heart rate", unit: "bpm", normal: "60–100" },
  { key: "sbp", label: "Systolic BP", unit: "mmHg", normal: "90–140" },
  { key: "resp", label: "Respiration", unit: "br/min", normal: "12–20" },
  { key: "temp", label: "Temperature", unit: "°C", normal: "36.5–37.5" },
  { key: "wbc", label: "WBC", unit: "10³/µL", normal: "4.5–11.0" },
];

const LIVE_PATH_FALLBACK = "Live path: Logistic Regression baseline (research prototype)";

const STATE_LABEL: Record<CellState, string> = {
  measured: "measured",
  imputed: "not measured · causal fill",
  missing: "not measured · baseline fill",
};

interface ObservationForm {
  patientId: string;
  hour: string;
  hr: string;
  sbp: string;
  resp: string;
  temp: string;
  wbc: string;
}

const EMPTY_OBSERVATION: ObservationForm = {
  patientId: "",
  hour: "",
  hr: "",
  sbp: "",
  resp: "",
  temp: "",
  wbc: "",
};

const OBSERVATION_FIELDS: {
  key: keyof ObservationForm;
  label: string;
  unit?: string;
  placeholder?: string;
  inputMode?: "numeric" | "decimal";
}[] = [
  { key: "patientId", label: "Patient ID", placeholder: "P001" },
  { key: "hour", label: "Hour", placeholder: "1", inputMode: "numeric" },
  { key: "hr", label: "HR", unit: "bpm", inputMode: "decimal" },
  { key: "sbp", label: "SBP", unit: "mmHg", inputMode: "decimal" },
  { key: "resp", label: "Resp", unit: "br/min", inputMode: "decimal" },
  { key: "temp", label: "Temp", unit: "°C", inputMode: "decimal" },
  { key: "wbc", label: "WBC", unit: "10³/µL", inputMode: "decimal" },
];

const FIELD_INPUT =
  "min-h-[42px] w-full min-w-0 rounded-panel border border-rule bg-surface px-3.5 py-2.5 font-label text-[12.5px] text-ink transition-colors";

const STATE_BASE =
  "inline-flex items-center gap-1.5 whitespace-nowrap font-label text-[10.5px] uppercase tracking-[0.04em] before:block before:h-[7px] before:w-[7px] before:shrink-0 before:rounded-full before:content-['']";

const STATE_CLASS: Record<CellState, string> = {
  measured: "text-signal before:bg-signal",
  imputed: "text-ink before:border-[1.5px] before:border-dashed before:border-imputed",
  missing: "text-muted before:rounded-none before:border before:border-muted",
};

const TABLE_WRAP = "mt-2 overflow-x-auto rounded-panel border border-rule bg-surface";
const TABLE = "w-full border-collapse text-[13px]";
const TABLE_TH =
  "whitespace-nowrap border-b border-rule bg-app px-4 py-2.5 text-left font-label text-[10.5px] font-medium uppercase tracking-[0.06em] text-muted";
const TABLE_ROW =
  "transition-colors hover:bg-app [&:last-child>th]:border-b-0 [&:last-child>td]:border-b-0";
const TABLE_CELL = "border-b border-rule-soft px-4 py-3 text-left align-top";
const TABLE_HEAD_CELL = `${TABLE_CELL} font-medium text-ink`;
const VITAL_CELL = "border-b border-rule-soft px-4 py-3 text-left align-middle";
const VITAL_HEAD_CELL = `${VITAL_CELL} font-medium text-ink`;

function formatValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function metricValue(value: number | string | undefined): string {
  if (value === undefined) return "—";
  return typeof value === "number" ? value.toFixed(4) : value;
}

function RiskChart({
  series,
  threshold,
  cutoff,
}: {
  series: { hour: number; probability: number; alert: number }[];
  threshold: number | null;
  cutoff: number;
}) {
  if (series.length === 0) {
    return <p className="py-7 text-[13px] text-muted">No estimate stored for this hour yet.</p>;
  }

  const height = 200;
  const pad = { top: 18, right: 22, bottom: 28, left: 38 };
  const width = Math.max(640, cutoff * 56);
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = (hour: number) =>
    pad.left + (cutoff <= 1 ? plotW / 2 : ((hour - 1) / (cutoff - 1)) * plotW);
  const y = (probability: number) => pad.top + (1 - probability) * plotH;
  const step = cutoff <= 10 ? 1 : Math.ceil(cutoff / 8);

  return (
    <div className="overflow-x-auto">
      <svg
        className="mx-auto block"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Model risk estimate from hour 1 to hour ${cutoff}`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((value) => (
          <g key={value}>
            <line
              className="stroke-rule-soft stroke-1"
              x1={pad.left}
              x2={width - pad.right}
              y1={y(value)}
              y2={y(value)}
            />
            <text className="fill-muted font-label text-[10px]" x={pad.left - 8} y={y(value) + 3.5} textAnchor="end">
              {value.toFixed(2)}
            </text>
          </g>
        ))}

        {Array.from({ length: cutoff }, (_, index) => index + 1)
          .filter((hour) => hour === 1 || hour === cutoff || hour % step === 0)
          .map((hour) => (
            <text
              key={hour}
              className="fill-muted font-label text-[10px]"
              x={x(hour)}
              y={height - 8}
              textAnchor="middle"
            >
              {hour}
            </text>
          ))}

        {threshold !== null && (
          <>
            <line
              className="stroke-alert stroke-1 [stroke-dasharray:5_4]"
              x1={pad.left}
              x2={width - pad.right}
              y1={y(threshold)}
              y2={y(threshold)}
            />
            <text
              className="fill-alert font-label text-[10px]"
              x={width - pad.right}
              y={Math.max(pad.top + 10, y(threshold) - 6)}
              textAnchor="end"
            >
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
            className={
              point.alert === 1
                ? "fill-alert stroke-surface stroke-2"
                : "fill-accent stroke-surface stroke-2"
            }
            cx={x(point.hour)}
            cy={y(point.probability)}
            r={point.hour === cutoff ? 5 : 3.5}
          />
        ))}
      </svg>
    </div>
  );
}

function HourScrubber({
  id,
  hour,
  maxHour,
  onChange,
}: {
  id: string;
  hour: number;
  maxHour: number;
  onChange: (hour: number) => void;
}) {
  const progress =
    maxHour <= 1 ? 0 : Math.min(1, Math.max(0, (hour - 1) / (maxHour - 1))) * 100;

  return (
    <div className="mt-[26px] flex max-w-[440px] items-center gap-2">
      <button
        type="button"
        className="inline-flex h-9 w-9 flex-none cursor-pointer items-center justify-center rounded-panel text-muted transition-colors hover:bg-surface hover:text-ink focus-visible:outline-2 focus-visible:outline-rule disabled:cursor-not-allowed disabled:text-missing"
        onClick={() => onChange(Math.max(1, hour - 1))}
        disabled={hour <= 1}
        aria-label="Previous hour"
      >
        <ChevronLeft size={18} strokeWidth={1.8} aria-hidden="true" />
      </button>
      <input
        id={id}
        className="hour-range h-6 min-w-[120px] flex-1 cursor-pointer"
        type="range"
        min={1}
        max={Math.max(1, maxHour)}
        value={Math.min(hour, Math.max(1, maxHour))}
        onChange={(event) => onChange(Number.parseInt(event.target.value, 10))}
        style={{ "--range-progress": `${progress}%` } as React.CSSProperties}
        aria-label="Hour of stay"
        aria-valuetext={`Hour ${hour} of ${maxHour}`}
      />
      <button
        type="button"
        className="inline-flex h-9 w-9 flex-none cursor-pointer items-center justify-center rounded-panel text-muted transition-colors hover:bg-surface hover:text-ink focus-visible:outline-2 focus-visible:outline-rule disabled:cursor-not-allowed disabled:text-missing"
        onClick={() => onChange(Math.min(maxHour, hour + 1))}
        disabled={hour >= maxHour}
        aria-label="Next hour"
      >
        <ChevronRight size={18} strokeWidth={1.8} aria-hidden="true" />
      </button>
    </div>
  );
}

function signalCells(signal: Signal, points: TimelinePoint[], cutoff: number) {
  const byHour = new Map(points.map((point) => [point.hour, point]));
  const cells: SignalCell[] = [];
  let lastValue: number | null = null;

  for (let hour = 1; hour <= cutoff; hour++) {
    const value = byHour.get(hour)?.values[signal.key] ?? null;
    if (value !== null) {
      lastValue = value;
      cells.push({ hour, value, state: "measured" });
    } else {
      cells.push({ hour, value: lastValue, state: lastValue === null ? "missing" : "imputed" });
    }
  }

  return { cells, current: cells[cells.length - 1] };
}

function SignalTrend({ cells, cutoff, label }: { cells: SignalCell[]; cutoff: number; label: string }) {
  const measured = cells
    .map((cell) => cell.value)
    .filter((value): value is number => value !== null);
  const min = measured.length ? Math.min(...measured) : 0;
  const max = measured.length ? Math.max(...measured) : 1;
  const span = max - min || 1;
  const height = 52;
  const padX = 10;
  const padY = 10;
  const width = Math.max(280, cutoff * 30);
  const x = (hour: number) =>
    padX + (cutoff <= 1 ? (width - padX * 2) / 2 : ((hour - 1) / (cutoff - 1)) * (width - padX * 2));
  const y = (value: number) => height - padY - ((value - min) / span) * (height - padY * 2);

  return (
    <svg
      className="block"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${label}, hours 1 to ${cutoff}`}
    >
      <line
        className="stroke-rule stroke-1"
        x1={padX}
        x2={width - padX}
        y1={height - padY}
        y2={height - padY}
      />
      {cells.map((cell, index) => {
        if (cell.state === "missing") {
          return (
            <text
              key={`m-${cell.hour}`}
              className="fill-missing font-label text-xs"
              x={x(cell.hour)}
              y={height - padY - 3}
              textAnchor="middle"
            >
              ×
            </text>
          );
        }
        if (index === 0) return null;
        const previous = cells[index - 1];
        if (previous.value === null) return null;
        return (
          <line
            key={`l-${cell.hour}`}
            className={
              cell.state === "imputed"
                ? "stroke-imputed stroke-[1.5] [stroke-dasharray:3_3]"
                : "stroke-signal stroke-[1.5]"
            }
            x1={x(previous.hour)}
            y1={y(previous.value)}
            x2={x(cell.hour)}
            y2={y(cell.value as number)}
          />
        );
      })}
      {cells.map((cell) =>
        cell.state === "measured" ? (
          <circle
            key={`p-${cell.hour}`}
            className="fill-signal"
            cx={x(cell.hour)}
            cy={y(cell.value as number)}
            r={2.6}
          />
        ) : cell.state === "imputed" ? (
          <circle
            key={`p-${cell.hour}`}
            className="fill-surface stroke-imputed stroke-[1.5] [stroke-dasharray:2_2]"
            cx={x(cell.hour)}
            cy={y(cell.value as number)}
            r={2.6}
          />
        ) : null
      )}
    </svg>
  );
}

export default function SepsisSignalDashboard() {
  const pathname = usePathname();
  const activePage = pathname === "/live" ? "live" : pathname === "/evaluation" ? "evaluation" : "overview";
  const pageTitle = activePage === "live" ? "Live input" : activePage === "evaluation" ? "Evaluation" : "Overview";
  const { selectedPatientId, setSelectedPatientId, currentHour, setCurrentHour } = useDashboardState();
  const patientSelectId = useId();
  const hourScrubberId = useId();

  const [patients, setPatients] = useState<Patient[]>([]);
  const [maxHour, setMaxHour] = useState(1);
  const [revision, setRevision] = useState(0);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [clockError, setClockError] = useState<string | null>(null);

  const [form, setForm] = useState<ObservationForm>(() => ({
    ...EMPTY_OBSERVATION,
    patientId: selectedPatientId,
  }));
  const [reply, setReply] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch(`${API_BASE}/patients`);
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (cancelled) return;
        const list: Patient[] = data.patients ?? [];
        setPatients(list);
        if (list.length > 0) {
          setSelectedPatientId(list[0].patient_id);
          setMaxHour(list[0].max_hour || 1);
          setCurrentHour(list[0].max_hour || 1);
        }
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }

      try {
        const res = await fetch(`${API_BASE}/metrics`);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) setMetrics(data.models ?? []);
        }
      } catch {
        if (!cancelled) setMetrics([]);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedPatientId) return;
    const controller = new AbortController();

    async function loadHour() {
      try {
        const [timelineRes, predictRes] = await Promise.all([
          fetch(
            `${API_BASE}/timeline?patient_id=${encodeURIComponent(selectedPatientId)}&upto=${currentHour}`,
            { signal: controller.signal }
          ),
          fetch(
            `${API_BASE}/predict?patient_id=${encodeURIComponent(selectedPatientId)}&hour=${currentHour}`,
            { signal: controller.signal }
          ),
        ]);
        if (!timelineRes.ok || !predictRes.ok) throw new Error("bad response");
        const [timelineData, predictData]: [TimelineResponse, PredictResponse] = await Promise.all([
          timelineRes.json(),
          predictRes.json(),
        ]);
        setTimeline(timelineData);
        setPrediction(predictData);
        setClockError(null);
      } catch {
        if (!controller.signal.aborted) {
          setClockError("Could not refresh this hour. Is the local API running?");
        }
      }
    }

    loadHour();
    return () => controller.abort();
  }, [selectedPatientId, currentHour, revision]);

  useEffect(() => {
    if (status !== "ready") return;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/patients`);
        if (res.ok) {
          const data = await res.json();
          setPatients(data.patients ?? []);
        }
      } catch {
        // API hiccup: keep the last known patients.
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [status]);

  useEffect(() => {
    const mine = patients.find((patient) => patient.patient_id === selectedPatientId);
    if (mine && mine.max_hour > maxHour) setMaxHour(mine.max_hour);
  }, [patients, selectedPatientId, maxHour]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (event.key === "ArrowLeft") {
        setCurrentHour((hour) => Math.max(1, hour - 1));
      } else if (event.key === "ArrowRight") {
        setCurrentHour((hour) => Math.min(maxHour, hour + 1));
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [maxHour]);

  function choosePatient(patientId: string) {
    const found = patients.find((patient) => patient.patient_id === patientId);
    setSelectedPatientId(patientId);
    setForm((previous) => ({ ...previous, patientId }));
    setMaxHour(found?.max_hour || 1);
    setCurrentHour(found?.max_hour || 1);
  }

  function updateField(key: keyof ObservationForm, value: string) {
    setForm((previous) => ({ ...previous, [key]: value }));
  }

  async function submitLine(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const patientId = form.patientId.trim();
    const hourText = form.hour.trim();
    const signalText = [form.hr, form.sbp, form.resp, form.temp, form.wbc].map((text) =>
      text.trim()
    );

    if (!patientId) {
      setReply("E|patient ID is required");
      return;
    }
    if (patientId.includes("|")) {
      setReply("E|patient ID cannot contain a pipe character");
      return;
    }

    const hour = Number(hourText);
    if (!hourText || !Number.isInteger(hour) || hour < 1) {
      setReply("E|hour must be a whole number of at least 1");
      return;
    }

    const values = signalText.map((text) => (text === "" ? null : Number(text)));
    if (values.some((value) => value !== null && !Number.isFinite(value))) {
      setReply("E|signal values must be numeric or empty");
      return;
    }

    const emptyCount = signalText.filter((text) => text === "").length;
    setSending(true);

    try {
      const res = await fetch(`${API_BASE}/observation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          hour,
          hr: values[0],
          sbp: values[1],
          resp: values[2],
          temp: values[3],
          wbc: values[4],
        }),
      });

      if (!res.ok) {
        setReply(`E|server returned ${res.status}`);
        return;
      }

      const data: PredictResponse = await res.json();
      setReply(`R|${patientId}|${hour}|${data.probability.toFixed(4)}|${data.alert}|${emptyCount}`);

      try {
        const listRes = await fetch(`${API_BASE}/patients`);
        if (listRes.ok) {
          const listData = await listRes.json();
          const list: Patient[] = listData.patients ?? [];
          setPatients(list);
          const found = list.find((patient) => patient.patient_id === patientId);
          setMaxHour(found?.max_hour ?? hour);
        }
      } catch {
        setMaxHour((previous) => Math.max(previous, hour));
      }

      setSelectedPatientId(patientId);
      setCurrentHour(hour);
      setRevision((value) => value + 1);
    } catch {
      setReply("E|network error");
    } finally {
      setSending(false);
    }
  }

  const selectedPatient = patients.find((patient) => patient.patient_id === selectedPatientId);
  const shownTimeline =
    timeline && timeline.patient_id === selectedPatientId && timeline.upto === currentHour
      ? timeline
      : null;
  const shownPrediction =
    prediction &&
    prediction.patient_id === selectedPatientId &&
    prediction.hour === currentHour
      ? prediction
      : null;
  const threshold = shownPrediction?.threshold ?? shownTimeline?.threshold ?? null;
  const livePathLabel =
    shownPrediction?.live_path_label ?? shownTimeline?.live_path_label ?? LIVE_PATH_FALLBACK;
  const alertActive = shownPrediction?.alert === 1;

  const riskSeries = useMemo(() => {
    const byHour = new Map<number, { hour: number; probability: number; alert: number }>();
    for (const point of shownTimeline?.points ?? []) {
      if (point.hour <= currentHour && point.probability !== null) {
        byHour.set(point.hour, {
          hour: point.hour,
          probability: point.probability,
          alert: point.alert ?? 0,
        });
      }
    }
    if (shownPrediction) {
      byHour.set(shownPrediction.hour, {
        hour: shownPrediction.hour,
        probability: shownPrediction.probability,
        alert: shownPrediction.alert,
      });
    }
    return [...byHour.values()].sort((a, b) => a.hour - b.hour);
  }, [shownTimeline, shownPrediction, currentHour]);

  return (
    <>
      <main className="mx-auto max-w-[1080px] px-8 pt-8 pb-[72px] max-[860px]:px-5 max-[860px]:pb-14">
      <header className="flex min-h-[76px] items-center justify-between gap-6 py-4 max-[600px]:flex-wrap">
        <h1 className="text-[30px] font-semibold leading-none tracking-[-0.04em]">{pageTitle}</h1>

        {activePage !== "evaluation" && <div className="flex min-w-0 items-center gap-2.5">
          <label className="shrink-0 font-label text-[11px] font-medium text-muted" htmlFor={patientSelectId}>
            Patient
          </label>
          {patients.length > 0 ? (
            <Select
              id={patientSelectId}
              value={selectedPatientId}
              options={patients.map((patient) => ({
                value: patient.patient_id,
                label: `${patient.patient_id} · ${patient.split === "test" ? "test" : "live"}`,
              }))}
              onChange={choosePatient}
              ariaLabel="Patients"
            />
          ) : (
            <span className="text-[13px] text-muted">No patient records</span>
          )}
        </div>}
      </header>

      {status === "error" && (
        <p className="mt-5 border-l-[3px] border-alert bg-surface px-3.5 py-2.5 text-[13px] text-alert">
          Cannot reach the local API at {API_BASE}. Start it, then reload this page.
        </p>
      )}

      {activePage === "overview" && <section
        className="grid grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)] gap-12 py-11 max-[860px]:grid-cols-1 max-[860px]:gap-7 max-[860px]:py-8"
        aria-labelledby="clock-heading"
      >
        <div>
          <p className="font-label text-[11px] font-medium uppercase tracking-[0.07em] text-muted" id="clock-heading">
            Hour of stay
          </p>
          <div className="mt-2.5 flex items-baseline gap-2.5">
            <span className="font-label text-[clamp(56px,8vw,88px)] font-medium leading-[0.95] tracking-[-0.045em] tabular-nums">
              {currentHour}
            </span>
            <span className="font-label text-lg text-muted tabular-nums">/ {maxHour}</span>
          </div>

          <HourScrubber
            id={hourScrubberId}
            hour={currentHour}
            maxHour={maxHour}
            onChange={setCurrentHour}
          />

        </div>

        <div
          className="pl-12 max-[860px]:pl-0"
          aria-live="polite"
        >
          <p className="font-label text-[11px] font-medium uppercase tracking-[0.07em] text-muted">
            Risk · hour {currentHour}
          </p>
          {shownPrediction ? (
            <p
              className="mt-2.5 animate-[fadeIn_0.18s_ease-out] font-label text-[clamp(44px,6vw,64px)] font-medium leading-none tracking-[-0.04em] text-accent tabular-nums"
              key={currentHour}
            >
              {shownPrediction.probability.toFixed(4)}
            </p>
          ) : (
            <p className="mt-[18px] py-[18px] text-[15px] text-muted">No estimate yet</p>
          )}
          <div className="mt-3.5 flex flex-wrap items-center gap-2.5 font-label text-xs" data-alert={alertActive}>
            <span
              className={`h-[9px] w-[9px] flex-none border-[1.5px] ${
                alertActive ? "rounded-none border-alert bg-alert" : "rounded-full border-accent"
              }`}
              aria-hidden="true"
            />
            <span className={`text-xs font-medium uppercase tracking-[0.07em] ${alertActive ? "text-alert" : ""}`}>
              {shownPrediction ? (shownPrediction.alert === 1 ? "alert" : "no alert") : "no estimate"}
            </span>
            <span className="text-[11px] tracking-[0.02em] text-muted">
              threshold {threshold !== null ? threshold.toFixed(2) : "—"}
            </span>
          </div>
          {shownPrediction && (
            <p className="mt-3 text-xs text-muted">
              {shownPrediction.empty_count} empty {shownPrediction.empty_count === 1 ? "field" : "fields"} · {shownPrediction.model_name}
            </p>
          )}
          <p className="mt-2 text-[11px] text-muted">{livePathLabel}</p>
          {clockError && <p className="mt-1.5 text-xs text-alert">{clockError}</p>}
        </div>
      </section>}

      {activePage === "overview" && patients.length === 0 && status !== "loading" ? (
        <section className="pt-11">
          <p className="py-5 text-sm text-muted">No data received for this patient yet.</p>
        </section>
      ) : (
        <>
          <section className={activePage === "overview" ? "pt-11" : "hidden"} aria-labelledby="timeline-heading">
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2.5">
              <div>
                <h2 className="text-xl font-semibold tracking-[-0.02em]" id="timeline-heading">
                  Timeline
                </h2>
                <p className="mt-1 max-w-[62ch] text-[13px] text-muted">
                  Vitals and estimates through hour {currentHour}.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap gap-x-[22px] gap-y-2 py-3 text-xs text-muted">
              <span className="inline-flex items-center gap-[7px]">
                <span className="h-[7px] w-[7px] flex-none rounded-full bg-signal" aria-hidden="true" />
                Measured
              </span>
              <span className="inline-flex items-center gap-[7px]">
                <span
                  className="inline-flex h-3.5 w-3.5 flex-none items-center justify-center font-label text-xs leading-none text-missing"
                  aria-hidden="true"
                >
                  ×
                </span>
                Missing
              </span>
              <span className="inline-flex items-center gap-[7px]">
                <span className="h-[7px] w-[7px] flex-none rounded-full border-[1.5px] border-dashed border-imputed" aria-hidden="true" />
                Imputed
              </span>
              <span className="inline-flex items-center gap-[7px]">
                <span className="h-0.5 w-4 flex-none bg-accent" aria-hidden="true" />
                Estimate
              </span>
              <span className="inline-flex items-center gap-[7px]">
                <span className="h-0 w-4 flex-none border-t-2 border-dashed border-alert" aria-hidden="true" />
                Threshold
              </span>
              <span className="inline-flex items-center gap-[7px]">
                <span className="h-2 w-2 flex-none border border-dashed border-missing" aria-hidden="true" />
                Benchmark label · retrospective
              </span>
            </div>

            <div className="mx-auto mt-5 flex w-full flex-wrap items-baseline flex-col gap-y-0.5 font-label text-[10.5px] text-muted">
              <span className="tracking-[0.03em]">
                {currentHour === 1 ? "Hour 1" : `Hours 1–${currentHour}`}
              </span>
            </div>
            <div className="mt-2.5 rounded-panel border border-rule bg-surface p-4 w-fit mx-auto">
              <RiskChart series={riskSeries} threshold={threshold} cutoff={currentHour} />
            </div>

            <p className="mt-7 mb-2 font-label text-[11px] font-medium uppercase tracking-[0.07em] text-muted">
              Vitals · hour {currentHour}
            </p>
            <div className={TABLE_WRAP}>
              <table className={TABLE} aria-label={`Vitals at hour ${currentHour}`}>
                <thead>
                  <tr>
                    <th scope="col" className={`${TABLE_TH} w-[140px]`}>
                      Signal
                    </th>
                    <th scope="col" className={`${TABLE_TH} w-[120px]`}>
                      Typical range
                    </th>
                    <th scope="col" className={TABLE_TH}>
                      Trend
                    </th>
                    <th scope="col" className={`${TABLE_TH} w-[110px]`}>
                      Value
                    </th>
                    <th scope="col" className={`${TABLE_TH} w-[210px]`}>
                      State
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {SIGNALS.map((signal) => {
                    const { cells, current } = signalCells(
                      signal,
                      shownTimeline?.points ?? [],
                      currentHour
                    );
                    return (
                      <tr key={signal.key} className={TABLE_ROW}>
                        <th scope="row" className={VITAL_HEAD_CELL}>
                          {signal.label}
                        </th>
                        <td className={`${VITAL_CELL} text-muted`}>{signal.normal}</td>
                        <td className={VITAL_CELL}>
                          <SignalTrend cells={cells} cutoff={currentHour} label={signal.label} />
                        </td>
                        <td className={`${VITAL_CELL} font-label tabular-nums`}>
                          {current.value !== null
                            ? `${formatValue(current.value)} ${signal.unit}`
                            : "—"}
                        </td>
                        <td className={VITAL_CELL}>
                          <span className={`${STATE_BASE} ${STATE_CLASS[current.state]}`}>
                            {STATE_LABEL[current.state]}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className={activePage === "live" ? "pt-11" : "hidden"} aria-labelledby="stream-heading">
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2.5">
              <div>
                <h2 className="text-xl font-semibold tracking-[-0.02em]" id="stream-heading">
                  Patient hour
                </h2>
                <p className="mt-1 text-[12px] text-muted">{livePathLabel}</p>
              </div>
            </div>

            <form onSubmit={submitLine}>
              <div className="grid grid-cols-2 gap-x-3 gap-y-3 sm:grid-cols-4 lg:grid-cols-7">
                {OBSERVATION_FIELDS.map((field) => (
                  <label key={field.key} className="flex min-w-0 flex-col gap-1.5">
                    <span className="font-label text-[10.5px] font-medium uppercase tracking-[0.06em] text-muted">
                      {field.label}
                      {field.unit && (
                        <span className="ml-1 font-normal normal-case tracking-normal">
                          ({field.unit})
                        </span>
                      )}
                    </span>
                    <input
                      className={FIELD_INPUT}
                      type="text"
                      inputMode={field.inputMode}
                      value={form[field.key]}
                      onChange={(event) => updateField(field.key, event.target.value)}
                      placeholder={field.placeholder}
                      spellCheck={false}
                      autoComplete="off"
                    />
                  </label>
                ))}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2.5">
                <button
                  type="submit"
                  className="min-h-[42px] cursor-pointer rounded-panel border border-accent bg-accent px-[18px] text-[13px] font-medium text-surface disabled:cursor-progress disabled:opacity-60"
                  disabled={sending}
                >
                  {sending ? "Sending…" : "Send line"}
                </button>
                <button
                  type="button"
                  className="min-h-[42px] cursor-pointer rounded-panel border border-rule bg-transparent px-3.5 text-[13px] text-muted hover:text-ink"
                  onClick={() =>
                    setForm({
                      patientId: selectedPatientId || "P001",
                      hour: String(maxHour + 1),
                      hr: "88",
                      sbp: "112",
                      resp: "",
                      temp: "",
                      wbc: "",
                    })
                  }
                >
                  3 blanks
                </button>
                <span className="font-label text-[11px] text-muted">Blank = missing.</span>
              </div>
            </form>

            {reply && (
              <p
                className="mt-4 overflow-x-auto whitespace-pre-wrap break-words rounded-panel border border-rule border-l-[3px] border-l-accent bg-surface px-3.5 py-3 font-label text-[12.5px] data-[error=true]:border-l-alert data-[error=true]:text-alert"
                data-error={reply.startsWith("E|")}
                role="status"
              >
                {reply}
              </p>
            )}
          </section>

          <section className={activePage === "evaluation" ? "pt-11" : "hidden"} aria-labelledby="evaluation-heading">
            <div className="mb-5 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2.5">
              <div>
                <h2 className="text-xl font-semibold tracking-[-0.02em]" id="evaluation-heading">
                  Held-out evaluation
                </h2>
              </div>
              <p className="font-label text-[11px] font-medium uppercase tracking-[0.07em] text-muted">
                TEST split
              </p>
            </div>

            {metrics.length === 0 ? (
              <p className="py-5 text-sm text-muted">Evaluation results unavailable.</p>
            ) : (
              <div className={TABLE_WRAP}>
                <table className={TABLE}>
                  <thead>
                    <tr>
                      <th scope="col" className={TABLE_TH}>
                        Model
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        AUPRC
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        Precision
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        Recall
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        Specificity
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        Threshold
                      </th>
                      <th scope="col" className={TABLE_TH}>
                        Alert burden
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((metric) => (
                      <tr key={metric.name} className={TABLE_ROW}>
                        <th scope="row" className={`${TABLE_HEAD_CELL} min-w-[220px]`}>
                          <span className="block">{metric.name}</span>
                          {metric.notes && (
                            <span className="mt-[3px] block max-w-[280px] text-[11.5px] font-normal text-muted">
                              {metric.notes}
                            </span>
                          )}
                        </th>
                        <td className={`${TABLE_CELL} font-label font-medium text-accent tabular-nums`}>
                          {metricValue(metric.auprc)}
                        </td>
                        <td className={`${TABLE_CELL} font-label tabular-nums`}>{metricValue(metric.precision)}</td>
                        <td className={`${TABLE_CELL} font-label tabular-nums`}>{metricValue(metric.recall)}</td>
                        <td className={`${TABLE_CELL} font-label tabular-nums`}>{metricValue(metric.specificity)}</td>
                        <td className={`${TABLE_CELL} font-label tabular-nums`}>{metricValue(metric.threshold)}</td>
                        <td className={`${TABLE_CELL} text-muted`}>{metric.alert_burden ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <p className="mt-5 max-w-[70ch] text-xs text-muted">
              Live estimates use five signals and can differ from offline results.
            </p>
          </section>
        </>
      )}

      <footer className="flex flex-wrap justify-between gap-x-10 gap-y-4 pt-7 text-xs leading-[1.6] text-muted">
        <div>
          PhysioNet Challenge 2019 (CC BY 4.0) · Reyna et al., Crit Care Med 2020.
        </div>
      </footer>
      </main>
    </>
  );
}
