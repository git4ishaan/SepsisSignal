"use client";

import React, { useEffect, useState, useId } from "react";

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

export default function SepsisSignalDashboard() {
  const patientSelectId = useId();
  const hourScrubberId = useId();
  const simulateInputId = useId();

  // State
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("P001");
  const [currentHour, setCurrentHour] = useState<number>(1);
  const [maxHour, setMaxHour] = useState<number>(10);
  
  // Data for current hour and timeline
  const [currentPrediction, setCurrentPrediction] = useState<PredictResponse | null>(null);
  const [timelinePoints, setTimelinePoints] = useState<TimelinePoint[]>([]);
  const [threshold, setThreshold] = useState<number>(0.25);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  
  // Status & live replay
  const [loading, setLoading] = useState<boolean>(true);
  const [apiOnline, setApiOnline] = useState<boolean>(true);
  const [incomingLine, setIncomingLine] = useState<string>("");
  const [socketReply, setSocketReply] = useState<string>("");
  const [isSubmittingLine, setIsSubmittingLine] = useState<boolean>(false);

  // 1. Fetch patient list and metrics on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [ptsRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE}/patients`),
          fetch(`${API_BASE}/metrics`),
        ]);

        if (ptsRes.ok) {
          const ptsData = await ptsRes.json();
          if (ptsData.patients && ptsData.patients.length > 0) {
            setPatients(ptsData.patients);
            const initial = ptsData.patients[0];
            setSelectedPatientId(initial.patient_id);
            setMaxHour(initial.max_hour || 1);
            setCurrentHour(initial.max_hour || 1);
          }
        }
        if (metricsRes.ok) {
          const mData = await metricsRes.json();
          if (mData.models) {
            setMetrics(mData.models);
          }
        }
        setApiOnline(true);
      } catch (err) {
        console.warn("API offline or unreachable, using offline mock state", err);
        setApiOnline(false);
        // Fallback demo state
        setPatients([
          { patient_id: "P001", max_hour: 10, total_observations: 10, split: "test" },
          { patient_id: "P002", max_hour: 8, total_observations: 8, split: "test" },
        ]);
        setSelectedPatientId("P001");
        setMaxHour(10);
        setCurrentHour(5);
      } finally {
        setLoading(false);
      }
    }
    loadInitialData();
  }, []);

  // 2. Fetch timeline and prediction for selectedPatientId and currentHour
  useEffect(() => {
    if (!selectedPatientId) return;

    async function fetchClockData() {
      try {
        // Causal timeline: strictly points with hour <= currentHour
        const timelineRes = await fetch(`${API_BASE}/timeline?patient_id=${selectedPatientId}&upto=${currentHour}`);
        if (timelineRes.ok) {
          const tData: TimelineResponse = await timelineRes.json();
          setTimelinePoints(tData.points || []);
          if (tData.threshold) setThreshold(tData.threshold);
        }

        // Live prediction for currentHour
        const predRes = await fetch(`${API_BASE}/predict?patient_id=${selectedPatientId}&hour=${currentHour}`);
        if (predRes.ok) {
          const pData: PredictResponse = await predRes.json();
          setCurrentPrediction(pData);
        }
        setApiOnline(true);
      } catch (err) {
        console.warn("Could not fetch clock data from API", err);
        setApiOnline(false);
      }
    }

    fetchClockData();
  }, [selectedPatientId, currentHour]);

  // Handle patient change
  const handlePatientSelect = (pid: string) => {
    setSelectedPatientId(pid);
    const found = patients.find((p) => p.patient_id === pid);
    const m = found?.max_hour || 10;
    setMaxHour(m);
    setCurrentHour(m);
  };

  // Handle custom line submission (simulate socket protocol)
  const handleSendCustomLine = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!incomingLine.trim()) return;

    setIsSubmittingLine(true);
    setSocketReply("");

    try {
      // Parse line in frontend format check: H|patient_id|hour|hr|sbp|resp|temp|wbc
      const parts = incomingLine.trim().split("|");
      if (parts.length !== 8 || parts[0] !== "H") {
        setSocketReply(`E|malformed line: expected 8 fields starting with H`);
        setIsSubmittingLine(false);
        return;
      }

      const pid = parts[1];
      const hr = parseInt(parts[2], 10);
      const hr_val = parts[3] ? parseFloat(parts[3]) : null;
      const sbp_val = parts[4] ? parseFloat(parts[4]) : null;
      const resp_val = parts[5] ? parseFloat(parts[5]) : null;
      const temp_val = parts[6] ? parseFloat(parts[6]) : null;
      const wbc_val = parts[7] ? parseFloat(parts[7]) : null;

      const res = await fetch(`${API_BASE}/observation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: pid,
          hour: hr,
          hr: hr_val,
          sbp: sbp_val,
          resp: resp_val,
          temp: temp_val,
          wbc: wbc_val,
        }),
      });

      if (res.ok) {
        const data: PredictResponse = await res.json();
        const emptyCount = [hr_val, sbp_val, resp_val, temp_val, wbc_val].filter((v) => v === null).length;
        const reply = `R|${pid}|${hr}|${data.probability.toFixed(4)}|${data.alert}|${emptyCount}`;
        setSocketReply(reply);

        // Refresh patient list and view
        const ptsRes = await fetch(`${API_BASE}/patients`);
        if (ptsRes.ok) {
          const ptsData = await ptsRes.json();
          setPatients(ptsData.patients || []);
        }
        setSelectedPatientId(pid);
        setMaxHour((prev) => Math.max(prev, hr));
        setCurrentHour(hr);
      } else {
        setSocketReply(`E|failed to process observation (${res.status})`);
      }
    } catch (err: unknown) {
      setSocketReply(`E|network error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsSubmittingLine(false);
    }
  };

  // Keyboard navigation for the clock
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowLeft") {
        setCurrentHour((prev) => Math.max(1, prev - 1));
      } else if (e.key === "ArrowRight") {
        setCurrentHour((prev) => Math.min(maxHour, prev + 1));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [maxHour]);

  // Current values for point at currentHour
  const currentPoint = timelinePoints.find((p) => p.hour === currentHour);

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "24px 20px 80px" }}>
      {/* 1. Mandatory Disclaimer Banner */}
      <div
        style={{
          backgroundColor: "var(--notice-bg)",
          color: "var(--notice-text)",
          padding: "10px 16px",
          borderRadius: "6px",
          fontSize: "13px",
          fontWeight: 500,
          marginBottom: "24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          border: "1px solid rgba(217, 164, 65, 0.3)",
        }}
      >
        <span>
          <strong>Research prototype, not for clinical decision-making.</strong> All evaluations conducted retrospectively on public benchmark data.
        </span>
        <span style={{ fontSize: "12px", opacity: 0.9 }}>PhysioNet 2019 · Temp in Celsius (°C)</span>
      </div>

      {/* 2. Header */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          paddingBottom: "18px",
          borderBottom: "1px solid var(--border)",
          marginBottom: "28px",
          flexWrap: "wrap",
          gap: "16px",
        }}
      >
        <div>
          <h1 style={{ fontSize: "32px", color: "var(--text)", lineHeight: 1.15, marginBottom: "4px" }}>
            SepsisSignal
          </h1>
          <p style={{ color: "var(--muted)", fontSize: "14px" }}>
            Explainable early-warning research prototype on ICU longitudinal time-series
          </p>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              marginTop: "8px",
              padding: "4px 10px",
              borderRadius: "4px",
              backgroundColor: "rgba(15, 118, 110, 0.08)",
              color: "var(--accent)",
              fontSize: "12px",
              fontWeight: 500,
            }}
          >
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: "var(--accent)" }} />
            Live path: Logistic Regression baseline (research prototype)
          </div>
        </div>

        {/* Patient Picker */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <label htmlFor={patientSelectId} style={{ fontSize: "13px", color: "var(--muted)", fontWeight: 500 }}>Patient:</label>
          <select
            id={patientSelectId}
            value={selectedPatientId}
            onChange={(e) => handlePatientSelect(e.target.value)}
            style={{
              padding: "8px 14px",
              borderRadius: "6px",
              border: "1px solid var(--border)",
              backgroundColor: "var(--surface)",
              color: "var(--text)",
              fontSize: "14px",
              fontWeight: 500,
            }}
          >
            {patients.map((p) => (
              <option key={p.patient_id} value={p.patient_id}>
                {p.patient_id} ({p.split === "test" ? "Held-out Test" : "Live Stream"}) · {p.max_hour}h
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* 3. The Clock / Time Scrubber Section */}
      <section
        style={{
          backgroundColor: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          padding: "24px",
          marginBottom: "32px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <div style={{ fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)", fontWeight: 600, marginBottom: "4px" }}>
              Time Scrubber (Point-in-Time Inference)
            </div>
            <h2 style={{ fontSize: "22px", color: "var(--text)" }}>
              Hour <span className="mono">{currentHour}</span> of stay
            </h2>
            <p style={{ fontSize: "13px", color: "var(--muted)", marginTop: "2px" }}>
              Showing causal point-in-time observations where hour &le; {currentHour}. Future records are strictly withheld.
            </p>
          </div>

          {/* Current Hour Model Probability & Alert Card */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "20px",
              backgroundColor: "var(--bg)",
              padding: "12px 20px",
              borderRadius: "8px",
              border: "1px solid var(--border)",
            }}
          >
            <div>
              <div style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase", fontWeight: 600 }}>
                Estimated Risk
              </div>
              <div className="mono" style={{ fontSize: "28px", fontWeight: 600, color: "var(--score)" }}>
                {currentPrediction ? currentPrediction.probability.toFixed(4) : "0.0000"}
              </div>
            </div>

            <div style={{ borderLeft: "1px solid var(--border)", paddingLeft: "16px" }}>
              <div style={{ fontSize: "11px", color: "var(--muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: "4px" }}>
                Alert State (Threshold &ge; {threshold.toFixed(2)})
              </div>
              <span
                style={{
                  display: "inline-block",
                  padding: "4px 10px",
                  borderRadius: "4px",
                  fontSize: "12px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  backgroundColor:
                    currentPrediction?.alert === 1 ? "rgba(180, 95, 107, 0.15)" : "rgba(15, 118, 110, 0.12)",
                  color: currentPrediction?.alert === 1 ? "var(--alert)" : "var(--accent)",
                }}
              >
                {currentPrediction?.alert === 1 ? "Alert Active" : "No Alert"}
              </span>
            </div>
          </div>
        </div>

        {/* Clock Scrubber Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "16px", marginTop: "12px" }}>
          <button
            onClick={() => setCurrentHour((prev) => Math.max(1, prev - 1))}
            disabled={currentHour <= 1}
            style={{
              padding: "6px 14px",
              borderRadius: "6px",
              border: "1px solid var(--border)",
              backgroundColor: "var(--surface)",
              color: currentHour <= 1 ? "var(--muted)" : "var(--text)",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            &larr; Prev Hour
          </button>

          <input
            id={hourScrubberId}
            type="range"
            min={1}
            max={Math.max(1, maxHour)}
            value={currentHour}
            onChange={(e) => setCurrentHour(parseInt(e.target.value, 10))}
            style={{
              flex: 1,
              accentColor: "var(--score)",
              height: "6px",
              cursor: "pointer",
            }}
          />

          <button
            onClick={() => setCurrentHour((prev) => Math.min(maxHour, prev + 1))}
            disabled={currentHour >= maxHour}
            style={{
              padding: "6px 14px",
              borderRadius: "6px",
              border: "1px solid var(--border)",
              backgroundColor: "var(--surface)",
              color: currentHour >= maxHour ? "var(--muted)" : "var(--text)",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            Next Hour &rarr;
          </button>
        </div>

        {/* Hour markers indicators */}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "8px", fontSize: "11px", color: "var(--muted)" }}>
          <span>Hour 1 (Admission)</span>
          <span>(Use &larr; / &rarr; keys to scrub through ICU stay)</span>
          <span>Hour {maxHour}</span>
        </div>
      </section>

      {/* 4. Timeline Visualizations & Trajectory */}
      <section
        style={{
          backgroundColor: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          padding: "24px",
          marginBottom: "32px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h2 style={{ fontSize: "18px", color: "var(--text)" }}>Patient Timeline & Vital Trajectory</h2>
            <p style={{ fontSize: "13px", color: "var(--muted)" }}>
              Data strictly rendered up to Hour {currentHour}. Future hours are withheld from the timeline.
            </p>
          </div>

          {/* Legend */}
          <div style={{ display: "flex", gap: "14px", fontSize: "12px", color: "var(--muted)", flexWrap: "wrap" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--blue)" }} />
              Measured
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", border: "1.5px dashed var(--missing)" }} />
              Not Measured (Missing)
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--imputed)" }} />
              Causal Forward-Fill
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "12px", height: "2px", backgroundColor: "var(--score)" }} />
              Risk Score
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "12px", height: "2px", borderTop: "2px dashed var(--alert)" }} />
              Alert Threshold ({threshold.toFixed(2)})
            </span>
          </div>
        </div>

        {/* Risk Score Trajectory Chart (SVG) */}
        <div style={{ marginBottom: "28px" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "var(--muted)", marginBottom: "6px" }}>
            Estimated Risk Score (0.0 to 1.0)
          </div>
          <div style={{ position: "relative", width: "100%", height: "130px", border: "1px solid var(--border)", borderRadius: "6px", backgroundColor: "var(--bg)", overflow: "hidden" }}>
            <svg width="100%" height="100%" viewBox={`0 0 ${Math.max(10, maxHour) * 40} 130`} preserveAspectRatio="none">
              {/* Threshold line */}
              <line
                x1="0"
                y1={130 - threshold * 120}
                x2={Math.max(10, maxHour) * 40}
                y2={130 - threshold * 120}
                stroke="var(--alert)"
                strokeWidth="1.5"
                strokeDasharray="4 3"
              />

              {/* Trajectory line up to currentHour */}
              {timelinePoints.length > 1 && (
                <polyline
                  fill="none"
                  stroke="var(--score)"
                  strokeWidth="2.5"
                  points={timelinePoints
                    .filter((p) => p.hour <= currentHour && p.probability !== null)
                    .map((p) => `${(p.hour - 0.5) * 40},${130 - (p.probability || 0) * 120}`)
                    .join(" ")}
                />
              )}

              {/* Data points */}
              {timelinePoints
                .filter((p) => p.hour <= currentHour && p.probability !== null)
                .map((p) => (
                  <circle
                    key={p.hour}
                    cx={(p.hour - 0.5) * 40}
                    cy={130 - (p.probability || 0) * 120}
                    r={p.hour === currentHour ? "5" : "3.5"}
                    fill={p.alert === 1 ? "var(--alert)" : "var(--score)"}
                    stroke="#FFFFFF"
                    strokeWidth="1"
                  />
                ))}
            </svg>
          </div>
        </div>

        {/* Signal Readings Table / Grid */}
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left", color: "var(--muted)" }}>
                <th style={{ padding: "8px 12px", width: "140px" }}>Signal</th>
                <th style={{ padding: "8px 12px", width: "100px" }}>Normal</th>
                <th style={{ padding: "8px 12px", width: "130px" }}>At Hour {currentHour}</th>
                <th style={{ padding: "8px 12px" }}>Measurement State</th>
              </tr>
            </thead>
            <tbody>
              {[
                { key: "hr", label: "Heart Rate (HR)", unit: "bpm", normal: "60 - 100" },
                { key: "sbp", label: "Systolic BP (SBP)", unit: "mmHg", normal: "90 - 140" },
                { key: "resp", label: "Respiration (Resp)", unit: "br/min", normal: "12 - 20" },
                { key: "temp", label: "Temperature", unit: "°C", normal: "36.5 - 37.5" },
                { key: "wbc", label: "WBC Count", unit: "10³/µL", normal: "4.5 - 11.0" },
              ].map((sig) => {
                const val = currentPoint ? (currentPoint.values as any)[sig.key] : null;
                const isMissing = currentPoint?.missing?.includes(sig.key);
                return (
                  <tr key={sig.key} style={{ borderBottom: "1px solid var(--border-light)" }}>
                    <td style={{ padding: "10px 12px", fontWeight: 500, color: "var(--text)" }}>{sig.label}</td>
                    <td style={{ padding: "10px 12px", color: "var(--muted)", fontSize: "12px" }}>{sig.normal}</td>
                    <td style={{ padding: "10px 12px" }}>
                      <span className="mono" style={{ fontWeight: 600, color: val !== null ? "var(--text)" : "var(--missing)" }}>
                        {val !== null ? `${val} ${sig.unit}` : "—"}
                      </span>
                    </td>
                    <td style={{ padding: "10px 12px" }}>
                      {isMissing ? (
                        <span
                          style={{
                            display: "inline-block",
                            padding: "2px 8px",
                            borderRadius: "4px",
                            fontSize: "11px",
                            backgroundColor: "rgba(156, 163, 175, 0.15)",
                            color: "var(--missing)",
                          }}
                        >
                          Not Measured (Imputed via Causal Fill)
                        </span>
                      ) : (
                        <span
                          style={{
                            display: "inline-block",
                            padding: "2px 8px",
                            borderRadius: "4px",
                            fontSize: "11px",
                            backgroundColor: "rgba(59, 110, 165, 0.12)",
                            color: "var(--blue)",
                          }}
                        >
                          Directly Measured
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. Live Stream / Socket Testing Terminal */}
      <section
        style={{
          backgroundColor: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          padding: "24px",
          marginBottom: "32px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <h2 style={{ fontSize: "18px", color: "var(--text)", marginBottom: "4px" }}>
          Live Line Protocol Tester (Jury Verification)
        </h2>
        <p style={{ fontSize: "13px", color: "var(--muted)", marginBottom: "16px" }}>
          Simulates streaming an incoming <code className="mono">H|...</code> record through the line protocol. Observations are stored in SQLite and evaluated via <code className="mono">predict()</code>.
        </p>

        <form onSubmit={handleSendCustomLine} style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginBottom: "12px" }}>
          <input
            id={simulateInputId}
            type="text"
            value={incomingLine}
            onChange={(e) => setIncomingLine(e.target.value)}
            placeholder="H|P001|11|98|110|22|38.2|12.5"
            className="mono"
            style={{
              flex: 1,
              minWidth: "280px",
              padding: "9px 14px",
              borderRadius: "6px",
              border: "1px solid var(--border)",
              backgroundColor: "var(--bg)",
              color: "var(--text)",
              fontSize: "13px",
            }}
          />
          <button
            type="submit"
            disabled={isSubmittingLine}
            style={{
              padding: "9px 18px",
              borderRadius: "6px",
              backgroundColor: "var(--accent)",
              color: "#FFFFFF",
              border: "none",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            {isSubmittingLine ? "Processing..." : "Stream Record"}
          </button>
          <button
            type="button"
            onClick={() => setIncomingLine(`H|${selectedPatientId}|${maxHour + 1}|104|98||38.5|`)}
            style={{
              padding: "9px 14px",
              borderRadius: "6px",
              backgroundColor: "transparent",
              color: "var(--muted)",
              border: "1px solid var(--border)",
              fontSize: "13px",
            }}
          >
            Load Example with Empty Fields
          </button>
        </form>

        {socketReply && (
          <div
            className="mono"
            style={{
              padding: "10px 14px",
              borderRadius: "6px",
              backgroundColor: socketReply.startsWith("R|") ? "rgba(15, 118, 110, 0.08)" : "rgba(180, 95, 107, 0.12)",
              color: socketReply.startsWith("R|") ? "var(--accent)" : "var(--alert)",
              fontSize: "13px",
              border: "1px solid var(--border)",
            }}
          >
            <strong>Socket Reply:</strong> {socketReply}
          </div>
        )}
      </section>

      {/* 6. Evaluation Panel (Read from models/results/metrics.json) */}
      <section
        style={{
          backgroundColor: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          padding: "24px",
          marginBottom: "32px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.03)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "12px", flexWrap: "wrap", gap: "8px" }}>
          <div>
            <h2 style={{ fontSize: "18px", color: "var(--text)" }}>Held-Out Test Set Evaluation</h2>
            <p style={{ fontSize: "13px", color: "var(--muted)" }}>
              Numbers read directly from saved result files (<code className="mono">models/results/metrics.json</code>). Zero fabricated numbers.
            </p>
          </div>
          <span style={{ fontSize: "12px", color: "var(--muted)" }}>Patient-level split · Evaluated on held-out test cohort</span>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left", color: "var(--muted)" }}>
                <th style={{ padding: "8px 12px" }}>Model</th>
                <th style={{ padding: "8px 12px" }}>AUPRC (Primary)</th>
                <th style={{ padding: "8px 12px" }}>Precision</th>
                <th style={{ padding: "8px 12px" }}>Recall</th>
                <th style={{ padding: "8px 12px" }}>Specificity</th>
                <th style={{ padding: "8px 12px" }}>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--border-light)" }}>
                  <td style={{ padding: "10px 12px", fontWeight: 600, color: "var(--text)" }}>{m.name}</td>
                  <td className="mono" style={{ padding: "10px 12px", color: "var(--score)", fontWeight: 600 }}>{typeof m.auprc === "number" ? m.auprc.toFixed(4) : m.auprc}</td>
                  <td className="mono" style={{ padding: "10px 12px" }}>{typeof m.precision === "number" ? m.precision.toFixed(4) : m.precision}</td>
                  <td className="mono" style={{ padding: "10px 12px" }}>{typeof m.recall === "number" ? m.recall.toFixed(4) : m.recall}</td>
                  <td className="mono" style={{ padding: "10px 12px" }}>{typeof m.specificity === "number" ? m.specificity.toFixed(4) : m.specificity}</td>
                  <td className="mono" style={{ padding: "10px 12px" }}>{typeof m.threshold === "number" ? m.threshold.toFixed(4) : m.threshold}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 7. Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border)",
          paddingTop: "20px",
          display: "flex",
          justifyContent: "space-between",
          color: "var(--muted)",
          fontSize: "12px",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div>
          <strong>SepsisSignal</strong> · Team HackerSIM (Soham Paranjape, Ishaan Patil, Mitansh Bheda)
        </div>
        <div>
          Dataset: PhysioNet/Computing in Cardiology Challenge 2019 (CC BY 4.0). Reyna et al., Crit Care Med 2020.
        </div>
      </footer>
    </div>
  );
}
