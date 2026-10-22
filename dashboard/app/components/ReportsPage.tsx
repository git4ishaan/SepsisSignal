"use client";

import { useEffect, useState } from "react";
import { FileText, Search, Trash2, X } from "lucide-react";
import ShiftReportModal from "./ShiftReportModal";
import { readSavedReports, removeSavedReport, type SavedReport } from "./reportStorage";

const TABLE_WRAP = "mt-5 overflow-x-auto rounded-panel border border-rule bg-surface";
const TABLE = "w-full border-collapse text-[13px]";
const TABLE_TH =
  "whitespace-nowrap border-b border-rule bg-app px-4 py-2.5 text-left font-label text-[10.5px] font-medium uppercase tracking-[0.06em] text-muted";
const TABLE_CELL = "border-b border-rule-soft px-4 py-3 text-left align-middle";

type SortMode = "newest" | "oldest" | "patient" | "risk" | "hour";

export default function ReportsPage() {
  const [reports, setReports] = useState<SavedReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<SavedReport | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("newest");

  useEffect(() => {
    setReports(readSavedReports());
  }, []);

  function deleteReport(id: string) {
    setReports(removeSavedReport(id));
    if (selectedReport?.id === id) setSelectedReport(null);
  }

  const normalizedSearch = searchTerm.trim().toLocaleLowerCase();
  const visibleReports = reports
    .filter((report) => {
      if (!normalizedSearch) return true;
      let status = "no estimate";
      if (report.prediction) status = report.prediction.alert === 1 ? "alert" : "no alert";
      const searchable = [
        report.patientId,
        report.split,
        `hour ${report.currentHour}`,
        String(report.currentHour),
        status,
        report.prediction?.probability.toFixed(4) ?? "",
        new Date(report.generatedAt).toLocaleString(),
      ].join(" ").toLocaleLowerCase();
      return searchable.includes(normalizedSearch);
    })
    .sort((a, b) => {
      switch (sortMode) {
        case "oldest":
          return Date.parse(a.generatedAt) - Date.parse(b.generatedAt);
        case "patient":
          return a.patientId.localeCompare(b.patientId, undefined, { numeric: true }) || b.currentHour - a.currentHour;
        case "risk":
          return (b.prediction?.probability ?? -1) - (a.prediction?.probability ?? -1);
        case "hour":
          return b.currentHour - a.currentHour || Date.parse(b.generatedAt) - Date.parse(a.generatedAt);
        case "newest":
        default:
          return Date.parse(b.generatedAt) - Date.parse(a.generatedAt);
      }
    });

  return (
    <>
      <main className="mx-auto max-w-[1080px] px-8 pt-8 pb-[72px] max-[860px]:px-5 max-[860px]:pb-14">
        <header className="flex min-h-[76px] items-center justify-between gap-6 py-4 max-[600px]:flex-wrap">
          <div>
            <h1 className="text-[30px] font-semibold leading-none tracking-[-0.04em]">Reports</h1>
            <p className="mt-2 text-[13px] text-muted">Generated patient-hour summaries saved on this device.</p>
          </div>
        </header>

        {reports.length === 0 ? (
          <div className="mt-8 border-t border-rule pt-6">
            <div className="flex items-start gap-3">
              <FileText className="mt-0.5 text-muted" size={18} strokeWidth={1.8} aria-hidden="true" />
              <div>
                <h2 className="text-sm font-medium">No reports yet</h2>
                <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-muted">
                  Generate a report from the Overview page. It will appear here so you can reopen or print it later.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="mt-5 flex flex-wrap items-end justify-between gap-3">
              <div className="relative min-w-[220px] flex-1 sm:max-w-sm">
                <Search className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-muted" size={15} aria-hidden="true" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                  placeholder="Search reports"
                  aria-label="Search reports by patient, hour, status, risk, or date"
                  className="min-h-[42px] w-full rounded-panel border border-rule bg-surface pr-9 pl-9 text-[13px] text-ink placeholder:text-muted focus-visible:outline-2 focus-visible:outline-rule"
                />
                {searchTerm && (
                  <button
                    type="button"
                    onClick={() => setSearchTerm("")}
                    className="absolute top-1/2 right-2 inline-flex h-7 w-7 -translate-y-1/2 cursor-pointer items-center justify-center rounded-panel text-muted hover:bg-app hover:text-ink"
                    aria-label="Clear report search"
                  >
                    <X size={14} aria-hidden="true" />
                  </button>
                )}
              </div>
              <label className="flex items-center gap-2 text-xs text-muted">
                Sort
                <select
                  value={sortMode}
                  onChange={(event) => setSortMode(event.target.value as SortMode)}
                  className="min-h-[42px] cursor-pointer rounded-panel border border-rule bg-surface px-3 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-rule"
                  aria-label="Sort reports"
                >
                  <option value="newest">Newest first</option>
                  <option value="oldest">Oldest first</option>
                  <option value="patient">Patient</option>
                  <option value="risk">Highest risk</option>
                  <option value="hour">Highest hour</option>
                </select>
              </label>
            </div>

            <p className="mt-3 text-[11px] text-muted" aria-live="polite">
              Showing {visibleReports.length} of {reports.length} {reports.length === 1 ? "report" : "reports"}
            </p>

            {visibleReports.length > 0 ? (
              <div className={TABLE_WRAP}>
                <table className={TABLE}>
                  <thead>
                    <tr>
                      <th scope="col" className={TABLE_TH}>Generated</th>
                      <th scope="col" className={TABLE_TH}>Patient</th>
                      <th scope="col" className={TABLE_TH}>Hour</th>
                      <th scope="col" className={TABLE_TH}>Risk</th>
                      <th scope="col" className={TABLE_TH}>Status</th>
                      <th scope="col" className={`${TABLE_TH} text-right`}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleReports.map((report) => (
                      <tr key={report.id} className="transition-colors hover:bg-app [&:last-child>th]:border-b-0 [&:last-child>td]:border-b-0">
                        <td className={`${TABLE_CELL} whitespace-nowrap text-muted`}>
                          {new Date(report.generatedAt).toLocaleString()}
                        </td>
                        <th scope="row" className={`${TABLE_CELL} font-medium`}>{report.patientId || "Unknown"}</th>
                        <td className={`${TABLE_CELL} tabular-nums`}>{report.currentHour}</td>
                        <td className={`${TABLE_CELL} font-label tabular-nums`}>
                          {report.prediction ? report.prediction.probability.toFixed(4) : "—"}
                        </td>
                        <td className={`${TABLE_CELL} ${report.prediction?.alert === 1 ? "font-medium text-alert" : "text-muted"}`}>
                          {report.prediction ? (report.prediction.alert === 1 ? "Alert" : "No alert") : "No estimate"}
                        </td>
                        <td className={`${TABLE_CELL} text-right`}>
                          <div className="inline-flex items-center gap-1">
                            <button
                              type="button"
                              className="min-h-9 cursor-pointer rounded-panel px-2.5 text-xs font-medium text-accent hover:bg-app focus-visible:outline-2 focus-visible:outline-rule"
                              onClick={() => setSelectedReport(report)}
                            >
                              Open
                            </button>
                            <button
                              type="button"
                              className="inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-panel text-muted hover:bg-app hover:text-alert focus-visible:outline-2 focus-visible:outline-rule"
                              onClick={() => deleteReport(report.id)}
                              aria-label={`Delete report for ${report.patientId}, hour ${report.currentHour}`}
                            >
                              <Trash2 size={15} strokeWidth={1.8} aria-hidden="true" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="mt-4 border-t border-rule pt-5">
                <p className="text-[13px] text-muted">No reports match “{searchTerm}”.</p>
              </div>
            )}
          </>
        )}
      </main>

      {selectedReport && (
        <ShiftReportModal
          isOpen
          onClose={() => setSelectedReport(null)}
          patientId={selectedReport.patientId}
          currentHour={selectedReport.currentHour}
          maxHour={selectedReport.maxHour}
          split={selectedReport.split}
          prediction={selectedReport.prediction}
          timelinePoints={selectedReport.timelinePoints}
          generatedAt={selectedReport.generatedAt}
        />
      )}
    </>
  );
}
