import type { ShiftReportData } from "./ShiftReportModal";

const STORAGE_KEY = "sepsis-signal-generated-reports";

export interface SavedReport extends ShiftReportData {
  id: string;
  generatedAt: string;
}

export function readSavedReports(): SavedReport[] {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is SavedReport =>
        typeof item === "object" && item !== null &&
        typeof item.id === "string" && typeof item.generatedAt === "string" &&
        typeof item.patientId === "string" && typeof item.currentHour === "number" &&
        Array.isArray(item.timelinePoints)
    );
  } catch {
    return [];
  }
}

export function saveReport(report: SavedReport): void {
  const reports = readSavedReports();
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify([report, ...reports]));
}

export function removeSavedReport(id: string): SavedReport[] {
  const reports = readSavedReports().filter((report) => report.id !== id);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
  return reports;
}
