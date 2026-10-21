"use client";

import { createContext, useContext, useState, type ReactNode } from "react";

const DashboardStateContext = createContext<{
  selectedPatientId: string;
  setSelectedPatientId: (patientId: string) => void;
  currentHour: number;
  setCurrentHour: (hour: number | ((current: number) => number)) => void;
} | null>(null);

export function DashboardStateProvider({ children }: { children: ReactNode }) {
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [currentHour, setCurrentHour] = useState(1);

  return (
    <DashboardStateContext.Provider
      value={{ selectedPatientId, setSelectedPatientId, currentHour, setCurrentHour }}
    >
      {children}
    </DashboardStateContext.Provider>
  );
}

export function useDashboardState() {
  const state = useContext(DashboardStateContext);
  if (!state) throw new Error("Dashboard state must be used inside DashboardStateProvider");
  return state;
}
