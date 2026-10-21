import type { Metadata } from "next";
import "./globals.css";
import { DM_Sans, Rubik } from "next/font/google";
import DashboardShell from "./components/DashboardShell";

const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-dm-sans" });
const rubik = Rubik({ subsets: ["latin"], variable: "--font-rubik" });

export const metadata: Metadata = {
  title: "SepsisSignal — Explainable ICU Early-Warning Research Prototype",
  description: "Research prototype for studying early sepsis risk estimation on ICU time-series. Not for clinical decision-making.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${dmSans.variable} ${rubik.variable}`}>
      <body className="min-h-screen overflow-x-hidden bg-app font-sans leading-[1.5] text-ink antialiased">
        <DashboardShell>{children}</DashboardShell>
      </body>
    </html>
  );
}
