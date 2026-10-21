import type { Metadata } from "next";
import "./globals.css";

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
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
