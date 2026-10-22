"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, ChartNoAxesCombined } from "lucide-react";
import type { ReactNode } from "react";
import { DashboardStateProvider } from "./DashboardState";
import Image from "next/image";

const navigation = [
  { href: "/", label: "Overview", Icon: Activity },
  { href: "/evaluation", label: "Evaluation", Icon: ChartNoAxesCombined },
];

export default function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex h-screen flex-col overflow-hidden md:flex-row">
      <aside className="w-full shrink-0 bg-app px-5 py-5 md:sticky md:top-0 md:flex md:h-screen md:w-[210px] md:flex-col md:px-5 md:py-7">
        <Link href="/" className="inline-flex items-center gap-3 rounded-panel focus-visible:outline-2 focus-visible:outline-accent">
          <Image src="/logo.png" alt="SepsisSignal Logo" width={32} height={32} className="h-8 w-8 rounded-[10px]" />
          <span className="text-[17px] font-semibold tracking-tighter">SepsisSignal</span>
        </Link>

        <nav aria-label="Main navigation" className="mt-10 flex gap-1 max-md:mt-4 max-md:overflow-x-auto md:flex-col">
          {navigation.map(({ href, label, Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`flex min-h-11 min-w-fit items-center gap-3 rounded-panel px-3 text-[13px] font-medium transition-colors max-md:flex-1 max-md:justify-center ${
                  active ? "bg-app text-accent" : "text-muted hover:bg-app hover:text-ink"
                }`}
              >
                <Icon size={17} strokeWidth={1.8} aria-hidden="true" />
                {label}
              </Link>
            );
          })}
        </nav>

      </aside>

      <div className="min-h-0 min-w-0 flex-1 p-3 max-md:p-2">
        <div className="dashboard-scroll h-full overflow-y-auto overscroll-contain rounded-panel bg-surface shadow-md">
          <div className="bg-notice text-notice-ink" role="note">
            <div className="mx-auto flex max-w-[1200px] flex-wrap justify-between gap-x-6 gap-y-1.5 px-8 py-2.5 text-[13px] leading-[1.5] max-[860px]:px-5">
              <strong className="font-semibold">Research prototype, not for clinical decision-making.</strong>
              <span className="opacity-90">Temperature assumes Celsius (°C).</span>
            </div>
          </div>
          <DashboardStateProvider>{children}</DashboardStateProvider>
        </div>
      </div>
    </div>
  );
}
