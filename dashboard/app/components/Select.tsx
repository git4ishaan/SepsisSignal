"use client";

import React, { useEffect, useId, useRef, useState } from "react";

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps {
  id?: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  ariaLabel?: string;
  placeholder?: string;
  disabled?: boolean;
}

export default function Select({
  id,
  value,
  options,
  onChange,
  ariaLabel,
  placeholder = "Select an option",
  disabled = false,
}: SelectProps) {
  const generatedId = useId();
  const triggerId = id ?? generatedId;
  const listboxId = `${triggerId}-listbox`;

  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const selectedIndex = options.findIndex((option) => option.value === value);
  const selected = selectedIndex >= 0 ? options[selectedIndex] : null;
  const isDisabled = disabled || options.length === 0;

  function openMenu() {
    if (isDisabled) return;
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0);
    setOpen(true);
  }

  function closeMenu() {
    setOpen(false);
    setActiveIndex(-1);
  }

  function choose(index: number) {
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    closeMenu();
    triggerRef.current?.focus();
  }

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) closeMenu();
    }
    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [open]);

  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!open || activeIndex < 0) return;
    const option = listRef.current?.querySelector<HTMLElement>(`[data-index="${activeIndex}"]`);
    option?.scrollIntoView({ block: "nearest" });
  }, [open, activeIndex]);

  function onTriggerKeyDown(event: React.KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openMenu();
    }
  }

  function onListKeyDown(event: React.KeyboardEvent<HTMLUListElement>) {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setActiveIndex((index) => Math.min(options.length - 1, index + 1));
        break;
      case "ArrowUp":
        event.preventDefault();
        setActiveIndex((index) => Math.max(0, index - 1));
        break;
      case "Home":
        event.preventDefault();
        setActiveIndex(0);
        break;
      case "End":
        event.preventDefault();
        setActiveIndex(options.length - 1);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        choose(activeIndex);
        break;
      case "Escape":
        event.preventDefault();
        closeMenu();
        triggerRef.current?.focus();
        break;
      case "Tab":
        closeMenu();
        triggerRef.current?.focus();
        break;
      default:
        break;
    }
  }

  return (
    <div className="relative inline-flex min-w-0" ref={rootRef}>
      <button
        type="button"
        id={triggerId}
        ref={triggerRef}
        className="group relative flex min-h-[42px] min-w-[250px] max-w-full cursor-pointer items-center justify-between gap-3 rounded-panel border border-rule bg-surface py-[9px] pr-[38px] pl-3.5 text-left text-sm text-ink transition-colors aria-expanded:outline-2 aria-expanded:outline-rule aria-expanded:outline-offset-0 disabled:cursor-not-allowed disabled:border-rule-soft disabled:text-missing"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        disabled={isDisabled}
        onClick={() => (open ? closeMenu() : openMenu())}
        onKeyDown={onTriggerKeyDown}
      >
        <span className="min-w-0 truncate">{selected ? selected.label : placeholder}</span>
        <svg
          className="pointer-events-none absolute top-1/2 right-3.5 h-3 w-3 -translate-y-1/2 text-muted transition-transform group-aria-expanded:rotate-180"
          viewBox="0 0 12 12"
          aria-hidden="true"
          focusable="false"
        >
          <path
            d="M2.5 4.5 6 8l3.5-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {open && (
        <ul
          id={listboxId}
          ref={listRef}
          className="absolute top-[calc(100%_+_6px)] left-0 z-30 max-h-[260px] w-max min-w-full max-w-[min(360px,calc(100vw_-_40px))] list-none overflow-y-auto overscroll-contain rounded-panel border border-rule bg-surface p-1 animate-[open_0.12s_ease-out] focus-visible:outline-none"
          role="listbox"
          tabIndex={-1}
          aria-label={ariaLabel ?? "Options"}
          aria-activedescendant={
            activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined
          }
          onKeyDown={onListKeyDown}
        >
          {options.map((option, index) => (
            <li
              key={option.value}
              id={`${listboxId}-option-${index}`}
              data-index={index}
              data-active={index === activeIndex}
              className="flex cursor-pointer items-center justify-between gap-3 rounded-lg px-2.5 py-[9px] text-[13px] leading-[1.35] text-ink data-[active=true]:bg-app aria-selected:font-medium"
              role="option"
              aria-selected={option.value === value}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => choose(index)}
            >
              <span className="min-w-0 truncate">{option.label}</span>
              <span className="min-w-3 flex-none text-right font-label text-xs text-muted" aria-hidden="true">
                {option.value === value ? "✓" : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
