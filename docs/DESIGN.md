# Design

Applies to the dashboard, README screenshots and slides so everything looks like one product.
Research prototype, not for clinical decision-making.

## Direction
Calm, editorial, data-first. Plenty of white space, one idea per screen, thin hairline rules.
Looks hand-made by a student team, not like a generic template.
No gradients, glassmorphism, drop-shadow cards, emoji, stock icon grids or decorative blobs.
No alarm styling: avoid saturated red and flashing elements. A model alert is not a clinical alarm.

## Color tokens
| Token | Use | Hex |
|---|---|---|
| `--bg` | Page background | #F7F9FB (or paper #FAF9F6) |
| `--text` | Primary text | #1A1A1A |
| `--muted` | Secondary text, rules | #6B7280 |
| `--accent` | Primary accent (teal) | #0F766E |
| `--blue` | Secondary accent | #3B6EA5 |
| `--missing` | Not measured | #9CA3AF |
| `--imputed` | Filled value | #D9A441 |
| `--alert` | Threshold line and alert state (muted rose) | #B45F6B |
| `--notice-bg` / `--notice-text` | Disclaimer banner | #FEF3C7 / #78350F |

Keep the meaning of each color the same everywhere: measured, missing, imputed, model estimate.
Model risk uses `--accent`; there is no separate score colour.
Define dark values under `prefers-color-scheme: dark`. Check text contrast.

## Typography
- Headlines: a clean serif (DM Sans)
- Body: DM Sans
- Accents/Labels (use instead of monospaced/serif fonts): Rubik
- Numbers, probabilities, hours, metrics: Rubik
- Typographic apostrophes and quotes in all visible text.

## Surfaces and controls
One shape language for cards, tables, fields and buttons: **10px radius**, 1px `--border`,
`--surface` fill, no shadows.
- **Tables:** the rounded border wraps the whole table; the table title sits outside it.
  Header cells use `--bg`, row dividers use `--border-light`, and rows get a `--bg` hover.
  The model-estimate chart uses the same card shell; its label and hour range sit outside, w-fit and centered.
- **Inputs:** 42px minimum height, 14px horizontal padding, same 10px radius and border.
  Labels sit above the field, with units in parentheses.
- **Selects:** a separate custom component (`dashboard/app/components/Select.tsx`), never the native
  control. The trigger matches the inputs; its chevron sits 14px from the right edge. The listbox popover
  uses the same 10px radius and 1px border, scrolls past 260px, highlights the active option with `--bg`,
  and marks the selected option with a check. Keyboard: arrows, Home/End, Enter/Space, Escape, Tab.
  The trigger keeps the same focus ring visible while its popover is open.
- **Buttons:** same 10px radius and 1px border weight as the fields.
- **Focus:** solid 2px `--border` ring, flush with the control (no offset, no accent colour).
- Do not use pill shapes, filled grey fields, floating labels, inner shadows or drop shadows.

## Dashboard layout
The disclaimer banner spans the full viewport width and sits flush at the top, with no outer padding.
1. **Header:** SepsisSignal wordmark, live-path label, patient picker.
   Label text: "Live path: Logistic Regression baseline (research prototype)".
2. **Clock:** slider or scrubber for hour T, step buttons (previous, next), current hour in mono type.
   At hour T show the probability (mono, large) and the alert state with a plain word ("alert" or "no alert").
3. **Timeline:** a model-estimate card and a vitals table (sparkline, typical range, current value, state).
   Missing points are marked and imputed values are distinct. Only data up to the selected hour is drawn,
   so the chart cannot show the future.
4. **Legend:** measured, missing, imputed, model estimate, benchmark label (retrospective only).
5. **Evaluation panel:** held-out comparison table read from `models/results/`, never typed by hand.
6. **Footer on every page:** "Research prototype, not for clinical decision-making". Dataset citation.

## Behavior
- Moving the clock updates probability and alert from `predict`, never from a precomputed file.
- Hours after T are greyed out or hidden.
- Sending a line over the socket makes that patient and hour appear without a reload (poll or push).
- Empty states are plain sentences ("No data received for this patient yet").
- Keep motion minimal: a short transition on the value change is enough.
- Honest copy: say "estimate", "benchmark label" and "research prototype". Never say "diagnosis" or "detects sepsis".

## Slides
- Same palette and fonts as the dashboard, 10 slides maximum.
- One idea per slide, varied layouts, big screenshot frames, short plain text.
- Footer on every slide: "Research prototype, not for clinical decision-making".
- Business and competitor numbers must be sourced, or left out.

## Accessibility
- Never rely on color alone: use shape, dash style or label for missing, imputed and alert states.
- Keyboard support for the clock (arrow keys) and visible focus states.
- Responsive: charts scroll inside their own container, page body never scrolls sideways.
