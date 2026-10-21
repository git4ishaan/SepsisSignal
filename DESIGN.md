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
| `--score` | Model risk line | #4F46E5 |
| `--missing` | Not measured | #9CA3AF |
| `--imputed` | Filled value | #D9A441 |
| `--alert` | Threshold line and alert state (muted rose) | #B45F6B |
| `--notice-bg` / `--notice-text` | Disclaimer banner | #FEF3C7 / #78350F |

Keep the meaning of each color the same everywhere: measured, missing, imputed, model estimate.
Define dark values under `prefers-color-scheme: dark`. Check text contrast.

## Typography
- Headlines: a clean serif (DM Serif Display or Fraunces)
- Body: Inter
- Numbers, probabilities, hours, metrics: JetBrains Mono
- Sizes: titles 32 to 36, body 16 to 20 on screen, nothing below 12. Left aligned.
- Typographic apostrophes and quotes in all visible text.

## Dashboard layout
1. **Header:** SepsisSignal wordmark, live-path label, patient picker.
   Label text: "Live path: Logistic Regression baseline (research prototype)".
2. **Clock:** slider or scrubber for hour T, step buttons (previous, next), current hour in mono type.
   At hour T show the probability (mono, large) and the alert state with a plain word ("alert" or "no alert").
3. **Timeline:** vitals over time with missing points marked, imputed values distinct, risk score line with a
   threshold line. Only data up to the selected hour is drawn, so the chart cannot show the future.
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
