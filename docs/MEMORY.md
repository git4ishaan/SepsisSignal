# Project Memory

Persistent notes for the agent and the team. Read at the start of every session.
Append dated entries at the bottom of each section. Do not delete history, mark old entries "superseded".
Keep each entry to one or two lines. Never store raw patient data, credentials or API keys here.

## Project in one paragraph
SepsisSignal is a research prototype for early sepsis risk estimation on the public PhysioNet 2019 ICU dataset.
Team HackerSIM: Soham Paranjape, Ishaan Patil, Mitansh Bheda.
Round 2 twist: store patient-hours in SQLite, compute risk causally per hour (rows <= T only),
answer a line protocol over TCP or WebSocket, and drive the dashboard clock from the same predict path.
Final round: 22 October, in person at the college. Research prototype, not for clinical decision-making.

## Fixed constraints (do not change without asking)
- Patient-level split is preserved. Predictions for a patient come from a model trained on other patients.
- A request for hour T uses only that patient's rows with hour <= T.
- Past predictions are immutable.
- Live path is the Logistic Regression baseline and is labeled on screen.
- Jury input `H|<patient id>|<hour>|<HR>|<SBP>|<Resp>|<Temp>|<WBC>`; output
  `R|<patient id>|<hour>|<probability>|<alert 0 or 1>|<count of empty fields>`.
- Raw dataset lives outside the repo. No patient data goes to any external service.
- No invented metrics. Numbers come from saved result files.
- Slide deck: 10 slides maximum.

## Decisions log
Format: `YYYY-MM-DD | decision | reason`
- 2026-10-21 | Live path uses Logistic Regression baseline | Twist allows it if labeled; lowest risk in 24 hours
- 2026-10-21 | Sequence model stays offline comparison only | Not worth the risk in the live path this round
- 2026-10-21 | 5-signal feature mapping defined in features.py | Causal forward-fill, missing flag, time-since, delta (22 total features)
- 2026-10-21 | Threshold set to 0.2500 on validation split | Balances sensitivity (target recall ~0.80) with low alert burden
- 2026-10-21 | TCP line server with asyncio implemented | Robust parsing, non-stripping of trailing pipes, E|... error lines
- 2026-10-21 | FastAPI endpoints and Next.js clock dashboard built | Real-time scrubber, causal timeline, live stream tester, metrics table

## Current state
Update this block, do not append to it.
- Done: Block 1 (Foundation), Block 2 (Core line server `src/server.py`), Block 3 (Tests), Block 4 (FastAPI backend `api/main.py` + Next.js Dashboard `dashboard/`)
- In progress: Block 5 (Final rehearsal, pitch preparation, presentation readiness)
- Blocked: None
- Next: End-to-end dry run (Server + API + Dashboard) and presentation rehearsal
- Last verified acceptance tests: 2026-10-21: 7/7 acceptance tests passed in 1.64s. Next.js production build succeeded in 2.0s with zero errors.




## Known issues and gotchas
- Trailing pipes matter: `H|P001|2|88|112|||` has 8 fields and 3 empty signals.
- Empty count is taken from the incoming record, before imputation.
- Live path sees five signals only, so probabilities can differ from full-feature offline results.
- [add as you find them]

## Open questions for the team
- [ ] 

## Things learned (for the pitch and Q&A)
- [one-line findings worth saying to the jury, e.g. what parity testing showed]

## Submission checklist
- [ ] GitHub link submitted on the Google Form
- [ ] Joined the official WhatsApp group
- [ ] README updated with real commands, ports and results
- [ ] Slides updated, 10 or fewer, disclaimer on every slide
- [ ] Demo backup recording saved
