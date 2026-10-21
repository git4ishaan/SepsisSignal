# Tasks: Round 2 Sprint (24 hours)

Team HackerSIM: Soham Paranjape, Ishaan Patil, Mitansh Bheda
Final round: 22 October, report to the college. Research prototype, not for clinical decision-making.

## Do right now
- [ ] Submit the GitHub repo link on the Google Form (mandatory)
- [ ] At least one teammate joins the official WhatsApp group
- [x] Put the dataset OUTSIDE the repo folder; confirm `data/` is in .gitignore

## Owners (fill in)
- Pipeline and features: Ishaan Patil / Team HackerSIM
- Model, predict function and tests: Ishaan Patil / Team HackerSIM
- Server, API and dashboard clock: Ishaan Patil / Team HackerSIM

## Definition of done (jury checks)
1. `H|P001|2|88|112|||` returns an empty-field count greater than zero.
2. Sending hour 3 does not change the stored risk for hour 2.
3. The dashboard clock shows the same probability and alert as the socket reply.
4. Live path is labeled "Logistic Regression baseline (research prototype)".
5. Held-out patient comparison is still shown.

## Order of work

### Block 1: Foundation (first 3 to 4 hours)
- [x] Locate the existing trained model, imputer values, threshold and split files. Copy into `models/`.
- [x] Write `src/features.py`: ONE point-in-time feature function using only rows with hour <= T.
- [x] Check training code for look-ahead: backfill, whole-stay rolling stats, scalers fit on more than train.
- [x] Decide the five-signal mapping in `docs/DATA_CONTRACT.md` and list features that become "unmeasured".
- [x] If the full-feature model cannot run on five signals, decide: treat extras as unmeasured, or retrain a reduced Logistic Regression on the same patient split (state it on screen).

### Block 2: Core (next 4 to 5 hours)
- [x] `src/store.py`: SQLite with WAL, `observations` and `predictions` tables.
- [x] `src/predict.py`: `predict(patient_id, hour)` reads rows <= hour, computes, stores once, returns stored value on repeat.
- [x] `src/server.py`: TCP (or WebSocket) line server: parse, count empties, call predict, reply `R|...`.
- [x] Error handling: malformed line returns `E|...`, server stays up, `\r\n` accepted.

### Block 3: Tests (before building more UI)
- [x] Test A: hours 1, 2, 3 for one patient. Hour 2 reply identical before and after hour 3.
- [x] Test B: `H|P001|2|88|112|||` returns empty count 3.
- [x] Test C: stream a held-out patient hour by hour, compare with offline batch features, record max difference.
- [x] Test D: unseen patient ID works; first hour with no history works; duplicate hour returns stored value.
- [x] Manual check with `nc localhost <port>` as the jury might.


### Block 4: Dashboard (next 4 to 5 hours)
- [x] FastAPI: `/predict`, `/patients`, `/timeline` as in `docs/DATA_CONTRACT.md`.
- [x] Add the clock (slider or scrubber) to the existing dashboard. It calls the same predict path.
- [x] Show live-path label, missing vs imputed legend, threshold line, disclaimer.
- [x] Patients sent through the socket appear in the dashboard.
- [x] Held-out comparison view still works (test-split patients only).
- [x] Remove any code path that reads a precomputed per-stay score file.


### Block 5: Presentation (last 4 to 5 hours, leave buffer)
- [ ] Update slides: causal per-hour inference, immutable past predictions, live-path label, held-out comparison.
- [ ] Remove unsourced business numbers; keep the honest-limits framing.
- [ ] Fill real metrics from `models/results/` (never typed by hand).
- [ ] Rehearse the 3 jury checks live, in the order the jury listed them.
- [ ] Prepare backup: short screen recording of the demo, and a fallback if the port is blocked.
- [ ] Update README (setup, run commands, limits, citation, AI tools used).

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| Features leak future rows | Rewrite features through the single point-in-time function; rerun Test C |
| Model needs more than 5 signals | Treat extras as unmeasured, or retrain reduced model on the same split and say so |
| Probabilities look low vs offline | Explain: live path sees 5 signals only; report openly |
| WebSocket trouble | Use plain TCP |
| Port blocked at the venue | Pick another local port; test before the demo |
| Out of time | Ship baseline live path + dashboard clock first; skip extras |

## Notes / unfinished (agent writes here)
- [ ] 
