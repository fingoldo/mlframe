# mlframe `reporting/` audit — master tracker

Read-only parallel audit of `src/mlframe/reporting/**` (90 files, ~22.4k LOC): 5 subsystem clusters plus 1 cross-cutting consumer-value/UI specialist.

Audit dimensions per cluster: correctness, computational efficiency, architecture and clarity, **informativeness and consumer value to a data scientist**, **UI/UX best practices**, edge cases, test-coverage gaps, OSS hygiene.

Every finding gets an explicit, non-silent disposition per the project's multi-agent-review convention: **RESOLVED** (implemented), **FUTURE** (with reason), **DOC** (documented rather than changed), or **REJECTED** (with reason). Nothing is dropped for being low severity. Status starts at TODO.

A cluster file moves to `implemented/` only once EVERY finding in it carries a final disposition.

## Per-cluster status

| Cluster | Scope | Report | Findings | Status |
|---|---|---|---|---|
| reporting_core | `spec.py`, `output.py`, `colors.py`, `catalog.py`, `auto_dispatch.py`, `diagnostics_dispatch.py`, `report_html.py`, `_benchmarks/` | [implemented/reporting_core.md](implemented/reporting_core.md) | 30 (0/0/8/22) | **COMPLETE** (22 RESOLVED, 6 FUTURE, 1 DOC, 1 REJECTED) |
| reporting_renderers | `renderers/**` (matplotlib, plotly, kaleido, save dispatch, shared helpers) | [reporting_renderers.md](reporting_renderers.md) | 29 (0/5/9/15) | IN PROGRESS (all 5 P1 RESOLVED) |
| reporting_charts_a | `charts/` shared kernels + binary/calibration family | [reporting_charts_a.md](reporting_charts_a.md) | 36 (0/4/15/17) | IN PROGRESS (all 4 P1 RESOLVED) |
| reporting_charts_b | `charts/` class-structure through multilabel (incl. `model_card`, `decision_curve`) | [implemented/reporting_charts_b.md](implemented/reporting_charts_b.md) | 68 (0/11/34/23) | **COMPLETE** (61 RESOLVED, 6 FUTURE, 1 DOC) -> [implemented/](implemented/reporting_charts_b.md) |
| reporting_charts_c | `charts/` pdp through training_curve (incl. `risk_coverage`, `slice_finder`) | [reporting_charts_c.md](reporting_charts_c.md) | 40 (0/2/22/16) | IN PROGRESS (both P1 RESOLVED) |
| reporting_ux_crosscutting | repo-wide caption inventory, verdict surfacing, degenerate cases, tooltips, colour accessibility, backend parity | [reporting_ux_crosscutting.md](reporting_ux_crosscutting.md) | 74 (0/4/41/29) | IN PROGRESS (1 RESOLVED) |

**Every P1 in the campaign is now RESOLVED (26 of 26).** The remaining work is P2/P3.

**Total: 277 findings — 0 P0, 26 P1, 138 P2, 113 P3.** Counts are `(P0/P1/P2/P3)`.

## Disposition legend

- **RESOLVED** — fixed in code, with a regression test that fails before the fix and passes after.
- **FUTURE** — real but deferred; the reason and the trigger for revisiting are written next to the finding.
- **DOC** — the behaviour is correct as-is and the finding is answered by documenting WHY, in the code.
- **REJECTED** — not a real problem; the disproof is written next to the finding.

## Already fixed before this audit started

Recorded here so the audit does not re-report them and so the tracker is a complete record of the campaign.

| Item | Where |
|---|---|
| plotly figures rendered 20% smaller than the matplotlib twin (80 vs 100 px/inch) | `renderers/plotly.py` |
| Panel titles collided with the suptitle (top margin sized from the suptitle alone) | `renderers/plotly.py` |
| Bar tick labels truncated at 24 chars in plotly only, matplotlib never truncated | `renderers/plotly.py` |
| Panel-title wrap width was a flat 46 chars regardless of panel width; matplotlib also collapsed explicit line breaks | `renderers/_shared_helpers.py` |
| Time axes rendered raw epoch nanoseconds (`1.62e18`) in both backends | `renderers/_shared_helpers.py` |
| matplotlib figures were also shown inline in notebooks, duplicating every chart | `renderers/save.py` |
| Save-only backends were rendered even when nothing was being saved | `renderers/save.py` |
| Single labelled panel had no legend in interactive HTML | `renderers/plotly.py` |
| Heatmap tooltips showed raw `x`/`y`/`z` and an internal trace id | `renderers/plotly.py` |
| 2-D PDP cells gave no support count, so unsupported cells looked identical to well-supported ones | `charts/pdp_ice.py` |
| Decision-curve usefulness margin was a flat `1e-3`, so a random score was declared USEFUL at n=2000 | `charts/decision_curve.py` |
| Decision curve had no how-to-read caption or verdict text | `charts/decision_curve.py` |
| WoE chart rendered empty axes plus a duplicated title when nothing cleared `min_support` | `charts/category_discriminability.py` |


## Resolved so far

| Finding | What was wrong | Fix |
|---|---|---|
| REPORTING_CORE-1 | The feature-column cap probed a pandas-only attribute, so it silently no-opped on every polars frame and the dense-matrix builders received ALL columns. | Probe the capability (`.columns` + a select attempt), not the attribute. Verified: polars and pandas now both cap to the requested width. |
| REPORTING_CORE-2 | The metric flattener accepted only built-in int/float, so `np.float32` and `np.int64` metrics were silently dropped from the leaderboard — the dtypes LightGBM and XGBoost routinely emit. | Test against the numeric ABCs plus numpy's scalar base, excluding bools. |
| REPORTING_RENDERERS-1 / RUX-62 | `LinePanelSpec.ylim` was read by neither backend, so the decision curve's deliberately clipped y-window was discarded — the very readability fix its author wrote. Found independently by two agents. | Apply the limit in both line renderers. Verified on both backends. |
| REPORTING_RENDERERS-2 | `fig.autofmt_xdate()` is a FIGURE-level call: it hid the x tick labels of every non-last-row axes and cleared their labels, erasing the epoch-date ticks the recent time-axis fix had just computed and stripping labels off unrelated panels in the same row. | Rotate per-axes instead; the rotation was that call's only remaining contribution. |

## Found while fixing, not by the audit

| What | Where | Detail |
|---|---|---|
| A fixed PIT-panel seed could reproduce the caller's own RNG stream, cancelling the fix for CHARTS_A-1. | `charts/binary.py` | The randomised PIT needs a uniform draw per row, and a module constant `_PIT_SEED = 0` is the obvious way to keep it reproducible. But a caller (or a test) generating scores from `default_rng(0)` gets the identical stream, so `u == p` exactly, the randomisation cancels and KS jumps straight back to the 0.25 the fix removes -- observed live while verifying it. The draw is now seeded from a hash of the scores: still reproducible for a given dataset, and impossible to collide with a caller's seed. |
| The CUSUM drift chart raised a change-point on roughly half of all DRIFT-FREE series of the length it targets. | `charts/drift.py` | Chasing CHARTS_B-23 (the change-point reported as an internal row index) meant rendering a real series, which showed a crossing at row 1,109 on data with no injected shift. The fixed `h=8` gives a two-sided ARL_0 near 9,500 by Siegmund's approximation, so a 6,000-row series false-alarms about 47% of the time — measured, 3 of 4 pure-noise seeds crossed — directly contradicting the constant's own comment ("h=8 sigma pushes ARL_0 into the tens of thousands so a no-drift series stays quiet"). `h` is now solved from the series length for a 5% whole-series false-alarm probability and floored at the old 8.0, so it can only ever be quieter. 8/8 pure-noise seeds now stay silent; an injected 1.5-sigma shift is still caught 8 steps later. |
