# Wave 24 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on areas no earlier wave read: the shap-proxy interaction / GPU / heuristics modules, the plotly and matplotlib scatter and network renderers, and the classification-report / iteration / quantile metric blocks. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W24REN-1 | renderer | Medium | plotly scatter: a per-point colour list was not reordered with the downsampled x / y, so colours landed on the wrong points |
| **RESOLVED** | W24REN-2 | renderer | Medium | plotly scatter: a per-point size list raised `TypeError` (also in the highlight ring) |
| **RESOLVED** | W24REN-3 | renderer | Low | plotly scatter: one NaN in x or y gave the perfect-fit diagonal NaN endpoints |
| **RESOLVED** | W24REN-4 | renderer | Low | plotly network: NaN edge weights sent every edge to an invalid bin, so no edge line was drawn |
| **RESOLVED** | W24REN-5 | renderer | Low | matplotlib network: a panel with no nodes raised on the empty extent reduction |
| **FUTURE** | W24SHP-1 | search | High | `beam_search` grows subsets only to `max_card - min_card + 1` when `min_features > 1` |
| **FUTURE** | W24SHP-2 | binning | High | `_su_bin`: a NaN in a continuous column gives all-NaN quantile edges, then one bin per row |
| **FUTURE** | W24SHP-3 | leakage | Medium | sparse-interaction candidates are scored in-sample and ranked against out-of-fold scores |
| **FUTURE** | W24SHP-4 | api | Medium | `residual_passes=2` is accepted but runs one pass |
| **FUTURE** | W24SHP-5 | memory | Medium | the interaction pair table is allocated P x P x n instead of top-k |
| **FUTURE** | W24SHP-6 | search | Medium | interaction searches return subsets below `min_features` |
| **FUTURE** | W24SHP-7 | dtype | Low | sanitised overflow products reach the booster as inf after the float32 cast |
| **FUTURE** | W24MET-1 | metrics | Medium | `_multiclass_metrics`: argmax columns vs `n_classes` / raw `classes_` labels are not validated |
| **FUTURE** | W24MET-2 | metrics | Medium | out-of-range labels are dropped from log-loss only, so one dict mixes row sets |
| **FUTURE** | W24MET-3 | metrics | Medium | NaN labels are cast to int64 INT_MIN and never filtered |
| **FUTURE** | W24MET-4 | metrics | Low | empty input: accuracy / kappa NaN but MCC, balanced accuracy, F1 0.0 |
| **FUTURE** | W24MET-5 | metrics | Low | single-class y_true gives balanced accuracy 0.5 and G-mean 0 for a perfect classifier |
| **FUTURE** | W24MET-6 | metrics | Low | the degenerate calibration report returns plausible-looking Brier / ECE / ll of 1.0 |

## Implemented

| ID | Regression test |
|---|---|
| W24REN-1..5 | `tests/reporting/test_renderer_edge_inputs_wave24.py` |
