# Wave 24 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on areas no earlier wave read: the shap-proxy interaction / GPU / heuristics modules, the plotly and matplotlib scatter and network renderers, and the classification-report / iteration / quantile metric blocks. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W24REN-1 | renderer | Medium | plotly scatter: a per-point colour list was not reordered with the downsampled x / y, so colours landed on the wrong points |
| **RESOLVED** | W24REN-2 | renderer | Medium | plotly scatter: a per-point size list raised `TypeError` (also in the highlight ring) |
| **RESOLVED** | W24REN-3 | renderer | Low | plotly scatter: one NaN in x or y gave the perfect-fit diagonal NaN endpoints |
| **RESOLVED** | W24REN-4 | renderer | Low | plotly network: NaN edge weights sent every edge to an invalid bin, so no edge line was drawn |
| **RESOLVED** | W24REN-5 | renderer | Low | matplotlib network: a panel with no nodes raised on the empty extent reduction |
| **RESOLVED** | W24SHP-2 | `tests/feature_selection/shap_proxied/test_su_bin_nan_column_keeps_its_bins.py` |
| W24SHP-4 | `tests/feature_selection/shap_proxied/test_residual_passes_two_runs_two_passes.py` |
| W24SHP-6 | `tests/feature_selection/shap_proxied/test_interaction_searches_respect_min_card.py` |
| W24SHP-7 | `tests/feature_selection/shap_proxied/test_sparse_interaction_products_stay_inside_float32.py` |
| W24MET-1..3 | `tests/metrics/test_iteration_metrics_label_row_sets.py` |
| W24MET-4, 5 | `tests/metrics/test_binary_confusion_block_single_class_and_empty.py` |
| W24SHP-1 | search | High | `beam_search` grew the size-1 seed beam for `max_card - min_card` layers, so with `min_features=4, max_features=6` it stopped at size 3 and returned only subsets below the minimum (and single features when min = max); it now grows `max_card - 1` layers and filters the result to sizes >= `min_card` (`tests/feature_selection/shap_proxied/test_beam_search_respects_min_and_max_card.py`) |
| **RESOLVED** | W24SHP-2 | binning | High | `_su_bin` on a continuous column with a NaN got all-NaN quantile edges and fell to the rank-dense branch, one bin per distinct value (a 50k-row pair would allocate a ~2.5e9-cell joint); the finite rows are binned on their own and the missing rows share one extra bin (`tests/feature_selection/shap_proxied/test_su_bin_nan_column_keeps_its_bins.py`) |
| **DOC** | W24SHP-3 | leakage | Medium | sparse-interaction candidates are scored by ONE in-sample model on `[X | products]` by design (`sparse_interaction_candidates` says so) and their losses are merged with the OOF additive candidates; the downstream honest revalidation of the top-N re-scores every survivor out of fold, so the in-sample optimism can only reorder which candidates reach it. Not changed: an out-of-fold fit of the augmented frame would multiply the SHAP cost by the fold count for a ranking refinement, and a revalidation widening is the proportionate lever if a bed shows displacement |
| **RESOLVED** | W24SHP-4 | api | Medium | `residual_passes=2` was accepted but `run_residual_pass` did one pass while the report said `n_passes=2`; the pass body is now `_residual_round` and the driver runs it `residual_passes` times, each pass explaining what the previous one left unexplained, unioning the rescued columns and summing the gated importances for the blend; the report gains `n_passes_run` (`tests/feature_selection/shap_proxied/test_residual_passes_two_runs_two_passes.py`; the existing residual-pass tests pass) |
| **RESOLVED** | W24SHP-5 | memory | Medium | `build_pair_table` allocated a full P x P x n copy of the interaction tensor although only the top-k gate is read (a second copy as large as `Phi` itself, then a MemoryError that silently fell back to additive); it now returns a (k, k, n) block plus a feature-to-position map and `_interaction_extra` indexes through it; the two existing unit tests were moved to the compact layout (`tests/feature_selection/shap_proxied/test_shap_proxy_interaction_proxy.py::test_pair_table_memory_is_bounded_by_the_gate_not_the_feature_count`) |
| **RESOLVED** | W24SHP-6 | search | Medium | `interaction_proxy_top_n` and `interaction_top_n` cached every subset size from 1 through the greedy pass and returned them unfiltered, so `min_features` was broken and, with an empty exhaustive window, only below-minimum subsets came back; both now filter to `min_card <= size <= max_card` and the pair sweep only runs when 2 is inside the window (`tests/feature_selection/shap_proxied/test_interaction_searches_respect_min_card.py`) |
| **RESOLVED** | W24SHP-7 | dtype | Low | the sanitised interaction product was clipped to float64's maximum, which is inf again after the boosters' float32 cast; it is now clipped to the float32 range (`tests/feature_selection/shap_proxied/test_sparse_interaction_products_stay_inside_float32.py`) |
| **RESOLVED** | W24MET-1 | metrics | Medium | `_multiclass_metrics` took `n_classes` as the class count while the predictions are argmaxed over the score columns, so a mismatch left argmax codes outside `[0, k)`; the class count is now the column count and a throttled warning names the mismatch |
| **RESOLVED** | W24MET-2 | metrics | Medium | out-of-range labels were dropped from log-loss only; accuracy, MCC and the per-class averages kept them; every metric now runs on one row set (finite scores, integer label in `[0, k)`) and the warning says the rows left every metric |
| **RESOLVED** | W24MET-3 | metrics | Medium | a NaN label cast to int64 became INT_MIN and entered every kernel (binary `n_pos` went hugely negative); non-finite labels are dropped with the non-finite scores in both the binary and the multiclass suite |
| **RESOLVED** | W24MET-4 | metrics | Low | on empty input the binary confusion block returned accuracy / kappa NaN but MCC, balanced accuracy, F-scores, precision and recall 0.0; every rate is NaN now and the integer counts stay 0 |
| **RESOLVED** | W24MET-5 | metrics | Low | with a single-class `y_true` the undefined recall was counted as 0, so a perfect classifier scored balanced accuracy 0.5 and G-mean 0; both average over the classes present, as sklearn's balanced accuracy does |
| **DOC** | W24MET-6 | metrics | Low | the degenerate calibration report returns Brier / ECE / log-loss of 1.0 as worst-case placeholders so that plots, threshold search and string builders keep receiving finite floats; its docstring now says none of those numbers is a measurement and that the empty `metrics_string` marks the report as degenerate |

## Implemented

| ID | Regression test |
|---|---|
| W24REN-1..5 | `tests/reporting/test_renderer_edge_inputs_wave24.py` |
| W24SHP-1 | `tests/feature_selection/shap_proxied/test_beam_search_respects_min_and_max_card.py` |
