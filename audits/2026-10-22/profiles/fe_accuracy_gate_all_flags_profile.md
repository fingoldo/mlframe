# fe accuracy gate (`_fe_accuracy_gate.measure_feature_uplift`): profile inside the all-flags fit (n=20000)

Measured 2026-10-28 with `D:/Temp/cg_allflags.py` (all FE flags on, warm fit on other data, default device mode).

The engineered-accuracy gate (`_gate_engineered_accuracy`, `keep_engineered_over_source`) takes 4.21 s of cProfile time in this fit, more than most FE families. It is not in the family table because it is a fit-level gate, not an FE family.

| part | cum s | calls | note |
|---|---:|---:|---|
| `measure_feature_uplift` | 4.18 | 38 | one probe per candidate |
| `_score` (fit one `StandardScaler` + `LogisticRegression(C=1.0, max_iter=200)` per fold, ROC-AUC) | 4.04 | 670 | one sklearn lbfgs fit per candidate per fold |

Reading: the cost is the sklearn lbfgs fit repeated for each candidate and fold (about 6 ms per fit). The usability greedy already has a batched Newton path (`_usability_greedy_clf_batched`, host twin `_usability_greedy_clf_host_batched`) that fits all candidates of a fold at once and falls back to the exact per-candidate refit on non-convergence. This gate could use it, but it scores ROC-AUC from predicted probabilities and the batch returns only logloss, so the batch would have to return probabilities too. That needs a selection-equivalence check (AUC ties could move at the Newton-vs-lbfgs tolerance). Not changed here; this is the next lever for this gate.
