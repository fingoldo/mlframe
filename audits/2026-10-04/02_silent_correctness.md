# 02 Silent correctness (read-only audit, 2026-10-04)

## Scope and method
Deduped against audits/full_audit_2026-09-20/_TRACKER.md (158 RESOLVED, 4 REJECTED, 3 NOT A DEFECT; none re-reported).
Patterns grepped over src/mlframe (counts): bare/broad `except ...: pass|continue|return` single-line forms (1940 lines matched the loose
regex, 0 `except..: pass` one-liners outside benchmarks); `logger.debug` fallbacks after `except Exception` (30+ sampled, all GPU->CPU or
perf-path downgrades with exc_info, results-equivalent, not reported); mutable defaults (0); `df.drop(x)` without axis (19 hits, all polars
or pandas-guarded, verified basic.py:394 uses columns= for pandas); `.min()==.max()` constant checks (11 hits; calibration.py:137,
local_classifier.py:176 are NaN-safe in effect); `astype(int/int8/int32)` (sampled ~20); hardcoded random_state (25 hits).
Verified findings: 2 (1 demonstrated by probe). The tree is heavily hardened; most hits already carry fix comments.

## Findings
| ID | file:line | sev | evidence | demonstrated failure | proposed fix | Disposition |
|---|---|---|---|---|---|---|
| SC-01 | feature_engineering/ensemble_features.py:142-150 (_bin_counts; used by predictor_consensus_entropy :177, top2_gap :190, predictor_disagreement_features :353) | P2 | `lo = arr.min(axis=1,...)`; `((arr - lo)/span*n_bins).astype(np.int32)` | row `[0.1,0.5,0.9,nan]`, n_bins=4 -> counts `[4,0,0,0]` (all 4 predictors in bin 0, incl. the 3 valid ones); entropy 0 / "full consensus" for a row that has a missing predictor; only a RuntimeWarning, no error | use nanmin/nanmax, mask NaN cells out of the histogram (or emit NaN for the row) | OPEN |
| SC-02 | feature_selection/filters/_hermite_fe_optimise.py:175,178,349,351,783,784; _hermite_fe_optimise_pair.py:405,407; fe_baselines.py:76,80 | P3 | `mutual_info_classif(..., random_state=42, ...)` while search fns take `seed` | user seed does not reach the KSG jitter; different `seed` runs share the same MI noise (determinism fine, variance understated across seeds) | thread the existing seed into these calls | OPEN |

## Verdict
No P1 found; swallowed-error and drop-axis siblings of the known history appear already closed. Not worth fixing: GPU->CPU `logger.debug`
fallbacks (equivalent results, perf-only), calibration/charts `min()==max()` (NaN falls through harmlessly), `int8` casts of 0/1 labels.
Unverified: depth in ~190 sample_weight-touching files was not read exhaustively.
