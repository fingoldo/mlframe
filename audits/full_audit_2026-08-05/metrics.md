# Audit: metrics

**Scope**: `src/mlframe/metrics/**` (classification / regression / calibration / ranking metrics, ~18.4k LOC).
Explicitly out of scope: `feature_selection/filters/**` (MRMR), `feature_selection/shap_proxied_fs/**` (already
audited 2026-07-25, all findings closed).

**Files reviewed**: 85 `.py` files (39 production modules read line-by-line in full; the `_benchmarks/`
subfolder, 44 standalone dev/timing scripts, was grep-scanned for common anti-patterns plus several files
spot-checked in full — no production code path imports from `_benchmarks/`).

**LOC reviewed**: 18,421 (per `wc -l`).

All findings below were verified either by direct code reading with cross-file consistency checks, or by
live reproduction in a Python REPL against the installed `mlframe` package (noted per finding).

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|---|---|---|---|---|---|
| METRICS-1 | P0 | `calibration/_calibration_plot.py:667-850` | `show_calibration_plot(backend="plotly", show_plots=False)` with no `plot_file`/`plot_outputs`+`base_path` raises `UnboundLocalError: fig` — confirmed live. | Initialize `fig = None` at function top, or raise a clear `ValueError` for the unsupported combination instead of falling through to an unbound read. | AST/CFG checker: for every function with >1 `return`/fallthrough path, flag any local name read on a path where no preceding assignment dominates it (a lightweight def-before-use dataflow check) — catches this class of "identifier bound only in one branch, read unconditionally" bug automatically. |
| METRICS-2 | P0 | `_core_precision_mape.py:136-157` (`maximum_absolute_percentage_error` via `_max_abs_pct_error_kernel`) | Empty `y_true`/`y_pred` raises `ValueError: zero-size array to reduction operation` (from `np.nanmax` inside the njit kernel) instead of returning `NaN` like every sibling metric in the same file (`fast_brier_score_loss`, `fast_log_loss`, ...) and like the function's own docstring implies. Confirmed live. | Add the same `if len(y_true)==0: return float(np.nan)` guard the wrapper already has for every other metric in this module, before dispatching to the kernel. | Property test: for every public `fast_*`/`maximum_*`/metric function taking `(y_true, y_pred)` arrays, call it with `np.array([])`/`np.array([])` and assert it returns NaN (or raises a documented `ValueError`) rather than an undocumented exception type — run this as a generic parametrized sweep over every exported metric callable. |
| METRICS-3 | P0 | `regression/_regression_corr.py:154-177` (`fast_concordance_index`) | `fast_concordance_index` computes `(kendall_tau_b + 1) / 2`. Kendall's tau-b denominator excludes pairs tied in **either** `y_true` or `y_pred`, but the textbook / Harrell concordance index used in survival & risk modelling (the exact use case cited in the docstring) excludes only pairs tied in `y_true` and credits `y_pred`-tied pairs with 0.5. The two formulas coincide only when `y_pred` has no ties. Verified live: `y_true=[1,2,3,4]`, `y_pred=[1,1,2,2]` (predictions tied, true values distinct) gives `fast_concordance_index == 0.9082`, while the brute-force Harrell C-index on the same data is `0.8333` — a 9% systematic inflation. Tree-ensemble / quantized risk scores routinely have tied predictions, so this is a common, not exotic, input shape. | Implement the direct pairwise definition (or its O(N log N) equivalent): exclude only `y_true`-tied pairs from the denominator, credit `y_pred`-tied pairs with 0.5 in the numerator. Do not derive C-index from tau-b's tie-symmetric formula. | Add a fuzz/property test comparing `fast_concordance_index` against a brute-force reference C-index (0.5 credit for `y_pred` ties, `y_true`-tied pairs excluded) across randomized data **including deliberately-duplicated prediction values** — not just tie-free Gaussian noise. |
| METRICS-4 | P1 | `calibration/_calibration_metrics.py:432-481` (`integral_calibration_error_from_metrics`) | `calibration_coverage` is accepted as a parameter (and is always computed and passed in by every caller: `fast_calibration_report`, `fast_ice_only`, `_batch_per_class_ice_kernel`) but is **never referenced** in the function body — it has zero effect on the returned ICE value. Verified live: calling with `calibration_coverage=0.9`, `0.1`, and `0.0` (all else equal) returns the identical float. `git log -S` traces this back to the function's very first version (2024, `mlframe/metrics.py`), so this predates all the module splits — it is not a refactor regression, it is a ~2-year-old dead parameter in the framework's core early-stopping metric ("Integral Calibration Error"). | Either wire `calibration_coverage` into the formula with an explicit `coverage_weight` knob (if the omission was accidental), or drop the parameter from the signature and all call sites and document explicitly that ICE does not consider bin coverage (if intentional) so callers/readers stop assuming it does. | Static "unused-parameter" scanner for `@njit`-and-plain functions with a stable public signature: flag any parameter name that never appears as a `Name`/`Load` node anywhere in the function body's AST (excluding the signature itself) — would have caught this on day one. |
| METRICS-5 | P2 | `classification/_classification_report.py:83-95` (`format_classification_report`) | The "MACRO-AVG CAVEAT" docstring says macro avg is "the UNWEIGHTED mean ... over ALL nclasses, INCLUDING any class with support 0 ... DEFLATES macro precision/recall/f1" and claims this "matches sklearn". This describes the `macro_over_present=False` legacy behaviour, but `format_classification_report` calls `fast_classification_report(...)` without passing that flag, so it gets the kernel's actual default, `macro_over_present=True` (present-classes-only — which is what actually matches `sklearn.metrics.classification_report`). Verified live with a 3-class example where class 2 has zero support: printed macro avg is `0.875` (present-only, undeflated), directly contradicting the docstring's description of a deflated value. | Rewrite the caveat to describe the present-only default (mirroring the accurate docstring already on `fast_classification_report` itself), and note that `macro_over_present=False` (not exposed as a `format_classification_report` parameter today) is the deflating legacy path. | Docstring-vs-behavior consistency check: for functions whose docstring makes a falsifiable numeric claim ("X deflates when Y"), a doctest or dedicated unit test should assert exactly that claim; add one here asserting macro avg does NOT deflate when a declared class is absent (as the code actually behaves), which will fail loudly if the underlying default ever silently flips again. |
| METRICS-6 | P2 | `rank_correlation.py:118-154` (`_spearmanr_scalar_njit`) | The kernel's leading NaN pre-check (`for i in prange(n): if not (...): return np.nan`) triggers Numba's `NumbaPerformanceWarning: loop will not be executed in parallel due to there being more than one entry to or exit from the loop` — confirmed live (isolating just this loop reproduces the warning; removing the early `return` makes it disappear). Results are still correct (verified NaN propagates), but the docstring's "two independent argsort-based rankings run concurrently (2-way prange)... ~1.5-2.7x over the scipy path" claim overstates the parallelism actually achieved for the whole function on any input containing NaN/inf, since Numba silently serializes that first loop instead of the intended thread-parallel scan. | Split the NaN check into its own loop with no early return (accumulate a bad-flag via `+=`, check after the loop, matching the pattern already used correctly in `_batch_per_class_ice_kernel`/`_average_rank_inplace`'s caller in `_spearmanr_batched_njit`), so Numba can parallelize it like the other two loops in the same function. | Reuse the existing internal convention (documented in `_multilabel_metrics.py`'s "numba 0.63.1 parfor workaround" comments): grep every `@njit(parallel=True)` function for a `return` statement nested inside a `prange` loop body, and assert (via a build-time or CI check capturing `NumbaPerformanceWarning`) that none fire when the function is exercised with a warnings-as-errors filter. |
| METRICS-7 | P2 | `classification/_threshold_optimization.py:88-116` (`optimal_threshold`), `classification/_ordinal_cutpoints.py:61-127` (`optimal_ordinal_cutpoints`) | Both functions fit a tunable parameter (decision threshold / ordinal cut thresholds) by directly maximizing the target metric on the **same** `(y_true, y_score)` arrays passed in — classic threshold-selection-on-evaluation-data leakage if a caller passes a test/OOS split and then reports the resulting metric as the model's test performance. Neither docstring carries the "HOLDOUT CONTRACT" warning this exact codebase already uses for the analogous risk in `quantile.py` (`coverage()`: *"computed on training rows it is optimistically inflated ... the function cannot detect which rows are which -- the CALLER must pass a holdout split"*, and `pit_values()`). | Add the same HOLDOUT CONTRACT paragraph to both docstrings: fit on train/val, apply (`apply_cutpoints`) to a held-out set before reporting. | Add a grep-based lint rule: any public function whose name matches `optimal_*`/`fit_*`/`*_cutpoints`/`*_threshold` and that both (a) accepts `y_true` and (b) returns a fitted scalar/array used later to score new data, must contain the literal string "HOLDOUT" or "holdout" in its docstring — flag violations for manual doc review. |
| METRICS-8 | P3 | `_multilabel_extras.py:480-502,564` (`fast_multilabel_classification_metrics_block`'s `jaccard_macro`) vs `_multilabel_metrics.py:118-136` (`jaccard_score_multilabel`) | The two Jaccard implementations in the same package use opposite empty/zero-support conventions: `jaccard_score_multilabel` (per-sample average) treats an empty-union row as a perfect score (`1.0`, explicitly documented as a deliberate deviation from sklearn's default), while `fast_multilabel_classification_metrics_block`'s per-label macro Jaccard treats a zero-support label as `0.0` with no cross-reference to the other convention in either docstring. Not a bug in isolation (they're different averaging axes — per-sample vs per-label — so some difference is expected), but the silent 1.0-vs-0.0 empty-case divergence is easy to trip over when a caller compares the two numbers on a rare-label fold. | Cross-reference both docstrings ("see X for the sibling metric's DIFFERENT empty-case convention") or align the empty-case constant if there's no principled reason for the difference. | none beyond a doc-linkage lint (not independently automatable). |
| METRICS-9 | P3 | `classification/_classification_extras.py:645-671` (`_rps_kernel` / `ranked_probability_score`) | `_rps_kernel` reads `y_true[i]` as an ordinal class index (`cum_y = 1.0 if ti <= k else 0.0`) with **no bounds validation**, unlike its sibling `_multiclass_confusion_kernel` in the same file (`if 0 <= t < K and 0 <= p < K`) and unlike `fast_classification_report`'s explicit out-of-range guard. A negative or out-of-range `y_true` value silently produces a well-formed-looking but wrong RPS instead of raising. | Validate `0 <= y_true.min()` and `y_true.max() < K` in `ranked_probability_score`'s Python wrapper before dispatching to the njit kernel (mirroring the pattern already used elsewhere in this file). | Generic property test: for every ordinal/multiclass metric taking an integer `y_true`/class-index array, feed a single out-of-range label (e.g. `-1` or `K`) and assert either a raised `ValueError` or an explicitly-documented "silently ignored" contract — currently this one silently corrupts instead of either. |
| METRICS-10 | P3 | `regression/_regression_benchmark.py:173-199` (`fast_rmspe`) | `fast_rmspe` silently excludes `y_true == 0` rows (`if y_true[i] != 0.0`) with **no warning**, while its direct siblings for the identical zero-denominator situation — `fast_mape_mean` (same subpackage) and `maximum_absolute_percentage_error` — both emit a rate-limited `RuntimeWarning` naming the affected row count. | Add the same rate-limited warning pattern (`_MAPE_ZERO_WARN_SEEN`-style set) used by `fast_mape_mean` in the same package. | Grep sweep: for every `fast_*`/`*_error`/`*_deviance` function whose kernel silently skips rows on a documented "undefined at 0" condition, assert a matching `warnings.warn(...RuntimeWarning)` call exists in the same function — flags the inconsistency directly. |
| METRICS-11 | P3 | `_core_precision_mape.py:78,104-109` (`fast_classification_report`) | `accuracy = hits.sum() / len(y_true)` divides by the raw sample count, while `weighted_averages` (computed a few lines later in the same function) divides by `support_total` — the count of samples whose true label was in-range `[0, nclasses)`. When out-of-range labels are present (the defensive path this same function explicitly guards against elsewhere), `accuracy` and the weighted P/R/F1 averages are silently computed on two different denominators from the identical call. | Either drop out-of-range rows from `len(y_true)` too (use `support_total`), or document explicitly that `accuracy` intentionally counts them in the denominator (penalizing them as misses) while weighted-avg does not. | Unit test: call `fast_classification_report` with a label vector containing one out-of-range value, and assert `accuracy` and `weighted_averages` were computed on a consistent, documented denominator. |
| METRICS-12 | P3 | `calibration/_calibration_plot.py` (7 lines: 220,418,686,773,785,790,805), `classification/_classification_report.py` (5 lines: 200,261,538,539,738), `classification/_gains_lift.py` (2 lines: 10,18), `regression/_regression_metrics.py:1`, `_benchmarks/bench_average_rank_tie_scan_cpx22.py:1`, `_benchmarks/bench_ndcg_sort_count_hoist_cpx23.py:1,22`, `_benchmarks/bench_prob_separation_seq_fused.py:1,11` | Unicode em-dash (`—`) characters used as prose punctuation in docstrings/comments, inconsistent with the ASCII `--`/`-` convention this repository otherwise enforces (nearly every other file in this cluster uses `--` for the same rhetorical pause). | Mechanical find/replace of `—`/`–` with ` -- ` in prose contexts in the listed files (already the dominant style elsewhere in the same modules — no behavior change). | Add a repo-wide (or per-cluster) grep gate for the em-dash/en-dash Unicode code points in tracked `.py` files, run as an advisory pre-commit/CI check. |
| METRICS-13 | P3 | `classification/_classification_extras_blocks.py:311-315` | Comment reads: "...the exact bug fast_classification_report's macro_over_present flag was added to fix, **left unfixed in this sibling fused block**." — but the code immediately below (`present_macro = (row_sums > 0) \| (col_sums > 0)`) actually *does* apply the present-only fix. The comment's trailing clause reads as if the bug is still live when it is not; a future maintainer skimming comments during an incident could waste time re-diagnosing an already-fixed issue (or, worse, "fix" already-correct code). | Reword to something like "...the exact bug fast_classification_report's macro_over_present flag was added to fix; applied here too via `present_macro` below." | none beyond human/LLM comment-review; not mechanically detectable in general, but a targeted grep for the phrase "left unfixed" co-located with code that contradicts it is a cheap one-off check. |
| METRICS-14 | P3 | `iteration_metrics.py:130-182` (`_multiclass_metrics`), module docstring lines 1-19 | The module docstring promises: *"This is a thin delegating aggregator, NOT a reimplementation: every number comes from an existing public metric function"*. `_multiclass_metrics`'s `log_loss` value (lines 150-157) is computed via a fresh inline numpy formula (row-normalize, clip, gather, `-log(...).mean()`), not delegated to any existing `mlframe.metrics` primitive — because no public multiclass log-loss function exists anywhere in this package (only `fast_log_loss`/`fast_log_loss_binary`, both binary-only). Separately, rows whose true label falls outside `[0, k)` are silently excluded from the log-loss mean (`valid = (yt>=0) & (yt<k)`) with no count/warning surfaced, unlike the rest of the package's established "surface the drop count" pattern (`maximum_absolute_percentage_error`, `fast_rmsle`, `fast_mape_mean`). | Either (a) extract this inline formula into a proper `mlframe.metrics.classification` primitive (`fast_log_loss_multiclass`) so the docstring's claim becomes true and the logic gets its own direct unit tests, or (b) soften the docstring to acknowledge the one reimplemented exception; either way, surface the out-of-range-label drop count via a debug log for diagnosability. | Add a direct unit test for the multiclass log-loss branch of `compute_all_metrics` comparing against `sklearn.metrics.log_loss(labels=range(k))` on random multiclass data (currently only indirectly exercised, if at all, through the per-iteration capture integration path). |

**Counts**: P0 = 3, P1 = 1, P2 = 3, P3 = 7 (14 total).

## Narrative

### METRICS-1 — `show_calibration_plot` crashes with `backend="plotly", show_plots=False` (P0)
`show_calibration_plot` is a public, multiply-re-exported function (`mlframe.metrics.core`,
`mlframe.metrics.calibration`, `mlframe.metrics`). Its body only assigns the local `fig` inside the
`if backend == "matplotlib":` block (line 667 onward); the unconditional `return fig` sits at the same
indentation as that `if`, i.e. outside it. The one internal call site (`fast_calibration_report`) happens
to always avoid the crash because it only invokes `show_calibration_plot` when `plot_file or show_plots or
_dsl_render` is truthy, and the DSL branch (lines 584-617) intercepts every `backend="plotly"` call where
`show_plots` is `True`. But nothing stops an external caller — or a future internal caller — from passing
`backend="plotly", show_plots=False` with no `plot_file`/`plot_outputs`+`base_path`, which sails past both
early-return guards and reaches the bare `return fig`. Reproduced directly:
```
>>> show_calibration_plot(freqs_predicted, freqs_true, hits, backend="plotly", show_plots=False)
UnboundLocalError: cannot access local variable 'fig' where it is not associated with a value
```

### METRICS-2 — `maximum_absolute_percentage_error` crashes on empty input (P0)
Every sibling scalar metric in `mlframe.metrics` (Brier, log-loss, R2, etc.) explicitly guards
`len(y_true) == 0` in its Python wrapper and returns `NaN`. `maximum_absolute_percentage_error`'s wrapper
does not — it dispatches straight to `_max_abs_pct_error_kernel`, whose body calls `np.nanmax(mape)` on a
zero-length array, which numpy (even inside `@njit`) raises `ValueError: zero-size array to reduction
operation fmax which has no identity` for. Reproduced directly: `maximum_absolute_percentage_error(np.array([]),
np.array([]))` raises instead of returning `nan`. An empty validation/OOF slice is a completely ordinary
occurrence in a training pipeline (e.g. a fold with zero rows after a filter), so this is a realistic crash
path, not a contrived one.

### METRICS-3 — `fast_concordance_index` diverges from the true C-index on tied predictions (P0)
The function's docstring claims it is "Equivalent to (Kendall tau-b + 1) / 2 after tie correction" and that
this is standard for "survival / risk modelling". That equivalence only holds when there are zero ties
anywhere in the data. Kendall's tau-b's denominator is `sqrt((P-Tx)(P-Ty))`, symmetrically discounting BOTH
`y_true`-tied and `y_pred`-tied pairs from the comparable-pair count, whereas the textbook / Harrell
concordance index used in survival analysis only excludes `y_true`-tied pairs (non-comparable) and credits
`y_pred`-tied pairs with 0.5 concordance each. Verified with a minimal example (`y_true=[1,2,3,4]`,
`y_pred=[1,1,2,2]`, i.e. two prediction ties, zero true-value ties):
`fast_concordance_index` returns `0.9082`, while a brute-force reference C-index computed directly from the
pairwise definition returns `0.8333` — a 9-point overstatement. Predicted risk scores in real survival /
risk models very commonly have ties (quantized tree-ensemble leaf outputs, rounded scores, discretized risk
buckets), so this is not a corner case. The regression tests for this function
(`tests/metrics/regression/test_regression_extras.py::test_concordance_index_range`,
`tests/training/reporting/test_regression_report_cindex_derived_from_kendall.py`) only check the output's
numeric *range* and that two internal call sites of the SAME `(tau_b+1)/2` formula agree with each other —
neither test compares against an independent C-index oracle, and neither exercises tied predictions, so the
bug has no test that would catch it.

### METRICS-4 — `integral_calibration_error_from_metrics`'s `calibration_coverage` parameter is dead code (P1)
The public "Integral Calibration Error" aggregator accepts `calibration_coverage` as its third parameter,
and every one of its three call sites (`fast_calibration_report`, `fast_ice_only`,
`_batch_per_class_ice_kernel`'s inline duplicate) computes a real per-call coverage value and passes it in.
The function body, however, never reads that parameter anywhere in its arithmetic — confirmed by calling it
three times with `calibration_coverage` set to `0.9`, `0.1`, and `0.0` (all other arguments fixed) and
observing an identical return value each time. `git log -S"calibration_coverage"` traces this back to the
function's original 2024 form in the pre-split `mlframe/metrics.py`, where the parameter was already present
and already unused in the formula (`res = brier_loss*brier_loss_weight + calibration_mae*mae_weight +
calibration_std*std_weight - np.abs(roc_auc-0.5)*roc_auc_weight`, no `calibration_coverage` anywhere) —
so this predates every subsequent module split and is not a refactor regression. Given ICE is the framework's
primary early-stopping / model-selection objective (see the module's own extensive documentation elsewhere on
how heavily it's relied on), a knob that looks like it should reward well-covered calibration curves but
silently does nothing is a real, long-lived design gap worth resolving one way or the other.

### METRICS-5 — `format_classification_report`'s macro-avg docstring contradicts its actual (default) behavior (P2)
`format_classification_report` calls `fast_classification_report(y_true, y_pred, nclasses=nclasses,
zero_division=zero_division)` without passing `macro_over_present`, so it inherits that kernel's actual
default, `macro_over_present=True` — averaging macro P/R/F1 only over classes present in `y_true` OR
`y_pred` (the kernel's own docstring: "matching sklearn.metrics.classification_report"). But
`format_classification_report`'s own "MACRO-AVG CAVEAT" describes the opposite: an unweighted mean "over ALL
nclasses, INCLUDING any class with support 0", claiming this deflates the macro average and claiming this
"matches sklearn" (it does not — sklearn's default label set is also present-classes-only). Reproduced
directly with a 3-class example where class `2` never appears in either array: the printed report shows
`macro avg = 0.875` (present-only, undeflated, matching the ACTUAL kernel default), while the
`macro_over_present=False` legacy value on the identical data is `0.583` (deflated) — exactly the number the
stale docstring implies is being returned. A reader trusting the docstring would misjudge how a rare-label
split's macro scores respond to absent classes.

### METRICS-6 — `_spearmanr_scalar_njit`'s NaN pre-check silently loses its intended parallelism (P2)
The scalar Spearman kernel's docstring claims "the two independent argsort-based rankings run concurrently
(2-way prange)" and cites a 1.5-2.7x speedup. Its FIRST loop (`for i in prange(n): if not (...): return
np.nan`), however, contains an early `return` inside the `prange` body — a pattern Numba's parfor pass
explicitly cannot parallelize (multiple loop exits). Isolating exactly this loop and compiling it reproduces
`NumbaPerformanceWarning: prange or pndindex loop will not be executed in parallel due to there being more
than one entry to or exit from the loop`; removing the early return makes the warning disappear. The final
result is still numerically correct (NaN propagates correctly for NaN-containing input, verified), so this is
a silent perf regression rather than a correctness bug — but it is exactly the "numba parallel loop silently
downgrades to sequential with no visible error" class of issue this codebase's own conventions elsewhere are
built to catch (see `_multilabel_metrics.py`'s explicit numba-0.63.1 parfor-reduction workaround comments,
and `_core_numba_warmup.py::_assert_numba_nogil_active`, which exists precisely to catch a sibling silent-
degradation failure mode for `nogil`).

### METRICS-7 — `optimal_threshold` / `optimal_ordinal_cutpoints` have no HOLDOUT CONTRACT warning (P2)
Both functions search a threshold/cutpoint set that directly maximizes a metric on the exact `(y_true,
y_pred)` arrays the caller supplies — the textbook shape of threshold-selection-on-evaluation-data leakage if
a caller naively fits on a test/OOS split and reports the resulting metric as that split's honest
performance. This exact class of risk is *already* a named, established pattern in this very package:
`quantile.py::coverage()` carries an explicit "HOLDOUT CONTRACT" docstring paragraph ("computed on training
rows it is optimistically inflated... the function cannot detect which rows are which -- the CALLER must pass
a holdout split"), and `pit_values()` repeats the same guidance. Neither `optimal_threshold` nor
`optimal_ordinal_cutpoints` — both of which literally *fit a parameter* on the input, a strictly higher-risk
operation than `coverage()`'s pure read-only diagnostic — carries any equivalent warning, an inconsistency
within the same codebase's own established documentation convention.

### METRICS-8 through METRICS-14 (P3)
See the table; these are hygiene / consistency / minor-robustness items: (8) two Jaccard variants in the
package disagree on the empty-case convention with no cross-reference; (9) `ranked_probability_score`'s
kernel omits the out-of-range-label bounds check its sibling multiclass-confusion kernel in the same file
already has; (10) `fast_rmspe` silently drops zero-`y_true` rows without the rate-limited warning its direct
siblings (`fast_mape_mean`, `maximum_absolute_percentage_error`) provide for the identical situation;
(11) `fast_classification_report`'s `accuracy` and `weighted_averages` use two different denominators
whenever out-of-range labels are present; (12) em-dash/en-dash Unicode characters in prose across several
files, against this repo's ASCII-dash convention; (13) a confusing/self-contradictory comment in
`_classification_extras_blocks.py` that reads as describing a still-open bug the code directly below it
already fixes; (14) `iteration_metrics.py`'s multiclass log-loss is a genuine (undocumented) reimplementation
— because no public multiclass log-loss primitive exists in the package for it to delegate to — that also
silently drops out-of-range-label rows with no surfaced count, unlike this package's established pattern.

## Dimension coverage notes

- **Correctness bugs**: 3 found (METRICS-1, -2, -3), all confirmed by live reproduction against the
  installed package.
- **ML correctness (leakage / reproducibility / calibration)**: METRICS-4 (dead ICE coverage term),
  METRICS-7 (missing holdout-contract documentation on two threshold-fitting functions). No unseeded-RNG or
  hidden-global-state reproducibility bugs were found in production code paths — the one intentionally
  seeded RNG (`create_robustness_standard_bins`'s `**RANDOM**` pseudo-feature) is explicitly seeded
  (`np.random.default_rng(seed)`), and no other `np.random.*` call appears outside `_benchmarks/`.
- **Computational efficiency**: METRICS-6 (silent loss of intended `prange` parallelism). No unnecessary
  `.copy()`/materialization, missed-vectorization, or wrong-dispatch issues were found beyond what the
  codebase's own extensive prior perf-audit comments already document and resolve — this cluster shows
  unusually heavy, already-applied optimization work (fused single-pass blocks, GPU/CPU auto-dispatch,
  kernel_tuning_cache integration, documented bench-rejected alternatives throughout).
- **Edge cases and robustness**: METRICS-1, -2 (crashes), -9, -10, -11 (silent-exclusion / inconsistent-
  denominator gaps). Constant-column, single-class, single-row, and NaN/Inf inputs are otherwise very
  thoroughly handled throughout this cluster (explicit NaN/degenerate-input sentinels are the dominant
  pattern in nearly every kernel reviewed).
- **Test coverage gaps**: the existing test for `fast_concordance_index` (METRICS-3) is the clearest instance
  of "asserts against the code's own output instead of an independent oracle" found in this cluster — it
  pins `(tau_b+1)/2` internal self-consistency, not correctness against a real C-index reference. No empty-
  input test exists for `maximum_absolute_percentage_error` (METRICS-2). No `backend="plotly", show_plots=
  False` test exists for `show_calibration_plot` (METRICS-1).
- **Code quality / architecture**: METRICS-4 (dead parameter), METRICS-5 (stale docstring), METRICS-13
  (self-contradictory comment), METRICS-14 (docstring overclaim). No dead code, broad `except:` clauses
  (bare `except:` — zero occurrences), or mutable-default-argument bugs were found anywhere in this cluster
  (`except Exception` is used pervasively but always narrowly scoped, logged, and documented as best-effort
  per the project's own established convention).
- **OSS/hygiene**: METRICS-12 (em-dash/en-dash usage). No mojibake or stale audit-wave markers were found.
  Docstring quality is otherwise unusually high across this cluster (extensive, precise, evidence-cited).

## Zero-finding dimensions (explicitly confirmed, not merely omitted)

- No bare `except:` clauses anywhere in `src/mlframe/metrics/**`.
- No mutable-default-argument bugs (`def f(x=[])` / `def f(x={})` pattern) anywhere in this cluster.
- No unseeded RNG in any production (non-`_benchmarks/`, non-test) code path.
- No O(n^2)-where-O(n log n)-available algorithmic issues found; the reviewed code already implements and
  documents several such fusions (e.g. `_wasserstein_1d_fused`/`_ks_distance_fused`'s O(n+m) merge instead of
  the naive O((n+m) log(n+m)) concatenate+sort+searchsorted it replaced).

Report path: `C:/Users/Admin/Machine learning/mlframe/audits/full_audit_2026-08-05/metrics.md`
