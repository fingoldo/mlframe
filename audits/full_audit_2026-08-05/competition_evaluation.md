# Audit report: competition_evaluation

**Scope**: `src/mlframe/competition/**` and `src/mlframe/evaluation/**` (~11.3k LOC combined per the
assignment; `feature_selection/filters/**` and `feature_selection/shap_proxied_fs/**` excluded as already
audited).

**Files reviewed in full** (via the Read tool, line-by-line): 43
- All 17 non-benchmark files under `src/mlframe/competition/` (16 modules + `__init__.py`).
- All 23 non-benchmark files under `src/mlframe/evaluation/` (19 modules + `__init__.py` + the 3
  underscore-prefixed `_bootstrap_*` support modules).
- 3 of the 33 `_benchmarks/*.py` harness scripts (`_cprofile_bench_shared.py`,
  `competition/_benchmarks/bench_leak_scan.py`, `evaluation/_benchmarks/bench_group_leakage_guard.py`), read
  in full as representative samples.

**Lighter-pass coverage**: the remaining 30 `_benchmarks/*.py` files (~2,705 LOC) are cProfile/timing
harnesses only (no production call sites reach them) and received a targeted grep sweep instead of a full
line-by-line read — specifically for bare `except:`/`except Exception:`, mutable default arguments, and the
`pd.DataFrame(rows).sort_values(...)`-without-empty-guard pattern found elsewhere in this cluster (see
COMPETITION_EVALUATION-5..8). No instances of any of those were found in `_benchmarks/`. Given this is a
disclosed reduced-depth pass, P3-level nits in the unread 30 files may have been missed; every dimension
below was still evaluated across the fully-read 43 files.

**LOC reviewed**: ~8,557 (8,406 LOC across the 40 non-benchmark modules, per `wc -l`, plus 151 LOC across the
3 sampled benchmark scripts).

## Summary by review dimension

1. **Correctness bugs**: 4 confirmed, all empirically reproduced (COMPETITION_EVALUATION-1/2/3/4).
2. **ML correctness** (leakage/reproducibility/calibration/class-imbalance): COMPETITION_EVALUATION-4 (leak
   detector accuracy) and COMPETITION_EVALUATION-9 (bootstrap CI diagnosability under class imbalance) are
   the findings in this dimension. No unseeded-RNG, no OOF-boundary violation, and no sample-weight-threading
   gap was found anywhere in the 40 reviewed modules — this cluster is unusually disciplined about causality
   (every leak-prone competition trick in `competition/` is explicitly gated with an OOF-only warning in its
   own docstring, and every `evaluation/` diagnostic that fits a model does so per-fold with fresh
   `clone()`/`model_factory()` calls).
3. **Computational efficiency**: no correctness-impacting or > trivial efficiency issue found. This cluster
   is heavily perf-audited already (dozens of documented, benchmarked optimizations in code comments); no new
   O(n^2)-where-O(n log n)-exists pattern, no avoidable large-frame `.copy()`, and no missed
   njit/parallel/GPU dispatch opportunity on a hot path was found in the reviewed files.
4. **Edge cases and robustness**: 4 confirmed, all the same recurring bug class (COMPETITION_EVALUATION-5/6/7/8).
5. **Test coverage gaps**: every numbered finding below is also, by construction, a test-coverage gap (none
   of the 4 correctness bugs or the 4 edge-case bugs have a regression test); called out per-finding rather
   than listed separately.
6. **Code quality/architecture**: no bare `except:`/`except Exception:` and no mutable-default-argument bug
   anywhere in the 40 reviewed modules (confirmed via full read + a repo-wide grep sweep). One API-consistency
   issue (COMPETITION_EVALUATION-2) and minor hygiene items (COMPETITION_EVALUATION-10/11).
7. **OSS/hygiene**: no mojibake, no stale audit/wave/phase markers, no stray dashes-in-prose found in this
   cluster's comments — 2 minor items (COMPETITION_EVALUATION-10/11).

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| COMPETITION_EVALUATION-1 | P1 | `evaluation/group_leakage_guard.py:64` | `assert_no_group_leakage` never materializes `cv_splits` to a list despite its own docstring promising exactly that, so passing a real generator silently exhausts it for the caller. | Add `cv_splits = list(cv_splits)` at the top of the function, matching the documented contract. | Grep every docstring phrase "materialized to a list"/"consumed eagerly"/"without exhausting" and assert the corresponding parameter is wrapped in `list(...)`/`tuple(...)` before the first iteration in the function body. |
| COMPETITION_EVALUATION-2 | P1 | `evaluation/reports.py:289` | `evaluate_estimators`'s `classification_thresholds: Optional[list]` param is used two incompatible ways: `.map()` (needs dict-like) in the classifier branch vs. `np.asarray(...)` (needs array-like) in the regressor branch — passing the documented `list` type to a classifier crashes with `TypeError: 'list' object is not callable`. | Split into two differently-named/typed parameters, or coerce consistently (e.g. accept only a dict and build the bin-edge array from `sorted(dict.values())` for the regressor path). | AST scanner: flag any single function parameter that is both passed to `pandas.Series.map()` and to `np.asarray()`/used as a numeric array in different branches of the same function — a strong signature of a split API contract. |
| COMPETITION_EVALUATION-3 | P1 | `evaluation/expanding_window_leakage.py:152-155` | `leaky_row_ranges` collapses each fold's leaked original-df positions to `(min, max+1)`, which is only a true contiguous range when the input `df` was already time-sorted; on realistic (unsorted) input it reports a range spanning nearly the whole dataset instead of the actual leaked rows, directly contradicting the docstring's "the exact rows whose held-out score was inflated" claim. | Return the actual (possibly non-contiguous) sorted `np.ndarray` of original positions instead of a `(start, end)` envelope, or explicitly document/detect the "df not time-sorted" case and fall back to an index list. | Property test: build a fixture with a SHUFFLED `time_col`/original-row-order relationship, run with `auto_remediate=True`, and assert every reported range's width is within a small constant factor of the true per-fold row count (not close to `len(df)`). |
| COMPETITION_EVALUATION-4 | P1 | `evaluation/leak_scan.py:19-38` | `_rank_columns`'s "single argsort + scatter" rank shortcut does not average ties, yet `scan_temporal_leak`'s own docstring says `split_labels` may be "a binary train=0/test=1 indicator" (i.e. exactly 2 tied values) — its primary documented use case. Demonstrated on a synthetic leaky feature: this module reports correlation 0.996 vs. `scipy.stats.spearmanr`'s tie-aware 0.865 for the identical data, a ~0.13 absolute (≈15% relative) systematic inflation. | Use `scipy.stats.rankdata(method="average")` (or an equivalent midrank scatter) whenever `split_labels`'s cardinality is low relative to `n` (the exact regime the docstring calls out), or document the approximation's error bound explicitly and narrow the docstring's claimed use case. | Differential property test: for every custom rank-correlation helper in the repo, compare its output against `scipy.stats.spearmanr` on both a continuous fixture and a heavily-tied (≤5 distinct values) fixture, asserting divergence stays under a fixed tolerance (e.g. 0.02) on the tied case, not just the continuous one. |
| COMPETITION_EVALUATION-5 | P2 | `evaluation/constant_group_leak_scan.py:130-131` | `constant_group_target_scan(df, y, candidate_cols=[])` raises `KeyError: 'min_group_variance_ratio'` (confirmed) instead of returning an empty, correctly-columned `DataFrame`, because `pd.DataFrame([])` has no columns for `.sort_values(...)` to find. | Guard: `if not rows: return pd.DataFrame(columns=["column","n_groups","min_group_variance_ratio","worst_group_value","worst_group_size","flagged"])` before the `sort_values` call. | Parametrized test that calls every public "rank_*"/"*_scan" function in `evaluation/` with its list-typed input argument set to `[]` and asserts either a clean empty-DataFrame return or a documented `ValueError` — never a `KeyError`. |
| COMPETITION_EVALUATION-6 | P2 | `evaluation/subpopulation_drift.py:96-97` | `subpopulation_ratio_drift_check` raises `KeyError: 'prevalence_ratio'` (confirmed) when `train_df`/`test_df` have zero effective values for `subgroup_col` (e.g. both frames empty), same `pd.DataFrame(rows).sort_values(...)`-without-empty-guard pattern as COMPETITION_EVALUATION-5. | Same fix pattern: return an explicitly-columned empty `DataFrame` when `all_values` is empty, before the `sort_values` call. | Same meta-test as COMPETITION_EVALUATION-5, applied to `subpopulation_ratio_drift_check` with zero-row `train_df`/`test_df`. |
| COMPETITION_EVALUATION-7 | P2 | `evaluation/subpopulation_drift.py:157-158` | `rank_subpopulation_drift_severity(train_df, test_df, subgroup_cols=[])` raises `KeyError: 'drift_severity_score'` (confirmed) instead of returning an empty ranking. | Same fix pattern as COMPETITION_EVALUATION-5/6, applied to this function's `ranking = pd.DataFrame(rows)` / `.sort_values("drift_severity_score", ...)`. | Same meta-test as COMPETITION_EVALUATION-5, applied to `rank_subpopulation_drift_severity` with `subgroup_cols=[]`. |
| COMPETITION_EVALUATION-8 | P2 | `evaluation/subgroup_feature_overfit_risk.py:165-166` | `rank_subgroup_feature_overfit_risk(train_df, test_df, candidates=[])` raises `KeyError: 'risk_score'` (confirmed) instead of returning an empty ranking — same recurring bug class as COMPETITION_EVALUATION-5/6/7 (4th independent occurrence in this cluster). | Same fix pattern, applied to this function's `ranking = pd.DataFrame(rows)` / `.sort_values("risk_score", ...)`. | Same meta-test as COMPETITION_EVALUATION-5, applied to `rank_subgroup_feature_overfit_risk` with `candidates=[]`; given 4 independent occurrences, consider a `code_audit` scanner rule: flag any `pd.DataFrame(rows)` (where `rows` is a `list` built by a preceding loop) immediately followed by `.sort_values(...)` without an intervening `if not rows:` guard. |
| COMPETITION_EVALUATION-9 | P2 | `evaluation/_bootstrap_fused_binary_bundle.py:288-320` | `bootstrap_auc_brier_ll_ece_batch` drops non-finite per-resample values (`finite = raw[np.isfinite(raw)]`) with no failure-count logging, unlike its documented "bit-identical" sibling `bootstrap_metrics`, which explicitly `log_throttle`s a warning once resample failures exceed 25% (relevant here because AUC on an unstratified resample of a rare/imbalanced binary target can legitimately collapse to a single class and go `NaN` at a non-trivial rate). A caller relying on the module's own "bit-identical to `bootstrap_metrics`" docstring claim gets a silently narrower/biased CI with no diagnostic signal on exactly the class-imbalanced inputs where this matters most. | Track a per-metric failure count in the two `_bootstrap_batch_*` njit kernels (or count `~np.isfinite(raw)` in the Python wrapper) and call the same `log_throttle(... "bootstrap_metrics_resamples_failed" ...)` path `bootstrap_metrics` uses when failures exceed `n_bootstrap // 4`. | Differential test: run both `bootstrap_auc_brier_ll_ece_batch` and `bootstrap_metrics` on a captured-logging fixture with a rare-positive (~1%), small-`n`, unstratified setup chosen to produce a high single-class-resample rate, and assert both emit a comparable failure-rate warning. |
| COMPETITION_EVALUATION-10 | P3 | `evaluation/reports.py:90-92` | `get_predicted_classes`'s docstring doctest is malformed: a `>>>_,preds=...;preds` line immediately followed by a second `>>>preds` line with the expected output attached only to the second — the first statement's own `preds` echo has no expected-output block, so this doctest would fail if ever collected (e.g. via `pytest --doctest-modules`). | Rewrite as one clean `>>> _, preds = get_predicted_classes(...)` / `>>> preds` / expected-output triple. | Add `--doctest-modules` (or `xdoctest`) to the OSS-hygiene lint pass so a malformed doctest fails CI instead of bit-rotting silently. |
| COMPETITION_EVALUATION-11 | P3 | `evaluation/reports.py:312-313` | Dead commented-out debug lines (`# print(mes)`, `# logger.info(mes)`) left in `evaluate_estimators`, immediately above the `display(Markdown(...))` call that superseded them. | Delete the two commented-out lines. | Grep-based hygiene rule: flag `^\s*#\s*(print|logger\.\w+)\(` lines outside of clearly-marked "example usage" docstring blocks. |

## Narrative detail

**COMPETITION_EVALUATION-1** (`group_leakage_guard.py`). The function's own docstring states: "Consumed
eagerly (materialized to a list) so a generator can still be checked without exhausting it for the caller."
The implementation, however, does `for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):` directly on
the parameter with no `list(...)` conversion anywhere in the function. I reproduced this live: constructing a
one-shot generator of `(train_idx, test_idx)` tuples, calling `assert_no_group_leakage(gen, groups)`, then
calling `list(gen)` afterward returns `[]` — the generator is fully consumed. Any caller who follows the
documented contract (build a CV-splitter generator once, pass it to this guard, then reuse the same generator
object for the actual training loop — a natural pattern for `KFold(...).split(X, groups=...)`-style one-shot
generators) will see their downstream CV loop silently execute zero folds with no exception and no warning.
This is a "silently wrong result" class bug in a function whose entire purpose is a safety check meant to run
before every nested/child-table CV fit.

**COMPETITION_EVALUATION-2** (`reports.py`, `evaluate_estimators`). The `classification_thresholds` parameter
is typed `Optional[list]` and used in the regressor branch as `_thresholds_arr =
np.asarray(classification_thresholds)` fed to `get_predicted_classes(..., thresholds=_thresholds_arr)` — a
sorted array of bin-edge thresholds is the correct call shape there. But in the classifier branch (reached
first when `is_classification` is True), the SAME parameter is passed to
`pd.Series(y_test_test).map(classification_thresholds)`, which requires a dict-like/callable argument;
`pandas.Series.map()` raises `TypeError: 'list' object is not callable` when given a plain list (confirmed
standalone: `pd.Series([0,1,2]).map([10,20,30])` raises exactly this). I reproduced the full crash end-to-end
with a real `LogisticRegression` classifier and `classification_thresholds=[0.0, 0.5, 1.0]` (a value matching
both the parameter's own type annotation and the regressor branch's usage pattern) — `evaluate_estimators`
crashes with `TypeError: 'list' object is not callable` as soon as it reaches line 289. No test in the repo
exercises this parameter with a classifier estimator (confirmed via `grep -rn classification_thresholds`,
which shows only regressor-context and unrelated-module usages).

**COMPETITION_EVALUATION-3** (`expanding_window_leakage.py`). `detect_expanding_window_feature_leakage`
internally sorts `df` by `time_col` before running its expanding-window folds, then maps each fold's flagged
validation-slice positions back to ORIGINAL (pre-sort) row positions via `inverse_order`. The `auto_remediate`
path reports `leaky_row_ranges` as `(orig_positions[0], orig_positions[-1] + 1)` — i.e. it takes the min and
max original position touched by the fold and calls that a "range," per the docstring: "the exact rows whose
'held-out' score was inflated by future information." This is only a valid contiguous range when the original
`df` row order already equals time order. I built a concrete repro (200 rows, original row order a random
permutation of the time order, with a genuine leak: `fit_transform_fn` encodes a category whose true predictive
value is only recoverable from future-inclusive statistics) — `leak_detected` correctly fired (`inflation
≈ 169`), but the reported `leaky_row_ranges` were `[(4, 199), (6, 198), (1, 200), (0, 191), (2, 195)]` on a
200-row frame: every single fold's "range" spans essentially the ENTIRE dataset, even though each fold's
actual validation slice is only ~1/5 of the rows. A caller trying to act on this diagnostic (e.g. "drop these
row ranges and only recompute the feature for them") would be told to touch nearly every row in the dataset.

**COMPETITION_EVALUATION-4** (`leak_scan.py`, `scan_temporal_leak`/`_rank_columns`). `_rank_columns`'s
docstring explicitly acknowledges it skips tie-averaging and calls this "acceptable for a screening diagnostic
on continuous/near-continuous id/date-like columns" — but the caller, `scan_temporal_leak`, documents its
`split_labels` parameter as accepting "a binary train=0/test=1 indicator," which is the opposite of
continuous: exactly 2 distinct values, i.e. maximal possible tie density. I built a fixture with `n=2000`,
`split_labels` a clean 50/50 binary train/test indicator, and a feature deliberately correlated with row order
(the classic "id/date-like leak" pattern the module's own docstring targets). `scan_temporal_leak` reported
correlation `0.996`; `scipy.stats.spearmanr` (which correctly midrank-averages ties) reported `0.865` on the
IDENTICAL data — a 0.13 absolute, ~15% relative overstatement. Because `np.argsort`'s tie-breaking for equal
values is not random but follows the array's own internal structure, the non-averaged ranks partially reproduce
whatever ordering correlates with the leaky feature, systematically inflating (not just adding noise to) the
reported correlation for exactly this heavily-tied primary use case. A user relying on this scanner's absolute
correlation values to compare or triage multiple flagged columns, or using a threshold near 0.9-0.95 as a
"very confident leak" cutoff, gets a systematically distorted read on precisely the split-label shape the tool
is built around.

**COMPETITION_EVALUATION-5 through -8** (recurring `KeyError` on empty list input). Four independent functions
across `evaluation/` build a `list` of per-row `dict`s (`rows`), construct `pd.DataFrame(rows)`, and
immediately call `.sort_values("<some_column>", ...)` with no guard for `rows == []`. `pd.DataFrame([])`
produces a `DataFrame` with NO columns at all (confirmed: `pd.DataFrame([]).columns` is empty), so
`.sort_values("<any column name>")` always raises `KeyError` on that column name rather than returning a
sensibly-typed empty result. I confirmed all 4 occurrences empirically in one script:
`constant_group_target_scan(df, y, candidate_cols=[])` → `KeyError: 'min_group_variance_ratio'`;
`subpopulation_ratio_drift_check(train_df, test_df, 'g')` with both frames having zero effective values for
`'g'` → `KeyError: 'prevalence_ratio'`; `rank_subpopulation_drift_severity(train_df, test_df, [])` →
`KeyError: 'drift_severity_score'`; `rank_subgroup_feature_overfit_risk(train_df, test_df, [])` →
`KeyError: 'risk_score'`. All four are one-line-fixable with an early-return guard, and all four are
reachable from realistic caller code (an upstream candidate-generation step that happens to produce zero
candidates, or an empty/filtered subgroup column) — this is exactly the kind of empty-input edge case the
audit brief calls out, occurring 4 times independently rather than once.

**COMPETITION_EVALUATION-9** (`_bootstrap_fused_binary_bundle.py`). `bootstrap_metrics` (the generic path in
`bootstrap.py`) explicitly tracks a per-metric `failures` counter and calls `log_throttle(...)` with a
`WARNING` once failures exceed `n_bootstrap // 4`, specifically so an operator sees precision degradation
rather than a misleadingly narrow CI. `bootstrap_auc_brier_ll_ece_batch` (the fused fast-path introduced to
replace `bootstrap_metrics` for the common roc_auc/brier/log_loss/ece bundle, and documented as
"Bit-identical to `bootstrap_metrics(...)`") instead does `finite = raw[np.isfinite(raw)]` per metric with no
failure counting or logging at all — the only failure signal is `results[name] = {"error": ...}` in the
all-resamples-failed case. AUC specifically can go `NaN` on an unstratified bootstrap resample of an
imbalanced binary target whenever a resample happens to draw zero examples of one class (`tmp = tps * fps * 2`
→ division guarded to `NaN`, not an exception) — a realistic scenario at the project's own documented
"rare-1pct needs n>=5000" scale with smaller `n`. Because this fast path is now the DEFAULT route for the
roc_auc/brier/log_loss/ece bundle (per the module's docstring and the "PERF WIN (2026-08-04)" project history
noting it "routes nearly every `_bootstrap_block` call through this module"), the missing warning is a real,
silent regression in observability versus the generic path it replaced for the majority of real calls.

**COMPETITION_EVALUATION-10/11** (hygiene). Minor stand-alone items: a malformed doctest in
`get_predicted_classes` (two consecutive `>>>` lines where the first statement's own echoed output has no
expected-output block attached) that would fail were doctests ever collected for this module, and two
dead commented-out debug lines (`# print(mes)`, `# logger.info(mes)`) immediately preceding the `display(...)`
call that replaced them in `evaluate_estimators`.

## Explicitly-checked, zero-finding dimensions

- **Mutable default arguments**: none found (`grep`-verified across both directories: no `def f(x=[])`/`def
  f(x={})` patterns).
- **Bare/overly-broad `except`**: none found (`grep`-verified: zero `except:` or `except Exception:` in either
  directory; every `except Exception as exc:` site logs the exception at `debug`/`warning` before continuing,
  consistent with the project's error-swallowing-hygiene convention).
- **Unseeded RNG / reproducibility**: every stochastic function reviewed (bootstrap resampling,
  `distribution_matching_subset_search`, `optimize_group_blend_weight`, `AdversarialValidator`,
  `build_test_like_validation_fold`) takes an explicit `random_state`/`seed` and threads it through
  `np.random.default_rng(...)` or sklearn's `random_state=` consistently; no hidden global-RNG usage found.
- **GPU/CPU dispatch correctness**: the one GPU-adjacent guard in this cluster
  (`bootstrap_metrics`'s `callable_looks_gpu_bound` check before parallelizing) is correct and matches the
  project's documented joblib/GPU-contention convention; no dispatch bug found.
- **Sample-weight threading**: no function in this cluster accepts or is expected to thread `sample_weight`
  (none of the reviewed diagnostics/tricks are weighted-loss-fitting code paths); not applicable.
