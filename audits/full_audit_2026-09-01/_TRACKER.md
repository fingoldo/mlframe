# mlframe wide audit 2026-09-01 -- master tracker

Read-only parallel audit on a worktree of `bc2de5068`: 9 subsystem clusters plus 3 cross-cutting sweeps by
defect class. Every finding below will be implemented (RESOLVED) or given an explicit non-silent disposition
(FUTURE / DOC / REJECTED) per the multi-agent-review disposition convention. Status starts at TODO for all.

**Totals: 7 P0, 46 P1, 76 P2, 73 P3 -- 202 findings.**

## Per-cluster summary

| Cluster | P0 | P1 | P2 | P3 | Total | Report |
|---|----|----|----|----|-------|--------|
| feature_engineering | 0 | 4 | 6 | 7 | 17 | [feature_engineering.md](feature_engineering.md) |
| fs_filters_mrmr | 0 | 1 | 5 | 5 | 11 | [fs_filters_mrmr.md](fs_filters_mrmr.md) |
| metrics | 1 | 4 | 5 | 5 | 15 | [metrics.md](metrics.md) |
| models_estimators | 3 | 6 | 1 | 2 | 12 | [models_estimators.md](models_estimators.md) |
| preprocessing_data | 2 | 7 | 11 | 8 | 28 | [preprocessing_data.md](preprocessing_data.md) |
| remaining_subsystems | 0 | 3 | 7 | 9 | 19 | [remaining_subsystems.md](remaining_subsystems.md) |
| training_core | 0 | 3 | 3 | 8 | 14 | [training_core.md](training_core.md) |
| training_reporting_targets | 0 | 4 | 5 | 6 | 15 | [training_reporting_targets.md](training_reporting_targets.md) |
| xcut_numerical_stability | 0 | 1 | 3 | 2 | 6 | [xcut_numerical_stability.md](xcut_numerical_stability.md) |
| xcut_swallowed_failures | 1 | 5 | 16 | 4 | 26 | [xcut_swallowed_failures.md](xcut_swallowed_failures.md) |
| reporting | 0 | 1 | 5 | 12 | 18 | [reporting.md](reporting.md) |
| xcut_test_quality | 0 | 7 | 9 | 5 | 21 | [xcut_test_quality.md](xcut_test_quality.md) |

## The seven P0s

| ID | File | One line |
|---|---|---|
| METRICS-1 | `metrics/classification/_threshold_optimization.py:212` | Bootstrap CI on the tuned threshold sweeps each resample in RANDOM order while its kernel's incremental confusion counts are valid only in descending-score order. Measured 8.7x too wide on a well-identified threshold. Feeds the per-model `95% CI [lo, hi]` log line. No test exists for this function. |
| MODELS_ESTIMATORS-1 | `models/ensembling/selection.py:137` | Default blend metric feeds raw class labels into a kernel that requires strictly 0/1; `{1,2}` or `{-1,+1}` yields NaN, and since `nan > best` is always false the greedy walk silently degenerates to "first model". |
| MODELS_ESTIMATORS-2 | `models/additive_interaction_diagnostic.py:96` | `cv_splits` is iterated per call but documented as an Iterable; a `KFold().split()` generator is exhausted after the first use, giving `np.mean([]) = nan`. The recommendation flag flips True -> False. |
| MODELS_ESTIMATORS-3 | `estimators/early_stopping.py:243` | The `staged` backend's truncation sets a hyperparameter sklearn's `predict` ignores, so early stopping has NO effect and the fully-overfit model is returned while the reported stop looks plausible. |
| PREPROCESSING_DATA-1 | `core/helpers.py:289` | `np.nan_to_num(..., posinf=, neginf=)` without `nan=` rewrites every NaN to 0.0 in any column that happens to contain an infinity. Reproduced. |
| PREPROCESSING_DATA-2 | `preprocessing/auto_transform_select.py:229` | The fold-local transform fit exists to stop scaler statistics leaking across the CV split -- and its docstring says so -- but the NaN median fill one line above the fold loop is computed over the whole column. |
| XCUT_SWALLOWED_FAILURES-1 | `feature_selection/filters/_kernel_tuning.py:93` | `ImportError` is already narrowed one line above, so the residual `except Exception` can only fire on a corrupt tuning file, a concurrent-process file lock, or a genuine bug -- and it latches the singleton off PERMANENTLY, dropping every per-host kernel-tuning lookup in the package to hardcoded defaults, at `debug` level. |

## What the cross-cutting sweeps changed about the picture

**Numerical stability.** All three instances named in the brief are already fixed, including
`_derive_cell_stats`, which `CLAUDE.md:443` and `docs/NUMERICAL_STABILITY_REPORT.md:233` still describe as OPEN
(verified independently). The sweep instead found the same class one order down -- variance from raw power sums,
`var = E[x^2] - E[x]^2` -- at nine live sites in four files. Three prior fix rounds missed these because they
grepped for skew and kurtosis. **The shape is `sum(x^k)` minus a power of the mean for any k >= 2.** The two
stale doc entries should be corrected in the same pass, because it was their three-instance framing that
anchored the earlier searches.

**Swallowed failures.** The fully-unlogged form of this defect is gone from the tree: of 2,359 broad handlers,
only 8 have no logging and all 8 are inert. The live form is different and was not in the brief -- **fail-open
statistical gates**: `_mrmr_fit_impl/_finalise.py:227` substitutes `_p_value = 0.0` when the
permutation-significance probe raises, and the next line is `if _p_value >= alpha: continue`, so a broken
estimator makes every rescue candidate pass a gate that exists precisely because plug-in MI is upward-biased.

## Verification notes

Findings I checked first-hand rather than relaying: XCUT_NUMERICAL_STABILITY-1 (five copies of the variance gate,
`val[i]` already materialised so the centred second pass is free), the `_derive_cell_stats` already-fixed claim,
PREPROCESSING_DATA-1 (reproduced), REPORTING-2 (a defect in code written the same day -- fixed, with a
regression test proven to fail without the fix), and METRICS-1.

METRICS-1 needs a correction to what the agent reported. The defect is real and provable by reading the kernel:
`tn = N - fp` and `fn = P - tp` hold only if the sweep is in descending-score order, and the kernel's docstring
asserts an invariant -- "resampling positions keeps every resample in sorted order for free" -- that is false for
a random position draw. But the reported magnitude (~800x) came from a perfectly separable fixture where the
honest reference is tie-degenerate. On a fixture where the optimal threshold is well identified, the measured
width ratio is **8.7x**, not 800x. The one-line fix (`np.sort(idx, axis=1)`) is unchanged: it preserves the
resample multiset, which is what a bootstrap resample is, and restores the order the kernel requires.


## What the test-quality sweep changed about the picture

The two originally-confirmed defects are genuinely fixed, and the meta-gates are real -- **within their stated
scope**. The damage is in their blind spots, and it is large:

- **322 source-text assertions across 33 files** in `tests/training/test_audit_*.py` are invisible to the
  `getsource` gate because each routes its production-file read through a local `def _read()`. Both scanners key
  on a `read_text()` call bound in the ASSERTING scope, so one level of helper indirection defeats them. These
  assert on exact lambda bodies, on log f-strings including embedded newlines and indentation, and on the
  presence of explanatory comments -- i.e. they break on every harmless refactor while passing for
  implementations that are wrong.
- The single-shot-timing gate only matches a DIVISION compared to a constant; its own detector sample documents
  `assert elapsed < 5.0` as deliberately unflagged. Roughly 40 such sites plus three `cProfile.total_tt < X`
  sites pass through.
- Conversely, the four assertions in `test_biz_val_runtime_budget.py` that ARE genuine `max_runtime_mins`
  contracts `pytest.skip` themselves under xdist -- so on CI, the only place the full matrix runs, they assert
  nothing.
- `tests/training/_fuzz_combo/combo.py:2319` disables RFECV for every non-balanced binary target with the sklearn
  exception text quoted verbatim as justification -- the illegitimate-canon pattern the harness's own docstring
  forbids in capitals.

Two items were verified by execution rather than reading: a `monkeypatch.setattr(..., raising=False)` targeting a
module with zero references to that name (so it silently creates a phantom attribute, and the test passes because
a sibling patch masks it), and a static resolution of all 105 `patch("mlframe...")` targets against an AST symbol
table -- the 9 that fail statically resolve legitimately at runtime through the lazy `__getattr__` table.

Declared incomplete by the agent: an AST pass for non-discriminating assertions returned 903 candidate functions,
of which about 30 were hand-triaged. That is the largest surface the sweep did not exhaust.

## Coverage gaps the agents named

The largest unexamined surfaces, in the agents' own words:

- `feature_selection/filters/`: the MRMR greedy core (`_mi_greedy_cmi_fe.py`, `evaluation.py`,
  `_usability_aware_selection.py`, `_screen_predictors.py`), the GPU-resident materialise/select subsystem, and
  the ~40 `_orthogonal_*_fe.py` families -- swept for the four named bug classes only, never read for selection
  integrity or contract drift.
- `feature_selection/shap_proxied_fs/` (~7,500 LOC): the search / revalidate / objective trio.
- `reporting/`: `diagnostics_dispatch.py` and `_diagnostics_dispatch_extra.py` (1,790 LOC of suite gating logic),
  scanned only for the listed defect families.
- `models/ensembling/`: `score_flavours.py`, `score_validate.py`, `float_aggregation.py`, and
  `base.py:260-990`.
