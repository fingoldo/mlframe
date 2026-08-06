# Audit: calibration

**Scope:** `src/mlframe/calibration/**` (probability/quantile calibration methods) — every `.py` file
under this tree, including the `_benchmarks/` subpackage. `feature_selection/filters/**` and
`feature_selection/shap_proxied_fs/**` are out of scope per the assignment (already audited
2026-07-25) and were not touched.

**Files reviewed:** 45 (`.py` files, `__pycache__` excluded) — all read in full.
**LOC reviewed:** 7,154 (per `wc -l` over the same file set).

**Files:**
`__init__.py`, `policy.py`, `post.py`, `_post_train_calibrators.py`, `quality.py`, `probabilities.py`,
`ensembling.py`, `threshold_optimizer.py`, `sticky_state_persistence_floor.py`,
`prediction_band_correction.py`, `smoothed_override.py`, `smoothed_override_backtest.py`,
`group_bias_correction.py`, `confidence_shrinkage.py`, `asymmetric_rescale.py`, `isotonic_risk.py`,
`group_zero_sum_constraint.py`, `_ktc_dispatch.py`, `_independence_check.py`, plus 26 files under
`_benchmarks/`.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| CALIBRATION-1 | P1 | `group_bias_correction.py:20-35` (`_canonical_group_key_series`), used by `fit_group_bias_correction`/`apply_group_bias_correction` | For non-floating (object/string) `group` dtype, a genuine missing group label (`None`/`np.nan`) is stringified to the literal string `"None"`/`"nan"` instead of being preserved as NaN — silently included as a real group in the fitted ratio table and matched at apply time, contradicting the function's own docstring and its own logged warning message. | In `_canonical_group_key_series`'s non-floating branch, detect `pd.isna` per-element the same way the floating branch does and leave those positions as an actual `np.nan` (not `str(None)`) before building the `pd.Series`. | Property test: for every group-bearing calibration `fit_*`/`apply_*` pair, build an object-dtype group array containing `None`/`np.nan`, assert the resulting NaN mask survives canonicalization (`pd.isna(canonical) == pd.isna(original)`) and that fitted-table keys never include a `"nan"`/`"None"` string. |
| CALIBRATION-2 | P1 | `policy.py:442-498` (`_bootstrap_ece_with_indices`), reached from `pick_best_calibrator` when `selection="same_oof"` | The idx-aware fused-njit bootstrap path (`n_bins is not None` branch) feeds the njit kernel `y_true` **without** running it through `_normalize_binary_labels`, while the point estimate two lines above (`metric_fn(y_true, y_pred)` → `_ece_score`) does normalize. For non-`{0,1}`-encoded labels (`{-1,+1}`, `{1,2}`, …) the bootstrap CI is computed on a completely different numeric scale than the point estimate it is supposed to bracket. | Normalize `y_true` via `_normalize_binary_labels` once before building `yb` in the `n_bins is not None` branch, mirroring what `_ece_score`/the point estimate already do. | Unit test: call `pick_best_calibrator(..., selection="same_oof")` with `oof_y` encoded as `{-1, +1}` vs. the same data encoded as `{0, 1}`; assert the reported `ece_ci` for every candidate is (near-)identical between the two encodings, the same invariant `_normalize_binary_labels`'s own docstring already promises for the point estimate. |
| CALIBRATION-3 | P2 | `sticky_state_persistence_floor.py:44-61` (`apply_sticky_state_persistence_floor`) | A per-class `floor` vector's length is never validated against `k` (number of classes) before `floor_arr[active]` is indexed — a mismatched-length vector raises an opaque `IndexError` deep inside fancy indexing instead of a clear `ValueError`, unlike every other shape mismatch in this module which is checked explicitly. | Add `if floor_arr.ndim == 1 and floor_arr.shape[0] != k: raise ValueError(...)` alongside the existing range check. | Parametrized edge-case test: call with `floor=np.ones(k-1)` (and `k+1`) and assert a `ValueError` naming the shape mismatch, not a bare `IndexError`. |
| CALIBRATION-4 | P2 | `policy.py:600-651` (`_emit_reliability_plot`) called unguarded at `policy.py:871` | `build_reliability_overlay_spec(...)` (spec construction) is **not** wrapped in the function's try/except — only the later `render_and_save` call is. A spec-build failure (e.g. a malformed `calibrated_probs` shape from one candidate) propagates all the way out of `pick_best_calibrator`, crashing the entire calibrator-selection call even though selection itself already succeeded and `emit_plot` is documented as an opt-in diagnostic extra. | Widen the try/except in `_emit_reliability_plot` to also cover the `build_reliability_overlay_spec` call (or wrap the whole function body), returning `None` + a warning on any exception, consistent with how the `os.makedirs`/render failures are already handled. | Regression test: monkeypatch `build_reliability_overlay_spec` to raise, call `pick_best_calibrator(..., emit_plot=True)`, assert it still returns a normal result dict with `plot_path=None` instead of propagating the exception. |
| CALIBRATION-5 | P2 | `policy.py:411-418` (`_build_resample_indices`, non-stratified branch) | The `(n_bootstrap, n)` resample-index matrix is filled via a Python `for b in range(n_bootstrap): out[b] = rng.integers(...)` loop — one small `rng.integers` call per row — instead of a single vectorized `rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)` call. This is exactly the "per-iteration Python dispatch instead of one fused call" inefficiency this same codebase calls out and fixes elsewhere (e.g. the documented bootstrap-AUC fusion). The stratified branch below has an explicit comment defending its own per-`(b,c)` loop as RNG-draw-order-critical; the non-stratified branch has no such justification and a single vectorized call over the same `Generator` instance preserves the identical draw order (NumPy's `Generator.integers` fills multi-dim output row-major from one continuous draw stream). | Replace the loop with one `rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)` call in the non-stratified branch; verify bit-identity against the current loop before shipping. | A/B bench + identity test: assert the vectorized draw is bit-identical to the current per-row loop for several `(n, n_bootstrap, seed)` combinations, then measure wall-time at `n_bootstrap=1000`. |
| CALIBRATION-6 | P3 | `post.py:111-115` vs. `post.py:183-186, 215-216` (`BinaryPostCalibrator`) | `fit()` dynamically sets `self.classes_`, `self.n_features_in_`, and (Venn-Abers branch) `self.y_cal`/`self.p_cal`, but only `_resolved_transform_method_name` is declared in the class-level type annotations block — violating this repo's own "declare dynamically-set attributes at class scope" mypy-cleanliness convention. | Add `classes_: np.ndarray`, `n_features_in_: int`, `y_cal: Optional[np.ndarray]`, `p_cal: Optional[np.ndarray]` to the class-level annotation block. | Static check: an AST/mypy-based scanner that flags any `self.<name> = ...` assignment inside a method whose `<name>` has no matching class-level annotation anywhere in the class body. |
| CALIBRATION-7 | P3 | `post.py:36-38` (`_INCLUDE_RE`) | A module-level sentinel (`_INCLUDE_RE: "re.Pattern" = re.compile("")`) is dead code — never read anywhere in the module — kept, per its own comment, purely so an (implied) source-inspection meta-test can confirm a refactor "landed." This is the exact anti-pattern this repo's own convention explicitly rejects (behavioral tests over `inspect.getsource()`/attribute-presence checks). | Remove the sentinel; if a regression test for the precompile-cache refactor is wanted, assert on behavior instead (e.g. `_compile_pattern` is an `lru_cache`d function and repeated calls with the same pattern return the identical `re.Pattern` object). | Grep-based scanner: flag any module-level variable whose only justification (per its own comment) is "so a test can confirm X landed" — a smell that the corresponding test is source-inspecting rather than behavioral. |
| CALIBRATION-8 | P3 | `quality.py:271-276, 319-334` (`estimate_calibration_quality_binned`, `show_classifier_calibration`) | Both functions default `metrics_to_show: dict = METRICS_TO_SHOW` — a mutable, module-level dict object used directly as a parameter default (the classic mutable-default-argument footgun this audit's own checklist calls out). Currently harmless (neither function mutates the dict, only iterates `.items()`), but any future edit that does (`.pop`, `.update`, adding a computed key) would silently corrupt the shared global for every caller relying on the default. | `metrics_to_show: Optional[dict] = None` + `metrics_to_show = metrics_to_show if metrics_to_show is not None else METRICS_TO_SHOW` inside the function body (or `.copy()` the default at the call boundary). | AST scanner: flag any `def f(..., param: dict = <module-level-name>, ...)` where the default is a `Name` reference to a mutable module-level literal (dict/list/set), not just a literal `{}`/`[]` default. |
| CALIBRATION-9 | P3 | `quality.py:228-268` (`bin_predictions`), guarded (incompletely) by `estimate_calibration_quality_binned:302` | The njit kernel computes `bin_size = s // nbins` with no guard against `nbins <= 0`. The caller clamps `nbins = min(nbins, n_samples)` but never validates `nbins >= 1`, so a caller-supplied `nbins=0` (or a negative value, since `min` doesn't floor it either) reaches the compiled kernel and raises an opaque `ZeroDivisionError` from inside `@njit` code instead of a clear `ValueError` at the Python boundary. | Add `if nbins < 1: raise ValueError(...)` in `estimate_calibration_quality_binned` before the `min(nbins, n_samples)` clamp. | Edge-case test: call `estimate_calibration_quality_binned(y, p, nbins=0)` and `nbins=-1`, assert a `ValueError` is raised (not a `ZeroDivisionError` surfacing from compiled code). |
| CALIBRATION-10 | P3 | `threshold_optimizer.py:72-75` (`_threshold_stability_report`) | `coeff_of_variation = std / mean` is not protected against `mean < 0`. With a caller-supplied `threshold_range` that spans negative values (a legitimate use for e.g. a cost-score threshold, not just probability cutoffs), a negative mean fold-threshold can sign-flip the ratio negative, which then trivially passes `is_stable = coeff_of_variation <= stability_cv_threshold` regardless of the true relative spread of the fold thresholds. | Use `abs(mean)` in the denominator (matching the pattern already used in `asymmetric_rescale.cross_validate_asymmetric_rescale`'s own `factor_cv` and `prediction_band_correction`'s `relative_std`, both of which divide by `abs(...)`). | Property test: construct fold thresholds with a negative mean and a genuinely large spread; assert `is_stable` is `False` (currently would report `True` due to the sign flip). |
| CALIBRATION-11 | P3 | `group_zero_sum_constraint.py:75-84, 116-128` (`apply_group_zero_sum_constraint_multi` docstring/algorithm) | The docstring claims the multi-constraint solver works "exactly like Dykstra's alternating projection algorithm onto multiple convex sets." The per-constraint step it alternates (`apply_group_zero_sum_constraint`) applies a **constant** per-group shift (by design, to preserve within-group rank order), not the true weighted L2-orthogonal projection onto the affine constraint set (which — for a constraint on a *weighted* sum with an *unweighted* L2 objective — would distribute the correction proportionally to each row's weight, not uniformly). Verified empirically that the iteration still converges to a jointly-feasible point (residuals ~1e-11 after enough sweeps even with non-uniform weights), so this is a documentation/naming precision issue, not a functional failure: the algorithm delivers what its own "constant shift" design promises, but "Dykstra" over-claims a minimal-distortion optimality property it doesn't have when weights are non-uniform. | Reword the docstring to describe the method as "alternating affine correction" (or similar), dropping the "exactly like Dykstra's...algorithm" framing, or note explicitly that the per-set operator is a rank-preserving constant shift rather than an orthogonal projection. | N/A (documentation-only); if kept as a functional claim, a property test asserting the multi-constraint output equals the true weighted-least-squares solution under non-uniform weights would fail and should not be added without fixing the algorithm to match. |
| CALIBRATION-12 | P3 | `quality.py:252,614`; `policy.py:247-248,284-285`; `__init__.py:13`; `post.py:544,618,674`; `_post_train_calibrators.py:284`; `_benchmarks/bench_compare_postcalibrators_parallel.py:16` | Pervasive process/audit metadata embedded directly in source comments — `Wave 21 P2`, `Wave 47`, `Wave 19 P1`, `iter309`/`iter308`/`iter598`/`iter595/596/597`/`iter631`, and bare finding IDs (`P1-5`, `P2-1`, `P1-4`) — which this repo's own CLAUDE.md explicitly forbids ("No process/audit metadata in code comments: no phase/wave markers, finding IDs, date stamps... that belongs in git history / the PR description"). | Strip the wave/iter/finding-ID prefixes from these comments, keeping only the substantive WHY (the numeric/technical justification that follows each marker is usually fine to keep). | Grep-based scanner (easy to centralize): flag any comment matching `Wave \d+|iter\d{3,}|P[0-3]-\d+` inside `.py` source (excluding `CHANGELOG`/git-history files), fail CI on new occurrences. |
| CALIBRATION-13 | P3 | Pervasive across nearly every file in the cluster (124 occurrences across 24 files, e.g. `policy.py` x19, `post.py` x14, `_post_train_calibrators.py` x12) | Docstrings/comments throughout the cluster use a literal `" -- "` as an em-dash substitute in prose, which this repo's own CLAUDE.md convention (`Never use -- in prose (CRITICAL)`) explicitly disallows in favor of a single ` - `. | Repo-wide (or at least per-file, on next edit) mechanical replacement of `" -- "` with `" - "` in comments/docstrings — do NOT run this as an unscoped repo-wide rewrite without explicit approval per this repo's own "no repo-wide reformat" rule; batch it into the same changeset as other touches to each file. | Grep-based scanner: flag ` -- ` (space-dash-dash-space) inside triple-quoted docstrings/`#` comments; easy to run as a pre-commit hygiene check limited to touched files. |

**Severity counts:** P0: 0 · P1: 2 · P2: 3 · P3: 8 (13 total)

## Dimension coverage notes

- **Data leakage / calib-test overlap:** `train_postcalibrators`'s calib==test overlap guard
  (`_post_train_calibrators.py`, using `_CalibTestOverlapError`/`_values_overlap_fraction` from
  `post.py`) is thorough — checks exact-array equality, reshuffled-multiset equality, row-index
  overlap, and probability-row overlap independently of the target check. No leakage-guard gaps found.
- **Sample-weight threading:** `BinaryPostCalibrator.fit`/`compare_postcalibrators`/
  `train_postcalibrators` all thread `sample_weight` correctly with an explicit
  `inspect.signature`-based capability check and a logged warning when a wrapped calibrator can't use
  it. No gaps found.
- **Reproducibility / RNG:** every RNG use in the cluster goes through a per-call
  `np.random.default_rng(seed)` (or `sklearn.utils.check_random_state`); no unseeded global
  `np.random` calls or process-global RNG mutation were found (the docstring in `probabilities.py`
  explicitly documents having fixed exactly this class of bug previously). No new instances found.
- **Class-imbalance / single-class handling:** systematically guarded throughout (`_normalize_binary_labels`,
  `_stratified_inner_folds`, `compare_postcalibrators`'s 2-class check, `optimize_persistence_floor*`,
  `compute_oof_confidence`'s empty-class fallback to neutral `1.0`). No gaps found beyond CALIBRATION-2.
- **GPU/CPU dispatch (`ensembling.py`, `_ktc_dispatch.py`):** the KTC-based dispatcher, its
  `MLFRAME_ODDS_COMBINE_BACKEND` override, and the try/except GPU-failure-falls-back-to-CPU path were
  all read and are correctly wired; no bug found.
- **Numerical stability of the correlation estimator (`_independence_check.py`):** the closed-form
  member-vs-consensus correlation uses a single-pass `E[X^2]-E[X]^2` variance formula — the same
  *shape* of computation this codebase has hit catastrophic-cancellation bugs in three times before
  (skew/kurtosis on large-offset targets). Empirically stress-tested with a "large offset, tiny
  spread" logit scenario (offset~16, spread~0.02, the most extreme case the bounded `[clip, 1-clip]`
  logit range permits) and found the precision loss negligible (~1.8e-9 vs. a direct `np.corrcoef`
  reference) — the bounded dynamic range of a clipped logit prevents the offset/scale ratio from
  reaching the catastrophic regime seen in unbounded regression targets. Not flagged as a finding;
  documented here since a reviewer double-checking this bug class would otherwise re-derive the same
  question.
- **Test coverage:** `tests/calibration/` has 44 files with solid `biz_val`/edge-case/regression
  coverage. Confirmed via grep that CALIBRATION-1 (object-dtype NaN group canonicalization) and
  CALIBRATION-2 (non-0/1 label encoding under `same_oof` bootstrap) have no corresponding test anywhere
  in the suite — both are genuine coverage gaps, not just latent bugs with a masking test.

## Narrative detail

**CALIBRATION-1** (`group_bias_correction.py`): `_canonical_group_key_series`'s docstring promises "A
genuine NaN group label is preserved as NaN... so it still falls through pandas' own dropna-groupby /
dict-lookup-miss behavior unchanged," and `fit_group_bias_correction` logs a warning claiming excluded
NaN rows "will silently receive default_ratio at apply time." Both claims are false for non-floating
`group` dtypes. Reproduced directly: `group = np.array(['a','b', None, 'a', 'b', None], dtype=object)`
→ `_canonical_group_key_series(group)` returns `['a', 'b', 'None', 'a', 'b', 'None']` with
`pd.isna(...)` all `False`; `fit_group_bias_correction` then returns a ratio table containing a real
`'None'` key computed from those two rows, and `apply_group_bias_correction` on a fresh array with the
same `None` marker maps to that computed ratio instead of the documented `default_ratio` fallback. The
same reproduces with a literal `np.nan` sentinel in an object array (a very plausible real shape: any
pandas `object`/string categorical column with missing values, e.g. `df['store_id'].to_numpy()`). Only
the `np.issubdtype(arr.dtype, np.floating)` branch (a pure-float array) gets the documented NaN-safe
behavior; everything else — the common case for a categorical group key — does not.

**CALIBRATION-2** (`policy.py`): Traced the two ECE code paths side by side. The point estimate
(`metric_fn(y_true, y_pred)` at `_bootstrap_ece_with_indices:466`) calls `_ece_score`, which always
normalizes `y_true` via `_normalize_binary_labels` before the njit kernel. The per-resample loop three
lines later, when `n_bins is not None` (the fused idx-aware fast path `pick_best_calibrator` always
uses), builds `yb = np.ascontiguousarray(np.asarray(y_true).ravel())` and passes that RAW array
straight into `_ece_score_idx_numba_serial`, which accumulates `sum_y[b] += yi` assuming `yi ∈ {0,1}`.
For labels like `{-1, +1}`, the point estimate correctly remaps them to `{0,1}` first, but every
bootstrap resample's `sum_y` accumulates raw `-1`/`+1` values — an entirely different quantity, not a
resample of the point estimate's own statistic. This only affects `selection="same_oof"` (the
`inner_cv` default recomputes and overwrites both `rank_ece` and `ece_ci` from
`_heldout_ece_inner_cv`/`_heldout_ece_ci`, which do call the normalizing `_ece_score`), but `same_oof`
is a fully supported, documented (if "legacy") public option, and the corrupted CI feeds directly into
the CI-overlap tie-break logic (`_cis_overlap`) that decides whether the Kull-2017 default rule fires.

**CALIBRATION-3** (`sticky_state_persistence_floor.py`): Every other shape/range violation in
`apply_sticky_state_persistence_floor` (scalar-floor range, `active_class` bounds) raises a clear
`ValueError` with a message; the per-class floor vector's length is the one dimension left
unvalidated, so `floor_arr[active]` at line 61 is the failure site for a caller who passes e.g. a
floor array sized for the wrong number of classes — a plausible mistake when `k` is inferred
differently by two different call sites in a caller's own code.

**CALIBRATION-4** (`policy.py`): Read `_emit_reliability_plot` end to end: the `try/except OSError`
block only wraps `os.makedirs` and `render_and_save`; `build_reliability_overlay_spec` (which touches
every candidate's `calibrated_probs`/label array) executes outside any handler, and `pick_best_calibrator`
itself calls `_emit_reliability_plot` with no enclosing try/except at all (line 871). Every other
optional/diagnostic step in this module (secondary-ECE scoring, per-candidate fit/predict/bootstrap)
is independently try/excepted with a warning-and-continue; the plotting step is the one place that
isn't, despite being explicitly documented as an opt-in extra that "returns... None if the render
dependency is missing or the write fails" — implying failure tolerance that the code doesn't fully
deliver.

**CALIBRATION-5** (`policy.py`): Read `_build_resample_indices`'s two branches side by side. The
stratified branch has an explicit multi-line comment defending why it stays as a Python double-loop
(preserving the exact per-`(b, c)` draw order downstream CIs are pinned to). The non-stratified branch
has no equivalent justification and no correctness reason apparent from reading it — `np.random.Generator`
methods consume entropy from the underlying bit generator in a fixed sequential stream regardless of
whether the caller requests it via N separate `size=n` calls or one `size=(N, n)` call, so a single
vectorized call over the same `Generator` object should produce bit-identical draws while removing
`n_bootstrap` (default 1000) Python-level dispatch overheads — this is the exact "GIL-bound
per-resample dispatch loop, fuse into one call" pattern already fixed in several other locations of
this codebase.

**CALIBRATION-6 through CALIBRATION-13** are code-quality / hygiene items surfaced during the
line-by-line read (mutable default argument, dead sentinel code, unvalidated `nbins`, unguarded
`mean<0` sign flip, an over-claimed algorithm name, and pervasive process-metadata / `--`-in-prose
comment style violations of this repo's own documented conventions). None reproduce a wrong numeric
result in the code paths actually exercised by the existing test suite; they are included per the
audit's explicit instruction to capture every real P3 rather than fold them away.

## Report path

`C:/Users/Admin/Machine learning/mlframe/audits/full_audit_2026-08-05/calibration.md`
