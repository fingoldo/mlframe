# Composite Targets audit - Performance & Memory dimension (2026-06-10)

Scope: src/mlframe/training/composite/** (discovery/_fit.py, screening.py, _screening_tiny.py,
_tiny_rerank.py, _eval.py, _auto_base.py, _filter.py, ensemble/, transforms/nonlinear.py, cache.py)
+ benchmarks/composite_*.py + training/_benchmarks (oof_cache_reuse, ct_ensemble_residual_dedup).

Default-config call-count context used for severity calibration (verified in
_composite_target_discovery_config.py): mi_sample_n=100_000, mi_nbins=16, mi_estimator="bin",
auto_base_top_k=3, transforms default list = 24 entries (~19 bivariate x 3 bases + 5 unary ~= 60
work items per target), top_k_after_mi=32, screening="hybrid" (rerank ON), tiny_model_sample_n=20_000,
tiny_model_n_seed_repeats=3, tiny_screening_families=("lightgbm","linear"), tiny_model_cv_folds=3,
tiny_model_n_jobs=1, tiny_rerank_n_jobs=0 (auto), mi_gain_bootstrap_n=0, per_bin_n_bins=0.

Known floors NOT re-flagged: _mi_pair_bin numpy floor (documented at screening.py:197-220, numba +
np.partition attempts rejected), pandas per-column to_numpy in data_signature (bench-attempt-rejected
note cache.py:413), _touch_lru O(N)-rewrite-per-touch (documented decision cache.py:630-637),
_quantile_residual_per_bin_stats dispatcher (benched, nonlinear.py:269), EWMA/frac-diff backend ladder
(benched 2026-05-24), OOF cache intentionally unwired on suite path (ensemble/__init__.py:303-317 +
bench_oof_cache_reuse.json), residual dedup default OFF (bench_ct_ensemble_residual_dedup.json),
items in tests/composite_discovery_audit_notes.md.

---

## P1 - severity P1 - perf
file: src/mlframe/training/composite/discovery/_eval.py:162-167
title: Per-work-item dead x_screen_valid float32 matrix copy + unconditional prebinned int64 slice in the default config

x_screen_valid = x_remaining_matrix[valid_screen] (line 162) is computed unconditionally for every
(base, transform) work item. With the default mi_estimator="bin" (_x_prebinned is not None) and
mi_gain_bootstrap_n=0, this array is NEVER used: mi_t (line 165), mi_y_compare (line 183) and the
bootstrap (line 231) all take the prebinned branch. Boolean indexing always allocates, so at default
scale (100k screen rows x n_feat) this is an ~80 MB (n_feat=200) to ~200 MB (n_feat=500) dead copy x
~60 work items per target ~= 5-12 GB of pure allocation churn per discovery fit. Additionally
_x_pb_valid = _x_prebinned[valid_screen] (line 164) copies the full int64 prebinned matrix (2x the
float32 size) even when valid_screen is all-True - which it is for the most common transforms (diff,
additive/median/linear residual on finite data).

fix: Move line 162 inside the else (non-prebinned) branch; gate the prebinned slice on
valid_screen.all() (_x_pb_valid = _x_prebinned if valid_screen.all() else _x_prebinned[valid_screen]
- downstream is read-only). Bit-identical by construction. Bench in benchmarks/composite_profile.py
discovery workload with --n large + mi_sample_n=100k-shaped synthetic.

## P2 - severity P1 - perf
file: src/mlframe/training/composite/discovery/screening.py:313-332
title: _mi_to_target_prebinned copies the whole int64 matrix + per-column double gathers without all-true gates

Line 317 fb_f = feature_binned[finite] boolean-slices the full (n_rows x n_feat) int64 matrix on every
call even when finite.all() (typical: target is y_screen / t_screen which are finite by construction
after the upstream checks). At 100k x 200 cols that is 160 MB per call; the function is called once per
base for the baseline (3x), once per work item for mi_t (~60x), plus the shrunk-domain recompute -
roughly 10 GB churn per fit. Inside the per-column loop, col_b[col_valid] and t_idx[col_valid]
(lines 330-331) allocate two more 100k gathers per column even when col_valid.all() (no -1 sentinel
in the column).

fix: fb_f = feature_binned if finite.all() else feature_binned[finite] (read-only downstream);
per-column "if col_valid.all(): use col_b, t_idx directly". Both bit-identical. Hot by construction:
call count ~= 60-70 per fit x O(n_rows x n_feat) work.

## P3 - severity P1 - perf
file: src/mlframe/training/composite/discovery/_fit.py:349-389
title: Per-base np.delete full-matrix copies + per-base mi_y_for_base recomputed from scratch (decomposable to one per-feature pass)

For each of K base candidates (default 3; more with hints), the loop materialises
np.delete(_full_x_matrix, idx, axis=1) (line 355, ~200 MB float32 at 100k x 500) AND
np.delete(_full_x_prebinned, idx, axis=1) (line 357, ~400 MB int64), then calls
_mi_to_target_prebinned(..., y_screen) (line 371) which re-bins the SAME y_screen and recomputes
per-column MI(x_j, y) for all shared columns. The per-feature MI values are identical across bases -
only which single column is excluded changes. So the K x (n_feat-1) MI-pair computations collapse to
ONE n_feat-length per-feature pass: mi_y_for_base = np.mean(np.delete(per_feat_vec, drop_idx)) (exact
for both "mean" and "sum" aggregation; same per-column MI floats, so bit-identical). At K=3 it wastes
~2/3 of the baseline MI wall (each per-column _mi_from_binned_pair ~= bincount over 100k); with
hint-driven larger base pools it scales linearly. The np.delete copies can also be removed entirely by
giving _mi_to_target_prebinned / eval_one_transform an exclude_col parameter (the per-column loop skips
j == exclude_col), saving ~1.8 GB of transient copies per fit at defaults.

fix: (1) compute per_feat_mi once on the full prebinned matrix (reuse the _mi_per_feature_y_fixed
hoist pattern that _auto_base already uses - measured 1.67x bit-exact); derive per-base baselines
arithmetically. (2) add exclude_col to the MI helpers and drop the np.delete materialisation. Bench:
extend benchmarks/composite_profile.py discovery workload with n_features=200+.

## P4 - severity P1 - perf
file: src/mlframe/training/composite/discovery/_tiny_rerank.py:299-303
title: Default-config thread oversubscription: rerank parallelises up to cpu_count spec workers, each fitting LGBM/XGB with n_jobs=-1

tiny_rerank_n_jobs=0 (config default) auto-picks min(len(kept_specs), cpu_count) joblib threads
(lines 285-297). Each _rerank_one_spec worker runs _tiny_cv_rmse_y_scale_multiseed with
n_jobs=tiny_model_n_jobs=1, so the existing inner-thread cap (if n_jobs > 1: model.set_params(n_jobs=1),
_screening_tiny.py:646) NEVER fires - and _build_tiny_model hardcodes n_jobs=-1 for LightGBM
(_screening_tiny.py:144) and XGBoost (line 161). On a 16-core host the default path runs up to 16
concurrent boosters each requesting 16 OMP threads (256-way demand). The file documents the 4-8x
wallclock blow-up risk for exactly this pattern (lines 304-307), but only covers fold-level
parallelism. The rerank loop is the dominant discovery phase (~32 specs x 2 families x 3 seeds x
3 folds ~= 576 booster fits), so this hits the default config of every hybrid-screening run.

fix: thread inner_n_jobs = max(1, cpu_count() // _rerank_n_jobs) from _tiny_model_rerank down
to _build_tiny_model whenever _rerank_n_jobs > 1 (mirror the fold-level cap). Measure before/after
wallclock on a >=8-core box (32 specs, 20k rows) per feedback_perf_measure_first.

## P5 - severity P1 - perf
file: src/mlframe/training/composite/discovery/_tiny_rerank.py:171-183
title: Rerank rebuilds per-base feature matrices column-by-column from df (B+1 full extraction passes) instead of build-once + delete

The per-base cache loop extracts x_remaining for every unique base via
self._build_feature_matrix(df, x_remaining, train_idx_screen) (line 180), and the raw baseline later
builds x_full from df AGAIN (lines 430-432). Each _build_feature_matrix call runs
_extract_column_array(df, c) per column, which materialises the FULL N-row column (see P6) before
slicing 20k sample rows. With B=3 unique bases + 1 raw pass over ~500 columns on a 4M-row frame, that
is ~2000 full-column materialisations (~16 MB each when astype copies) ~= 30+ GB cumulative allocation
churn - for matrices that differ from each other by exactly one column. _fit.py:330-359 already solved
this exact problem in Phase A with the build-full-matrix-once + np.delete pattern (comment at line 327
documents the rationale).

fix: build x_full ONCE at the top of _tiny_model_rerank, derive each per-base matrix via
np.delete(x_full, col_idx, axis=1) (at 20k rows the delete copy is only ~40 MB) and reuse x_full for
the raw baseline; slice base_screen out of x_full too. Bit-identical.

## P6 - severity P1 - perf
file: src/mlframe/training/composite/discovery/screening.py:43-69
title: _extract_column_array always materialises the full N-row column even when the caller keeps only a small screen sample

The polars path does df.get_column(col).to_numpy().astype(np.float32, copy=False): for any non-float32
source dtype (float64/int - the common case) this allocates the entire column (N rows), and callers
then immediately discard all but a sample: _build_feature_matrix slices [idx] with idx=100k of possibly
4M+ rows (discovery/__init__.py:368), _tiny_model_rerank slices [train_idx_screen] = 20k rows
(_tiny_rerank.py:175). On a 4M-row x 500-col frame that is ~500 full-column transients per matrix build
where O(sample) gathers suffice. Repo precedent for the fix: cache.py CACHE-P0-1 (lines 319-326)
replaced exactly this whole-column to_numpy() with col.gather(sample_idx) and measured ~100x on
data_signature.

fix: add an optional rows parameter; polars path
df.get_column(col).gather(rows).to_numpy().astype(np.float32, copy=False) (O(rows) materialisation),
pandas path df[col].to_numpy(...)[rows]. Wire through _build_feature_matrix and the rerank base_screen
pull. Bit-identical (gather-then-cast == cast-then-gather for these dtypes).

## P7 - severity P1 - perf
file: src/mlframe/training/composite/discovery/screening.py:138-156
title: _safe_abs_corr_all allocates three full-size matrix temporaries - tripling the allocation the leak-corr RAM sampler explicitly budgets for

_filter_features goes to great lengths (_maybe_sample_for_leak_corr, _filter.py:41-123) to keep the
ONE np.column_stack allocation under 30% of available RAM, and frees candidate_arrays early
(_filter.py:199). But the very next call, _safe_abs_corr_all(y, X_train) (_filter.py:212), then
allocates: (1) X_f = X[y_finite] (line 143) - boolean gather copies the FULL matrix even when y is
all-finite; (2) X_dev = X_f - X_means (line 148) - second full copy; (3) (X_dev * X_dev) (line 149)
- third full copy; plus X_dev[:, safe] (line 153) when any column is degenerate. At the 4M x 500
float32 prod shape that is 3 x 8 GB transients against the sampler 8 GB budget - the same
fragmentation-MemoryError class the sampler was built to prevent (its needed_bytes accounting at
_filter.py:79 covers only the stack itself).

fix: (a) gate the gather: X_f = X if y_finite.all() else X[y_finite]; (b) since y_dev sums to
zero, cov_j = x_j . y_dev EXACTLY without centering X - compute cov = X_f.T @ y_dev (BLAS, no
temporary) and var_X = np.einsum(ij,ij->j) on X_f minus n*X_means**2 (no temporary; document the
conditioning caveat, acceptable for a |corr|-threshold filter) - or at minimum replace
(X_dev*X_dev).sum(0) with np.einsum. Verify equivalence on the existing corr-filter tests at rtol=1e-5.

## P8 - severity P1 - perf
file: src/mlframe/training/composite/cache.py:150-158
title: _row_order_fingerprint (polars) hashes EVERY row of the whole frame to keep only the first 256 hashes

df.hash_rows().slice(0, n_take) computes a u64 hash for all N rows across ALL columns - an O(N x C)
full-frame scan plus an N x 8-byte Series allocation - and then throws away everything past row 256.
On the 100+ GB frames this package is designed for, that is a multi-second full read + multi-GB alloc
per data_signature call, silently undoing most of the CACHE-P0-1 ~100x optimisation (lines 319-326)
that removed full-column materialisation from the same function. The docstring even claims
"bounded-cost fingerprint" (line 139). hash_rows is row-local (each row hash depends only on that row
values), so slicing first produces identical bytes.

fix: df.slice(0, n_take).hash_rows() - O(256 x C) instead of O(N x C), digest-identical. Add an
assertion to tests/training/_benchmarks/bench_data_signature.py comparing digests pre/post on a fixture.

## P9 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_screening_tiny.py:429-434
title: Multiseed wrappers re-run domain_check + transform.forward + full-matrix valid-slices per seed; only the KFold seed changes

_tiny_cv_rmse_y_scale_multiseed calls _tiny_cv_rmse_y_scale n_seed_repeats (default 3) times with
identical (y_train, base_train, transform, fitted_params, x_train_matrix). Each call recomputes
transform.domain_check (line 604), y_train[valid]/base_train[valid] float64 copies (607-608),
x_clean = x_train_matrix[valid] - a (20k x ~500) ~= 40 MB matrix copy (line 609, allocated even when
valid.all(), unlike _tiny_cv_rmse_raw_y which has the all-finite shortcut at lines 243-248) - and
transform.forward over the full sample (line 610). With 32 specs x 2 families x 3 seeds, that is ~128
redundant matrix copies + forward evaluations (~5 GB churn + repeated PCHIP/spline forwards for the
heavier transforms) per rerank, all invariant across seeds.

fix: in the multiseed wrapper, compute (valid, y_clean, base_clean, x_clean, t_clean) once and pass
them via optional precomputed kwargs (or refactor _tiny_cv_rmse_y_scale into prepare() + run(seed)).
Also add the valid.all() no-copy gate symmetric with _tiny_cv_rmse_raw_y. Bit-identical.

## P10 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_tiny_rerank.py:494-522
title: Raw-y per-bin baseline refits the K-fold tiny model once per unique base although the fits are bin_var-independent

When the regime gate is enabled (per_bin_n_bins>0, off by default), the loop calls
_tiny_cv_rmse_raw_y(..., return_per_bin=True, bin_var=base_screen) for every unique base column. The
model fits and fold predictions inside are IDENTICAL across bases (same x_full, y_screen, folds,
random_state); only the final _per_bin_rmse(y, y_hat, bin_var) re-aggregation differs. B unique bases
mean B x cv_folds redundant LGBM fits where 1 x cv_folds + B cheap re-binnings suffice. This is the
raw-side twin of the composite-side dedup that ENS-P2-5 already landed (first-pass per-bin reuse,
lines 144-163) - the asymmetry is the leftover.

fix: extend _tiny_cv_rmse_raw_y to accept bin_vars: dict[str, ndarray] (or return per-fold
(val_idx, y_hat) pairs) and compute all per-base breakdowns from one fit pass. Bit-identical.

## P11 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_screening_tiny.py:575
title: Pack #7 serial early-stop (early_stop_threshold) is dead-wired - no caller in src/ ever passes it

The early-exit machinery in _tiny_cv_rmse_y_scale (lines 738-754, documented as "saves 30-66% of
fold-fit compute on candidates the gate will reject anyway") is unreachable: grep over src/mlframe
shows early_stop_threshold appears only in its own definition/comments; _rerank_one_spec and the
multiseed wrappers never pass it, so it is always inf. The natural threshold (raw_baseline x
tolerance) IS computed in _tiny_model_rerank - but only AFTER the per-spec sweep, so it cannot be
threaded.

fix: reorder _tiny_model_rerank to compute the raw-y baseline (lines 426-490) BEFORE the per-spec
loop and pass early_stop_threshold = raw_baseline * raw_baseline_tolerance into
_tiny_cv_rmse_y_scale_multiseed -> _tiny_cv_rmse_y_scale. Note the early-stop only works on the serial
fold path (n_jobs==1, the default config). Add a regression test asserting fewer folds are fitted for
a hopeless spec.

## P12 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_eval.py:226
title: MI bootstrap gathers x_boot (full float32 screen matrix) on every replicate even on the prebinned path where it is unused

Inside the bootstrap loop, x_boot = x_screen_valid[idx_b] (line 226) runs unconditionally; when
_x_pb_valid_const is not None (default bin estimator) the prebinned branch (lines 230-237) never reads
it. With mi_gain_bootstrap_n enabled at e.g. 100 replicates and mi_sample_n=100k x 200 feat, that is
100 x 80 MB = 8 GB of dead fancy-index gathers per work item. Bootstrap is off by default
(mi_gain_bootstrap_n=0), hence P2 not P1.

fix: move the x_boot gather into the non-prebinned else branch (lines 238-252). Bit-identical.

## P13 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_eval.py:113-121
title: Residual-std degeneracy probe runs at FULL train scale per work item (2 float64 full-train copies + transform.forward), where the screen sample suffices

Lines 114-118: y_train[valid].astype(np.float64), base_train[valid].astype(np.float64) (two
n_train-sized copies - 64 MB at 4M rows), then transform.forward over all train rows - per (base,
transform) work item (~60/target), just to compute a std ratio gated at 0.001. The comment claims
"cheap: one transform.forward call", but for monotonic_residual (PCHIP eval), smoothing_spline_residual
and rank_residual, forward at 4M rows is O(n)-heavy; total = several GB churn + seconds per target. A
std-ratio estimate at the 100k screen sample is statistically indistinguishable for a 3-orders-of-
magnitude threshold. Also y_train[valid]/base_train[valid] were ALREADY gathered at line 56 for
transform.fit - the probe re-gathers the same arrays.

fix: compute the probe from t_screen (the forward over the screen sample is computed at line 156
anyway - reuse it: T_std/y_std from t_screen and y_screen[valid_screen]), or at minimum reuse the
line-56 gathers. Document the sample-based probe; thresholds unchanged.

## P14 - severity P2 - perf
file: src/mlframe/training/composite/discovery/_fit.py:507
title: Alpha-drift and alpha~1-collapse loops re-extract full base columns from df per spec instead of reusing _auto_base_pool

The drift detector does base_full = _extract_column_array(df, s.base_column) (line 507) per
linear_residual spec, then gathers base_full[idx1], base_full[idx2], base_full[train_idx] - but
self._auto_base_pool[s.base_column] (populated at line 351) already holds exactly base[train_idx];
idx1/idx2 are contiguous halves of train_idx, so all three gathers are pool[:half], pool[half:], pool -
zero new extraction needed. The collapse loop repeats the pattern at line 616
(_extract_column_array(df, s.base_column)[train_idx]). With up to ~32 kept specs and a 4M-row frame
that is up to ~32 full-column materialisations + train-sized gathers, all redundant.

fix: base_t = self._auto_base_pool.get(s.base_column) with the extract call as fallback for explicit
bases outside the pool. Bit-identical (pool stores the same float32 values; _linear_residual_fit
promotes to float64 internally either way).

## P15 - severity P2 - perf
file: src/mlframe/training/composite/discovery/screening.py:270
title: Prebinned bin-index matrices stored as int64 - 4x the memory needed for nbins=16 codes (sentinel -1 fits int16)

_prebin_feature_columns allocates the (n_rows x n_feat) code matrix as int64 (lines 268, 270). Bin
codes are in [-1, nbins-1] with nbins=16 default - int16 holds them with huge headroom. The int64
choice multiplies every downstream cost: the resident matrix at 100k x 500 is 400 MB (int16: 100 MB),
each per-base np.delete copy (P3), each [valid_screen] slice (P1), each [finite] slice (P2) and each
bootstrap gather scales with it. Only _mi_from_binned_pair needs widening:
combo = x_idx.astype(np.int64) * nbins + y_idx (or int32 - max combo = nbins^2 - 1 = 255).

fix: dtype=np.int16 in _prebin_feature_columns (+ same in the _auto_base null-perm prebinning,
_auto_base.py:403-422), upcast inside _mi_from_binned_pair combo computation. MI values bit-identical
(same integer codes). Bench gather/bincount before/after per feedback_perf_measure_first - int16
gathers are also ~2-4x faster from cache effects.

## P16 - severity P2 - perf
file: src/mlframe/training/composite/ensemble/__init__.py:266-272
title: _carve_inner_eval_split runs np.unique twice on the same group array (two O(n log n) sorts per component per fold)

Line 266: uniq, first_idx = np.unique(g, return_index=True); line 272:
_, _, counts_orig = np.unique(g, return_index=True, return_counts=True) - the second call re-sorts the
identical array and discards two of three outputs. g is fold-train-sized (can be millions of rows);
the helper is called once per raw component per fold in the K-fold OOF path (ensemble/__init__.py:698)
and per component in the external-holdout/single-split paths (lines 422, 863, 897), so with K=5 folds
x C=6 components that is 30 redundant full sorts per ensemble build.

fix: single call uniq, first_idx, counts_orig = np.unique(g, return_index=True, return_counts=True).
Bit-identical.

## P17 - severity LOW - perf
file: src/mlframe/training/composite/discovery/_fit.py:288
title: y_train_for_strat = y_full[train_idx] duplicates the y_train gather from line 219

Both lines fancy-index the same full-train target; one train-sized float32 copy (16 MB at 4M rows) per
fit is wasted. fix: reuse y_train at line 292. Bit-identical.

## P18 - severity LOW - extension
file: src/mlframe/training/composite/discovery/_eval.py:181-195
title: Shrunk-domain mi_y_compare is recomputed per (base, transform) even when several transforms share the identical valid_screen mask

Transforms with the same domain constraint on the same base (e.g. ratio/centered_ratio/
reciprocal_residual share base!=0-style masks; logratio/chain variants share base>0 and y>0) each
recompute the full per-column MI baseline on the same masked rows. A memo keyed on
(base, hash(valid_screen.tobytes())) inside _base_contexts would collapse these to one computation per
distinct mask. Worth doing after P1-P3 land (same path); expected saving = one _mi_to_target_prebinned
call (~0.2-0.5 s at default scale) per duplicated mask per base. Thread-safety: populate lazily under
a lock or precompute masks serially in the per-base setup.

## P19 - severity LOW - perf
file: src/mlframe/training/composite/discovery/screening.py:277
title: _prebin_feature_columns uses np.nanquantile per column even for all-finite columns

np.nanquantile is several times slower than np.quantile on NaN-free data (per-call NaN scan +
nan-aware path). The loop already computes col_finite (line 273); gating to np.quantile when
col_finite.all() is free and bit-identical (nanquantile == quantile on finite input). ~n_feat calls x
O(n log n) per fit; measure before shipping per feedback_perf_measure_first (expected ~1.5-3x on the
prebin phase for clean data).

---

### Not flagged (checked, at floor or by design)
- _mi_pair_bin / _mi_from_binned_pair: numpy floor documented with rejected numba + np.partition
  attempts (screening.py:197-220).
- compute_oof_holdout_predictions module cache intentionally unwired on suite path - bench
  (bench_oof_cache_reuse.json, hit speedups 1760-4645x) and RAM-rule rationale documented at
  ensemble/__init__.py:303-317.
- CompositeCrossTargetEnsemble: solver-copy dedup, __getstate__ train-matrix strip, hoisted sklearn
  imports - all present (_cross_target.py:255-258, 677-691).
- EWMA / frac-diff kernels: backend ladder + KTC integration + batched variants present and benched
  (nonlinear.py:50-124, 584-784); KTC lookup goes through a process singleton (_kernel_tuning.py),
  no per-call subprocess.
- data_signature polars numeric stats single-select + gather (CACHE-P0-1) is sound; only the
  row-order fingerprint regression (P8) undermines it.
- _quantile_residual_per_bin_stats: measured dispatcher with bench-attempt-rejected note.
- forward_stepwise_multi_base: per-trial column_stack copies exist but trial counts are tiny
  (pool <= auto_base_top_k+1, max_k=3) - below the measure-first bar.