# Composite targets audit, direction 5: performance (speed and memory)

Date: 2026-09-19. Code: `origin/master` a950f7e47 (clean worktree). Scope: `src/mlframe/training/composite/` (discovery, transforms, estimator, ensemble, root modules) and the composite phases in `src/mlframe/training/core/`.

**Method.** I read the code paths that run under the default `CompositeTargetDiscoveryConfig`: the MI screen, per-candidate eval, auto-base, tiny rerank, post-rerank gates, honest holdout, opt-in steps, the wrap pass, CT_ENSEMBLE OOF and MoE. I looked for work that is repeated, discarded, copied or strided. Correctness issues already filed in `transforms.md`, `discovery.md` and `estimator_ensemble.md` are not repeated here. Where a finding touches one of them, the cross-reference is given.

**Measurements.** Micro-benchmarks were run in one Python process at a time, with `OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=2` and `LOKY_MAX_CPU_COUNT=1`. Each figure is the median of 3 to 5 runs. Environment: numpy 2.x, lightgbm 4.6.0, pandas 3.0.3, polars. The host is a shared 16 GB box with another agent running, so absolute timings are inflated. Only the ratios matter. The scripts are in the session scratchpad (`bench_perf_audit.py`, `bench_coll.py`, `prof_sig.py`) and are not committed. Anything marked "estimate" was not measured.

Default counts used in the estimates:
- `mi_sample_n=100_000`, `auto_base_top_k=3` bases.
- 25 bivariate transforms plus 5 unary transforms, so about 80 MI candidates per target.
- `top_k_after_mi=32`, `tiny_screening_families=("lightgbm","linear")`, `tiny_model_n_seed_repeats=3`, `tiny_model_cv_folds=3`, `tiny_model_sample_n=20_000`.
- `dedup_x_remaining_for_mi_baseline=True`, `honest_holdout_frac=0.2`, `oof_holdout_source="kfold"` with `oof_kfold=5`, `transform_waic_validation_enabled=True`.

---

### PRF-01 [P1] Near-collinear dedup runs a serial, strided O(B²·n) njit walk once per base; a single GEMM correlation matrix per target is about 70x faster

- **Where**:
  - `discovery/_fit.py:433-443`: the per-base `near_collinear_keep_mask(x_remaining_matrix, ...)` call, on by default.
  - `discovery/_collinear_numba.py:134-181`: `_keep_mask_kernel_allfinite`, which is serial and reads `fm[i, j]` / `fm[i, k]` down columns of a C-order matrix.
  - `discovery/_collinear_numba.py:324`: `np.ascontiguousarray(..., dtype=np.float64)`, a float64 copy per base.
  - `discovery/_collinear_numba.py:326-331`: the content-hash cache.
- **What**:
  - For every base candidate, the walk compares each column with every column already kept, one full pass over n per pair. The pass is serial, and the inner reads stride across the row-major matrix.
  - Measured on 80k x 120 all-finite float32: **11.5 s per call**. The in-code note says "~17 s at 80k x 120" on another box.
  - A float64 `Z.T @ Z` correlation matrix plus the same greedy walk on the (B, B) matrix took **165 ms** and gave the identical keep-mask.
  - The per-base matrices differ only by the dropped base column, and pairwise correlations do not depend on which column is absent. One correlation matrix per target therefore serves every base.
  - The keep-mask cache is keyed by a blake2b hash of the float64 copy of the base-dropped matrix. It misses across bases by construction and still pays for the copy and the hash on each call.
  - The cost scales as F². At 500 features, one base is an estimated few minutes on the current kernel.
- **Expected win** (estimate): from 3 x 11.5 s to about 0.2 s per target at 80k x 120; roughly 50 to 70x on this step. Memory also drops: no per-base float64 copy.
- **Suggested fix**:
  - Compute column means and centred sums of squares once, then the full |corr| matrix with one BLAS GEMM on the float64 standardised screen matrix. For NaN-holed matrices, get the per-pair joint-finite moments from masked GEMMs: `X0ᵀX0`, `X0ᵀM`, `(X0²)ᵀM` and `MᵀM` with zero-filled `X0` and finite mask `M`.
  - Run each base's greedy walk on the sub-matrix that excludes that base.
  - Keep the existing borderline band (1e-9) and the exact numpy re-decision, so the mask stays bit-identical. GEMM rounding is around 1e-13, well inside the band.
  - Keep the current kernel as `_keep_mask_kernel_v2` per the keep-all-kernels rule, behind the KTC dispatcher.
- **Test/benchmark to add**:
  - `_benchmarks/bench_near_collinear_gemm.py`: kernel vs GEMM at (80k, 120) and (100k, 500), all-finite and 5% NaN, plus 3-base amortisation.
  - A parity test: the GEMM mask equals the reference `_near_collinear_keep_mask_numpy` across seeds, exact duplicates, a constant column, NaN holes and a pair placed exactly on the threshold.
- **Disposition**: REJECTED as specified - the 70x does not reproduce on this host. Measured medians (2 threads, shared box, _benchmarks/bench_near_collinear_gemm.py, masks identical everywhere): 80k x 120 all-finite kernel 239 ms vs matrix 163 ms (1.5x), 20k x 60 30.2 vs 16.1 ms (1.9x), 80k x 300 624 vs 476 ms (1.3x), 100k x 500 1115 vs 1034 ms (1.1x), and with 5% NaN the masked-GEMM path is SLOWER (31.9 vs 41.7 ms, 0.8x). The report's 11.5 s per call is 48x the kernel time measured here, so the premise of the win is absent; swapping a tuned njit kernel for a path that loses on holed matrices is a tradeoff, not a free win. The benchmark is committed with the numbers; the one idea still worth pursuing (amortise ONE matrix per target across its bases) would save about 0.5 s per target at b=120, which is noise against a minutes-long discovery, and is recorded here rather than implemented.

### PRF-02 [P1] The prebinned code matrix is C-order, so every per-feature MI call reads strided columns; F-order is 8.7x faster and bit-identical

- **Where**:
  - `discovery/screening.py:342` and `:405`: `binned = np.empty((n_rows, n_cols))` in the eager and lazy prebin paths.
  - Consumers of `fb_f[:, j]` in `_mi_per_feature_prebinned` (`screening.py:536-557`):
    - `discovery/_fit.py:374-381`: `_per_feat_y_full`.
    - `discovery/_eval.py:515`: per-candidate `MI(T, X_remaining)`.
    - `discovery/_eval.py:548`: `mi_y_compare`.
  - `discovery/_fit.py:416`: per-base `np.delete`, which keeps C order.
- **What**:
  - `_mi_from_binned_pair` walks one int16 column element by element. On a C-order (n, F) matrix each element sits `2*F` bytes after the previous one, so every read touches a new cache line.
  - Measured `_mi_per_feature_prebinned` at n=100k, F=100: **302 ms (C-order) vs 35 ms (`np.asfortranarray`), max abs diff 0.0**.
  - The eval loop calls this once per candidate, about 80 times per target, plus the baselines. That is about 24 s vs about 3 s per target at F=100 on this host, and it grows linearly with F.
- **Expected win** (measured kernel, estimated end to end): about 8x on the MI(T, X) part of the screen. `_prebin_feature_columns` would also speed up if it read and wrote contiguous columns.
- **Suggested fix**:
  - Allocate the code matrix with `order="F"` in both prebin paths.
  - Have `_build_feature_matrix` fill an F-order `np.empty` column by column instead of `np.column_stack`. `np.delete` along axis 1 keeps F order.
  - Check the row-gather users, which are the opt-in bootstrap `_x_pb_valid_const[idx_b]` and the boolean `valid_screen` masks. Keep a C-order copy only where row gathers dominate.
- **Test/benchmark to add**: extend `bench_iter94_mi_binned_pair_strided.py` with a C vs F layout A/B at (100k, 100) and (100k, 500). Add a test that per-feature MI is bit-identical across the two layouts.
- **Disposition**: COMPLETED - both prebin paths allocate the code matrix column-major, which is how every consumer reads it. Measured on this host (2 threads): the per-feature MI pass on the shipped prebin output went 239 ms -> 19.1 ms at n=100k, F=100, and an isolated A/B gives 6.8x at F=100 and 23.9x at F=300, max|diff| 0.0. The one path that reads ROWS (the opt-in bootstrap replicate loop) takes a C-order copy once, since a row gather costs 102 ms against 14 ms in this layout (test_prebinned_matrix_is_column_major.py, 4 tests: layout, bit-identical MI, np.delete keeps the layout, exclude_col parity)

### PRF-03 [P1] With group ids, the tiny rerank runs the full multi-family CV for every spec, then honest-OOF replaces nearly all of those scores

- **Where**:
  - `discovery/_tiny_rerank.py:368-470` and `:504`: the per-spec CV sweep.
  - `discovery/_tiny_rerank.py:571-601`: honest-OOF override of `agg_scores` and of the gate baseline.
  - `discovery/_tiny_rerank.py:353-363`: sequential early-stop is disabled whenever honest-OOF will run.
- **What**:
  - When `_group_ids_for_rerank` and `honest_holdout_idx_` exist (the grouped production case), every measured spec's CV score is overwritten by `honest_oof_reconstruction_rmse`. The raw-baseline threshold is replaced by the honest floor.
  - The CV scores survive only as a fallback for specs whose honest measurement degenerated. Two other consumers use them, but both are off by default: the per-bin gate (`per_bin_n_bins=0`) and the Wilcoxon gate.
  - Under groups the seed repeats collapse to one, so the discarded work is 32 specs x 2 families x 3 folds, about 192 tiny fits. The multiseed early-stop that would trim it is also disabled on this path.
  - The honest step itself costs 1 raw fit plus one fit per spec.
- **Expected win** (estimate): with groups, the rerank drops from about 192 CV fits to about 33 honest fits plus CV only for the degenerate specs. That is roughly 5x on the rerank phase.
- **Suggested fix**:
  - Compute `honest_oof_reconstruction_rmse` before the per-spec CV sweep. It only needs `kept_specs`, the frame and the indices.
  - Run `_rerank_one_spec` only for specs missing from the honest dict, plus any spec a still-enabled consumer needs: per-bin, Wilcoxon, or the WAIC band in PRF-11.
- **Test/benchmark to add**:
  - A test with synthetic groups asserting that the CV sweep runs only for specs absent from the honest dict, and that the final `specs_` equal the current code path.
  - A bench of rerank wall time with and without group ids.
- **Disposition**: COMPLETED - honest-OOF is now measured BEFORE the sweep and every spec it measures skips the CV fits entirely. Measured on the grouped biz fixture (40 groups x 300 rows, 2 threads): 16 kept specs, all 16 scores overwritten, CV sweep 11.6 s of fits against 0.26 s for the honest measurement; after the change the sweep runs 0 times and the full fit went 8.44 s -> 6.47 s at default `tiny_model_n_estimators`, with a bit-identical spec order and bit-identical `tiny_rerank_scores_`. The skip is gated on the sweep having no other consumer: with the per-bin regime gate or the Wilcoxon gate on, every spec is still fitted (WAIC recomputes its own folds, so it needs nothing from the sweep). test_honest_oof_skips_discarded_cv.py, 4 tests: no CV fits on the default grouped path (fails pre-fix with 32 calls), scores are the honest values, Wilcoxon keeps the sweep, per-bin keeps the sweep

### PRF-04 [P1] The OOF pre-screen that skips refitting hopeless components never runs under the default `oof_holdout_source="kfold"`

- **Where**:
  - `core/_phase_composite_post_xt_ensemble/__init__.py:525`: the guard `if _ext_X is not None and _ext_y is not None and len(_components) >= 4`.
  - `_ext_X` is set only in the `external_val` branch, `:481-489`.
  - The default is set in `_composite_target_discovery_config.py:477`.
- **What**:
  - The code comment says the end-of-function dummy-floor gate drops 60-70% of components after they are OOF-refit, costing "~30-50 minutes of pure waste per target".
  - The cheap pre-screen written to avoid this (leaky val RMSE with a 1.5x safety margin) is reachable only when the OOF source is `external_val`.
  - Under the default K-fold source, `_ext_X` stays `None`. Every component therefore goes through `oof_kfold=5` full refits before the floor gate discards most of them.
  - The val frame `filtered_val_df` is still available on the K-fold path.
- **Expected win** (the in-code estimate, not re-measured here): 5 refits saved for each component the floor gate would drop, up to tens of minutes per target on large frames.
- **Suggested fix**:
  - Decouple the pre-screen from the OOF source. Whenever `filtered_val_df` and the val y exist, compute the leaky val RMSE from the already-trained components; reuse cached predictions per PRF-13. Drop components by the same 1.5x rule before `compute_oof_holdout_predictions`.
  - This is independent of EST-11, the val-vs-OOF unit mismatch in the final floor gate.
- **Test/benchmark to add**: a test in which a component clearly loses to the dummy on val and the OOF source is `"kfold"`. Assert that its OOF refit is never invoked, by counting fits on a mock, and that the surviving ensemble is unchanged.
- **Disposition**: COMPLETED - the pre-screen no longer reads the OOF frame directly: it takes whichever frame is available, falling back to `filtered_val_df` and its y, so it runs under the default `oof_holdout_source="kfold"` as well. The screen itself was carved into `_phase_composite_post_xt_ensemble/_prescreen.py` (frame choice, leaky-RMSE keep mask, dummy floor lookup) so its rules are directly testable; the caller keeps the same drop conditions (>=4 components, >=2 survivors, honest floor gate unchanged afterwards). test_xt_ensemble_leaky_prescreen.py, 8 tests: val fallback, OOF frame takes precedence, no frame means no screen, hopeless dropped / good kept, safety margin, predict failure kept, too few finite rows kept, floor lookup. The wiring test fails pre-fix by construction (the helper did not exist); no timing was taken, as the win is refits not run and the in-code estimate was not re-measured here

### PRF-05 [P1] The honest-holdout re-score gathers the full, uncapped holdout feature matrix once per spec, in parallel threads, and re-bins every column twice per spec

- **Where**:
  - `discovery/_honest_holdout.py:159-177`: `_build_x_remaining_holdout`.
  - `discovery/_honest_holdout.py:252-338`: `_rescore_one`, with its `x_remaining[valid]` copy and two `_mi_to_target` calls, run under `Parallel(n_jobs=min(len(specs), cores))`.
- **What**:
  - For each final spec, the code pulls every usable feature on all holdout rows from the frame. The holdout is 20% of train and has no row cap, unlike the 100k screen.
  - It copies that matrix again through the valid mask, then runs `_mi_to_target(estimator="bin")`. That calls `_mi_pair_bin`, which re-quantiles each X column for both `mi_t` and `mi_y`. The `mi_y` memo saves the second only on a hit.
  - Specs on the same base rebuild the identical matrix.
  - Measured at 40k x 50: `_mi_to_target(bin)` **430 ms per call**, against 26 ms with a matrix prebinned once, plus 275 ms one-time prebin.
  - Memory: on a 4M-row train, each spec holds about 800k x F x 4 B twice. With 8 concurrent specs at F=500, that is an estimated 25 GB transient.
- **Expected win** (estimate): time drops about 10x on this step. Peak memory goes from `n_jobs x 2 x holdout x F x 4 B` to one prebinned int16 matrix.
- **Suggested fix**:
  - Build the holdout matrix once, capped at `mi_sample_n` rows with a seeded draw, and prebin it once in F order (PRF-02).
  - Per spec, pass the base column(s) as excluded indices; `_mi_per_feature_prebinned` already supports `exclude_col`, so extend it to a list. Apply `valid` as a row mask only when it is not all-true.
- **Test/benchmark to add**:
  - A bench of the re-score on 10 specs over 3 bases at a 200k holdout with 100 features, recording wall time and peak RSS.
  - A test that the honest gains equal the current values when the cap exceeds the holdout size.
- **Disposition**: COMPLETED - three changes, each measured on a 200k-row holdout with 60 features and 10 specs sharing one base: the X-remaining matrix is built once per base set instead of once per spec (the lock is held across the build, or the threads all miss the empty cache together); the rows are capped by `mi_sample_n`, the same cap the in-screen MI uses, with a seeded draw above it; and the bin estimator quantises the shared matrix once, so each MI is a histogram pass. Wall: 10.77 s -> 0.75 s at the default cap (1.50 s with the cap off, so 7.2x from the shared bin codes alone), with the stamped gain 0.026222 against 0.026228 uncapped, and the prebinned MI is exact to 8e-17 against the per-call binning at 100k x 60 (0.97 s -> 0.14 s per call plus a 0.60 s one-time prebin). The shared codes are used only when a spec keeps every holdout row: the bin edges are quantiles of the rows actually scored, so a spec whose domain filter drops rows bins its own subset as before. test_honest_holdout_rescore_cost.py, 5 tests: one build for four specs, gains unchanged by the cache, a cap above the holdout changes nothing, a cap below it bounds the rows, the draw is reproducible

### PRF-06 [P1] `_filter_features` holds every numeric feature over all train rows, then stacks a second full copy for a leak-corr test that a sample would decide

- **Where**: `discovery/_filter.py:143-178`, which holds `candidate_arrays` and then runs `np.column_stack` and `~np.isfinite(X_train)`. The sampling helper is at `:41-121`.
- **What**:
  - Each numeric column is extracted on all `train_idx` rows (the 80% screening pool, not the 100k sample) and kept in `candidate_arrays`.
  - After the loop, `np.column_stack` materialises a second (n_train, F) float32 copy, plus an (n_train, F) bool non-finite mask.
  - `_maybe_sample_for_leak_corr` runs only after every full column is already held, and samples only when the stack would exceed 30% of available RAM.
  - Peak is therefore at least `n_train x F x 4 B` and usually `2x` that plus `n_train x F` bytes. At 3.2M x 500 that is 6.4 GB + 6.4 GB + 1.6 GB.
  - The constancy check (ptp, finite count) needs no stacking at all. The leak test is `|corr| >= 0.99999`, which is decided exactly as well on a few hundred thousand rows: the standard error of r near 1 is about (1-r²)/√n.
- **Expected win** (estimate): peak memory in discovery fit falls from about 2 x n_train x F x 4 B to one column plus the sample matrix; the full-frame gather time also goes away.
- **Suggested fix**:
  - Compute min, max and non-null count per column with one lazy polars `select`, or per-column numba stats on pandas, without keeping the arrays.
  - Run the leak-corr on a fixed seeded row sample, e.g. 500k rows or `mi_sample_n`, through `_extract_column_array(rows=...)`.
  - Keep the per-pair NaN correlation branch on the sample.
- **Test/benchmark to add**:
  - A peak-RSS bench of `_filter_features` at 2M x 200.
  - A test that the drop list, including exact-copy and y-derived leak columns, equals the full-row result.
- **Disposition**: PARTIAL - the leak-corr rows are now chosen BEFORE the columns are gathered, so each column is released after its constancy and finite-row checks and only the sampled block is retained: peak goes from every column over every train row plus a second stacked copy to one full column plus a (sample x F) matrix. The stride keeps at least `_LEAK_CORR_MIN_SAMPLE_ROWS` = 500k rows, where the standard error of r near 1 is far inside the 0.99999 threshold, and frames at or below that still read every row, so their drop lists are bit-identical. NOT done: the per-column stats still go through `_extract_column_array` rather than one lazy polars `select`, and no peak-RSS bench was run - the remaining gather is one column at a time, which is no longer the dominant allocation. test_filter_leak_corr_sampling.py, 5 tests: no sampling below the floor, strided and bounded above it, the drop list (exact copy, near-copy, y-derived) is unchanged when sampling kicks in, the constancy check still reads every row, only the sample is held per column

### PRF-07 [P2] Tiny-model LightGBM fits re-bin the same feature matrix for every spec, seed and fold; `LgbFoldCache` exists but only auto-chain uses it

- **Where**:
  - `discovery/_screening_tiny_perbin.py:282-316`: the y-scale CV, once per spec x seed x fold.
  - `discovery/_screening_tiny.py:373-399`: the raw-y CV.
  - `discovery/_honest_oof_select.py:101-157`, `discovery/_honest_rmse_gate.py:116-127`, `discovery/_yscale_holdout_gate.py:333-392`: one x_fit per gate, a raw fit plus one fit per spec.
  - `discovery/_eval_waic.py:218-229`: WAIC, same X per base, 4 folds per spec.
  - The reference implementation is `discovery/_lgb_fold_cache.py`, used by `_auto_chain.py`.
- **What**:
  - Within one base, every spec's tiny model trains on the same X rows per (seed, fold), and only the label differs. `LGBMRegressor.fit` rebuilds the binned `Dataset` each time.
  - In the rerank that is about 288 LightGBM fits over at most 9 distinct fold matrices per base.
  - Measured with 10 targets on 13.3k x 60: **7.50 s per-fit vs 5.00 s with one constructed Dataset plus `set_label` (1.50x)**; one Dataset construction takes 133 ms. The auto-chain docstring measured construction at "a quarter" of wall time.
- **Expected win** (measured on this shape): about 1.3-1.5x on all tiny LightGBM work, which is the dominant rerank cost.
- **Suggested fix**:
  - Generalise `LgbFoldCache` into a shared helper keyed by (x identity, fold rows).
  - In the rerank, restructure so each (base, seed, fold) builds one Dataset and loops specs over it. Use `subset(fit_rows)` for specs whose domain mask removes rows. Keep threads on (base, seed, fold) instead of specs, because `set_label` mutates the Dataset.
  - Apply the same pattern to the three holdout gates: build the gate's x_fit Dataset once and reuse it for the raw fit and each spec.
- **Test/benchmark to add**:
  - A bench of the rerank at 20k x 100 with 32 specs.
  - A parity test that per-spec RMSEs match the per-fit path to about 1e-12, since LightGBM with the same params and bins is deterministic.
- **Disposition**: PARTIAL - the y-scale tiny CV (`_tiny_cv_rmse_y_scale`, the rerank's per-spec fits) now builds each fold's binned dataset once per thread and swaps the label, through the new `_lgb_shared_fold.py`. Measured parity is exact, not 1e-12: max abs diff 0.0 over 32 specs, and a full tiny-model discovery on 20k x 41 gave identical specs and identical `tiny_rerank_scores_` to 10 decimals. Isolated fit cost 38.83 s -> 29.31 s (1.32x) for 32 specs on one 13.3k x 100 fold at the defaults, where one construction is 328 ms; the whole discovery went 68.73 s -> 66.31 s, with 177 of its 281 LightGBM fits on the shared path. Only a fold that trains on all its rows shares: a row `subset` keeps the parent's bin boundaries, which a fresh fit on fewer rows places differently (measured max diff 8.05), so masked folds keep the fresh fit. The cache is per thread because `set_label` mutates the dataset and the rerank scores specs from a thread pool. NOT done: the raw-y CV and the three holdout gates plus WAIC (the other 104 fits) still fit per call. While probing this I found and fixed a real bug in the auto-chain `LgbFoldCache`: a subset whose label is set before construction trains on the parent's all-zeros label, so masked candidates predicted a constant (commit 4a240d450). test_tiny_cv_shares_lgb_fold_dataset.py, 3 tests: bit-identical to the sklearn per-fit path, one build per fold across specs, a freed matrix never hits a stale dataset

### PRF-08 [P2] The `"linear"` screening family refits `SimpleImputer + Ridge` for every spec on the same X; one multi-output solve is 9.4x faster

- **Where**: `discovery/_screening_tiny.py:236-254` (the pipeline), called from `_screening_tiny_perbin.py:282-316` and `_screening_tiny.py:373-399` for each spec x seed x fold.
- **What**:
  - `tiny_screening_families` defaults to `("lightgbm", "linear")`, so the rerank also runs about 288 Ridge pipelines. Each one re-imputes and re-factorises the same fold X; only the target column differs.
  - Measured with 10 targets on 13.3k x 60 (1% NaN): **427 ms per-target vs 45 ms for one multi-output `Pipeline.fit(X, T)` (9.4x)**.
- **Expected win** (measured): about 9x on the linear-family share of the rerank.
- **Suggested fix**:
  - Group specs by (base, seed, fold, fit-row mask). The all-valid-domain specs share one mask, which is the common case.
  - Fit one imputer and one multi-target Ridge per group, then invert each column with its own transform. Specs with a different domain mask fall back to the per-spec path.
- **Test/benchmark to add**: a parity test that multi-output Ridge predictions equal per-target predictions to 1e-10, plus a rerank bench with the `"linear"` family only.
- **Disposition**: COMPLETED - rather than regroup the parallel rerank by (base, seed, fold), each thread now caches the fold's imputer means and the Cholesky factor of its regularised Gram matrix (`_ridge_shared_fold.py`), and every spec is one right-hand side on it: 32 targets on 13.3k x 60 went 1836 ms -> 83 ms (22x) in isolation. The cache holds only two length-F vectors and an F x F factor, never the fold's rows. The solve runs in float64, where the old pipeline inherited the screening block's float32: against a float64 sklearn pipeline predictions agree to 1e-7 (test tolerance; 2.4e-8 measured) and a spec's CV-RMSE to 1e-9 relative, while against the old float32 path they move by float32 round-off. A full default-family discovery on 20k x 41 kept identical specs, scores within 1.9e-7 relative, wall 65.73 s -> 64.60 s (the linear family is a small share of the fits). Masked folds keep the per-spec pipeline. The raw-y CV fits the linear family once per seed and fold, not per spec, so it has nothing to share. test_tiny_cv_shares_ridge_factorisation.py, 4 tests: predictions match a float64 pipeline, an all-missing column matches the imputer's drop, one factorisation per fold across specs, a spec's score matches the float64 sklearn path

### PRF-09 [P2] Per-base float and prebinned matrix copies stay alive through the rerank and all later gates, although the default bin path never reads the float values; the lazy prebin path is dead under defaults

- **Where**:
  - `discovery/_fit.py:409-510`: per-base `np.delete` copies of both `_full_x_matrix` and `_full_x_prebinned`, stored in `_base_contexts`.
  - `discovery/_fit.py:319`: lazy-prebin gate that requires `not _dedup_x_remaining`.
  - `discovery/_eval.py:512`: the float copy is read only when `_x_prebinned is None`.
- **What**:
  - With the default bin estimator, the base-dropped float32 `x_remaining_matrix` is consumed only by the dedup mask and a `.shape[1]` check. It is still stored in each base context next to its int16 prebinned copy.
  - These locals and the full matrices stay referenced until `fit` returns, so they are alive during the tiny rerank, the holdout gates, auto-chain and the honest re-score.
  - Resident extra is about `(bases+1) x n_screen x F x 6 B`; an estimated 1.2 GB at 100k x 500 with 3 bases.
  - The polars lazy-prebin path, built to avoid the float plane, is gated off whenever `dedup_x_remaining_for_mi_baseline` is on, which is the default. The path is therefore unreachable in default runs.
- **Expected win** (estimate): about 1 GB lower peak during the most memory-heavy discovery phases at 100k x 500.
- **Suggested fix**:
  - After the dedup mask, replace the float copy with the zero-row proxy the lazy path already uses on the bin path.
  - Use `exclude_col` / a column-index array instead of materialising per-base prebinned copies.
  - `del _base_contexts, _full_x_matrix, _full_x_prebinned` right after the candidate loop.
  - With PRF-01, dedup needs only one correlation matrix per target. That matrix can be built one column pair block at a time, which lets the lazy path run together with dedup.
- **Test/benchmark to add**: extend `bench_lazy_prebin_memory.py` to record RSS at the `transforms_evaluated` and `tiny_model_rerank_done` checkpoints with dedup on.
- **Disposition**: PARTIAL - the per-base contexts and the full float and prebinned matrices are now released right after the candidate loop (the work-item build and dispatch moved into `_evaluate_work_items` so `fit` could take the release line; it went 788 -> 711 lines). Measured RSS at the start of the tiny rerank on 100k x 300 with two bases: 1080 MB -> 725 MB. NOT done: the float copy still lives alongside its prebinned twin during the candidate loop itself, the per-base prebinned copies are still materialised rather than addressed through `exclude_col`, and the lazy prebin path is still gated off by the default dedup, so the peak inside the candidate loop is unchanged. test_base_matrices_released_before_rerank.py: a weak reference to the screening matrix is dead when the rerank starts (fails pre-fix)

### PRF-10 [P2] The honest-OOF selector, the honest RMSE gate and the y-scale gate each rebuild feature matrices from the frame and refit a raw baseline and each spec on nearly the same screen-to-holdout design

- **Where**:
  - `discovery/_honest_oof_select.py:71-111`: 30k cap; `x_fit`/`x_eval` built; raw fit plus one fit per spec, up to 32.
  - `discovery/_honest_rmse_gate.py:95-126`: 20k cap; its own `x_fit`/`x_eval`; raw fit plus one fit per spec.
  - `discovery/_yscale_holdout_gate.py:324-392`.
  - Call order in `discovery/_fit.py:700-722`.
- **What**:
  - With group ids, honest-OOF has already fit a tiny model per spec on screen rows, predicted the honest holdout, inverted, and scored RMSE against a raw-y baseline.
  - The honest RMSE gate then repeats the same measurement on a different subsample, with a 20k instead of 30k cap and its own RNG draw. It does this for the surviving specs and refits the raw baseline.
  - Each of the three gates also gathers two full-feature matrices from the frame.
- **Expected win** (estimate): 1 raw fit plus up to `top_m_after_tiny=10` spec fits, plus 2 frame gathers, saved per target when groups exist. On non-grouped runs, the shared gather still saves one matrix build per gate.
- **Suggested fix**:
  - Use one cap for both honest measurements and cache `{spec.name -> (honest rmse, raw rmse)}` on the instance. The gate reuses the value for any spec already measured.
  - Keep a per-fit row-index-keyed cache of built feature matrices for the gates.
  - The DSC-03/DSC-07 correctness discussion about using the holdout for selection is separate; this finding is only about duplicated compute.
- **Test/benchmark to add**: count tiny-model fits per discovery fit on a grouped fixture, and assert gate verdicts are unchanged when the caps are equal.
- **Disposition**: PARTIAL - verified first: on the grouped fixture the honest RMSE gate's per-spec y-RMSE and its raw baseline were identical to honest-OOF's to full precision, because whenever both samples are under their caps neither draws at random and they fit the same models on the same rows. Honest-OOF now caches its holdout predictions with the rows and each spec's fit-mask fingerprint, and the gate reuses one only when fit rows, eval rows and mask all match, then runs its own finite / collapse / duplicate-of-raw checks on it unchanged. Measured on the grouped fixture: 93 -> 84 sklearn LightGBM fits per discovery, identical specs and honest numbers (wall 6.72 s -> 4.07 s on a noisy host; the fit count is the reliable figure). NOT done: the caps are still 30k vs 20k, so above 20k rows the two samples differ and the gate still refits, and the y-scale group gate and the per-gate frame gathers are untouched - unifying the caps would change the gate's sample and so its verdicts, which is a selection change rather than a pure compute saving. test_honest_rmse_gate_reuses_honest_oof.py, 3 tests: verdicts equal a forced-refit run with fewer fits, reuse withheld when rows differ, reuse withheld when the fit mask differs

### PRF-11 [P2] The WAIC tie-break scores every kept spec, although the score is used only inside multi-member RMSE bands

- **Where**:
  - `discovery/_tiny_rerank.py:930`: call site.
  - `discovery/_tiny_rerank_waic.py:72-110`: `_waic_for`, run for all specs.
  - `discovery/_tiny_rerank_waic.py:93`: float64 copy of the base matrix per spec.
  - `discovery/_tiny_rerank_waic.py:117-128`: band logic.
- **What**:
  - `transform_waic_validation_enabled` is on by default. `_apply_waic_tiebreak` runs 4 LightGBM folds per spec, with an extra train-set predict per fold, for up to 32 specs.
  - The resulting WAIC changes the order only inside bands where RMSEs sit within 2% of each other. Singleton bands, and bands wholly below `top_m_after_tiny`, discard their WAIC.
  - Each spec also builds its own `np.asarray(x_mat, dtype=np.float64)[valid]` copy of the shared per-base matrix.
  - The in-code note records the serial version at "73s of a 139s discovery". It is now threaded, but the work is unchanged.
- **Expected win** (estimate): proportional to the fraction of specs outside multi-member bands. When RMSEs are spread out, most of the WAIC work disappears.
- **Suggested fix**: compute the bands from `agg_scores` first. Score WAIC only for specs in bands of size at least 2 that intersect the top-`top_m` window. Hoist the float64 conversion to once per base, and reuse the fold Dataset (PRF-07).
- **Test/benchmark to add**: a test that the final order is identical when WAIC runs only on band members; a bench counting WAIC fits on a spread-RMSE fixture.
- **Disposition**: PARTIAL - the RMSE bands are now computed first (`rmse_bands`) and WAIC is scored only for members of multi-spec bands that start inside the top-m window; a singleton band has nothing to re-order and a band past the cut is trimmed away whatever its order. Measured over three seeds of the WAIC biz fixture with a six-transform pool: identical final specs, WAIC computations 18 -> 6. `_tiny_rerank_waic_scores` now holds only the scored specs, and the WAIC wiring test was reframed from "every reranked spec is scored" to "the tied specs near the top are scored and nothing else". NOT done: the per-spec float64 copy of the base matrix and reusing the PRF-07 fold dataset inside WAIC. test_biz_val_discovery_waic_validation.py: +2 tests (only tied specs are scored, which fails pre-fix; band grouping)

### PRF-12 [P2] The auto-base permutation null is a serial Python loop over features x 20 permutations

- **Where**: `discovery/_auto_base.py:512-561`.
- **What**:
  - `auto_base_null_perms=20` by default. For every usable feature, the code block-shuffles its codes and calls `_mi_from_binned_pair` 20 times, one Python iteration each, on the 100k screen.
  - The codes are int64.
  - Measured at 481 µs per (column, permutation) at n=100k: 0.48 s for F=50, an estimated 4.8 s for F=500 per target. All of it runs on one thread.
- **Expected win** (estimate): about the core count on this step, since (column, permutation) pairs are independent, plus a smaller constant from fusing and int16 codes.
- **Suggested fix**:
  - Pre-draw the permutation matrix from `rng_perm` in the same order to keep the null bit-identical.
  - Run one `@njit(parallel=True)` kernel that prange-iterates columns, gathers each block-shuffled column into a thread-local buffer and builds the joint histogram.
  - Store codes as int16.
  - Keep the per-pair NaN fallback path as is.
- **Test/benchmark to add**: extend `bench_unary_mi_memo.py` or add `bench_auto_base_null_njit.py` for F in {50, 500}, with a bit-identity test of `null_means`/`null_stds` against the loop.
- **Disposition**: COMPLETED - each column's permutations are now drawn up front in the exact order the loop consumed `rng_perm` (`draw_null_permutations`), and one `@njit(parallel=True)` kernel scores all of a column's permutations, calling the same gather and MI kernels the loop called (`_null_mi_numba.py`). Every null MI is bit-identical to the loop for element shuffles and block shuffles, the generator ends in the same state so the next column's draws are unchanged, and the value-based fallback paths get the identical shuffled arrays from the same pre-drawn permutations. Measured at F=50, n=100k, 20 permutations: 0.40 s -> 0.24 s at 2 threads, 0.42 s -> 0.18 s at 8 (a shared host, so the core scaling is muted). Codes stay int64: the MI kernel takes any integer dtype, and narrowing them would only save the copy. test_auto_base_null_parallel.py, 6 tests: bit-identical null MIs for block_len 1/7/50, generator state preserved, value fallback identical

### PRF-13 [P2] The composite post-phases re-predict the same models on the same val and test frames several times

- **Where**:
  - `core/_phase_composite_wrapping.py:258-263`: per-model immediate `wrapper.predict` on val and test.
  - `core/_phase_composite_post_xt_ensemble/__init__.py:1120`: `_ensemble.predict(val)` and `_ensemble.predict(test)`, which runs every component predict.
  - `core/_phase_composite_post_moe.py:183-193`: `_ens_model.predict(filtered_val_df)` and `_raw_shim.predict(filtered_val_df)` again.
  - `core/_phase_composite_post_xt_ensemble/__init__.py:544`: the pre-screen's component predicts on val (see PRF-04).
- **What**:
  - Each component's y-scale val prediction is computed once in the per-model hook. It is computed again inside the CT_ENSEMBLE report, and a third time by the MoE gate.
  - Every one of these calls also re-runs the shim's `pre_pipeline.transform` on the full val frame.
  - The only existing cache (`_train_pred_cache` / `_build_pred_cache`) covers the train frame only.
- **Expected win** (estimate): 2 of 3 val predict passes and 1 of 2 test passes per component, per target. On MLP/CatBoost components with large val/test frames, that is minutes per target.
- **Suggested fix**:
  - Extend the build-scoped cache to `(id(inner), split_name, frame identity)` for val and test, filled by the per-model hook.
  - Give the ensemble a `predict_from_component_preds` path, reusing the "gate==deploy" combine helper, so the report and MoE combine cached columns instead of calling `predict`.
- **Test/benchmark to add**: a mock-counted predict test over one composite target, asserting each (component, split) is predicted once; a wall-time bench on a 3-component fixture.
- **Disposition**: PARTIAL - `run_composite_post_processing` now runs under a phase-scoped prediction memo (`core/_prediction_memo.py`): the wrap pass's val/test predicts, the CT_ENSEMBLE report, the MoE gate's ensemble and raw-model predicts and the refit pre-screen all go through `memo_predict`, keyed on the (model, frame) pair with strong references held so an `id()` cannot be reused, and dropped when the phase exits. Measured on the composite integration fixture: the MoE gate's ensemble val prediction is now served from the report's call, which saves a full ensemble predict (every component, plus the shim's pre-pipeline) per target; ensemble predicts on the val frame went 3 -> 2. The pre-screen needs four or more components and the wrap-pass metric loop did not run under that config, so component-level reuse is wired but not exercised end-to-end there. NOT done: a `predict_from_component_preds` path for the ensemble, which would let the report combine cached component columns instead of calling the components at all. test_composite_post_prediction_memo.py, 6 tests: one predict per pair, pairs kept apart, no caching outside the phase, memo dropped at phase end, copies per caller, decorator scoping

### PRF-14 [P2] The wrap-pass metric block runs four full inner predicts per (entry, split) where one would do

- **Where**: `core/_phase_composite_wrapping.py:518-527` (`predict` and then `predict_pre_clip`), `:651-655` (watchdog universal `_wi_uni.predict`) and `:699-702` (additive watchdog `_wi.predict`). The block runs when `skip_wrap_pass_predict=False`.
- **What**:
  - For each wrapped entry and each of train, val and test, the code calls:
    1. `wrapper.predict(X)`, which runs the inner predict and the inverse.
    2. `wrapper.predict_pre_clip(X)`, the same inner predict and inverse without the final clip.
    3. `estimator_.predict(X)` for the universal watchdog.
    4. `estimator_.predict(X)` again for the additive watchdog, whose result equals the third call.
  - Train is the largest split. The docstring prices the block at "~5-15 min".
- **Expected win** (estimate): about 4x on this block when it is enabled.
- **Suggested fix**:
  - Call `_predict_unclipped` once, derive the clipped value with the same `np.clip`, and return `t_hat` in `meta`.
  - Feed that `t_hat` to both watchdog checks.
  - Keep the watchdog semantics by comparing `inverse(t_hat)` to the wrapper output computed from the same `t_hat`. Alternatively, keep one independent inner predict for the universal check only, if its purpose is to catch wrapper state loss (EST-08 discusses what the watchdog can detect).
- **Test/benchmark to add**: a mock-counted inner-predict test for `_run_composite_target_wrapping(skip_predict=False)` expecting 1 or 2 predicts per (entry, split) instead of 4.
- **Disposition**: PARTIAL - the universal and additive watchdogs now take the inner prediction through the phase's prediction memo (PRF-13), so their two identical `estimator_.predict` calls on a split cost one. Measured on the integration fixture with `skip_wrap_pass_predict=False`: inner predicts per additive composite went train 4 -> 3, val 10 -> 9, test 9 -> 8 (43 -> 40 over the run). The wrapper's own `predict` and `predict_pre_clip` are left as two calls on purpose: `predict` records the clip-violation counters in `runtime_stats_`, which the report and the model card surface, so deriving the clipped value from the unclipped one would change those numbers; and the universal watchdog keeps an inner predict independent of the wrapper's, since catching wrapper state loss is its job (EST-08). The block is off by default (`skip_wrap_pass_predict=True`), so this only matters when it is enabled. test_wrap_pass_watchdog_single_inner_predict.py: fewer inner predicts than a run whose watchdogs predict independently, never more

### PRF-15 [P2] Under the supported pandas range, the per-target discovery frame is a full copy of the train frame

- **Where**: `core/_phase_composite_discovery_helpers.py:88-112` (`_build_disc_df_for_target`), called per regression target at `core/_phase_composite_discovery.py:478`.
- **What**:
  - For pandas input, the function returns `pd.concat([filtered_train_df[cols_wo_target], target_series], axis=1)`.
  - `pyproject.toml` pins `pandas>=1.5,<3.0`. There, list-column selection materialises a copy, and `concat` with its default `copy=True` can copy again. That is one or two transient full-frame copies per target just to attach the y column.
  - Measured on pandas 3.0.3 (Copy-on-Write default): **zero-copy** (5 ms, +5 MB RSS, `shares_memory=True` on a 200 MB frame). I did not measure on pandas 2.x, so the copy is inferred from pandas semantics.
  - Discovery reads the target only through `_extract_column_array(df, target_col)`.
- **Expected win** (estimate, pandas 1.5-2.x only): up to 2x the train-frame size in transient RAM per regression target.
- **Suggested fix**: pass `y` to `CompositeTargetDiscovery.fit` as an array (an optional `y=` argument used instead of `df[target_col]`) and stop injecting the column. Or use `df.copy(deep=False)` plus assigning the new, previously absent column, which does not touch existing blocks.
- **Test/benchmark to add**: a pandas-2.x CI job running a peak-RSS bench of the discovery phase on a 1M x 200 pandas frame, and asserting the caller's frame is unmodified.
- **Disposition**: COMPLETED - `_build_disc_df_for_target` no longer selects a column list when the target is not already a column (the usual case), and passes `copy=False` to `pd.concat` on pandas < 3, where the keyword is not deprecated; on pandas 3 it passes nothing, since copy-on-write already makes this zero-copy. The result is still a new frame, so the caller's is untouched. Verified on pandas 3.0.3 only (this host): feature columns share memory with the train frame, the caller's frame gains no target column, an existing target column is replaced not duplicated. The pandas-2.x copy is inferred from pandas semantics as the finding says; the same `np.shares_memory` assertion is what a pandas-2 CI leg would check. test_disc_df_does_not_copy_train_frame.py, 3 tests

### PRF-16 [P2] K-fold OOF re-runs a shared `pre_pipeline.transform` for every component on every fold

- **Where**: `composite/ensemble/__init__.py:558` (K-fold loop), `:306` (external holdout) and `:818` (single split), all through `_transform_pair_via` (`:62-96`).
- **What**:
  - Inside each fold, the code calls `pp.transform(X_stack)` and `pp.transform(X_holdout)` separately for every component.
  - Components that hold the same fitted pre-pipeline object repeat identical transforms over about 160k rows each, at the default `oof_max_train_rows=200_000` and K=5.
  - I did not confirm how often a suite shares one pre-pipeline object across components. It depends on the per-strategy `pre_pipeline` construction in `_phase_train_one_target_body.py`.
- **Expected win** (estimate): per fold, `(components sharing a pp - 1) x 2` pipeline transforms. Zero when pipelines are not shared.
- **Suggested fix**: memoise `_transform_pair_via` results per fold in a dict keyed by `id(pp)`, only for fitted pipelines; unfitted pipelines are fold-fit clones and stay per component. Release the memo at the end of each fold.
- **Test/benchmark to add**: a test with two components sharing one fitted pipeline that asserts `transform` is called 2 times per fold instead of 4, and that OOF matrices are unchanged.
- **Disposition**: COMPLETED - premise measured first: on the composite integration suite half of all `_transform_pair_via` calls repeated an earlier one on the same fitted pipeline object and the same fold slices (5 of 10 with `linear`, 10 of 20 with `linear,lgb`). All three OOF loops now go through `_transform_pair_cached` with a memo per fold keyed on (pipeline, stack slice, holdout slice), for fitted pipelines only; an unfitted pipeline is fit as a fold clone per component and never enters the cache. On the `linear,lgb` suite transform calls went 20 -> 15 (the other five are pipeline-less components, a pass-through) and the OOF prediction matrices are bit-identical. test_oof_shared_pipeline_transformed_once.py, 4 tests: shared fitted pipeline transforms once per slice, a new fold transforms again, an unfitted pipeline is never cached, cached values equal a direct transform

### PRF-17 [P3] Every `fit` computes a full `data_signature` that only `discover_incremental` reads

- **Where**: `discovery/_fit.py:817-823`; `composite/cache.py:158-240`.
- **What**:
  - After every discovery fit (and every stability or per-group replicate), `data_signature(df, target_col, feature_cols)` runs. It computes whole-column min/max/null statistics, a head/tail row fingerprint, and a per-column encoded sample of up to `sample_n` rows.
  - Measured at 200k x 50: **308 ms on polars, 84 ms on pandas**. The polars cost is per column (about 2 ms/column of small `select`/`collect` calls in `encode_polars_slice` and `row_order_fingerprint`), so it is an estimated 1-3 s at 500 columns.
  - When the disk cache is on, the phase also computes its own signature of the same frame (`core/_phase_composite_discovery.py:533-536`).
- **Expected win** (estimate): about 1-3 s per target on wide polars frames, plus the duplicate when caching.
- **Suggested fix**: compute it lazily in `discover_incremental` from the retained `self._df_ref`, or reuse the phase's `_df_sig` when present. Batch the per-column polars sample encodes into one `select`.
- **Test/benchmark to add**: a test that `discover_incremental` still hits the byte-identical fast path; a bench of `data_signature` at 200k x 500 on polars.
- **Disposition**: OPEN

### PRF-18 [P3] Auto-base and `fit` build the identical 100k-row screen feature matrix twice

- **Where**: `discovery/_auto_base.py:158-167` and `discovery/_fit.py:225-348`. Both call `_sample_indices(train_idx.size, mi_sample_n, random_state, strategy, y=y_train, n_strata)` and then `_build_feature_matrix(df, usable_features, ...)`.
- **What**: both sites draw the same sample; `fit` only reorders it by time when `time_ordering` is given. Each then gathers every usable column on those rows from the frame, a polars `gather` per column on a possibly multi-million-row frame.
- **Expected win** (estimate): one `n_screen x F` gather per target, roughly 1-2 s at 100k x 500 on a large polars frame.
- **Suggested fix**: have `_auto_base` stash `(train_idx_screen, x_matrix)` on the instance. In `fit`, reuse it by permuting rows with the time order. Mind that `_auto_base` imputes a copy, so stash the pristine matrix.
- **Test/benchmark to add**: count `_build_feature_matrix` calls per fit.
- **Disposition**: OPEN

### PRF-19 [P3] Auto-chain upcasts each base's matrix to float64 and copies `x_tr` for every candidate on every fold, even when the fold Dataset is already cached

- **Where**: `discovery/_auto_chain.py:440` (`x_matrix = np.asarray(x_matrix, dtype=np.float64)`) and `:254` (`x_tr, x_va = x_matrix[tr_idx], x_matrix[va_idx]` inside `_y_scale_cv_rmse`).
- **What**:
  - Bases run in parallel threads, and each doubles its float32 tiny-sample matrix to float64.
  - Every one of the roughly 12 candidates then fancy-index-copies both fold slices, although `LgbFoldCache` needs `x_tr` only on its first call per fold.
- **Expected win** (estimate): half the per-base matrix memory, and about `12 x folds - folds` avoided `x_tr` copies per base.
- **Suggested fix**: keep float32, since LightGBM bins it anyway. Build `x_va` once per fold and `x_tr` only when the fold cache is cold.
- **Test/benchmark to add**: a parity test that chain RMSEs are unchanged; an RSS bench at 20k x 400.
- **Disposition**: OPEN

### PRF-20 [P3] The interaction-base step, on by default, synthesises its columns twice, and its output never becomes a spec

- **Where**: `discovery/_interaction_bases.py:123` (`score_interaction_pairs`) and `:196` (`discover_interaction_bases` calls `generate_interaction_bases` again). Stashed only at `discovery/_opt_in_steps.py:119-155`.
- **What**: every pairwise product/ratio column over the top-4 bases is generated in the scoring pass and again in the selection pass. The result is stored on `interaction_bases_` for reporting and never re-screened into `specs_`.
- **Expected win** (estimate): small, about 24 column syntheses plus MI at 100k rows per target.
- **Suggested fix**: return the synthesised columns from `score_interaction_pairs` and reuse them. Consider gating the step behind a reporting flag until it feeds specs.
- **Test/benchmark to add**: a test that `generate_interaction_bases` is called once per fit.
- **Disposition**: OPEN

### PRF-21 [P3] The opt-in bootstrap MI recomputes the same `MI(y, X)` replicates for every transform on a base

- **Where**: `discovery/_eval.py:585-611`.
- **What**:
  - The bootstrap RNG is re-seeded per candidate from the fixed `mi_gain_bootstrap_random_state`. For every transform on a base with the same `valid_screen` mask, the `idx_b` draws and therefore the `mi_y_b` replicates are identical, yet each candidate recomputes them.
  - `_x_prebinned[valid_screen]` is also copied even when the mask is all-true.
- **Expected win** (estimate): about 2x on bootstrap cost when `mi_gain_bootstrap_n > 0`; off by default.
- **Suggested fix**: memoise the `mi_y_b` replicate vector in the base context keyed by `hash(valid_screen.tobytes())`, as `_mi_y_compare_memo` already does. Skip the copy when the mask is all-true.
- **Test/benchmark to add**: a bit-identity test of `mi_gain_lcb` and `bootstrap_p_value` with and without the memo.
- **Disposition**: OPEN

### PRF-22 [P3] The prebin content cache hashes the screen matrix on every fit but misses across targets

- **Where**: `discovery/screening.py:434-463` (`_prebin_feature_columns_cached`) and `discovery/_fit.py:355-363`.
- **What**:
  - Each fit hashes the whole float32 screen matrix, an estimated 0.2-0.4 s at 100k x 500.
  - The default `mi_sample_strategy="stratified_quantile"` stratifies on each target's own y. Different targets therefore draw different rows, and the cache can only hit when the same target is re-discovered.
- **Expected win** (estimate): the hash cost per target in multi-target suites.
- **Suggested fix**: key on `(frame signature, train_idx_screen hash, nbins)`, which is cheap, instead of the matrix bytes. Or skip the lookup when the row set differs from the last put.
- **Test/benchmark to add**: a counter test on a 3-target suite showing hits/misses; a hash-cost micro-bench.
- **Disposition**: OPEN

### PRF-23 [P3] Region-adaptive, which is opt-in, fits full-region parameters for every candidate in every region, then keeps only the winner's

- **Where**: `discovery/_region_adaptive.py:199-202` (`_oof_score_transform` returns `tr.fit(y, base)` for each candidate) and `:248-253`.
- **What**: every candidate refits on the full region after its OOF folds, and every non-winning fit is discarded. The winner-seeding `linear_residual` fit is also done up front.
- **Expected win** (estimate): `len(candidates) - 1` full-region fits per region. Small.
- **Suggested fix**: return only the OOF score, then fit the winner once per region.
- **Test/benchmark to add**: a test that the fitted `RegionAdaptiveSpec` is unchanged.
- **Disposition**: OPEN

### PRF-24 [P3] The multi-target OOF polars slice converts fold indices to a Python list

- **Where**: `core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py:43-45`.
- **What**:
  - `X[idx.tolist()]` builds a Python list of about 160k ints per fold and component slice. Polars row-indexing accepts an integer ndarray directly.
  - The `hasattr(X, "__getitem__")` guard is always true, so the `filter` fallback is unreachable.
- **Expected win** (estimate): a few ms per fold. Small, but free.
- **Suggested fix**: use `X[idx]` (or `X.gather(idx)`) and drop the dead branch.
- **Test/benchmark to add**: a parity test that the slices are equal.
- **Disposition**: OPEN
