"""Documentation payload for ``screen_predictors``: its full docstring and the per-parameter design notes.

Kept out of ``_screen_predictors.py`` so the function itself stays within the function-length gate.
"""

from __future__ import annotations

# Notes on the screen_predictors parameters (kept beside the definition; the signature itself is packed).

# [factors_data]
# factors
# [targets_data]
# targets
# [mrmr_relevance_algo]
# algorithm
# [extra_x_shuffling]
# performance
# [subsample_idx]
# ONE shared FE subsample ("score the screen on the same rows"). When supplied, the order-1 relevance SWEEP + its maxT FDR floor are computed on
# THESE rows (consistent estimator scale, and the full-n permutation work disappears); the RETURNED target encodings (classes_y / freqs_y) are recomputed at
# FULL n so the downstream FE pipeline stays row-aligned. None -> full-n screen.
# [min_occupancy]
# confidence Statistical defaults aligned with the MRMR constructor (mrmr.py) so a direct ``screen_predictors`` caller - and any consumer that does not
# override every knob (e.g. ad-hoc screening) - gets the same behaviour MRMR.fit produces, rather than the far-stricter legacy 1000/100 permutation counts
# that made standalone screening ~300x slower and over-reject. MRMR.fit still explicitly overrides each of these.
# [fe_confirm_undersample_rows_per_cell]
# Sample-size-aware Fleuret confirmation threshold (rows per occupied cell of the conditioning joint). Below it the conditional-MI
# permutation gate is finite-sample unreliable and ``confirm_candidate`` falls back to a marginal-MI permutation test. 0.0 = always use the strict
# conditional test (legacy). Threaded into ``ScreenContext``.
# [min_relevance_gain]
# stopping conditions
# [min_relevance_gain_relative_to_first]
# diminishing-returns stop. Stops greedy selection when the
# current candidate's gain falls below this fraction of the FIRST-selected
# feature's gain. Catches "trailing noise" leakage on imbalanced y where
# tiny-but-statistically-positive gains squeeze past min_relevance_gain
# (e.g. Layer 13: signal gain 0.0176, noise gain 0.0004 - both clear the
# H(y)-relative floor at 1% imbalance, but noise is 2.5% of signal).
# 0.0 disables; default 0.05 = stop once gain drops below 5% of first
# gain. Only applies from the second selected feature onward.
# [cardinality_bias_correction]
# Miller-Madow MI bias correction at the selection gate.
# Plug-in mutual information OVERESTIMATES MI for high-cardinality
# features (Paninski 2003, Miller 1955). For a binned feature with
# |X| bins, target with |Y| classes, and n samples the plug-in MI
# picks up a bias of ~(|X|-1)*(|Y|-1)/(2n). On a 1200-level user_id
# at n=2500 with binary y that's ~0.24 nats - enough to make pure
# noise outrank real numeric signal (Layer 10 seed=101 hijack:
# user_id raw gain 0.328, after MM correction 0.088; num_signal_1
# raw 0.187 -> corrected 0.185; the corrected ordering puts the
# real signal first and demotes user_id to #3, where the relative-
# gain floor then excludes it). True = subtract MM bias from gains
# at the floor comparison only (does NOT mutate mrmr_gains_ which
# remains the raw plug-in value for downstream consumers).
# [raw_cardinality_cols]
# Column NAMES whose bin count is a raw level count (categoricals). Only these are eligible for the
# cardinality ceiling: a numeric column discretised by a SUPERVISED strategy earns bins for EXPLAINING the
# target, so judging it by bin count drops the strongest feature. None = judge every column by bin count.
# [max_confirmation_cand_nbins]
# Confirmation-step cardinality cutoff. ``None`` falls back to ``MAX_CONFIRMATION_CAND_NBINS``; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2``.
# [fe_fallback_to_all]
# When screening returns zero selected_vars, legacy FE fell back to running on ALL features. False skips FE instead (safer default: FE on empty screen amplifies noise). Aligned with the MRMR ctor default (False).
# [verbose]
# verbosity and formatting
# [use_simple_mode]
# Aligned with the MRMR ctor (False): full Fleuret conditional-MI redundancy is the point of MRMR. True is the faster dedup-free path on very wide pools.
# [engineered_lineage]
# ``engineered_lineage`` - mapping ``{engineered_col_idx: frozenset(parent_indices)}``. When set, k-way candidate enumeration skips combinations of an engineered
# column with one of its own parents (e.g. ``(orig_i, kway(orig_i, orig_j))`` is redundant since the engineered col already contains orig_i's info). Threaded
# through ``should_skip_candidate``. ``None`` preserves legacy behaviour.
# [dcd_config]
# Dynamic Cluster Discovery (DCD) config dict. When
# not None, DCD is active for this screen call. Keys: enable, tau_cluster,
# distance, cluster_size_threshold, swap_gain_threshold, swap_method,
# pairwise_cache_max, min_cluster_size, max_cluster_size, X_raw,
# quantization_method, quantization_nbins, quantization_dtype, factors_cols.
# None preserves pre-Wave-9 behaviour bit-stable.
# [engineered_recipes]
# Layer 43 (PART A) — host-MRMR engineered_recipes dict (name ->
# EngineeredRecipe). Threaded into ``commit_swap`` so the DCD PC1 aggregate
# gets registered as a replayable recipe. Pre-fix it was hardcoded to None
# at the commit_swap callsite -> aggregate appeared in ``selected_vars`` /
# ``cols`` but the remap in _mrmr_fit_impl.py dropped it from
# ``_engineered_recipes_`` so ``get_feature_names_out`` silently lost it.
# [raw_feature_names]
# — directed-FE tie-break (see ScreenContext.raw_feature_names).
# ``raw_feature_names`` is the set of ORIGINAL (pre-FE) input column names;
# any candidate whose ``factors_names`` entry is not in it is engineered.
# On a near-tie in selection gain (within ``prefer_engineered_rel_eps``
# relative tolerance) the greedy pick prefers the engineered candidate over a
# raw one - deterministic, and the whole point of directed FE: surface the
# nonlinear combination, not its raw parent (which a shallow downstream can't
# use). ``None`` raw-name set falls back to the syntactic ``(``/``__``
# heuristic; rel-eps ``0.0`` restores the legacy pure-index tie-break.
# [prefer_engineered_rel_eps]
# RC1: widened 0.01 -> 0.15. On non-monotone targets (e.g.
# y=sign(x1^2-1)) the binned-MI of the raw parent x1 edges out its engineered
# linearizer x1__He2 by more than 1% (estimation noise; in EXACT MI they are
# equal because He2 is a deterministic function of x1), so a 1% band left the
# raw parent winning and the engineered linearizer pruned - the downstream a
# linear model needs (x1^2-1) was lost (AUC ~0.5 vs ~0.99). A 0.15 band lets
# the engineered linearizer win over its near-tied raw parent. Still gated to
# the leading band + engineered-only promotion (clear raw winners untouched).
# [screen_fdr_null_permutations]
# — Westfall-Young maxT permutation-null gain floor. In a wide
# candidate pool (embedding / TF-IDF matrices, p >> sqrt(n)) the MAX marginal
# MI over p pure-noise columns is a positive order statistic that grows with
# p; per-candidate Miller-Madow correction centres each column's EXPECTED
# bias near zero but cannot remove this max-over-p selection inflation, so the
# best-of-p noise clears the abs/rel floors by chance and the greedy admits a
# noise cloud (Layer-20 p=500: 15 noise dims pass a <=15 bound). The null
# shuffles y K times, records the per-shuffle MAX corrected marginal MI over
# the pool, and floors order-1 selection at the q-th quantile of that
# distribution - the chance ceiling for THIS pool. SELF-GATING: tiny at small
# p (keeps weak genuine signals), large at high p (rejects the noise cloud).
# Applied when the pool has >= ``screen_fdr_min_features`` candidates (wide pool: embedding / TF-IDF best-of-p bias) OR when a NARROW pool meets the high-cardinality-target
# gate (``screen_fdr_target_oversplit_ratio`` / ``screen_fdr_min_rows_per_joint_cell``). The narrow-pool gate catches a distinct finite-sample-bias regime: a heavy-tailed
# (log-normal) regression target whose quantile binning yields a high-cardinality target (~10 equal-frequency bins, matching feature cardinality), lifting pure-noise columns
# past the abs/rel gain floors after the genuine signals are picked. The gate fires ONLY when the target is high-cardinality (nbins_y >= median feature nbins - a continuous
# regression target, not a low-card classification one) AND the (X,y) joint table is dense enough that the floor is itself reliable - so a dense weak-signal regression pool
# like sklearn diabetes (MDLP nbins_y=64 but ~1.0-1.5 rows per joint cell at n=330) keeps the floor OFF and preserves its 10 weak features, while lognormal (7.8 rows per
# joint cell at n=2500) fires. The original ratio=3 keyed on the MDLP ~30-bin over-split that the target-rebin guard now removes. See
# the narrow-pool rule in ``_screen_predictors_prescreen.compute_fdr_gain_floor``. ``screen_fdr_null_permutations=0`` disables. Default 200 (raised from 25): the floor is the 95th
# percentile of a per-shuffle MAX, an extreme upper-tail order statistic whose K=25 estimate is high-variance (~1.25 draws above the quantile); 200 draws stabilise the
# floor several-fold run-to-run (bench ``_benchmarks/bench_maxt_floor_stability.py``) - a lower-variance noise floor is the correct behavior, the rescore cost stays
# sub-second at production widths.
# [screen_fdr_min_rows_per_joint_cell]
# 8.0 -> 7.0. MDLP now bins the lognormal biz_val target into 64 bins (not the
# ~10-30 assumed at this gate's original calibration), which pushed the lognormal fixture's
# measured rows/cell to exactly 7.8125 - just under the old 8.0 bar, silently disabling the
# gate it exists to trigger. Diabetes (the no-regression side) measures ~1.0-1.5 rows/cell,
# so 7.0 keeps its comfortable separation while restoring the lognormal win.
# [existing_dcd_state]
# When MRMR.fit re-screens after a feature-engineering step (confirm-rescreen
# loop), the DCDState from the prior pass is threaded back in here so cluster
# discovery accumulates across passes instead of being rebuilt empty each
# time (which dropped the screen-1 dup cluster from the published summary).
# [seed_caches]
# fix: analogous to ``existing_dcd_state`` above - thread the PRIOR screen round's
# relevance/redundancy caches back in instead of starting empty every round. A given cache key
# (a column-index tuple, or an (X,Z) conditional-MI key) is a deterministic function of the data,
# which does not change across rounds (only grows via appended engineered columns; existing
# indices' content is stable), so a cached value computed in round N-1 is EXACT, not approximate,
# for round N - the same guarantee this memoization already provides WITHIN one round, just not
# previously carried across the 2-3 rounds a typical fit runs. ``seed_caches``, when given, is the
# 4-tuple ``(entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs)`` returned by a
# PRIOR ``screen_predictors`` call in the same fit. ``None`` (default) preserves the legacy
# fresh-empty-caches behaviour.
# [seed_maxt_floor_cache]
# fix: optional cross-round cache for the maxT permutation-null gain floor (see
# ``compute_fdr_gain_floor``'s ``maxt_floor_cache`` docstring). A plain dict the caller owns and
# threads unchanged into every ``screen_predictors`` call within one fit (mutated in place - no
# need to read it back from the return tuple). ``None`` (default) disables caching, legacy behaviour.
# [seed_workers_pool]
# fix: thread back a joblib ``Parallel`` pool built by a PRIOR ``screen_predictors``
# call in the same fit (only meaningful when ``n_workers>1``). Building a fresh pool + eager
# warmup dispatch on every one of the ~3 screen/re-screen rounds per fit respawns ``n_workers``
# threads each time for no benefit - the pool config (backend/n_jobs) is fixed by the MRMR
# instance for the whole ``fit()`` call, so a warmed pool from round 1 is safe to reuse verbatim
# in rounds 2-3. ``None`` (default) preserves the legacy fresh-pool-per-call behaviour.

SCREEN_PREDICTORS_DOC = """Finds best predictors for the target. ``factors_data`` must be an n-by-m array of integers (ordinal encoded).

    ``max_confirmation_cand_nbins=None`` falls back to the module constant for backward compat; ``MRMR.fit`` overrides explicitly. ``fe_fallback_to_all`` is consumed
    by ``MRMR.fit`` and only threaded here for caller pass-through.

    Args:
        factors_data: n-by-m array of ordinal-encoded integer factor columns.
        factors_nbins: number of bins (cardinality) of each factor column.
        factors_names: names of the factor columns; generated when None.
        factors_names_to_use: restrict screening to these column names.
        factors_to_use: restrict screening to these column indices.
        targets_data: integer-encoded target columns; built from ``y`` when None.
        targets_nbins: number of bins of each target column.
        y: indices of the target columns inside ``factors_data``.
        mrmr_relevance_algo: relevance scheme.
            "fleuret": max(min(I(X,Y|Z)),max(I(X,Y|Z)-I(X,Y))) Possible to use n-way interactions here.
            "pld": I(X,Y)
        mrmr_redundancy_algo: redundancy scheme.
            "fleuret": 0 ('cause redundancy already accounted for)
            "pld_max": max(I(veterane,cand)) Possible to use n-way interactions here.
            "pld_mean": mean(I(veterane,cand)) Possible to use n-way interactions here.
        reduce_gain_on_subelement_chosen: reduce a candidate's gain when one of its sub-elements was already chosen.
        extra_x_shuffling: also shuffle the candidate column in permutation tests.
        dtype: integer dtype used for the working copy of the data.
        random_seed: seeds numpy, numba and (with ``use_gpu``) cupy for the duration of the screen; None leaves RNG state untouched.
        subsample_idx: row indices of one shared FE subsample on which the order-1 sweep and its maxT floor are computed; None means full n.
        use_gpu: run the MI kernels on the GPU.
        n_workers: number of joblib workers for candidate evaluation.
        min_occupancy: minimum rows per occupied cell for a candidate to be considered; None disables the check.
        min_nonzero_confidence: if in random permutation tests this or higher % of cases had worse current_gain than original, current_gain value is considered valid, otherwise, it's set to zero.
        full_npermutations: when computing every MI, repeat calculations with randomly shuffled indices that many times
        baseline_npermutations: number of permutations used for the cheaper baseline (order-1) significance checks.
        fe_confirm_undersample_rows_per_cell: rows per occupied conditioning cell below which confirmation falls back to a marginal-MI permutation test; 0.0 always uses the strict conditional test.
        min_relevance_gain: absolute gain below which selection stops.
        min_relevance_gain_relative_to_first: stop once a candidate's gain drops below this fraction of the first selected feature's gain; 0.0 disables.
        cardinality_bias_correction: subtract the Miller-Madow bias from gains at the floor comparison (the raw gains stay unmodified).
        raw_cardinality_cols: names of columns whose bin count is a raw level count and so eligible for the cardinality ceiling; None judges every column by bin count.
        max_consec_unconfirmed: stop after this many consecutive candidates fail confirmation.
        max_runtime_mins: wall-clock budget in minutes; None means unlimited.
        interactions_min_order: smallest interaction order enumerated.
        interactions_max_order: largest interaction order enumerated.
        interactions_order_reversed: enumerate interaction orders from largest to smallest.
        max_veteranes_interactions_order: maximum interaction order evaluated against already selected features.
        only_unknown_interactions: True for speed, False for completeness of higher order interactions discovery.
        max_confirmation_cand_nbins: confirmation-step cardinality cutoff; None falls back to ``MAX_CONFIRMATION_CAND_NBINS``.
        fe_fallback_to_all: consumed by ``MRMR.fit``; only passed through here.
        verbose: int  1=log only important info,>1=also log additional details
        ndigits: decimals used when logging numeric values.
        parallel_kwargs: extra keyword arguments for the joblib ``Parallel`` pool.
        stop_file: path of a file whose existence requests an early stop.
        use_simple_mode: True selects the faster dedup-free path on very wide pools instead of the full Fleuret conditional-MI redundancy.
        engineered_lineage: mapping ``{engineered_col_idx: frozenset(parent_indices)}`` used to skip combinations of an engineered column with its own parents.
        dcd_config: Dynamic Cluster Discovery configuration; None leaves DCD off.
        engineered_recipes: host ``engineered_recipes`` dict (name to recipe) into which DCD aggregates are registered.
        raw_feature_names: original (pre-FE) input column names; candidates outside it are treated as engineered for the near-tie preference.
        prefer_engineered_rel_eps: relative gain tolerance within which an engineered candidate is preferred over a raw one; 0.0 restores pure-index tie-breaking.
        screen_fdr_null_permutations: number of y shuffles for the maxT permutation-null gain floor; 0 disables it.
        screen_fdr_null_quantile: quantile of the per-shuffle maximum used as the floor.
        honest_null_codes_fn: callable re-binning every column on a row-permuted raw target (supervised binning); None keeps the plain permutation null.
        screen_fdr_min_features: pool width at which the maxT floor is applied.
        screen_fdr_target_oversplit_ratio: target-cardinality ratio that triggers the floor on narrow pools.
        screen_fdr_min_rows_per_joint_cell: minimum rows per (X, y) joint cell for the narrow-pool floor to be considered reliable.
        existing_dcd_state: DCD state from the prior screening pass, so cluster discovery accumulates across passes.
        seed_caches: 4-tuple ``(entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs)`` from a prior call in the same fit; None starts with empty caches.
        seed_maxt_floor_cache: caller-owned dict caching the maxT floor across calls, mutated in place; None disables caching.
        seed_workers_pool: warmed joblib ``Parallel`` pool from a prior call, reused when ``n_workers`` > 1; None builds a fresh pool.

    Returns:
        1) best set of non-redundant single features influencing the target
        2) subsets of size 2..interactions_max_order influencing the target. Such subsets will be candidates for predictors and OtherVarsEncoding.
        3) all 1-vs-1 influencers (not necessarily in mRMR)
    """
