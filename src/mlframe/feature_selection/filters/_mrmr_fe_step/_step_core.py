"""MRMR._run_fe_step -- the FE-step orchestrator for mlframe.feature_selection.filters.mrmr.

The single irreducible ~1.36k-line _run_fe_step function lives here, verbatim. The package
__init__ re-exports it (and the small _helpers symbols), and mrmr.py binds it back onto
the MRMR class at its module bottom, so self._run_fe_step(...) call sites are unchanged.

The parent module helpers (_lazy_chunks etc.) and every other intra-filters dependency are
imported lazily in-body (from ..mrmr import ... etc.) since .mrmr re-imports this package at
its bottom for method binding -- a top-level import would create a hard cycle. The self-contained FE
sub-blocks (synergy bootstrap, order-2 maxT floor, FE summary log, cluster-aggregate emission) live in
the sibling _mrmr_fe_step_helpers module; the two small operand-pool helpers live in ._helpers.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import delayed

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._mrmr_fe_step_helpers import (
    apply_interaction_information_routing,
    apply_surrogate_gbm_seeder,
    apply_synergy_bootstrap,
    compute_pair_maxt_floor,
    log_fe_summary,
    run_cluster_aggregate_emission,
)
from .._gradient_interaction_seeder import propose_gradient_interaction_pairs
from ._helpers import _non_numeric_column_indices, _synergy_bootstrap_can_supply_pool


def _run_fe_step(
    self,
    *,
    # Mutable state from MRMR.fit (returned updated as tuple).
    data, cols, nbins, X,
    target_names, target_indices,
    selected_vars, categorical_vars,
    classes_y, classes_y_safe, freqs_y,
    cached_MIs, cached_confident_MIs,
    unary_transformations, binary_transformations,
    engineered_features, checked_pairs,
    # Parallel dict (name -> EngineeredRecipe) populated as new columns are added to data / cols so
    # transform() can replay them on test data. Mutated in place; MRMR.fit reads it after the FE loop
    # and copies surviving recipes into self._engineered_recipes_. ``None`` skips recipe construction.
    engineered_recipes=None,
    times_spent,
    num_fs_steps,
    # Service.
    n_jobs, prefetch_factor, parallel_kwargs,
    _is_polars_input, verbose,
    # FE config (frozen per fit).
    fe_max_steps, fe_npermutations, fe_max_pair_features,
    fe_print_best_mis_only, fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    fe_min_pair_mi, fe_min_pair_mi_prevalence,
    fe_smart_polynom_iters, fe_smart_polynom_optimization_steps,
    fe_min_polynom_degree, fe_max_polynom_degree,
    fe_min_polynom_coeff, fe_max_polynom_coeff,
    # Preset-name snapshot so recipes can rebuild the correct registry at replay time. Default "minimal"
    # matches MRMR.__init__ defaults; callers that override via self.fe_unary_preset / self.fe_binary_preset
    # get the actual values threaded through by fit().
    fe_unary_preset: str = "medium",
    fe_binary_preset: str = "minimal",
):
    """One Feature Engineering iteration. Extracted from ``MRMR.fit`` for testability and FE experimentation.

    Returns ``None`` if the FE step should not run (empty-screen + ``fe_fallback_to_all=False``); otherwise
    ``(data, cols, nbins, X, selected_vars, n_recommended_features)``. ``n_recommended_features == 0`` signals
    the outer loop to stop. Private; external callers should use ``MRMR.fit()`` or ``MRMR.fit_transform()``.
    """
    # Lazy import: ``.mrmr`` re-imports this module at its bottom for method
    # binding -> any top-level ``from .mrmr import ...`` here creates a hard
    # import cycle that ``tests/test_meta/test_no_import_cycles.py`` flags.
    from ..mrmr import (
        _lazy_chunks,
        _MRMR_BATCH_PRECOMPUTE_MAX_K,
        _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
        check_prospective_fe_pairs,
        compute_pairs_mis,
        discretize_array,
        get_new_feature_name,
        parallel_run,
        sort_dict_by_value,
        tqdmu,
    )
    if verbose:
        logger.info("MRMR+ selected %d out of %d features before the Feature Engineering step.", len(selected_vars), self.n_features_in_)

    _screening_returned_empty = False
    if len(selected_vars) == 0:
        if self.fe_fallback_to_all:
            logger.info("Proceeding with all features though (fe_fallback_to_all=True).")
            selected_vars = np.array([cols.index(col) for col in cols if col not in target_names])
        elif (getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0):
            # cluster_aggregate operates on raw ``feature_names_in_`` columns
            # (correlation-based clusters) and does NOT need ``selected_vars``.
            # When screening returns 0 features (heavily-correlated reflection
            # clusters routinely trigger this, since every member's marginal
            # MI is near-zero), the legacy ``return None`` skipped the
            # cluster-aggregate step too -- the test contract (and the
            # documented "ON by default and fires on a clean reflection
            # cluster" promise) requires running it regardless. Continue with
            # an empty ``selected_vars`` and let the cluster-aggregate block
            # at the bottom of this function fire; the pair / hermite blocks
            # in between are no-ops on empty ``selected_vars``.
            logger.info(
                "Screening returned 0 features but cluster_aggregate_enable=True; "
                "running cluster-aggregate step anyway (operates on raw "
                "feature_names_in_, not on selected_vars).",
            )
            selected_vars = []
            # 2026-06-03 (wave-9 follow-up, default_filtering.py:165): screening
            # selected nothing but we continue (interaction-only signal / cluster
            # aggregate). Flag this so the smart-polynom optimiser does NOT treat
            # the raw-seeded pool as "speculative synergy" to withhold (which
            # would exclude EVERY pair and the polynom search would never fire).
            _screening_returned_empty = True
        elif _synergy_bootstrap_can_supply_pool(self, num_fs_steps, data):
            # INTERACTION-ONLY SIGNAL (2026-06-05): screening returned 0 features because every operand of a pure-interaction target (a*b + c*d + ...) has ~0 MARGINAL MI -- and the
            # empirical-null debiasing (Fix B) now correctly demotes those near-zero marginals to exactly 0, so even the weak pre-debiasing marginal that used to slip one operand
            # through the screen no longer does. The signal is entirely in the PAIRS, which the synergy bootstrap below surfaces by seeding the all-pairs joint-MI sweep with the raw
            # numeric columns. Continue into the FE step (rather than skipping it) so that sweep + the smart-polynom / pair search can recover the interaction features. Without this,
            # the default-on DCD suppresses the cluster_aggregate fallback branch above, so a pure-interaction frame would silently engineer NOTHING. ``_screening_returned_empty``
            # disables the speculative-synergy withhold downstream (every operand is bootstrap-added here, so the withhold would otherwise drop every pair).
            logger.info(
                "Screening returned 0 features but the synergy bootstrap can seed an interaction-only pool "
                "(fe_synergy_screen_max_features>0); running the FE pair / smart-polynom search anyway.",
            )
            selected_vars = []
            _screening_returned_empty = True
        else:
            logger.info("Skipping Feature Engineering (screening returned 0 features and fe_fallback_to_all=False).")
            return None

    if _is_polars_input:
        import polars as pl  # noqa: F401  -- pl is used in the polars dispatch branches below

    # RAW-RETENTION capture (2026-06-03): record the SCREENING-confirmed genuine
    # raw features on the first FE step. The post-FE re-selection can drop a
    # screening-confirmed (permutation-validated) weak feature when an engineered
    # feature absorbs its signal as a redundant near-duplicate (measured: a genuine
    # X5 / pair operand absorbed into a noise-paired engineered feature, raw column
    # dropped). ``MRMR.fit`` re-adds these at support finalisation unless a
    # SINGLE-PARENT engineered child substitutes them (the prefer-engineered case).
    if num_fs_steps == 0:
        try:
            _raw_set = set(self.feature_names_in_)
            self._prefe_screened_raw_ = [cols[v] for v in selected_vars if cols[v] in _raw_set]
        except Exception:
            self._prefe_screened_raw_ = []

    n_recommended_features = 0
    if verbose >= 2:
        logger.info("Computing prospective FE pairs...")

    # bench-attempt-rejected (2026-06-11, FS backlog #6 surrogate-GBM split-co-occurrence 3-way seeder + #7 order-3 maxT floor):
    # Hypothesis was that a shallow GBM's gain-weighted root->leaf path co-occurrence could propose pure zero-marginal
    # 3-way synergy triples directly into this FE pool, bypassing the blind univariate top-N seed below. Decisive Stage-1
    # measurement (n=4000, p=200 iid, y=sign(x7*x42*x113)+0.3*noise; ALL univariate |corr| <= 0.026 and all
    # pairwise-product |corr| <= 0.026) shows the needle {7,42,113} NEVER surfaces:
    #   baseline150 / 1000r / extra_trees / colsample0.1 / extra+colsample : needle_rank=None, top-triple recall 0/3.
    #   best case (depth5 extra_trees 1000r): rank 18778/32947, score 12 vs top ~150 -- still 0/3.
    # Positive control (same walker) DOES rank a detectable pairwise XOR 0/1640 and a 3-way-with-pairwise-leak needle
    # 0/766, proving the null is the pure-3way's greedy-invisibility, not a code bug. Tree-path co-occurrence inherits
    # the greedy blindness it claimed to bypass (consistent with beam-search-useless-for-MRMR: conditional-MI is not
    # greedy-trappable). The order-3 Westfall-Young maxT floor (#7) is the mandatory rail for a 3-way proposer, but with
    # no working proposer it ships nothing -- both backlog items rejected/not-built together. A detect-without-enumerate
    # screen (#9 RFF) is the open non-greedy path to pure 3-way synergy.
    if self.fe_ntop_features:
        numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
    else:
        numeric_vars_to_consider = selected_vars

    numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

    # Drop any operand whose source column is not numeric in X, even if it was not flagged in ``categorical_vars``:
    # the polynomial / Hermite pair search applies numeric basis transforms (np.isfinite, z-score) that raise on a
    # string / categorical column (e.g. ``"ZZZ_UNSEEN"`` / ``"NA"``). Mirrors the per-stage numeric guards the
    # triplet / quadruplet stages already apply (_mrmr_fit_impl.py is_numeric_dtype filters).
    _non_numeric_idx = _non_numeric_column_indices(X, cols)
    if _non_numeric_idx:
        numeric_vars_to_consider = numeric_vars_to_consider - _non_numeric_idx

    # Honor factors_to_use / factors_names_to_use in the FE step too; intersect the FE pool with the user's
    # restriction so the contract matches the screening step.
    if self.factors_to_use is not None:
        numeric_vars_to_consider = numeric_vars_to_consider & set(self.factors_to_use)
    if self.factors_names_to_use is not None:
        allowed = {cols.index(n) for n in self.factors_names_to_use if n in cols}
        numeric_vars_to_consider = numeric_vars_to_consider & allowed

    numeric_vars_to_consider, _synergy_added_idx = apply_synergy_bootstrap(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        cols=cols,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # SURROGATE-GBM SPLIT-CO-OCCURRENCE SEEDER (2026-06-09, backlog #6) + its mandatory
    # order-3 maxT rail (#7). Fits one shallow LightGBM on the discretised matrix, walks
    # root-to-leaf paths, and tallies depth-discounted split-gain co-occurrence to propose
    # interaction PAIRS + TRIPLES whose operands have ~0 univariate MI -- the zero-marginal
    # needles (``y = sign(x_a*x_b*x_c)``) the univariate seed_count never reaches. Seeded PAIR
    # operands are MERGED into ``numeric_vars_to_consider`` so their joint MI is screened by the
    # existing pair pipeline below (bypassing seed_count); seeded TRIPLES are order-3-floored and
    # stamped onto ``self._seeded_triplets_`` for the triplet FE stage. SELF-GATES on a permuted-y
    # OOF comparison (pure noise -> no seeds -> pool not polluted); the order-2/order-3 maxT floors
    # then gate every emitted candidate. OPT-IN (``fe_gbm_seeder_enable``); self-routes OFF below
    # ``fe_gbm_seeder_min_features`` where seed_count is not the blocker. No-op otherwise.
    numeric_vars_to_consider, _gbm_seeded_pairs = apply_surrogate_gbm_seeder(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        nbins=nbins,
        cols=cols,
        categorical_vars=categorical_vars,
        target_indices=target_indices,
        classes_y=classes_y,
        freqs_y=freqs_y,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # GRADIENT-INTERACTION (MIXED SECOND PARTIALS) SEEDER (2026-06-10, backlog #21). Fits one
    # smooth differentiable RFF+ridge surrogate on a row sample, estimates E[(d2f/dxa dxb)^2] per
    # pair (a large mixed partial is the calculus definition of a non-additive interaction), and
    # proposes the operands of high-energy pairs into the SAME pool as #6 so their joint MI is
    # screened by the pipeline below. Targets SMOOTH/ROTATED interactions (a*b saddles, sin(a)*b).
    # Carries its OWN self-gate: an OOF-R2 vs permuted-y check (no learning -> 0 proposals), a
    # GAM additive-residual baseline (additive targets -> ~0 mixed partials), and a permutation
    # null on the max mixed-partial energy. OPT-IN (``fe_gradient_interaction_enable``, default
    # OFF -- see the module's bench-reject note: on the prescribed sin(x5)*x31 fixture the gradient
    # detector ranks the saddle #1 but the #6 GBM seeder does NOT under-rank it, so the two are
    # equally good rather than complementary there, and the full self-gated proposer is ~8 s at
    # n=2000/p=60). Routed OFF outside its size regime by ``_route_gradient_seeder``. No-op otherwise.
    numeric_vars_to_consider, _gradient_seeded_idx = propose_gradient_interaction_pairs(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        # The smooth surrogate REQUIRES the RAW CONTINUOUS values: ``data`` here is the
        # discretised (few-bin int) matrix every FE-gate MI is scored on, and a piecewise-constant
        # ~5-level step function has ~zero mixed partials regardless of the underlying smooth
        # interaction. ``X`` is the raw float frame (indexed by ``cols``); pass it so the RFF fit
        # can resolve the smooth saddle. (This is the architectural reason the detector is niche --
        # unlike the joint-MI sweep / GBM seeder, it cannot operate on the discretised pool.)
        X_continuous=X,
        cols=cols,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # bench-rejected (2026-06-09) -- backlog #9 "RFF / random-projection interaction pre-screen
    # (detect-without-enumerate)": a sibling proposer to the GBM seeder above that would draw R
    # random SPARSE (support 2-4 cols) hyperplane projections phi_r(x)=cos(w_r^T x_std + b_r), score
    # each phi_r's MI vs y, and promote supports whose best-of-trials MI uplift over the additive
    # baseline clears a permutation-null maxT floor -- meant to recover zero-marginal/smooth
    # interaction supports the all-pairs synergy bootstrap SKIPS when p > fe_synergy_screen_max_features.
    # Prototyped + benchmarked end-to-end (D:/Temp/rff_proto*.py, rff_coverage.py, rff_vs_allpairs.py,
    # rff_smooth_check.py, rff_largep_birthday.py; n=2000, p=500-3000). The MECHANISM is sound -- when
    # the needle support is COVERED, best-of-T RFF ranks it #0 with uplift 0.146 vs a null floor 0.0145
    # (10x), 0 noise false-promote -- but it loses in EVERY regime against the direct joint-MI sweep:
    #   1) COVERAGE is the binding constraint, not enumeration. A SPECIFIC localized k=2 needle in p=500
    #      has hit-probability R / C(p,2): R=2000 -> 1.6%, R=16000 -> 12%, R=124750 (== full all-pairs) ->
    #      63%; reaching 0.95 recall of ONE needle needs ~374k random draws. Random sparse supports cannot
    #      guarantee recall of a specific needle without ~enumerating all pairs.
    #   2) The BIRTHDAY-PARADOX rationale is wrong for sparse planted needles: with K=30 planted needles at
    #      p=3000 (C(p,2)=4.5M), R=20000 x T=6 = 120k evals (65s) DREW 0/30 needles -> recall 0/30. Hitting
    #      a fixed K-needle set needs draws ~ C(p,2)/K (~150k here), not R << p^2; the birthday paradox is
    #      about collisions AMONG draws, not hitting a fixed small target set.
    #   3) COST: one RFF best-of-T (T=8) support evaluation is 17x the cost of one direct joint-MI pair
    #      evaluation (4.1ms vs 0.24ms). 0.95-recall of one needle ~1533s of RFF vs ~30s for the full
    #      deterministic C(500,2) joint-MI sweep.
    #   4) NO EXCLUSIVE SWEET SPOT: the 2-D joint-MI estimator is a nonparametric (x_i, x_j, y) contingency
    #      MI, so it recovers the SMOOTH sin(x1*x2) needle (rank #0/2000, jMI 1.29) exactly as well as the
    #      zero-marginal product x3*x400 (rank #0/2000, jMI 1.51). The "smooth/rotated interactions trees
    #      miss" argument applies to the GBM seeder's AXIS-ALIGNED trees (#6/#21), NOT to the joint-MI sweep
    #      that #9 was meant to replace.
    # The correct fix for "needle missed when p > fe_synergy_screen_max_features" is to RAISE/relax that cap
    # (and its fe_synergy_max_sweep_cost n*p^2 budget): the direct all-pairs joint-MI bootstrap recovers
    # both the zero-marginal and the smooth needle DETERMINISTICALLY and ~17x cheaper at p<=1000 (full sweep
    # ~30s at p=500, ~120s at p=1000). RFF random-support prescreen would only conceivably help at p >~ 5000
    # where C(p,2) joint-MI is itself infeasible -- but even there it is coverage-bound (test #2 above) and
    # never recovers a SPECIFIC needle, so it is not implemented. Do NOT re-attempt as a default or opt-in
    # proposer without a fundamentally different (non-random-support) coverage scheme.

    # bench-rejected (2026-06-09) -- backlog #10 "Conditional-MI complementarity growth (Apriori lattice)":
    # grow triples from the order-2 SURVIVOR frontier (the prospective pairs below) by testing only third
    # columns c maximising the conditional-MI uplift I((a,b,c);y) - I((a,b);y) (reusing batch_triple_mi_prange
    # for scale-consistency with the #7 order-3 maxT floor), keeping triples above that floor; meant to catch a
    # 3rd operand that matters ONLY given (a,b) -- pure higher-order synergy the univariate triplet seed_count
    # misses. Prototyped + benchmarked end-to-end (D:/Temp/cmi_lattice_proto{,2,3,4,5}.py, cmi_lattice_cost.py;
    # n=3000, p=40, the backlog fixture). REJECTED: the Apriori ANTI-MONOTONE premise is FALSE for the exact
    # signal class #10 targets, and the part that DOES work is already shipped as #6.
    #   1) NOT GREEDY-TRAPPABLE (the binding wall). The backlog fixture y=sign(x1*x2)*x3>0 is, mathematically, a
    #      PURE 3-way sign XOR (verified 100% agreement with sign(x0)*sign(x1)*sign(x2)>0), NOT a "detectable
    #      2-way (x1,x2) + conditional x3" as the backlog text claims. For a pure k-way interaction ALL its
    #      (k-1)-way sub-tuples have joint MI AT THE NOISE FLOOR: the needle pair ranks ~39th of C(40,2) (jMI
    #      0.0002-0.0107 across nbins 2..8, indistinguishable from the noise-pair max), so it NEVER enters the
    #      top-N order-2 frontier -> the needle triple is NEVER grown (frontier in {6,8,12} all miss it). This is
    #      the same conditional-MI-is-not-greedy-trappable wall that killed beam-search for MRMR -- Apriori
    #      pruning ("grow only from surviving (k-1)-tuples") structurally CANNOT reach a pure higher-order
    #      synergy whose every lower-order projection is null.
    #   2) The GROWTH KERNEL itself is sound but moot. Given a base pair holding 2 of the 3 needle legs, the
    #      CMI uplift ranks the true 3rd operand RANK-0 every time (6/6, uplift 0.21-0.61 >> noise) -- but that
    #      base pair is exactly what the frontier never contains (point 1).
    #   3) NOISE ADMISSION on the one fixture where the lattice CAN grow the needle (a marginal-signal base x0
    #      plus a conditional-only x2, y=(x0>0)^((x1>0)&(x2>0))): the lattice grows (0,1,2) as the #1 triple
    #      (jMI 0.4495) BUT the order-3 maxT floor (0.0287) admits 31/31 grown noise triples, ALL sharing the
    #      marginal-signal leg x0 (a (0,noise,noise) triple inherits x0's MI and clears the floor). This is not
    #      a lattice-specific defect (#6's seeded triples admit 8/8 on the same fixture) -- it is the downstream
    #      per-triplet uplift gate's job, NOT the maxT floor's, so the lattice adds NO admission advantage.
    #   4) FULLY REDUNDANT with the shipped #6 GBM seeder. On the pure-3-way-XOR fixture the GBM
    #      split-co-occurrence proposer recovers the SAME needle {x1,x2,x3} as its RANK-0 triple (z=5.46),
    #      order-3-floored -- because path co-occurrence conditions a zero-marginal operand on its co-splitter
    #      WITHOUT requiring a detectable (k-1)-tuple, so it is immune to the anti-monotone wall that blocks #10.
    #      #10's whole purpose (reach a 3-way the univariate seeder misses) is already delivered by #6.
    # The lattice's ONLY edge is cost (19.5 ms growth vs 1641 ms for the GBM fit+self-gate at n=3000/p=40) -- but
    # that merely buys a proposer that misses its own target signal class. Do NOT re-implement as the backlog
    # specifies. A non-Apriori conditional-growth scheme (e.g. seed triples from the GBM's co-occurrence pool,
    # THEN refine the 3rd operand by CMI uplift) could compose with #6, but the recovery is already complete via
    # #6 alone, so there is no measured gap to close.

    # ENGINEERED-OPERAND FEED-FORWARD CAP (2026-06-08). At FE step k>1 the operand
    # pool now also carries the engineered columns selected by the prior step(s)
    # (their cols-space indices are promoted into ``selected_vars`` at this
    # function's bottom and re-confirmed by the next screening pass). Feeding them
    # back lets the pair search build COMPOSITES of two engineered features -- e.g.
    # the additive ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the
    # entire deterministic signal. But engineered columns accumulate across steps,
    # so an uncapped feed-forward makes the O(k^2) pair count blow up. Keep only the
    # top-K engineered operands BY THEIR MARGINAL SCREENING MI (``cached_MIs[(idx,)]``,
    # populated by screen_predictors for every selected var); the rest still reach
    # ``support_`` -- they just don't seed further composites. ``fe_max_engineered_operands``:
    # 0 -> raw-only pool (legacy, no composites), <0 -> no cap, >0 -> top-K cap.
    _raw_name_set = set(getattr(self, "feature_names_in_", []) or [])
    _eng_cap = int(getattr(self, "fe_max_engineered_operands", 8))
    _engineered_in_pool = [v for v in numeric_vars_to_consider if cols[v] not in _raw_name_set]
    if _engineered_in_pool and _eng_cap >= 0:
        if _eng_cap == 0:
            _keep_eng: set = set()
        else:
            # Rank by marginal MI; missing single-var MI sorts last (kept only if it fits the cap).
            _ranked = sorted(_engineered_in_pool, key=lambda v: cached_MIs.get((v,), 0.0), reverse=True)
            _keep_eng = set(_ranked[:_eng_cap])
        _drop_eng = set(_engineered_in_pool) - _keep_eng
        if _drop_eng:
            numeric_vars_to_consider = set(numeric_vars_to_consider) - _drop_eng
            if verbose:
                logger.info(
                    "MRMR FE feed-forward: kept top %d engineered operand(s) by marginal MI for composite pairing, "
                    "dropped %d below fe_max_engineered_operands=%d to bound the pair count (they remain selected).",
                    len(_keep_eng), len(_drop_eng), _eng_cap,
                )

    # `combinations(...)` is consumed lazily by tqdmu (small path) or by
    # `_lazy_chunks` (large path). Pair count is closed-form, avoiding
    # `list(combinations(...))` materialisation (O(k^2) tuples, ~300 MB at
    # k=5000) before chunking even starts.
    _k = len(numeric_vars_to_consider)
    n_pairs = (_k * (_k - 1)) // 2

    if verbose:
        logger.info("Feature Engineering: Computing MIs of %d most prospective feature pairs...", n_pairs)

    # ---------------------------------------------------------------------------------------------------------------
    # Layer 3 pre-batch: compute pair MIs for every (a, b) in numeric_vars_to_consider via dispatch_batch_pair_mi
    # (CUDA / CPU njit prange by size). Pre-fills cached_MIs[pair] so the per-pair compute_pairs_mis loop below skips
    # the permutation-test branch entirely (since "pair in cached_MIs" short-circuits at feature_engineering.py:394).
    #
    # Semantic change vs the legacy path: pairs no longer go through the permutation-test confidence filter
    # (min_nonzero_confidence). The raw original_mi is used as the FE-pair signal. Bench (commit 57f772c) shows
    # 10-30x speedup over the per-pair joblib loop; downstream MRMR FE pair selection is regression-validated by the
    # MRMR test suite. Disable by setting MLFRAME_MRMR_BATCH_PAIR_MI=0 (the env-var is the emergency rollback knob).
    #
    # Guards:
    #   * _k > _MRMR_BATCH_PRECOMPUTE_MAX_K: the dispatcher would have to materialise O(k^2) pair tuples; for very
    #     wide FE pools we keep the legacy lazy combinations + joblib chunking instead.
    #   * n_pairs < _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS: pair count too small to amortise the dispatcher overhead.
    #   * Any backend failure (CUDA driver hiccup, dtype mismatch): logged WARN, fall through to legacy path.
    # Accept the common truthy/falsy spellings rather than require the operator
    # to remember the exact literals we sliced earlier. Empty / missing env
    # var defaults to ENABLED (the new behaviour).
    _BATCH_PRECOMPUTE_ENABLED = os.environ.get(
        "MLFRAME_MRMR_BATCH_PAIR_MI", "1",
    ).strip().lower() not in ("0", "false", "no", "off", "")
    _batch_prefill_count = 0
    if (
        _BATCH_PRECOMPUTE_ENABLED
        and _k <= _MRMR_BATCH_PRECOMPUTE_MAX_K
        and n_pairs >= _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS
    ):
        try:
            from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi

            _pairs_list = list(combinations(numeric_vars_to_consider, 2))
            _pair_a_arr = np.fromiter((p[0] for p in _pairs_list), dtype=np.int64, count=len(_pairs_list))
            _pair_b_arr = np.fromiter((p[1] for p in _pairs_list), dtype=np.int64, count=len(_pairs_list))
            _pair_mi_batch, _backend_used = dispatch_batch_pair_mi(
                factors_data=data,
                pair_a=_pair_a_arr,
                pair_b=_pair_b_arr,
                nbins=nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
            )
            # Populate cached_MIs to short-circuit compute_pairs_mis's per-pair mi_direct call.
            # Skip pairs already in cached_confident_MIs (those had a confident permutation outcome).
            for _i, _p in enumerate(_pairs_list):
                if _p not in cached_confident_MIs and _p not in cached_MIs:
                    cached_MIs[_p] = float(_pair_mi_batch[_i])
                    _batch_prefill_count += 1
            if verbose:
                logger.info(
                    "MRMR FE: batch-prefilled %d/%d pair MIs via %s backend (permutation test skipped for these pairs)",
                    _batch_prefill_count, len(_pairs_list), _backend_used,
                )
        except Exception as _exc:
            if verbose:
                logger.warning(
                    "MRMR FE: dispatch_batch_pair_mi failed (%s: %s); falling back to legacy per-pair path "
                    "[n_pairs=%d, n_rows=%d, n_classes_y=%d]",
                    type(_exc).__name__, _exc,
                    n_pairs, int(data.shape[0]) if hasattr(data, "shape") else -1,
                    int(freqs_y.shape[0]) if hasattr(freqs_y, "shape") else -1,
                )

    # Parallelise whenever (a) more than one worker is configured and
    # (b) we have at least n_jobs pairs to spread; per-pair MI compute is
    # ~35 s with default fe_npermutations on a wide frame, so parallel
    # overhead is amortised even at very small _k. Previously this took
    # the single-thread branch up to _k=50 (1225 pairs), serialising what
    # should be a 4-minute job into ~1 h on a 16-core box.
    if n_jobs <= 1 or n_pairs < max(2, n_jobs):
        compute_pairs_mis(
            all_pairs=tqdmu(
                combinations(numeric_vars_to_consider, 2),
                total=n_pairs,
                desc="getting pairs MIs",
                leave=False,
                mininterval=5,
                disable=not verbose,
            ),
            data=data,
            target_indices=target_indices,
            nbins=nbins,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            fe_npermutations=fe_npermutations,
            cached_confident_MIs=cached_confident_MIs,
            cached_MIs=cached_MIs,
            fe_min_pair_mi=fe_min_pair_mi,
            fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        )
    else:
        chunk_size = max(1, n_pairs // (n_jobs * prefetch_factor))
        dicts = parallel_run(
            [
                delayed(compute_pairs_mis)(
                    all_pairs=chunk,
                    data=data,
                    target_indices=target_indices,
                    nbins=nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                    fe_npermutations=fe_npermutations,
                    cached_confident_MIs=cached_confident_MIs,
                    cached_MIs=cached_MIs,
                    fe_min_pair_mi=fe_min_pair_mi,
                    fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                )
                for chunk in _lazy_chunks(combinations(numeric_vars_to_consider, 2), chunk_size)
            ],
            n_jobs=n_jobs,
            **parallel_kwargs,
        )
        for next_dict in dicts:
            cached_MIs.update(next_dict)

    # ---------------------------------------------------------------------------------------------------------------
    # ORDER-2 Westfall-Young maxT permutation-null floor on the PROSPECTIVE-PAIR
    # JOINT MI (2026-06-03). The gating loop below ranks O(p^2) candidate pairs by
    # JOINT MI(x_i, x_j; y); at high p the MAX joint MI over PURE-NOISE pairs is a
    # positive order statistic that grows with the pool size -- the same best-of-p
    # selection bias the order-1 screening floor rejects, now at order 2. The
    # per-pair prevalence gates (``fe_min_pair_mi_prevalence`` /
    # ``fe_synergy_min_prevalence``) are PER-PAIR; they do NOT account for the
    # max-over-pool selection, so a wide noise matrix still surfaces
    # "synergistic-looking" noise pairs. Compute the floor ONCE here over the WHOLE
    # candidate pool: shuffle the discretised target K times, take the per-shuffle
    # MAX joint MI via the SAME batched plug-in estimator the screen scores
    # ``pair_mi`` with, floor at the q-th quantile. Applied IN ADDITION to the
    # prevalence gates in BOTH the zero-individual-MI (XOR) branch and the uplift
    # branch below. SELF-GATING: below ``fe_pair_maxt_min_pairs`` candidate pairs
    # the floor is 0.0 (no-op => byte-identical narrow pools), mirroring
    # ``screen_fdr_min_features``. ``fe_pair_maxt_null_permutations=0`` disables.
    _pair_maxt_floor, _pair_mm_bias = compute_pair_maxt_floor(
        self,
        numeric_vars_to_consider=numeric_vars_to_consider,
        n_pairs=n_pairs,
        data=data,
        nbins=nbins,
        classes_y=classes_y,
        freqs_y=freqs_y,
        verbose=verbose,
    )

    # ---------------------------------------------------------------------------------------------------------------
    # For every pair of factors (A,B), select ones having MI((A,B),Y)>MI(A,Y)+MI(B,Y). Such ones must posess more special connection!
    # ---------------------------------------------------------------------------------------------------------------

    vars_usage_counter = defaultdict(int)
    prospective_pairs = {}
    for raw_vars_pair, pair_mi in sort_dict_by_value(cached_MIs).items():
        if len(raw_vars_pair) == 2:
            if raw_vars_pair in checked_pairs:
                continue
            if raw_vars_pair[0] in numeric_vars_to_consider and raw_vars_pair[1] in numeric_vars_to_consider:
                ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
                # Guard against ZeroDivisionError: when both individual features have zero MI with target
                # (canonical 3-way XOR case: MI(x_i, y) = 0 for all i but the joint signal exists), any positive pair_mi
                # qualifies as infinite uplift -- keep the pair.
                # MM-DEBIAS (2026-06-09, IRON RULE): the maxT floor was computed on the
                # Miller-Madow-debiased joint-MI scale (per-pair bias subtracted inside the
                # null kernel), so subtract the SAME per-pair joint-MI bias from the observed
                # ``pair_mi`` before the ``>= floor`` comparison -- consistent debias on both
                # sides keeps the outer best-of-pool guard at full strength even though the
                # prevalence ratio bar downstream was lowered. No-op (0.0) when MM is off or
                # the pool is below the floor's min-pairs self-gate.
                _mm_pair_bias = _pair_mm_bias.get(tuple(sorted(raw_vars_pair)), 0.0)
                _pair_mi_floor_cmp = pair_mi - _mm_pair_bias
                if ind_elems_mi_sum <= 0:
                    # ORDER-2 maxT floor (computed once above): a zero-individual-MI
                    # pair enters via the canonical XOR branch on ANY positive joint
                    # MI, so on a wide noise matrix a noise pair whose joint MI is
                    # merely the best chance hit slips through. Require the joint MI
                    # to clear the pool's permutation-null max before keeping it;
                    # genuine pure-synergy (XOR / sign product) joint MI is FAR above
                    # the chance ceiling, so it survives. No-op when floor==0.0.
                    if pair_mi > 0 and _pair_mi_floor_cmp >= _pair_maxt_floor:
                        uplift = float("inf")
                        if verbose >= 2:
                            logger.info(
                                "Factors pair %s has zero individual MI but pair_mi=%.4f -- canonical hidden-pair case (e.g. XOR), keeping for FE",
                                raw_vars_pair, pair_mi,
                            )
                        prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                        for var in raw_vars_pair:
                            vars_usage_counter[var] += 1
                    continue
                # SYNERGY pairs (>=1 bootstrap-added operand) must clear a STRICTER
                # uplift bar than selected-selected pairs: their operands are
                # unselected (usually noise), and adding one as a 2nd joint
                # dimension inflates the finite-sample joint MI by ~5-15% bias,
                # which would clear the lenient 1.05 gate and inject a spurious
                # feature (observed regressing F-MONO). Genuine synergy has joint MI
                # far above the marginal sum, so the stricter bar keeps it.
                _is_synergy_pair = bool(_synergy_added_idx) and (
                    raw_vars_pair[0] in _synergy_added_idx or raw_vars_pair[1] in _synergy_added_idx
                )
                _prev_thresh = fe_min_pair_mi_prevalence
                if _is_synergy_pair:
                    _prev_thresh = max(fe_min_pair_mi_prevalence, float(getattr(self, "fe_synergy_min_prevalence", 1.15)))
                # ORDER-2 maxT floor (computed once above) applied IN ADDITION to the
                # per-pair prevalence gate: the pair's JOINT MI must clear the pool's
                # permutation-null max as well, rejecting best-of-p chance-max noise
                # pairs the per-pair prevalence bar misses. No-op when floor==0.0.
                # ``_pair_mi_floor_cmp`` is the MM-debiased joint MI (IRON RULE, see above).
                if pair_mi > ind_elems_mi_sum * _prev_thresh and _pair_mi_floor_cmp >= _pair_maxt_floor:
                    uplift = pair_mi / ind_elems_mi_sum
                    if verbose >= 2:
                        logger.info(
                            "Factors pair %s will be considered for Feature Engineering, %.4f->%.4f, rat=%.2f",
                            raw_vars_pair, ind_elems_mi_sum, pair_mi, uplift,
                        )
                    prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                    for var in raw_vars_pair:
                        vars_usage_counter[var] += 1

    # SIGNED INTERACTION-INFORMATION ROUTING (2026-06-09, backlog idea #8). Among the
    # pairs that PASSED the ratio gate + order-2 maxT floor above, separate genuine
    # synergy (II > null floor) from the ADDITIVE cross-mix (II <= floor: a feeds one
    # independent term of y, b a DIFFERENT one -- the weak-F2 ``add(invqubed(a),
    # invsqrt(c))`` surrogate that mixes an (a,b)-term operand with a (c,d)-term operand)
    # and from REDUNDANCY (II < 0). ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)`` is the
    # signed difference of terms the gate ALREADY computed (cached marginals + pair_mi),
    # Miller-Madow corrected per term so the joint's nbins_a*nbins_b-bin finite-sample
    # bias does not masquerade as synergy. Positive II is floored by a permutation null on
    # the per-shuffle MAX II (the additive-completion + finite-sample chance ceiling).
    # DEMOTE additive speculative (synergy-added) cross-mix pairs out of the FE search so
    # no cross-mix surrogate is built; this is a ROUTING change, the maxT floor + ratio
    # gate stay as the detection guards. Runs BEFORE the synergy budget cap so demoted
    # pairs do not consume budget. SELF-GATING => byte-stable on narrow pools / when
    # disabled. See ``_interaction_information.py``.
    #
    # bench-rejected (2026-06-09) as a DEFAULT (now ``fe_ii_routing_enable=False``): on the
    # user's WEAK F2 the cross-mix pair (b,c) has HIGHER interaction information than the
    # genuine (a,b) a**2/b pair on every cross-mix seed (II +0.0132/+0.0135/+0.0139 vs
    # +0.0114/+0.0120/+0.0132 at n=20000, seeds 0/6/8), and the y-shuffle null floor sits at
    # ~0.0007 -- so no II threshold demotes the cross-mix without also dropping the genuine
    # pair. F2 10-seed result NEUTRAL (cross_mix 3/10 -> 3/10). The router is correct on
    # STRONG synergy (synthetic synergy II +0.55 vs additive +0.03 below floor) and stays an
    # opt-in; see the default-off rationale in ``mrmr.py`` (fe_ii_routing_enable). The call
    # below is a structural no-op while disabled (returns the un-routed pairs unchanged).
    prospective_pairs = apply_interaction_information_routing(
        self,
        prospective_pairs=prospective_pairs,
        cached_MIs=cached_MIs,
        nbins=nbins,
        freqs_y=freqs_y,
        classes_y=classes_y,
        data=data,
        synergy_added_idx=_synergy_added_idx,
        verbose=verbose,
    )

    # SYNERGY-PAIR BUDGET (2026-06-02): cap the synergy pairs (>=1 operand is a
    # bootstrap-added unselected column) at ``fe_synergy_max_pairs`` top-joint-MI
    # pairs before the expensive per-pair search, so a noise-heavy frame cannot
    # flood ``check_prospective_fe_pairs``. Selected-selected pairs are kept in
    # full. ``key`` is ``(raw_vars_pair, pair_mi)``; rank synergy pairs by pair_mi.
    if _synergy_added_idx:
        _synergy_budget = int(getattr(self, "fe_synergy_max_pairs", 16) or 0)
        _synergy_keys = [k for k in prospective_pairs if (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)]
        if _synergy_budget >= 0 and len(_synergy_keys) > _synergy_budget:
            _keep_synergy = set(sorted(_synergy_keys, key=lambda k: k[1], reverse=True)[:_synergy_budget])
            _dropped = 0
            for k in _synergy_keys:
                if k not in _keep_synergy:
                    del prospective_pairs[k]
                    _dropped += 1
            if verbose and _dropped:
                logger.info(
                    "MRMR FE synergy bootstrap: kept top %d synergy pairs by joint MI, "
                    "dropped %d below budget (fe_synergy_max_pairs) to bound FE search cost.",
                    min(_synergy_budget, len(_synergy_keys)), _dropped,
                )

    # Now need to sort prospective_pairs by the uplift, to check most promising pairs within the time budget.
    # Also need to sort them by their members usage frequency+members ids sum. this way, their splitting will benefit more from caching.
    prospective_pairs = sort_dict_by_value(prospective_pairs, reverse=True)

    # SUCCESSIVE-HALVING / RUNG-SCHEDULE FE-search budget (2026-06-10, backlog #16).
    # ON by default. Before the EXPENSIVE per-pair operator search below
    # (``check_prospective_fe_pairs`` -- all unary x binary transforms / CMA-ES / full
    # discretize / prewarp, ~4-50s per pair), run a CHEAP rung-0 SCREEN: rank the
    # gate-surviving prospective pairs by their JOINT MI ``pair_mi`` (key[1] -- a
    # monotone-ish proxy of the operator-search outcome ALREADY computed by the pair-MI
    # gate, so the screen is FREE) and keep only the top fraction (UNION a relative-MI
    # floor so a moderate-MI genuine winner is never cut). The expensive search then runs
    # on the survivors only. GATES UNCHANGED -- this changes WHERE the compute goes, not
    # admission (a pair the rung-0 screen drops simply never gets the operator search,
    # exactly as the synergy budget already caps synergy pairs; this generalises that to
    # the whole pool). Measured 1.7-2.2x at keep_frac=0.5 with NO genuine signal pair
    # dropped (n=5000/p=40 canonical fixture + noise, 5 seeds). Self-gates to a no-op
    # below ``fe_rung_min_pairs`` pairs / all-zero pair_mi (byte-identical flat sweep).
    if bool(getattr(self, "fe_rung_schedule_enable", True)) and len(prospective_pairs) >= int(getattr(self, "fe_rung_min_pairs", 6)):
        from .._fe_rung_schedule import apply_rung_schedule
        _rung_n_rows = int(data.shape[0]) if hasattr(data, "shape") else 0
        prospective_pairs, _rung_info = apply_rung_schedule(
            prospective_pairs,
            n_rows=_rung_n_rows,
            keep_frac=getattr(self, "fe_rung_keep_frac", None),
            rel_floor=float(getattr(self, "fe_rung_rel_floor", 0.40)),
            min_pairs=int(getattr(self, "fe_rung_min_pairs", 6)),
            verbose=verbose,
        )

    # cols-space indices of polynom-pair engineered columns appended by the
    # ``run_polynom_pair_fe`` block below; promoted into ``selected_vars``
    # alongside the unary/binary indices so a polynom feature that cleared the
    # FE gates actually reaches ``support_`` (see promotion at the bottom).
    _polynom_engineered_indices: list[int] = []
    if fe_smart_polynom_iters:
        # Orthogonal-polynomial pair FE: Chebyshev default basis (empirically robust); tight coef range [-2, 2],
        # fixed degree per study, L2 regularisation, identity-baseline filter. Override basis via
        # ``self.fe_polynomial_basis``. See feature_selection.filters.hermite_fe and bench_polynomial_bases.
        #
        # 2026-05-18: extracted from inline ~200 LOC block into
        # ``polynom_pair_fe.run_polynom_pair_fe`` (joblib-threaded pair
        # eval + serial inject). ``self._hermite_features_`` is fed
        # through as a target list so the helper stays method-free.
        from ..polynom_pair_fe import run_polynom_pair_fe
        if not hasattr(self, "_hermite_features_"):
            self._hermite_features_ = []
        # SYNERGY pairs feed the STANDARD unary/binary+prewarp search below, but NOT
        # this orthogonal-poly optimiser. The synergy bootstrap adds SPECULATIVE
        # pairs (>=1 unselected, often-noise operand); the cma_batch / optuna
        # optimiser here is powerful enough to fit a high-MI atan2/poly cell to
        # PURE NOISE on such a pair (the saturating-penalty relaxation makes that
        # reachable), fabricating a spurious feature. Every measured synergy WIN
        # (sign_prod, gauss_prod, ratio_abs, ...) is recovered by the standard
        # mul/div search, NOT a _polynom_ cell, so withholding synergy pairs from
        # the optimiser loses no recovery while keeping the pure-noise control clean.
        _prospective_for_polynom = prospective_pairs
        # Withhold SPECULATIVE synergy pairs from the powerful poly optimiser
        # (it can fit a high-MI cell to pure noise on a noise-operand pair) --
        # but ONLY when there was a genuine selected pool to augment. When
        # screening returned 0 features (interaction-only signal), EVERY operand
        # is "synergy-added", so this exclusion would withhold every pair and the
        # polynom search would never fire (default_filtering.py:165). In that
        # case keep the pairs: they ARE the signal, and the synergy max-pairs cap
        # + the downstream pair-MI / engineered-MI / uplift gates already bound
        # the pure-noise risk.
        if _synergy_added_idx and not _screening_returned_empty:
            _filtered_for_polynom = {
                k: v for k, v in prospective_pairs.items()
                if not (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)
            }
            # 2026-06-03 (wave-9 follow-up, default_filtering.py:165): apply the
            # speculative-synergy exclusion ONLY if it leaves a non-empty pool.
            # When the selected pool is too small to form any NON-synergy pair
            # (screening kept 0-1 features on an interaction-only target, so
            # every surviving pair has a synergy-added operand), excluding them
            # would withhold EVERY pair and silently disable the polynom search
            # -- yet those pairs ARE the signal. Keep them in that case; the
            # synergy max-pairs cap + the downstream pair-MI / engineered-MI /
            # uplift gates already bound the pure-noise risk.
            if _filtered_for_polynom:
                _prospective_for_polynom = _filtered_for_polynom
        # None / 0 / negative all map to "no subsample" (use full data).
        _subsample_raw = getattr(self, "fe_smart_polynom_subsample_n", 0)
        _subsample_n = int(_subsample_raw) if _subsample_raw and _subsample_raw > 0 else 0
        # Capture cols width before the polynom block so we can promote the
        # polynom-injected engineered column indices into ``selected_vars``
        # below (same "ROOT CAUSE 5" promotion the unary/binary block does for
        # its own appended cols). Without this, a polynom-pair feature that
        # cleared every polynom-FE gate (pair-MI prevalence + engineered-MI
        # prevalence + uplift) was appended to ``data``/``cols`` and tracked in
        # ``_hermite_features_`` but never reached ``support_`` under the default
        # single-step path, because only the unary/binary indices were promoted.
        _n_cols_before_polynom = len(cols)
        data, nbins, cols, X = run_polynom_pair_fe(
            X=X, is_polars_input=_is_polars_input,
            prospective_pairs=_prospective_for_polynom,
            classes_y=classes_y,
            cols=cols, nbins=nbins, data=data,
            engineered_features=engineered_features,
            engineered_recipes=engineered_recipes,
            hermite_features_list=self._hermite_features_,
            feature_names_in=self.feature_names_in_,
            fe_smart_polynom_iters=fe_smart_polynom_iters,
            fe_smart_polynom_optimization_steps=fe_smart_polynom_optimization_steps,
            fe_min_polynom_degree=fe_min_polynom_degree,
            fe_max_polynom_degree=fe_max_polynom_degree,
            fe_min_polynom_coeff=fe_min_polynom_coeff,
            fe_max_polynom_coeff=fe_max_polynom_coeff,
            fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
            fe_hermite_l2_penalty=getattr(self, "fe_hermite_l2_penalty", 0.05),
            fe_polynomial_basis=getattr(self, "fe_polynomial_basis", "chebyshev"),
            fe_mi_estimator=getattr(self, "fe_mi_estimator", "plugin"),
            # 2026-05-22: cma_batch is the new default (20.58x faster than
            # optuna, 1.09x faster than per-solution cma, within_1%=1.00
            # on a 12-pair benchmark). See profiling/bench_polynom_optimizers.py.
            fe_optimizer=getattr(self, "fe_optimizer", "cma_batch"),
            fe_warm_start=getattr(self, "fe_warm_start", True),
            fe_multi_fidelity=getattr(self, "fe_multi_fidelity", True),
            quantization_nbins=self.quantization_nbins,
            quantization_method=self.quantization_method,
            quantization_dtype=self.quantization_dtype,
            n_jobs=int(n_jobs) if n_jobs and n_jobs > 0 else 1,
            verbose=int(verbose),
            subsample_n=_subsample_n,
            # Cheap-first dispatch: skip the expensive CMA/Optuna search for pairs
            # whose trivial baseline already saturates the joint-MI ceiling. getattr
            # default keeps the knob optional (no ctor flag required); 1.0 disables.
            poly_cheap_skip_ratio=float(getattr(self, "fe_poly_cheap_skip_ratio", 0.97)),
            # Linear-usability guard on the skip: only skip when the trivial feature
            # is ALSO linearly useful (|corr| to y >= this), so an MI-saturated-but-
            # non-linear trivial (atan2, etc.) still falls through to the optimiser.
            poly_cheap_skip_min_corr=float(getattr(self, "fe_poly_cheap_skip_min_corr", 0.90)),
        )
        # Columns appended by the polynom block (its own gates already accepted
        # them). Promote into selected_vars below so they reach support_.
        _polynom_engineered_indices = list(range(_n_cols_before_polynom, len(cols)))

    # The standard check_prospective_fe_pairs path used to live in
    # ``else:`` of the Hermite block, which meant enabling
    # ``fe_smart_polynom_iters > 0`` silently DISABLED all standard
    # unary/binary FE (cbrt, sqrt, log, hypot, atan2, ...). De-dented the
    # block so the standard pipeline always runs after the Hermite block;
    # users get the unary/binary FE they asked for via
    # ``fe_unary_preset='medium'`` regardless of whether Hermite ran.
    if True:
        original_cols = {i: self.feature_names_in_.index(col) for i, col in enumerate(cols) if col in self.feature_names_in_}
        if verbose >= 1:
            logger.info("Checking %d most prospective_pairs for feature engineering...", len(prospective_pairs))

        # PER-OPERAND PRE-WARP (2026-06-02): read the opt-in flag + knobs off the
        # MRMR instance (getattr keeps _run_fe_step's signature stable, mirroring
        # ``fe_check_pairs_subsample_n``). When enabled, the discretised target
        # (``classes_y`` -- the SAME codes the MI sweep scores against) is handed
        # to ``check_prospective_fe_pairs`` so it can fit a learned 1-D pre-warp
        # per operand; ``_prewarp_specs`` collects the fitted coeffs (by cols-space
        # var index) for leak-safe recipe construction. Default OFF.
        _prewarp_enable = bool(getattr(self, "fe_pair_prewarp_enable", False))
        _prewarp_basis = str(getattr(self, "fe_pair_prewarp_basis", "chebyshev"))
        _prewarp_max_degree = int(getattr(self, "fe_pair_prewarp_max_degree", 4))
        _prewarp_uplift = float(getattr(self, "fe_pair_prewarp_uplift_threshold", 1.20))
        _prewarp_min_val_corr = float(getattr(self, "fe_pair_prewarp_min_val_corr", 0.08))
        # PREWARP-SPEC PERSISTENCE (2026-06-08 fix). ``_mrmr_fe_step`` is re-entered
        # once per FE-bearing MRMR iteration, and each call's ``check_prospective_fe_pairs``
        # fits prewarp specs ONLY for the operands of THIS iteration's prospective pairs
        # (a var whose pair isn't prospective this round is not re-fit). But the recipe
        # build below runs per iteration over ``this_pair_features`` and can construct a
        # recipe whose operand used the ``prewarp`` pseudo-unary in an EARLIER iteration --
        # if ``_prewarp_specs`` were a fresh per-call dict, that operand's coeffs would be
        # absent and the recipe would be built with ``prewarp_a/b=None``, producing a
        # recipe that names ``prewarp`` but lacks ``prewarp_*_coef`` in ``extra`` -> a
        # KeyError at transform()-time replay (observed: ``sub(prewarp(informative_3),
        # prewarp(noise_5))`` with informative_3's spec missing). Back the local dict with
        # a SELF-LEVEL accumulator (mirrors ``self._engineered_continuous_``) so every
        # spec fit in any prior iteration stays available for recipe construction. Seeded
        # from the accumulator and written back after each call; specs are keyed by
        # cols-space var index, which is stable across iterations (no cat reorder mid-fit).
        _prewarp_specs: dict = getattr(self, "_prewarp_specs_accum_", None)
        if _prewarp_specs is None:
            _prewarp_specs = {}
            self._prewarp_specs_accum_ = _prewarp_specs

        # PER-OPERAND MEDIAN GATE (2026-06-04): opt-in flag off the MRMR instance
        # (getattr keeps the signature stable, mirroring the prewarp wiring). When
        # enabled, ``check_prospective_fe_pairs`` fits one TRAIN median per operand
        # and exposes a ``gate_med`` pseudo-unary; ``_gate_med_specs`` collects the
        # fitted medians (by cols-space var index) for leak-safe recipe
        # construction. Default OFF -> byte-identical legacy path. Same cross-iteration
        # persistence as the prewarp specs above (a gate_med operand selected in an
        # earlier iteration must keep its median available for recipe replay).
        _gate_med_enable = bool(getattr(self, "fe_gate_med_enable", False))
        _gate_med_specs: dict = getattr(self, "_gate_med_specs_accum_", None)
        if _gate_med_specs is None:
            _gate_med_specs = {}
            self._gate_med_specs_accum_ = _gate_med_specs

        # SERIAL-vs-JOBLIB DISPATCH (2026-06-08 narrow+tall fix). The ``else`` branch
        # spreads the prospective PAIRS across ``n_jobs`` joblib ``backend="threading"``
        # workers, each running the SERIAL (no-prange) FE kernels (a numba prange would
        # nest inside the threading layer and deadlock -- see ``_fe_use_parallel_kernels``).
        # That is the right lever ONLY when there are enough pairs to actually fill the
        # worker pool. In the NARROW+TALL regime (few features -> 1-2 prospective pairs,
        # but n large) the joblib path span only 1-2 jobs over 16 threads (14 idle) AND
        # each job is pinned to a single-core serial kernel -> ~1-2 of 16 cores busy, the
        # measured 56s-per-pair / fully-idle-HW pathology. The serial-main-thread path
        # instead runs the WHOLE candidate sweep on the main thread where a numba prange
        # is safe, so the materialise / searchsorted / MI kernels dispatch to their
        # BYTE-IDENTICAL ``parallel=True`` column-prange twins and spread each pair's K
        # candidates across ALL cores. So route to it whenever there are FEWER pairs than
        # workers (joblib-over-pairs cannot saturate the pool): selection is unchanged
        # (same ``check_prospective_fe_pairs`` over the same pairs; only kernel
        # parallelism + chunk-merge differ, both byte-identical). The pair-count crossover
        # subsumes the old fixed ``len(X) < 50000`` row gate -- a tall narrow frame now
        # takes the all-cores path it always should have.
        _fe_serial_min_pairs_per_worker = max(2, int(n_jobs) if n_jobs and n_jobs > 0 else 1)
        if len(prospective_pairs) < _fe_serial_min_pairs_per_worker:
            prospective_additions = check_prospective_fe_pairs(
                prospective_pairs,
                X,
                unary_transformations,
                binary_transformations,
                classes_y,
                classes_y_safe,
                freqs_y,
                num_fs_steps,
                cols,
                original_cols,
                fe_max_steps,
                fe_npermutations,
                fe_max_pair_features,
                fe_print_best_mis_only,
                fe_min_nonzero_confidence,
                fe_min_engineered_mi_prevalence,
                fe_good_to_best_feature_mi_threshold,
                fe_max_external_validation_factors,
                numeric_vars_to_consider,
                self.quantization_nbins,
                self.quantization_method,
                self.quantization_dtype,
                times_spent,
                verbose,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                prewarp_enable=_prewarp_enable,
                prewarp_y=classes_y if _prewarp_enable else None,
                prewarp_basis=_prewarp_basis,
                prewarp_max_degree=_prewarp_max_degree,
                prewarp_uplift_threshold=_prewarp_uplift,
                prewarp_min_val_corr=_prewarp_min_val_corr,
                prewarp_specs_out=_prewarp_specs,
                fe_gate_med_enable=_gate_med_enable,
                gate_med_specs_out=_gate_med_specs,
                # OPT-A (2026-06-07): THIS is the serial-main-thread branch -- the whole FE
                # search runs here with NO joblib threads (the ``else`` below is the
                # ``len(X) >= 50000`` joblib ``backend="threading"`` path). On this branch a
                # numba prange does not nest inside Python threads, so the FE materialise /
                # searchsorted kernels may dispatch to their byte-identical ``parallel=True``
                # column-prange twins (gated further by the per-host crossover). The joblib
                # path below leaves ``serial_main_thread`` at its default False -> serial kernels.
                serial_main_thread=True,
                # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): resolve engineered operands by
                # name so (eng_i, eng_j) composites materialise. Off when the cap is 0 (raw-only
                # pool); the pool already excludes them in that case. ``engineered_operand_values``
                # supplies the CONTINUOUS engineered values so the composite is built on them
                # rather than the lossy bin codes.
                allow_engineered_operands=(_eng_cap != 0),
                engineered_operand_values=getattr(self, "_engineered_continuous_", None),
                # MM-DEBIAS (2026-06-09, backlog #1 + #4): debias the joint-prevalence
                # ratio gate (see check_prospective_fe_pairs). Co-updated with the maxT
                # floor below (IRON RULE). Default-on; ``fe_mm_debias_prevalence=False``
                # byte-reproduces pre-2026-06-09 fits.
                fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
            )
        else:

            prospective_additions = {}
            desired_nitems = max(1, len(prospective_pairs) // (n_jobs * prefetch_factor))

            jobs_list = []

            nitems = 0
            cur_dict = {}
            for key, value in prospective_pairs.items():
                nitems += 1
                cur_dict[key] = value
                if nitems >= desired_nitems:
                    jobs_list.append(cur_dict)
                    nitems = 0
                    cur_dict = {}
            if cur_dict:
                jobs_list.append(cur_dict)

            if verbose:
                logger.info(
                    "Using %d items per thread for checking %d prospective_pairs with gain>%.2f.",
                    desired_nitems, len(prospective_pairs), fe_min_pair_mi_prevalence,
                )

            dicts = parallel_run(
                [
                    delayed(check_prospective_fe_pairs)(
                        chunk,
                        X,
                        unary_transformations,
                        binary_transformations,
                        classes_y,
                        classes_y_safe,
                        freqs_y,
                        num_fs_steps,
                        cols,
                        original_cols,
                        fe_max_steps,
                        fe_npermutations,
                        fe_max_pair_features,
                        fe_print_best_mis_only,
                        fe_min_nonzero_confidence,
                        fe_min_engineered_mi_prevalence,
                        fe_good_to_best_feature_mi_threshold,
                        fe_max_external_validation_factors,
                        numeric_vars_to_consider,
                        self.quantization_nbins,
                        self.quantization_method,
                        self.quantization_dtype,
                        times_spent,
                        verbose,
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                        subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                        prewarp_enable=_prewarp_enable,
                        prewarp_y=classes_y if _prewarp_enable else None,
                        prewarp_basis=_prewarp_basis,
                        prewarp_max_degree=_prewarp_max_degree,
                        prewarp_uplift_threshold=_prewarp_uplift,
                        prewarp_min_val_corr=_prewarp_min_val_corr,
                        prewarp_specs_out=None,  # loky: recovered from result dict below
                        fe_gate_med_enable=_gate_med_enable,
                        gate_med_specs_out=None,  # loky: recovered from result dict below
                        # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): see the serial branch above.
                        allow_engineered_operands=(_eng_cap != 0),
                        engineered_operand_values=getattr(self, "_engineered_continuous_", None),
                        # MM-DEBIAS (2026-06-09, backlog #1 + #4): see the serial branch above.
                        fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
                        # LARGE-N PEAK-MEMORY FIX (2026-06-08): joblib ``backend="threading"``
                        # runs up to ``n_jobs`` of these CONCURRENTLY in the shared address space,
                        # each allocating its own candidate/chunk/disc/MI buffers; divide the
                        # per-call RAM budget by the worker count so N threads don't collectively
                        # blow past 0.4*available and OOM a worker.
                        concurrent_workers=int(n_jobs) if n_jobs and n_jobs > 0 else 1,
                    )
                    for chunk in jobs_list
                ],
                # max_nbytes=0,
                n_jobs=n_jobs,
                **parallel_kwargs,
            )
            # PREWARP/GATE-MED SPEC-MERGE FIX (2026-06-08). Each chunk's result dict
            # carries its OWN fitted-spec payload under the SAME reserved key
            # (``_PREWARP_SPECS_RESULT_KEY`` / ``_GATE_MED_SPECS_RESULT_KEY``). A bare
            # ``prospective_additions.update(next_dict)`` lets each chunk's reserved-key
            # entry CLOBBER the previous chunk's -- only the LAST chunk's specs survive,
            # so an operand whose ``prewarp`` warp was fit in an EARLIER chunk loses its
            # coeffs. The recipe builder below then constructs a recipe that names
            # ``prewarp`` but has no ``prewarp_*_coef`` in ``extra`` -> a KeyError at
            # transform()-time replay (observed with n_jobs=1, which still chunks the
            # pairs into multiple sequential ``parallel_run`` jobs: e.g.
            # ``sub(prewarp(informative_3),prewarp(noise_5))`` with informative_3's spec
            # dropped). Fix: MERGE each chunk's reserved spec payload into the
            # accumulators DURING the merge loop, before ``update`` overwrites the key.
            from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY
            for next_dict in dicts:
                _pw_chunk = next_dict.pop(_PREWARP_SPECS_RESULT_KEY, None)
                if _pw_chunk:
                    _prewarp_specs.update(_pw_chunk)
                _gm_chunk = next_dict.pop(_GATE_MED_SPECS_RESULT_KEY, None)
                if _gm_chunk:
                    _gate_med_specs.update(_gm_chunk)
                prospective_additions.update(next_dict)

        # Extract any reserved pre-warp / gate-med spec entry the SERIAL path may have
        # left in ``prospective_additions`` (the serial branch returns a single dict that
        # also carries the reserved key) and merge it, so the pair loop below never
        # treats it as a ``raw_vars_pair``. The joblib branch already drained the key
        # per-chunk above; this pop is then a harmless no-op.
        from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY
        _pw_from_res = prospective_additions.pop(_PREWARP_SPECS_RESULT_KEY, None)
        if _pw_from_res:
            _prewarp_specs.update(_pw_from_res)
        # Same recovery for the per-operand TRAIN medians (loky-parallel path).
        _gm_from_res = prospective_additions.pop(_GATE_MED_SPECS_RESULT_KEY, None)
        if _gm_from_res:
            _gate_med_specs.update(_gm_from_res)

        # CONDITIONAL-MI REDUNDANCY GATE (strategy S5, 2026-06-08). The PRINCIPLED,
        # constant-free replacement for the hardcoded ``fe_min_engineered_mi_prevalence``
        # joint-prevalence ratio. After the per-pair acceptance machinery has selected one
        # best engineered column per pair, run a greedy CMI-MRMR over the SURVIVING pool:
        # admit a candidate iff its CONDITIONAL MI with y GIVEN the already-admitted
        # ENGINEERED features clears (1) a conditional-permutation floor AND (2) a scale-free
        # fraction (TAU=``fe_engineered_cmi_retain_frac``, default 0.15) of the weakest
        # admitted feature's CMI. A redundant engineered column whose y-information is wholly
        # carried by the admitted features collapses to ~0 CMI and is dropped here; a genuine
        # column carrying a PRIVATE interaction term keeps a large CMI and is kept. Default
        # path (``fe_acceptance == 'conditional_mi'``); the old ratio remains available via
        # ``fe_acceptance == 'prevalence_ratio'`` (then this block is skipped and the per-pair
        # ratio gate alone decides, exactly as before). Validated 10/10 vs four failing
        # approaches across 16 (seed, formula) cells; see ``_fe_cmi_redundancy_gate``.
        _fe_acceptance = str(getattr(self, "fe_acceptance", "conditional_mi"))
        _cmi_dropped: set = set()
        if _fe_acceptance == "conditional_mi" and prospective_additions:
            from .._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate
            from ..mrmr import discretize_array  # already imported above; re-bind for clarity

            # Build the surviving-candidate pool: {engineered_col_name -> (continuous_vals,
            # marginal_mi)}. The continuous values are the pair search's ``transformed_vals``
            # (full-n float, NOT pre-binned). Marginal MI is computed cheaply from the binned
            # values via the same plug-in primitive (z=None) so the seed/relative-bar anchor
            # matches the production CMI estimator -- no separate MI kernel.
            from .._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

            # y codes: reuse the discretised target the MI sweep scored against.
            _y_codes = np.asarray(classes_y).ravel()
            _, _y_dense = np.unique(_y_codes, return_inverse=True)
            _y_dense = _y_dense.astype(np.int64)

            _cmi_cands: dict = {}
            for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                if not _tpf or _tvals is None or not _ncols:
                    continue
                for _jc, _cname in enumerate(_ncols):
                    if _tvals.shape[1] <= _jc:
                        continue
                    _vals = np.asarray(_tvals[:, _jc], dtype=np.float64)
                    _vb = _quantile_bin(_vals, nbins=int(self.quantization_nbins))
                    _marg = float(_cmi_from_binned(_vb, _y_dense, None))
                    _cmi_cands[_cname] = (_vals, _marg)

            if len(_cmi_cands) >= 2:
                _retain = float(getattr(self, "fe_engineered_cmi_retain_frac", 0.15))
                _escape = float(getattr(self, "fe_engineered_cmi_significance_escape_margin", 3.0))
                _cmi_max_cands = int(getattr(self, "fe_engineered_cmi_max_candidates", 64))
                _accepted, _diag = apply_cmi_redundancy_gate(
                    _cmi_cands, _y_dense,
                    nbins=int(self.quantization_nbins),
                    retain_frac=_retain,
                    significance_escape_margin=_escape,
                    max_candidates=_cmi_max_cands,
                    seed=int(getattr(self, "random_state", 0) or 0),
                    verbose=int(bool(verbose)),
                )
                _cmi_dropped = set(_cmi_cands) - _accepted
                if _cmi_dropped and verbose:
                    logger.info(
                        "CMI-redundancy gate: dropped %d/%d engineered survivors as redundant "
                        "given the admitted engineered support (TAU=%.3f): %s",
                        len(_cmi_dropped), len(_cmi_cands), _retain, sorted(_cmi_dropped),
                    )

        # Apply the CMI-redundancy drops to ``prospective_additions`` IN PLACE so the
        # materialise / recipe loop below never appends a redundant engineered column.
        # Each entry's parallel arrays (``this_pair_features`` set of (config, j),
        # ``transformed_vals`` columns, ``new_cols`` names, ``new_nbins``) are filtered to
        # the surviving columns by NAME. ``new_cols[i]`` is the name of the i-th
        # ``transformed_vals`` column; the matching ``(config, j)`` is the one whose
        # ``get_new_feature_name(config, cols)`` equals that name. Both downstream
        # consumers index ``transformed_vals`` by the per-column position
        # (materialise: ``for j in range(len(this_pair_features))``; recipe: the tuple's
        # stored ``j``), so the kept tuples are re-emitted as ``(config, new_position)``
        # with the new packed column position. Entries whose every column was dropped
        # are removed entirely. In the common one-best-per-pair case a pair holds a
        # single column, so this reduces to keep-entry / drop-entry.
        if _cmi_dropped:
            from ..mrmr import get_new_feature_name as _get_new_feature_name
            _filtered_additions: dict = {}
            for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                if not _tpf or _tvals is None or not _ncols:
                    _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                _keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _cmi_dropped]
                if not _keep_idx:
                    continue  # whole pair redundant -> drop the entry
                if len(_keep_idx) == len(_ncols):
                    _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                # Name -> config map (authoritative column<->config link).
                _name_to_cfg = {
                    _get_new_feature_name(_cfg, cols): _cfg for _cfg, _ in _tpf
                }
                _new_tpf = set()
                for _new_pos, _old_i in enumerate(_keep_idx):
                    _cfg = _name_to_cfg.get(_ncols[_old_i])
                    if _cfg is not None:
                        _new_tpf.add((_cfg, _new_pos))
                _new_tvals = _tvals[:, _keep_idx]
                _new_ncols = [_ncols[i] for i in _keep_idx]
                _new_nnb = (
                    [_nnb[i] for i in _keep_idx]
                    if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols)
                    else _nnb
                )
                _filtered_additions[_rp] = (_new_tpf, _new_tvals, _new_ncols, _new_nnb, _msgs)
            prospective_additions = _filtered_additions

        # ROOT CAUSE 5 fix (2026-06-01): collect the cols-space indices of the
        # engineered columns appended below so they can be added DIRECTLY to
        # ``selected_vars`` for the default single-step (``fe_max_steps==1``)
        # path. The screening re-run that would normally promote appended cols
        # only happens on the NEXT outer-loop iteration; with the default
        # ``fe_max_steps=1`` the loop breaks before re-screening, so a recommended
        # engineered column never reached ``_engineered_features_``. Mirroring the
        # cluster_aggregate pattern (which already self-selects its aggregate),
        # we promote the FE survivors here. On multi-step (``> 1``) the next
        # screening pass re-evaluates them as usual and may drop weak ones.
        # Seed with the polynom-pair engineered indices captured above so they
        # are promoted into ``selected_vars`` together with the unary/binary
        # ones below (ROOT CAUSE 5). They already cleared every polynom-FE gate.
        _newly_engineered_indices: list[int] = list(_polynom_engineered_indices)
        # 2026-06-02: a fit() MUST NOT mutate the caller's input. The pandas
        # branch below appends engineered columns via ``X[col] = ...`` IN PLACE;
        # without this guard the user's DataFrame silently grows engineered
        # columns after ``MRMR().fit(df, y)`` (and the leak bled across fits that
        # reused one frame). Copy ONCE, up front, only when at least one pair
        # actually produced an engineered column. Polars (``.with_columns``)
        # already returns a fresh frame, and the ndarray path never appends to X.
        if (
            not _is_polars_input
            and hasattr(X, "columns")
            and any(v[0] for v in prospective_additions.values())
        ):
            X = X.copy()
        for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, new_nbins, messages) in prospective_additions.items():
            if this_pair_features:
                engineered_features.update(this_pair_features)
                if verbose:
                    for mes in messages:
                        logger.info(mes)
                    # logger.info(f"Features {new_cols} are recommended to use as new features!")
                if fe_max_steps >= 1:
                    new_vals = np.empty(shape=(len(X), len(this_pair_features)), dtype=self.quantization_dtype)
                    for j in range(len(this_pair_features)):
                        new_vals[:, j] = discretize_array(
                            arr=transformed_vals[:, j],
                            n_bins=self.quantization_nbins,
                            method=self.quantization_method,
                            dtype=self.quantization_dtype,
                        )
                    _n_cols_before = len(cols)
                    data = np.append(data, new_vals, axis=1)
                    # ``nbins`` is a numpy.ndarray (returned by categorize_dataset), so plain ``+`` does
                    # element-wise addition / broadcasting, not concatenation. Use np.concatenate so nbins
                    # grows in lockstep with data.shape[1] (otherwise screen_predictors trips its
                    # targets_data.shape[1] == len(targets_nbins) assertion when engineered cols feed back).
                    nbins = np.concatenate([
                        np.asarray(nbins),
                        np.asarray(new_nbins, dtype=nbins.dtype),
                    ])
                    cols = cols + new_cols
                    # cols-space indices of the freshly appended engineered columns.
                    _newly_engineered_indices.extend(range(_n_cols_before, len(cols)))
                    # Use the DISCRETISED codes (``new_vals``) for the augmented
                    # output frame, NOT the raw ``transformed_vals``. The fit-time
                    # frame must match what ``transform()`` reproduces on test data
                    # (the recipe replay emits quantised bin codes), otherwise a
                    # consumer reading the fit-time augmented frame would see raw
                    # floats while transform() emits codes -- a silent fit/transform
                    # skew. ``transformed_vals`` (raw) is still used below to pin the
                    # recipe's quantile edges.
                    if _is_polars_input:
                        # Polars is immutable: with_columns returns a new frame sharing buffers; caller's X untouched.
                        _series_to_add = [
                            pl.Series(col, new_vals[:, j])
                            for j, col in enumerate(new_cols)
                        ]
                        X = X.with_columns(_series_to_add)
                    else:
                        # 2026-06-01: index by the per-column position, not the
                        # leaked loop variable ``j`` (which held len-1 after the
                        # discretize loop above, so EVERY appended pandas column
                        # silently received the LAST survivor's values).
                        for _jc, col in enumerate(new_cols):
                            X[col] = new_vals[:, _jc]

                    # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): stash the CONTINUOUS
                    # engineered values (``transformed_vals``) keyed by column name. The
                    # augmented frame ``X`` only carries the DISCRETISED bin codes (needed
                    # for screening), but the NEXT FE step's pair search must combine the
                    # CONTINUOUS values: ``add(bin_codes(eng1), bin_codes(eng2))`` is
                    # severely lossy (measured: the additive composite of the two real
                    # step-1 features keeps MI 0.88 from bin codes vs 1.81 -- the full
                    # signal -- from continuous values, so the code form fails the
                    # engineered-MI gate). ``check_prospective_fe_pairs`` reads this store
                    # (threaded as ``engineered_operand_values``) so ``(eng_i, eng_j)``
                    # composites are built on the continuous values and recover the signal.
                    _eng_cont_store = getattr(self, "_engineered_continuous_", None)
                    if _eng_cont_store is None:
                        _eng_cont_store = {}
                        self._engineered_continuous_ = _eng_cont_store
                    for _jc, col in enumerate(new_cols):
                        if transformed_vals.shape[1] > _jc:
                            _eng_cont_store[col] = np.asarray(transformed_vals[:, _jc], dtype=np.float64)

                    # Build EngineeredRecipe for each newly-appended column so transform() can replay it.
                    # Runs whenever columns were added (fe_max_steps >= 1). NESTED-ENGINEERED PARENTS
                    # (2026-06-08): a parent that is itself an engineered column (a higher-order
                    # composite, e.g. add(div(sqr(a),abs(b)), mul(log(c),sin(d)))) is now REPLAYABLE --
                    # we pass the parent's own EngineeredRecipe (already in ``engineered_recipes`` from
                    # the prior step) so replay recomputes it recursively. Only when a parent is
                    # engineered AND has no replayable recipe do we skip (cannot reconstruct it).
                    if engineered_recipes is not None:
                        from ..engineered_recipes import build_unary_binary_recipe
                        _raw_names = set(self.feature_names_in_)
                        for config, _j in this_pair_features:
                            # config = (transformations_pair, bin_func_name, i)
                            # transformations_pair = ((var_a_idx, unary_a_name),
                            #                        (var_b_idx, unary_b_name))
                            transformations_pair, bin_func_name, _ = config
                            (var_a_idx, unary_a_name) = transformations_pair[0]
                            (var_b_idx, unary_b_name) = transformations_pair[1]
                            # Map cols-index -> name. A RAW parent resolves to a ``feature_names_in_``
                            # name; an ENGINEERED parent resolves to its prior recipe (nested replay).
                            src_a_name_raw = cols[var_a_idx]
                            src_b_name_raw = cols[var_b_idx]
                            _nested_a = None if src_a_name_raw in _raw_names else engineered_recipes.get(src_a_name_raw)
                            _nested_b = None if src_b_name_raw in _raw_names else engineered_recipes.get(src_b_name_raw)
                            # Skip only when an operand is engineered but its parent recipe is missing
                            # (un-replayable) -- e.g. a parent from a stage that did not register one.
                            _a_unreplayable = (src_a_name_raw not in _raw_names) and (_nested_a is None)
                            _b_unreplayable = (src_b_name_raw not in _raw_names) and (_nested_b is None)
                            if _a_unreplayable or _b_unreplayable:
                                if verbose:
                                    logger.info(
                                        "Skipping recipe construction for nested engineered feature "
                                        "'%s' (parent %s has no replayable recipe).",
                                        get_new_feature_name(config, cols),
                                        src_a_name_raw if _a_unreplayable else src_b_name_raw,
                                    )
                                continue
                            eng_name = get_new_feature_name(config, cols)
                            # 2026-05-30 Wave 9.1 fix (loop iter 28):
                            # pass the fit-time engineered values
                            # ``transformed_vals[:, _j]`` so the recipe
                            # persists the quantile edges. Pre-fix replay
                            # re-quantiled on test data, silently shifting
                            # bin codes between fit and transform under
                            # distribution drift.
                            _fit_vals = transformed_vals[:, _j] \
                                if transformed_vals.shape[1] > _j else None
                            # Per-operand pre-warp: when a side used the learned
                            # ``prewarp`` pseudo-unary, hand its fitted spec to the
                            # recipe so replay reproduces the closed-form warp.
                            _pw_a = _prewarp_specs.get(var_a_idx) if unary_a_name == "prewarp" else None
                            _pw_b = _prewarp_specs.get(var_b_idx) if unary_b_name == "prewarp" else None
                            # Per-operand median gate: when a side used the
                            # ``gate_med`` pseudo-unary, hand its fitted TRAIN
                            # median to the recipe so replay reproduces the
                            # closed-form ``(x > median)`` gate.
                            _gm_a = _gate_med_specs.get(var_a_idx) if unary_a_name == "gate_med" else None
                            _gm_b = _gate_med_specs.get(var_b_idx) if unary_b_name == "gate_med" else None
                            engineered_recipes[eng_name] = build_unary_binary_recipe(
                                name=eng_name,
                                src_a_name=src_a_name_raw,
                                src_b_name=src_b_name_raw,
                                unary_a_name=unary_a_name,
                                unary_b_name=unary_b_name,
                                binary_name=bin_func_name,
                                unary_preset=fe_unary_preset,
                                binary_preset=fe_binary_preset,
                                quantization_nbins=self.quantization_nbins,
                                quantization_method=self.quantization_method,
                                quantization_dtype=self.quantization_dtype,
                                fit_values_for_edges=_fit_vals,
                                prewarp_a=_pw_a,
                                prewarp_b=_pw_b,
                                gate_med_a=_gm_a,
                                gate_med_b=_gm_b,
                                # Nested-engineered parents (2026-06-08): None for raw operands,
                                # else the parent's recipe so replay recomputes it recursively.
                                nested_parent_a=_nested_a,
                                nested_parent_b=_nested_b,
                            )

                n_recommended_features += len(this_pair_features)

            # Wave 69 (2026-05-20): factors_to_use / factors_names_to_use are
            # already threaded through the upstream FE loop (MRMR.fit -> FE-pair
            # iteration consults these via `self.factors_to_use` and the
            # caller-supplied filter); no extra plumbing needed at this
            # bookkeeping site. The pair-cache only tracks "raw pair already
            # processed", which is name-agnostic.
            checked_pairs.add(raw_vars_pair)

        # AUTO-ESCALATION to the richer SHIPPED bases (2026-06-10, backlog idea B,
        # default-ON). A pair that PASSED the pair-MI prescreen (ratio gate + order-2
        # maxT floor) but for which the unary/binary search above admitted NOTHING used
        # to end in the log_fe_summary WARNING below -- detected signal, silently
        # abandoned. Escalate instead: PROPOSE candidates from the richer shipped basis
        # families (signal-adaptive orth-poly ALS warp across the 4 polynomial bases at
        # a higher degree + DEMODULATED adaptive-frequency Fourier/chirp warps -- e.g.
        # the sin(3.7*a)*b inner frequency no library unary can express) and let the
        # EXISTING gates decide (maxT floor on MM-debiased MI + marginal-permutation
        # floor + the S5 conditional-MI redundancy gate vs the admitted engineered
        # support). Structurally a no-op (one set-difference) when every surviving pair
        # produced an admitted column -- the common case. See ``_fe_auto_escalation``.
        if bool(getattr(self, "fe_auto_escalation_enable", True)) and prospective_pairs:
            try:
                # Per-fit escalation ledger: a pair escalated once is never re-escalated
                # in a later FE step of the SAME fit (a step-2 retry would re-propose the
                # identical candidate on identical data and emit a duplicate ``..._2``
                # column). Reset on the first FE step so re-fits start clean.
                if num_fs_steps == 0 or not hasattr(self, "_fe_escalation_done_pairs_"):
                    self._fe_escalation_done_pairs_ = set()
                    self.fe_escalation_history_ = []
                _esc_done = self._fe_escalation_done_pairs_
                _esc_pairs_with_additions = {
                    _rp for _rp, _v in prospective_additions.items() if _v[0]
                }
                _esc_failed = [
                    (_k[0], float(_k[1])) for _k in prospective_pairs
                    if _k[0] not in _esc_pairs_with_additions and _k[0] not in _esc_done
                ]
                # UNDERDELIVERY trigger (2026-06-10): a pair that DID admit a column but
                # whose best capture leaves SIGNIFICANT conditional pair MI on the table
                # (leftover CMI(joint(a,b); y | best admitted) above its conditional-
                # permutation null) is escalated too -- e.g. the ``y=sin(3.7a)*b``
                # envelope capture ``mul(sin(a),qubed(b))`` that the marginal-uplift
                # fallback admits while most of the detected signal stays unexpressed.
                # Stride-subsampled + 8-perm null keeps the every-pair-delivers common
                # case cheap; a false trigger only PROPOSES -- the full gates (incl. the
                # S5 CMI gate vs the pair's own admitted column) still decide. See
                # ``find_underdelivering_pairs``.
                if bool(getattr(self, "fe_escalation_underdelivery_enable", True)):
                    from .._fe_auto_escalation import find_underdelivering_pairs
                    _esc_failed.extend(find_underdelivering_pairs(
                        self,
                        prospective_pairs=prospective_pairs,
                        prospective_additions=prospective_additions,
                        X=X, cols=cols, classes_y=classes_y, done=_esc_done,
                    ))
                if _esc_failed:
                    from .._fe_auto_escalation import run_fe_auto_escalation
                    from .._mi_greedy_cmi_fe import _cmi_from_binned as _esc_mi, _quantile_bin as _esc_qbin
                    # Admitted-support context for the S5 gate: the engineered columns
                    # the main path just materialised (continuous values + marginal MI).
                    _esc_y = np.asarray(classes_y)
                    if not np.issubdtype(_esc_y.dtype, np.integer):
                        _esc_y = _esc_y.astype(np.int64)
                    _, _esc_y_dense = np.unique(_esc_y, return_inverse=True)
                    _esc_y_dense = _esc_y_dense.astype(np.int64)
                    _esc_admitted_pool: dict = {}
                    for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                        if not _tpf or _tvals is None or not _ncols:
                            continue
                        for _jc, _cname in enumerate(_ncols):
                            if _tvals.shape[1] <= _jc:
                                continue
                            _cv = np.asarray(_tvals[:, _jc], dtype=np.float64)
                            _cb = _esc_qbin(_cv, nbins=int(self.quantization_nbins))
                            _esc_admitted_pool[_cname] = (_cv, float(_esc_mi(_cb, _esc_y_dense, None)))
                    # Per-pair admitted-capture values: UNDERDELIVERY-triggered pairs
                    # get their proposers fit on the RESIDUAL of the target given the
                    # existing capture (see ``run_fe_auto_escalation``); zero-admission
                    # pairs have no entry and fit the full target.
                    _esc_capture_vals: dict = {}
                    for _pp, _pmi in _esc_failed:
                        _v = prospective_additions.get(_pp)
                        if _v and _v[0] and _v[1] is not None and _v[2]:
                            _esc_capture_vals[tuple(_pp)] = _v[1][:, : min(int(_v[1].shape[1]), len(_v[2]))]
                    _esc_admitted = run_fe_auto_escalation(
                        self,
                        failed_pairs=_esc_failed,
                        X=X, cols=cols,
                        classes_y=classes_y,
                        pair_maxt_floor=float(_pair_maxt_floor),
                        admitted_pool=_esc_admitted_pool,
                        verbose=verbose,
                        capture_vals=_esc_capture_vals,
                    )
                    # Mark the pairs escalation actually PROCESSED (budget-selected
                    # eligible) as done for this fit, admitted or not -- a retry on
                    # identical data cannot change the verdict.
                    _esc_done.update(
                        getattr(self, "fe_escalation_info_", {}).get("eligible_idx", []) or []
                    )
                    if _esc_admitted:
                        # Materialise exactly like the unary/binary survivors above:
                        # discretised codes into data/X, name into cols, nbins in
                        # lockstep, recipe registered, continuous values stashed for
                        # the engineered-operand feed-forward, index promoted below.
                        if not _is_polars_input and hasattr(X, "columns"):
                            X = X.copy()
                        _esc_new_codes = np.empty(
                            shape=(len(X), len(_esc_admitted)), dtype=self.quantization_dtype,
                        )
                        for _je, _ec in enumerate(_esc_admitted):
                            _esc_new_codes[:, _je] = discretize_array(
                                arr=np.asarray(_ec["values"], dtype=np.float64),
                                n_bins=self.quantization_nbins,
                                method=self.quantization_method,
                                dtype=self.quantization_dtype,
                            )
                        _n_cols_before_esc = len(cols)
                        data = np.append(data, _esc_new_codes, axis=1)
                        nbins = np.concatenate([
                            np.asarray(nbins),
                            np.asarray([self.quantization_nbins] * len(_esc_admitted), dtype=np.asarray(nbins).dtype),
                        ])
                        cols = cols + [_ec["name"] for _ec in _esc_admitted]
                        _newly_engineered_indices.extend(range(_n_cols_before_esc, len(cols)))
                        n_recommended_features += len(_esc_admitted)
                        _eng_cont_store = getattr(self, "_engineered_continuous_", None)
                        if _eng_cont_store is None:
                            _eng_cont_store = {}
                            self._engineered_continuous_ = _eng_cont_store
                        if _is_polars_input:
                            X = X.with_columns([
                                pl.Series(_ec["name"], _esc_new_codes[:, _je])
                                for _je, _ec in enumerate(_esc_admitted)
                            ])
                        for _je, _ec in enumerate(_esc_admitted):
                            if not _is_polars_input:
                                X[_ec["name"]] = _esc_new_codes[:, _je]
                            _eng_cont_store[_ec["name"]] = np.asarray(_ec["values"], dtype=np.float64)
                            if engineered_recipes is not None:
                                engineered_recipes[_ec["name"]] = _ec["recipe"]
            except Exception:
                logger.warning(
                    "MRMR FE auto-escalation failed; continuing with the unary/binary survivors only.",
                    exc_info=True,
                )

        # ROOT CAUSE 5 fix (2026-06-01): promote the freshly-appended engineered
        # columns directly into ``selected_vars`` (cols-space). They already
        # cleared every FE gate (pair-MI prevalence, engineered-MI prevalence,
        # external validation) -- the gates ARE the selection criterion for FE
        # survivors. Without this, the only path to ``support_`` was the
        # screening re-run at the top of the NEXT outer-loop iteration, which
        # never executes under the default ``fe_max_steps=1`` (the loop breaks
        # first), so ``_engineered_features_`` stayed empty. On multi-step the
        # re-screen still re-evaluates them and may prune weak ones. Mirrors the
        # cluster_aggregate self-selection pattern below.
        if _newly_engineered_indices:
            _sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
            _sv_set = set(_sv)
            selected_vars = _sv + [i for i in _newly_engineered_indices if i not in _sv_set]

        # CROSS-FOLD RECIPE STABILITY VOTING (2026-06-10, backlog #15). A near-free
        # consensus layer OVER the existing FE gates. The expensive search above ran
        # ONCE on the full data; here we add a cheap K-fold CONFIRMATION -- each
        # surviving unary_binary recipe is REPLAYED (leak-safe: the recipe is frozen,
        # only the rows change) on K held-out folds, its uplift gate statistic
        # recomputed per fold, and the recipe ADMITTED only if it clears the gate in
        # >= ceil(q*K) folds. This complements the order-2/order-3 maxT floors: maxT
        # kills the chance-MAX candidate WITHIN a fold (best-of-pool selection bias);
        # this kills a recipe that won only on a fold-specific QUIRK of the full-data
        # split (its uplift carried by a few rows in the train split, collapses on the
        # held-out folds). NO REFIT -- only K plug-in-MI replays per recipe, so the
        # cost is negligible. Failed recipes are dropped from BOTH ``engineered_recipes``
        # (so they never reach ``self._engineered_recipes_`` at fit-end) and from
        # ``selected_vars`` (so they never reach ``support_``). Default-ON; self-gates
        # to a no-op below 2 unary_binary survivors / k<2 / tiny n. ``fe_stability_vote_enable=False``
        # byte-reproduces the pre-vote support.
        if (
            engineered_recipes
            and bool(getattr(self, "fe_stability_vote_enable", True))
            and _newly_engineered_indices
        ):
            try:
                from .._fe_stability_vote import confirm_recipes_cross_fold

                _failed_eng = confirm_recipes_cross_fold(
                    recipes=engineered_recipes,
                    X=X,
                    y_codes=classes_y,
                    feature_names_in=list(self.feature_names_in_),
                    nbins=int(self.quantization_nbins),
                    k=int(getattr(self, "fe_stability_vote_k", 5)),
                    quorum=float(getattr(self, "fe_stability_vote_quorum", 0.6)),
                    rng=np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)),
                    verbose=int(verbose),
                )
                if _failed_eng:
                    # Drop the failed engineered names from selected_vars (by cols-index)
                    # and from the recipe dict so neither support_ nor _engineered_recipes_
                    # admits a fold-specific winner.
                    _failed_idx = {i for i in selected_vars if 0 <= i < len(cols) and cols[i] in _failed_eng}
                    if _failed_idx:
                        selected_vars = [i for i in selected_vars if i not in _failed_idx]
                    for _fn in _failed_eng:
                        engineered_recipes.pop(_fn, None)
            except Exception as _vote_exc:
                if verbose:
                    logger.warning(
                        "MRMR cross-fold stability vote failed (%s: %s); keeping the un-voted FE support.",
                        type(_vote_exc).__name__, _vote_exc,
                    )

        log_fe_summary(
            prospective_pairs=prospective_pairs,
            prospective_additions=prospective_additions,
            n_recommended_features=n_recommended_features,
            fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
            fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
            fe_acceptance=str(getattr(self, "fe_acceptance", "conditional_mi")),
            verbose=verbose,
        )

    data, cols, nbins, X, selected_vars, n_recommended_features = run_cluster_aggregate_emission(
        self,
        data=data,
        cols=cols,
        nbins=nbins,
        X=X,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        selected_vars=selected_vars,
        n_recommended_features=n_recommended_features,
        num_fs_steps=num_fs_steps,
        _is_polars_input=_is_polars_input,
        verbose=verbose,
    )

    return data, cols, nbins, X, selected_vars, n_recommended_features


