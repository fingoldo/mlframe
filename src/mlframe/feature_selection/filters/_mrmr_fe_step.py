"""``MRMR._run_fe_step`` -- FE-step orchestrator for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. ``_run_fe_step`` is bound back onto the ``MRMR`` class at the
parent's module bottom, so call sites that invoke ``self._run_fe_step(...)``
continue to work unchanged.

Carries the per-fold FE expansion logic (unary / binary / hermite / pysr
candidate generation, scoring, append-to-data) and the support-map
bookkeeping. The parent module's helpers (``_lazy_chunks``, ``MRMR`` for
class-level cache attrs) are imported eagerly since the parent is already
loaded by the time this sibling is imported.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import logging
import math
import os
import textwrap
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations, islice
from timeit import default_timer as timer
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


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
    fe_unary_preset: str = "minimal",
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
    from .mrmr import (
        MRMR,
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

    if self.fe_ntop_features:
        numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
    else:
        numeric_vars_to_consider = selected_vars

    numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

    # Honor factors_to_use / factors_names_to_use in the FE step too; intersect the FE pool with the user's
    # restriction so the contract matches the screening step.
    if self.factors_to_use is not None:
        numeric_vars_to_consider = numeric_vars_to_consider & set(self.factors_to_use)
    if self.factors_names_to_use is not None:
        allowed = {cols.index(n) for n in self.factors_names_to_use if n in cols}
        numeric_vars_to_consider = numeric_vars_to_consider & allowed

    # 2026-06-02 -- SYNERGY BOOTSTRAP (see ``fe_synergy_screen_max_features`` in
    # MRMR.__init__). Pure-synergy interactions (a*d, sign products, log(c)*sin(d))
    # have ~zero marginal MI per factor, so neither factor is screened-in and the
    # pair never reaches the prospective-pair screen below -- even though that
    # screen ALREADY keeps a zero-individual-MI pair whose JOINT MI is positive
    # (the canonical XOR branch). The fix is purely to widen the POOL: when the raw
    # numeric feature count is within the cap, add the UNSELECTED raw numeric columns
    # so the all-pairs joint-MI sweep screens the synergy pairs. Two cost/quality
    # guards live downstream: (1) synergy pairs (>=1 bootstrap-added operand) must
    # clear the STRICTER ``fe_synergy_min_prevalence`` uplift bar (rejects finite-
    # sample-bias noise pairs), and (2) the surviving synergy pairs are budget-capped
    # to ``fe_synergy_max_pairs`` by joint MI (bounds the expensive per-pair search).
    # Runs only on the FIRST FE step (where the bootstrap matters).
    _synergy_cap = int(getattr(self, "fe_synergy_screen_max_features", 0) or 0)
    _synergy_added_idx: set = set()
    # MIN-ROWS guard: a 2-D joint-MI estimate is dominated by finite-sample bias at
    # tiny n, so the synergy uplift gate admits a pure-NOISE pair and a spurious
    # feature is engineered (measured: ``max(neg(a),d)`` on random y at n=100). The
    # bootstrap needs enough rows for the joint MI to be meaningful; below
    # ``fe_synergy_min_rows`` it is disabled (genuine synergy is detected with the
    # samples it needs -- the recovery wins are at n>=2000, unaffected).
    _synergy_min_rows = int(getattr(self, "fe_synergy_min_rows", 300) or 0)
    _n_rows_for_synergy = int(data.shape[0]) if hasattr(data, "shape") else 0
    if _synergy_cap > 0 and num_fs_steps == 0 and _n_rows_for_synergy >= _synergy_min_rows:
        _raw_names = set(getattr(self, "feature_names_in_", []) or [])
        _target_idx_set = {int(t) for t in np.atleast_1d(target_indices)}
        _cat_set = set(categorical_vars)
        _raw_numeric_idx = {
            i for i, nm in enumerate(cols)
            if nm in _raw_names and i not in _target_idx_set and i not in _cat_set
        }
        if self.factors_to_use is not None:
            _raw_numeric_idx &= set(self.factors_to_use)
        if self.factors_names_to_use is not None:
            _raw_numeric_idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}
        if 0 < len(_raw_numeric_idx) <= _synergy_cap:
            _added = _raw_numeric_idx - numeric_vars_to_consider
            if _added:
                _synergy_added_idx = set(_added)
                numeric_vars_to_consider = numeric_vars_to_consider | _raw_numeric_idx
                if verbose:
                    logger.info(
                        "MRMR FE synergy bootstrap: augmented pair pool with %d unselected raw "
                        "numeric columns (%d raw <= cap %d) so zero-marginal synergy pairs "
                        "(a*d / sign products / log*sin) get joint-MI screened.",
                        len(_added), len(_raw_numeric_idx), _synergy_cap,
                    )
        elif len(_raw_numeric_idx) > _synergy_cap and verbose:
            logger.info(
                "MRMR FE synergy bootstrap: %d raw numeric columns > cap %d; skipping the "
                "all-pairs synergy sweep (keeping the selected-only pool). Raise "
                "fe_synergy_screen_max_features to enable it on this frame.",
                len(_raw_numeric_idx), _synergy_cap,
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
    _pair_maxt_floor = 0.0
    _pair_maxt_perms = int(getattr(self, "fe_pair_maxt_null_permutations", 25) or 0)
    if _pair_maxt_perms > 0 and len(numeric_vars_to_consider) >= 2 and n_pairs >= int(getattr(self, "fe_pair_maxt_min_pairs", 30)):
        try:
            from ._permutation_null import pooled_pair_permutation_null_joint_mi_floor

            _maxt_pairs = list(combinations(numeric_vars_to_consider, 2))
            _maxt_pa = np.fromiter((p[0] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            _maxt_pb = np.fromiter((p[1] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            _pair_maxt_floor = pooled_pair_permutation_null_joint_mi_floor(
                factors_data=data,
                nbins=nbins,
                pair_a=_maxt_pa,
                pair_b=_maxt_pb,
                classes_y=classes_y,
                freqs_y=freqs_y,
                n_permutations=_pair_maxt_perms,
                quantile=float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                random_seed=getattr(self, "random_seed", None),
            )
            if _pair_maxt_floor > 0.0 and verbose >= 1:
                logger.info(
                    "MRMR FE: order-2 maxT permutation-null joint-MI floor=%.5f over %d candidate "
                    "pairs (q=%.2f, K=%d) - rejects best-of-p chance-max noise pairs.",
                    _pair_maxt_floor, n_pairs, float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                    _pair_maxt_perms,
                )
        except Exception:
            logger.warning(
                "MRMR FE: order-2 maxT permutation-null floor failed; continuing without it.",
                exc_info=True,
            )
            _pair_maxt_floor = 0.0

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
                if ind_elems_mi_sum <= 0:
                    # ORDER-2 maxT floor (computed once above): a zero-individual-MI
                    # pair enters via the canonical XOR branch on ANY positive joint
                    # MI, so on a wide noise matrix a noise pair whose joint MI is
                    # merely the best chance hit slips through. Require the joint MI
                    # to clear the pool's permutation-null max before keeping it;
                    # genuine pure-synergy (XOR / sign product) joint MI is FAR above
                    # the chance ceiling, so it survives. No-op when floor==0.0.
                    if pair_mi > 0 and pair_mi >= _pair_maxt_floor:
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
                if pair_mi > ind_elems_mi_sum * _prev_thresh and pair_mi >= _pair_maxt_floor:
                    uplift = pair_mi / ind_elems_mi_sum
                    if verbose >= 2:
                        logger.info(
                            "Factors pair %s will be considered for Feature Engineering, %.4f->%.4f, rat=%.2f",
                            raw_vars_pair, ind_elems_mi_sum, pair_mi, uplift,
                        )
                    prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                    for var in raw_vars_pair:
                        vars_usage_counter[var] += 1

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
        from .polynom_pair_fe import run_polynom_pair_fe
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
        _prewarp_specs: dict = {}

        if len(X) < 50_000 or len(prospective_pairs) < 2:
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
                prewarp_enable=_prewarp_enable,
                prewarp_y=classes_y if _prewarp_enable else None,
                prewarp_basis=_prewarp_basis,
                prewarp_max_degree=_prewarp_max_degree,
                prewarp_uplift_threshold=_prewarp_uplift,
                prewarp_min_val_corr=_prewarp_min_val_corr,
                prewarp_specs_out=_prewarp_specs,
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
                        prewarp_enable=_prewarp_enable,
                        prewarp_y=classes_y if _prewarp_enable else None,
                        prewarp_basis=_prewarp_basis,
                        prewarp_max_degree=_prewarp_max_degree,
                        prewarp_uplift_threshold=_prewarp_uplift,
                        prewarp_min_val_corr=_prewarp_min_val_corr,
                        prewarp_specs_out=None,  # loky: recovered from result dict below
                    )
                    for chunk in jobs_list
                ],
                # max_nbytes=0,
                n_jobs=n_jobs,
                **parallel_kwargs,
            )
            for next_dict in dicts:
                prospective_additions.update(next_dict)

        # Extract the reserved pre-warp-specs entry the FE-pair helper stuffs
        # into its result dict (so loky-parallel chunks can return specs); merge
        # into ``_prewarp_specs`` and remove it so the pair loop below never
        # treats it as a ``raw_vars_pair``.
        from ._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY
        _pw_from_res = prospective_additions.pop(_PREWARP_SPECS_RESULT_KEY, None)
        if _pw_from_res:
            _prewarp_specs.update(_pw_from_res)

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

                    # Build EngineeredRecipe for each newly-appended column so transform() can replay it.
                    # Runs whenever columns were added (fe_max_steps >= 1). Best-effort: parents that are
                    # themselves engineered (higher-order interaction) are skipped (nested replay is future work).
                    if engineered_recipes is not None:
                        from .engineered_recipes import build_unary_binary_recipe
                        for config, _j in this_pair_features:
                            # config = (transformations_pair, bin_func_name, i)
                            # transformations_pair = ((var_a_idx, unary_a_name),
                            #                        (var_b_idx, unary_b_name))
                            transformations_pair, bin_func_name, _ = config
                            (var_a_idx, unary_a_name) = transformations_pair[0]
                            (var_b_idx, unary_b_name) = transformations_pair[1]
                            # Map cols-index -> feature_names_in_-name. If a parent is itself engineered,
                            # cols[var] is not in feature_names_in_; skip with a warning rather than produce
                            # an unreplayable recipe.
                            src_a_name_raw = cols[var_a_idx]
                            src_b_name_raw = cols[var_b_idx]
                            if (
                                src_a_name_raw not in self.feature_names_in_
                                or src_b_name_raw not in self.feature_names_in_
                            ):
                                if verbose:
                                    logger.info(
                                        "Skipping recipe construction for nested "
                                        "engineered feature '%s' (parents %s, %s "
                                        "are not in feature_names_in_); higher-"
                                        "order replay is future work.",
                                        get_new_feature_name(config, cols),
                                        src_a_name_raw, src_b_name_raw,
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
                            )

                n_recommended_features += len(this_pair_features)

            # Wave 69 (2026-05-20): factors_to_use / factors_names_to_use are
            # already threaded through the upstream FE loop (MRMR.fit -> FE-pair
            # iteration consults these via `self.factors_to_use` and the
            # caller-supplied filter); no extra plumbing needed at this
            # bookkeeping site. The pair-cache only tracks "raw pair already
            # processed", which is name-agnostic.
            checked_pairs.add(raw_vars_pair)

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

        # Surface WHY FE added 0 features when the operator configured it
        # explicitly. A prod log showed 88 min of Hermite Optuna yielding
        # 0 engineered cols with no visible
        # explanation (kept 25 cols, returned 25, dedup at downstream
        # marked MRMR identity-equivalent). The summary below explains:
        # n_pairs_considered: how many (a, b) pairs were screened
        # n_pairs_with_additions: how many pairs produced ANY recipe
        # n_engineered_features: total recipes that survived all gates
        # If 0 with verbose >= 1, also log the gate thresholds so an
        # operator can see which knob is too tight (often
        # ``fe_min_engineered_mi_prevalence=0.98`` is the culprit on
        # heavily-correlated feature sets).
        try:
            _n_pairs_considered = int(len(prospective_pairs))
        except Exception:
            _n_pairs_considered = -1
        try:
            _n_pairs_with_additions = sum(
                1 for v in prospective_additions.values()
                if v[0]  # this_pair_features non-empty
            )
        except Exception:
            _n_pairs_with_additions = -1
        if verbose >= 1:
            logger.info(
                "FE summary: %d pair(s) considered, %d produced engineered cols, "
                "n_total_engineered=%d. Gate thresholds: "
                "fe_min_pair_mi_prevalence=%.3f, "
                "fe_min_engineered_mi_prevalence=%.3f, "
                "fe_min_nonzero_confidence=%.3f, "
                "fe_good_to_best_feature_mi_threshold=%.3f.",
                _n_pairs_considered, _n_pairs_with_additions,
                n_recommended_features,
                float(fe_min_pair_mi_prevalence),
                float(fe_min_engineered_mi_prevalence),
                float(fe_min_nonzero_confidence),
                float(fe_good_to_best_feature_mi_threshold),
            )
            if n_recommended_features == 0 and _n_pairs_considered > 0:
                logger.warning(
                    "FE produced 0 engineered features despite %d pair(s) "
                    "passing the pair-MI gate. Likely cause: the "
                    "fe_min_engineered_mi_prevalence=%.3f threshold is "
                    "tight relative to the pair-level MI. Try lowering "
                    "to 0.90 (5%% under the default) or set "
                    "fe_min_pair_mi_prevalence=1.02 to widen the pool.",
                    _n_pairs_considered,
                    float(fe_min_engineered_mi_prevalence),
                )

    # Clustered-feature aggregation (opt-in): denoise correlated "reflection" clusters into one
    # aggregate column (a k-ary engineered recipe). Run once on the first FE step -- it clusters only
    # raw ``feature_names_in_`` columns, which don't change across FE steps. Guarded so a failure never
    # aborts fit, mirroring the friend-graph block.
    if getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0:
        try:
            from ._cluster_aggregate import run_cluster_aggregate_step

            data, cols, nbins, X, _ca_added, _ca_removed, _ca_indices, _ca_summary = run_cluster_aggregate_step(
                data=data, cols=cols, nbins=nbins, X=X, target_indices=target_indices,
                feature_names_in_=list(self.feature_names_in_), categorical_idx=categorical_vars,
                cached_MIs=cached_MIs, engineered_recipes=engineered_recipes,
                quantization_nbins=self.quantization_nbins, quantization_method=self.quantization_method,
                quantization_dtype=self.quantization_dtype,
                methods=tuple(self.cluster_aggregate_methods),
                mi_prevalence=self.cluster_aggregate_mi_prevalence,
                min_member_relevance=self.cluster_aggregate_min_member_relevance,
                min_cluster_size=self.cluster_aggregate_min_cluster_size,
                max_cluster_size=self.cluster_aggregate_max_cluster_size,
                corr_threshold=self.cluster_aggregate_corr_threshold,
                homogeneity_tau=self.cluster_aggregate_homogeneity_tau,
                max_candidates=self.cluster_aggregate_max_candidates,
                mode=self.cluster_aggregate_mode, is_polars_input=_is_polars_input,
                verbose=verbose, dtype=self.dtype,
            )
            n_recommended_features += int(_ca_added)
            if _ca_indices:
                # The aggregate already passed the supervised MI gate (beats best member), so select it
                # directly -- don't rely on a re-screen (with the default fe_max_steps=1 the loop breaks
                # before re-screening). Remap routes the engineered name into _engineered_recipes_.
                _sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
                selected_vars = _sv + [i for i in _ca_indices if i not in _sv]
            if _ca_removed:  # replace mode: consumed in _fit_impl before the cols->original remap
                self._cluster_aggregate_removals_ = list(getattr(self, "_cluster_aggregate_removals_", [])) + list(_ca_removed)
            if _ca_summary:
                # Fitted summary -> surfaced into meta_info["feature_selection_report"]["cluster_aggregate"].
                self.cluster_aggregate_ = list(getattr(self, "cluster_aggregate_", []) or []) + _ca_summary
                logger.info(
                    "MRMR cluster_aggregate (%s): built %d denoised aggregate(s) from correlated clusters: %s",
                    self.cluster_aggregate_mode, _ca_added,
                    [f"{r['name']} (method={r['method']}, k={len(r['members'])}, +MI {r['mi_gain']:.4f})" for r in _ca_summary],
                )
        except Exception:
            logger.warning("cluster_aggregate step failed; continuing without it.", exc_info=True)

    return data, cols, nbins, X, selected_vars, n_recommended_features


