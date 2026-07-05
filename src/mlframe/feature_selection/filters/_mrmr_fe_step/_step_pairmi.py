"""Operand-pool feed-forward cap + batch pair-MI computation + order-2 maxT floor stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` to keep that module under the 1k-LOC ceiling.
``compute_pair_mis_and_floor`` runs the engineered-operand feed-forward cap, the batch/per-pair joint-MI
computation (CUDA/CPU dispatch + legacy joblib fallback, populating ``cached_MIs`` in place), the order-2
Westfall-Young maxT permutation-null floor, and the "auto" prevalence MM-debias bias fill. The lazy
``..mrmr`` imports are done in-body exactly as in the inline block (the mrmr<->this-package import cycle
forbids a top-level import). Selection is byte-for-byte identical to the inline block.

Returns ``(numeric_vars_to_consider, _eng_cap, _pair_maxt_floor, _pair_mm_bias, _prevalence_debias_auto)``;
``cached_MIs`` is mutated in place.
"""
from __future__ import annotations

import logging
import os
from itertools import combinations

import numpy as np
from joblib import delayed

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._mrmr_fe_step_helpers import compute_pair_maxt_floor


def compute_pair_mis_and_floor(
    self,
    *,
    data, cols, nbins, X,
    classes_y, classes_y_safe, freqs_y,
    target_indices,
    cached_MIs, cached_confident_MIs,
    numeric_vars_to_consider,
    _prevalence_debias_auto,
    n_jobs, prefetch_factor, parallel_kwargs,
    fe_min_nonzero_confidence, fe_npermutations,
    fe_min_pair_mi, fe_min_pair_mi_prevalence,
    verbose,
):
    """Compute pair MIs + the order-2 maxT floor for the FE candidate pool. ``cached_MIs`` mutated in place."""
    # Lazy import: ``.mrmr`` re-imports this package at its bottom for method binding -> a top-level
    # ``from ..mrmr import ...`` here would create a hard import cycle (see _step_core).
    from ..mrmr import (
        _lazy_chunks,
        _MRMR_BATCH_PRECOMPUTE_MAX_K,
        _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
        compute_pairs_mis,
        parallel_run,
        tqdmu,
    )

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
    # ESCALATION FEATURES ARE TERMINAL -- never feed them forward as composite operands
    # (2026-06-12, F2 rescue). An ``esc_*`` escalation feature (orth-poly / adaptive-Fourier
    # pair warp) already captures a genuine richer-basis interaction the library could not
    # express; nesting it INTO a further pair composite (the feed-forward built
    # ``div(log(esc_poly_legendre_mul(a,b)),exp(mul(prewarp(c),prewarp(d))))`` on F2) fuses
    # two INDEPENDENT additive terms of the target into a single ratio whose joint MI tops
    # the greedy ranking, so MRMR then DROPS the clean raw predictors and the standalone
    # captures -- measured on F2: the standalone esc_poly(a,b) is a +0.05 downstream R^2 WIN
    # (raw c,d,f 0.943 -> +a**2/b 0.995), but the fed-forward nested form REGRESSES it to
    # 0.864 by restructuring the selection. The escalation feature stays selected (reaches
    # ``support_``); it just does not seed further composites. Gated on
    # ``fe_escalation_feedforward_enable`` (default OFF -- terminal is the safe default).
    if not bool(getattr(self, "fe_escalation_feedforward_enable", False)):
        _esc_idx = {
            v
            for v in _engineered_in_pool
            if str(cols[v]).startswith("esc_") or "esc_poly_" in str(cols[v]) or "esc_fourier_" in str(cols[v]) or "esc_chirp_" in str(cols[v])
        }
        if _esc_idx:
            numeric_vars_to_consider = set(numeric_vars_to_consider) - _esc_idx
            _engineered_in_pool = [v for v in _engineered_in_pool if v not in _esc_idx]
    if _engineered_in_pool and _eng_cap >= 0:
        if _eng_cap == 0:
            _keep_eng: set = set()
        else:
            # Rank by marginal MI desc; missing single-var MI sorts last. Tie-break on the var index so the
            # cap survivors are deterministic (the pool derives from a set, whose iteration order is not stable
            # across processes -- without the secondary key equal-MI operands kept an arbitrary, irreproducible subset).
            _ranked = sorted(_engineered_in_pool, key=lambda v: (-cached_MIs.get((v,), 0.0), v))
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
    # SECOND FUNNEL STAGE (2026-06-19): when the synergy bootstrap selected the GPU-exhaustive
    # sweep it set ``self._fe_synergy_exhaustive_active_`` -- bypass the _MRMR_BATCH_PRECOMPUTE_MAX_K
    # guard (which caps at 200 cols) and FORCE the cuda backend so the full C(p,2) joint-MI sweep
    # over ALL raw numeric columns runs on the measured CUDA kernel (the only path that recovers a
    # balanced L=0 interaction). The exhaustive decision already verified GPU availability + budget.
    _exhaustive_active = bool(getattr(self, "_fe_synergy_exhaustive_active_", False))
    # When exhaustive is active, run the full C(p,2) sweep on the measured CUDA kernel where a GPU is
    # present, else on the CPU njit-prange backend (decide_exhaustive_sweep made the choice hardware-
    # independent, so a GPU-less host runs exhaustive on CPU rather than silently dropping to the lossy
    # pre-rank -- otherwise a balanced L=0 interaction feature would exist only on CUDA hosts).
    _exhaustive_backend = None
    if _exhaustive_active:
        try:
            from mlframe.feature_selection.filters.batch_pair_mi_gpu import _CUDA_AVAIL as _exh_cuda
        except Exception:
            _exh_cuda = False
        _exhaustive_backend = "cuda" if _exh_cuda else "njit_parallel"
    if _BATCH_PRECOMPUTE_ENABLED and (_exhaustive_active or _k <= _MRMR_BATCH_PRECOMPUTE_MAX_K) and n_pairs >= _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS:
        try:
            from mlframe.feature_selection.filters.batch_pair_mi_gpu import dispatch_batch_pair_mi

            # Build the (a, b) id arrays via ``np.triu_indices`` over the materialised id list instead of
            # ``list(combinations(...))`` -- the exhaustive branch bypasses the _MRMR_BATCH_PRECOMPUTE_MAX_K
            # cap, so at large p the Python tuple list is O(p^2) tuples (~420 MB at p=3888). ``triu_indices``
            # yields ``(_ids[i], _ids[j])`` for i<j in iteration order, byte-for-byte the SAME pair sequence
            # ``combinations(_ids, 2)`` produces, so the ``cached_MIs`` keys + their MI values are identical.
            _ids = list(numeric_vars_to_consider)
            _ids_arr = np.fromiter(_ids, dtype=np.int64, count=len(_ids))
            _ia, _ib = np.triu_indices(len(_ids), k=1)
            _pair_a_arr = _ids_arr[_ia]
            _pair_b_arr = _ids_arr[_ib]
            _pair_mi_batch, _backend_used = dispatch_batch_pair_mi(
                factors_data=data,
                pair_a=_pair_a_arr,
                pair_b=_pair_b_arr,
                nbins=nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                force_backend=_exhaustive_backend,
            )
            # Populate cached_MIs to short-circuit compute_pairs_mis's per-pair mi_direct call.
            # Skip pairs already in cached_confident_MIs (those had a confident permutation outcome).
            # Reconstruct each pair key lazily (``_ids[i], _ids[j]``) -- same Python-int tuple keys as the
            # old ``combinations`` path, without materialising the full tuple list.
            _n_pairs_batch = _ia.shape[0]
            for _i in range(_n_pairs_batch):
                _p = (_ids[_ia[_i]], _ids[_ib[_i]])
                if _p not in cached_confident_MIs and _p not in cached_MIs:
                    cached_MIs[_p] = float(_pair_mi_batch[_i])
                    _batch_prefill_count += 1
            if verbose:
                logger.info(
                    "MRMR FE: batch-prefilled %d/%d pair MIs via %s backend (permutation test skipped for these pairs)",
                    _batch_prefill_count, _n_pairs_batch, _backend_used,
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

    # "auto" prevalence debias needs the per-pair MM joint-MI bias; compute_pair_maxt_floor only
    # populates it when ``fe_mm_debias_prevalence`` is on, so fill it here (analytic, no shuffles)
    # when "auto" is active and the maxT path left it empty. On failure fall back to the fixed bar
    # (degrade to the proven default rather than risk a wrong gate).
    if _prevalence_debias_auto and not _pair_mm_bias:
        try:
            from .._permutation_null import pairwise_mm_joint_bias
            _auto_pairs = list(combinations(numeric_vars_to_consider, 2))
            if _auto_pairs:
                _auto_pa = np.fromiter((p[0] for p in _auto_pairs), dtype=np.int64, count=len(_auto_pairs))
                _auto_pb = np.fromiter((p[1] for p in _auto_pairs), dtype=np.int64, count=len(_auto_pairs))
                _auto_ky = int(np.asarray(freqs_y).shape[0])
                _auto_bias = pairwise_mm_joint_bias(data, _auto_pa, _auto_pb, nbins, _auto_ky)
                # UNDER-SAMPLE GUARD (bias-variance, 2026-06-13): the MM bias (k_joint-1)(k_y-1)/2n is
                # only a reliable correction when the joint table is adequately occupied. At tiny n the
                # bias is large/noisy and over-tightening the prevalence gate feeds the synergy-rescue
                # path, which can ADMIT worse features (measured: F2 n=2500 0.917 -> 1.079, but n=8000
                # unchanged). So skip the debias (bias -> 0, raw pair_mi) for any pair whose rows-per-
                # occupied-joint-cell falls below ``fe_confirm_undersample_rows_per_cell`` (default 5),
                # mirroring the existing CMI-fallback rule. This FORGOES the (unreliable) tiny-n win
                # rather than risk the tiny-n harm -- the large-n win (bilinear n=8000 0.195 -> 0.052)
                # is preserved because there rows-per-cell clears the floor. k_joint is recovered from
                # the bias: k_joint = 1 + bias*2n/(k_y-1).
                _auto_n = int(data.shape[0])
                _auto_min_rpc = float(getattr(self, "fe_confirm_undersample_rows_per_cell", 5.0))
                for _api, _apr in enumerate(_auto_pairs):
                    _b = float(_auto_bias[_api])
                    if _b > 0.0 and _auto_ky > 1:
                        _kj = 1.0 + (_b * 2.0 * _auto_n) / float(_auto_ky - 1)
                        _rpc = _auto_n / max(1.0, _kj * _auto_ky)
                        if _rpc < _auto_min_rpc:
                            _b = 0.0  # under-sampled joint -> unreliable bias -> use raw pair_mi
                    _pair_mm_bias[tuple(sorted(_apr))] = _b
        except Exception:
            _prevalence_debias_auto = False

    return numeric_vars_to_consider, _eng_cap, _pair_maxt_floor, _pair_mm_bias, _prevalence_debias_auto
