"""Selection-gate and Dynamic Cluster Discovery (DCD) state construction, carved out of
``screen_predictors`` (``_screen_predictors.py``) to keep that file under the 1k-line gate.

Both helpers are pure w.r.t. the caller's loop state: ``build_dcd_state`` reads only its
explicit arguments and returns a fresh state (or ``None``); ``compute_selection_gate`` reads
only its explicit arguments and returns the accept/reject decision plus the MM-corrected gain
used for the relative-floor comparison on the NEXT candidate. Neither mutates caller locals.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _single_int(y) -> int:
    """Defensive fallback for a caller passing a bare int despite the Sequence[int] signature
    (SCREEN_CONFIRM_A-15 fix: factors out 3 verbatim-repeated inline copies)."""
    return int(y[0]) if hasattr(y, "__len__") else int(y)


def build_dcd_state(
    dcd_config: dict | None,
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str] | None,
    y: Sequence[int] | None,
    existing_dcd_state,
    verbose: int,
):
    """Construct the Wave 9 Dynamic Cluster Discovery state from ``dcd_config``, or return
    ``None`` when DCD is not requested / config is missing / init fails (fallback to legacy
    path -- DCD is an opt-in accelerator, never a hard requirement for screening to proceed)."""
    if dcd_config is None or not dcd_config.get("enable", False):
        return None
    try:
        from ._dynamic_cluster_discovery import make_dcd_state
        # Layer 47 (2026-05-31): tau_cluster passes through as-is so
        # the literal ``'auto'`` sentinel reaches make_dcd_state's
        # calibration branch. Numeric values are float()-coerced.
        _raw_tau = dcd_config.get("tau_cluster", 0.7)
        _tau_arg = _raw_tau if isinstance(_raw_tau, str) else float(_raw_tau)
        return make_dcd_state(
            X_raw=dcd_config.get("X_raw"),
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            cols=list(factors_names) if factors_names is not None else None,
            nbins=factors_nbins,
            target_indices=np.asarray(y, dtype=np.int64),
            quantization_method=dcd_config.get("quantization_method", "quantile"),
            quantization_nbins=int(dcd_config.get("quantization_nbins", 10)),
            quantization_dtype=dcd_config.get("quantization_dtype", np.int32),
            tau_cluster=_tau_arg,
            distance=str(dcd_config.get("distance", "su")),
            cluster_size_threshold=int(dcd_config.get("cluster_size_threshold", 4)),
            swap_gain_threshold=float(dcd_config.get("swap_gain_threshold", 0.05)),
            swap_method=str(dcd_config.get("swap_method", "pca_pc1")),
            pairwise_cache_max=int(dcd_config.get("pairwise_cache_max", 50_000)),
            min_cluster_size=int(dcd_config.get("min_cluster_size", 2)),
            max_cluster_size=int(dcd_config.get("max_cluster_size", 12)),
            # Layer 47 (2026-05-31): forward auto-tau calibration knobs.
            tau_calibration_n_pairs=int(dcd_config.get(
                "tau_calibration_n_pairs", 100,
            )),
            tau_calibration_seed=int(dcd_config.get(
                "tau_calibration_seed", 0,
            )),
            existing_state=existing_dcd_state,
        )
    except Exception as _dcd_init_exc:
        # SCREEN_CONFIRM_A-8 fix: always log at debug level (was silent at the
        # library's own verbose=0 default), promoting to warning only when verbose -- a DCD-init failure
        # should never be completely invisible.
        if verbose:
            logger.warning(
                "DCD init failed silently; falling back to legacy path: %r",
                _dcd_init_exc,
            )
        else:
            logger.debug("DCD init failed; falling back to legacy path: %r", _dcd_init_exc, exc_info=True)
        return None


def compute_selection_gate(
    *,
    min_relevance_gain: float,
    interactions_order: int,
    best_candidate,
    best_gain,
    cardinality_bias_correction: bool,
    factors_data: np.ndarray,
    y: Sequence[int] | None,
    factors_nbins: Sequence[int],
    min_relevance_gain_relative_to_first: float,
    selected_vars: list,
    predictors: list,
    fdr_gain_floor: float,
    cached_MIs: dict,
) -> tuple[bool, float]:
    """Decide whether ``best_candidate`` clears the abs/relative/maxT-FDR gain floors.

    Returns ``(passes_gate, best_gain_for_gate)`` where ``best_gain_for_gate`` is the
    Miller-Madow-corrected gain (order-1 candidates only; unchanged for joints) the caller
    stores alongside the accepted predictor for future relative-floor comparisons.
    """
    _abs_floor = min_relevance_gain if interactions_order == 1 else min_relevance_gain ** (1 / (interactions_order + 1))
    # 2026-05-30 Miller-Madow: subtract finite-sample bias from gain at gate. For
    # joint candidates (k-way interactions) use product-of-bin-counts as effective
    # cardinality. Bias = (nbins_x_eff - 1) * (nbins_y - 1) / (2*n). The same
    # correction is applied to the first-selected feature's stored gain so the
    # relative-floor comparison is consistent across cardinalities.
    _best_gain_for_gate = float(best_gain)
    # MM gate only applies to single-feature candidates (interactions_order=1).
    # For joint candidates (order >= 2) the bias (|joint_X|-1)*(|Y|-1)/(2n) grows
    # multiplicatively in component nbins (product), which over-corrects: a 39 x 39
    # joint at n=1500 carries bias 0.51 nats - enough to kill the XOR-product
    # synergy signal even when the joint MI is genuinely informative. The pre-
    # screen filter (cells > 0.5*n) already refuses high-cardinality SINGLE
    # columns before they're combined, so joints with all-safe components are
    # implicitly bounded; explicit MM correction on joints is double-counting.
    if cardinality_bias_correction and best_candidate is not None and interactions_order == 1:
        _n_samples_for_mm = int(factors_data.shape[0])
        _y_idx = _single_int(y)
        _nbins_y = int(factors_nbins[_y_idx])
        _nbins_x_eff = 1
        try:
            for _v in best_candidate:
                _nbins_x_eff *= int(factors_nbins[int(_v)])
            _mm_bias_cand = (_nbins_x_eff - 1) * (_nbins_y - 1) / (2.0 * _n_samples_for_mm)
            _best_gain_for_gate = float(best_gain) - _mm_bias_cand
        except (TypeError, ValueError):
            # best_candidate isn't iterable / contains non-int; skip MM gate
            pass
    # 2026-05-30 diminishing-returns floor: from the SECOND selected feature onward,
    # require corrected best_gain >= MAX(corrected gain over already-selected) *
    # min_relevance_gain_relative_to_first. Using MAX (not just first) is critical when
    # the first-picked feature has a cardinality-inflated raw MI that the Miller-Madow
    # correction collapses (Layer 10 seed=101: user_id raw 0.328 -> corrected 0.088,
    # num_signal_1 raw 0.187 -> corrected 0.185; the corrected MAX over the running
    # set is num_signal_1's 0.185, so the floor at 5% is 0.009 - high enough to exclude
    # both the cardinality-biased user_id residual AND any trailing noise). The absolute
    # floor catches "no signal at all"; the relative floor catches "trailing noise that
    # statistically clears the absolute floor but is 100x smaller than the strongest
    # already-selected signal". 0.0 disables.
    _rel_floor = 0.0
    if min_relevance_gain_relative_to_first and selected_vars and predictors:
        _max_corrected_gain = 0.0
        _n_samples_for_mm = int(factors_data.shape[0]) if cardinality_bias_correction else 0
        _y_idx_for_mm = _single_int(y)
        _nbins_y_for_mm = int(factors_nbins[_y_idx_for_mm]) if cardinality_bias_correction else 0
        for _pred in predictors:
            _g_raw = float(_pred.get("gain", 0.0))
            _p_indices = _pred.get("indices", ())
            if cardinality_bias_correction and len(_p_indices) == 1:
                _p_nbins_eff = 1
                for _v in _p_indices:
                    _p_nbins_eff *= int(factors_nbins[int(_v)])
                _g_corr = _g_raw - (_p_nbins_eff - 1) * (_nbins_y_for_mm - 1) / (2.0 * _n_samples_for_mm)
            else:
                _g_corr = _g_raw
            if _g_corr > _max_corrected_gain:
                _max_corrected_gain = _g_corr
        if _max_corrected_gain > 0.0:
            _rel_floor = _max_corrected_gain * float(min_relevance_gain_relative_to_first)
    # 2026-06-03 maxT permutation-null floor (order-1 single candidates
    # only). CRITICAL: compare the candidate's corrected MARGINAL MI -
    # the exact statistic the null is built on - NOT the Fleuret
    # conditional gain ``_best_gain_for_gate``. Once the genuine signals
    # are selected, the conditional gain of a noise column is dominated by
    # conditioning-bias on the sparse high-dim joint (3 selected x 14-bin
    # features => ~2700 cells at n=1500): the noise's conditional gain
    # inflates to ~2x its marginal MI and clears the abs/rel floors, which
    # is exactly the embedding noise-cloud hijack. The marginal MI carries
    # no such conditioning bias, so flooring it at the maxT chance ceiling
    # cleanly separates the genuine signals (marginal >> floor) from the
    # noise dims (marginal < floor). ``cached_MIs[X]`` is the direct
    # marginal I(X;Y) computed during candidate scoring (no extra cost).
    # The floor is 0.0 for narrow pools (< ``screen_fdr_min_features``), so
    # this is a no-op on the tabular suite; higher-order joints are
    # untouched (they keep their own ``_abs_floor``).
    _fdr_floor_eff = fdr_gain_floor if interactions_order == 1 else 0.0
    _fdr_pass = True
    if _fdr_floor_eff > 0.0 and best_candidate is not None and len(best_candidate) == 1:
        _cand_marg_raw = cached_MIs.get(best_candidate, None)
        if _cand_marg_raw is not None:
            _cand_marg_corr = float(_cand_marg_raw)
            if cardinality_bias_correction:
                _nb_x_fdr = int(factors_nbins[int(best_candidate[0])])
                _y_idx_fdr2 = _single_int(y)
                _nb_y_fdr = int(factors_nbins[_y_idx_fdr2])
                _cand_marg_corr -= (_nb_x_fdr - 1) * (_nb_y_fdr - 1) / (2.0 * int(factors_data.shape[0]))
            _fdr_pass = _cand_marg_corr >= _fdr_floor_eff
    return (_best_gain_for_gate >= _abs_floor and _best_gain_for_gate >= _rel_floor and _fdr_pass), _best_gain_for_gate
