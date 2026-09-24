"""Putting back columns the MRMR screen dropped, when they still earn their place.

Two families are screened out for reasons that are artefacts of how the screen measures rather than of the column's worth. A single adaptive
Fourier leg carries little marginal MI because the phase is split across its sin and cos twin, and a clean ``is_missing__`` indicator ties with
the raw NaN-bin source that encodes the same pattern, so the greedy keeps the raw column and drops the indicator the model can actually use.

Neither is a reason to re-add unconditionally. Whether a dropped column is a rescue or a regression depends on the design that survived, so
both passes ask the same held-out question every sibling protection in this package asks.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._heldout_gate import (
    build_heldout_incr_probe,
    candidate_values,
    coerce_gate_target,
    selected_design_columns,
)

logger = logging.getLogger(__name__)

# Held-out R^2 a protection pass must buy before it puts a screened-out column back into support. Matches the hinge protection's floor in
# _group2: these are the same question asked of different column families, so they answer to the same bar.
_ADAPTIVE_FOURIER_PROTECT_MIN_INCR_R2 = 0.003
_MISS_INDICATOR_PROTECT_MIN_INCR_R2 = 0.003


def readd_protected_columns(
    self,
    *,
    X,
    cols,
    data,
    selected_vars,
    _eng_continuous_snapshot,
    _y_np,
    hybrid_orth_pre_recipes,
    miss_ind_pre_recipes,
    verbose,
):
    """Re-add the adaptive-Fourier legs and missingness indicators that still lift a held-out fit over the selected design.

    Every candidate this leaves out is recorded on ``self.protection_readd_rejections_`` with the gain it measured and the
    bar it missed. The decision used to exist only as a verbose-gated log line, so a fitted estimator could not say whether
    a family's empty roster meant it produced nothing or produced a column the held-out fit showed to be redundant.
    """
    self.protection_readd_rejections_ = []

    # ADAPTIVE-FOURIER PROTECTION: re-add held-out-validated
    # ADAPTIVE Fourier columns the MRMR screen dropped. The adaptive detector
    # already confirmed the column's dominant frequency on a held-out slice;
    # the screen drops it anyway because a SINGLE sin OR cos has low marginal MI
    # (the phase is split across the two legs, so neither alone clears the
    # relevance floor and the screen prefers a lower-MI fixed-freq twin). We
    # re-add the index of every adaptive name that is a column in ``cols`` but
    # absent from ``selected_vars``; its recipe is already in
    # ``engineered_recipes`` (merged from ``hybrid_orth_pre_recipes`` above)
    # and survives into ``self._engineered_recipes_`` via the remap below, so
    # transform() replays the fit-time column byte-for-byte. Runs BEFORE the
    # ``selected_vars_names`` remap so the re-added index is routed correctly.
    _adaptive_fourier = getattr(self, "_adaptive_fourier_features_", None)
    if _adaptive_fourier and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _y_gate_af = coerce_gate_target(_y_np, data.shape[0])
        _af_probe = build_heldout_incr_probe(
            y_gate=_y_gate_af,
            sel_value_cols=selected_design_columns(
                X=X, cols=cols, selected_vars=selected_vars, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_af
            ),
            random_seed=getattr(self, "random_seed", 0),
        )
        # The legs of one adaptive frequency are judged TOGETHER, not one at a time: the phase is split across the sin and cos leg, so each
        # leg's individual marginal is low by construction (which is why the screen dropped them) and a per-leg gate would reject the very
        # pair this protection exists to rescue. Their JOINT lift over the selected design is the honest question, and a pair already
        # subsumed by a surviving composite adds ~0 to it.
        _adaptive_by_source: dict = {}
        for _an in _adaptive_fourier:
            _idx = _cols_index.get(_an)
            if _idx is None or _idx in _sv_set:
                continue
            _rec_af = hybrid_orth_pre_recipes.get(_an)
            _src_af = tuple(getattr(_rec_af, "src_names", ()) or ())
            _adaptive_by_source.setdefault(_src_af[0] if _src_af else _an, []).append((_an, _idx))
        _readd_adaptive = []
        for _src_name_af, _legs_af in _adaptive_by_source.items():
            _leg_vals_af = [candidate_values(_ln, X=X, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_af) for _ln, _ in _legs_af]
            if _y_gate_af is None or any(_v is None for _v in _leg_vals_af):
                # Unmeasurable: no usable target, or a leg whose values cannot be scored. Keep the pre-gate behaviour rather than dropping blind.
                _incr_af = float("inf")
            else:
                _incr_af = _af_probe(
                    np.column_stack(_leg_vals_af),
                    candidate_values(_src_name_af, X=X, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_af),
                )
            if _incr_af < _ADAPTIVE_FOURIER_PROTECT_MIN_INCR_R2:
                self.protection_readd_rejections_.append({
                    "pass": "adaptive_fourier",
                    "source": str(_src_name_af),
                    "columns": [str(_ln) for _ln, _ in _legs_af],
                    "heldout_r2_gain": float(_incr_af),
                    "min_gain": float(_ADAPTIVE_FOURIER_PROTECT_MIN_INCR_R2),
                })
                if verbose:
                    logger.info(
                        "MRMR adaptive-fourier protection: leaving %d leg(s) of %r out of support, held-out R^2 gain over the selected design %.5f < %.5f: %s",
                        len(_legs_af),
                        _src_name_af,
                        _incr_af,
                        _ADAPTIVE_FOURIER_PROTECT_MIN_INCR_R2,
                        [_ln for _ln, _ in _legs_af],
                    )
                continue
            for _ln, _idx in _legs_af:
                _readd_adaptive.append(_idx)
                _sv_set.add(_idx)
        if _readd_adaptive:
            selected_vars = list(selected_vars) + _readd_adaptive
            if verbose:
                logger.info(
                    "MRMR adaptive-fourier protection: re-added %d adaptive Fourier feature(s) the screen dropped, each lifting a held-out fit over the selected design: %s",
                    len(_readd_adaptive),
                    [cols[i] for i in _readd_adaptive],
                )

    # MISSINGNESS-INDICATOR PROTECTION: re-add the clean ``is_missing__{col}`` indicator the MRMR screen dropped IN FAVOUR OF its raw source. Under ``nan_strategy='separate_bin'``
    # the raw column's NaN bin already encodes the MNAR pattern, so the binned MI of the indicator and the raw source are near-identical (a true tie); the greedy screen keeps the raw column
    # and discards the indicator as redundant. But the raw column is mostly NaN - the downstream model cannot consume the missingness signal from it, only from the standalone numeric
    # indicator (the whole point of Layer 37). When the raw source IS selected, the indicator carries the SAME signal in a clean, model-ready form, so we re-add it. Gating on "the raw source
    # survived the screen" keeps a pure-noise indicator (MAR column the screen never selects) out of support. The count / pattern encoders have no single raw source and are screened normally.
    _miss_indicators = list(getattr(self, "missingness_indicator_features_", None) or [])
    if _miss_indicators and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _y_gate_mi = coerce_gate_target(_y_np, data.shape[0])
        _miss_probe = build_heldout_incr_probe(
            y_gate=_y_gate_mi,
            sel_value_cols=selected_design_columns(
                X=X, cols=cols, selected_vars=selected_vars, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_mi
            ),
            random_seed=getattr(self, "random_seed", 0),
        )
        _readd_miss = []
        for _mn in _miss_indicators:
            _idx = _cols_index.get(_mn)
            if _idx is None or _idx in _sv_set:
                continue
            _rec_mi = miss_ind_pre_recipes.get(_mn)
            _src_mi = tuple(getattr(_rec_mi, "src_names", ()) or ())
            # Re-add only when the indicator's raw source survived the screen (i.e. the signal is real and the screen kept the redundant raw twin
            # in its place) AND the indicator still lifts a held-out fit over the design we actually kept: "the source survived" is a membership
            # test, and on a multi-signal frame a surviving composite can already carry the pattern the indicator encodes.
            if _src_mi and _src_mi[0] in _sel_names_now:
                _mi_vals = candidate_values(_mn, X=X, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_mi)
                if _mi_vals is None:
                    _incr_mi = float("inf")  # unmeasurable values leave the pre-gate behaviour in place rather than dropping blind
                else:
                    _incr_mi = _miss_probe(_mi_vals, candidate_values(_src_mi[0], X=X, eng_continuous_snapshot=_eng_continuous_snapshot, y_ref=_y_gate_mi))
                if _incr_mi < _MISS_INDICATOR_PROTECT_MIN_INCR_R2:
                    self.protection_readd_rejections_.append({
                        "pass": "missingness_indicator",
                        "source": str(_src_mi[0]),
                        "columns": [str(_mn)],
                        "heldout_r2_gain": float(_incr_mi),
                        "min_gain": float(_MISS_INDICATOR_PROTECT_MIN_INCR_R2),
                    })
                    if verbose:
                        logger.info(
                            "MRMR missingness-indicator protection: leaving %r out of support, held-out R^2 gain over the selected design %.5f < %.5f",
                            _mn,
                            _incr_mi,
                            _MISS_INDICATOR_PROTECT_MIN_INCR_R2,
                        )
                    continue
                _readd_miss.append(_idx)
                _sv_set.add(_idx)
        if _readd_miss:
            selected_vars = list(selected_vars) + _readd_miss
            if verbose:
                logger.info(
                    "MRMR missingness-indicator protection: re-added %d clean " "is_missing__ indicator(s) the screen dropped in favour of " "the redundant raw NaN-bin source: %s",
                    len(_readd_miss),
                    [cols[i] for i in _readd_miss],
                )

    return selected_vars
