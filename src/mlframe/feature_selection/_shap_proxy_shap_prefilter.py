"""SHAP-pre-prefilter: tighten the effective prefilter budget to a SHAP-aware cap.

The OOF-SHAP attribution wall scales near-linearly in column count, but the downstream search only
consumes top-``brute_force_max_features`` by mean |phi|; columns between the search cap and the loose
default ``prefilter_top`` (2000) pay full TreeSHAP cost while contributing nothing to the final pick.

This module resolves the SHAP-aware cap ``shap_prefilter_top`` from the search budget plus a safety
cushion: ``max(brute_force_max_features * safety_factor, shap_prefilter_min_features)``. The selector
then passes ``min(prefilter_top, shap_prefilter_top)`` to ``prefilter_columns`` so the EXISTING
prefilter booster (two_stage's stage B / model / fast_model / gpu_model) already produces the tighter
output -- no second booster fit is paid.

A separate post-clustering booster pass was bench-attempt-rejected 2026-05-28: at width=1000,
n_rows=5000 the extra fit cost ~1.2s while OOF-SHAP savings were also ~1.3s, for a +0.1s wash at
seed=1 (cold-start seed=0 gained 17.6% from the same JIT-warmup amortisation that fooled the lever).
Tightening at the prefilter step amortises into work the pipeline already pays.

The ``safety_factor`` (default 4) guards against the cheap-importance pass dropping a column the
OOF-SHAP would have caught: 4x the search cap leaves OOF-SHAP attribution headroom to surface signal
the prefilter booster's ranking missed. ``min_features`` (default 40) is a floor for small
``brute_force_max_features`` configurations.
"""

from __future__ import annotations


def resolve_shap_prefilter_top(
    *, brute_force_max_features: int, safety_factor: int, min_features: int,
) -> int:
    """Resolve the SHAP-aware prefilter cap from the search budget + cushion.

    Returns ``max(brute_force_max_features * safety_factor, min_features)``. The selector then
    intersects this with the user's ``prefilter_top`` (never expands it) so a tight user setting
    still wins.
    """
    return max(int(brute_force_max_features) * int(safety_factor), int(min_features))


__all__ = ["resolve_shap_prefilter_top"]
