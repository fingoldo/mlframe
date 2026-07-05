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


def resolve_shap_aware_stage1_keep(
    *, effective_prefilter_top: int, stage1_cushion: int, stage1_floor: int,
    default_stage1_keep: int,
) -> int:
    """Resolve the two_stage prefilter's stage-A survivor count when the SHAP-aware lever is active.

    When ``effective_prefilter_top`` is much smaller than the legacy default (``min(2000,
    0.2*n_features)``), the stage-B booster's column-budget is dictated by ``effective_prefilter_top``
    anyway (stage B narrows ``stage1_keep -> effective_prefilter_top`` by importance) -- so keeping
    stage A at 2000 just makes the stage-B booster fit on 2000 columns when 88 * cushion would
    suffice. Iter33 measurement at C2 (width=10000, n_rows=5000) attributed 57.6% of fit-wall to the
    prefilter stage and cProfile pinned 14.8s of that to the stage-B XGBoost ``update`` on a
    2000-column matrix. With ``effective_prefilter_top=88`` and cushion 8 the stage-A funnel narrows
    to ``max(200, 88*8) = 704`` and the booster fit shrinks ~2.5-3x.

    Returns ``min(default_stage1_keep, max(stage1_floor, effective_prefilter_top * stage1_cushion))``
    so the lever is a strict TIGHTENING (never widens beyond the legacy default, never below the
    floor, never below the eventual output ``effective_prefilter_top``).

    ``stage1_cushion`` (default 2): per-survivor headroom over the eventual stage-B output.
    Iter76 measured cushion 8 -> 2 at C1/C2/C3 (width 5000-10000): prefilter wall 3.0-4.0x faster,
    e2e 1.42-1.58x faster, recall preserved at C1/C2 and +1 at C3. The 2x headroom is the empirical
    minimum that survives stage A's univariate F-rank for marginal-signal informatives that the
    stage-B interaction-aware booster then recovers; below 2x the floor=200 starts to dominate.
    ``stage1_floor`` (default 200): absolute lower bound on stage A's survivor count regardless of
    how small the SHAP cap is. Keeps a generous cohort for stage B's interaction-aware booster on
    pathological tight ``brute_force_max_features`` configurations.
    """
    if effective_prefilter_top is None:
        return int(default_stage1_keep)
    tightened = max(int(stage1_floor), int(effective_prefilter_top) * int(stage1_cushion), int(effective_prefilter_top))
    return min(int(default_stage1_keep), tightened)


__all__ = ["resolve_shap_prefilter_top", "resolve_shap_aware_stage1_keep"]
