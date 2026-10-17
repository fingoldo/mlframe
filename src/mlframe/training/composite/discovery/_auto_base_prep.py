"""Prep and result-assembly stages carved out of ``_auto_base._auto_base`` (function-length gate)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .screening import _sample_indices

# Same logger name as ``_auto_base`` so log filtering by module is unchanged.
logger = logging.getLogger(__name__.replace("_auto_base_prep", "_auto_base"))

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery


def resolve_hint_kept(self: "CompositeTargetDiscovery", usable_features: Sequence[str], top_k: int) -> list[str]:
    """Hint features that survive filtering, capped to half the slots unless the ablation strength signal is strong."""
    # Hint-aware ranking: BaselineDiagnostics ablation already
    # measured each feature's predictive contribution directly
    # (drop feature -> RMSE delta). That signal beats pairwise
    # MI(y, x), which gets fooled by features with global trend
    # but no structural residual signal (spatial coords on
    # geographically-trended y is the canonical case). When a
    # hint is provided, prepend hint features (preserving order)
    # then fill remaining slots with MI-ranked features.
    usable_set = set(usable_features)
    hint_raw = list(getattr(self.config, "dominant_features_hint", None) or [])
    hint_kept: list[str] = []
    hint_dropped: list[str] = []
    for c in hint_raw:
        if c in usable_set and c not in hint_kept:
            hint_kept.append(c)
        else:
            hint_dropped.append(c)
    if hint_dropped:
        logger.info(
            "[CompositeTargetDiscovery] dominant_features_hint dropped " "%d entries (filtered or not in feature_cols): %s",
            len(hint_dropped),
            hint_dropped[:5],
        )
    # Adaptive hint cap. Previous fixed cap of
    # ``max(1, top_k // 2)`` was too aggressive when BD ablation
    # confidently identified the dominant base (e.g. delta% > 100%
    # for the top-1 feature in a prod incident). Now: if the user
    # supplied a hint AND we have ablation strengths in metadata,
    # check the strength signal. Strong hint (top-1 delta_pct >
    # ``hint_strength_threshold_pct``, default 50%) -> use FULL
    # hint (no cap). Weak/absent strength info -> fall back to the
    # half-slot cap so MI-leaders still get evaluated.
    #
    # Rationale: BD ablation directly measures "drop feature -> RMSE
    # delta%" which is a high-quality signal. When it screams +501%
    # for a lag feature (real production case), trust it; don't dilute
    # with MI-leaders that may be lower-quality features.
    strong_hint_threshold = float(getattr(
        self.config, "hint_strength_threshold_pct", 50.0,
    ))
    # Strength info is plumbed via the suite-level hint precompute
    # at core.py and stored on the discovery instance for this fit.
    # Absent = treat as unknown strength -> use half-slot cap.
    hint_strengths = getattr(self, "_hint_strengths_pct", None)
    # hint_raw and hint_strengths are positionally aligned; hint_kept drops filtered entries, so realign strengths to the surviving hints before taking the max -- otherwise a dropped strong hint's strength leaks onto a surviving weak one.
    _kept_set = set(hint_kept)
    _aligned = [s for f, s in zip(hint_raw, (hint_strengths or [])) if f in _kept_set]
    is_strong_hint = hint_strengths is not None and len(_aligned) > 0 and max(_aligned) >= strong_hint_threshold
    if is_strong_hint:
        # Full hint -- no cap. Log so it's auditable.
        logger.info(
            "[CompositeTargetDiscovery] auto-base using FULL hint "
            "(%d candidates, max ablation delta%% = %.1f%% >= %.1f%% "
            "threshold; trusting BD over MI ranking).",
            len(hint_kept), max(_aligned),
            strong_hint_threshold,
        )
        hint_cap = top_k  # effectively no cap
    else:
        hint_cap = max(1, top_k // 2)
        if len(hint_kept) > hint_cap:
            logger.info(
                "[CompositeTargetDiscovery] auto-base capping hint "
                "contribution to %d/%d slots (was %d hint candidates; "
                "strength signal weak or absent) so MI-leaders also "
                "get evaluated; full hint list preserved as feature "
                "ordering source.",
                hint_cap, top_k, len(hint_kept),
            )
            hint_kept = hint_kept[:hint_cap]
    return hint_kept


def screen_sample_matrix(
    self: "CompositeTargetDiscovery", df: Any, usable_features: Sequence[str], y_train: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the screening sample and build its feature matrix; returns ``(y_screen, x_matrix)``."""
    sample_idx = _sample_indices(
        train_idx.size, self.config.mi_sample_n, self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", 'stratified_quantile'),
        y=y_train,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    train_idx_screen = train_idx[sample_idx]
    y_screen = y_train[sample_idx]

    x_matrix = self._build_feature_matrix(df, usable_features, train_idx_screen)
    # ``fit`` draws the same screening sample and gathers the same columns again; hand it this (never mutated: the steps
    # below reassign) matrix so the second full-frame gather is skipped when frame, columns and rows all match.
    self._screen_matrix_stash = (df, tuple(usable_features), np.asarray(train_idx_screen).copy(), x_matrix)
    return y_screen, x_matrix


def finite_mask_prep(
    self: "CompositeTargetDiscovery", usable_features: Sequence[str], x_matrix: np.ndarray, y_screen: np.ndarray, hint_kept: list[str], top_k: int
) -> tuple[list[str] | None, Sequence[str], np.ndarray, np.ndarray, bool, np.ndarray]:
    """Drop sparse columns and build the finite-row mask.

    Returns ``(early_result, usable_features, x_matrix, x_matrix_for_mi, use_per_pair, finite)``; ``early_result`` is the
    feature-list-order fallback when too few finite rows remain, else ``None``.
    """
    # Drop columns with ZERO observed values in the screening sample
    # BEFORE the all-row finite-mask. A single fully-NaN column made
    # the AND-mask return zero rows (observed in prod: 'auto-base:
    # only 0 finite rows in screening sample') AND every downstream
    # sklearn.SimpleImputer call sprayed UserWarnings about feature
    # index N. Per-column NaN tolerance: keep features with at least
    # ``_AUTO_BASE_MIN_FRACTION_FINITE`` non-NaN cells in the
    # screening sample; impute the rest per-column with the column
    # mean before the all-row finite mask.
    _MIN_FRAC_FINITE = 0.10  # at least 10% non-NaN cells
    _col_finite_frac = np.isfinite(x_matrix).mean(axis=0)
    _keep_cols = _col_finite_frac >= _MIN_FRAC_FINITE
    usable_features, x_matrix = _auto_base_step1_mean_before_all(_keep_cols, usable_features, _MIN_FRAC_FINITE, x_matrix)
    # The MI RANKING must be estimated with
    # PER-PAIR (per-column) finite masking, NOT the global all-column
    # ``np.all(isfinite(x_matrix), axis=1)`` intersection. For mid-range-NaN
    # columns the intersection is a non-random (MNAR) subset -- the rows
    # where EVERY feature happens to be observed -- so MI(y, x_j) estimated
    # on it is biased by the joint-observability pattern and silently shifts
    # which base wins. Per-pair masking estimates each column's MI on its
    # own observed rows (mirroring ``_mi_to_target`` / the prebinned
    # ``-1``-sentinel path) and is bit-identical when nothing is NaN.
    #
    # Capture the pristine (pre-impute) screening matrix for the MI ranking:
    # the legacy ``< 50``-global-finite fallback below imputes NaNs in
    # ``x_matrix`` (used by the demoters / dedup, which need a SHARED row
    # basis for pairwise correlations), but the MI ranking must see the real
    # NaNs so per-pair masking can drop only the truly-missing rows. The
    # impute reassigns ``x_matrix`` to a fresh ``np.where`` array, so this
    # reference keeps pointing at the original NaN-bearing screening sample.
    x_matrix_for_mi = x_matrix
    use_per_pair = bool(getattr(self.config, "auto_base_mi_per_pair_mask", True))
    finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
    # Audit the MNAR severity once: how much of the per-pair-available row
    # mass the global intersection discards. Below the configured fraction
    # the global mask would meaningfully under-sample some column, which is
    # exactly the per-pair masking case; log it so the divergence is auditable.
    _mnar_threshold = float(getattr(
        self.config, "auto_base_mnar_per_pair_threshold", 0.5,
    ))
    _n_global = int(finite.sum())
    _per_col_finite = np.isfinite(x_matrix) & np.isfinite(y_screen)[:, None]
    _per_col_counts = _per_col_finite.sum(axis=0)
    _max_per_col = int(_per_col_counts.max()) if _per_col_counts.size else 0
    if use_per_pair and _max_per_col > 0 and _n_global < _mnar_threshold * _max_per_col:
        logger.info(
            "[CompositeTargetDiscovery] auto-base: global all-column finite "
            "mask keeps %d row(s) vs %d per-pair-available (%.0f%%, below "
            "%.0f%% MNAR threshold); ranking by PER-PAIR MI to avoid "
            "selection bias.",
            _n_global, _max_per_col,
            100.0 * _n_global / _max_per_col, 100.0 * _mnar_threshold,
        )
    if finite.sum() < 50:
        # Even after the per-column drop, the all-row finite mask can
        # still be too tight when many features have sparse but
        # non-zero NaN density. Impute remaining NaNs with per-column
        # mean and proceed (correlation-quality features survive; truly
        # bad columns were already dropped above). NOTE: this only feeds
        # the demoters / dedup correlations; the MI ranking already uses
        # ``x_matrix_for_mi`` (pristine + per-pair masked) above.
        _col_means = np.nanmean(x_matrix, axis=0)
        _col_means = np.where(np.isfinite(_col_means), _col_means, 0.0)
        _nan_mask = ~np.isfinite(x_matrix)
        if _nan_mask.any():
            x_matrix = np.where(_nan_mask, _col_means[None, :], x_matrix)
        finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
        if finite.sum() < 50:
            logger.warning(
                "[CompositeTargetDiscovery] auto-base: only %d finite rows "
                "even after per-column NaN drop + mean impute; falling back "
                "to feature-list order.", int(finite.sum()),
            )
            # Keep hint features at the front: BD ablation is the one signal not derived from the broken screening sample, and a hint may not fall within the first top_k of the arbitrary column order.
            return (hint_kept + [c for c in usable_features if c not in hint_kept])[:top_k], usable_features, x_matrix, x_matrix_for_mi, use_per_pair, finite
    return None, usable_features, x_matrix, x_matrix_for_mi, use_per_pair, finite


def combine_hint_and_ranked(ranked: list, hint_kept: list[str], top_k: int) -> list[str]:
    """Hint candidates take the leading slots, MI-ranked candidates fill the rest up to ``top_k``."""
    # Combine hint (priority) + MI-ranked tail. Hint always wins
    # the leading slots; MI fills up to auto_base_top_k.
    if hint_kept:
        mi_tail: list[str] = []
        for _, c in ranked:
            if c in hint_kept:
                continue
            mi_tail.append(c)
            if len(hint_kept) + len(mi_tail) >= top_k:
                break
        top = hint_kept + mi_tail
        top = top[:top_k]
        mi_lookup = {c: mi for mi, c in ranked}
        scores = ", ".join(f"{c}={mi_lookup.get(c, float('nan')):.4f}{'(hint)' if c in hint_kept else ''}" for c in top)
        logger.info(
            "[CompositeTargetDiscovery] auto-base top-%d (%d hint, %d MI): %s",
            len(top), len(hint_kept), len(mi_tail), scores,
        )
        return top

    top = [c for _, c in ranked[:top_k]]
    if top:
        scores = ", ".join(f"{c}={mi:.4f}" for mi, c in ranked[:top_k])
        logger.info(
            "[CompositeTargetDiscovery] auto-base top-%d by MI(y, x): %s",
            len(top), scores,
        )
    return top


def _auto_base_step1_mean_before_all(_keep_cols, usable_features, _MIN_FRAC_FINITE, x_matrix):
    """Step 1 of _auto_base: lines starting at ``if not _keep_cols.all():``."""
    if not _keep_cols.all():
        _dropped = [usable_features[i] for i, k in enumerate(_keep_cols.tolist()) if not k]
        logger.info(
            "[CompositeTargetDiscovery] auto-base: dropping %d feature(s) "
            "with <%.0f%% finite cells in screening sample: %s",
            len(_dropped), _MIN_FRAC_FINITE * 100, _dropped[:10],
        )
        x_matrix = x_matrix[:, _keep_cols]
        usable_features = [f for f, k in zip(usable_features, _keep_cols.tolist()) if k]
    return usable_features, x_matrix
