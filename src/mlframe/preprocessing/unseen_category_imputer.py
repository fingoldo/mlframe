"""Replace unseen/rare categorical values at inference time with a known train-vocabulary category.

Source: 9th_home-credit-default-risk.md -- "Replace categories not found in train... replace all of these
with things previously encountered. Test set has no XNA genders..." A never-seen-at-train category (or one
seen too rarely to have a reliable encoding) forces most encoders into a NaN/sentinel-bucket fallback, which
discards the row's signal entirely for that column. Mapping it to the train vocabulary's MODE instead keeps
the row usable with the closest available prior, rather than an "unknown" bucket carrying no information.

Distinct from :func:`mlframe.preprocessing.rare_count_pruning.collapse_rare_categories`, which collapses rare
train-side categories into a constant sentinel bucket (``"__other__"``) that then gets its OWN encoding --
here the replacement is a real, previously-encountered category, so no new (untrained) encoding is needed at
inference time.

``similarity_mode="nearest"`` implements the "nearest" alternative floated above: instead of always falling
back to the single train-mode category, it maps an unseen category to whichever KNOWN category has the
closest average value on a companion numeric column (``value_column``), using that unseen row's OWN
``value_column`` reading as the query point. This beats the mode fallback whenever an unseen category's true
behavior is closer to a non-dominant known category than to the mode -- but it needs a value_column that
carries real signal about the row, so ``mode`` (needing nothing extra) stays the default.

``track_fallback_stats=True`` is an opt-in diagnostic: silent fallback with no visibility is a real production
blind spot -- a column whose unseen-value rate creeps up over time (schema drift, a new category rollout, an
upstream bug) degrades quietly since the imputer always produces *some* known category either way. When
enabled, each ``transform`` call records per-column fallback counts/rate in ``fallback_stats_``, so the caller
can alert when a column's fallback rate crosses a threshold that signals drift worth investigating.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


class UnseenCategoryImputer:
    """Fit a per-column train category vocabulary; at transform time, map unseen or below-``min_count``
    categories to a known train category.

    Parameters
    ----------
    columns
        Categorical columns to fit/transform.
    min_count
        Train categories occurring fewer than this many times are treated as unreliable and also mapped to
        the fallback (in addition to genuinely unseen categories). Default ``1`` (only genuinely-unseen values).
    similarity_mode
        ``"mode"`` (default) maps every unreliable value to the column's train-mode category, matching the
        original behavior exactly. ``"nearest"`` instead maps each unreliable row to the known category whose
        train-time average of ``value_column`` is closest to that row's own ``value_column`` reading -- a
        strictly better proxy when categories vary a lot and the unseen value behaves like a non-dominant
        known category, but it requires ``value_column`` to carry real per-row signal, so it is opt-in.
    value_column
        Companion numeric column (assumed present in both fit and transform frames) used only when
        ``similarity_mode="nearest"``. Ignored for ``"mode"``.
    track_fallback_stats
        If ``True`` (default ``False``), each ``transform`` call records per-column fallback counts in
        ``fallback_stats_`` -- ``{col: {"n_total": int, "n_fallback": int, "fallback_rate": float}}`` -- so the
        caller can monitor how often the fallback actually fires in production and catch category drift (a
        rising fallback rate on a column that was previously stable). Disabled by default since it is a pure
        diagnostic add-on with no effect on the transformed output.
    impute_nan
        ``True`` (default, prior behavior, bit-identical) means a genuine NaN cell is ALSO treated as
        "unreliable" and mode/nearest-imputed -- ``value_counts()`` never puts NaN in ``known_categories_``,
        so every NaN cell falls into the same unreliable-category fallback as an unseen/rare string value.
        Set ``False`` to leave NaN cells untouched (e.g. when a separate downstream imputer or missing-
        indicator step is meant to own NaN handling for this column).
    """

    def __init__(
        self,
        columns: Sequence[str],
        min_count: int = 1,
        similarity_mode: str = "mode",
        value_column: Optional[str] = None,
        track_fallback_stats: bool = False,
        impute_nan: bool = True,
    ) -> None:
        if similarity_mode not in ("mode", "nearest"):
            raise ValueError(f"similarity_mode must be 'mode' or 'nearest', got {similarity_mode!r}")
        if similarity_mode == "nearest" and value_column is None:
            raise ValueError("similarity_mode='nearest' requires value_column to be set")
        self.columns = list(columns)
        self.min_count = min_count
        self.similarity_mode = similarity_mode
        self.value_column = value_column
        self.track_fallback_stats = track_fallback_stats
        self.impute_nan = impute_nan
        self.known_categories_: Dict[str, set] = {}
        self.mode_: Dict[str, object] = {}
        # Per column: known categories sorted by their train-time average value_column, and that sorted average array.
        self.nearest_categories_sorted_: Dict[str, np.ndarray] = {}
        self.nearest_values_sorted_: Dict[str, np.ndarray] = {}
        # Populated by transform() when track_fallback_stats=True; holds the MOST RECENT transform call's stats.
        self.fallback_stats_: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "UnseenCategoryImputer":
        """Learn per-column reliable categories, fallback mode, and (if nearest mode) value-sorted category means."""
        for col in self.columns:
            counts = df[col].value_counts()
            reliable = counts[counts >= self.min_count]
            if len(reliable) == 0:
                reliable = counts
            self.known_categories_[col] = set(reliable.index)
            self.mode_[col] = reliable.index[0]

            if self.similarity_mode == "nearest":
                assert self.value_column is not None
                # groupby over the whole frame then reindex to reliable categories, instead of an isin() row-mask
                # scan of size n up front -- groupby already visits every row exactly once either way.
                cat_means = df.groupby(col, observed=True)[self.value_column].mean().reindex(reliable.index)
                cat_means = cat_means.sort_values()
                self.nearest_categories_sorted_[col] = cat_means.index.to_numpy()
                self.nearest_values_sorted_[col] = cat_means.to_numpy(dtype=float)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace unseen/unreliable categories with the fitted mode or nearest known category by value."""
        out = df.copy(deep=False)
        if self.track_fallback_stats:
            self.fallback_stats_ = {}
        for col in self.columns:
            known = self.known_categories_.get(col)
            if known is None:
                continue
            unreliable_mask = ~df[col].isin(known)
            if not self.impute_nan:
                unreliable_mask &= df[col].notna()
            if self.track_fallback_stats:
                n_total = len(df)
                n_fallback = int(unreliable_mask.sum())
                self.fallback_stats_[col] = {
                    "n_total": n_total,
                    "n_fallback": n_fallback,
                    "fallback_rate": (n_fallback / n_total) if n_total else 0.0,
                }
            if not unreliable_mask.any():
                continue

            if self.similarity_mode == "nearest":
                assert self.value_column is not None
                sorted_vals = self.nearest_values_sorted_[col]
                sorted_cats = self.nearest_categories_sorted_[col]
                query = df.loc[unreliable_mask, self.value_column].to_numpy(dtype=float)
                idx_right = np.searchsorted(sorted_vals, query)
                idx_left = np.clip(idx_right - 1, 0, len(sorted_vals) - 1)
                idx_right = np.clip(idx_right, 0, len(sorted_vals) - 1)
                left_dist = np.abs(sorted_vals[idx_left] - query)
                right_dist = np.abs(sorted_vals[idx_right] - query)
                nearest_idx = np.where(left_dist <= right_dist, idx_left, idx_right)
                replacement = pd.Series(sorted_cats[nearest_idx], index=df.index[unreliable_mask])
                out.loc[unreliable_mask, col] = replacement
            else:
                out[col] = df[col].where(~unreliable_mask, self.mode_[col])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` then immediately transform it."""
        return self.fit(df).transform(df)


__all__ = ["UnseenCategoryImputer"]
