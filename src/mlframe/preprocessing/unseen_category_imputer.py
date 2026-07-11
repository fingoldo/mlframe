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
"""
from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


class UnseenCategoryImputer:
    """Fit a per-column train category vocabulary; at transform time, map unseen or below-``min_count``
    categories to that column's train-mode (most frequent train category).

    Parameters
    ----------
    columns
        Categorical columns to fit/transform.
    min_count
        Train categories occurring fewer than this many times are treated as unreliable and also mapped to
        the mode (in addition to genuinely unseen categories). Default ``1`` (only genuinely-unseen values).
    """

    def __init__(self, columns: Sequence[str], min_count: int = 1) -> None:
        self.columns = list(columns)
        self.min_count = min_count
        self.known_categories_: Dict[str, set] = {}
        self.mode_: Dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "UnseenCategoryImputer":
        for col in self.columns:
            counts = df[col].value_counts()
            reliable = counts[counts >= self.min_count]
            if len(reliable) == 0:
                reliable = counts
            self.known_categories_[col] = set(reliable.index)
            self.mode_[col] = reliable.index[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy(deep=False)
        for col in self.columns:
            known = self.known_categories_.get(col)
            if known is None:
                continue
            unreliable_mask = ~df[col].isin(known)
            if unreliable_mask.any():
                out[col] = df[col].where(~unreliable_mask, self.mode_[col])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


__all__ = ["UnseenCategoryImputer"]
