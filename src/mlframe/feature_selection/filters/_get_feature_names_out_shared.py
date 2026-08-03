"""Shared ``get_feature_names_out`` sklearn-transformer-contract implementation for selectors that
expose an integer ``support_`` index array into ``feature_names_in_``: independently duplicated
across ``stability.py`` / ``group_aware.py``, consolidated here so a fix can't silently drift out of
sync across copies.
"""
from __future__ import annotations

import numpy as np


def get_feature_names_out_support_based(self, input_features=None):
    """Selected feature names (sklearn transformer contract). ``support_`` is an integer index array
    into ``feature_names_in_``; a passed ``input_features`` must match ``n_features_in_`` (sklearn
    column-drift contract) and, when correct-length, overrides the stored names."""
    names = getattr(self, "feature_names_in_", None)
    if input_features is not None:
        input_features = list(input_features)
        n_in = int(getattr(self, "n_features_in_", len(input_features)))
        if len(input_features) != n_in:
            raise ValueError(
                f"input_features has {len(input_features)} elements, expected {n_in} "
                f"(n_features_in_); names passed to get_feature_names_out must match the "
                f"feature set this selector was fit on (sklearn column-drift contract)."
            )
        return np.asarray([input_features[int(i)] for i in self.support_], dtype=object)
    if names is not None:
        return np.asarray([names[int(i)] for i in self.support_], dtype=object)
    return np.asarray([f"f_{int(i)}" for i in self.support_], dtype=object)
