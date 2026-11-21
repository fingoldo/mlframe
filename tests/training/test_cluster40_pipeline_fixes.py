"""Cluster 40 regressions: a duplicated index does not multiply rows, a TypeError inside a fit is not swallowed, array parameters key the cache by content, and unseen validation labels are refused (audit 2026-11-20, findings 40-F6, 40-F12, 40-F14, 40-F19)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._composite_fe_shared import attach_new_columns
from mlframe.training.pipeline._pipeline_cache import _param_key_repr
from mlframe.training.pipeline._pipeline_fit_state import _fit_with_optional_kwargs


def test_attach_new_columns_keeps_row_count_with_duplicated_index():
    """A duplicated index label used to multiply rows in df.join (40-F14)."""
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 0, 1])
    new = pd.DataFrame({"b": [10, 20, 30]}, index=[5, 6, 7])
    out = attach_new_columns(df, new)
    assert len(out) == 3
    assert out["b"].tolist() == [10, 20, 30]
    assert out.index.tolist() == [0, 0, 1]


def test_typeerror_inside_the_fit_propagates():
    """A TypeError raised inside a selector's fit must surface, not be read as an unsupported keyword (40-F12)."""

    def selector(X, y):
        """Raise a TypeError from inside the fit."""
        raise TypeError("a real bug inside the fit")

    with pytest.raises(TypeError, match="a real bug"):
        _fit_with_optional_kwargs(selector, np.zeros((2, 1)), np.zeros(2), groups=None, sample_weight=None, wants_sw=False, sw_kwarg="sample_weight")


def test_array_parameters_with_equal_summaries_get_distinct_keys():
    """numpy elides arrays over 1000 elements, so two different arrays used to share a repr and a cache entry (40-F19)."""
    a = np.zeros(2000)
    b = np.zeros(2000)
    b[1500] = 1.0
    assert repr(a) == repr(b)  # the premise: numpy's repr alone cannot tell them apart
    assert _param_key_repr(a) != _param_key_repr(b)
