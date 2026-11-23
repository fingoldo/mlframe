"""Cluster 22 regression: PdOrdinalEncoder still refuses NaN codes after the copy-free check (audit 2026-11-20, finding 22-F36)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

from mlframe.estimators.custom import PdOrdinalEncoder


def test_nan_codes_from_parent_are_refused(monkeypatch):
    """If the parent encoder yields NaN codes, the transform refuses them instead of casting to int32 (22-F36)."""
    enc = PdOrdinalEncoder(encoded_missing_value=np.nan)
    monkeypatch.setattr(OrdinalEncoder, "transform", lambda self, X: np.array([[0.0], [np.nan]]))
    with pytest.raises(ValueError, match="NaN codes"):
        enc.transform(pd.DataFrame({"a": ["x", "y"]}))


def test_clean_codes_pass_through():
    """Control: ordinary codes come back as int32 (22-F36)."""
    enc = PdOrdinalEncoder()
    enc.fit(pd.DataFrame({"a": ["x", "y", "x"]}))
    out = enc.transform(pd.DataFrame({"a": ["y", "x"]}))
    assert np.asarray(out).dtype == np.int32
