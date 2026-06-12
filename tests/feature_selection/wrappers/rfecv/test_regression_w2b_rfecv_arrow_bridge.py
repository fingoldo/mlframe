"""Regression sensor for w2b-percol-scattered _rfecv.py transform Arrow-bridge alignment (finding #34).

The sibling _rfecv_fit.py already uses use_pyarrow_extension_array=True / split_blocks=True / self_destruct=True; _rfecv.py:716 was the lone
bare ``X.to_pandas()`` call. Sensor: transforming polars X through a fitted RFECV must not crash and must produce the expected column subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")


def test_rfecv_transform_polars_input_with_arrow_bridge():
    """Construct an RFECV manually with pre-set support_/feature_names_in_ so the test exercises ONLY transform(); we don't need fit()."""
    from sklearn.linear_model import LinearRegression
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    feat_names = [f"f{i}" for i in range(5)]
    rfecv = RFECV(estimator=LinearRegression(), cv=3)
    rfecv.support_ = np.array([True, False, True, False, True])
    rfecv.feature_names_in_ = np.array(feat_names, dtype=object)

    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 5))
    pldf = pl.DataFrame({c: data[:, i] for i, c in enumerate(feat_names)})

    out = rfecv.transform(pldf)
    expected_cols = [c for c, keep in zip(feat_names, rfecv.support_) if keep]
    assert hasattr(out, "columns") or hasattr(out, "shape")
    if hasattr(out, "columns"):
        assert list(out.columns) == expected_cols
    assert out.shape == (40, 3)
