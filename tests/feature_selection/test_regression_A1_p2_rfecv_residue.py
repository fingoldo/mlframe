"""Regression sensors for A1 P2 RFECV residue: A1#9, A1#12, A1#13.

* A1#9: sample_weight length must be validated against y, not just X.
* A1#12: polars to_pandas(self_destruct=True) must NOT fire on caller-owned frames.
* A1#13: _x_hash signature must use full-content blake2b (no strided collisions).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


def _toy_xy(n=40, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] + 0.5 * X["f1"] > 0).astype(int), name="y")
    return X, y


def test_A1_9_sample_weight_y_length_mismatch_raises():
    X, y = _toy_xy(n=40)
    sw_bad = np.ones(len(y) + 5, dtype=np.float64)  # mismatch vs both X and y
    sel = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    with pytest.raises(ValueError, match="sample_weight"):
        sel.fit(X, y, sample_weight=sw_bad)


def test_A1_9_sample_weight_y_only_mismatch_raises():
    """When X-length matches but y-length is different (e.g. multi-column y coerced row-by-row),
    the mismatch must surface at fit entry, not deep in the per-fold slice.
    """
    X, y = _toy_xy(n=40)
    y_short = y.iloc[:35]  # mismatch only vs y
    sw = np.ones(40, dtype=np.float64)  # matches X but not y_short
    sel = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    with pytest.raises(ValueError, match="sample_weight"):
        sel.fit(X, y_short, sample_weight=sw)


def test_A1_12_polars_self_destruct_only_on_internally_owned():
    """Default: caller-owned polars frame is NOT consumed by RFECV (self_destruct gated off)."""
    X_pd, y = _toy_xy(n=40, p=3)
    X_pl = pl.from_pandas(X_pd)
    sel = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    sel.fit(X_pl, y)
    # After fit on caller-owned polars frame, the polars frame must still be usable. polars
    # consumed frames raise on .shape or operations; un-consumed frames retain their data.
    try:
        _shape = X_pl.shape
        _ = X_pl.head(2)
        consumed = False
    except Exception:
        consumed = True
    assert not consumed, "caller-owned polars frame must NOT be destroyed by RFECV self_destruct"


def test_A1_12_polars_self_destruct_opted_in_by_marker():
    """When the caller sets the opt-in marker, RFECV may consume the frame for the peak-RAM win."""
    X_pd, y = _toy_xy(n=40, p=3)
    X_pl = pl.from_pandas(X_pd)
    sel = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    sel._rfecv_owns_polars_frame_ = True
    sel.fit(X_pl, y)
    assert hasattr(sel, "support_")


def test_A1_13_x_hash_full_content_no_collision_after_outlier_clip():
    """Two X frames that differ only at non-strided positions must produce DIFFERENT signatures
    so the skip-retrain shortcut cannot replay the wrong fit."""
    rng = np.random.default_rng(0)
    n, p = 200, 5
    X1 = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    X2 = X1.copy()
    # Mutate a position that the strided 1024 sample would have missed in the legacy code:
    # any middle index that isn't on the 1024-stride grid for n=200 is hit (since stride <1).
    # Here pick a single off-stride cell -- with full-content hash, ANY differing cell flips the digest.
    X2.iloc[37, 2] = X1.iloc[37, 2] + 1e3
    y = pd.Series((X1["f0"] > 0).astype(int))
    sel1 = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    sel1.fit(X1, y)
    sig1 = sel1.signature
    sel2 = RFECV(estimator=LogisticRegression(max_iter=50), cv=3, max_nfeatures=2)
    sel2.fit(X2, y)
    sig2 = sel2.signature
    assert sig1 != sig2, "full-content X hash must differ for X1 vs X2 (one cell changed)"
