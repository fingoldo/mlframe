"""Regression: an all-empty / all-missing TF-IDF text column must NOT crash.

sklearn ``TfidfVectorizer.fit`` raises ``ValueError: empty vocabulary`` when
every document is empty (all-NaN / all-"" column, common on a tiny inner-CV
fold). The encoder must degrade gracefully to a faithful, fixed-width matrix so
the downstream concat layer keeps a stable schema instead of aborting the fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.handlers import HashingParams, TfidfParams
from mlframe.training.feature_handling.text_encoder import TextColumnEncoder


def _frame(values):
    """Build a single-column polars frame named 't' from the given text values, skipping if polars is absent."""
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"t": values})


def test_tfidf_all_empty_column_does_not_crash():
    """An all-empty/None text column degrades to a 0-width fitted vocab instead of raising, and stays stable on transform."""
    df = _frame(["", "", None, ""])
    enc = TextColumnEncoder("t", TfidfParams(max_features=10))
    out = enc.fit_transform(df)  # pre-fix: ValueError empty vocabulary
    assert out.shape[0] == 4
    # Empty vocab -> width 0, faithfully reported.
    assert out.shape[1] == enc.n_features_out
    assert enc.is_fitted
    # transform on a (non-empty) test frame keeps the fitted 0-width schema.
    df2 = _frame(["hello world", "more text"])
    out2 = enc.transform(df2)
    assert out2.shape == (2, enc.n_features_out)


def test_tfidf_all_empty_then_transform_idempotent():
    """Re-transforming the same all-None frame through a fitted 0-width TF-IDF encoder yields the identical output."""
    df = _frame([None, None])
    enc = TextColumnEncoder("t", TfidfParams(max_features=5))
    out = enc.fit_transform(df)
    assert np.allclose(enc.transform(df).toarray(), out.toarray())


def test_hashing_all_empty_keeps_fixed_width():
    """Hashing encoder keeps its fixed n_features width on an all-empty column, unlike TF-IDF's 0-width degrade."""
    # Hashing never builds a vocab; width must stay == n_features regardless.
    df = _frame(["", None, ""])
    enc = TextColumnEncoder("t", HashingParams(n_features=16))
    out = enc.fit_transform(df)
    assert out.shape == (3, 16)
    assert enc.n_features_out == 16
