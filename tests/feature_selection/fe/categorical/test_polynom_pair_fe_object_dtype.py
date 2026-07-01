"""Regression: polynomial-pair FE must not crash when the frame has a string column.

A frame with any string/categorical column makes ``X.to_numpy()`` return an *object* ndarray, so even a numeric
operand extracts as an object slice; the Hermite/polynomial basis (``np.isfinite`` / z-score / minmax) then raised
``TypeError: ufunc 'isfinite' not supported`` on that object slice. The numeric-pair guard correctly keeps only
numeric pairs, but the extracted slice was still object-typed. The fix coerces the extracted operands to float64.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_smart_polynom_pair_fe_survives_object_dtype_frame():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    # Polynomial interaction so the (a, b) pair is a genuine prospective polynomial-FE candidate.
    logit = 1.5 * (a * a - b)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    # The string column is what forces X.to_numpy() to object dtype, reproducing the crash condition.
    df = pd.DataFrame({"a": a, "b": b, "cat": rng.choice(["x", "y", "z"], size=n)})

    fs = MRMR(
        verbose=0,
        n_jobs=1,
        random_seed=0,
        fe_smart_polynom_iters=1,
        fe_smart_polynom_optimization_steps=4,
        fe_max_polynom_degree=3,
    )
    # Pre-fix this raised TypeError: ufunc 'isfinite' not supported (object slice). Must complete cleanly now.
    fs.fit(df, pd.Series(y, name="y"))
    assert fs.support_ is not None
