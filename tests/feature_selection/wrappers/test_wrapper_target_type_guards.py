"""Target-type / edge-case robustness guards for the wrapper selectors.

A supported target type or realistic degenerate frame must never crash a public wrapper selector with an OBSCURE
internal traceback (numba TypingError, sklearn stratify "least populated class", LGBM 1d-array). Unsupported target
shapes must raise a CLEAR, actionable error at fit entry instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _Xy(n=400, seed=0):
    """Build a 4-column frame with two signal columns (s0, s1) and two noise columns, for the target-type guard tests."""
    rng = np.random.default_rng(seed)
    s0, s1 = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame({"s0": s0, "s1": s1, "n0": rng.normal(size=n), "n1": rng.normal(size=n)})
    return X, s0, s1


# --------------------------------------------------------------- ShapProxiedFS


def test_shapproxied_rejects_2d_multilabel_target_clearly():
    """Pre-fix a 2D y survived np.unique/astype and blew up deep in a numba kernel with an opaque TypingError;
    now it must raise a clear single-output ValueError at fit entry."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, s0, s1 = _Xy()
    y2d = np.c_[(s0 > 0).astype(int), (s1 > 0).astype(int)]
    sel = ShapProxiedFS(classification=True, n_splits=3, top_n=8, random_state=0)
    with pytest.raises(ValueError, match="single-output 1D target"):
        sel.fit(X, y2d)


def test_shapproxied_accepts_single_column_2d_with_hint():
    """Shapproxied accepts single column 2d with hint."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, s0, _ = _Xy()
    y = (s0 > 0).astype(int).reshape(-1, 1)
    sel = ShapProxiedFS(classification=True, n_splits=3, top_n=8, random_state=0)
    with pytest.raises(ValueError, match="ravel"):
        sel.fit(X, y)


# --------------------------------------------------------------- HybridSelector


def test_hybrid_rejects_regression_target_clearly():
    """Pre-fix a continuous target crashed inside train_test_split's stratify with an opaque message; now it must
    raise a clear classification-only ValueError."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    X, s0, s1 = _Xy()
    y = s0 + 0.8 * s1  # continuous
    with pytest.raises(ValueError, match="classification targets only"):
        HybridSelector(use_fe=False, fe_max_steps=0, random_state=0).fit(X, y)


def test_hybrid_rejects_multilabel_target_clearly():
    """Hybrid rejects multilabel target clearly."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    X, s0, s1 = _Xy()
    y2d = np.c_[(s0 > 0).astype(int), (s1 > 0).astype(int)]
    with pytest.raises(ValueError, match="single-output classification"):
        HybridSelector(use_fe=False, fe_max_steps=0, random_state=0).fit(X, y2d)


@pytest.mark.timeout(300)
def test_hybrid_count_target_with_rare_classes_completes():
    """A count / high-cardinality multiclass target has rare classes (some with a single member). Pre-fix the shared
    permutation-FI's unconditional stratified split crashed; now it falls back to an unstratified split and completes."""
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    X, s0, _ = _Xy()
    y = np.abs(np.round(s0 * 3)).astype(int)  # 0..~10 with rare tails
    assert (np.bincount(y) == 1).any(), "fixture must contain a single-member class to exercise the guard"
    sel = HybridSelector(use_fe=False, fe_max_steps=0, random_state=0).fit(X, y)
    assert hasattr(sel, "selected_features_")
