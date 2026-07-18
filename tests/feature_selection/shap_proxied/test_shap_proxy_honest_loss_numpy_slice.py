"""Regression: the honest-retrain path slices feature columns to a NAMELESS numpy array before
fitting / predicting the booster (speed lever 2026-06-08 -- strips xgboost's per-call
``from_cstr_to_pystr`` + ``_validate_features`` feature-name marshalling). The lever is gated on
being BIT-IDENTICAL to the legacy named-DataFrame path: tree splits depend only on column values +
positions, never names, and fit / predict use the same ``cols`` order so positional == named.

These tests pin that invariant so a future refactor that re-introduces named-DataFrame slicing (or
breaks the numpy gather) is caught. They also exercise the ndarray-input branch of the helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from sklearn.base import clone

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
    _honest_loss,
    _permutation_importance_ranking,
    _slice_cols_to_numpy,
)


def _make_xy(n=2000, p=24, seed=0):
    """Make xy."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"feat_{i}" for i in range(p)])
    logit = X.iloc[:, 0] - 0.7 * X.iloc[:, 1] + 0.4 * X.iloc[:, 3]
    y = (logit + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def test_slice_cols_to_numpy_dataframe_and_ndarray_equivalent():
    """Slice cols to numpy dataframe and ndarray equivalent."""
    X, _ = _make_xy()
    cols = [2, 5, 0, 11]
    from_df = _slice_cols_to_numpy(X, cols)
    from_np = _slice_cols_to_numpy(X.to_numpy(), cols)
    assert from_df.shape == (X.shape[0], len(cols))
    assert np.array_equal(from_df, from_np)
    # Order must follow ``cols`` exactly (positional == named contract).
    assert np.array_equal(from_df[:, 0], X.iloc[:, 2].to_numpy())
    assert np.array_equal(from_df[:, 2], X.iloc[:, 0].to_numpy())


def _honest_loss_named_reference(model_template, X_tr, y_tr, X_ev, y_ev, idx, metric):
    """The pre-lever named-DataFrame honest-loss, recomputed locally as the bit-identity oracle."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    cols = list(idx)
    est = clone(model_template)
    est.fit(X_tr.iloc[:, cols], y_tr)
    p = est.predict_proba(X_ev.iloc[:, cols])[:, 1]
    return _loss_from_predictions(p, y_ev, True, metric)


def test_honest_loss_numpy_bit_identical_to_named_dataframe():
    """Honest loss numpy bit identical to named dataframe."""
    X, y = _make_xy()
    Xtr, Xev = X.iloc[:1500], X.iloc[1500:]
    ytr, yev = y[:1500], y[1500:]
    model = make_default_estimator(True, random_state=0)
    for idx in ([0, 1, 3], [0, 3, 5, 7, 11], list(range(12))):
        got = _honest_loss(model, Xtr, ytr, Xev, yev, idx, True, "brier")
        ref = _honest_loss_named_reference(model, Xtr, ytr, Xev, yev, idx, "brier")
        assert got == ref, f"idx={idx}: numpy path {got!r} != named-DF reference {ref!r}"


def test_perm_importance_numpy_bit_identical_to_named_dataframe():
    """The shuffle-predict loop now feeds numpy directly (no per-shuffle DataFrame rewrap)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
        _loss_from_predictions,
    )

    X, y = _make_xy()
    Xtr, Xev = X.iloc[:1500], X.iloc[1500:]
    ytr, yev = y[:1500], y[1500:]
    model = make_default_estimator(True, random_state=0)
    cols = [0, 1, 3, 5, 7]

    # Reference: pre-lever named-DataFrame fit + per-shuffle DataFrame rewrap.
    est = clone(model)
    est.fit(Xtr.iloc[:, cols], ytr)
    Xev_sub = Xev.iloc[:, cols]
    Xev_arr = Xev_sub.to_numpy(copy=False)
    base_p = est.predict_proba(Xev_sub)[:, 1]
    base_loss_ref = _loss_from_predictions(base_p, yev, True, "brier")
    rng = np.random.default_rng(0)
    Xperm = Xev_arr.copy()
    perm = rng.permutation(Xperm.shape[0])
    imp_ref = np.zeros(len(cols))
    names = list(Xev_sub.columns)
    for j in range(len(cols)):
        orig = Xperm[:, j].copy()
        Xperm[:, j] = orig[perm]
        shuf_df = pd.DataFrame(Xperm, columns=names, index=Xev_sub.index, copy=False)
        p = est.predict_proba(shuf_df)[:, 1]
        imp_ref[j] = _loss_from_predictions(p, yev, True, "brier") - base_loss_ref
        Xperm[:, j] = orig

    base_loss, imp = _permutation_importance_ranking(model, Xtr, ytr, Xev, yev, cols, True, "brier", seed=0)

    assert base_loss == base_loss_ref
    assert np.array_equal(imp, imp_ref), f"perm importances differ: {imp} vs {imp_ref}"
