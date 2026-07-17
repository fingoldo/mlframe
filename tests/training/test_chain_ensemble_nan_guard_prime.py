"""Regression: _ChainEnsemble.fit must prime the predict-time NaN guard.

Fuzz combo c0146 (multilabel chain + inject_inf_nan) crashed at predict with
NanGuardNotPrimedError: the chain imputed its own fit frame but never primed the
outer guard, so a NaN-bearing predict frame had no persisted imputer/scaler and
the leak-safe guard refused. fit() now calls prime_nan_guard_stats(self, X).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fit_chain():
    lgb = pytest.importorskip("lightgbm")
    from mlframe.training._classif_helpers import _ChainEnsemble

    rng = np.random.default_rng(0)
    n, p, K = 240, 6, 3
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    Y = (rng.random((n, K)) < 0.4).astype(int)
    model = _ChainEnsemble(
        lgb.LGBMClassifier(n_estimators=10, verbose=-1),
        n_labels=K,
        n_chains=2,
    )
    model.fit(X, Y)
    return model, X, n


def test_chain_ensemble_fit_primes_nan_guard():
    model, _X, _n = _fit_chain()
    assert hasattr(model, "_mlframe_nan_imputer"), "fit must prime the NaN-guard imputer"
    assert hasattr(model, "_mlframe_nan_scaler"), "fit must prime the NaN-guard scaler"


def test_chain_ensemble_nan_predict_frame_not_refused():
    model, X, n = _fit_chain()
    from mlframe.training._predict_guards import _apply_nan_guard, NanGuardNotPrimedError

    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan  # NaN-bearing predict frame
    try:
        _apply_nan_guard(model, X_nan, fn=lambda Xi: np.asarray(Xi), n_rows=n)
    except NanGuardNotPrimedError:
        pytest.fail("primed fit must let the guard impute a NaN predict frame, not raise NanGuardNotPrimedError")
