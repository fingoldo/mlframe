"""FE_TRANSFORMER_A-2 related finding (2026-08-05 audit): ``fisher_weighted_residual.py``'s ``_process``
computed ``resid_train`` from the full-fit model's IN-SAMPLE predictions (``p_train = model.predict(Xt_s)``),
the same in-sample leakage class already fixed for the sibling ``bidir_residual_band.py``. This
``resid_train`` sets the quantile-band thresholds for the weighted-residual banding, so understated
in-sample residuals bias which train rows define each band. Fixed by computing ``resid_train`` from a
separate inner-KFold(3) OOF pass (``_oof_train_predictions``), while keeping the FULL-fit model for the
genuinely-held-out query path and the Fisher-gradient probe (mirrors ``sign_residual_baseline.py``'s
pattern of separating the query-path model from the train-side-statistic OOF pass).
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.fisher_weighted_residual import (
    _oof_train_predictions,
    compute_fisher_weighted_residual_features,
)


def _in_sample_reference(Xt, y_t, is_binary, seed):
    """Pre-fix reference: fit once on Xt/y_t and predict on the SAME Xt (in-sample)."""
    import lightgbm as lgb

    if is_binary:
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        model.fit(Xt, y_t.astype(np.int32))
        return np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
    model.fit(Xt, y_t)
    return np.asarray(model.predict(Xt)).astype(np.float32)


def test_oof_train_predictions_differs_from_in_sample():
    """The fix must change real numeric output, not just add a docstring caveat."""
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(n)).astype(np.float32)

    preds_oof = _oof_train_predictions(X, y, is_binary=False, seed=0)
    preds_in_sample = _in_sample_reference(X, y, is_binary=False, seed=0)

    assert not np.allclose(preds_oof, preds_in_sample), "OOF and in-sample train predictions must differ -- the fix changes real numeric output"
    assert preds_oof.shape == (n,)


def test_oof_train_predictions_small_n_falls_back_to_in_sample():
    """n < 3 cannot support a 3-fold inner split; must fall back to a single in-sample fit+predict."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    preds = _oof_train_predictions(X, y, is_binary=False, seed=0)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))


def test_compute_fisher_weighted_residual_features_end_to_end_finite():
    """Public API still produces finite output end-to-end after the OOF wiring."""
    rng = np.random.default_rng(2)
    n_train, n_query, d = 120, 10, 4
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.1 * rng.standard_normal(n_train)).astype(np.float32)
    X_query = rng.standard_normal((n_query, d)).astype(np.float32)

    out = compute_fisher_weighted_residual_features(X_train, y_train, X_query, seed=0, task="regression")
    arr = out.to_numpy()
    assert arr.shape[0] == n_query
    assert np.all(np.isfinite(arr))
