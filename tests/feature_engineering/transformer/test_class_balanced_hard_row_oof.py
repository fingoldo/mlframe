"""FE_TRANSFORMER_A-2 related finding (2026-08-05 audit): ``class_balanced_hard_row.py``'s
``_fit_baseline_predict`` docstring literally said it returned "IN-SAMPLE predictions, used only to rank
rows by |residual| hardness" -- the same in-sample-leakage class already fixed for the sibling
``bidir_residual_band.py``. Fixed by porting the identical inner-KFold(3) OOF pattern.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.class_balanced_hard_row import _fit_baseline_predict


def _fit_in_sample_reference(Xt, y_t, task, seed, n_estimators=50, max_depth=3):
    """Pre-fix reference: fit once on Xt/y_t and predict on the SAME Xt (in-sample)."""
    import lightgbm as lgb

    if task == "binary":
        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        model.fit(Xt, y_t.astype(np.int32))
        return np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
    model.fit(Xt, y_t)
    return np.asarray(model.predict(Xt)).astype(np.float32)


def test_class_balanced_hard_row_oof_differs_from_in_sample():
    """The fix must change real numeric output, not just add a docstring caveat: OOF predictions (each
    row scored by a model that never saw that row's own label) must diverge from the pre-fix in-sample
    predictions."""
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(n)).astype(np.float32)

    preds_oof = _fit_baseline_predict(X, y, task="regression", seed=0)
    preds_in_sample = _fit_in_sample_reference(X, y, task="regression", seed=0)

    assert not np.allclose(preds_oof, preds_in_sample), "OOF and in-sample predictions must differ -- the fix changes real numeric output"
    assert preds_oof.shape == (n,)


def test_class_balanced_hard_row_small_n_falls_back_to_in_sample():
    """n < 3 cannot support a 3-fold inner split; must fall back to a single in-sample fit+predict
    rather than raising."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    preds = _fit_baseline_predict(X, y, task="regression", seed=0)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
