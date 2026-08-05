"""FE_TRANSFORMER_A-2 related finding (2026-08-05 audit): ``residual_band_attention.py`` is iter 60, the
ROOT of this whole band-attention family, and its own docstring explicitly defended the in-sample
fit-then-predict choice as "intentional -- measure residual under the LGB hypothesis class, not forecast
unseen test residuals." Every later sibling (bidir_residual_band.py, decision_region_depth.py,
sign_residual_baseline.py, multi_temp_cbhr.py, class_balanced_hard_row.py, multi_baseline_hard_row.py,
disagreement_band.py) treats the identical pattern as a bug and fixes it via inner-KFold(3) OOF. Fixed
this root the same way for consistency, retracting the "intentional" framing.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.residual_band_attention import _fit_baseline_predict


def _in_sample_reference(Xt, y_t, task, seed, n_estimators=50, max_depth=3):
    """Pre-fix reference: fit once on Xt/y_t and predict on the SAME Xt (in-sample)."""
    import lightgbm as lgb

    if task == "binary":
        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        model.fit(Xt, y_t.astype(np.int32))
        return np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
    model.fit(Xt, y_t)
    return np.asarray(model.predict(Xt)).astype(np.float32)


def test_residual_band_attention_oof_differs_from_in_sample():
    """The fix must change real numeric output, not just the docstring framing."""
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(n)).astype(np.float32)

    preds_oof = _fit_baseline_predict(X, y, task="regression", seed=0)
    preds_in_sample = _in_sample_reference(X, y, task="regression", seed=0)

    assert not np.allclose(preds_oof, preds_in_sample), "OOF and in-sample predictions must differ -- the fix changes real numeric output"
    assert preds_oof.shape == (n,)


def test_residual_band_attention_small_n_falls_back_to_in_sample():
    """n < 3 cannot support a 3-fold inner split; must fall back to a single in-sample fit+predict."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    preds = _fit_baseline_predict(X, y, task="regression", seed=0)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))
