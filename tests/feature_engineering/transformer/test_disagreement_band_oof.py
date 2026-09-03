"""FE_TRANSFORMER_A-2 (2026-08-05 audit): ``disagreement_band.py``'s 3-baseline disagreement signal was
computed IN-SAMPLE (fit and predict on the same train rows), which systematically understates each
baseline's true disagreement -- all 3 models partially memorize each row's own label, suppressing measured
disagreement precisely on the rows that are genuinely hardest/most ambiguous under honest generalization.
Fixed by porting the inner-KFold(3) OOF pattern already shipped in the sibling
``bidir_residual_band.py::_fit_baseline_predict``.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.disagreement_band import _fit_3baselines_oof


def _fit_3baselines_in_sample_reference(Xt, y_t, task, seed):
    """Pre-fix reference: fit 3 baselines on (Xt, y_t) and predict on the SAME Xt (in-sample), the exact
    mechanism ``_fit_3baselines_oof`` replaced -- kept here only to demonstrate the fix changes real
    numeric behaviour, not just the function name."""
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    preds = np.zeros((Xt.shape[0], 3), dtype=np.float32)
    m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
    m1.fit(Xt, y_t)
    preds[:, 0] = np.asarray(m1.predict(Xt)).astype(np.float32)
    m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
    m2.fit(Xt, y_t)
    preds[:, 1] = np.asarray(m2.predict(Xt)).astype(np.float32)
    m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
    m3.fit(Xt, y_t)
    preds[:, 2] = m3.predict(Xt).astype(np.float32)
    return preds


def test_disagreement_band_oof_differs_from_in_sample():
    """The fix must change real numeric output, not just the function name: OOF disagreement (each row's
    prediction comes from a model that never saw that row's own label) must diverge from the pre-fix
    in-sample disagreement (each row's prediction comes from a model fit on that exact row)."""
    rng = np.random.default_rng(0)
    n_clean, n_noisy, d = 300, 100, 4
    X_clean = rng.standard_normal((n_clean, d)).astype(np.float32)
    y_clean = (X_clean[:, 0] + 0.05 * rng.standard_normal(n_clean)).astype(np.float32)
    X_noisy = rng.standard_normal((n_noisy, d)).astype(np.float32)
    y_noisy = rng.standard_normal(n_noisy).astype(np.float32)
    X = np.vstack([X_clean, X_noisy])
    y = np.concatenate([y_clean, y_noisy])

    preds_oof = _fit_3baselines_oof(X, y, task="regression", seed=0)
    preds_in_sample = _fit_3baselines_in_sample_reference(X, y, task="regression", seed=0)
    disagreement_oof = preds_oof.std(axis=1)
    disagreement_in_sample = preds_in_sample.std(axis=1)

    assert not np.allclose(
        disagreement_oof, disagreement_in_sample
    ), "OOF and in-sample disagreement must differ -- the fix changes real numeric output, not just naming"
    assert preds_oof.shape == (n_clean + n_noisy, 3)


def test_disagreement_band_small_n_falls_back_to_in_sample():
    """n < 3 cannot support a 3-fold inner split; must fall back to a single in-sample fit+predict
    rather than raising."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    preds = _fit_3baselines_oof(X, y, task="regression", seed=0)
    assert preds.shape == (2, 3)
    assert np.all(np.isfinite(preds))
