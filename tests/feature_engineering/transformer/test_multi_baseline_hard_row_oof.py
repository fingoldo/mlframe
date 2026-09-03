"""FE_TRANSFORMER_A-2 related finding (2026-08-05 audit): ``multi_baseline_hard_row.py``'s
``_fit_3baselines_predict`` fit 3 baselines on Xt and predicted on the SAME Xt (in-sample) -- the direct
3-baseline-ensemble analogue of ``disagreement_band.py``'s own FE_TRANSFORMER_A-2 bug. Fixed by porting
the identical inner-KFold(3) OOF pattern.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.multi_baseline_hard_row import _fit_3baselines_predict


def _fit_3baselines_in_sample_reference(Xt, y_t, task, seed):
    """Pre-fix reference: fit 3 baselines on Xt/y_t and predict on the SAME Xt (in-sample)."""
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    preds_list = []
    m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
    m1.fit(Xt, y_t)
    preds_list.append(np.asarray(m1.predict(Xt)).astype(np.float32))
    m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
    m2.fit(Xt, y_t)
    preds_list.append(np.asarray(m2.predict(Xt)).astype(np.float32))
    m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
    m3.fit(Xt, y_t)
    preds_list.append(m3.predict(Xt).astype(np.float32))
    return preds_list


def test_multi_baseline_hard_row_oof_differs_from_in_sample():
    """The fix must change real numeric output, not just naming: OOF predictions must diverge from the
    pre-fix in-sample predictions for all 3 baselines."""
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(n)).astype(np.float32)

    preds_oof = _fit_3baselines_predict(X, y, task="regression", seed=0)
    preds_in_sample = _fit_3baselines_in_sample_reference(X, y, task="regression", seed=0)

    assert len(preds_oof) == 3
    for b in range(3):
        assert not np.allclose(
            preds_oof[b], preds_in_sample[b]
        ), f"baseline {b}: OOF and in-sample predictions must differ -- the fix changes real numeric output"
        assert preds_oof[b].shape == (n,)


def test_multi_baseline_hard_row_small_n_falls_back_to_in_sample():
    """n < 3 cannot support a 3-fold inner split; must fall back to a single in-sample fit+predict."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    preds_list = _fit_3baselines_predict(X, y, task="regression", seed=0)
    assert len(preds_list) == 3
    for p in preds_list:
        assert p.shape == (2,)
        assert np.all(np.isfinite(p))
