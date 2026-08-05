"""FE_TRANSFORMER_B-2 (2026-08-05 audit): the documented "quantile-loss LightGBM (L1-regularized, at
median)" third aux model for regression was never implemented. ``_fit_aux_lgb``'s ``focal=True`` branch
only special-cased ``task == "binary"``, so for ``task == "regression"`` execution fell through into the
SAME vanilla ``LGBMRegressor`` config the non-focal aux model already uses (identical hyperparameters/
objective, differing only by ``random_state``) -- silently weakening the ``proba_std``/``proba_range``
cross-model-disagreement signal that is this module's entire stated purpose.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.multi_aux_ensemble import _fit_aux_lgb


def test_fit_aux_lgb_regression_focal_uses_quantile_median_not_duplicate():
    """The regression focal aux model must be configured with quantile loss at the median, distinct from
    the vanilla aux model's default squared-error loss."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] * 2.0 + rng.normal(scale=0.1, size=300)).astype(np.float32)

    m_vanilla = _fit_aux_lgb(X, y, task="regression", seed=0, focal=False, n_estimators=30, max_depth=3)
    m_focal = _fit_aux_lgb(X, y, task="regression", seed=1, focal=True, n_estimators=30, max_depth=3)

    focal_params = m_focal.get_params()
    vanilla_params = m_vanilla.get_params()

    assert focal_params.get("objective") == "quantile", f"expected the regression focal model to use quantile loss, got {focal_params.get('objective')!r}"
    assert focal_params.get("alpha") == 0.5, f"expected quantile loss at the median (alpha=0.5), got {focal_params.get('alpha')!r}"
    # Pre-fix these were identical (both None/default squared-error), differing only by random_state.
    assert focal_params.get("objective") != vanilla_params.get("objective"), "regression focal model must NOT be a near-duplicate of the vanilla aux model"


def test_fit_aux_lgb_regression_focal_is_more_robust_to_outliers_than_vanilla():
    """Functional check, not just config: quantile-at-median loss must pull predictions LESS toward
    injected large outliers than squared-error loss does -- the actual behavioral difference the fix is
    supposed to buy, not merely a different objective string."""
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] * 1.5 + rng.normal(scale=0.1, size=n)).astype(np.float32)
    # Inject a handful of extreme-outlier rows: squared-error loss is pulled hard toward them, quantile
    # (median) loss is much more robust.
    outlier_idx = rng.choice(n, size=15, replace=False)
    y[outlier_idx] += rng.choice([-1, 1], size=15) * rng.uniform(50, 100, size=15).astype(np.float32)

    m_vanilla = _fit_aux_lgb(X, y, task="regression", seed=0, focal=False, n_estimators=50, max_depth=3)
    m_focal = _fit_aux_lgb(X, y, task="regression", seed=1, focal=True, n_estimators=50, max_depth=3)

    clean_mask = np.ones(n, dtype=bool)
    clean_mask[outlier_idx] = False
    true_clean = X[clean_mask, 0] * 1.5

    mae_vanilla = float(np.mean(np.abs(m_vanilla.predict(X[clean_mask]) - true_clean)))
    mae_focal = float(np.mean(np.abs(m_focal.predict(X[clean_mask]) - true_clean)))

    assert mae_focal < mae_vanilla, (
        f"expected the median-quantile focal model to fit the clean majority better than the outlier-pulled "
        f"squared-error vanilla model, got focal_mae={mae_focal:.4f} vanilla_mae={mae_vanilla:.4f}"
    )


def test_fit_aux_lgb_binary_focal_still_returns_raw_booster():
    """Sanity: the pre-existing binary focal path (raw Booster via the custom focal objective) is
    unaffected by adding the regression branch."""
    import lightgbm as lgb

    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    m_focal = _fit_aux_lgb(X, y, task="binary", seed=0, focal=True, n_estimators=20, max_depth=3)
    assert isinstance(m_focal, lgb.Booster)
