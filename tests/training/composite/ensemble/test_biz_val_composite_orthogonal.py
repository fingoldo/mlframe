"""biz_value test for OrthogonalizedCompositeEstimator.

The trick: under CONFOUNDING (base and a feature share a latent cause), the plain OLS
base coefficient is biased; the Neyman-orthogonal / FWL estimate recovers the true
causal coefficient. We assert the orthogonalized base coefficient is MUCH closer to the
true value (bias >= 3x lower) and that OOS RMSE is no worse than the naive composite."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from mlframe.training.composite.orthogonal import OrthogonalizedCompositeEstimator


TRUE_BASE_COEF = 1.0


def _confounded_frame(n, seed):
    """base and f_conf both driven by a latent confounder U; y depends causally on base
    (coef 1.0) and on f_conf. OLS of y~base alone over-credits base because base proxies U."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)  # latent confounder
    f_conf = u + rng.normal(scale=0.3, size=n)
    base = u + rng.normal(scale=0.3, size=n)
    f_other = rng.normal(size=n)
    y = TRUE_BASE_COEF * base + 1.8 * f_conf + 0.5 * f_other + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"base": base, "f_conf": f_conf, "f_other": f_other})
    return X, y


def test_biz_val_orthogonal_debiases_base_coef_under_confounding():
    """Biz val orthogonal debiases base coef under confounding."""
    Xtr, ytr = _confounded_frame(4000, seed=11)
    Xte, yte = _confounded_frame(4000, seed=22)

    est = OrthogonalizedCompositeEstimator(
        base_column="base",
        inner_estimator=GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0),
        base_nuisance_estimator=GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0),
        y_nuisance_estimator=GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0),
        n_folds=5,
        random_state=0,
    )
    est.fit(Xtr, ytr)

    ortho_bias = abs(est.base_coef_ - TRUE_BASE_COEF)
    naive_bias = abs(est.naive_base_coef_ - TRUE_BASE_COEF)

    # The naive OLS base coefficient must be substantially biased (sanity: confounding present).
    assert naive_bias > 0.3, f"expected real confounding bias, got naive_bias={naive_bias:.3f}"
    # The orthogonalized estimate must cut the bias by at least 3x (measured ~10x+).
    assert ortho_bias <= naive_bias / 3.0, f"orthogonal bias {ortho_bias:.3f} should be <= naive bias {naive_bias:.3f} / 3"
    # And it should be close to the true causal coefficient in absolute terms.
    assert ortho_bias < 0.2, f"orthogonal base_coef_={est.base_coef_:.3f} far from true 1.0"

    # OOS RMSE no worse than a naive composite that uses the (biased) OLS base coefficient.
    naive_base = est.naive_base_coef_
    X_inner_tr = Xtr.drop(columns=["base"])
    X_inner_te = Xte.drop(columns=["base"])
    naive_inner = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0)
    naive_inner.fit(X_inner_tr, ytr - naive_base * Xtr["base"].to_numpy())
    naive_pred = naive_base * Xte["base"].to_numpy() + naive_inner.predict(X_inner_te)
    naive_rmse = mean_squared_error(yte, naive_pred) ** 0.5

    ortho_rmse = mean_squared_error(yte, est.predict(Xte)) ** 0.5
    assert ortho_rmse <= naive_rmse * 1.05, f"orthogonal OOS RMSE {ortho_rmse:.4f} must be no worse than naive {naive_rmse:.4f} (+5%)"
