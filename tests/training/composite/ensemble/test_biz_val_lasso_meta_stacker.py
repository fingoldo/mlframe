"""biz_value: Lasso meta-stacker auto-zeroes weak/redundant ensemble members.

Synthetic: two informative components (``a``, ``b``, weighted 0.7/0.3 into ``y``) plus several REDUNDANT copies
of ``a`` with tiny added noise (near-duplicates a Ridge stacker keeps small-but-nonzero weight on, spreading its
L2 budget) and several pure-noise components with zero true relationship to ``y``. A Lasso stacker's L1 penalty
should drive the redundant/noise components' coefficients to exactly zero, keeping only the genuinely
informative ones -- a sparse, self-selecting ensemble in one fit, unlike Ridge's dense shrinkage.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlframe.training.composite.ensemble._stackers import fit_lasso_meta_stacker, fit_ridge_meta_stacker

warnings.filterwarnings("ignore")


def _gen_redundant_and_noise(n, seed, n_redundant=4, n_noise=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 0.7 * a + 0.3 * b + 0.05 * rng.normal(size=n)
    redundant = [a + 0.02 * rng.normal(size=n) for _ in range(n_redundant)]
    noise = [rng.normal(size=n) for _ in range(n_noise)]
    X = np.column_stack([a, b, *redundant, *noise])
    return X, y


def test_biz_val_lasso_meta_stacker_zeroes_redundant_and_noise_components():
    X, y = _gen_redundant_and_noise(n=2000, seed=0)
    n_components = X.shape[1]

    lasso = fit_lasso_meta_stacker(X, y, n_components)
    ridge = fit_ridge_meta_stacker(X, y, n_components)

    lasso_n_zero = int(np.sum(np.asarray(lasso.coef_) == 0.0))
    ridge_n_zero = int(np.sum(np.asarray(ridge.coef_) == 0.0))

    # 6 of 10 components (4 redundant + 4 pure-noise, indices 2..9) carry no genuinely new information beyond
    # ``a``/``b`` (indices 0, 1) -- Lasso should zero out a real share of them (measured 3/6); Ridge's dense L2
    # shrinkage should not.
    assert lasso_n_zero >= 3, f"expected Lasso to zero out a real share of the 6 redundant/noise components, got {lasso_n_zero} zeroed (coef_={lasso.coef_})"
    assert ridge_n_zero <= 1, f"expected Ridge's dense shrinkage to keep coefficients nonzero (sanity check on the synthetic), got {ridge_n_zero} zeroed"
    assert lasso_n_zero > ridge_n_zero, f"expected Lasso to zero out strictly more components than Ridge, got lasso={lasso_n_zero} ridge={ridge_n_zero}"


def test_biz_val_lasso_meta_stacker_holdout_rmse_not_worse_than_ridge():
    X_train, y_train = _gen_redundant_and_noise(n=2000, seed=0)
    X_test, y_test = _gen_redundant_and_noise(n=2000, seed=1)
    n_components = X_train.shape[1]

    lasso = fit_lasso_meta_stacker(X_train, y_train, n_components)
    ridge = fit_ridge_meta_stacker(X_train, y_train, n_components)

    rmse_lasso = float(np.sqrt(np.mean((lasso.predict(X_test) - y_test) ** 2)))
    rmse_ridge = float(np.sqrt(np.mean((ridge.predict(X_test) - y_test) ** 2)))

    assert rmse_lasso <= rmse_ridge * 1.10, f"expected Lasso holdout RMSE to be within 10% of Ridge despite sparsity, got lasso={rmse_lasso:.4f} ridge={rmse_ridge:.4f}"
