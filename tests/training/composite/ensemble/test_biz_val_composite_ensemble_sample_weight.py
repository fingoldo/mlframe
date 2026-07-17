"""biz_value tests for ``CompositeCrossTargetEnsemble.from_linear_stack`` and ``from_nnls_stack``.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test": when sample_weight is non-uniform,
the Ridge stack's coefficients must differ from the unweighted Ridge stack's coefficients; the NNLS
stack's sqrt-weight row-scaling must yield the same minimiser as the weighted-LS reference.

Naming: ``test_biz_val_<solver>_sample_weight_<scenario>``.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


class _NullModel:
    def predict(self, X):
        return np.zeros(len(X))


def _components(k):
    return [_NullModel() for _ in range(k)], [f"c{i}" for i in range(k)]


def _two_regime_dataset(n=500, seed=0):
    """Two component predictions, two regimes: regime-1 is recent and only c0 predicts well;
    regime-0 is older and only c1 predicts well. Linear stack's coefficient under recency weighting
    should pull c0's coefficient up and push c1's down vs uniform."""
    rng = np.random.default_rng(seed)
    n_recent = n // 3
    is_recent = np.zeros(n, dtype=bool)
    is_recent[-n_recent:] = True
    # y_true depends on different latent linear function in each regime.
    z = rng.normal(size=n)
    noise = 0.1 * rng.normal(size=n)
    y = np.where(is_recent, 2.0 * z, -2.0 * z) + noise
    # Components: c0 predicts z (right answer in recent regime); c1 predicts -z (right in older regime).
    c0 = z + 0.05 * rng.normal(size=n)
    c1 = -z + 0.05 * rng.normal(size=n)
    X_stack = np.column_stack([c0, c1])
    return X_stack, y, is_recent


def test_biz_val_ridge_stack_sample_weight_shifts_weights_toward_recent_regime():
    """Under recency weighting, the Ridge stack's c1 coefficient must FLIP SIGN: unweighted-Ridge fits the
    cross-regime average so it inflates c1's positive weight (it correlates with y on the older slice);
    recency-weighted Ridge sees only the recent regime where c1 = -z (anti-correlated with y=2z) so the
    optimal c1 coefficient is negative. This sign flip is the smoking-gun biz_value win."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y, is_recent = _two_regime_dataset()
    sw_recency = np.where(is_recent, 1.0, 0.001)
    models, names = _components(2)
    unweighted = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=0.1)
    weighted = CompositeCrossTargetEnsemble.from_linear_stack(models, names, X, y, ridge_alpha=0.1, sample_weight=sw_recency)
    # c1's coefficient must flip from positive (cross-regime average) to negative (recent-regime alignment).
    assert unweighted.weights[1] > 0, f"unweighted Ridge should have positive c1 weight; got {unweighted.weights}"
    assert weighted.weights[1] < 0, f"recency-weighted Ridge should have negative c1 weight; got {weighted.weights}"


def test_biz_val_nnls_stack_row_scaling_matches_weighted_least_squares_minimiser():
    """The sqrt-w row-scaling trick must equal the weighted-LS minimiser to high precision."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
    from scipy.optimize import nnls

    X, y, is_recent = _two_regime_dataset()
    sw = np.where(is_recent, 1.0, 0.1)
    models, names = _components(2)
    fitted = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=sw)
    # Reference: solve the scaled system directly.
    sqrt_w = np.sqrt(sw).reshape(-1, 1)
    A_scaled = X * sqrt_w
    b_scaled = y * sqrt_w.reshape(-1)
    w_ref, _ = nnls(A_scaled, b_scaled)
    np.testing.assert_allclose(fitted.weights, w_ref, rtol=1e-8)


def test_biz_val_nnls_weighted_predictor_outperforms_unweighted_on_recent_slice():
    """The deployment win: training NNLS with recency sample_weight must produce a stack predictor that has
    lower RMSE on the recent slice than the unweighted-trained stack."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    X, y, is_recent = _two_regime_dataset(n=600, seed=3)
    sw_recency = np.where(is_recent, 1.0, 0.001)
    models, names = _components(2)
    unweighted = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y)
    weighted = CompositeCrossTargetEnsemble.from_nnls_stack(models, names, X, y, sample_weight=sw_recency)

    def _rmse_on_recent(w):
        recent_X = X[is_recent]
        recent_y = y[is_recent]
        preds = recent_X @ w.weights
        return float(np.sqrt(np.mean((preds - recent_y) ** 2)))

    rmse_uniform = _rmse_on_recent(unweighted)
    rmse_recency = _rmse_on_recent(weighted)
    assert rmse_recency < rmse_uniform, (
        f"recency-weighted NNLS must beat unweighted NNLS on recent-slice RMSE; got uniform={rmse_uniform:.4f}, weighted={rmse_recency:.4f}"
    )
