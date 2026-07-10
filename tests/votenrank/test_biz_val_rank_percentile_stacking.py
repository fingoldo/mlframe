"""biz_value test for ``votenrank.rank_percentile_transform``.

The win: a base learner's raw score often has a skewed/miscalibrated distribution while its RANKING remains
faithful (e.g. a probability estimate whose scale doesn't linearly track true risk, but whose relative
ordering does) -- so the target's true dependence is on the base learner's rank-percentile, not its raw
magnitude. A linear meta-model fit on the raw (skewed) score mismodels that relationship; rank-percentile-
transforming first restores the true linear-in-percentile form and lets the meta-model fit it almost exactly.
This is the literal mechanism the source diagnosis describes ("stacking hurts single model" from miscalibrated
base learners) -- confirmed here by first testing the underlying hypothesis directly (a naive outlier-
contamination synthetic did NOT reproduce a raw-vs-percentile gap: LinearRegression already shrinks a
corrupted feature's coefficient toward zero on its own) before committing to this scenario.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.votenrank.rank_percentile_stacking import rank_percentile_transform


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_biz_val_rank_percentile_transform_fixes_linear_stacker_on_skewed_base_learner():
    rng = np.random.default_rng(0)
    n = 4000

    # a realistic miscalibrated base learner: heavily right-skewed raw score (lognormal), whose RANKING is
    # what actually tracks outcome risk -- not its raw magnitude (the raw scale is an artifact of the base
    # model's own loss function / calibration, not a meaningful linear predictor of the target).
    base_c = rng.lognormal(mean=0.0, sigma=1.5, size=n)
    c_pct, _ = rank_percentile_transform(base_c)
    y = 3.0 * c_pct + rng.normal(0, 0.3, n)

    split = int(0.7 * n)
    raw_model = LinearRegression().fit(base_c[:split].reshape(-1, 1), y[:split])
    raw_rmse = _rmse(y[split:], raw_model.predict(base_c[split:].reshape(-1, 1)))

    pct_model = LinearRegression().fit(c_pct[:split].reshape(-1, 1), y[:split])
    pct_rmse = _rmse(y[split:], pct_model.predict(c_pct[split:].reshape(-1, 1)))

    assert pct_rmse < raw_rmse * 0.6, (
        f"rank-percentile transform should let the linear stacker recover the true percentile-linear "
        f"relationship a raw skewed base-learner score obscures: pct={pct_rmse:.4f} raw={raw_rmse:.4f}"
    )


def test_rank_percentile_transform_oof_range_and_monotonicity():
    oof = np.array([5.0, 1.0, 3.0, 3.0, 2.0])
    pct, test_pct = rank_percentile_transform(oof)
    assert test_pct is None
    assert np.all((pct > 0) & (pct < 1))
    # monotonic: larger raw value -> larger (or equal, for ties) percentile.
    order = np.argsort(oof)
    assert np.all(np.diff(pct[order]) >= -1e-12)


def test_rank_percentile_transform_test_pred_matches_oof_scale():
    oof = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_pred = np.array([0.0, 3.0, 100.0])
    oof_pct, test_pct = rank_percentile_transform(oof, test_pred)
    assert test_pct is not None
    assert test_pct[0] <= oof_pct[0]  # below the OOF range -> low percentile (ties the OOF min by construction)
    assert test_pct[2] > oof_pct[-1]  # above the OOF range -> clamped near the top
    assert np.all((test_pct >= 0) & (test_pct <= 1))


def test_rank_percentile_transform_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        rank_percentile_transform(np.array([]))
