"""Regression: polars + TimeSeriesSplit OOF must build the holdout from an explicit
fold_holdout_idx mask, not ``~train_mask``.

Under ``TimeSeriesSplit`` the complement of a fold's train indices includes ALL future
rows, not just that fold's holdout window. The pre-fix polars branch built the holdout
via ``train_X.filter(~fold_train_mask)`` -> a frame LONGER than ``fold_holdout_idx``, so
``buf[fold_holdout_idx] = preds`` raised a length mismatch that the per-component except
swallowed, silently dropping every component on every fold. The pandas / ndarray branches
already indexed by ``fold_holdout_idx`` and were unaffected.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from sklearn.linear_model import LinearRegression

from mlframe.training.composite.ensemble import compute_oof_holdout_predictions


def _fitted_linear(X, y):
    """Fitted linear."""
    return LinearRegression().fit(np.asarray(X, dtype=np.float64), y)


def test_polars_timeseries_oof_keeps_components_and_matches_pandas():
    """Polars timeseries oof keeps components and matches pandas."""
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=n)).astype(np.float64)
    # Monotone time_ordering -> the kfold path uses TimeSeriesSplit.
    time_ordering = np.arange(n, dtype=np.float64)

    import pandas as pd

    cols = ["a", "b", "c"]
    X_pd = pd.DataFrame(X, columns=cols)
    X_pl = pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)})

    def _run(train_X):
        """Computes OOF-holdout predictions for a single fitted linear component given either a pandas or polars train_X."""
        model = _fitted_linear(train_X if not hasattr(train_X, "to_pandas") else X, y)
        return compute_oof_holdout_predictions(
            component_models=[model],
            component_names=["lin"],
            component_specs=[None],
            train_X=train_X,
            y_train_full=y,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=0,
            time_ordering=time_ordering,
            kfold=3,
        )

    oof_pl, _y_pl, survived_pl = _run(X_pl)
    oof_pd, _y_pd, _survived_pd = _run(X_pd)

    # The component must survive (pre-fix: dropped -> empty survivor list / (0,0) result).
    assert survived_pl == ["lin"], survived_pl
    assert oof_pl.shape == oof_pd.shape and oof_pl.shape[1] == 1
    # Polars and pandas OOF must agree on the populated holdout rows.
    mask = np.isfinite(oof_pd[:, 0])
    assert mask.sum() > 0
    np.testing.assert_allclose(oof_pl[mask, 0], oof_pd[mask, 0], rtol=1e-6, atol=1e-6)
