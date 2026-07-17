"""N21: time-respecting K-fold OOF (forward-walking TimeSeriesSplit) instead of
the single trailing-slice downgrade."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from mlframe.training.composite.ensemble import compute_oof_holdout_predictions


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    y = b + 0.5 * feat + rng.normal(0.0, 0.1, n)
    X = pd.DataFrame({"b": b, "feat": feat})
    return X, y


class TestN21TimeKfold:
    def test_monotone_time_kfold_forward_walk(self) -> None:
        X, y = _data()
        inner = LinearRegression().fit(X, y)
        ts = np.arange(len(X))  # monotone
        oof, _yh, names = compute_oof_holdout_predictions(
            component_models=[inner, inner],
            component_names=["c0", "c1"],
            component_specs=[None, None],
            train_X=X,
            y_train_full=y,
            base_train_full_per_spec={},
            holdout_frac=0.2,
            random_state=0,
            kfold=3,
            time_ordering=ts,
        )
        assert names == ["c0", "c1"]
        # Forward-walk leaves the FIRST fold's rows train-only, so coverage is
        # < n but spans MULTIPLE holdout blocks (> a single trailing slice).
        n_cov = int(np.isfinite(oof).any(axis=1).sum())
        assert oof.shape[0] > 0
        assert n_cov >= int(0.5 * len(X)), "forward-walk OOF should cover most rows"

    def test_non_monotone_time_downgrades_with_warn(self, caplog) -> None:
        X, y = _data(n=2000)
        inner = LinearRegression().fit(X, y)
        ts = np.random.default_rng(1).permutation(len(X))  # NON-monotone
        import logging

        with caplog.at_level(logging.WARNING):
            _oof, _yh, _names = compute_oof_holdout_predictions(
                component_models=[inner, inner],
                component_names=["c0", "c1"],
                component_specs=[None, None],
                train_X=X,
                y_train_full=y,
                base_train_full_per_spec={},
                holdout_frac=0.2,
                random_state=0,
                kfold=3,
                time_ordering=ts,
            )
        assert any("NON-monotone" in r.message or "Downgrading" in r.message for r in caplog.records)
