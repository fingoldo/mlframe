"""Binned LightGBM fold datasets shared by every candidate a CV scorer evaluates on the same feature matrix.

``discover_chains`` scores 12 transforms (raw, residuals, unaries, chains) with the same KFold over the same ``x_matrix``;
only the TARGET differs. Fitting ``LGBMRegressor`` per candidate re-bins the fold's features every time: 36 dataset
constructions of 0.33s each on a 26k x 80 screen sample, a quarter of the 48s one base took. The fold's binned
dataset is built once here and each candidate trains on a row ``subset`` of it with its own label.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class LgbFoldCache:
    """Lazily-built binned ``lgb.Dataset`` per fold; ``fit_predict`` mirrors ``_build_tiny_model('lgb', ...)``."""

    def __init__(self, *, n_estimators: int, num_leaves: int, learning_rate: float, random_state: int, n_jobs: int = 1) -> None:
        self.n_estimators = int(n_estimators)
        # Same parameters the sklearn ``LGBMRegressor`` built by ``_build_tiny_model`` hands to the booster.
        self.params: Dict[str, Any] = {
            "objective": "regression",
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "seed": int(random_state),
            "num_threads": int(n_jobs),
            "verbose": -1,
            "force_col_wise": True,
        }
        self._folds: Dict[int, Any] = {}

    def _fold_dataset(self, fold_id: int, x_tr: np.ndarray) -> Any:
        ds = self._folds.get(fold_id)
        if ds is None:
            import lightgbm as lgb

            ds = lgb.Dataset(x_tr, label=np.zeros(x_tr.shape[0]), params=self.params, free_raw_data=False).construct()
            self._folds[fold_id] = ds
        return ds

    def fit_predict(self, fold_id: int, x_tr: np.ndarray, target: np.ndarray, fit_mask: np.ndarray, x_va: np.ndarray) -> np.ndarray:
        """Train on the ``fit_mask`` rows of fold ``fold_id`` with ``target`` as label; predict ``x_va``."""
        import lightgbm as lgb

        full = self._fold_dataset(fold_id, x_tr)
        rows = np.flatnonzero(fit_mask)
        train = full.subset(rows.tolist()) if rows.size < x_tr.shape[0] else full
        train.set_label(np.asarray(target, dtype=np.float64)[rows])
        booster = lgb.train(self.params, train, num_boost_round=self.n_estimators)
        return np.asarray(booster.predict(x_va, num_threads=self.params["num_threads"]), dtype=np.float64)
