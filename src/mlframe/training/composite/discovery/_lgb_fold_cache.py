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
        self._holdouts: Dict[int, np.ndarray] = {}

    def has_fold(self, fold_id: int) -> bool:
        """True once fold ``fold_id``'s dataset is built, so the caller can skip slicing its train rows."""
        return fold_id in self._folds

    def holdout(self, fold_id: int) -> np.ndarray:
        """Fold ``fold_id``'s holdout feature slice, kept from the call that built the fold."""
        return self._holdouts[fold_id]

    def _fold_dataset(self, fold_id: int, x_tr: np.ndarray) -> Any:
        """The LightGBM ``Dataset`` for fold ``fold_id``, built once from ``x_tr`` and reused."""
        ds = self._folds.get(fold_id)
        if ds is None:
            import lightgbm as lgb

            ds = lgb.Dataset(x_tr, label=np.zeros(x_tr.shape[0]), params=self.params, free_raw_data=False).construct()
            self._folds[fold_id] = ds
        return ds

    def fit_predict(self, fold_id: int, x_tr: np.ndarray | None, target: np.ndarray, fit_mask: np.ndarray, x_va: np.ndarray) -> np.ndarray:
        """Train on the ``fit_mask`` rows of fold ``fold_id`` with ``target`` as label; predict ``x_va``.

        ``x_tr`` may be ``None`` once the fold is built (:meth:`has_fold`): the binned dataset already holds its rows.
        """
        import lightgbm as lgb

        if x_tr is None:
            full = self._folds[fold_id]
        else:
            full = self._fold_dataset(fold_id, x_tr)
            self._holdouts.setdefault(fold_id, x_va)
        rows = np.flatnonzero(fit_mask)
        # The subset must be constructed BEFORE its label is set: a lazy subset builds itself from the parent at train
        # time and takes the parent's label with it -- the all-zeros placeholder -- so every masked candidate trained on
        # zeros and predicted a constant, while ``get_label()`` still reported the label that was set.
        train = full.subset(rows.tolist()).construct() if rows.size < fit_mask.shape[0] else full
        train.set_label(np.asarray(target, dtype=np.float64)[rows])
        booster = lgb.train(self.params, train, num_boost_round=self.n_estimators)
        return np.asarray(booster.predict(x_va, num_threads=self.params["num_threads"]), dtype=np.float64)
