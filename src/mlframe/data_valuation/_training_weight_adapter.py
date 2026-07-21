"""End-to-end adapter: KNN-Shapley row valuation -> a ``sample_weight`` vector for the FULL training set,
scalable to millions of training rows.

``knn_shapley`` itself is ``O(n_val * n_train)`` (a full ``cdist`` per validation batch) -- exact and
cheap at the sizes its own biz_val tests use (n_train in the low thousands), but that cost is
prohibitive once ``n_train`` reaches production scale (mlframe datasets routinely have millions of
rows). This module never runs the closed-form recursion against the full training set: it values a
capped, seeded RANDOM SUBSAMPLE of training rows against the caller's held-out ``X_val``/``y_val``
(never ``X_test`` -- valuation must be computed against genuinely unseen validation data, exactly the
same OOF discipline every other honest-metric helper in this repo already requires), then extends that
subsample's values to every other training row via :func:`propagate_subsample_values`'s
nearest-neighbor imputation (itself ``O(n_train * max_valued_rows)`` -- linear in ``n_train``, not
quadratic).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mlframe.data_valuation._knn_shapley import knn_shapley
from mlframe.data_valuation._mc_sampling import propagate_subsample_values
from mlframe.data_valuation._weights import valuation_sample_weight


def training_sample_weight_from_valuation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    k: int = 5,
    max_valued_rows: int = 20_000,
    propagate_k: int = 1,
    weight_mode: str = "clip_negative",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """KNN-Shapley-derived ``sample_weight``, shape ``(n_train,)``, for the FULL ``X_train`` regardless of size.

    ``X_val``/``y_val`` MUST be a validation split disjoint from both ``X_train`` and the caller's test
    set -- this function computes valuations against whatever it is given, but never against a test set
    itself; the caller owns that split (mirrors every OOF-discipline helper elsewhere in this repo, e.g.
    ``stacking_aware_gate``'s leakage contract).

    When ``X_train.shape[0] <= max_valued_rows``, every row is valued directly (matches
    :func:`mlframe.data_valuation.knn_shapley`'s own biz_val exactly). Above the cap, a seeded random
    subsample of ``max_valued_rows`` training rows is valued, standardized on the FULL train's own
    mean/std (consistent distances for both the valuation call and the propagation step below), and
    every other row's value is imputed from its ``propagate_k`` nearest valued neighbors via
    :func:`propagate_subsample_values` -- turning the ``O(n_val * n_train)`` cost into
    ``O(n_val * max_valued_rows + n_train * max_valued_rows)``, linear in ``n_train``.

    Returns a ``(n_train,)`` non-negative weight vector, mean ~= 1 (see :func:`valuation_sample_weight`
    for ``weight_mode`` semantics), ready to pass straight into ``model.fit(sample_weight=...)`` -- the
    existing ``_setup_sample_weight`` choke point in ``training/_data_helpers.py`` already forwards
    whatever ``sample_weight`` array it is given, regardless of source.
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    n_train = X_train.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    X_train_std = (X_train - mu) / sigma
    X_val_std = (np.ascontiguousarray(X_val, dtype=np.float64) - mu) / sigma

    if n_train <= max_valued_rows:
        values = knn_shapley(X_train_std, y_train, X_val_std, y_val, k=k, standardize=False)
        return valuation_sample_weight(values, mode=weight_mode)

    valued_idx = rng.choice(n_train, size=max_valued_rows, replace=False)
    sub_values = knn_shapley(X_train_std[valued_idx], y_train[valued_idx], X_val_std, y_val, k=k, standardize=False)
    values = propagate_subsample_values(X_train_std, X_train_std[valued_idx], sub_values, k=propagate_k)
    return valuation_sample_weight(values, mode=weight_mode)


__all__ = ["training_sample_weight_from_valuation"]
