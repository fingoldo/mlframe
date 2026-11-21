"""Synthetic-fixture builder shared by the data-valuation benchmark scripts, so one fixture rule serves all of them.

Consumers: ``profile_data_valuation.py`` and ``profile_training_weight_adapter.py`` (and
``profile_adversarial_reweighting.py`` through its delegating helper). Audit 05-F3 wording.
"""
from __future__ import annotations

import numpy as np


def make_blob_fixture(n_train: int, n_val: int, n_features: int, seed: int):
    """Synthetic 2-class blob-style fixture of the given (n_train, n_val, n_features) shape."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)
    return X_train, y_train, X_val, y_val
