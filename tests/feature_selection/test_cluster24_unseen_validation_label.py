"""A validation-only class gets the probability floor, not a neighbouring class's probability (audit 2026-11-20, finding 24-F48)."""

import numpy as np

from mlframe.feature_selection._hybrid_twin_prune import _row_losses


def test_validation_only_class_is_not_scored_with_a_neighbour_column():
    """Class 2 occurs only in fold 0's validation rows: its log-loss must be the floor, not a neighbouring column's probability."""
    rng = np.random.default_rng(0)
    n = 100
    Z = rng.normal(size=(n, 2))
    y = (Z[:, 0] > 0).astype(int)
    folds = np.arange(n) % 5
    y[(folds == 0) & (np.arange(n) >= 60)] = 2
    out = _row_losses(Z, y, classification=True, folds=folds, model_kind="linear")
    assert out is not None
    unseen_rows = y == 2
    assert unseen_rows.any()
    assert np.all(out[unseen_rows] > 20.0)  # -log(1e-12) is about 27.6
