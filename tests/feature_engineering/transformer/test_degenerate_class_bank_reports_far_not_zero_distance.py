"""A fold whose positive class has fewer than two rows reports the far sentinel, not a zero distance to the positive manifold."""
import numpy as np
import pytest

from mlframe.feature_engineering.transformer.active_virtual import compute_active_virtual_features
from mlframe.feature_engineering.transformer.adasyn_smote import compute_adasyn_smote_features
from mlframe.feature_engineering.transformer.mixup_boundary import compute_mixup_boundary_features
from mlframe.feature_engineering.transformer.multiscale_smote import compute_multiscale_smote_features
from mlframe.feature_engineering.transformer.smote_distance import compute_smote_distance_features
from mlframe.feature_engineering.transformer.class_distance import compute_class_distance_features


def _one_positive_row():
    """Fifty negatives and a single positive, so every positive bank is below the two-row minimum."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((51, 3)).astype(np.float32)
    y = np.zeros(51, dtype=np.float32)
    y[0] = 1.0
    return X, y


@pytest.mark.parametrize("fn", [compute_class_distance_features, compute_active_virtual_features, compute_adasyn_smote_features, compute_mixup_boundary_features, compute_multiscale_smote_features, compute_smote_distance_features])
def test_a_single_positive_row_gives_far_distances(fn):
    """The positive-distance columns must be 1e6 ("very far") and no column may be a zero distance."""
    X, y = _one_positive_row()
    df = fn(X, y, X[10:20], seed=1, task="binary")
    pos_cols = [c for c in df.columns if "_pos_k" in c]
    assert pos_cols
    assert all(float(df[c].min()) >= 1e5 for c in pos_cols), {c: float(df[c].min()) for c in pos_cols}
