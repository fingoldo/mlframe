"""Equal values must get equal extremality, and the within-batch and reference conventions must agree."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.row_wise_extremality import _compute_extremality_matrix
from mlframe.feature_engineering.row_wise_extremality_reference import extremality_matrix_from_reference, fit_extremality_reference


def test_a_binary_column_gives_every_zero_the_same_score():
    """Ordinal ranks gave two rows holding the same 0 scores near 1.0 and near 0.0, by argsort position."""
    X = pd.DataFrame({"flag": np.array([0.0] * 50 + [1.0] * 50)})
    ext, _ = _compute_extremality_matrix(X, ["flag"])
    assert np.unique(ext[:50, 0]).size == 1 and np.unique(ext[50:, 0]).size == 1


def test_within_batch_matches_the_reference_on_the_fitting_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"c": rng.integers(0, 5, size=300).astype(float)})
    batch, _ = _compute_extremality_matrix(X, ["c"])
    ref = fit_extremality_reference(X, ["c"])
    vs_ref, _ = extremality_matrix_from_reference(X, ref, ["c"])
    np.testing.assert_allclose(batch, vs_ref, atol=1e-12)
