"""spectral_attention returns a full-width embedding when the training fold has fewer rows than k_graph or n_eigvecs."""
import numpy as np

from mlframe.feature_engineering.transformer.spectral_attention import compute_spectral_attention


def test_six_train_rows_with_default_graph_size_still_give_n_eigvecs_columns():
    """6 train rows, k_graph=10, n_eigvecs=8: NearestNeighbors needs more rows than that and eigsh returns fewer vectors than asked."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 3)).astype(np.float32)
    df = compute_spectral_attention(X, np.zeros(6, dtype=np.float32), rng.standard_normal((4, 3)).astype(np.float32), seed=1, n_eigvecs=8, k_graph=10)
    assert df.shape == (4, 8) and np.isfinite(df.to_numpy()).all()
