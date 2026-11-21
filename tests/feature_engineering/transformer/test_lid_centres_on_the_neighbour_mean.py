"""Local PCA centres on the neighbour mean, so neighbours on a line give one dimension even when the query is off it (audit 2026-11-20, finding 09-F13)."""
import numpy as np

from mlframe.feature_engineering.transformer.local_intrinsic_dim import compute_local_intrinsic_dim_features


def test_neighbours_on_a_line_give_one_dimension_with_an_off_line_query():
    """Points on a line far from the origin, query a little off it: the top eigenvalue must carry all the local variance (top1_ratio ~ 1)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 300)
    X = np.column_stack([t, 2.0 * t, np.full_like(t, 50.0)]) + 0.001 * rng.standard_normal((300, 3))
    Xq = np.array([[5.0, 10.0, 50.5]])
    df = compute_local_intrinsic_dim_features(X, np.zeros(300), Xq, splitter=None, seed=0, k_neighbors=20)
    top1_ratio = df.to_numpy()[0, 1]
    assert top1_ratio > 0.95, top1_ratio
