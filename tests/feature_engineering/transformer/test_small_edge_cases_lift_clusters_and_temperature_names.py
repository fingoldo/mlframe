"""Edge cases: lift against a negative global mean, k-means on a tiny fold, and close softmax temperatures."""
import numpy as np

from mlframe.feature_engineering.transformer.local_lift import _local_lift_and_pr_auc
from mlframe.feature_engineering.transformer.multi_temperature import compute_multi_temperature_attention


def test_lift_against_a_negative_global_mean_is_a_plain_ratio():
    """Global mean -0.01 and a local mean of -0.02 give a lift of 2, not -2e7 from a 1e-9 divisor."""
    y_nb = np.full((3, 4), -0.02, dtype=np.float32)
    lift, _pr, _top1 = _local_lift_and_pr_auc(y_nb, np.ones((3, 4), dtype=np.float32), -0.01, "regression")
    assert np.allclose(lift, 2.0, atol=1e-5)


def test_close_temperatures_get_distinct_column_names():
    """Temperatures 0.301 and 0.304 both formatted as 0.30 and made pl.concat raise on duplicate names."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    df = compute_multi_temperature_attention(X, y, X[:5], None, seed=1, temperatures=(0.301, 0.304), n_heads=1, head_dim=4, k=5)
    assert len(set(df.columns)) == len(df.columns)


def test_kmeans_codebook_with_three_train_rows_does_not_ask_for_more_clusters_than_rows():
    """The cluster count was capped at 4 for tiny folds but never at the row count, so 3 rows raised in KMeans."""
    from mlframe.feature_engineering.transformer.target_kmeans_codebook import compute_target_kmeans_codebook_features

    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 2)).astype(np.float32)
    y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    df = compute_target_kmeans_codebook_features(X, y, rng.standard_normal((4, 2)).astype(np.float32), seed=1)
    assert df.height == 4
