"""target_kmeans_codebook clusters on out-of-fold baseline predictions and reports cluster sizes as shares of the train rows."""
import numpy as np

from mlframe.feature_engineering.transformer.target_kmeans_codebook import compute_target_kmeans_codebook_features


def test_clustering_space_does_not_carry_the_in_sample_fit_of_a_noise_target(monkeypatch):
    """y is noise; the baseline-prediction coordinate the k-means sees must be uncorrelated with it (in-sample it tracked y closely)."""
    from sklearn.cluster import MiniBatchKMeans

    seen = {}
    real_fit = MiniBatchKMeans.fit

    def spy(self, X, *args, **kwargs):
        """Keep the matrix the clustering is fitted on."""
        seen["X"] = np.asarray(X)
        return real_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(MiniBatchKMeans, "fit", spy)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)
    compute_target_kmeans_codebook_features(X, y, X[:10], seed=1, task="regression")
    assert abs(float(np.corrcoef(seen["X"][:, -1], y)[0, 1])) < 0.3


def test_cluster_size_is_a_share_of_the_train_rows():
    """A raw count was fold-sized in Mode A and full-sized in Mode B; shares are comparable and lie in [0, 1]."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)
    df = compute_target_kmeans_codebook_features(X, y, X[:20], seed=1, task="regression")
    size = df["tkmc_cluster_size"].to_numpy()
    assert size.min() > 0.0 and size.max() <= 1.0
