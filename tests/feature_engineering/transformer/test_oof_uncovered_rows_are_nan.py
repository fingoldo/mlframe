"""Mode-A out-of-fold outputs are NaN for rows no validation fold covers, not a real 0.0 (audit 2026-11-20, finding 08-F2)."""
import numpy as np

from mlframe.feature_engineering.transformer.baseline_surprise import compute_baseline_surprise_features


class _SecondHalfOnly:
    """One fold whose validation rows are the second half of the data: the first half is covered by no fold."""

    def split(self, X):
        """Yield a single (train, validation) index pair."""
        n = len(X)
        yield np.arange(n // 2), np.arange(n // 2, n)


def test_rows_outside_every_validation_fold_are_nan():
    """The first half is never validated, so its out-of-fold features must be NaN; the covered half must be finite."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    df = compute_baseline_surprise_features(X, y, None, splitter=_SecondHalfOnly(), seed=0, task="regression")
    col = df.to_numpy()[:, 0]
    assert np.isnan(col[: n // 2]).all()
    assert np.isfinite(col[n // 2 :]).all()
