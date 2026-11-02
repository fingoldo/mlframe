"""The blocked correlation clustering standardises in place (tracker 2026-11-02 wave 16, MEM16-7)."""

import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.hybrid_selector import corr_clusters


def _frame():
    """A float32 frame with a duplicated column pair and a constant column."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20_000, 40)).astype(np.float32)
    X[:, 5] = X[:, 4] + rng.normal(scale=0.01, size=20_000).astype(np.float32)
    X[:, 7] = 3.0
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(40)])


def test_blocked_path_matches_the_dense_path():
    """Same reps and members whether the blocked path (threshold below p) or the dense path runs."""
    df = _frame()
    assert corr_clusters(df, 0.92, block_threshold=10) == corr_clusters(df, 0.92, block_threshold=10_000)


def test_blocked_path_peak_memory_is_about_one_extra_copy():
    """Z is the one full-size array; the old division built two more temporaries (about 3x the matrix)."""
    df = _frame()
    tracemalloc.start()
    corr_clusters(df, 0.92, block_threshold=10)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 1.6 * df.shape[0] * df.shape[1] * 4
