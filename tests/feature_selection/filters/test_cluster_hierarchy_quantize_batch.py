"""The batched quantiser of the cluster hierarchy gives the codes and bin counts of the per-column one (finite, tied, binary, constant and NaN-bearing columns; frame and array input)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cluster_hierarchy import _quantize_for_su
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def _reference(X):
    """The former per-column quantiser."""
    n_bins = 10
    cols, nbins = [], []
    arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    for j in range(arr.shape[1]):
        col = arr[:, j].astype(_crit_np_dtype(), copy=False)
        edges = np.unique(np.quantile(col[np.isfinite(col)] if np.isfinite(col).any() else col, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 3:
            binned, nb = np.zeros(col.shape, dtype=np.int32), 1
        else:
            binned = np.searchsorted(edges[1:-1], col, side="right").astype(np.int32)
            nb = int(binned.max()) + 1
        cols.append(binned)
        nbins.append(nb)
    return np.column_stack(cols), np.asarray(nbins, dtype=np.int64)


def _data(n=4000, seed=0):
    """Continuous, rounded (ties), binary, constant, few-valued, skewed and NaN-bearing columns."""
    rng = np.random.default_rng(seed)
    d = {"cont": rng.normal(size=n), "round": np.round(rng.normal(size=n), 1), "binary": (rng.random(n) > 0.3).astype(float), "const": np.full(n, 2.0), "few": rng.integers(0, 4, n).astype(float), "skew": np.exp(rng.normal(size=n))}
    nan_col = rng.normal(size=n)
    nan_col[7] = np.nan
    d["nan"] = nan_col
    return d


def test_frame_and_array_inputs_match_the_per_column_quantiser():
    """Same factor matrix and bin counts as the reference for both input kinds."""
    frame = pd.DataFrame(_data()).astype(np.float32)
    for X in (frame, frame.to_numpy()):
        fd, fn, names = _quantize_for_su(X)
        ref_fd, ref_fn = _reference(X)
        np.testing.assert_array_equal(fd, ref_fd)
        np.testing.assert_array_equal(fn, ref_fn)
        assert len(names) == frame.shape[1]
