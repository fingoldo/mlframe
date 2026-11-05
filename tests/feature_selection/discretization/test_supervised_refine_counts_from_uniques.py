"""The near-collapse refinement counts its bins from the column's distinct values when the caller has them; the edges equal the per-row count."""

import numpy as np

from mlframe.feature_selection.filters._supervised_collapse_refine import refine_near_collapsed_supervised_edges


def test_counts_from_uniques_give_the_per_row_edges():
    """Dominated and balanced bins, ties, float32 columns and edges equal to column values: the same edges either way."""
    rng = np.random.default_rng(0)
    for t in range(300):
        n = int(rng.integers(20, 3000))
        dtype = np.float32 if t % 2 else np.float64
        x = np.round(rng.exponential(size=n) ** int(rng.integers(1, 4)), int(rng.integers(0, 3))).astype(dtype)
        uniq, counts = np.unique(x, return_counts=True)
        k = int(rng.integers(1, 4))
        edges = np.sort(rng.choice(uniq, size=min(k, uniq.size), replace=False).astype(np.float64)) if t % 3 else np.quantile(x, [0.97])
        plain = refine_near_collapsed_supervised_edges(edges, x, "quantile", 5)
        fast = refine_near_collapsed_supervised_edges(edges, x, "quantile", 5, uniq, counts)
        assert np.array_equal(plain, fast)
