"""The SIS redundancy dedup correlates survivors on a row-capped strided subset, not a full-height float64 copy.

The gather ``Xarr[:, survivors]`` upcast to float64 scaled with n (16 GB at n=1M, m=2000) while the survivor RAM cap only budgeted an int16
matrix. A near-duplicate pair must still collapse on the subset.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _mrmr_sis_screen
from mlframe.feature_selection import hybrid_selector


def test_sis_dedup_gather_respects_the_row_cap(monkeypatch):
    """With the cap shrunk to 300 rows on a 2000-row frame, corr_clusters sees at most 300 rows and the duplicate still collapses."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    X = np.column_stack([x, x + 1e-3 * rng.normal(size=n), rng.normal(size=(n, 8))])
    y = (x > 0).astype(np.int64)
    seen = {}
    real = hybrid_selector.corr_clusters

    def spy(df, thr):
        """Record the gathered frame's row count."""
        seen["rows"] = df.shape[0]
        return real(df, thr=thr)

    monkeypatch.setattr(hybrid_selector, "corr_clusters", spy)
    monkeypatch.setattr(_mrmr_sis_screen, "_SIS_DEDUP_MAX_ROWS", 300, raising=False)
    survivors = np.asarray(_mrmr_sis_screen.sis_screen(X, y, target_survivors=10, chunk_width=10, dedup_corr_thr=0.92))
    assert "rows" in seen, "fixture precondition: the dedup gather must run"
    assert seen["rows"] <= 300, f"dedup gathered {seen['rows']} rows despite the 300-row cap"
    assert not ({0, 1} <= set(survivors.tolist())), f"the near-duplicate pair was not collapsed on the subset: {survivors}"
