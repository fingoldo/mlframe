"""The raw-column MI baseline memo of the orthogonal families is one table per fit scope, visible to every thread."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as backends
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_scoring_memo as memo


def _data() -> "tuple[list, np.ndarray, np.ndarray]":
    """Six float32 columns, the first informative, and a binary target."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(4000) > 0).astype(np.int64)
    return [f"c{j}" for j in range(6)], X, y


def test_second_family_reads_the_baseline_from_the_table(monkeypatch):
    """Inside a scope the columns are scored once: a second caller on a pool thread, with a copy of the matrix and a column subset, launches nothing and gets the same values."""
    cols, X, y = _data()
    want = memo.cached_raw_mi_baseline(cols, X, y, nbins=10)  # outside a scope: a plain batch call
    widths = []
    real = backends._mi_classif_batch

    def spy(mat, *args, **kwargs):
        """Record the width of each batch MI call."""
        widths.append(int(mat.shape[1]))
        return real(mat, *args, **kwargs)

    monkeypatch.setattr(backends, "_mi_classif_batch", spy)
    with memo.orth_scoring_memo_scope():
        first = memo.cached_raw_mi_baseline(cols, X, y, nbins=10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            second = pool.submit(memo.cached_raw_mi_baseline, cols[1:4], np.ascontiguousarray(X[:, 1:4]).copy(), y.copy(), nbins=10).result()
        with memo.orth_scoring_memo_scope():  # a nested scope reuses the table and does not clear it on exit
            pass
        third = memo.cached_raw_mi_baseline(cols[:2], X[:, :2], y, nbins=10)
    assert widths == [6]
    assert first == want
    assert second == {c: want[c] for c in cols[1:4]} and third == {c: want[c] for c in cols[:2]}
    assert memo._SCORING_MEMO["cache"] is None and memo._SCORING_MEMO["depth"] == 0
    assert max(want, key=want.get) == "c0"
