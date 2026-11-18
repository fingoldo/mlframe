"""The spike-contamination verdict of a column is computed once per scope, whichever thread or array object asks."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import _hermite_robust as robust


def _columns() -> "tuple[np.ndarray, np.ndarray]":
    """A clean normal column and the same column with a handful of far spikes (large enough for the numpy body)."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(20000)
    spiked = clean.copy()
    spiked[:20] = 1e6
    return clean, spiked


def test_one_examination_per_column_across_pool_threads(monkeypatch):
    """Eight pool threads asking about copies of two columns examine each column once; the verdicts are the unmemoised ones."""
    clean, spiked = _columns()
    want = (robust._detect_heavy_tail(clean), robust._detect_heavy_tail(spiked))
    assert want == (False, True)
    examined = []
    real = robust._detect_heavy_tail_numpy

    def spy(x):
        """Count the examinations that were not served from the table."""
        examined.append(x.shape)
        return real(x)

    monkeypatch.setattr(robust, "_detect_heavy_tail_numpy", spy)
    with robust.heavy_tail_memo_scope():
        robust._detect_heavy_tail(clean)
        robust._detect_heavy_tail(spiked)
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = list(pool.map(lambda i: robust._detect_heavy_tail((spiked if i % 2 else clean).copy()), range(32)))
    assert got == [bool(i % 2) for i in range(32)]
    assert len(examined) == 2
    assert robust._HEAVY_TAIL_MEMO["cache"] is None and robust._HEAVY_TAIL_MEMO["depth"] == 0


def test_no_scope_no_memo(monkeypatch):
    """Outside a scope every call examines the column (the table exists only inside a fit)."""
    clean, _ = _columns()
    examined = []
    real = robust._detect_heavy_tail_numpy
    monkeypatch.setattr(robust, "_detect_heavy_tail_numpy", lambda x: (examined.append(1), real(x))[1])
    robust._detect_heavy_tail(clean)
    robust._detect_heavy_tail(clean)
    assert len(examined) == 2


def test_strided_column_views_are_served_from_the_table(monkeypatch):
    """A column cut out of a 2-D block (strided, the shape of ``frame[c].to_numpy()``) has a content key: asking about fresh views of the same values examines the column once."""
    clean, spiked = _columns()
    block = np.ascontiguousarray(np.column_stack([clean, spiked, clean]))  # C order: a column of it is strided
    views = [block[:, 0], block[:, 1], block[:, 2]]
    assert all(not v.flags.c_contiguous for v in views)
    examined = []
    real = robust._detect_heavy_tail_numpy
    monkeypatch.setattr(robust, "_detect_heavy_tail_numpy", lambda x: (examined.append(1), real(x))[1])
    with robust.heavy_tail_memo_scope():
        got = [robust._detect_heavy_tail(block[:, j]) for j in (0, 1, 2, 0, 1, 2, 0)]
    assert got == [False, True, False, False, True, False, False]
    assert len(examined) == 2, "views 0 and 2 hold the same values, so the table serves the second one and the repeats"
    assert robust._detect_heavy_tail(views[1]) is True  # outside a scope the verdict is still computed correctly
