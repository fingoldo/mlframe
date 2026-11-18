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
