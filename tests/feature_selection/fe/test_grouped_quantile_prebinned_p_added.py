"""The batched quantile-codes binner of the grouped-quantile p-value path gives the same p-values as the per-column host binner."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _grouped_quantile_fe as gq


def _frames(n=3000, seed=0):
    """Raw frame, engineered frame (continuous, tied, few-valued and NaN-holding columns) and the integer target."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.exponential(size=n), "c": rng.integers(0, 9, n).astype(float)})
    y = (raw["a"] + 0.5 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    eng = pd.DataFrame(
        {
            "e1": raw["a"] ** 2 + rng.standard_normal(n),
            "e2": np.round(raw["b"], 1),
            "e3": rng.integers(0, 5, n).astype(float),
            "e5": rng.integers(0, 6, n) / 7.0,
            "e4": np.where(rng.random(n) < 0.05, np.nan, raw["b"]),
        }
    )
    return raw, eng, y, {"e1": "a", "e2": "b", "e3": "c", "e4": "b", "e5": "c"}


def test_prebinned_p_added_equals_host_binner(monkeypatch):
    """p_added with the batched binner equals p_added with every column binned by the host closure."""
    raw, eng, y, src = _frames()
    pre = gq._prebinned_codes(eng, list(eng.columns), 10)
    assert "e4" not in pre and {"e1", "e2"} <= set(pre) and not {"e3", "e5"} & set(pre), "non-finite and few-valued columns must stay on the host binner"

    def host_bin(arr):
        """The score function's per-column binner, restated."""
        a = np.asarray(arr, dtype=np.float32)
        fin = np.isfinite(a)
        if np.all(a[fin] == np.round(a[fin])) and np.unique(a[fin]).size <= 10:  # integer-valued with few levels: used as-is
            return np.unique(np.where(fin, a, 0.0), return_inverse=True)[1].astype(np.int64)
        q = np.unique(np.quantile(a[fin], np.linspace(0, 1, 11)))
        codes = np.searchsorted(q[1:-1], a, side="right")
        codes[~fin] = 0
        return codes.astype(np.int64)

    with_pre = gq._p_added_by_column(eng, raw, y, src, host_bin, 10)
    monkeypatch.setattr(gq, "_prebinned_codes", lambda *a, **k: {})
    without = gq._p_added_by_column(eng, raw, y, src, host_bin, 10)
    for c in eng.columns:
        assert np.isclose(with_pre[c], without[c], rtol=1e-9, atol=1e-12, equal_nan=True), (c, with_pre[c], without[c])


def test_added_information_p_with_shared_dense_codes_is_identical():
    """Passing the pre-renumbered target and source gives the same p-value as renumbering inside the call."""
    from mlframe.feature_selection.filters._added_information_gate import _dense_codes, added_information_p

    rng = np.random.default_rng(3)
    n = 4000
    y = rng.integers(0, 3, n)
    src = rng.integers(0, 7, n)
    cand = (src + rng.integers(0, 4, n)) % 9
    base = added_information_p(cand, src, y)
    shared = added_information_p(cand, src, y, y_dense=_dense_codes(y), source_dense=_dense_codes(src))
    assert shared == base
    assert added_information_p(cand, None, y, y_dense=_dense_codes(y)) == added_information_p(cand, None, y)
