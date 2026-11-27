"""The device warp scan stacks the host columns in chunks: the numbers equal those of one call over all columns, and a candidate does not keep the stacked columns alive."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _oof_warp_fe as W


def _frame(n=6000, k=9, seed=0):
    """A float32 frame whose first columns carry a non-monotone effect on the binary target."""
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((n, k)).astype(np.float32)
    y = (np.sin(3.0 * xs[:, 0]) + np.abs(xs[:, 1]) + xs[:, 2] ** 2 + rng.standard_normal(n) > 1.5).astype(np.int64)
    return pd.DataFrame(xs, columns=[f"c{j}" for j in range(k)]), y


def _stats(cands):
    """The per-candidate numbers the scan returns."""
    return [(c["col"], c["cls"], c["mi_raw"], c["mi_warp"], c["se"], c["bulk_gain"], c["bulk_se"], c["scored_on"]) for c in cands]


@pytest.fixture(scope="module")
def device_scan():
    """The frame and the one-call device scan of all its columns; skips the module when the device scan is not available on this host."""
    X, y = _frame()
    whole = W.warp_candidates(X, y, list(X.columns), device=True)
    if not whole or whole[0]["scored_on"] != "device":
        pytest.skip("the device warp scan is not available on this host")
    return X, y, whole


def test_chunks_give_the_numbers_of_one_call_and_candidates_do_not_hold_the_columns(device_scan, monkeypatch):
    """A chunk budget of two columns against the whole frame at once: the same candidates (numbers to 1e-12); the device scan's candidates read their column from the frame on demand."""
    X, y, whole = device_scan
    n_bytes = len(X) * np.dtype(W._crit_np_dtype()).itemsize
    monkeypatch.setattr(W, "_STACK_CHUNK_BYTES", 2 * n_bytes)
    chunked = W.warp_candidates(X, y, list(X.columns), device=True)
    assert len(chunked) == len(whole)
    for a, b in zip(_stats(chunked), _stats(whole)):
        assert a[:2] == b[:2] and a[-1] == b[-1]
        np.testing.assert_allclose(a[2:-1], b[2:-1], rtol=1e-12, atol=1e-14)  # device reductions over another column partition: equal to the last bit or two
    assert all(c["x"] is None for c in chunked)
    cand = chunked[0]
    fit = W.candidate_fit(cand)
    assert np.array_equal(W.training_column(cand).shape, (len(X),)) and "oof" in fit
