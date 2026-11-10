"""The per-pair usability table serves later calls over other pairs of the same operand rows with the values a fresh call computes."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _pair_usability_table as T
from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as U

pytestmark = pytest.mark.skipif(not U._CUDA_AVAIL, reason="the table stores values of the warp kernel")


def _fresh(y, ops, pa, pb, forms):
    """The warp kernel's table for the pairs, without any memory."""
    return U._same_dtype(U._dispatch_batch_pair_usability_corr_uncached(y, ops, pa, pb, forms, "cuda_warp"))[0]


def test_subsets_reorders_and_supersets_equal_fresh_calls(monkeypatch):
    """Overlapping pair sets over permuted copies of the same rows return the fresh values; only unseen pairs reach the kernel."""
    T.clear()
    monkeypatch.setattr(U, "_CORR_MEMO_MIN_WORK", 0)
    rng = np.random.default_rng(0)
    n, k = 3000, 24
    y = rng.standard_normal(n).astype(np.float32)
    ops = (rng.standard_normal((k, n)) + 3.0).astype(np.float32)
    ops[5, 7] = np.nan  # an operand the double kernel scores
    forms = np.asarray(U.ALL_PAIR_FORM_IDS, dtype=np.int64)
    ia, ib = np.triu_indices(k, 1)
    first = rng.permutation(len(ia))[:200]
    pa1, pb1 = ia[first].astype(np.int64), ib[first].astype(np.int64)
    out1, be1 = U.dispatch_batch_pair_usability_corr(y, ops, pa1, pb1, forms)
    assert be1 == "cuda_warp"
    assert np.array_equal(out1, _fresh(y, ops, pa1, pb1, forms))

    # a subset of the first pairs over a permuted copy of the rows: served from the table
    perm = rng.permutation(k)
    inv = np.argsort(perm)
    ops2 = np.ascontiguousarray(ops[perm])
    sub = pa1[:80], pb1[:80]
    calls = []
    real = U._dispatch_batch_pair_usability_corr_uncached
    monkeypatch.setattr(U, "_dispatch_batch_pair_usability_corr_uncached", lambda *a, **kw: (calls.append(len(a[2])), real(*a, **kw))[1])
    out2, _ = U.dispatch_batch_pair_usability_corr(y, ops2, inv[sub[0]], inv[sub[1]], forms)
    assert not calls, "a subset of known pairs must not reach the kernel"
    assert np.array_equal(out2, out1[:80])

    # a superset: known pairs from the table, new pairs from the kernel, all equal to a fresh call
    pa3, pb3 = ia[:120].astype(np.int64), ib[:120].astype(np.int64)
    out3, _ = U.dispatch_batch_pair_usability_corr(y, ops, pa3, pb3, forms)
    known = {(int(a), int(b)) for a, b in zip(pa1, pb1)}
    assert calls == [sum((int(a), int(b)) not in known for a, b in zip(pa3, pb3))]
    monkeypatch.setattr(U, "_dispatch_batch_pair_usability_corr_uncached", real)
    assert np.array_equal(out3, _fresh(y, ops, pa3, pb3, forms))
    T.clear()


def test_prefetch_fills_the_table_with_the_values_a_scan_computes(monkeypatch):
    """After the prefetch thread ran, a scan-style call over the same raw columns is served from the table and equals the kernel's values."""
    import pandas as pd

    T.clear()
    monkeypatch.setattr(T, "_PREFETCH_MIN_WORK", 0)
    monkeypatch.setattr(T, "_PREFETCH_BLOCK", 7)
    monkeypatch.setattr(U, "_CORR_MEMO_MIN_WORK", 0)
    rng = np.random.default_rng(1)
    n, k = 2500, 20
    X = pd.DataFrame((rng.standard_normal((n, k)) + 2.0).astype(np.float32), columns=[f"x{i}" for i in range(k)])

    class Est:
        """Carries the continuous target like the fitted estimator."""

        _fe_prewarp_y_continuous_ = rng.standard_normal(n).astype(np.float32)

    est = Est()
    T.start_prefetch(est, X)
    T.join_prefetch(est)
    y = np.asarray(est._fe_prewarp_y_continuous_, dtype=np.float32)
    forms = np.asarray(U.ALL_PAIR_FORM_IDS, dtype=np.int64)
    ops = np.ascontiguousarray(X.to_numpy(dtype=np.float32).T)
    ia, ib = np.triu_indices(k, 1)
    pa, pb = ia.astype(np.int64), ib.astype(np.int64)
    real = U._dispatch_batch_pair_usability_corr_uncached
    calls = []
    monkeypatch.setattr(U, "_dispatch_batch_pair_usability_corr_uncached", lambda *a, **kw: (calls.append(len(a[2])), real(*a, **kw))[1])
    out, _ = U.dispatch_batch_pair_usability_corr(y, ops, pa, pb, forms)
    assert not calls, "every pair of the raw columns must come from the prefetched table"
    monkeypatch.setattr(U, "_dispatch_batch_pair_usability_corr_uncached", real)
    assert np.array_equal(out, _fresh(y, ops, pa, pb, forms))
    T.clear()
