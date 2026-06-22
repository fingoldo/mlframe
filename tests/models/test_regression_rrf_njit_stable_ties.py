"""Regression: the RRF njit kernel must tie-break identically to the numpy fallback.

Pre-fix the njit kernel used ``np.argsort(kind="quicksort")`` (unstable) while the numpy fallback used
``kind="stable"``; the dispatcher routed float64 / C-contiguous inputs to njit by default, so production diverged
from the fallback on TIED probabilities.
"""
import numpy as np
import pytest

from mlframe.models.ensembling import base as _base


@pytest.mark.skipif(_base._rrf_aggregate_probs_njit is None, reason="numba per-member kernel unavailable")
def test_rrf_njit_matches_numpy_fallback_on_tied_probabilities():
    rng = np.random.default_rng(0)
    M, N, K = 4, 40, 3
    preds = rng.random((M, N, K))
    # Force heavy ties: snap probabilities to a small grid so many rows share identical column values.
    preds = np.round(preds * 3) / 3.0
    preds = np.ascontiguousarray(preds, dtype=np.float64)

    njit_out = _base._rrf_aggregate_probs_njit(preds, 60)

    # Reproduce the numpy fallback exactly (the dispatcher's else-branch).
    aggregated = np.zeros((N, K), dtype=np.float64)
    for kc in range(K):
        col = preds[:, :, kc]
        order = np.argsort(-col, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(N), axis=1)
        rr = 1.0 / (60 + (ranks + 1).astype(np.float64))
        aggregated[:, kc] = rr.sum(axis=0)
    for n in range(N):
        s = aggregated[n].sum()
        if s > 0.0:
            aggregated[n] /= s

    np.testing.assert_allclose(njit_out, aggregated, rtol=0, atol=1e-12)
