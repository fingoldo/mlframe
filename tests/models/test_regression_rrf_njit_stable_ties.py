"""Regression: the RRF njit kernel must produce the same AVERAGE-rank tie handling as the numpy fallback.

Canonical RRF gives genuinely TIED items EQUAL (averaged) ranks so ties contribute identical
reciprocal-rank mass regardless of array index. Both the njit kernel and the numpy fallback now use
scipy ``rankdata(method="average")`` semantics; the dispatcher routes float64 / C-contiguous inputs to
njit by default, so this pins njit == numpy on TIED probabilities.
"""
import numpy as np
import pytest
from scipy.stats import rankdata

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

    # Reproduce the numpy fallback exactly (the dispatcher's else-branch): average-rank RRF.
    aggregated = np.zeros((N, K), dtype=np.float64)
    for kc in range(K):
        col = preds[:, :, kc]
        ranks = rankdata(-col, method="average", axis=1)  # 1-based averaged ranks
        rr = 1.0 / (60 + ranks.astype(np.float64))
        aggregated[:, kc] = rr.sum(axis=0)
    for n in range(N):
        s = aggregated[n].sum()
        if s > 0.0:
            aggregated[n] /= s

    np.testing.assert_allclose(njit_out, aggregated, rtol=0, atol=1e-12)


@pytest.mark.skipif(_base._rrf_aggregate_probs_njit is None, reason="numba per-member kernel unavailable")
def test_rrf_tied_rows_get_identical_contribution():
    """Two rows with IDENTICAL member probabilities across all members must receive identical RRF output.

    The pre-fix positional tie-break assigned distinct ordinal ranks to equal scores (broken by array
    index), so two genuinely-tied rows got DIFFERENT fused mass -- position-dependent fusion noise. The
    average-rank fix makes tied rows bit-identical.
    """
    # rows 0 and 1 are identical across every member; K=2.
    preds = np.array(
        [
            [[0.5, 0.5], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]],
            [[0.5, 0.5], [0.5, 0.5], [0.7, 0.3], [0.1, 0.9]],
            [[0.5, 0.5], [0.5, 0.5], [0.6, 0.4], [0.3, 0.7]],
        ],
        dtype=np.float64,
    )
    njit_out = _base._rrf_aggregate_probs_njit(preds, 60)
    np.testing.assert_allclose(njit_out[0], njit_out[1], rtol=0, atol=0.0)

    # numpy fallback path (float32 bypasses the njit dispatcher) agrees and is also tie-identical.
    numpy_out = _base._rrf_aggregate_probs(preds.astype(np.float32), 60)
    np.testing.assert_allclose(numpy_out[0], numpy_out[1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(njit_out, numpy_out, rtol=0, atol=1e-6)
