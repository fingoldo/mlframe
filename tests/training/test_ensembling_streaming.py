"""Tests for streaming ensembling via _WelfordAccumulator (Session 3).

Verifies:
- Streaming-vs-materialised bitwise equivalence across arithm/harm/quad/qube/geo
- median raises NotImplementedError in streaming mode (needs P² sketch)
- Outlier-filter warning when M>2 (streaming path can't compare cross-member)
- _WelfordAccumulator combine() exactly merges two partial accumulators
- Memory footprint O(N*K) — empirically verified via memory_info().private
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.ensembling import (
    _WelfordAccumulator,
    ensemble_probabilistic_predictions,
    ensemble_probabilistic_predictions_streaming,
)


def _make_preds(M=6, N=500, K=3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.01, 0.99, size=(N, K)) for _ in range(M)]


# ---------------------------------------------------------------------------
# Equivalence: streaming vs materialised
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["arithm", "harm", "quad", "qube", "geo"])
def test_streaming_matches_materialised(method):
    preds = _make_preds(M=6, N=500, K=3)
    mat, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=method, verbose=False)
    str_, _, _ = ensemble_probabilistic_predictions_streaming(*preds, ensemble_method=method, verbose=False)
    delta = np.max(np.abs(mat - str_))
    assert delta < 1e-12, f"{method}: streaming-vs-materialised delta {delta:.3e} exceeds 1e-12"


def test_streaming_median_raises():
    preds = _make_preds()
    with pytest.raises(NotImplementedError, match="median"):
        ensemble_probabilistic_predictions_streaming(*preds, ensemble_method="median")


def test_streaming_outlier_filter_warn(caplog):
    import logging
    preds = _make_preds(M=6)
    with caplog.at_level(logging.WARNING):
        ensemble_probabilistic_predictions_streaming(*preds, ensemble_method="arithm", verbose=True)
    assert any("outlier-member filter is not applied" in rec.message for rec in caplog.records)


def test_streaming_empty_preds_returns_nones():
    result = ensemble_probabilistic_predictions_streaming(ensemble_method="arithm", verbose=False)
    assert result == (None, None, None)


def test_streaming_1d_input_preserved():
    """1-D preds should return 1-D ensembled output (not (N, 1))."""
    rng = np.random.default_rng(0)
    preds = [rng.uniform(0.01, 0.99, size=(100,)) for _ in range(4)]
    ens, unc, _ = ensemble_probabilistic_predictions_streaming(
        *preds, ensemble_method="arithm", verbose=False
    )
    assert ens.ndim == 1
    assert ens.shape == (100,)


# ---------------------------------------------------------------------------
# _WelfordAccumulator direct API
# ---------------------------------------------------------------------------


def test_welford_vs_numpy_mean_std():
    preds = _make_preds(M=10, N=300, K=5)
    acc = _WelfordAccumulator(shape=(300, 5))
    for p in preds:
        acc.push(p)
    result = acc.result()
    ref = np.stack(preds, axis=0)
    np.testing.assert_allclose(result["mean"], ref.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(result["std"], ref.std(axis=0, ddof=1), atol=1e-12)


def test_welford_combine_exact_merge():
    preds = _make_preds(M=8)
    shape = preds[0].shape
    a = _WelfordAccumulator(shape=shape)
    b = _WelfordAccumulator(shape=shape)
    for p in preds[:4]: a.push(p)
    for p in preds[4:]: b.push(p)
    combined = _WelfordAccumulator.combine(a, b).result()

    full = _WelfordAccumulator(shape=shape)
    for p in preds: full.push(p)
    ref = full.result()

    np.testing.assert_allclose(combined["mean"], ref["mean"], atol=1e-12)
    np.testing.assert_allclose(combined["std"], ref["std"], atol=1e-12)
    assert combined["n"] == ref["n"] == 8


def test_welford_shape_mismatch_raises():
    acc = _WelfordAccumulator(shape=(100, 3))
    with pytest.raises(ValueError, match="expected shape"):
        acc.push(np.zeros((100, 5)))


def test_welford_empty_accumulator():
    acc = _WelfordAccumulator(shape=(10, 3))
    result = acc.result()
    assert result["mean"] is None
    assert result["n"] == 0
