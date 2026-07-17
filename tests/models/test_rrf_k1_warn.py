"""Regression sensor for A3#4 (S43): `_rrf_aggregate_probs` K=1 binary destroys calibration.

The K=1 / scalar 1-column path returns the raw reciprocal-rank score (rank-monotone
but NOT a probability). Callers stamping AUC / logloss on the result hit silently
miscalibrated outputs. W11D adds a one-line WARN so the path is grep-able in suite
logs. This sensor pins the WARN in place.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.models.ensembling.base import _rrf_aggregate_probs


def test_k1_input_emits_warn(caplog):
    rng = np.random.default_rng(0)
    preds_2d = rng.random((3, 50)).astype(np.float64)  # (M, N) -> auto-promoted to K=1
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        out = _rrf_aggregate_probs(preds_2d, k=60)
    assert out.shape == (50,)
    msgs = [r.getMessage() for r in caplog.records if "K=1" in r.getMessage()]
    assert msgs, f"K=1 path must emit a WARN; got {[r.getMessage() for r in caplog.records]}"


def test_k2_binary_does_not_emit_k1_warn(caplog):
    rng = np.random.default_rng(0)
    preds = rng.random((3, 50, 2)).astype(np.float64)
    # Normalise each row to sum 1 so we feed real probabilities.
    preds /= preds.sum(axis=2, keepdims=True)
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        out = _rrf_aggregate_probs(preds, k=60)
    assert out.shape == (50, 2)
    msgs = [r.getMessage() for r in caplog.records if "K=1" in r.getMessage()]
    assert not msgs, f"K=2 path must not emit the K=1 WARN; got {msgs}"
    # Output rows are normalised.
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
