"""Regression: a failing numba/GPU batch dispatch inside ``_mi_classif_batch_numba`` must WARN before
falling back to the sklearn reference loop, not fail silently.

BUG FOUND AND FIXED (2026-08-02, incidental to a profiling cycle): the ``except Exception`` around the
dense-column batch dispatch call in ``_mi_classif_batch_numba`` swallowed every exception with zero
logging, so a 2M-row cProfile (combo `c0056_f76bf491`) showed `_mi_classif_batch_sklearn` costing 149.3s
cumtime across 23 calls (~12% of a 1271s run) with no trail explaining why the ~53x-faster batch path was
skipped that often. Fixed by adding a `logger.warning` capturing the exception type/message before the
fallback runs; the fallback's own correctness (still returns the right MI values via sklearn) is
unchanged.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np


def test_dispatch_failure_falls_back_to_sklearn_and_warns(caplog):
    """A failing plugin_mi_classif_batch_dispatch call must log a warning naming the exception, then still
    return correct MI values via the sklearn fallback."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as m

    rng = np.random.default_rng(0)
    n, p = 200, 3
    X = rng.standard_normal((n, p))
    y = (X[:, 0] > 0).astype(np.int64)

    def raise_runtime_error(*args, **kwargs):
        """Simulate a dispatch-layer failure (e.g. transient GPU/kernel-tuning fault)."""
        raise RuntimeError("simulated: kernel tuning cache miss")

    with (
        patch("mlframe.feature_selection.filters.hermite_fe.plugin_mi_classif_batch_dispatch", side_effect=raise_runtime_error),
        caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends"),
    ):
        mis = m._mi_classif_batch_numba(X, y)

    assert any(
        "falling back to the sklearn reference loop" in rec.message for rec in caplog.records
    ), f"expected a warning naming the sklearn fallback; got: {[rec.message for rec in caplog.records]}"
    assert any("RuntimeError" in rec.message for rec in caplog.records), "warning must name the exception type that triggered the fallback"

    expected = m._mi_classif_batch_sklearn(np.ascontiguousarray(X), np.ascontiguousarray(y, dtype=np.int64))
    np.testing.assert_allclose(mis, expected)


def test_successful_dispatch_does_not_warn(caplog):
    """The happy path (batch dispatch succeeds) must not log the fallback warning."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as m

    rng = np.random.default_rng(1)
    n, p = 200, 3
    X = rng.standard_normal((n, p))
    y = (X[:, 0] > 0).astype(np.int64)

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends"):
        m._mi_classif_batch_numba(X, y)

    assert not any("falling back to the sklearn reference loop" in rec.message for rec in caplog.records)
