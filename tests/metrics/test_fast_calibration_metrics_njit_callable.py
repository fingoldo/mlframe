"""Regression test: ``fast_calibration_metrics`` must be a working njit function.

iter136 SUITE-ORCHESTRATION profiling surfaced that ``fast_calibration_metrics``
(``@numba.njit``) called ``fast_calibration_binning`` -- a plain-Python size
dispatcher, not an njit kernel. Referencing a non-njit global from inside a
nopython body fails type inference with ``TypingError: Untyped global name
'fast_calibration_binning'``, so:

  1. the public ``fast_calibration_metrics`` API was completely broken (any
     call raised TypingError); and
  2. that call lived in the first un-guarded loop of ``prewarm_numba_cache``,
     so it ABORTED the entire suite metric-kernel prewarm -- every kernel after
     it in the ~480-line warmup body never compiled, defeating the on-disk
     numba-cache pre-warm optimization (``[dummy-baselines] metric kernel
     pre-warmup failed`` logged once per suite).

Fix: call the serial njit binning kernel ``_fast_calibration_binning_serial``
directly. Output is bit-identical to the dispatcher path at n below the prange
threshold.
"""
import numpy as np


def test_fast_calibration_metrics_is_njit_callable_and_matches_dispatcher():
    """``fast_calibration_metrics`` must compile + run (pre-fix: raised numba
    TypingError) AND produce output bit-identical to the public dispatcher path."""
    from mlframe.metrics.core import (
        fast_calibration_metrics,
        fast_calibration_binning,
        calibration_metrics_from_freqs,
    )

    rng = np.random.default_rng(0)
    y_pred = rng.random(500).astype(np.float64)
    y_true = (rng.random(500) < y_pred).astype(np.float64)

    out = fast_calibration_metrics(y_true, y_pred, nbins=10)

    fp, ft, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
    ref = calibration_metrics_from_freqs(
        freqs_predicted=fp, freqs_true=ft, hits=hits,
        nbins=10, array_size=len(y_true), use_weights=False,
    )
    assert out == ref, f"fast_calibration_metrics output {out} != dispatcher path {ref}"


def test_prewarm_numba_cache_completes_without_aborting():
    """The metric-kernel prewarm must run to completion. Pre-fix the
    ``fast_calibration_metrics`` call in the first prewarm loop raised
    TypingError, aborting every subsequent kernel warmup."""
    from mlframe.metrics.core import prewarm_numba_cache
    prewarm_numba_cache()  # pre-fix: raised numba.core.errors.TypingError
