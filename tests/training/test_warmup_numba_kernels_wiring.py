"""Regression sensor: ``_warmup_numba_kernels()`` MUST call
``mlframe.metrics.core.prewarm_numba_cache()`` to pre-compile the
metric / calibration / classification-report kernel chain.

Pre-fix the suite's warmup path only warmed the dummy_baselines
bootstrap kernels (RMSE / MAE / log-loss). Fast_calibration_report's
~15 inner kernels JIT-compiled LAZILY inside report_probabilistic_model_perf
on the first binary_classification fit, charging 10-16s of compile cost
to the training-phase profile. The 500k binary_classification x lgb
fuzz profile 2026-05-19 measured this as 27.2s cold vs 10.1s warm on
the SAME combo + seed -- the 17.1s delta sat entirely in
``numba.dispatcher._compile_for_args`` under fast_calibration_report.

This sensor pins the wiring by replacing prewarm_numba_cache with a
counter and asserting _warmup_numba_kernels() bumped it. A regression
that drops the import + call would fail by leaving the counter at 0.
"""

from __future__ import annotations


import pytest


def test_warmup_numba_kernels_calls_prewarm_metric_cache(monkeypatch):
    """The suite's warmup must trigger BOTH the dummy_baselines bootstrap
    kernels AND the metric / calibration kernel chain. A regression that
    drops the ``from ..metrics.core import prewarm_numba_cache`` import
    or its invocation would fail here."""
    pytest.importorskip("numba")

    # Replace the metric prewarm BEFORE _warmup_numba_kernels imports it
    # (the import is local to the warmup function body so monkeypatching
    # the module attribute reaches the call site).
    seen = {"count": 0}

    def _spy_prewarm():
        seen["count"] += 1

    monkeypatch.setattr(
        "mlframe.metrics.core.prewarm_numba_cache",
        _spy_prewarm,
    )

    from mlframe.training.baselines.dummy import _warmup_numba_kernels

    _warmup_numba_kernels(verbose=False)

    assert seen["count"] >= 1, (
        "_warmup_numba_kernels did NOT invoke prewarm_numba_cache; metric "
        "/ calibration / classification-report kernels will JIT-compile "
        "lazily during the first suite call, charging 10-16s of compile "
        "to the training-phase profile."
    )


def test_warmup_numba_kernels_survives_metric_prewarm_failure(monkeypatch):
    """If the metric prewarm raises (numba runtime hiccup / stale .nbc /
    library upgrade), _warmup_numba_kernels must NOT propagate the
    exception -- the suite continues with lazy compile as the fallback.
    """
    pytest.importorskip("numba")

    def _broken_prewarm():
        raise RuntimeError("simulated cache miss")

    monkeypatch.setattr(
        "mlframe.metrics.core.prewarm_numba_cache",
        _broken_prewarm,
    )

    from mlframe.training.baselines.dummy import _warmup_numba_kernels

    # Must not raise.
    _warmup_numba_kernels(verbose=False)
