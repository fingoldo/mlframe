"""Audit 2026-11-20 finding 25-F39: a failing numba warmup kernel does not skip the others.

numba_warmup used to share one try across its six compiles. The test makes the first kernel fail and checks that the others are still
attempted, each one warmed through its own guard.
"""

import mlframe.metrics._core_numba_warmup as warmup_module
import mlframe.metrics.core as core


def test_one_failing_kernel_does_not_skip_the_rest(monkeypatch):
    """The binary seq kernel raises; the parallel kernel and the batch ICE kernels are still called."""
    called = []

    def failing(*args, **kwargs):
        """Stand in for a kernel whose signature no longer matches."""
        called.append("binary_seq")
        raise TypeError("signature mismatch")

    def recording(name):
        """Build a kernel stand-in that records its name and returns."""

        def _kernel(*args, **kwargs):
            """Record this kernel as called."""
            called.append(name)

        return _kernel

    monkeypatch.setattr(core, "_cb_logits_to_probs_binary_seq", failing)
    monkeypatch.setattr(core, "_cb_logits_to_probs_binary_par", recording("binary_par"))
    monkeypatch.setattr(core, "_cb_logits_to_probs_multiclass_seq", recording("multiclass_seq"))
    monkeypatch.setattr(core, "_cb_logits_to_probs_multiclass_par", recording("multiclass_par"))
    monkeypatch.setattr(core, "_batch_per_class_ice_kernel", recording("ice"))
    monkeypatch.setattr(core, "_batch_per_class_ice_kernel_serial", recording("ice_serial"))
    warmup_module.numba_warmup()
    assert called[0] == "binary_seq"
    assert {"binary_par", "multiclass_seq", "multiclass_par", "ice", "ice_serial"} <= set(called)
