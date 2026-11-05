"""Disabling CUDA twice and restoring once gives back the original availability check.

The save of the original ``torch.cuda.is_available`` now happens under a lock. A race between two threads cannot be forced deterministically, so
this pins the behaviour the lock protects: a second disable must not overwrite the saved original.
"""

from __future__ import annotations

import torch

from mlframe.training.neural.base import _cuda_fallback as fb


def test_second_disable_keeps_the_saved_original(monkeypatch):
    """Disable, disable again, restore: ``torch.cuda.is_available`` is the function it was before."""
    original = torch.cuda.is_available
    monkeypatch.setattr(fb, "_CUDA_DISABLED_STATE", None)
    try:
        fb._disable_cuda_globally()
        fb._disable_cuda_globally()
        assert torch.cuda.is_available() is False
        assert fb.restore_cuda_visibility() is True
        assert torch.cuda.is_available is original
    finally:
        torch.cuda.is_available = original
