"""Smoke test for mlframe.votenrank.fairness_computation (W5-4)."""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_import_fairness_computation_module():
    """Module imports cleanly and exposes its public callables.

    Heavy deps (transformers, torch) gated behind importorskip so CI without GPU/HF
    stack still passes.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mlframe.votenrank import fairness_computation as fc

    for name in (
        "naive_masking_score",
        "naive_t5_score",
        "naive_gpt2_score",
        "naive_model_scores",
        "crows_pipeline",
        "stereo_pipeline",
        "winobias_pipeline",
    ):
        assert callable(getattr(fc, name)), f"{name} not callable"
