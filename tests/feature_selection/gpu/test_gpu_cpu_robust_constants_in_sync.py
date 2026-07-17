"""Drift guard: the GPU robust-axis constants must equal the CPU reference.

The GPU heavy-tail / robust-scale batched ports (_gpu_resident_fe._GPU_ROBUST_AXIS_*) re-declare the
thresholds that the CPU reference (_hermite_robust._ROBUST_AXIS_*) owns. They are intentionally NOT
single-sourced via import (the GPU module deliberately avoids a top-level hermite_fe import -- a cycle the
lazy in-function imports work around). This test is the cheap substitute: if anyone retunes the CPU
thresholds without mirroring them on the GPU side (or vice-versa), the GPU heavy-tail verdict silently
diverges from the CPU one -> selection drift on the default-on GPU routing / basis-MI path. Flagged by the
code-quality + effectiveness critique agents (2026-06-22).
"""

from __future__ import annotations


def test_gpu_robust_axis_constants_match_cpu():
    """The GPU-side heavy-tail/robust-scale constants (re-declared to avoid an import cycle) must equal the CPU reference exactly, or GPU routing silently drifts."""
    from mlframe.feature_selection.filters import _gpu_resident_fe as gpu
    from mlframe.feature_selection.filters.hermite_fe import _hermite_robust as cpu

    assert gpu._GPU_ROBUST_AXIS_OUTER_K == cpu._ROBUST_AXIS_OUTER_K, f"OUTER_K drift: GPU {gpu._GPU_ROBUST_AXIS_OUTER_K} != CPU {cpu._ROBUST_AXIS_OUTER_K}"
    assert gpu._GPU_ROBUST_AXIS_GAP == cpu._ROBUST_AXIS_GAP, f"GAP drift: GPU {gpu._GPU_ROBUST_AXIS_GAP} != CPU {cpu._ROBUST_AXIS_GAP}"
    assert gpu._GPU_ROBUST_AXIS_MAX_FRAC == cpu._ROBUST_AXIS_MAX_FRAC, f"MAX_FRAC drift: GPU {gpu._GPU_ROBUST_AXIS_MAX_FRAC} != CPU {cpu._ROBUST_AXIS_MAX_FRAC}"
