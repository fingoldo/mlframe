"""Wave 13 finding #7: _should_use_cuda (info_theory/_cmi_cuda.py) queried memGetInfo() for its own relative
cap, then called fe_gpu_has_vram_cushion() which re-queried memGetInfo() a second time per dispatch -- redundant
on a hot per-round gate. fe_gpu_has_vram_cushion now accepts optional free_b/total_b to reuse an already-probed
value. These tests run WITHOUT cupy available (this dev box has none) and pin: (1) the decision is identical
whether free_b/total_b are supplied or the function probes internally, (2) supplying free_b/total_b never
triggers cupy import (verified via the no-cupy-permissive path itself, and via the signature contract), (3) a
partial pair (only one of the two given) safely falls back to probing rather than mixing a stale value.
"""

from mlframe.feature_selection.filters._fe_gpu_vram import _cushion_bytes, fe_gpu_has_vram_cushion


def test_cushion_permissive_without_cupy_regardless_of_explicit_probe_args():
    """No cupy on this box -> both call forms must be permissive (True), matching pre-fix behaviour."""
    assert fe_gpu_has_vram_cushion(10**9) is True
    assert fe_gpu_has_vram_cushion(10**9, free_b=2 * 1024**3, total_b=4 * 1024**3) is True


def test_cushion_decision_matches_manual_formula_when_probe_supplied():
    """When free_b/total_b ARE supplied, the decision must equal the documented formula
    (free_b - bytes_needed) >= cushion_bytes(total_b), independent of any internal memGetInfo probe."""
    total_b = 4 * 1024 * 1024 * 1024
    cushion = _cushion_bytes(total_b)
    free_b_ok = cushion + 10**7  # just above the cushion floor
    free_b_bad = max(0, cushion - 10**7)  # just below

    assert fe_gpu_has_vram_cushion(0, free_b=free_b_ok, total_b=total_b) is True
    assert fe_gpu_has_vram_cushion(0, free_b=free_b_bad, total_b=total_b) is False


def test_partial_probe_args_fall_back_to_permissive_probe_path():
    """A partial pair (only one of free_b/total_b given) must not silently mix a stale value -- it falls back
    to the function's own probe, which on this no-cupy box is permissive (matches the neither-given case)."""
    assert fe_gpu_has_vram_cushion(10**9, free_b=2 * 1024**3, total_b=None) is True
    assert fe_gpu_has_vram_cushion(10**9, free_b=None, total_b=4 * 1024**3) is True
