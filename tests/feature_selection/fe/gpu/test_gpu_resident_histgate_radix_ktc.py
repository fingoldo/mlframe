"""GPU_INFRA_B-6 fix (mrmr_audit_2026-07-22): _gpu_resident_histgate_ktc.py / _gpu_resident_radix_ktc.py had
zero test coverage anywhere in the suite, unlike their structurally-identical sibling
_gpu_resident_k_chunk_ktc.py (test_gpu_k_chunk_vram_fraction_ktc.py). Mirrors that sibling's pattern:
tuner-spec-registered smoke test + fallback-choice sanity + a CPU-only "safe default" check -- this file
class already had one self-documented regression (_gpu_resident_radix_ktc.py's dead sweep-probe override,
fixed 2026-07-18), so the untested siblings carry the same class of risk with no safety net.
"""

from __future__ import annotations

from mlframe.feature_selection.filters import _gpu_resident_histgate_ktc as hg
from mlframe.feature_selection.filters import _gpu_resident_radix_ktc as rx


def test_histgate_threads_fallback_choice_is_historical_default():
    """Pre-sweep / no-cache fallback returns the historical hardcoded 128 threads/block."""
    assert hg._histgate_threads_fallback_choice(100_000) == f"th_{hg._HISTGATE_THREADS_DEFAULT}"


def test_histgate_threads_returns_int_and_defaults_safely():
    """histgate_threads always returns a positive int, falling back to the historical default when the
    tuner spec is unavailable (no cupy / lookup failure)."""
    n = hg.histgate_threads(100_000)
    assert isinstance(n, int)
    assert n > 0
    if hg._HISTGATE_THREADS_SPEC is None:
        assert n == hg._HISTGATE_THREADS_DEFAULT


def test_histgate_threads_variants_include_the_default():
    """The HW-occupancy-derived candidate set always keeps the historical default as a reference variant."""
    assert hg._HISTGATE_THREADS_DEFAULT in hg._HISTGATE_THREADS_VARIANTS


def test_radix_select_threads_fallback_choice_is_historical_default():
    """Pre-sweep / no-cache fallback returns the historical hardcoded 512 threads/block."""
    assert rx._radix_threads_fallback_choice(100_000) == f"th_{rx._RADIX_THREADS_DEFAULT}"


def test_radix_select_threads_returns_int_and_defaults_safely():
    """radix_select_threads always returns a positive int, falling back to the historical default when the
    tuner spec is unavailable (no cupy / lookup failure)."""
    n = rx.radix_select_threads(100_000)
    assert isinstance(n, int)
    assert n > 0
    if rx._RADIX_THREADS_SPEC is None:
        assert n == rx._RADIX_THREADS_DEFAULT


def test_radix_select_f32_variant_fallback_choice_is_v3():
    """Pre-sweep / no-cache fallback returns the historical default f32 variant ('v3')."""
    assert rx._radix_f32_variant_fallback_choice(100_000) == rx._RADIX_F32_VARIANT_DEFAULT


def test_radix_select_f32_variant_returns_known_variant():
    """radix_select_f32_variant always returns one of the registered f32 variants."""
    variant = rx.radix_select_f32_variant(100_000)
    assert variant in rx._RADIX_F32_VARIANTS


def test_radix_threads_variants_include_the_default():
    """The HW-occupancy-derived candidate set always keeps the historical default as a reference variant."""
    assert rx._RADIX_THREADS_DEFAULT in rx._RADIX_THREADS_VARIANTS
