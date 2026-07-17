"""G3 regression (2026-06-22): GPU FE K-chunk width is KTC-tuned via a VRAM fraction, default-safe.

The K-chunk width that bounds the resident candidate-MI working set was governed by a hardcoded
0.25 * free_VRAM fraction. Per ``feedback_use_kernel_tuning_cache_for_gpu`` that fraction is now
looked up per-host from the kernel_tuning_cache (wider = fewer launches on high-VRAM cards), with the
conservative 0.25 as the source-code fallback. These tests pin:

* the default fraction reproduces the legacy 0.25 chunk-width math byte-for-byte (no regression on the
  un-tuned / no-cache-entry path);
* the fraction resolver clamps/falls back safely (never zeros the budget, never exceeds 90% VRAM);
* the tuner spec is registered;
* (GPU) the candidate MI is selection-INVARIANT across fractions (chunk width is per-column-independent),
  so a wider tuned fraction never changes which features are selected.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_resident_fe as g


def test_default_fraction_matches_legacy_025_math():
    fb = 4_000_000_000
    for n in (50_000, 100_000, 300_000):
        legacy = min(len(g._COMBOS), max(1, int(fb * 0.25) // (n * 8 * g._GPU_MI_WORKING_MULTIPLE)))
        got = g._gpu_k_chunk(n, free_bytes=fb)  # no vram_fraction -> resolver -> default 0.25 (no cache)
        assert got == legacy, f"n={n}: default-fraction k_chunk {got} != legacy {legacy}"


def test_explicit_wider_fraction_widens_chunk():
    fb = 4_000_000_000
    n = 100_000
    narrow = g._gpu_k_chunk(n, free_bytes=fb, vram_fraction=0.25)
    wide = g._gpu_k_chunk(n, free_bytes=fb, vram_fraction=0.70)
    assert wide >= narrow


def test_fraction_resolver_clamps_and_defaults():
    # No/garbage cache entry -> conservative default.
    assert g._gpu_k_chunk_vram_fraction(100_000) == g._GPU_K_CHUNK_VRAM_FRACTION_DEFAULT
    # Clamp: a corrupt huge/zero/negative fraction must never zero or blow the budget.
    fb = 4_000_000_000
    assert g._gpu_k_chunk(100_000, free_bytes=fb, vram_fraction=0.0) >= 1
    assert g._gpu_k_chunk(100_000, free_bytes=fb, vram_fraction=-5.0) >= 1
    cap = len(g._COMBOS)
    assert g._gpu_k_chunk(100_000, free_bytes=fb, vram_fraction=100.0) <= cap


def test_tuner_spec_registered():
    assert g._GPU_K_CHUNK_SPEC is not None
    # fallback returns the safe default fraction string
    assert g._gpu_k_chunk_fallback_choice(100_000) == f"frac_{g._GPU_K_CHUNK_VRAM_FRACTION_DEFAULT}"


def test_candidate_mi_selection_invariant_across_fractions():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(0)
    n = 60_000
    a = rng.normal(size=n).astype(np.float64)
    b = rng.normal(size=n).astype(np.float64)
    y = (a * b > 0).astype(np.int64)
    _, mi_default = g.gpu_resident_pair_candidate_mi(a, b, y, nbins=20)
    _, mi_wide = g.gpu_resident_pair_candidate_mi_vram_fraction(a, b, y, nbins=20, vram_fraction=0.70)
    # Chunk width is per-column-independent -> MI must be bit-identical regardless of chunking.
    np.testing.assert_array_equal(np.argsort(-mi_default), np.argsort(-mi_wide))
    np.testing.assert_allclose(mi_default, mi_wide, rtol=1e-9, atol=1e-9)
