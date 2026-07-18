"""RESIDENT UPLOAD (2026-07-13): ``mi_direct_gpu_batched`` / ``mi_direct_gpu_batched_streamed`` must upload
the MRMR target (``classes_y``/``freqs_y``) ONCE across repeated calls scoring DIFFERENT candidate columns
(mirrors the greedy screening loop, which evaluates many candidates against the same fixed target within
one round), instead of a fresh ``cp.asarray`` every call. ``classes_x``/``freqs_x`` genuinely vary per
candidate and must stay a fresh per-call upload (never cached). Proves the ``resident_operand`` adoption
fix in ``_gpu_batched.py`` engages and stays bit-identical to the pre-fix raw-upload path.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters._gpu_batched import (
    mi_direct_gpu_batched,
    mi_direct_gpu_batched_streamed,
)
from mlframe.feature_selection.filters.info_theory import merge_vars


def _gpu_available() -> bool:
    """Gpu available."""
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:  # pragma: no cover - no driver / no GPU
        return False


_GPU_AVAILABLE = _gpu_available()
if not _GPU_AVAILABLE:  # pragma: no cover - guarded at collection time
    pytest.skip("No CUDA device available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_resident_cache():
    """Clear resident cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _build_two_candidate_inputs(n=2000, seed=1):
    """(factors_data, factors_nbins) with TWO candidate columns (0, 1) scored against the SAME target (2)."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 5, size=n).astype(np.int32)
    x1 = rng.integers(0, 5, size=n).astype(np.int32)
    y_raw = ((x0 + rng.normal(scale=0.6, size=n)) > 2.0).astype(np.int32)
    factors = np.column_stack([x0, x1, y_raw]).astype(np.int32)
    factors_nbins = np.array([5, 5, 2], dtype=np.int64)
    return factors, factors_nbins


def _count_asarray_calls_matching(monkeypatch, target_values):
    """Count ``cp.asarray`` calls whose input equals ``target_values`` exactly (content-based, not just
    shape -- ``classes_x``/``classes_y`` share the same 1-D shape, so shape alone can't distinguish them)."""
    calls = {"n": 0}
    orig = cp.asarray
    target = np.ascontiguousarray(target_values)

    def spy(a, *args, **kw):
        """Helper that spy."""
        if isinstance(a, np.ndarray) and a.shape == target.shape and a.dtype == target.dtype and np.array_equal(a, target):
            calls["n"] += 1
        return orig(a, *args, **kw)

    monkeypatch.setattr(cp, "asarray", spy)
    return calls


@pytest.mark.gpu
@pytest.mark.parametrize("fn", [mi_direct_gpu_batched, mi_direct_gpu_batched_streamed])
def test_target_uploaded_once_across_candidate_calls(monkeypatch, fn):
    """Target uploaded once across candidate calls."""
    factors, factors_nbins = _build_two_candidate_inputs(seed=2)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors,
        vars_indices=(2,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    classes_y_i32 = np.ascontiguousarray(classes_y.astype(np.int32))

    y_calls = _count_asarray_calls_matching(monkeypatch, classes_y_i32)

    mi0, _ = fn(
        factors,
        (0,),
        (2,),
        factors_nbins,
        npermutations=16,
        batch_size=8,
        classes_y=classes_y.copy(),
        freqs_y=freqs_y.copy(),
    )
    mi1, _ = fn(
        factors,
        (1,),
        (2,),
        factors_nbins,
        npermutations=16,
        batch_size=8,
        classes_y=classes_y.copy(),
        freqs_y=freqs_y.copy(),
    )

    assert y_calls["n"] == 1, f"classes_y-content cp.asarray called {y_calls['n']}x across 2 candidate calls (expected 1, resident)"
    # The significance gate can legitimately zero a weak-signal candidate's MI depending on the (per-call,
    # unseeded) permutation draws -- not this test's concern (that's the confidence/gate logic, unrelated to
    # the resident-upload fix). Only pin that both calls completed and returned sane, finite, non-negative MI.
    assert np.isfinite(mi0) and mi0 >= 0.0
    assert np.isfinite(mi1) and mi1 >= 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("fn", [mi_direct_gpu_batched, mi_direct_gpu_batched_streamed])
def test_candidate_x_still_uploaded_fresh_every_call(monkeypatch, fn):
    """classes_x/freqs_x genuinely vary per candidate -- must NOT be resident-cached (would either never
    hit, since content differs every call, or grow the cache unboundedly across a wide candidate pool)."""
    factors, factors_nbins = _build_two_candidate_inputs(seed=5)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors,
        vars_indices=(2,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    classes_x0, _freqs_x0, _ = merge_vars(
        factors_data=factors,
        vars_indices=(0,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    x_calls = _count_asarray_calls_matching(monkeypatch, np.ascontiguousarray(classes_x0.astype(np.int32)))

    fn(factors, (0,), (2,), factors_nbins, npermutations=16, batch_size=8, classes_y=classes_y.copy(), freqs_y=freqs_y.copy())
    fn(factors, (0,), (2,), factors_nbins, npermutations=16, batch_size=8, classes_y=classes_y.copy(), freqs_y=freqs_y.copy())

    assert x_calls["n"] == 2, f"classes_x-content cp.asarray called {x_calls['n']}x across 2 identical-candidate calls (expected 2, NOT cached)"


@pytest.mark.gpu
@pytest.mark.parametrize("fn", [mi_direct_gpu_batched, mi_direct_gpu_batched_streamed])
def test_resident_upload_bit_identical_to_disabled_cache(monkeypatch, fn):
    """Cache ON vs the diagnostic OFF switch (the pre-fix raw-upload behaviour) must return the SAME ``mi``
    -- the selection-relevant deliverable -- since the cached device buffer holds the exact same values as
    a fresh upload would.

    ``confidence`` is NOT compared: empirically confirmed (direct 5-run repro with the resident cache fully
    DISABLED on every run, i.e. with NO involvement of this fix at all) that ``mi_direct_gpu_batched_streamed``
    returns a bit-identical ``mi`` every time but a run-to-run VARYING ``confidence`` (0.75 vs 1.0 seen across
    otherwise-identical calls) -- a pre-existing property of its two-concurrent-CUDA-stream design (GPU
    atomic-add histogram accumulation is not bit-order-reproducible across concurrently-scheduled streams;
    a documented CUDA characteristic, not a resident-caching regression). ``mi`` itself is unaffected because
    ``original_mi`` only gets gated to 0.0 past a LARGE ``nfailed`` threshold (``min_nonzero_confidence=0.5``
    here keeps ``max_failed`` comfortably above the observed nfailed variance); asserting equality on ``mi``
    is therefore the correct, non-flaky proof that resident caching changed nothing about the computed
    result. See the project convention: GPU MI comparisons are selection-equivalent, not bit-identical on
    every stochastic byproduct."""
    rng = np.random.default_rng(9)
    n = 2000
    x0 = rng.integers(0, 5, size=n).astype(np.int32)
    x1 = rng.integers(0, 5, size=n).astype(np.int32)
    y_raw = x0  # deterministic relationship: MI(x0; y) is far above any permutation's chance MI
    factors = np.column_stack([x0, x1, y_raw]).astype(np.int32)
    factors_nbins = np.array([5, 5, 5], dtype=np.int64)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=factors,
        vars_indices=(2,),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )

    _orig_default_rng = cp.random.default_rng

    def _fixed_rng(*a, **kw):
        """Fixed rng."""
        return _orig_default_rng(1234)

    monkeypatch.setattr(cp.random, "default_rng", _fixed_rng)

    clear_fe_resident_operands()
    mi_cached, _conf_cached = fn(
        factors,
        (0,),
        (2,),
        factors_nbins,
        npermutations=32,
        batch_size=8,
        min_nonzero_confidence=0.5,
        classes_y=classes_y.copy(),
        freqs_y=freqs_y.copy(),
    )

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    clear_fe_resident_operands()
    mi_fresh, _conf_fresh = fn(
        factors,
        (0,),
        (2,),
        factors_nbins,
        npermutations=32,
        batch_size=8,
        min_nonzero_confidence=0.5,
        classes_y=classes_y.copy(),
        freqs_y=freqs_y.copy(),
    )

    assert mi_cached == mi_fresh


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
