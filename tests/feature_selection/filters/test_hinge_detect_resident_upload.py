"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``detect_hinge_breakpoints_gpu`` previously re-uploaded the
FIT-CONSTANT target ``y`` via a raw ``cp.asarray`` on every candidate column call (the candidate column
``x`` itself is genuinely fresh per call and stays raw). ``y`` now routes through ``resident_operand`` so
repeated calls on the SAME target share ONE device buffer.

Skips when cupy is unavailable (CI without a GPU)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._hinge_detect_gpu_resident import detect_hinge_breakpoints_gpu
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands

_KW = dict(
    max_breakpoints=1,
    min_heldout_r2_uplift=0.01,
    precheck_qs=(0.25, 0.5, 0.75),
    precheck_min_sse_drop=0.0,
    cand_q_lo=0.1,
    cand_q_hi=0.9,
    n_candidates=16,
    min_rows=100,
    min_seg_rows=20,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _hinge_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.where(x > 0.5, 2.0 * (x - 0.5), 0.0) + rng.normal(0, 0.05, n)
    return x, y


def test_y_dedups_upload_across_candidate_columns(monkeypatch):
    """Two detect_hinge_breakpoints_gpu calls scoring DIFFERENT candidate columns (x1, x2) against the SAME
    target y -- the realistic per-column FE-scan loop -- must upload y ONLY ONCE."""
    n = 5000
    x1, y = _hinge_data(n, seed=1)
    x2, _y_unused = _hinge_data(n, seed=2)

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if isinstance(arr, np.ndarray) and arr.shape == y.shape and arr.dtype == np.float64 and np.array_equal(arr, y):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    taus1 = detect_hinge_breakpoints_gpu(x1, y, **_KW)
    taus2 = detect_hinge_breakpoints_gpu(x2, y, **_KW)

    assert taus1 is not None and taus2 is not None, "cupy fault -> host fallback (GPU path not exercised)"
    assert upload_calls["n"] == 1, f"y upload called {upload_calls['n']} times across 2 candidate columns (expected 1)"


def test_resident_y_bit_identical_to_fresh_upload():
    """A cache HIT (warm resident y) must produce the exact same taus as a cache MISS (fresh upload) -- the
    caching change only skips the redundant H2D, never alters the uploaded bytes."""
    n = 5000
    x, y = _hinge_data(n, seed=3)

    clear_fe_resident_operands()
    taus_cold = detect_hinge_breakpoints_gpu(x, y, **_KW)  # cold: uploads y fresh
    taus_warm = detect_hinge_breakpoints_gpu(x, y, **_KW)  # warm: same content -> cache HIT

    assert taus_cold is not None and taus_warm is not None
    assert taus_cold == taus_warm, f"cold={taus_cold} warm={taus_warm}"


def test_bit_identical_vs_resident_cache_disabled(monkeypatch):
    """MLFRAME_FE_RESIDENT_OPERANDS=0 reproduces the EXACT pre-fix raw-upload behavior (resident_operand
    always does a fresh cp.asarray, never touching the cache). The fix's output must be bit-identical to
    that baseline."""
    n = 6000
    x, y = _hinge_data(n, seed=4)

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    taus_raw = detect_hinge_breakpoints_gpu(x, y, **_KW)
    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "1")
    taus_cached = detect_hinge_breakpoints_gpu(x, y, **_KW)

    assert taus_raw is not None and taus_cached is not None, "cupy fault -> host fallback (GPU path not exercised)"
    assert taus_raw == taus_cached, f"raw={taus_raw} cached={taus_cached}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
