"""Regression tests for the VRAM safety gate + row-chunked GPU fallback in ``dispatch_batch_pair_mi``.

Root cause (2026-07-10 wellbore production run, 2_425_537 rows x 423 columns): ``batch_pair_mi_cuda``/
``batch_pair_mi_cupy`` upload the ENTIRE ``factors_data`` matrix to the device on every call, with no
check against free VRAM. On this host's 4GB card the upload alone (~4GB as int32) consumes essentially
the whole budget. Windows/WDDM can transparently over-subscribe device memory via host-paging instead of
raising a catchable CUDA OOM, so the upload "succeeds" and the kernel launch then grinds through
PCIe-paged memory for minutes before the OS kills the process -- no Python exception, no traceback, no
Windows Event Log entry, just a silent ``EXIT_CODE=1`` (reproduced live: the process ran silently for
~14 minutes, matching the isolated CUDA batch-precompute duration at this exact scale, then vanished).

This module was the one remaining GPU dispatch site in the package without the ``_fe_gpu_vram`` ABSOLUTE
VRAM-cushion guard that ``_cmi_cuda.py`` / ``gpu.py`` / ``hermite_fe`` / ``friend_graph_gpu.py`` /
``batch_mi_noise_gate_gpu.py`` / ``_permutation_null_pair_resident.py`` already use (2026-07-05 sweep).
``_gpu_upload_fits`` closes that gap using the SAME two-layer pattern as ``_cmi_cuda._should_use_cuda``:
a relative free-VRAM cap, then the shared absolute cushion floor from ``_fe_gpu_vram.fe_gpu_has_vram_cushion``.

A rejection is never silent: every ``_gpu_upload_fits`` REJECT logs the full sizing context (n_samples,
n_cols, n_pairs, requested GB, free/total VRAM GB) at WARNING, per explicit user feedback that a quiet
fallback is unacceptable for a decision this consequential.

Also, a rejection no longer means "give up on the GPU": ``dispatch_batch_pair_mi`` first tries
:func:`batch_pair_mi_cuda_row_chunked`, which uploads ``factors_data`` in VRAM-sized row-blocks and
accumulates the joint histogram across them (counts are additive over rows; bit-identical to the
full-upload kernel to within a single-ULP libm rounding difference -- verified against real hardware:
max abs diff ~5e-18, far below float64 machine epsilon-scale significance). Only when even THAT fails
(no CUDA, or a genuine runtime/driver fault) does the CPU njit kernel run.

A second, independent bug is also pinned: the CUDA except clause used to only catch ``(ValueError,
RuntimeError)``, but numba's ``CudaAPIError``/``CudaDriverError`` derive directly from ``Exception`` --
a genuine CUDA driver fault would have skipped that handler and propagated to the caller uncaught.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest

import mlframe.feature_selection.filters.batch_pair_mi_gpu as bpmg
from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
    _required_gpu_bytes,
    dispatch_batch_pair_mi,
)


def _build_pair_inputs(n_samples=500, nbins_per_col=(4, 4, 4, 4), n_classes_y=2, seed=0):
    """Build pair inputs."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, nb, size=n_samples) for nb in nbins_per_col]
    data = np.column_stack(cols).astype(np.int32)
    nbins = np.asarray(nbins_per_col, dtype=np.int32)
    y_raw = rng.integers(0, n_classes_y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(y_raw, minlength=n_classes_y).astype(np.float64) / n_samples
    pairs = list(itertools.combinations(range(len(nbins_per_col)), 2))
    pair_a = np.array([p[0] for p in pairs], dtype=np.int64)
    pair_b = np.array([p[1] for p in pairs], dtype=np.int64)
    return data, nbins, y_raw, freqs_y, pair_a, pair_b


def test_required_gpu_bytes_dominated_by_factors_data_upload():
    """The estimate must scale with the FULL factors_data upload (n_samples * n_cols * 4 bytes as
    int32), not just the (much smaller) pair-index/output arrays -- that upload is what actually
    exhausts VRAM in production. At production-representative row counts (n_samples >> n_cols, mirroring
    the 2.4M-row wellbore run) the pair/aux terms are negligible; a tiny n_samples with many columns
    would let C(n_cols, 2) pair overhead dominate instead, which is NOT the shape that crashed."""
    data, nbins, classes_y, freqs_y, pair_a, _pair_b = _build_pair_inputs(n_samples=50_000, nbins_per_col=(4,) * 50)
    required = _required_gpu_bytes(data, pair_a, nbins, classes_y, freqs_y)
    factors_bytes = data.size * 4
    assert required >= factors_bytes, "factors_data upload must dominate the byte estimate"
    assert required < factors_bytes * 1.05, "pair/aux arrays should not meaningfully inflate the estimate at production-representative scale"


def test_gpu_upload_fits_rejection_is_never_silent(monkeypatch, caplog):
    """A REJECT must log full sizing context (rows/cols/pairs/requested-GB/free-GB/total-GB) at WARNING --
    not a bare 'insufficient VRAM' with no numbers, and not silent (DEBUG-only)."""
    _data, _nbins, _classes_y, _freqs_y, _pair_a, _pair_b = _build_pair_inputs(n_samples=1000, nbins_per_col=(4,) * 20)

    class _FakeCupyRuntime:
        """Groups tests covering FakeCupyRuntime."""
        @staticmethod
        def memGetInfo():
            """Helper that memGetInfo."""
            return 100 * 1024**2, 4 * 1024**3  # 100 MiB free / 4 GiB total -- tiny, forces a reject

    class _FakeCupy:
        """Groups tests covering FakeCupy."""
        cuda = type("cuda", (), {"runtime": _FakeCupyRuntime})

    monkeypatch.setitem(__import__("sys").modules, "cupy", _FakeCupy)

    with caplog.at_level(logging.WARNING):
        ok = bpmg._gpu_upload_fits(2 * 1024**3, n_samples=1000, n_cols=20, n_pairs=190, context="test_ctx")

    assert ok is False
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "a rejection must produce at least one WARNING-level log record"
    msg = warnings[0]
    assert "test_ctx" in msg
    assert "1000" in msg  # n_samples
    assert "20" in msg  # n_cols
    assert "190" in msg  # n_pairs
    assert "GB" in msg
    assert "free" in msg.lower()


def test_forced_cuda_vram_insufficient_uses_row_chunked_not_bare_fallback(monkeypatch):
    """When the full upload doesn't fit but CUDA is available, the row-chunked GPU path must be tried
    BEFORE giving up on the GPU entirely -- the old behavior (straight to CPU njit) left real GPU speed
    on the table whenever a row-chunked pass could have handled it."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: False)

    def _boom_full(*a, **kw):
        """Boom full."""
        raise AssertionError("batch_pair_mi_cuda (full-upload) must NOT be invoked when the VRAM guard fails")

    calls = {"row_chunked": 0}

    def _fake_row_chunked(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y):
        """Fake row chunked."""
        calls["row_chunked"] += 1
        return np.zeros(pair_a.shape[0], dtype=np.float64)

    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda", _boom_full)
    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda_row_chunked", _fake_row_chunked)

    mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cuda")

    assert backend == "cuda_row_chunked"
    assert calls["row_chunked"] == 1
    assert mi.shape[0] == pair_a.shape[0]


def test_falls_back_to_cpu_only_when_row_chunked_also_fails(monkeypatch, caplog):
    """Only when row-chunked GPU ALSO fails (or CUDA is unavailable) does the CPU njit kernel run."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: False)

    def _boom_full(*a, **kw):
        """Boom full."""
        raise AssertionError("batch_pair_mi_cuda (full-upload) must NOT be invoked when the VRAM guard fails")

    def _boom_row_chunked(*a, **kw):
        """Boom row chunked."""
        raise RuntimeError("simulated row-chunked failure too")

    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda", _boom_full)
    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda_row_chunked", _boom_row_chunked)

    with caplog.at_level(logging.WARNING):
        mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cuda")

    assert backend == "njit"
    assert mi.shape[0] == pair_a.shape[0]
    assert any("row-chunked CUDA also failed" in r.message for r in caplog.records)


def test_no_cuda_at_all_skips_row_chunked_attempt_cleanly(monkeypatch):
    """When CUDA is entirely unavailable, the row-chunked attempt must be skipped (not raise trying to
    call into numba.cuda) and the CPU njit path must run directly."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", False)
    monkeypatch.setattr(bpmg, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: False)

    mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    assert backend == "njit"
    assert mi.shape[0] == pair_a.shape[0]


def test_forced_cupy_falls_back_to_cpu_when_vram_insufficient(monkeypatch, caplog):
    """cupy has no row-chunked variant (yet) -- a VRAM-insufficient forced-cupy request falls straight to
    CPU njit, still with a WARNING logged."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUPY_AVAIL", True)
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", False)

    def _boom(*a, **kw):
        """Helper that boom."""
        raise AssertionError("batch_pair_mi_cupy must NOT be invoked when the VRAM guard fails")

    monkeypatch.setattr(bpmg, "batch_pair_mi_cupy", _boom)
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: False)

    with caplog.at_level(logging.WARNING):
        mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y, force_backend="cupy")

    assert backend == "njit"
    assert mi.shape[0] == pair_a.shape[0]


def test_auto_choice_cuda_falls_back_via_row_chunked_when_vram_insufficient(monkeypatch):
    """Auto choice cuda falls back via row chunked when vram insufficient."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpmg, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bpmg, "_batch_pair_mi_backend_choice", lambda n_samples, n_pairs: "cuda")
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: False)

    def _boom_full(*a, **kw):
        """Boom full."""
        raise AssertionError("batch_pair_mi_cuda (full-upload) must NOT be invoked when the VRAM guard fails")

    def _fake_row_chunked(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y):
        """Fake row chunked."""
        return np.zeros(pair_a.shape[0], dtype=np.float64)

    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda", _boom_full)
    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda_row_chunked", _fake_row_chunked)

    mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    assert backend == "cuda_row_chunked"
    assert mi.shape[0] == pair_a.shape[0]


def test_auto_choice_cuda_uses_gpu_when_vram_fits(monkeypatch):
    """Sanity check the guard is not a blanket disable: when VRAM fits, the cuda backend still fires."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpmg, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bpmg, "_batch_pair_mi_backend_choice", lambda n_samples, n_pairs: "cuda")
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: True)

    calls = {"n": 0}

    def _fake_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y):
        """Fake cuda."""
        calls["n"] += 1
        return np.zeros(pair_a.shape[0], dtype=np.float64)

    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda", _fake_cuda)

    _mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    assert backend == "cuda"
    assert calls["n"] == 1


class _SimulatedCudaDriverFault(Exception):
    """Mirrors numba's ``CudaAPIError``/``CudaDriverError`` MRO: derives directly from ``Exception``,
    NOT ``RuntimeError`` or ``ValueError`` -- the exact shape the pre-fix narrow except clause missed."""


def test_cuda_driver_fault_falls_through_row_chunked_then_cpu(monkeypatch):
    """A CUDA driver fault on the full-upload kernel (neither ValueError nor RuntimeError, as numba's real
    exceptions are) must not propagate -- it tries row-chunked next, then CPU. Fails on the pre-fix
    ``except (ValueError, RuntimeError):`` clause (would propagate the fault uncaught)."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs()
    monkeypatch.setattr(bpmg, "_CUDA_AVAIL", True)
    monkeypatch.setattr(bpmg, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bpmg, "_batch_pair_mi_backend_choice", lambda n_samples, n_pairs: "cuda")
    monkeypatch.setattr(bpmg, "_gpu_upload_fits", lambda required_bytes, **kw: True)

    def _fault(*a, **kw):
        """Helper that fault."""
        raise _SimulatedCudaDriverFault("CUDA_ERROR_LAUNCH_FAILED")

    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda", _fault)
    monkeypatch.setattr(bpmg, "batch_pair_mi_cuda_row_chunked", _fault)

    mi, backend = dispatch_batch_pair_mi(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    assert backend == "njit"
    assert mi.shape[0] == pair_a.shape[0]


@pytest.mark.skipif(not (bpmg._CUDA_AVAIL or bpmg._CUPY_AVAIL), reason="no GPU backend available on this host")
def test_real_gpu_upload_fits_rejects_absurd_byte_request():
    """End-to-end sanity check against the REAL device probe (no mocking): an absurd byte request must
    be rejected by ``_gpu_upload_fits`` on any real GPU, proving the guard actually queries hardware."""
    assert bpmg._gpu_upload_fits(999 * 1024**4) is False  # 999 TB -- no real GPU has this much VRAM


@pytest.mark.skipif(not (bpmg._CUDA_AVAIL or bpmg._CUPY_AVAIL), reason="no GPU backend available on this host")
def test_real_gpu_upload_fits_accepts_tiny_request():
    """Sanity check the real probe is not a blanket rejection: a trivially small request must pass."""
    assert bpmg._gpu_upload_fits(1024) is True


@pytest.mark.skipif(not bpmg._CUDA_AVAIL, reason="numba.cuda not available on this host")
def test_row_chunked_cuda_matches_full_upload_cuda_on_real_hardware():
    """Real-hardware bit-identity (within single-ULP libm rounding) check: the row-chunked kernel must
    reproduce the full-upload kernel's MI values even when forced into many small row-chunks."""
    data, nbins, classes_y, freqs_y, pair_a, pair_b = _build_pair_inputs(
        n_samples=3000,
        nbins_per_col=(5,) * 8,
        n_classes_y=3,
        seed=11,
    )
    mi_full = bpmg.batch_pair_mi_cuda(data, pair_a, pair_b, nbins, classes_y, freqs_y)

    orig_choose = bpmg._choose_row_chunk_rows
    bpmg._choose_row_chunk_rows = lambda *a, **kw: 97  # tiny -> forces real multi-chunk accumulation
    try:
        mi_chunked = bpmg.batch_pair_mi_cuda_row_chunked(data, pair_a, pair_b, nbins, classes_y, freqs_y)
    finally:
        bpmg._choose_row_chunk_rows = orig_choose

    np.testing.assert_allclose(
        mi_full,
        mi_chunked,
        rtol=0,
        atol=1e-9,
        err_msg="row-chunked CUDA must reproduce the full-upload CUDA kernel's MI to within single-ULP rounding",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
