"""A `force_backend` request that cannot be honoured must say so.

`dispatch_batch_pair_mi(force_backend="cupy")` was gated as one condition -- `force_backend == "cupy" and
_CUPY_AVAIL and _vram_ok` -- so a request that could not be honoured matched no branch and fell straight
through to the njit return with no log line at all. A caller benchmarking or pinning a backend on a large
frame got the CPU kernel, and only the returned `backend_name` said so, which many call sites discard.

The adjacent forced-CUDA branch already warns on every downgrade, and the forced-cupy branch's own `except`
logs too -- the silent path was the one where the branch was never entered.

Two more requests disappeared the same way and are covered here: `force_backend="cuda"` on a host without
numba.cuda, and a value that is neither.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_mi_gpu as bp


@pytest.fixture
def pair_inputs():
    """A tiny valid input set: two features, one pair, binary target."""
    rng = np.random.default_rng(0)
    n = 200
    factors = rng.integers(0, 4, size=(n, 2)).astype(np.int32)
    pair_a = np.array([0], dtype=np.int64)
    pair_b = np.array([1], dtype=np.int64)
    nbins = np.array([4, 4], dtype=np.int64)
    classes_y = rng.integers(0, 2, size=n).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64) / n
    return factors, pair_a, pair_b, nbins, classes_y, freqs_y


@pytest.fixture(autouse=True)
def _gpu_not_globally_disabled(monkeypatch):
    """Reach the override block at all.

    `gpu_globally_disabled()` is checked first and, by design, wins even over an explicit force_backend --
    that is the whole point of MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES="". The suite runs with the opt-out
    set, so without this every test below would exercise that early return instead of the branch it names.
    The contract itself is pinned separately at the bottom of this file.
    """
    import mlframe.feature_selection.filters._gpu_policy as policy

    monkeypatch.setattr(policy, "gpu_globally_disabled", lambda: False)


def _dispatch(pair_inputs, force_backend):
    """Run the dispatcher on the fixture."""
    return bp.dispatch_batch_pair_mi(*pair_inputs, force_backend=force_backend)


def test_a_forced_cupy_request_that_does_not_fit_vram_is_reported(pair_inputs, monkeypatch, caplog):
    """The exact silent path: cupy present, VRAM estimate says no."""
    monkeypatch.setattr(bp, "_CUPY_AVAIL", True)
    monkeypatch.setattr(bp, "_gpu_upload_fits", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _, backend = _dispatch(pair_inputs, "cupy")
    assert backend == "njit"
    assert any(
        "forced cupy backend requested but not honoured" in r.getMessage() for r in caplog.records
    ), f"the downgrade was silent; warnings seen: {[r.getMessage()[:80] for r in caplog.records]}"
    assert any("does not fit VRAM" in r.getMessage() for r in caplog.records), "the message must say WHY it was not honoured"


def test_a_forced_cupy_request_without_cupy_is_reported(pair_inputs, monkeypatch, caplog):
    """The other half of the same condition."""
    monkeypatch.setattr(bp, "_CUPY_AVAIL", False)
    monkeypatch.setattr(bp, "_gpu_upload_fits", lambda *a, **k: True)
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _, backend = _dispatch(pair_inputs, "cupy")
    assert backend == "njit"
    assert any("cupy is unavailable" in r.getMessage() for r in caplog.records)


def test_a_forced_cuda_request_without_numba_cuda_is_reported(pair_inputs, monkeypatch, caplog):
    """The forced-CUDA branch is entered only when _CUDA_AVAIL; without it the request vanished as quietly."""
    monkeypatch.setattr(bp, "_CUDA_AVAIL", False)
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _, backend = _dispatch(pair_inputs, "cuda")
    assert backend == "njit"
    assert any("numba.cuda is unavailable" in r.getMessage() for r in caplog.records)


def test_an_unrecognised_backend_name_is_reported(pair_inputs, caplog):
    """A typo silently produced the CPU kernel and a 'njit' name nobody reads."""
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _, backend = _dispatch(pair_inputs, "cuppy")
    assert backend == "njit"
    assert any("is not one of" in r.getMessage() for r in caplog.records)


def test_the_result_is_still_correct_when_the_request_is_downgraded(pair_inputs, monkeypatch):
    """Reporting the downgrade must not change what the njit fallback computes."""
    monkeypatch.setattr(bp, "_CUPY_AVAIL", True)
    monkeypatch.setattr(bp, "_gpu_upload_fits", lambda *a, **k: False)
    forced, _ = _dispatch(pair_inputs, "cupy")
    direct = bp.batch_pair_mi_njit_prange(*pair_inputs)
    assert np.allclose(forced, direct)


def test_no_warning_when_nothing_was_forced(pair_inputs, caplog):
    """The default path must stay quiet; a gate that cries on every call gets muted."""
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _dispatch(pair_inputs, None)
    assert not [r for r in caplog.records if "force_backend" in r.getMessage() or "forced" in r.getMessage()]


def test_the_global_opt_out_still_outranks_a_forced_backend(pair_inputs, monkeypatch, caplog):
    """The early return must keep winning: an explicit force_backend does not override the opt-out.

    Pinned here because every other test in this file disables that check to reach the branch it needs, and
    a fixture that switches something off should not be the only statement about it.
    """
    import mlframe.feature_selection.filters._gpu_policy as policy

    monkeypatch.setattr(policy, "gpu_globally_disabled", lambda: True)
    monkeypatch.setattr(bp, "_CUPY_AVAIL", True)
    monkeypatch.setattr(bp, "_gpu_upload_fits", lambda *a, **k: True)
    with caplog.at_level(logging.WARNING, logger=bp.logger.name):
        _, backend = _dispatch(pair_inputs, "cupy")
    assert backend == "njit", "the global GPU opt-out was overridden by force_backend"
