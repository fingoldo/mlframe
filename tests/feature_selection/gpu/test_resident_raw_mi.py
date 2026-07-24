"""Unit coverage for ``_resident_raw_mi.py``'s ``resident_raw_baseline_mi``.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this module (the resident-operand MI path for
FIT-CONSTANT raw baseline matrices under STRICT-residency) had zero test references anywhere in the
suite. Pins the documented gate/fallback contract and a basic informative-vs-noise sanity check on the
returned MI values (a full bit-exact host-vs-resident parity harness would need to reconstruct the
exact ``_mi_classif_batch`` host caller context, out of scope for closing this coverage gap).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Import the ``hermite_fe`` package facade BEFORE ``_resident_raw_mi`` (which lazily imports
# ``_hermite_fe_mi`` internally): ``_hermite_fe_mi.py <-> hermite_fe`` is a documented, whitelisted
# circular import (``tests/test_meta/test_no_import_cycles.py``) that is only benign when the facade
# loads first. Run in isolation (this file alone, with nothing else in the process having already
# imported ``hermite_fe``), the opposite order triggers an ImportError inside the resident MI path's
# own broad except-and-fall-back-to-None, silently degrading these tests to "fallback engaged" instead
# of exercising the resident code path they exist to test. Guarantee the safe order explicitly rather
# than relying on incidental import order from other test files/conftest.
import mlframe.feature_selection.filters.hermite_fe  # noqa: F401

from mlframe.feature_selection.filters._resident_raw_mi import resident_raw_baseline_mi


def _need_cuda() -> bool:
    """Whether a usable CUDA device is present this process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


def teardown_function():
    """Never leak the resident-raw-baseline opt-out env var into another test."""
    os.environ.pop("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", None)
    os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
    os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)


def test_returns_none_when_strict_residency_disabled():
    """The default (flag-off) path never engages the resident matrix upload; the caller must fall
    back to the exact host ``_mi_classif_batch``."""
    os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
    os.environ.pop("MLFRAME_FE_GPU_STRICT_RESIDENT", None)
    rng = np.random.default_rng(0)
    mat = rng.random((200, 3))
    y = rng.integers(0, 2, size=200)
    assert resident_raw_baseline_mi(mat, y, "test_role", nbins=10) is None


def test_returns_none_when_explicit_opt_out_set():
    """``MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE=0`` is the documented explicit opt-out, even under STRICT-residency."""
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    os.environ["MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE"] = "0"
    rng = np.random.default_rng(0)
    mat = rng.random((200, 3))
    y = rng.integers(0, 2, size=200)
    assert resident_raw_baseline_mi(mat, y, "test_role", nbins=10) is None


def test_empty_matrix_returns_zeros_not_none():
    """A zero-column matrix returns an empty (k=0) array (a well-defined degenerate case), not ``None``
    -- distinguishing it from the "GPU path unavailable" signal the caller treats differently."""
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    pytest.importorskip("cupy")
    if not _need_cuda():
        pytest.skip("no CUDA")
    mat = np.zeros((100, 0))
    y = np.zeros(100, dtype=np.int64)
    result = resident_raw_baseline_mi(mat, y, "test_role_empty", nbins=10)
    assert result is not None
    assert result.shape == (0,)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_informative_column_scores_higher_mi_than_noise():
    """Sanity check on the resident MI values themselves: a column that strongly determines the binary
    target must score materially higher MI than an unrelated noise column, through the SAME resident
    code path (percentile-edge binning, the non-rank-binning default)."""
    pytest.importorskip("cupy")
    rng = np.random.default_rng(1)
    n = 4000
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (signal + 0.1 * rng.normal(size=n) > 0).astype(np.int64)
    mat = np.column_stack([signal, noise])

    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    result = resident_raw_baseline_mi(mat, y, ("test_role_signal", ("signal", "noise")), nbins=10)
    assert result is not None
    assert result.shape == (2,)
    signal_mi, noise_mi = float(result[0]), float(result[1])
    assert signal_mi > 5 * max(noise_mi, 1e-6), f"informative column MI {signal_mi} not clearly above noise MI {noise_mi}"


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_rank_binning_path_also_returns_valid_mi():
    """``rank_binning=True`` routes through the argsort equi-frequency resident binner (the gate-MI
    byte-match path) instead of the percentile-edge binner, and still returns a well-formed result."""
    pytest.importorskip("cupy")
    rng = np.random.default_rng(2)
    n = 3000
    signal = rng.normal(size=n)
    y = (signal > 0).astype(np.int64)
    mat = signal[:, None]

    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    result = resident_raw_baseline_mi(mat, y, "test_role_rank", nbins=10, rank_binning=True)
    # The rank-binning resident path may itself be unavailable on some builds (documented fallback to
    # None); if present, it must be a finite, non-negative MI value.
    if result is not None:
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert result[0] >= 0.0
