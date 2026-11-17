"""A singular system must surface as ``LinAlgError`` so the callers' existing lstsq fallbacks run."""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.feature_selection.filters._cupy_solve_checked import cupy_solve_checked


def _stub_cp(solve):
    """A numpy-backed stand-in for the cupy module with the given ``linalg.solve``."""
    return types.SimpleNamespace(linalg=types.SimpleNamespace(solve=solve), isfinite=np.isfinite)


def test_non_finite_solution_raises_linalg_error() -> None:
    """cupy returns NaN for a singular matrix instead of raising; the helper turns that into the exception the callers catch."""
    cp = _stub_cp(lambda a, b: np.full_like(b, np.nan))
    with pytest.raises(np.linalg.LinAlgError):
        cupy_solve_checked(cp, np.zeros((2, 2)), np.ones(2))


def test_finite_solution_is_returned_unchanged() -> None:
    """A regular system returns exactly what the solver returned."""
    a = np.array([[2.0, 0.0], [0.0, 4.0]])
    b = np.array([2.0, 4.0])
    cp = _stub_cp(np.linalg.solve)
    np.testing.assert_array_equal(cupy_solve_checked(cp, a, b), np.array([1.0, 1.0]))


def test_real_cupy_singular_matrix_raises() -> None:
    """On a real device a singular matrix gives NaN from cupy itself and the helper raises."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("no usable CUDA runtime")
    with pytest.raises(np.linalg.LinAlgError):
        cupy_solve_checked(cp, cp.zeros((3, 3)), cp.ones(3))


def test_callers_import_the_checked_helper() -> None:
    """The three resident solvers import and use the helper (a name error here would only show at the first device call)."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_batch_gpu, _fourier_detect_gpu_resident
    from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp_gpu_resident

    for mod in (_fourier_batch_gpu, _fourier_detect_gpu_resident, _hermite_prewarp_gpu_resident):
        assert mod.cupy_solve_checked is cupy_solve_checked
