"""Wave 11 (Category 3) H1: ``_bspline_basis_values`` was rewritten from a raw per-point Python loop
(Cox-de Boor recursion re-derived per point, with an inner degree/knot-span scan) to a fully vectorised
recursion mirroring the GPU-resident twin ``_extra_basis_resident.py::_bspline_col_gpu``. Pins the new
vectorised implementation against a frozen copy of the pre-fix per-point loop across random knot/degree/
column configurations, including edge cases (constant column, all-boundary values).
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import (
    _bspline_basis_values,
    _fit_spline_knots,
)


def _bspline_basis_values_ref(z: np.ndarray, knots: np.ndarray, idx: int, degree: int = 3) -> np.ndarray:
    """Frozen copy of the pre-Wave-11 per-point Cox-de Boor loop (the reference this fix must match)."""
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nk = len(knots)
    for i in range(n):
        zi = z[i]
        if zi >= knots[nk - degree - 1]:
            zi_eff = knots[nk - degree - 1] - 1e-12
        elif zi <= knots[degree]:
            zi_eff = knots[degree] + 1e-12
        else:
            zi_eff = zi
        k = degree
        for kk in range(degree, nk - degree - 1):
            if knots[kk] <= zi_eff < knots[kk + 1]:
                k = kk
                break
        else:
            k = nk - degree - 2
        N = np.zeros(degree + 1, dtype=np.float64)
        N[0] = 1.0
        for d in range(1, degree + 1):
            saved = 0.0
            for r in range(d):
                t_left = knots[k + 1 + r - d]
                t_right = knots[k + 1 + r]
                denom = t_right - t_left
                temp = 0.0 if denom <= 1e-12 else N[r] / denom
                N[r] = saved + (t_right - zi_eff) * temp
                saved = (zi_eff - t_left) * temp
            N[d] = saved
        rel = idx - (k - degree)
        if 0 <= rel <= degree:
            out[i] = N[rel]
    return out


def test_bspline_vectorized_matches_reference_across_random_configs():
    rng = np.random.default_rng(0)
    max_abs_diff = 0.0
    n_cases = 0
    for trial in range(15):
        n = int(rng.integers(200, 1500))
        n_inner = int(rng.integers(2, 10))
        degree = 3
        x = rng.normal(size=n) * rng.choice([1.0, 10.0, 0.01])
        knots, lo, hi = _fit_spline_knots(x, n_inner, degree=degree)
        span = max(hi - lo, 1e-12)
        z = np.clip((x - lo) / span, 0.0, 1.0)
        n_basis = len(knots) - degree - 1
        for idx in range(n_basis):
            ref = _bspline_basis_values_ref(z, knots, idx, degree=degree)
            new = _bspline_basis_values(z, knots, idx, degree=degree)
            diff = float(np.max(np.abs(ref - new)))
            max_abs_diff = max(max_abs_diff, diff)
            n_cases += 1
            assert diff < 1e-9, f"trial={trial} idx={idx} diff={diff}"
    assert n_cases > 50
    assert max_abs_diff < 1e-9


def test_bspline_vectorized_matches_reference_on_constant_column():
    x_const = np.full(500, 5.0)
    knots, lo, hi = _fit_spline_knots(x_const, 4, degree=3)
    span = max(hi - lo, 1e-12)
    z = np.clip((x_const - lo) / span, 0.0, 1.0)
    n_basis = len(knots) - 3 - 1
    for idx in range(n_basis):
        ref = _bspline_basis_values_ref(z, knots, idx, degree=3)
        new = _bspline_basis_values(z, knots, idx, degree=3)
        assert np.max(np.abs(ref - new)) < 1e-9


def test_bspline_vectorized_matches_reference_on_boundary_values():
    rng = np.random.default_rng(1)
    knots, _lo, _hi = _fit_spline_knots(rng.normal(size=1000), 5, degree=3)
    z_boundary = np.array([0.0, 1.0, 0.0, 1.0] * 100)
    n_basis = len(knots) - 3 - 1
    for idx in range(n_basis):
        ref = _bspline_basis_values_ref(z_boundary, knots, idx, degree=3)
        new = _bspline_basis_values(z_boundary, knots, idx, degree=3)
        assert np.max(np.abs(ref - new)) < 1e-9
