"""Parity: the GPU batched basis router (_gpu_route_bases_batched) must pick the SAME
orth-polynomial basis per column as the host basis_route_by_signal.

Routing is SELECTION-BEARING (the chosen basis is baked into the EngineeredRecipe), and the GPU basis
eval is parity-<1e-6 (not bit-identical), so a near-tie between two bases can legitimately flip the
argmax. This test asserts identical per-column choices on columns where the host decision has a clear
margin (top-2 |corr| gap >= GAP); a flip on a genuine <GAP tie is reported, not failed, since that is the
exact case the opt-in default (MLFRAME_FE_GPU_ROUTING off) guards. A clean run here is the selection-
equivalence evidence required before flipping the default to opt-out.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gpu_resident_fe import _cuda_present

cupy = pytest.importorskip("cupy") if _cuda_present() else None
if not _cuda_present():
    pytest.skip("cupy not available", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_route_bases_batched
from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
    _POLY_BASES,
    basis_route_by_signal,
    _evaluate_basis_column,
)

_GAP = 1e-3  # host top-2 |corr| gap below which a basis flip is a genuine tie (reported, not failed)
_DEGREES = (2, 3)


def _host_basis_corrs(x, y):
    """Per-basis host bcorr (max over degrees), so the test can tell a clear win from a near-tie."""
    yv = np.asarray(y, dtype=np.float64).ravel()
    out = {}
    for basis in _POLY_BASES:
        bcorr = 0.0
        for d in _DEGREES:
            try:
                v = np.asarray(_evaluate_basis_column(np.asarray(x, dtype=np.float64), basis, int(d)), dtype=np.float64)
            except Exception:
                continue
            if v.size != yv.size or not np.all(np.isfinite(v)) or float(np.std(v)) < 1e-12:
                continue
            c = abs(float(np.corrcoef(v, yv)[0, 1]))
            if np.isfinite(c) and c > bcorr:
                bcorr = c
        out[basis] = bcorr
    return out


def _make_columns(rng, n):
    """A spread of distributions whose best linearising basis varies."""
    return {
        "uniform": rng.uniform(-3, 3, n),
        "gaussian": rng.normal(0, 1, n),
        "lognormal": rng.lognormal(0, 1, n),
        "gamma": rng.gamma(2.0, 1.5, n),
        "heavytail_t": rng.standard_t(3, n),
        "skewed_sq": rng.normal(0, 1, n) ** 2,
    }


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_gpu_routing_matches_host(seed):
    import cupy as cp

    rng = np.random.default_rng(seed)
    n = 4000
    cols = _make_columns(rng, n)
    names = list(cols)
    X = [np.ascontiguousarray(cols[c], dtype=np.float64) for c in names]
    # A continuous target with mixed polynomial structure so routing has real signal.
    base = cols["gaussian"]
    y = (base**2) - 0.5 * cols["uniform"] + 0.3 * np.log1p(np.abs(cols["gamma"])) + rng.normal(0, 0.1, n)

    host_choice = [basis_route_by_signal(x, y, degrees=_DEGREES) for x in X]

    M = cp.asarray(np.ascontiguousarray(np.column_stack(X)))
    y_gpu = cp.asarray(np.asarray(y, dtype=np.float64))
    gpu_choice = _gpu_route_bases_batched(cp, M, y_gpu, list(_POLY_BASES), _DEGREES, robust_axis=True)

    assert len(gpu_choice) == len(host_choice) == len(names)
    ties = []
    for nm, h, g in zip(names, host_choice, gpu_choice):
        if h == g:
            continue
        corrs = _host_basis_corrs(cols[nm], y)
        top2 = sorted(corrs.values(), reverse=True)[:2]
        gap = (top2[0] - top2[1]) if len(top2) == 2 else 1.0
        if gap < _GAP:
            ties.append((nm, h, g, gap))
            continue
        pytest.fail(
            f"seed={seed} col={nm!r}: GPU routed {g!r} but host routed {h!r} with a CLEAR margin "
            f"(top-2 |corr| gap={gap:.2e} >= {_GAP}); per-basis corrs={corrs}"
        )
    if ties:
        print(f"seed={seed}: {len(ties)} genuine near-tie flip(s) (gap<{_GAP}), allowed: {ties}")
