"""Regression pin (audit3 cross-backend-P1): polyeval_dispatch's njit single-thread kernel and its
njit_parallel twin diverge only by FP reassociation (~1e-14, self-documented "not bit-identical"), and the
result feeds the basis/degree MI selection. There was NO direct par-vs-single equivalence pin at the
crossover (only a GEMV<->Horner small-n check). This pins that the two CPU backends agree to a tolerance far
below anything that could flip a downstream selection, for every basis -- so a future kernel change cannot
silently introduce a selection-altering divergence between the size-gated backends.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch, _NJIT_FUNCS

_BASES = sorted(_NJIT_FUNCS.keys())


@pytest.mark.parametrize("basis", _BASES)
def test_njit_single_and_parallel_agree_within_selection_tolerance(basis, monkeypatch):
    rng = np.random.default_rng(0)
    # n large enough that the parallel kernel does real multi-chunk work (its chunking is what reassociates).
    x = rng.uniform(-0.9, 0.9, size=20000).astype(np.float64)
    c = np.array([0.5, -0.3, 0.2, -0.1, 0.05, -0.02], dtype=np.float64)  # degree-5 coefficients

    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
    r_single = np.asarray(polyeval_dispatch(basis, x, c), dtype=np.float64)
    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit_par")
    r_par = np.asarray(polyeval_dispatch(basis, x, c), dtype=np.float64)

    assert r_single.shape == r_par.shape == x.shape
    # ~1e-14 documented divergence; pin at rtol 1e-10 -- passes with huge margin yet trips loudly on any
    # future change that introduces a selection-flipping (>=1e-3) divergence between the two CPU backends.
    max_rel = np.max(np.abs(r_par - r_single) / (np.abs(r_single) + 1e-12))
    assert np.allclose(r_single, r_par, rtol=1e-10, atol=1e-12), (
        f"{basis}: njit vs njit_par diverge (max_rel={max_rel:.2e}) -- a selection-altering backend split"
    )
