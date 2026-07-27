"""GPU_INFRA_C-3 fix (mrmr_audit_2026-07-22): best_existing_op_mi_resident had zero dedicated
parity/correctness test anywhere in tests/, unlike its sibling gate_grid_mi_resident in the same file (which
has test_device_born_gate_grid_parity.py). Compounding this, the KTC sweep gating production engagement of
this path (_run_rescand_sweep) checks equivalence at equiv_rtol=5e-2/equiv_atol=5e-2 -- 4-7 orders of
magnitude looser than nearly every sibling KTC sweep in this cluster, so a genuinely wrong resident
implementation could be crowned "fastest" by that loose tolerance alone. This file adds a SEPARATE,
tight-tolerance correctness gate independent of the sweep's wall-clock ranking tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is present, so the module can skip itself when there is none."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


@pytest.mark.parametrize("rank_binning", [False, True], ids=["edge", "rank"])
@pytest.mark.parametrize("nclasses", [2, 4])
def test_best_existing_op_mi_resident_close_to_host(rank_binning, nclasses, monkeypatch):
    """The resident device-born max-MI must be CLOSE (tight tolerance, not the sweep's loose 5e-2 ranking
    tolerance) to the host njit best_existing_op_mi on the same operand set, and never silently return a
    wildly different value that the loose sweep tolerance alone could mask."""
    from mlframe.feature_selection.filters._conditional_gate_fe import best_existing_op_mi
    from mlframe.feature_selection.filters._resident_candidate_mi import best_existing_op_mi_resident
    import mlframe.feature_selection.filters._conditional_gate_fe as cg

    monkeypatch.setattr(cg, "_gate_rank_binning", lambda: rank_binning)

    rng = np.random.default_rng(20260722)
    n = 4000
    a = rng.uniform(0.1, 1.1, n)
    b = rng.uniform(0.1, 1.1, n)
    c = rng.uniform(0.1, 1.1, n)
    yi = rng.integers(0, nclasses, n).astype(np.int64)
    arrs = {"a": a, "b": b, "c": c}
    names = ("a", "b", "c")
    nbins = 20

    host_mi = float(best_existing_op_mi(arrs, list(names), yi, nbins))
    dev_mi = best_existing_op_mi_resident(arrs, names, yi, nbins, rank_binning=rank_binning)
    assert dev_mi is not None, "resident best_existing_op_mi_resident returned None on a CUDA host"

    # Tight-ish tolerance: the two paths use different binning schemes (percentile-edge vs rank) which are
    # selection-equivalent but not bit-identical, so some FP slack is legitimate -- but nowhere near the
    # sweep's 5e-2 ranking tolerance. 1e-2 catches a genuinely broken implementation while tolerating the
    # documented binning-scheme divergence.
    assert abs(dev_mi - host_mi) < 1e-2, (
        f"resident best_existing_op_mi_resident diverged from host ({'rank' if rank_binning else 'edge'}, "
        f"nclasses={nclasses}): resident={dev_mi:.6f} host={host_mi:.6f}"
    )
