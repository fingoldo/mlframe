"""Parity: the matrix-native GPU orth-FE basis-column builder must match the host
``_evaluate_basis_column`` to fp round-off across data shapes x bases x degrees.

This is the correctness gate for the resident-candidate path (Piece 2): the orth-FE
candidate matrix is built ON the device (so it feeds _plugin_mi_classif_batch_cuda_resident
with no H2D) only if its values equal the host basis eval -- basis values are
selection-bearing (the MI ranking decides which basis/degree survives), so a silent
GPU/host drift would change selection. It also guards the Laguerre recurrence (the
prior Clenshaw form was wrong and no canonical pin exercised it -- this test caught it).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import _CUDA_AVAILABLE

if not _CUDA_AVAILABLE:
    pytest.skip("CUDA required for the GPU basis builder", allow_module_level=True)

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _evaluate_basis_column
from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_evaluate_basis_column
from mlframe.feature_selection.filters.hermite_fe._hermite_robust import _robust_axis_enabled


def _datasets(seed: int, n: int = 8000) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "uniform": rng.random(n) + 0.1,
        "gaussian": rng.standard_normal(n),
        "heavytail": np.concatenate([rng.standard_normal(n - 50), rng.standard_normal(50) * 500.0]),
        "lognormal": rng.lognormal(0.0, 1.0, n),
    }


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
@pytest.mark.parametrize("degree", [2, 3])
def test_gpu_basis_column_parity(seed, basis, degree):
    ra = _robust_axis_enabled()
    for dname, x in _datasets(seed).items():
        host = np.asarray(_evaluate_basis_column(x, basis, degree), dtype=np.float64)
        gpu = cp.asnumpy(
            _gpu_evaluate_basis_column(cp, cp.asarray(x), basis, degree, robust_axis=ra)
        ).astype(np.float64)
        denom = float(np.max(np.abs(host))) + 1e-12
        rel = float(np.max(np.abs(host - gpu))) / denom
        assert rel < 1e-6, (
            f"{dname}/{basis}/d={degree} GPU basis column drifted from host: rel={rel:.2e}"
        )
