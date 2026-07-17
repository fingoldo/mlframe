"""Parity: the device within-stratum perm-null (argsort on GPU, host RNG keys) == the CPU per-perm loop.

The conditional-permutation floor falls back to permutations on sparse conditional joints (where the
analytic chi-square null is unreliable). Under STRICT/CMI_GPU the within-stratum shuffle argsort + gather
+ CMI run on the device, but the RNG KEY DRAW stays on the host (per-perm np ``rng.random``), so the
permutations -- and therefore the floor and null-mean -- are BIT-IDENTICAL to the CPU loop (no ties in
``z_rank + keys`` -> cp.argsort matches np.argsort(stable)). This pins that the GPU path does not perturb
the redundancy-gate selection. cupy-gated; the analytic null is disabled to force the permutation path."""

import os
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_conditional_perm_null_gpu_bit_identical_to_cpu(seed, monkeypatch):
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null

    # Force the permutation fallback (not the analytic chi-square null) on a sparse conditional joint.
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    rng = np.random.default_rng(seed)
    n = 4000
    x = rng.integers(0, 8, n).astype(np.int64)
    y = rng.integers(0, 6, n).astype(np.int64)
    z = rng.integers(0, 40, n).astype(np.int64)  # high-cardinality -> sparse strata -> perm fallback

    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    fh, mh = _conditional_perm_null(x, y, z, n_permutations=25, seed=seed, salt=3)
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    fg, mg = _conditional_perm_null(x, y, z, n_permutations=25, seed=seed, salt=3)

    assert abs(fh - fg) < 1e-9, f"seed={seed} floor host={fh} gpu={fg}"
    assert abs(mh - mg) < 1e-9, f"seed={seed} null_mean host={mh} gpu={mg}"
