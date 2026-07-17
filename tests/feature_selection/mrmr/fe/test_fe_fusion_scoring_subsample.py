"""Pins for the 2026-07-03 additive-fusion scoring row-cap (``MLFRAME_FE_FUSION_MAX_ROWS``).

The GPU-resident additive-fusion twin only DECIDES which disjoint half-pairs to fuse (per-half relevance MI +
floor, then the O(H^2) per-pair add/sub MI + OLS-R separability razor) -- wide-margin MI/floor comparisons
that are selection-equivalent under a large strided subsample. It scores halves AND fused pairs on the SAME
strided subsample above the cap, BUT an admitted compound's materialised ``values`` MUST stay full-n. These
pins assert (1) the stride formula (incl the =0 full-n opt-out) and (2) the critical correctness property:
under a cap well below n the fusion still fires and every admitted compound's ``values`` are full-n (the
scoring subsample never leaks into the output the caller materialises).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def _canonical_fixture(seed, n):
    """Canonical fixture."""
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def _fusion_stride(n, max_rows):
    """The exact strided-subsample formula used in propose_additive_fusions_gpu."""
    return int(n // max_rows) if max_rows > 0 and n > max_rows else 1


@pytest.mark.parametrize(
    "n, max_rows, expect_stride",
    [
        (1_000_000, 250_000, 4),
        (1_000_000, 0, 1),  # opt-out -> full-n
        (80_000, 250_000, 1),  # below the cap -> untouched
        (500_000, 250_000, 2),
    ],
)
def test_fusion_stride_formula(n, max_rows, expect_stride):
    """Fusion stride formula."""
    st = _fusion_stride(n, max_rows)
    assert st == expect_stride
    if max_rows == 0:
        assert np.arange(n)[::st].shape[0] == n
    elif st > 1:
        assert np.arange(n)[::st].shape[0] < n


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_fusion_scoring_subsample_keeps_output_full_n():
    """Under MLFRAME_FE_FUSION_MAX_ROWS well below n, the resident fusion twin still fuses the two canonical
    halves into one compound AND every admitted compound's materialised ``values`` are full-n (scoring is
    subsampled; output is not)."""
    pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._fe_additive_fusion_gpu_resident as TWIN
    from mlframe.feature_selection.filters.mrmr import MRMR

    n = 100_000
    cap = 30_000
    captured = {"admitted": None}
    orig = TWIN.propose_additive_fusions_gpu

    def spy(self, **kw):
        """Helper that spy."""
        res = orig(self, **kw)
        admitted = res[0]
        if admitted:
            captured["admitted"] = [np.asarray(a["values"]).shape[0] for a in admitted]
        return res

    TWIN.propose_additive_fusions_gpu = spy
    prev = os.environ.get("MLFRAME_FE_FUSION_MAX_ROWS")
    os.environ["MLFRAME_FE_FUSION_MAX_ROWS"] = str(cap)
    # Ensure the resident twin path is actually exercised.
    strict_prev = {k: os.environ.get(k) for k in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_FE_GPU_STRICT_RESIDENT")}
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    try:
        df, y = _canonical_fixture(seed=0, n=n)
        fs = MRMR(verbose=0, fe_max_steps=1, n_workers=1)
        fs.fit(df, y)
    finally:
        TWIN.propose_additive_fusions_gpu = orig
        for k, v in strict_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if prev is None:
            os.environ.pop("MLFRAME_FE_FUSION_MAX_ROWS", None)
        else:
            os.environ["MLFRAME_FE_FUSION_MAX_ROWS"] = prev

    if captured["admitted"] is None:
        pytest.skip("resident fusion twin admitted no compound (CPU fallback or no fusion on this box)")
    assert cap < n
    for _len in captured["admitted"]:
        assert _len == n, f"admitted compound values are {_len} rows, must be full-n {n} (scoring subsample leaked)"
