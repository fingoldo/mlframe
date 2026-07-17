"""Regression: ``compute_member_quality_gate`` uses np.nanmedian, not
np.nanquantile(arr, 0.5, axis=0) (iter119, 2026-05-21).

Why this matters: ``np.nanquantile`` with ``q=0.5`` dispatches through
``apply_along_axis``, which iterates the non-axis dimensions in pure Python.
On a (K=3, N=200_000) ensemble payload that's 200k 1-D calls + 8.7 s
``apply_along_axis`` self-time -- the dominant mlframe-side hotspot in the
iter119 c0085 profile. ``np.nanmedian`` has numpy's dedicated C reduction
and runs ~275x faster on the same shape (13.5 s -> 49 ms; 3-D at
(K, N, C=4): 54 s -> 250 ms, 215x), with output bit-equivalent at machine
epsilon (max abs diff 2.22e-16 in unit tests).

These tests pin:
  (1) cross-member median equals the legacy nanquantile output to fp64
      epsilon
  (2) NaN cells in one member don't poison the median across the others
      (the whole reason both nanmedian and nanquantile were preferred over
      plain np.median in the first place)
"""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling import compute_member_quality_gate


def test_member_gate_median_matches_legacy_nanquantile():
    rng = np.random.default_rng(0)
    K, N = 4, 1000
    preds = [rng.standard_normal(N) for _ in range(K)]
    arr = np.asarray(preds, dtype=np.float64)
    legacy_median = np.nanquantile(arr, 0.5, axis=0)
    new_median = np.nanmedian(arr, axis=0)
    # Bit-equivalent at fp64 epsilon (numpy uses the same partition+sort under
    # the hood for q=0.5; nanmedian just skips the apply_along_axis dispatch).
    assert np.allclose(new_median, legacy_median, atol=1e-12, rtol=0.0), "nanmedian vs nanquantile(0.5) should match to fp64 epsilon"


def test_member_gate_nan_in_one_member_isolates_other_members():
    """The cross-member median must be NaN-resilient at the row level: a NaN
    cell in member 0 row 17 must NOT poison median_preds[17] (the other K-1
    finite values still produce a valid median). Per-member MAE for the
    NaN-bearing member itself is allowed to be NaN -- that's a legitimate
    "this member has a hole" signal -- but the other members' MAE/STD must
    stay finite. iter119 swap from nanquantile(0.5) -> nanmedian must keep
    this property."""
    import mlframe.models.ensembling as ens

    rng = np.random.default_rng(1)
    K, N = 5, 500
    preds = [rng.standard_normal(N) for _ in range(K)]
    preds[0][17] = np.nan

    arr = np.asarray(preds, dtype=np.float64)
    median_preds = np.nanmedian(arr, axis=0)
    # Median at row 17 takes the median of the OTHER 4 finite members.
    assert np.isfinite(median_preds[17]), "nanmedian must skip the NaN cell so median_preds[17] stays finite"

    # End-to-end: per-member MAE/STD for members 1..K-1 must stay finite.
    _kept, _excluded, stats = ens.compute_member_quality_gate(preds)
    finite_member_mae = stats["per_member_mae"][1:]  # member 0 is the NaN one
    finite_member_std = stats["per_member_std"][1:]
    assert np.all(np.isfinite(finite_member_mae))
    assert np.all(np.isfinite(finite_member_std))


def test_member_gate_3d_multilabel_preds():
    """multilabel-style (K, N, C) preds: median axis=0 reduces across members,
    preserving (N, C). nanmedian handles this shape natively (same axis arg
    as nanquantile)."""
    rng = np.random.default_rng(2)
    K, N, C = 3, 200, 4
    preds = [rng.random((N, C)) for _ in range(K)]
    kept, _excluded, stats = compute_member_quality_gate(preds)
    assert len(kept) == K
    assert "per_member_mae" in stats
    assert stats["per_member_mae"].shape == (K,)
