"""A member with non-finite predictions must not reach the blend.

Measured before the fix: three good members plus one all-NaN member gave `kept [0, 1, 2, 3] excluded []` (every
`tot_mae > threshold` comparison against NaN is False), and `combine_probs(..., "arithm")` then returned NaN for every
row - the documented "NaN/inf fallback to arithmetic mean" was itself `np.mean` over the NaN-bearing stack. Downstream
`combined[:, 1] >= threshold` is False on a NaN row, so the whole target was classified 0 with nothing in the log.
"""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling.base import combine_probs
from mlframe.models.ensembling.quality_gate import compute_member_quality_gate


def _stack(n: int = 200, k: int = 4, nan_member: int | None = 3) -> np.ndarray:
    """A stacked probability array of ``k`` members, one of them all-NaN unless ``nan_member`` is None."""
    rng = np.random.default_rng(0)
    p = rng.random((k, n))
    stacked = np.stack([np.stack([1.0 - p[i], p[i]], axis=1) for i in range(k)], axis=0)
    if nan_member is not None:
        stacked[nan_member] = np.nan
    return stacked


def test_the_gate_excludes_the_non_finite_member():
    """The all-NaN member is dropped with a reason, rather than passing because every NaN comparison is False."""
    kept, excluded, info = compute_member_quality_gate(_stack())
    assert kept == [0, 1, 2]
    assert [i for i, _ in excluded] == [3]
    assert "non-finite" in excluded[0][1]
    assert not np.isfinite(info["per_member_mae"][3])


def test_the_too_restrictive_fallback_never_restores_a_non_finite_member():
    """With an impossible absolute bar every finite member is excluded too; the defensive "filter too tight" path
    restores those, but restoring the NaN member would reinstate the all-NaN blend."""
    kept, _, info = compute_member_quality_gate(_stack(), max_mae=1e-12)
    assert kept == [0, 1, 2]
    assert info.get("filter_too_restrictive") is True


def test_the_blend_fallback_ignores_a_non_finite_member_instead_of_reproducing_it():
    """If the gate is bypassed the blend must still skip the NaN member, not propagate NaN into every output row."""
    combined = combine_probs(_stack(), "arithm")
    assert np.isfinite(combined).all()
    np.testing.assert_allclose(combined, np.nanmean(_stack(), axis=0))


def test_the_weighted_blend_fallback_renormalises_over_the_finite_members():
    """Dropping a member must renormalise the weights over those that remain, not leave them summing to less than one."""
    stacked = _stack()
    weights = [0.4, 0.3, 0.2, 0.1]
    combined = combine_probs(stacked, "arithm", precomputed_weights=weights)
    assert np.isfinite(combined).all()
    expected = np.average(stacked[:3], axis=0, weights=weights[:3])
    np.testing.assert_allclose(combined, expected, rtol=1e-12)


def test_a_cell_no_member_could_predict_stays_non_finite():
    """There is nothing to fall back to when every member is NaN on a row; the fix must not invent a value."""
    stacked = _stack(nan_member=None)
    stacked[:, 5, :] = np.nan
    combined = combine_probs(stacked, "arithm")
    assert not np.isfinite(combined[5]).any()
    assert np.isfinite(np.delete(combined, 5, axis=0)).all()
