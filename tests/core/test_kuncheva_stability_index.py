"""Kuncheva stability index: chance correction, clamping, degenerate cases, and RFECV parity.

Pins the behaviour of the shared ``mlframe.core.set_similarity.kuncheva`` kernel extracted out of
``RFECV.selection_stability_(metric="kuncheva")``, including the clamp that folds the whole
below-chance half of the range onto 0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.set_similarity import kuncheva


def test_kuncheva_identical_sets_is_one():
    """The index is 1 only for identical subsets, independent of ``k`` and ``N``."""
    assert kuncheva({"a", "b", "c"}, {"a", "b", "c"}, 10) == pytest.approx(1.0)


def test_kuncheva_matches_closed_form():
    """Pinned against the published formula ``(r - k^2/N) / (k - k^2/N)`` computed by hand."""
    # k=3, N=10, r=1 -> expected = 0.9; (1 - 0.9) / (3 - 0.9)
    assert kuncheva({"a", "b", "c"}, {"a", "d", "e"}, 10) == pytest.approx((1 - 0.9) / (3 - 0.9))


def test_kuncheva_is_chance_corrected_unlike_jaccard():
    """Two disjoint 2-subsets of a 4-item universe are BELOW chance; jaccard reports a flat 0."""
    assert kuncheva({"a", "b"}, {"c", "d"}, 4, clamp=False) < 0.0


def test_kuncheva_clamp_destroys_the_below_chance_range():
    """Every below-chance pairing collapses to exactly 0.0 under the default clamp."""
    mildly_below = kuncheva({"a", "b", "c"}, {"c", "d", "e"}, 6, clamp=False)
    fully_disjoint = kuncheva({"a", "b", "c"}, {"d", "e", "f"}, 6, clamp=False)
    assert fully_disjoint < mildly_below < 0.0
    assert kuncheva({"a", "b", "c"}, {"c", "d", "e"}, 6) == 0.0
    assert kuncheva({"a", "b", "c"}, {"d", "e", "f"}, 6) == 0.0


def test_kuncheva_boolean_masks_agree_with_sets():
    """A boolean mask and the equivalent index set must give the same number; both are accepted representations."""
    a = np.array([True, True, False, False, False])
    b = np.array([True, False, True, False, False])
    assert kuncheva(a, b, 5) == pytest.approx(kuncheva({0, 1}, {0, 2}, 5))


@pytest.mark.parametrize("k, n", [(0, 10), (0, 0), (4, 4)])
def test_kuncheva_degenerate_denominator_returns_equality_indicator(k, n):
    """When ``k`` is 0, ``N`` is 0, or ``k == N`` the denominator is undefined because only one subset is possible."""
    a = set(range(k))
    assert kuncheva(a, set(a), n) == 1.0


def test_kuncheva_degenerate_unequal_content_returns_zero():
    """Same undefined denominator, but the subsets differ, so the degenerate branch reports 0 rather than 1."""
    assert kuncheva({0, 1, 2}, {1, 2, 3}, 3) == 0.0


def test_kuncheva_rejects_unequal_cardinality():
    """The index is defined only for equal-cardinality subsets; the original RFECV copy silently took ``len(a)``."""
    with pytest.raises(ValueError, match="equal-cardinality"):
        kuncheva({"a"}, {"a", "b"}, 5)


def test_kuncheva_rejects_universe_smaller_than_subset():
    """A universe smaller than the subset drawn from it is incoherent and would make the chance term exceed 1."""
    with pytest.raises(ValueError, match="universe_size"):
        kuncheva({"a", "b", "c"}, {"a", "b", "d"}, 2)


def test_rfecv_diagnostics_uses_the_shared_kernel():
    """The RFECV accessor's kuncheva branch must produce the same number as the free function."""
    from mlframe.feature_selection.wrappers.rfecv._diagnostics import selection_stability_

    class _Fake:
        """Minimal stand-in exposing only the three attributes ``selection_stability_`` reads."""
        n_features_ = 2
        n_features_in_ = 6
        feature_importances_ = {
            "2_0": {"f0": 1.0, "f1": 0.9, "f2": 0.1, "f3": 0.0, "f4": 0.0, "f5": 0.0},
            "2_1": {"f0": 1.0, "f2": 0.9, "f1": 0.1, "f3": 0.0, "f4": 0.0, "f5": 0.0},
        }

    got = selection_stability_(_Fake(), metric="kuncheva")
    assert got == pytest.approx(kuncheva({"f0", "f1"}, {"f0", "f2"}, 6))
