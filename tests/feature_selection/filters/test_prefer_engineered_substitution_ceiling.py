"""The prefer-engineered substitution scores the child with the budget-aware confidence of the normal confirmed path (audit 2026-11-20 A-F02).

At ``full_npermutations = 3`` the add-one confidence of a clean candidate is 0.75, which the normal path divides out; the substitution multiplied the gain by the raw 0.75 and could drop a clean child below the
gain floor its raw parent had cleared.
"""

from types import SimpleNamespace

import pytest

from mlframe.feature_selection.filters import _confirm_predictor as cp


def _run(monkeypatch, conf, npermutations):
    """Substitute one child (gain 0.0105, confidence ``conf``) and return the stored gain."""
    monkeypatch.setattr(cp, "confirm_candidate", lambda ctx, child_x, gain: (1.0, conf))
    ctx = SimpleNamespace(full_npermutations=npermutations)
    added: set = set()
    child = (7, ("child",), 0.0105)
    _x, gain, idx = cp._confirm_one_predic_becomes_redundant_marked_accepted(child, ctx, added, 3, 0, ("parent",), None, 6, 0.01, None, 0.0, -1)
    assert idx == 7 and 3 in added
    return gain


def test_clean_child_keeps_its_gain_at_a_tiny_budget(monkeypatch):
    """Confidence 0.75 at three permutations is the add-one ceiling: the clean child keeps the full gain."""
    assert _run(monkeypatch, 0.75, 3) == pytest.approx(0.0105)


def test_weaker_confidence_still_scales_the_gain(monkeypatch):
    """A child below the ceiling is still scaled by confidence / ceiling (0.5 / 0.75)."""
    assert _run(monkeypatch, 0.5, 3) == pytest.approx(0.0105 * 0.5 / 0.75)
