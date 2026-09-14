"""The never-empty raw re-attach must not act on an unavailable subsumption verdict (mrmr_audit_2026-09-14 NUM-18).

When only engineered features survive, MRMR re-attaches one raw operand as a stand-in, restricted to operands the conditional-redundancy
verdict does NOT judge subsumed by the engineered child (``a`` in ``a**2/b`` is subsumed and must not come back). On any failure the verdict
became an empty set at debug, which made EVERY operand eligible: the re-attach then resurrected exactly the operand the check exists to
exclude. The ``emit_both`` policy (no restriction) was signalled by raising and catching a ``RuntimeError``, so a genuine failure and the
policy were indistinguishable.
"""

from __future__ import annotations

import logging
import types

import pytest

import mlframe.feature_selection.filters._fe_raw_redundancy_drop as rrd
from mlframe.feature_selection.filters._mrmr_fit_impl import _assign_support as asup

_KW = dict(data=None, cols=["a", "b", "eng"], cols_idx={"a": 0, "b": 1, "eng": 2}, operand_idxs={0, 1}, raw_names={"a", "b"}, classes_y=None, y=[0, 1], engineered_continuous=None, X=None)


def _est(policy: str):
    """An estimator carrying one engineered recipe named 'eng'."""
    return types.SimpleNamespace(redundancy_policy=policy, _engineered_recipes_=["eng"], random_seed=0)


def test_failed_verdict_under_drop_is_unavailable_and_warned(monkeypatch, caplog):
    """A genuine failure is reported as 'no verdict', not as 'nothing is subsumed'."""

    def _boom(**kwargs):
        """Stand in for the redundancy verdict failing."""
        raise RuntimeError("synthetic verdict failure")

    monkeypatch.setattr(rrd, "drop_redundant_raw_operands", _boom)
    with caplog.at_level(logging.WARNING):
        verdict = asup._subsumed_operands_verdict(_est("drop"), recipe_name=str, **_KW)
    assert verdict is None, f"a failed verdict was reported as {verdict!r}, which makes every operand eligible"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING and "RuntimeError" in r.getMessage()], "the failed verdict was not reported"


def test_emit_both_applies_no_restriction_and_is_not_a_failure(monkeypatch, caplog):
    """Control: under emit_both the verdict is not computed at all, and that is a policy, not an error."""

    def _must_not_run(**kwargs):
        """The verdict must not be computed under emit_both."""
        raise AssertionError("the subsumption verdict ran under emit_both")

    monkeypatch.setattr(rrd, "drop_redundant_raw_operands", _must_not_run)
    with caplog.at_level(logging.DEBUG):
        verdict = asup._subsumed_operands_verdict(_est("emit_both"), recipe_name=str, **_KW)
    assert verdict == set()
    assert not [r for r in caplog.records if "failed" in r.getMessage() or "unavailable" in r.getMessage()], "emit_both was logged as a failure"


def test_successful_verdict_returns_the_subsumed_names(monkeypatch):
    """Control: a working verdict passes its dropped names through."""
    monkeypatch.setattr(rrd, "drop_redundant_raw_operands", lambda **kwargs: (None, ["a"]))
    assert asup._subsumed_operands_verdict(_est("drop"), recipe_name=str, **_KW) == {"a"}


@pytest.mark.parametrize(
    "subsumed, expected",
    [(None, []), (set(), [0, 1]), ({"a"}, [1])],
)
def test_never_empty_eligibility_fails_closed_without_a_verdict(subsumed, expected):
    """No verdict -> no stand-in; an empty verdict -> every operand; a real verdict -> only the operands it does not subsume."""
    assert sorted(asup._never_empty_eligible({0, 1}, ["a", "b", "eng"], subsumed, set())) == expected
