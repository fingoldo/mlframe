"""MODELS-4 regression test: check_rules's allow_if_values_or branch must not depend on the gate loop's
leftover ``condition`` variable.

The bug (fixed): after the ``for condition in conditions: ...`` gate loop, the branch re-checked
``check_condition(condition, params)`` using whatever ``condition`` happened to be left over from the
loop -- dead code (re-testing an already-true condition) when ``conditions`` was non-empty, and a
``NameError`` when ``conditions`` was an empty tuple (a legal DSL input meaning "no gate, always evaluate
the field conditions").
"""

from __future__ import annotations

import pytest

from mlframe.models.tuning import HashableDict, check_rules

pytestmark = pytest.mark.fast


def test_empty_gate_conditions_does_not_raise_nameerror():
    """An empty conditions tuple in allow_if_values_or must not raise NameError."""
    rule = {(): [{"x": 1}]}
    result = check_rules({"x": 1}, allow_if_values_or=rule)
    assert result is True


def test_empty_gate_conditions_rejects_when_no_field_condition_holds():
    """With an empty gate (always active), the candidate is rejected if no field condition holds."""
    rule = {(): [{"x": 999}]}
    result = check_rules({"x": 1}, allow_if_values_or=rule)
    assert result is False


def test_nonempty_gate_still_evaluates_field_conditions_correctly():
    """A non-empty gate that holds still correctly evaluates the OR over field conditions."""
    gate = (HashableDict({"gate": True}),)
    rule = {gate: [{"x": 1}, {"x": 2}]}
    assert check_rules({"gate": True, "x": 2}, allow_if_values_or=rule) is True
    assert check_rules({"gate": True, "x": 3}, allow_if_values_or=rule) is False
    # Gate not active -> rule is skipped entirely, candidate survives regardless of x.
    assert check_rules({"gate": False, "x": 3}, allow_if_values_or=rule) is True
