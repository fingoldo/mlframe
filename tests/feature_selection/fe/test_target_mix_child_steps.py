"""The target-mix second pass runs a single FE step: its candidates come from the first step, and a second step over the enlarged pool is the cost it must not pay."""

from mlframe.feature_selection.filters import _target_mix_fe as TM
from mlframe.feature_selection.filters.mrmr import MRMR


def test_child_runs_one_fe_step_and_no_recursion():
    """A clone of an estimator with two FE steps and the mix on gives a child with one step and the mix off; the parent is untouched."""
    parent = MRMR(fe_max_steps=2, fe_target_mix_enable=True, verbose=2)
    child = TM._child(parent)
    assert child.get_params()["fe_max_steps"] == 1
    assert child.get_params()["fe_target_mix_enable"] is False
    assert child.get_params()["verbose"] == 0
    assert parent.fe_max_steps == 2
    assert parent.fe_target_mix_enable is True
