"""Cluster 27 regression: mapping 'iterations' to 'max_iter' does not mutate the caller's dict (audit 2026-11-20, finding 27-F17)."""

from mlframe.training._model_configs import LinearModelConfig


def test_caller_dict_is_not_mutated():
    """The validator leaves the caller's 'iterations' key in place (27-F17)."""
    params = {"iterations": 50}
    cfg = LinearModelConfig.model_validate(params)
    assert params == {"iterations": 50}
    assert cfg.max_iter == 50


def test_explicit_max_iter_takes_precedence_and_keeps_input():
    """Control: when max_iter is also given, it wins and the input is still untouched (27-F17)."""
    params = {"iterations": 50, "max_iter": 200}
    cfg = LinearModelConfig.model_validate(params)
    assert params == {"iterations": 50, "max_iter": 200}
    assert cfg.max_iter == 200
