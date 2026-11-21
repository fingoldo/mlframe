"""Audit 2026-11-20 finding 04-F18: the oracle's third estimator is named for what it computes, a coverage-adjusted plug-in, not Chao-Shen entropy."""

from mlframe.data.datasets import _oracle


def test_the_estimator_is_named_as_the_coverage_adjusted_plugin_it_is():
    """The public estimator list names the coverage-adjusted plug-in; the Chao-Shen name is gone."""
    assert "coverage_adjusted_plugin" in _oracle.ORACLE_ESTIMATORS
    assert "chao_shen" not in _oracle.ORACLE_ESTIMATORS
