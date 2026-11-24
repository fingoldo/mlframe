"""Cluster 08 regression: the flip-distance docstring states the rule the code implements (audit 2026-11-20, finding 08-F6).

The regression flip is a move of the prediction by more than the 10th percentile of the training absolute residuals, not a crossing of a
residual-decile boundary. The module docstring now says so.
"""

from mlframe.feature_engineering.transformer import adversarial_flip as af


def test_docstring_states_the_implemented_regression_rule():
    """The regression rule is described as a threshold on the prediction change (08-F6)."""
    doc = af.__doc__ or ""
    assert "10th percentile of the train absolute residuals" in doc
    assert "across one residual-decile boundary" not in doc
