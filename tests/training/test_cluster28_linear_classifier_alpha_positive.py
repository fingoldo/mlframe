"""Audit 2026-11-20 finding 28-F22: lasso and elasticnet classifiers refuse alpha <= 0 instead of silently using C=1.0.

The old mapping turned alpha=0 into a moderate penalty the user did not ask for. The test checks both builders refuse alpha=0 and that a
positive alpha still builds.
"""

import pytest

from mlframe.training import models
from mlframe.training._model_configs import LinearModelConfig


@pytest.mark.parametrize("builder", ["_build_lasso_classifier", "_build_elasticnet_classifier"])
def test_zero_alpha_is_refused(builder):
    """alpha=0 raises ValueError naming the builder, rather than fitting with a penalty."""
    config = LinearModelConfig(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        getattr(models, builder)(config)


def test_positive_alpha_still_builds():
    """A positive alpha builds the lasso classifier with C = 1/alpha."""
    model = models._build_lasso_classifier(LinearModelConfig(alpha=0.5))
    assert model.C == pytest.approx(2.0)
