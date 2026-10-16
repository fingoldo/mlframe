"""The analyzer's inline fused kurtosis/skewness must equal the standalone _excess_kurtosis / _skewness helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.targets._target_distribution_analyzer import _excess_kurtosis, _skewness, analyze_target_distribution


def _samples():
    """Gaussian, heavy-tailed and right-skewed targets."""
    rng = np.random.default_rng(7)
    return {
        "gauss": rng.normal(5.0, 2.0, 4000),
        "heavy": rng.standard_t(3, 4000) * 3.0 + 1.0,
        "skewed": rng.lognormal(0.0, 0.9, 4000),
    }


@pytest.mark.parametrize("kind", ["gauss", "heavy", "skewed"])
def test_inline_moments_match_standalone_helpers(kind):
    """diagnostics["excess_kurtosis"] / ["skew"] equal the helper values to float precision."""
    y = _samples()[kind]
    diag = analyze_target_distribution(y, has_time_axis=False).diagnostics
    assert diag["excess_kurtosis"] == pytest.approx(_excess_kurtosis(y), rel=1e-9, abs=1e-12)
    assert diag["skew"] == pytest.approx(_skewness(y), rel=1e-9, abs=1e-12)


def test_helpers_distinguish_the_distributions():
    """Negative control: the parity above is not vacuous, the three samples have clearly different moments."""
    s = _samples()
    assert _excess_kurtosis(s["heavy"]) > _excess_kurtosis(s["gauss"]) + 2.0
    assert _skewness(s["skewed"]) > _skewness(s["gauss"]) + 1.0
