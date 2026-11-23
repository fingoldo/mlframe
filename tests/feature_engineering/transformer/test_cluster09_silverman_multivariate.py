"""Cluster 09 regression: the Silverman bandwidth uses the d-dimensional rule (audit 2026-11-20, finding 09-F38).

At d=1 the multivariate rule reduces to the 1-D rule (4/(3n))^(1/5), so that case is unchanged. For d>1 the bandwidth must equal
sigma * (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)).
"""

import numpy as np

from mlframe.feature_engineering.transformer.density_ratio import _silverman_h


def test_one_dimensional_case_matches_the_one_d_rule():
    """At d=1 the bandwidth equals the classic 1-D Silverman rule (09-F38 control)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    sigma = X.std(axis=0).mean()
    expected = sigma * (4.0 / (3.0 * 200)) ** 0.2
    assert np.isclose(_silverman_h(X), max(expected, 1e-3))


def test_multivariate_case_uses_the_d_dependent_exponent():
    """At d=20 the bandwidth follows (4/(d+2))^(1/(d+4)) n^(-1/(d+4)) (09-F38)."""
    rng = np.random.default_rng(1)
    n, d = 300, 20
    X = rng.normal(size=(n, d))
    sigma = X.std(axis=0).mean()
    expected = sigma * (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4))
    assert np.isclose(_silverman_h(X), max(expected, 1e-3))
    old_one_d = sigma * (4.0 / (3.0 * n)) ** 0.2
    assert not np.isclose(old_one_d, expected)
