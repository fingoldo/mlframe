"""Model-to-model SHAP variance must survive a spread that is small next to the phi values.

`var = sum(phi^2)/n - mean^2` is the textbook second-moment form, and it loses the answer to cancellation
exactly where this variance is computed: a dominant feature's phi sits at 3-30 in margin/log-odds space
while the spread across `n_models` fits of the SAME data is orders of magnitude smaller. Subtracting two
nearly-equal large numbers then leaves noise, and often a negative one.

Measured on eight values, the raw form returns -1.42e-14 at phi=10 with a spread of 1e-7 against a true
4.72e-15, and is out by eight orders of magnitude at phi=1000. The `np.clip(..., 0.0, None)` that stood
around it turned that into a confident zero -- a caller reading "the models agree exactly" where they
merely agree closely, which is the opposite of what a variance is asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _StreamingPhiVariance


def _raw_moment_variance(values: np.ndarray) -> np.ndarray:
    """The pre-fix form, kept so the tests can show what they are guarding against."""
    n = values.shape[0]
    mean = values.sum(axis=0) / n
    return np.clip((values * values).sum(axis=0) / n - mean * mean, 0.0, None)


def _accumulate(values: np.ndarray) -> np.ndarray:
    """Stream the same values through the production accumulator."""
    acc = _StreamingPhiVariance(values.shape[1:])
    for row in values:
        acc.update(row)
    return acc.variance(values.shape[0])


def _phis(level: float, spread: float, n_models: int = 8, seed: int = 0) -> np.ndarray:
    """`n_models` phi matrices agreeing to within `spread` around `level`."""
    rng = np.random.default_rng(seed)
    return level + spread * rng.standard_normal((n_models, 4, 3))


@pytest.mark.parametrize(
    "level,spread",
    [(10.0, 1e-7), (100.0, 1e-7), (1000.0, 1e-6), (30.0, 1e-8)],
    ids=["phi10", "phi100", "phi1000", "phi30_tight"],
)
def test_a_small_spread_around_a_large_phi_is_measured_not_lost(level: float, spread: float):
    """The regime the multi-model SHAP variance actually runs in."""
    values = _phis(level, spread)
    expected = values.var(axis=0)
    got = _accumulate(values)
    assert np.allclose(got, expected, rtol=1e-6), f"variance lost: {got.ravel()[:3]} against {expected.ravel()[:3]}"


@pytest.mark.parametrize(
    "level,spread",
    [(10.0, 1e-7), (100.0, 1e-7), (1000.0, 1e-6)],
    ids=["phi10", "phi100", "phi1000"],
)
def test_the_raw_moment_form_reports_zero_where_there_is_real_spread(level: float, spread: float):
    """Pins what the fix is for: the old form collapses to the clip, reporting perfect agreement.

    Without this the tests above could pass for the wrong reason -- a fixture whose spread is large enough
    that any formula survives it.
    """
    values = _phis(level, spread)
    assert np.any(values.var(axis=0) > 0.0), "the fixture has no spread to lose"
    assert np.any(_raw_moment_variance(values) == 0.0), "the fixture no longer reproduces the collapse it was built to show"


def test_an_ordinary_spread_is_unchanged():
    """The fix must not move the cases the old form already handled."""
    values = _phis(1.0, 1e-2)
    assert np.allclose(_accumulate(values), _raw_moment_variance(values), rtol=1e-9)


def test_identical_models_give_exactly_zero():
    """Models that agree exactly must still read as zero variance, not as a rounding smear."""
    values = np.repeat((3.0 + np.zeros((1, 4, 3))), 8, axis=0)
    assert np.array_equal(_accumulate(values), np.zeros((4, 3)))


def test_the_variance_is_never_negative():
    """M2 is a sum of deviation-times-deviation, so the clip is a floor against the last bit, not a mask."""
    values = _phis(1000.0, 1e-9)
    assert np.all(_accumulate(values) >= 0.0)


def test_the_accumulator_matches_numpy_on_a_wide_spread():
    """A direct cross-check against numpy's own population variance."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=(16, 5, 2)) * 10.0
    assert np.allclose(_accumulate(values), values.var(axis=0), rtol=1e-12)
