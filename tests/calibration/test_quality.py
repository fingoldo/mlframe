"""Smoke tests for mlframe.calibration.quality (W5-4)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

# B2#33 DRY: pre-fix every test body re-importorskip'd the same (properscoring + matplotlib) pair (7 sites = 14
# probes). Module-level skip via pytest.importorskip collects both contracts once at module import; if either dep
# is absent the whole file deselects with a single skip reason rather than re-skipping per test.
pytest.importorskip("properscoring")
pytest.importorskip("matplotlib")


@pytest.mark.fast
def test_import_quality_module():
    """Calibration quality module imports cleanly with public callables present."""
    from mlframe.calibration import quality as q

    for name in (
        "bin_predictions",
        "estimate_calibration_quality_binned",
        "kolmogorov_smirnov_statistic",
        "mean_squared_deviation",
        "entropy_calibration_index",
    ):
        assert callable(getattr(q, name)), f"{name} not callable"


@pytest.mark.fast
def test_msd_uniform_pit_below_quarter():
    """Mean squared deviation of (near-)uniform PIT values stays below the 0.25 worst case."""
    from mlframe.calibration.quality import mean_squared_deviation

    rng = np.random.default_rng(0)
    pit = rng.uniform(0.0, 1.0, size=2_000)
    msd = mean_squared_deviation(pit)
    # Worst case (point mass at 0 or 1) is 0.25; uniform should be near 1/12 ~ 0.083.
    assert 0.0 < msd < 0.20


@pytest.mark.fast
def test_ks_statistic_uniform_small():
    """KS stat against uniform CDF on a uniform sample stays small."""
    from mlframe.calibration.quality import kolmogorov_smirnov_statistic

    rng = np.random.default_rng(1)
    pit = rng.uniform(0.0, 1.0, size=2_000)
    stat = kolmogorov_smirnov_statistic(pit)
    assert 0.0 <= stat < 0.10


def _ad_reference(pit_values):
    """Reference numpy A-D statistic (the pre-fused implementation)."""
    n = len(pit_values)
    if n == 0:
        return float("nan")
    sorted_pit = np.sort(pit_values)
    i = np.arange(1, n + 1)
    eps = 1e-12
    sorted_pit = np.clip(sorted_pit, eps, 1.0 - eps)
    return -n - (1 / n) * np.sum((2 * i - 1) * (np.log(sorted_pit) + np.log(1 - sorted_pit[::-1])))


@pytest.mark.fast
@pytest.mark.parametrize("kind", ["uniform", "tied", "boundary", "all_zero", "all_one"])
def test_anderson_darling_fused_matches_numpy_reference(kind):
    """Fused njit A-D kernel reproduces the reference numpy formula to FP reduction-order (~1e-9)."""
    from mlframe.calibration.quality import anderson_darling_statistic

    rng = np.random.default_rng(7)
    n = 20_000
    if kind == "uniform":
        pit = rng.uniform(0.0, 1.0, size=n)
    elif kind == "tied":
        pit = np.round(rng.uniform(0.0, 1.0, size=n), 2)
    elif kind == "boundary":
        pit = np.clip(rng.normal(0.5, 2.0, size=n), 0.0, 1.0)
    elif kind == "all_zero":
        pit = np.zeros(n)
    else:
        pit = np.ones(n)

    got = anderson_darling_statistic(pit)
    ref = _ad_reference(pit)
    assert abs(got - ref) <= 1e-7 * max(abs(ref), 1.0), f"{kind}: {got} vs {ref}"


@pytest.mark.fast
def test_anderson_darling_uses_fused_njit_kernel():
    """Sensor: the public A-D path must route through the fused njit kernel, not the numpy loop."""
    from mlframe.calibration import quality as q

    called = {"n": 0}
    orig = q._anderson_darling_kernel

    def spy(sorted_pit, n):
        called["n"] += 1
        return orig(sorted_pit, n)

    q._anderson_darling_kernel = spy
    try:
        q.anderson_darling_statistic(np.linspace(0.01, 0.99, 1000))
    finally:
        q._anderson_darling_kernel = orig
    assert called["n"] == 1, "anderson_darling_statistic did not call the fused kernel"


@pytest.mark.fast
def test_anderson_darling_empty_is_nan():
    """Empty input returns NaN (guards the n==0 division)."""
    from mlframe.calibration.quality import anderson_darling_statistic

    import math

    assert math.isnan(anderson_darling_statistic(np.array([])))
