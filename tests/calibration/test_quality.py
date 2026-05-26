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
