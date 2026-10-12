"""PIT goodness-of-fit statistics reach the production PIT diagram through ``pit_gof_summary``.

``build_pit_diagram_spec`` is the path ``plot_pit_diagram`` renders through, so these tests drive that builder and
assert on the numbers it prints, checked against each standalone statistic function.
"""

import numpy as np
import pytest

from mlframe.calibration.quality import (
    anderson_darling_statistic,
    build_pit_diagram_spec,
    chi_square_statistic,
    cramer_von_mises_statistic,
    entropy_calibration_index,
    kolmogorov_smirnov_statistic,
    mean_squared_deviation,
    pit_gof_summary,
    weighted_pit_deviation,
)


def _uniform_pit(n: int, seed: int) -> np.ndarray:
    """Return ``n`` reproducible uniform PIT values drawn from a seeded generator."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)


def test_pit_gof_summary_matches_each_standalone_statistic():
    """Every summary entry equals the output of the statistic function it names, on the same input."""
    pit = _uniform_pit(3_000, seed=11)
    summary = pit_gof_summary(pit)

    assert summary["ks"] == pytest.approx(kolmogorov_smirnov_statistic(pit))
    assert summary["cvm"] == pytest.approx(cramer_von_mises_statistic(pit))
    assert summary["anderson_darling"] == pytest.approx(anderson_darling_statistic(pit))
    assert summary["chi_square"] == pytest.approx(chi_square_statistic(pit))
    assert summary["eci"] == pytest.approx(entropy_calibration_index(pit))
    assert summary["msd"] == pytest.approx(mean_squared_deviation(pit))
    assert summary["wpd"] == pytest.approx(weighted_pit_deviation(pit))


def test_pit_diagram_caption_prints_the_summary_statistics():
    """The caption built for the PIT diagram carries the CvM, AD, chi2, ECI, MSD and WPD values computed on the same PIT."""
    pit = _uniform_pit(2_000, seed=23)
    spec = build_pit_diagram_spec(pit)

    expected = (
        f"CvM={cramer_von_mises_statistic(pit):.4f}, "
        f"AD={anderson_darling_statistic(pit):.4f}, "
        f"chi2={chi_square_statistic(pit):.4f}, "
        f"ECI={entropy_calibration_index(pit):.4f}, "
        f"MSD={mean_squared_deviation(pit):.4f}, "
        f"WPD={weighted_pit_deviation(pit):.4f}"
    )
    assert expected in spec.caption


def test_pit_diagram_title_ks_reflects_miscalibration():
    """A PIT skewed toward 1 yields a larger KS-vs-uniform title value than uniform PIT, read back from the built spec."""
    rng = np.random.default_rng(5)
    uniform_spec = build_pit_diagram_spec(_uniform_pit(4_000, seed=5))
    skewed_spec = build_pit_diagram_spec(rng.beta(4.0, 1.0, size=4_000))

    def _ks_from_title(spec) -> float:
        """Parse the KS value printed in the PIT panel title."""
        title = spec.panels[0][0].title
        return float(title.split("KS-vs-uniform=")[1].rstrip(")"))

    assert _ks_from_title(skewed_spec) > _ks_from_title(uniform_spec) + 0.1


def test_pit_diagram_empty_input_has_no_statistics_line():
    """Empty PIT input builds a NaN KS title and omits the statistics sentence instead of raising."""
    spec = build_pit_diagram_spec(np.array([], dtype=np.float64))

    assert "KS-vs-uniform=nan" in spec.panels[0][0].title
    assert "Statistics:" not in spec.caption
