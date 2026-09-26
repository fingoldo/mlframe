"""The target distribution report shows the extremes and a point mass, not only p01 / p99.

A production ``total_charge`` was 74% zeros with a few refunds at -2.17: the report printed ``p01=0`` and nothing hinted
at the negatives, which then silently kept the zero-inflation model off.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.drift_report import compute_label_distribution_drift, format_drift_report


def _charges(n, seed, n_negative=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.74, 0.0, np.exp(rng.normal(4.0, 1.2, n)))
    y[:n_negative] = -2.17
    return y


def test_min_max_and_rows_below_the_point_mass_are_reported():
    report = compute_label_distribution_drift(_charges(20_000, 0, n_negative=7), _charges(2_000, 1), _charges(2_000, 2), "regression")
    train = report["splits"]["train"]
    assert train["min"] == -2.17 and train["max"] > 100
    assert train["atom"] == 0.0 and 0.70 < train["atom_share"] < 0.78 and train["n_below_atom"] == 7
    text = format_drift_report(report, "total_charge")
    line = next(l for l in text.splitlines() if l.strip().startswith("train"))
    assert "min=-2.17" in line and "point_mass=0" in line and "7 row(s) below it" in line


def test_a_continuous_target_reports_no_point_mass():
    rng = np.random.default_rng(3)
    report = compute_label_distribution_drift(rng.normal(size=5_000), rng.normal(size=500), rng.normal(size=500), "regression")
    assert report["splits"]["train"]["atom_share"] == 0.0
    assert "point_mass" not in format_drift_report(report)
