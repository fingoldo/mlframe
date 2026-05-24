"""Sensor test for the ``_target_distribution_analyzer`` monolith split
(Wave 6a).

The parent ``mlframe.training._target_distribution_analyzer`` was carved into
four new siblings:

- ``_target_distribution_analyzer_stats`` (moment + autocorr helpers)
- ``_target_distribution_analyzer_modes`` (multi-modal + variance-ratio +
  type classifier)
- ``_target_distribution_analyzer_target_fn`` (``analyze_target_distribution``)
- ``_target_distribution_analyzer_features`` (FeatureDistributionReport +
  ``analyze_feature_distribution`` + helpers)

This sensor pins:

1. Identity preserved: every re-exported symbol on the parent is the SAME
   object as on the sibling.
2. Facade LOC budget: parent stays under 750 lines per the carve plan.
3. Smoke: at least one call routed through the parent re-export per sibling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_target_dist_stats_identity_preserved():
    from mlframe.training import _target_distribution_analyzer as parent
    from mlframe.training import _target_distribution_analyzer_stats as stats

    for name in (
        "_excess_kurtosis", "_skewness",
        "_lag1_autocorr", "_lag_autocorr", "_max_abs_lag_autocorr",
        "_lag1_autocorr_grouped", "_check_within_group_ordering",
    ):
        assert getattr(parent, name) is getattr(stats, name), name


def test_target_dist_modes_identity_preserved():
    from mlframe.training import _target_distribution_analyzer as parent
    from mlframe.training import _target_distribution_analyzer_modes as modes

    for name in (
        "_detect_multi_modal",
        "_within_between_group_variance_ratio",
        "_classify_target_type",
    ):
        assert getattr(parent, name) is getattr(modes, name), name


def test_target_dist_target_fn_identity_preserved():
    from mlframe.training import _target_distribution_analyzer as parent
    from mlframe.training import _target_distribution_analyzer_target_fn as tfn

    assert parent.analyze_target_distribution is tfn.analyze_target_distribution


def test_target_dist_features_identity_preserved():
    from mlframe.training import _target_distribution_analyzer as parent
    from mlframe.training import _target_distribution_analyzer_features as feats

    for name in (
        "FeatureDistributionReport",
        "_pairwise_redundant_features",
        "_normalise_X",
        "analyze_feature_distribution",
    ):
        assert getattr(parent, name) is getattr(feats, name), name


def test_target_dist_facade_loc_budget():
    parent_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "mlframe" / "training" / "_target_distribution_analyzer.py"
    )
    n_lines = len(parent_path.read_text(encoding="utf-8").splitlines())
    # Plan target: <750; current carve lands ~190.
    assert n_lines <= 750, f"_target_distribution_analyzer.py grew to {n_lines} lines"


def test_target_dist_smoke_stats_via_parent():
    from mlframe.training._target_distribution_analyzer import (
        _excess_kurtosis,
        _lag1_autocorr,
        _skewness,
    )

    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert abs(_excess_kurtosis(y)) < 1.0
    assert abs(_skewness(y)) < 0.5
    assert abs(_lag1_autocorr(y)) < 0.2


def test_target_dist_smoke_modes_via_parent():
    from mlframe.training._target_distribution_analyzer import (
        _classify_target_type,
        _detect_multi_modal,
    )

    # Constant array: no peaks possible => unimodal.
    y_flat = np.zeros(200, dtype=np.float64)
    is_mm, n_peaks, sep = _detect_multi_modal(y_flat)
    assert is_mm is False
    assert isinstance(n_peaks, int)
    y_int = np.array([0, 1, 2] * 100, dtype=np.int64)
    assert _classify_target_type(y_int) == "classification"


def test_target_dist_smoke_target_fn_via_parent():
    from mlframe.training._target_distribution_analyzer import (
        TargetDistributionReport,
        analyze_target_distribution,
    )

    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    rep = analyze_target_distribution(y)
    assert isinstance(rep, TargetDistributionReport)
    assert rep.n_samples == 500


def test_target_dist_smoke_features_via_parent():
    from mlframe.training._target_distribution_analyzer import (
        FeatureDistributionReport,
        analyze_feature_distribution,
    )

    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "a": rng.normal(size=200),
        "b": rng.normal(size=200),
    })
    rep = analyze_feature_distribution(df)
    assert isinstance(rep, FeatureDistributionReport)
    assert rep.n_samples == 200
