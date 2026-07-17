"""Stacked-discovery no-win warning.

``use_stacked_discovery`` / ``use_stacked_discovery_residual`` have a measured
no-win (profiling/bench_stacked_discovery_default_flip.py). The config validator
warns when either is enabled so callers know the flag does not help on benchmarks.
"""

from __future__ import annotations

import warnings

import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_warns_when_use_stacked_discovery_true():
    with pytest.warns(UserWarning, match="no measurable improvement"):
        CompositeTargetDiscoveryConfig(use_stacked_discovery=True)


def test_warns_when_use_stacked_discovery_residual_true():
    with pytest.warns(UserWarning, match="no measurable improvement"):
        CompositeTargetDiscoveryConfig(use_stacked_discovery_residual=True)


def test_no_warning_when_both_false():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CompositeTargetDiscoveryConfig(use_stacked_discovery=False, use_stacked_discovery_residual=False)
