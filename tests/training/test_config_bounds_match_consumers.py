"""Config values that every consumer would silently ignore or mis-handle are rejected at construction."""

import pytest
from pydantic import ValidationError

from mlframe.training.configs import (
    ConformalConfig,
    EnsemblingConfig,
    ReportingConfig,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from mlframe.training._training_runtime_configs import MetricsConfig, SliceStableESConfig


@pytest.mark.parametrize(
    "build",
    [
        lambda: TrainingBehaviorConfig(oof_n_splits=1),  # every consumer tests >= 2
        lambda: TrainingBehaviorConfig(iteration_metrics_stride=0),
        lambda: MetricsConfig(nbins=1),
        lambda: SliceStableESConfig(k=1),  # silently fell back to classic ES
        lambda: TrainingSplitConfig(test_size=0.5, val_size=0.5),  # an empty train set
        lambda: ConformalConfig(alphas=(0.0, 0.1)),
        lambda: ConformalConfig(alphas=(0.2, 0.1)),
        lambda: ConformalConfig(alphas=()),
        lambda: EnsemblingConfig(degenerate_class_ratio=0.0),
        lambda: EnsemblingConfig(degenerate_class_ratio=1.5),
        lambda: ReportingConfig(heavy_diagnostics_for="everything"),
    ],
)
def test_the_value_is_rejected(build):
    with pytest.raises((ValidationError, ValueError)):
        build()


def test_the_valid_neighbours_still_construct():
    TrainingBehaviorConfig(oof_n_splits=0)
    TrainingBehaviorConfig(oof_n_splits=2)
    MetricsConfig(nbins=2)
    SliceStableESConfig(k=2)
    TrainingSplitConfig(test_size=0.2, val_size=0.2)
    ConformalConfig(alphas=(0.05, 0.1))
    assert ReportingConfig(heavy_diagnostics_for="all").heavy_diagnostics_for == "all"
