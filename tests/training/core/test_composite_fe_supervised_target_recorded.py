"""The target that supervises the suite-level composite-FE steps is recorded and, for several targets, logged."""

import logging

import numpy as np
import pandas as pd

from mlframe.training.core._phase_helpers_fit_pipeline import _composite_fe_supervised_target


def test_the_chosen_target_is_recorded_and_logged_for_a_multi_target_suite(caplog):
    targets = {"binary": {"churn": pd.Series([0, 1, 0]), "fraud": pd.Series([1, 1, 0])}}
    metadata: dict = {}
    with caplog.at_level(logging.INFO, logger="mlframe.training.core"):
        y = _composite_fe_supervised_target(targets, metadata)
    assert np.array_equal(y, [0, 1, 0])
    assert metadata["composite_fe_supervised_target"] == {"target_type": "binary", "target_name": "churn"}
    assert any("supervised by target binary/churn for all 2 targets" in r.getMessage() for r in caplog.records)


def test_a_single_target_is_recorded_without_a_log(caplog):
    metadata: dict = {}
    with caplog.at_level(logging.INFO, logger="mlframe.training.core"):
        _composite_fe_supervised_target({"regression": {"y": np.arange(3.0)}}, metadata)
    assert metadata["composite_fe_supervised_target"]["target_name"] == "y"
    assert not [r for r in caplog.records if "supervised by target" in r.getMessage()]
