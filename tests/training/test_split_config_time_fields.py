"""Unit tests for TrainingSplitConfig.time_column / cv_strategy (E2 config surface).

These declare temporal-split intent; they connect to the conformal structure-inference today
(so split-conformal validity is reported honestly on temporal splits) and to the make_train_test_split
routing in the E2 follow-up. Default preserves legacy behaviour.
"""

from __future__ import annotations

import pytest

from mlframe.training._conformal_finalize import infer_split_structure
from mlframe.training._preprocessing_configs import TrainingSplitConfig


def test_defaults_are_legacy():
    cfg = TrainingSplitConfig()
    assert cfg.time_column is None
    assert cfg.cv_strategy == "random"


def test_time_fields_accepted():
    cfg = TrainingSplitConfig(time_column="ts", cv_strategy="purged")
    assert cfg.time_column == "ts"
    assert cfg.cv_strategy == "purged"


def test_forward_walk_conflicts_with_backward_val():
    with pytest.raises(ValueError, match="val_placement"):
        TrainingSplitConfig(cv_strategy="timeseries", val_placement="backward")


def test_invalid_cv_strategy_rejected():
    with pytest.raises(ValueError):
        TrainingSplitConfig(cv_strategy="bogus")


def test_conformal_structure_inference_reads_these_fields():
    cfg = TrainingSplitConfig(time_column="ts")
    structure = infer_split_structure(
        time_column=cfg.time_column,
        cv_strategy=cfg.cv_strategy,
        use_groups=cfg.use_groups,
        bucket_stratify=cfg.bucket_stratify,
        wholeday_splitting=cfg.wholeday_splitting,
    )
    # use_groups defaults True -> time + groups => temporal_grouped; the point: it is NOT plain iid.
    assert structure in ("temporal", "temporal_grouped")
