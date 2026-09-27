"""Stratification keys treat a missing label as its own stratum.

Before: a single classification target with NaN reached sklearn's splitter, which rejects NaN; several classification
targets were stacked into a key where every NaN row was unique, so stratification was switched off; a multilabel NaN
was cast to True; a regression NaN landed in the top quantile bucket.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.core._phase_helpers_fit_split import _multilabel_for_stratify, _phase_train_val_test_split, _stratify_codes
from tests.training.test_bucket_stratify_default import _DummySplitConfig


class _Binary:
    name = "binary_classification"


def _split(target_by_type, n):
    df = pd.DataFrame({"x1": np.random.default_rng(0).standard_normal(n)})
    return _phase_train_val_test_split(
        df=df, target_by_type=target_by_type, timestamps=None, group_ids=None, group_ids_raw=None, artifacts=None,
        sequences=None, split_config=_DummySplitConfig(use_groups=False), behavior_config=type("B", (), {"fairness_features": None})(),
        metadata={}, data_dir=None, models_dir=None, target_name="y", model_name="m", df_size_mb=1.0, verbose=False,
    )


def test_a_target_without_missing_labels_is_passed_through_unchanged():
    y = np.array([0, 1, 1, 0])
    assert _stratify_codes(y) is y


def test_codes_keep_label_order_and_mark_missing():
    codes = _stratify_codes(np.array([2.0, np.nan, 0.0, 2.0, 1.0]))
    assert codes.tolist() == [2, -1, 0, 2, 1]


def test_multilabel_missing_reads_as_zero_not_one():
    assert _multilabel_for_stratify(np.array([[1.0, np.nan], [0.0, 1.0]])).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_one_classification_target_with_missing_labels_still_stratifies():
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.08).astype(float)
    y[rng.random(n) < 0.4] = np.nan
    res = _split({_Binary(): {"y": y}}, n)
    pos_rate = lambda idx: np.nanmean(y[idx])  # noqa: E731
    assert abs(pos_rate(res.val_idx) - pos_rate(res.train_idx)) < 0.02


def test_several_classification_targets_with_missing_labels_keep_stratification(caplog):
    rng = np.random.default_rng(2)
    n = 3000
    a = (rng.random(n) < 0.1).astype(float)
    b = (rng.random(n) < 0.3).astype(float)
    a[rng.random(n) < 0.3] = np.nan
    with caplog.at_level(logging.WARNING):
        _split({_Binary(): {"a": a, "b": b}}, n)
    assert not any("stratification disabled" in r.getMessage() for r in caplog.records)
