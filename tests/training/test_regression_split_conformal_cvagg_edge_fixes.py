"""Regression tests for degenerate-input fixes in training split/conformal/cv-agg.

Covers:
- _stratified_split: singleton class falls back to non-stratified with a warning (EDGE6).
- carve_calib_conformal_temporal: time_values length mismatch raises (EDGE-P2).
- carve_calib_conformal_grouped: non-zero frac flooring to 0 groups raises (EDGE-P2).
- select_from_pareto: empty per-iteration shard is skipped, no np.quantile([]) (EDGE-P2).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training._split_helpers import _stratified_split
from mlframe.training._conformal_split import (
    carve_calib_conformal_temporal,
    carve_calib_conformal_grouped,
)
from mlframe.training._cv_aggregation import select_from_pareto


def test_stratified_split_singleton_class_falls_back(caplog):
    # Pre-fix: sklearn raised opaque "least populated class has only 1 member".
    indices = np.arange(20)
    y = np.array([0] * 19 + [1])  # class 1 is a singleton
    with caplog.at_level(logging.WARNING):
        left, right = _stratified_split(indices, test_size=0.3, stratify_y=y, random_state=0)
    assert len(left) + len(right) == 20
    assert "only 1 member" in caplog.text


def test_stratified_split_normal_class_still_stratifies():
    indices = np.arange(40)
    y = np.array([0] * 20 + [1] * 20)
    left, right = _stratified_split(indices, test_size=0.25, stratify_y=y, random_state=0)
    assert len(right) == 10


def test_carve_temporal_time_values_length_mismatch_raises():
    # Pre-fix: no length check (grouped carver had one); mismatch silently misordered rows.
    train_idx = np.arange(10)
    bad_time = np.arange(5)
    with pytest.raises(ValueError, match="must align with train_idx"):
        carve_calib_conformal_temporal(train_idx, calib_frac=0.2, conformal_frac=0.2, time_values=bad_time)


def test_carve_grouped_nonzero_frac_floors_to_zero_groups_raises():
    # Pre-fix: conformal_frac flooring to 0 groups silently produced an empty conformal slice.
    train_idx = np.arange(6)
    groups = np.array([0, 0, 0, 1, 1, 1])  # only 2 groups
    with pytest.raises(ValueError, match="floors to 0 conformal groups"):
        carve_calib_conformal_grouped(train_idx, calib_frac=0.4, conformal_frac=0.1, group_values=groups)


def test_select_from_pareto_skips_empty_shard():
    # Pre-fix: np.quantile([]) on an empty shard warned and returned NaN, poisoning selection.
    frontier = [0, 1]
    iter_means = [0.5, 0.4]
    iter_stds = [0.1, 0.1]
    iter_shard_scores = [[], [0.4, 0.45, 0.5]]  # iter 0 has no shards
    best = select_from_pareto(frontier, iter_means, iter_stds, iter_shard_scores, risk_quantile=0.9, direction="min")
    assert best == 1
