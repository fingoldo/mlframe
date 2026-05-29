"""Unit tests for slice-stable ES integration in ``UniversalCallback``.

The booster-specific subclasses (LightGBMCallback / XGBoostCallback / CatBoostCallback) only
add the boilerplate that converts the booster-native eval payload into ``metric_history``;
all the slice-stable decision logic lives in the base class and is what we exercise here.

Covers:
  - patience auto-bump via ``effective_patience``
  - aggregator wiring (mean / t_lcb / quantile)
  - skip-iteration when at least one shard hasn't pushed a fresh value (CB metric_period>1)
  - improvement / no-improvement / patience-trigger flow on a constructed history
  - default ``slice_k=0`` is bit-identical to the legacy single-val path
  - ``disable_native_es_for_slice_stable`` strips ES knobs from per-booster param dicts
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from mlframe.training._callbacks import UniversalCallback
from mlframe.training._helpers_training_configs import disable_native_es_for_slice_stable


def _build_cb(*, patience=3, mode="min", slice_k=3, agg_mode="mean", **extra):
    """Construct a callback ready to receive metric history without booster boilerplate."""
    cb = UniversalCallback(
        patience=patience, min_delta=0.0, monitor_dataset="valid_0",
        monitor_metric="loss", mode=mode,
        slice_k=slice_k, slice_aggregate_mode=agg_mode, slice_aggregate_confidence=0.9,
        slice_persist_history=True, verbose=0, **extra,
    )
    cb.start_time = time.time()
    cb.last_reporting_ts = time.time()
    cb.iter = 0
    return cb


def _push_iter(cb: UniversalCallback, values: dict[str, float]) -> None:
    """Append one value per dataset to the metric history (mirrors what booster callbacks do)."""
    for name, v in values.items():
        cb.metric_history.setdefault(name, {}).setdefault("loss", []).append(float(v))


def test_default_no_slice_path_bit_identical() -> None:
    """slice_k=0 (default) must produce the same best_iter / stop decisions as legacy."""
    cb = UniversalCallback(
        patience=2, min_delta=0.0, monitor_dataset="valid_0",
        monitor_metric="loss", mode="min", slice_k=0, verbose=0,
    )
    cb.start_time = time.time(); cb.last_reporting_ts = time.time(); cb.iter = 0
    # Legacy history: monotone improvement
    for v in [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]:
        _push_iter(cb, {"valid_0": v})
        cb.should_stop()
    assert cb.best_metric == pytest.approx(0.85)
    assert cb.best_iter == 2


def test_patience_auto_bumped_when_slice_active() -> None:
    cb = _build_cb(patience=10, slice_k=5)
    # 1 + 1/sqrt(4) = 1.5 -> ceil(10*1.5) = 15
    assert cb.patience == 15


def test_patience_not_bumped_when_slice_inactive() -> None:
    cb = _build_cb(patience=10, slice_k=0)
    assert cb.patience == 10


def test_slice_aggregate_mean_drives_decision() -> None:
    """With aggregator='mean' over 3 shards, the aggregate is the unweighted average.

    Note: patience is auto-bumped by ``effective_patience(K=3) -> ceil(p * 1.707)``, so
    ``patience=1`` becomes effective 2: stop fires AFTER 2 consecutive no-improvement iters.
    """
    cb = _build_cb(patience=1, mode="min", slice_k=3, agg_mode="mean")
    assert cb.patience == 2, "patience=1 with K=3 must auto-bump to 2"
    # iter 1: all shards at 1.0 -> aggregate 1.0 (init best_metric)
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 1.0, "valid_2": 1.0, "valid_3": 1.0})
    assert cb.should_stop() is False
    assert cb.best_metric == pytest.approx(1.0)
    # iter 2: aggregate 0.9 -> improvement; best_metric = 0.9
    _push_iter(cb, {"valid_0": 0.9, "valid_1": 0.9, "valid_2": 0.9, "valid_3": 0.9})
    assert cb.should_stop() is False
    assert cb.best_metric == pytest.approx(0.9)
    # iter 3: aggregate 1.0 (worse) -> since_imp=1 (< effective patience 2)
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 1.0, "valid_2": 1.0, "valid_3": 1.0})
    assert cb.should_stop() is False
    # iter 4: aggregate 1.0 again -> since_imp=2 (== effective patience) -> stop
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 1.0, "valid_2": 1.0, "valid_3": 1.0})
    assert cb.should_stop() is True


def test_slice_aggregate_t_lcb_pushes_penalty_up_for_min() -> None:
    """With t-LCB on min-direction, the per-iteration aggregated score sits ABOVE the raw mean."""
    cb = _build_cb(patience=5, mode="min", slice_k=3, agg_mode="t_lcb")
    # iter 1: heterogeneous shards -> aggregate (mean + penalty) > mean
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 0.5, "valid_2": 1.0, "valid_3": 1.5})
    cb.should_stop()
    raw_mean = (0.5 + 1.0 + 1.5) / 3
    # best_metric was set to the t-LCB aggregate on iter 1
    assert cb.best_metric > raw_mean, (
        f"t-LCB for direction='min' must add penalty; got best={cb.best_metric} vs mean={raw_mean}"
    )


def test_slice_skip_when_shard_missing_value() -> None:
    """If a shard's history is shorter than monitor_dataset's, the aggregator returns None and
    the callback skips the decision (legacy single-val fallback would mix scales — see comment
    in ``should_stop``)."""
    cb = _build_cb(patience=2, mode="min", slice_k=3)
    # monitor has 2 values, one shard has only 1 -> aggregator returns None -> should_stop False
    cb.metric_history = {
        "valid_0": {"loss": [1.0, 0.9]},
        "valid_1": {"loss": [1.0, 0.9]},
        "valid_2": {"loss": [1.0]},  # phantom CB metric_period > 1
        "valid_3": {"loss": [1.0, 0.9]},
    }
    assert cb.should_stop() is False
    # best_metric should NOT have been initialized from the single-val number; remains None.
    assert cb.best_metric is None


def test_slice_quantile_aggregator() -> None:
    """Quantile aggregator on min-direction reads the upper tail of shard scores."""
    cb = _build_cb(patience=3, mode="min", slice_k=4, agg_mode="quantile",
                    slice_aggregate_quantile_level=0.75)
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 0.5, "valid_2": 0.7, "valid_3": 0.9})
    cb.should_stop()
    # Upper-quartile of [0.5, 0.7, 0.9] = 0.8 (linear interp)
    assert cb.best_metric == pytest.approx(0.8, rel=1e-9)


def test_slice_history_persisted_when_flag_set() -> None:
    cb = _build_cb(patience=3, slice_k=3)
    for offset in (0.0, -0.1, +0.1):
        _push_iter(cb, {"valid_0": 1.0 + offset, "valid_1": 1.0 + offset,
                         "valid_2": 1.0 + offset, "valid_3": 1.0 + offset})
        cb.should_stop()
    # 3 iterations -> 3 shard-score snapshots
    assert len(cb.slice_shard_score_history) == 3
    assert all(len(s) == 3 for s in cb.slice_shard_score_history)
    assert len(cb.slice_aggregate_history) == 3


def test_slice_dataset_names_override() -> None:
    """Caller may pass explicit dataset names instead of relying on insertion order."""
    cb = UniversalCallback(
        patience=2, min_delta=0.0, monitor_dataset="valid_0",
        monitor_metric="loss", mode="min",
        slice_k=2, slice_aggregate_mode="mean",
        slice_dataset_names=["shard_alpha", "shard_beta"], verbose=0,
    )
    cb.start_time = time.time(); cb.last_reporting_ts = time.time(); cb.iter = 0
    cb.metric_history = {
        "valid_0": {"loss": [1.0]},
        "shard_alpha": {"loss": [0.8]},
        "shard_beta": {"loss": [1.2]},
        "ignore_me": {"loss": [0.0]},  # should NOT be used
    }
    cb.should_stop()
    # mean over the named shards: (0.8 + 1.2) / 2 = 1.0
    assert cb.best_metric == pytest.approx(1.0)


def test_slice_diagnostic_only_skips_patience_bump_and_aggregator_drive() -> None:
    """In ``slice_diagnostic_only=True`` mode the callback logs per-shard history but ES
    decision reads single-val (legacy path). patience stays unbumped; the aggregate isn't
    consulted for stop logic.
    """
    cb = UniversalCallback(
        patience=10, min_delta=0.0, monitor_dataset="valid_0",
        monitor_metric="loss", mode="min",
        slice_k=5, slice_aggregate_mode="mean",
        slice_persist_history=True, slice_diagnostic_only=True, verbose=0,
    )
    assert cb.patience == 10, "patience must NOT be auto-bumped when diagnostic_only=True"
    cb.start_time = time.time(); cb.last_reporting_ts = time.time(); cb.iter = 0
    # iter 1: full val 1.0, shards arbitrary (worse). Decision must read full val (1.0), not aggregate.
    _push_iter(cb, {"valid_0": 1.0, "valid_1": 0.5, "valid_2": 1.5, "valid_3": 1.5,
                     "valid_4": 1.5, "valid_5": 1.5})
    assert cb.should_stop() is False
    assert cb.best_metric == pytest.approx(1.0), "diagnostic mode drives ES off the full val"
    # Per-shard history populated (so the Pareto artefact has data downstream).
    assert len(cb.slice_shard_score_history) == 1
    assert len(cb.slice_shard_score_history[0]) == 5


def test_disable_native_es_for_slice_stable_strips_keys() -> None:
    configs = SimpleNamespace(
        CB_GENERAL_PARAMS={"early_stopping_rounds": 50, "iterations": 1000},
        CB_REGR={"early_stopping_rounds": 50},
        CB_CLASSIF=None,
        CB_CALIB_CLASSIF=None,
        LGB_GENERAL_PARAMS={"early_stopping_rounds": 50},
        XGB_GENERAL_PARAMS={"early_stopping_rounds": 50},
        XGB_GENERAL_CLASSIF=None,
        XGB_CALIB_CLASSIF=None,
        HGB_GENERAL_PARAMS={"early_stopping": True, "n_iter_no_change": 10, "max_iter": 500},
        MLP_GENERAL_PARAMS=None,
        COMMON_RFECV_PARAMS=None,
        NGB_GENERAL_PARAMS=None,
    )
    disable_native_es_for_slice_stable(configs)
    assert "early_stopping_rounds" not in configs.CB_GENERAL_PARAMS
    assert "early_stopping_rounds" not in configs.CB_REGR
    assert "early_stopping_rounds" not in configs.LGB_GENERAL_PARAMS
    assert "early_stopping_rounds" not in configs.XGB_GENERAL_PARAMS
    assert configs.CB_GENERAL_PARAMS["iterations"] == 1000, "other keys preserved"
    assert configs.HGB_GENERAL_PARAMS["early_stopping"] is False
    assert "n_iter_no_change" not in configs.HGB_GENERAL_PARAMS
    assert configs.HGB_GENERAL_PARAMS["max_iter"] == 500
