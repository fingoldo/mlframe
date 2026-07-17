"""Regression coverage for TVT-run RAM optimizations (2026-05-30).

User-requested batch: leak-corr in-place mutation (1), PipelineCache budget
clamp (3), auto-drop analyzer candidates (4), auto-drop near-duplicate pairs (5).

Each test asserts the observable behaviour: dtype/shape preservation, drop set
correctness, train-only invariant.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fix 1 - leak-corr in-place mutation: _filter_features doesn't double the peak
# allocation. Black-box check: pre-fix logic still rejects strongly y-correlated
# columns, in-place mutation doesn't corrupt the survivor set.
# ---------------------------------------------------------------------------
def test_1_leak_corr_filter_drops_strong_y_correlate_after_in_place_fix():
    """The leak-corr filter must still drop a column with |corr(x, y)| >= threshold
    after the in-place mutation refactor; survivor set must NOT include the strong
    correlate. Regression check that removing the .copy() didn't break correctness."""
    pytest.importorskip("polars")
    import polars as pl
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 5000
    y = rng.standard_normal(n).astype(np.float32)
    # Strong-corr feature: y itself + tiny noise (|corr|>0.999).
    leak = y + 0.01 * rng.standard_normal(n).astype(np.float32)
    # Weak-corr feature: random.
    weak = rng.standard_normal(n).astype(np.float32)
    df = pl.DataFrame({"leak": leak, "weak": weak, "y": y})

    cfg = CompositeTargetDiscoveryConfig(forbidden_base_corr_threshold=0.95)
    inst = CompositeTargetDiscovery.__new__(CompositeTargetDiscovery)
    inst.config = cfg
    inst._target_col = "y"
    inst._patterns_compiled = []
    inst._filter_drops = []

    kept = inst._filter_features(df, ["leak", "weak"], y, np.arange(n))
    assert "leak" not in kept, "strong-correlate must be filtered out"
    assert "weak" in kept, "weak-correlate must survive"


# ---------------------------------------------------------------------------
# Fix 3 - PipelineCache default 0.15 + 16 GB floor.
# ---------------------------------------------------------------------------
def test_3_pipeline_cache_default_fraction_is_0p15():
    from mlframe.training.strategies.pipeline_cache import _DEFAULT_PIPELINE_CACHE_RAM_FRACTION

    assert _DEFAULT_PIPELINE_CACHE_RAM_FRACTION == 0.15, (
        "default RAM fraction must be 0.15 (was 0.4) to keep room for big-frame transient peaks during pipeline-fit + composite-discovery on 100+ GB processes."
    )


def test_3_pipeline_cache_default_in_behavior_config_is_0p15():
    from mlframe.training._model_configs import TrainingBehaviorConfig

    cfg = TrainingBehaviorConfig()
    assert cfg.pipeline_cache_ram_budget_fraction == 0.15


def test_3_pipeline_cache_budget_clamped_to_available_minus_16gb_floor():
    """When available is small (mock 20 GB), the clamp ensures the cache budget
    never exceeds available - 16 GB floor; the 2 GB minimum prevents the cap from
    collapsing to zero on a fully-saturated host."""
    from mlframe.training.strategies import pipeline_cache as mod

    fake_vm = SimpleNamespace(total=int(137 * 1024**3), available=int(20 * 1024**3))
    with (
        patch.dict(os.environ, {"MLFRAME_PIPELINE_CACHE_BYTES_LIMIT": ""}, clear=False),
        patch.object(mod, "_DEFAULT_PIPELINE_CACHE_RAM_FRACTION", 0.4),
        patch("psutil.virtual_memory", return_value=fake_vm),
    ):
        if "MLFRAME_PIPELINE_CACHE_BYTES_LIMIT" in os.environ:
            del os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"]
        if "MLFRAME_PIPELINE_CACHE_RAM_FRACTION" in os.environ:
            del os.environ["MLFRAME_PIPELINE_CACHE_RAM_FRACTION"]
        budget = mod._resolve_pipeline_cache_budget(fraction=0.4)
    # total * 0.4 = 54.8 GB, but available - 16 = 4 GB; clamp gives 4 GB.
    floor = 16 * 1024**3
    assert budget == max(2 * 1024**3, 20 * 1024**3 - floor), f"budget {budget / 1024**3:.2f} GB should equal min(total*fraction, available - 16 GB)"


# ---------------------------------------------------------------------------
# Fix 4 + 5 - analyzer auto-drop. Train-only derivation, then propagated to
# val and test for schema alignment.
# ---------------------------------------------------------------------------
def test_4_5_auto_drop_helper_strips_drop_candidates_and_near_duplicates():
    pl = pytest.importorskip("polars")
    from mlframe.training.core._main_train_suite_target_distribution import (
        _maybe_auto_drop_after_feature_analyzer,
    )

    train_df = pl.DataFrame(
        {
            "good_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "good_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "nan_heavy": [None, None, None, 1.0, 2.0],  # candidate
            "low_var": [0.0, 0.0, 0.0, 0.0, 0.0],  # candidate
            "dup_of_a": [1.0, 2.0, 3.0, 4.0, 5.0],  # exact duplicate of good_a
        }
    )
    val_df = train_df.head(3)
    test_df = train_df.head(2)
    fd_report = SimpleNamespace(
        drop_candidates=["nan_heavy", "low_var"],
        diagnostics={
            "redundant_feature_pairs": [
                {"a": "good_a", "b": "dup_of_a", "corr": 1.0},
            ],
        },
    )
    behavior_config = SimpleNamespace(
        auto_drop_distribution_analyzer_candidates=True,
        auto_drop_near_duplicate_threshold=0.999,
    )
    metadata: dict = {}
    train_out, val_out, test_out, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=fd_report,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        behavior_config=behavior_config,
        metadata=metadata,
        verbose=False,
    )
    # nan_heavy + low_var dropped (candidates), AND one of (good_a, dup_of_a).
    # The helper picks the alphabetically-larger of the pair => dup_of_a.
    assert set(dropped) == {"nan_heavy", "low_var", "dup_of_a"}, dropped
    # All three splits lose the same columns.
    for df, n_expected in ((train_out, 2), (val_out, 2), (test_out, 2)):
        assert set(df.columns) == {"good_a", "good_b"}, df.columns
    # Metadata records the drop list.
    assert metadata["feature_distribution_report"]["auto_dropped_columns"] == sorted(dropped)


def test_4_5_auto_drop_disabled_when_flags_off():
    pl = pytest.importorskip("polars")
    from mlframe.training.core._main_train_suite_target_distribution import (
        _maybe_auto_drop_after_feature_analyzer,
    )

    train_df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    fd_report = SimpleNamespace(
        drop_candidates=["a"],
        diagnostics={"redundant_feature_pairs": [{"a": "a", "b": "b", "corr": 1.0}]},
    )
    behavior_config = SimpleNamespace(
        auto_drop_distribution_analyzer_candidates=False,
        auto_drop_near_duplicate_threshold=2.0,  # disabled (>1.0)
    )
    train_out, _, _, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=fd_report,
        train_df=train_df,
        val_df=None,
        test_df=None,
        behavior_config=behavior_config,
        metadata={},
        verbose=False,
    )
    assert dropped == [], dropped
    assert set(train_out.columns) == {"a", "b"}


def test_4_5_default_behavior_config_enables_both_drops():
    from mlframe.training._model_configs import TrainingBehaviorConfig

    cfg = TrainingBehaviorConfig()
    assert cfg.auto_drop_distribution_analyzer_candidates is True
    assert cfg.auto_drop_near_duplicate_threshold == 0.999


def test_4_5_train_only_derivation_val_test_stats_never_consulted():
    """Train-only invariant: even if val/test contain a column with WORSE NaN
    fraction or zero variance, the drop set is computed from the TRAIN report
    only. The helper just propagates the train-derived drop list to val/test;
    val/test column stats never enter the decision."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._main_train_suite_target_distribution import (
        _maybe_auto_drop_after_feature_analyzer,
    )

    # Train: every column is healthy.
    train_df = pl.DataFrame({"feat": [1.0, 2.0, 3.0]})
    # Val + test: same column is now all-NaN. This is a DRIFT scenario, not
    # something the auto-drop should react to (val/test must never drive
    # train-time decisions).
    val_df = pl.DataFrame({"feat": [None, None, None]})
    test_df = pl.DataFrame({"feat": [None, None, None]})
    fd_report = SimpleNamespace(
        drop_candidates=[],  # train report says nothing to drop
        diagnostics={"redundant_feature_pairs": []},
    )
    behavior_config = SimpleNamespace(
        auto_drop_distribution_analyzer_candidates=True,
        auto_drop_near_duplicate_threshold=0.999,
    )
    _, _, _, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=fd_report,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        behavior_config=behavior_config,
        metadata={},
        verbose=False,
    )
    # No drops: val/test NaN is irrelevant; the helper trusts the train report.
    assert dropped == []
