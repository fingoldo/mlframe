"""Regression: tiny-rerank must follow the ACTUAL group-aware splitter, not only the analyzer recommendation (audit I11).

Pre-fix, ``run_composite_target_discovery`` activated the group-aware tiny-CV rerank ONLY when the target-distribution analyzer
stamped ``knob_overrides.split_config.prefer_group_aware=True`` (a HINT). A user who explicitly configured a group-aware split
(``split_config.use_groups=True`` + supplied ``group_ids``) WITHOUT the analyzer recommending it got a plain-KFold rerank, which
promotes per-group memorisers whose trained models then fail the production group-aware test (the documented prod failure).

The fix threads the real ``split_config`` into the phase and gates the rerank on EITHER the recommendation OR the actual splitter
config. These tests assert that ``CompositeTargetDiscovery._group_ids_for_rerank`` is set (group-aware rerank active) when the
splitter is group-aware-by-config even with no recommendation, and stays unset when neither signal is present.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes
from mlframe.training.core._phase_composite_discovery import (
    run_composite_target_discovery,
)


class _FitStub:
    """Minimal stand-in for a fitted ``CompositeTargetDiscovery`` so the phase's post-fit bookkeeping runs without a real fit."""

    specs_: list = []

    def export_specs(self):
        """Export specs."""
        return []

    def report(self):
        """Report."""
        return []

    def filter_drops(self):
        """Filter drops."""
        return {}


def _run_capture_rerank_groups(*, split_config, group_ids, metadata_extra=None):
    """Drive the real phase with ``fit`` patched to record ``self._group_ids_for_rerank`` at call time; return that captured value."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 120
    feature_df = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    target_y = rng.standard_normal(n)
    target_by_type = {TargetTypes.REGRESSION: {"y": target_y}}
    metadata: dict = {}
    if metadata_extra:
        metadata.update(metadata_extra)

    cfg = CompositeTargetDiscoveryConfig(enabled=True)

    captured = {}

    def _spy_fit(self, *a, **kw):
        """Spy fit."""
        captured["rerank_groups"] = getattr(self, "_group_ids_for_rerank", None)
        return _FitStub()

    from unittest.mock import patch

    _SENTINEL = object()
    captured["rerank_groups"] = _SENTINEL

    with patch.object(CompositeTargetDiscovery, "fit", _spy_fit):
        run_composite_target_discovery(
            composite_target_discovery_config=cfg,
            target_by_type=target_by_type,
            mlframe_models=["cb"],
            metadata=metadata,
            filtered_train_df=feature_df,
            filtered_train_idx=np.arange(n),
            train_df_pd=feature_df,
            val_df_pd=feature_df.iloc[: n // 5],
            test_df_pd=feature_df.iloc[: n // 5],
            train_idx=np.arange(n),
            val_idx=np.arange(n // 5),
            test_idx=np.arange(n // 5),
            baseline_diagnostics_config=None,
            cat_features=[],
            verbose=False,
            group_ids=group_ids,
            split_config=split_config,
        )
    val = captured["rerank_groups"]
    if val is _SENTINEL:
        raise AssertionError("fit was not invoked; test setup is wrong")
    return val


def test_group_aware_splitter_config_activates_rerank_without_recommendation():
    """``split_config.use_groups=True`` + group_ids, NO analyzer recommendation -> group-aware rerank MUST activate (I11 fix).

    Pre-fix the phase had no ``split_config`` parameter and gated solely on ``prefer_group_aware`` (absent here), so
    ``_group_ids_for_rerank`` was never set and the rerank silently used plain KFold on a group-aware production split.
    """
    n = 120
    groups = np.repeat(np.arange(6), n // 6)
    split_config = SimpleNamespace(use_groups=True)

    rerank_groups = _run_capture_rerank_groups(
        split_config=split_config,
        group_ids=groups,
    )
    assert rerank_groups is not None, (
        "group-aware splitter (use_groups=True + group_ids) did not activate the "
        "tiny-rerank GroupKFold path; _group_ids_for_rerank stayed unset (I11 regression)"
    )
    np.testing.assert_array_equal(np.asarray(rerank_groups), groups)


def test_group_aware_splitter_accepts_dict_split_config():
    """A raw-dict split_config (caller bypassing setup_configuration) is honoured, not silently treated as group-naive."""
    n = 120
    groups = np.repeat(np.arange(6), n // 6)

    rerank_groups = _run_capture_rerank_groups(
        split_config={"use_groups": True},
        group_ids=groups,
    )
    assert rerank_groups is not None, "dict split_config with use_groups=True was not honoured for the rerank gate"
    np.testing.assert_array_equal(np.asarray(rerank_groups), groups)


def test_no_group_signal_keeps_rerank_group_naive():
    """No recommendation AND no group-aware splitter -> rerank stays plain KFold (_group_ids_for_rerank unset)."""
    n = 120
    groups = np.repeat(np.arange(6), n // 6)

    rerank_groups = _run_capture_rerank_groups(
        split_config=SimpleNamespace(use_groups=False),
        group_ids=groups,
    )
    assert rerank_groups is None, "group-aware rerank fired with use_groups=False and no analyzer recommendation"


def test_recommendation_still_activates_rerank_without_split_config():
    """The legacy analyzer-recommendation path (prefer_group_aware=True) still activates the rerank even when split_config is None."""
    n = 120
    groups = np.repeat(np.arange(6), n // 6)
    metadata_extra = {
        "target_distribution_report": {
            "knob_overrides": {"split_config": {"prefer_group_aware": True}},
        },
    }

    rerank_groups = _run_capture_rerank_groups(
        split_config=None,
        group_ids=groups,
        metadata_extra=metadata_extra,
    )
    assert rerank_groups is not None, (
        "analyzer recommendation prefer_group_aware=True no longer activates the rerank when split_config is None (regression in the OR gate)"
    )
    np.testing.assert_array_equal(np.asarray(rerank_groups), groups)
