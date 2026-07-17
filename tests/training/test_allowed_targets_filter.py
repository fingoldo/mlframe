"""Sensor: SimpleFeaturesAndTargetsExtractor must honour the ``allowed_targets``
filter promised by its docstring.

Pre-fix shape (docstring-drift audit P1 #1): the subclass __init__ accepted
``allowed_targets`` but silently swallowed it -- the kwarg was NOT forwarded to
``super().__init__()``, never stored on self, and never read by ``build_targets``.
A caller writing
    SimpleFeaturesAndTargetsExtractor(classification_targets=["a","b","c"],
                                      allowed_targets=["a"])
expecting filtering got all three targets trained.

Post-fix: super().__init__() receives allowed_targets, the base class stores it via
store_params_in_object, and build_targets filters target_by_type to keep only allowed
names + WARNs on names that didn't match any built target (catches typos in the
allowlist).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _build_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x0": rng.normal(size=100),
            "x1": rng.normal(size=100),
            "a": rng.integers(0, 2, size=100),
            "b": rng.integers(0, 2, size=100),
            "c": rng.integers(0, 2, size=100),
        }
    )


def test_allowed_targets_none_keeps_everything():
    """Baseline: no allowlist -> all configured targets are built."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["a", "b", "c"],
    )
    out = fte.build_targets(_build_df())
    # All three should appear (sum across buckets)
    all_names = set()
    for _named in out.values():
        if isinstance(_named, dict):
            all_names.update(_named.keys())
    assert all_names == {"a", "b", "c"}


def test_allowed_targets_filters_to_subset():
    """REGRESSION: pre-fix the allowlist was silently ignored and all 3 were trained."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["a", "b", "c"],
        allowed_targets=["a"],
    )
    out = fte.build_targets(_build_df())
    all_names = set()
    for _named in out.values():
        if isinstance(_named, dict):
            all_names.update(_named.keys())
    assert all_names == {"a"}, f"allowed_targets=['a'] should keep only 'a'; got {all_names}. Pre-fix bug regression."


def test_allowed_targets_typo_warns(caplog):
    """Caller passing a typo in the allowlist gets a WARN naming the typo + the built names."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["a", "b"],
        allowed_targets=["a", "non_existent"],
        verbose=1,
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.extractors"):
        out = fte.build_targets(_build_df())
    # The 'non_existent' name should surface in the WARN
    found = any("non_existent" in (rec.getMessage() if hasattr(rec, "getMessage") else str(rec.message)) for rec in caplog.records)
    assert found, f"expected WARN naming the unmatched 'non_existent' allowlist entry; got records: {[rec.message for rec in caplog.records]}"


def test_allowed_targets_drops_target_type_with_no_match():
    """When the allowlist matches zero names in a target_type bucket, that bucket
    is excluded entirely from target_by_type (instead of an empty dict slot)."""
    df = _build_df()
    df["reg_y"] = np.random.default_rng(0).normal(size=len(df))
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["a", "b"],
        regression_targets=["reg_y"],
        allowed_targets=["a"],  # excludes 'b' AND 'reg_y'
    )
    out = fte.build_targets(df)
    # No bucket should contain 'b' or 'reg_y'.
    for _tt, _named in out.items():
        if isinstance(_named, dict):
            assert "b" not in _named
            assert "reg_y" not in _named


def test_allowed_targets_stored_on_self():
    """super().__init__ now stores self.allowed_targets via store_params_in_object.
    Verify attribute exists for downstream introspection."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["a"],
        allowed_targets=["a"],
    )
    assert hasattr(fte, "allowed_targets")
    assert list(fte.allowed_targets) == ["a"]
