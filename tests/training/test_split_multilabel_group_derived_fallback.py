"""Regression tests for TC15: multilabel stratify_y + groups without
iterative-stratification must NOT silently drop all stratification.

Pre-fix: ``make_train_test_split`` set ``stratify_y = None`` and logged an
INFO, silently degrading to GroupShuffleSplit (every label's balance lost).
Post-fix: it derives a 1-D composite label-combination id and routes through
sklearn ``StratifiedGroupKFold`` when feasible, emitting a WARNING; only when
the derived label is unusable does it fall back to GroupShuffleSplit, and then
with a loud WARNING that proportions are not preserved.
"""
from __future__ import annotations

import builtins
import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.splitting import make_train_test_split


def _block_iterstrat(monkeypatch):
    """Force the ``iterstrat`` import to fail so the fallback path runs even
    when iterative-stratification happens to be installed in the env."""
    _real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("iterstrat"):
            raise ImportError("blocked for test")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def _make_multilabel_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    # 60 groups, 5 rows each -> plenty of groups per class for StratifiedGroupKFold.
    groups = np.repeat(np.arange(n // 5), 5)
    # 2 labels; build a clear joint structure so derived composite has 2-4 classes
    # each spread over many groups.
    lab0 = (np.arange(n) % 2).astype(int)
    lab1 = ((np.arange(n) // 2) % 2).astype(int)
    y = np.stack([lab0, lab1], axis=1)
    return df, y, groups


def test_tc15_warns_and_uses_derived_stratification(monkeypatch, caplog):
    """When iterstrat is absent, the derived-label StratifiedGroupKFold path
    runs and a WARNING (not silent INFO) names the lost per-label guarantee."""
    _block_iterstrat(monkeypatch)
    df, y, groups = _make_multilabel_data()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df, test_size=0.2, val_size=0.2,
            stratify_y=y, groups=groups, random_seed=7,
        )
    # The fallback emitted a WARNING (pre-fix this was INFO-only -> caplog empty).
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("derived" in m.lower() and "composite" in m.lower() for m in msgs), msgs

    # Groups stay whole across splits (core group-aware invariant preserved).
    g = np.asarray(groups)
    train_g = set(g[train_idx].tolist())
    val_g = set(g[val_idx].tolist())
    test_g = set(g[test_idx].tolist())
    assert not (train_g & val_g)
    assert not (train_g & test_g)
    assert not (val_g & test_g)
    # Non-degenerate splits.
    assert len(val_idx) > 0 and len(test_idx) > 0


def test_tc15_unusable_derived_falls_back_with_loud_warning(monkeypatch, caplog):
    """When the derived composite is unusable (a class on a single group),
    fall back to GroupShuffleSplit but WARN loudly that proportions are lost."""
    _block_iterstrat(monkeypatch)
    n = 40
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"f0": rng.normal(size=n)})
    # Each group is its own singleton-ish class: a rare label combination
    # confined to a single group -> min_groups_per_class == 1 -> unusable.
    groups = np.arange(n)  # one row per group
    y = np.stack([(np.arange(n) % 2), (np.arange(n) // (n - 1))], axis=1)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.splitting"):
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df, test_size=0.2, val_size=0.2,
            stratify_y=y, groups=groups, random_seed=3,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("NOT PRESERVED" in m or "not preserved" in m.lower() for m in msgs), msgs
    # Still a valid group-disjoint split.
    g = np.asarray(groups)
    assert not (set(g[train_idx]) & set(g[test_idx]))
