"""Regression: per-fold / per-permutation dataset builds inside feature-selection loops must not log at INFO.

An RFECV fit on a CatBoost classifier logged one INFO ``[dataset-build] catboost.Pool`` line per fold train/val Pool,
per fold scoring predict, and per permutation-importance scorer call -- thousands of lines per fit. Those builds are
now DEBUG unless slow; the per-owner rollup still counts every one of them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training import _dataset_build_stats as dbs

_BUILD_LOGGER = "mlframe.training.trainer"


@pytest.fixture(autouse=True)
def _patched():
    pytest.importorskip("catboost")
    from mlframe.training._model_factories import apply_third_party_patches_once

    apply_third_party_patches_once()
    dbs.reset_dataset_build_stats()
    yield
    dbs.reset_dataset_build_stats()


def _build_records(caplog, min_level: int) -> list:
    return [r for r in caplog.records if r.name == _BUILD_LOGGER and "[dataset-build]" in r.getMessage() and r.levelno >= min_level]


def _pool_from_module(module_name: str):
    from catboost import Pool

    rng = np.random.default_rng(0)
    ns = {"__name__": module_name, "Pool": Pool, "X": rng.random((60, 3)), "y": rng.integers(0, 2, size=60)}
    src = "def _loop_build():\n    return Pool(data=X, label=y)\n_loop_build()"
    exec(compile(src, "<fs_loop>", "exec"), ns)  # nosec B102 -- literal test string, not untrusted input


def test_feature_selection_pool_build_demoted_to_debug(caplog):
    """A routine Pool build fired from a feature-selection wrapper module logs at DEBUG, not INFO."""
    with caplog.at_level(logging.DEBUG, logger=_BUILD_LOGGER):
        _pool_from_module("mlframe.feature_selection.wrappers.rfecv._fit_fold")
    assert _build_records(caplog, logging.INFO) == []
    assert len(_build_records(caplog, logging.DEBUG)) == 1


def test_slow_internal_loop_build_stays_info(caplog, monkeypatch):
    """A build slower than the threshold keeps its INFO line even inside an internal loop."""
    monkeypatch.setattr(dbs, "SLOW_INTERNAL_BUILD_SECONDS", 0.0)
    with caplog.at_level(logging.INFO, logger=_BUILD_LOGGER):
        _pool_from_module("mlframe.feature_selection.wrappers._helpers_importance")
    assert len(_build_records(caplog, logging.INFO)) == 1


def test_rfecv_catboost_fit_emits_no_info_build_lines_but_records_rollup(caplog):
    """End to end: an RFECV fit with CatBoost and permutation importance logs no INFO build line, yet every build is in the rollup."""
    from catboost import CatBoostClassifier

    from mlframe.feature_selection.wrappers import RFECV

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + X["f1"] > 0).astype(int).to_numpy()
    sel = RFECV(
        estimator=CatBoostClassifier(iterations=15, verbose=0, random_seed=0, thread_count=1),
        cv=2,
        importance_getter="permutation",
        max_refits=3,
    )
    with caplog.at_level(logging.INFO, logger=_BUILD_LOGGER):
        sel.fit(X, y)
    assert _build_records(caplog, logging.INFO) == []
    rollup = dbs.dataset_build_snapshot()
    assert sum(r["count"] for r in rollup) >= 6, rollup
    assert any("mlframe.feature_selection" in r["owner"] for r in rollup), rollup
