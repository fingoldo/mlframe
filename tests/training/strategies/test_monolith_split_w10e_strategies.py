"""Identity + facade sensors for the ``mlframe.training.strategies`` subpackage.

Verifies the package ``__init__`` re-exports point at the SAME class / function
objects the submodules own (so ``isinstance(x, strategies.HGBStrategy)`` keeps
working for objects built via the submodule import path), the facade stays under
its LOC budget, and basic constructors return the expected types.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training import strategies
    return strategies


@pytest.fixture(scope="module")
def siblings():
    from mlframe.training.strategies import (
        pipeline_cache,
        tree_cb,
        hgb,
        neural,
    )
    return {
        "pipeline_cache": pipeline_cache,
        "tree_cb": tree_cb,
        "hgb": hgb,
        "neural": neural,
    }


def test_pipeline_cache_identity(parent_module, siblings):
    assert parent_module.PipelineCache is siblings["pipeline_cache"].PipelineCache


def test_tree_strategies_identity(parent_module, siblings):
    tcb = siblings["tree_cb"]
    assert parent_module.TreeModelStrategy is tcb.TreeModelStrategy
    assert parent_module.CatBoostStrategy is tcb.CatBoostStrategy


def test_hgb_identity(parent_module, siblings):
    assert parent_module.HGBStrategy is siblings["hgb"].HGBStrategy


def test_neural_linear_recurrent_identity(parent_module, siblings):
    neu = siblings["neural"]
    assert parent_module.NeuralNetStrategy is neu.NeuralNetStrategy
    assert parent_module.LinearModelStrategy is neu.LinearModelStrategy
    assert parent_module.RecurrentModelStrategy is neu.RecurrentModelStrategy


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 600, f"strategies/__init__.py facade is {n_lines} LOC, expected <= 600"


def test_smoke_get_strategy_returns_catboost(parent_module):
    s = parent_module.get_strategy("cb")
    assert isinstance(s, parent_module.CatBoostStrategy)


def test_smoke_pipeline_cache_construct_and_set(parent_module):
    cache = parent_module.PipelineCache(verbose=False, bytes_limit=10_000_000)
    cache.set("key1", None, None, None)
    assert cache.has("key1")
    got = cache.get("key1")
    assert got == (None, None, None)


def test_smoke_isinstance_cross_module(parent_module, siblings):
    sibling_inst = siblings["hgb"].HGBStrategy()
    assert isinstance(sibling_inst, parent_module.HGBStrategy)
