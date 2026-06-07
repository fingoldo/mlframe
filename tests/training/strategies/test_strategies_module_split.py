"""Package-facade sensors for ``mlframe.training.strategies``.

The strategies subpackage re-exports every strategy class + ``get_strategy`` +
``PipelineCache`` from its ``__init__`` so historical
``from mlframe.training.strategies import X`` import paths keep working.
"""
from __future__ import annotations

from pathlib import Path


def test_all_strategy_classes_still_importable_from_facade() -> None:
    from mlframe.training.strategies import (
        ModelPipelineStrategy,
        TreeModelStrategy,
        CatBoostStrategy,
        XGBoostStrategy,
        HGBStrategy,
        NeuralNetStrategy,
        LinearModelStrategy,
        RecurrentModelStrategy,
    )
    from abc import ABC
    assert issubclass(ModelPipelineStrategy, ABC)
    for cls in (
        TreeModelStrategy,
        CatBoostStrategy,
        XGBoostStrategy,
        HGBStrategy,
        NeuralNetStrategy,
        LinearModelStrategy,
        RecurrentModelStrategy,
    ):
        assert issubclass(cls, ModelPipelineStrategy), cls


def test_factory_and_cache_still_importable() -> None:
    from mlframe.training.strategies import get_strategy, PipelineCache
    assert callable(get_strategy)
    assert PipelineCache is not None


def test_facade_below_1k_line_threshold() -> None:
    pkg = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "training" / "strategies"
    facade = pkg / "__init__.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"strategies/__init__.py is {n} lines, still over the 1k threshold"


def test_siblings_own_their_moved_classes() -> None:
    """Identity: facade and submodule expose the SAME class objects."""
    from mlframe.training import strategies
    from mlframe.training.strategies import base, xgboost
    assert strategies.ModelPipelineStrategy is base.ModelPipelineStrategy
    assert strategies.XGBoostStrategy is xgboost.XGBoostStrategy


def test_xgb_strategy_is_treemodel_subclass() -> None:
    """XGBoostStrategy -> TreeModelStrategy -> ModelPipelineStrategy chain stays correct."""
    from mlframe.training.strategies import XGBoostStrategy, TreeModelStrategy, ModelPipelineStrategy
    assert issubclass(XGBoostStrategy, TreeModelStrategy)
    assert issubclass(XGBoostStrategy, ModelPipelineStrategy)
