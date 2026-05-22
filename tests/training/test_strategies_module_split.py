"""Wave 104 (2026-05-21): split training/strategies.py (1492 lines)
into strategies.py (now 857 lines) + two new siblings:

  - _strategies_base.py (375 lines): ModelPipelineStrategy ABC
  - _strategies_xgboost.py (331 lines): XGBoostStrategy concrete impl

The other 6 concrete strategies (CatBoostStrategy, TreeModelStrategy,
HGBStrategy, NeuralNetStrategy, LinearModelStrategy,
RecurrentModelStrategy) plus get_strategy + PipelineCache stay in the
parent. Re-exports keep every existing import path working.
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
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training"
    facade = root / "strategies.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"strategies.py is {n} lines, still over the 1k threshold"


def test_siblings_own_their_moved_classes() -> None:
    """Identity: facade and sibling expose the SAME class objects."""
    from mlframe.training import strategies, _strategies_base, _strategies_xgboost
    assert strategies.ModelPipelineStrategy is _strategies_base.ModelPipelineStrategy
    assert strategies.XGBoostStrategy is _strategies_xgboost.XGBoostStrategy


def test_xgb_strategy_is_treemodel_subclass() -> None:
    """The XGBoostStrategy class chain stays correct across the split:
    XGBoostStrategy -> TreeModelStrategy -> ModelPipelineStrategy."""
    from mlframe.training.strategies import XGBoostStrategy, TreeModelStrategy, ModelPipelineStrategy
    assert issubclass(XGBoostStrategy, TreeModelStrategy)
    assert issubclass(XGBoostStrategy, ModelPipelineStrategy)
