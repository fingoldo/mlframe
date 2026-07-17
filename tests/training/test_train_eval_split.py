"""Wave 10e monolith-split sensor for ``mlframe.training.train_eval``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training import train_eval

    return train_eval


@pytest.fixture(scope="module")
def sibling():
    from mlframe.training.targets import _train_eval_select_target

    return _train_eval_select_target


def test_select_target_identity(parent_module, sibling):
    assert parent_module.select_target is sibling.select_target


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 700, f"train_eval.py facade is {n_lines} LOC, expected <= 700"


def test_smoke_select_target_callable(parent_module):
    import inspect

    sig = inspect.signature(parent_module.select_target)
    # Required first 4 positional params per public signature contract.
    required = ["model_name", "target", "target_type", "df"]
    params = list(sig.parameters)
    for name in required:
        assert name in params, f"select_target lost parameter {name} after carve"


def test_n_classes_from_target_in_sibling(sibling):
    import numpy as np
    from mlframe.training.configs import TargetTypes

    # Binary -> None (returns None for binary in current implementation)
    out = sibling._n_classes_from_target(np.array([0, 1, 0, 1]), TargetTypes.BINARY_CLASSIFICATION)
    assert out is None
    # Multiclass -> count of unique values
    out = sibling._n_classes_from_target(np.array([0, 1, 2, 3, 4]), TargetTypes.MULTICLASS_CLASSIFICATION)
    assert out == 5
