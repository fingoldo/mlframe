"""Wave 10e monolith-split sensor for ``mlframe.training.helpers``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    """Parent module."""
    from mlframe.training import helpers

    return helpers


@pytest.fixture(scope="module")
def sibling():
    """Sibling."""
    from mlframe.training import _helpers_training_configs

    return _helpers_training_configs


def test_get_training_configs_identity(parent_module, sibling):
    """Get training configs identity."""
    assert parent_module.get_training_configs is sibling.get_training_configs


def test_facade_loc_budget(parent_module):
    """Facade loc budget."""
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 400, f"helpers.py facade is {n_lines} LOC, expected <= 400"


def test_smoke_compute_cb_text_processing(parent_module):
    # Small training set -> non-None scaled config returned.
    """Smoke compute cb text processing."""
    cfg = parent_module.compute_cb_text_processing(n_train_rows=200)
    assert cfg is not None
    assert "dictionaries" in cfg
    # Production-sized training set -> CB's own defaults, returns None.
    assert parent_module.compute_cb_text_processing(n_train_rows=50_000) is None


def test_smoke_parse_catboost_devices_explicit():
    """Smoke parse catboost devices explicit."""
    from mlframe.training.helpers import parse_catboost_devices

    fake_gpus = [{"index": 0, "name": "g0"}, {"index": 1, "name": "g1"}, {"index": 2, "name": "g2"}]
    out = parse_catboost_devices("0:2", all_gpus=fake_gpus)
    assert [g["index"] for g in out] == [0, 2]


def test_get_training_configs_runs_for_multiclass(parent_module):
    """iter361 regression: the carve into _helpers_training_configs.py
    dropped the ``from ._classif_helpers import _classif_objective_kwargs``
    import. Binary / regression / LTR combos sidestepped the bug because the
    ``if _resolved_tt.is_classification and not _resolved_tt.is_binary:``
    block is skipped; any multiclass / multilabel combo crashed at fit time
    with ``NameError: name '_classif_objective_kwargs' is not defined``.
    Reproduced on fuzz combo c0023 (multiclass cb+linear+xgb)."""
    from mlframe.training.configs import TargetTypes

    cfg = parent_module.get_training_configs(
        iterations=10,
        early_stopping_rounds=0,
        target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        n_classes=3,
    )
    assert cfg is not None
    # The classification branch must have stamped a multiclass objective.
    xgb_params = cfg.XGB_GENERAL_CLASSIF
    assert xgb_params.get("objective") == "multi:softprob"
    assert xgb_params.get("num_class") == 3
