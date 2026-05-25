"""Wave 10e monolith-split sensor for ``mlframe.training.helpers``."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training import helpers
    return helpers


@pytest.fixture(scope="module")
def sibling():
    from mlframe.training import _helpers_training_configs
    return _helpers_training_configs


def test_get_training_configs_identity(parent_module, sibling):
    assert parent_module.get_training_configs is sibling.get_training_configs


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= 400, f"helpers.py facade is {n_lines} LOC, expected <= 400"


def test_smoke_compute_cb_text_processing(parent_module):
    # Small training set -> non-None scaled config returned.
    cfg = parent_module.compute_cb_text_processing(n_train_rows=200)
    assert cfg is not None
    assert "dictionaries" in cfg
    # Production-sized training set -> CB's own defaults, returns None.
    assert parent_module.compute_cb_text_processing(n_train_rows=50_000) is None


def test_smoke_parse_catboost_devices_explicit():
    from mlframe.training.helpers import parse_catboost_devices
    fake_gpus = [{"index": 0, "name": "g0"}, {"index": 1, "name": "g1"}, {"index": 2, "name": "g2"}]
    out = parse_catboost_devices("0:2", all_gpus=fake_gpus)
    assert [g["index"] for g in out] == [0, 2]
