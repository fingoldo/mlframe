"""Regression: get_training_configs in-scope gates (CB-GPU probe, MLP/recurrent config build) must recognise estimator INSTANCES
and the full alias set, not just literal "cb"/"mlp"/"recurrent" strings.

Pre-fix bugs in ``_helpers_training_configs.get_training_configs``:
  * ``_cb_in_scope`` used ``str(m).lower() in ("cb","catboost")`` -- a CatBoost passed as an instance stringifies to
    "catboostclassifier()" and was mis-classed out of scope, silently forcing task_type="CPU" even with a GPU present.
  * ``_mlp_in_scope`` matched only literal ``"mlp"``/``"recurrent"``. The actual recurrent aliases (lstm/gru/rnn/transformer)
    were never matched, so a recurrent-only suite skipped the MLP/pytorch config path and got MLP_GENERAL_PARAMS=None.
"""

import pytest

from mlframe.training._helpers_training_configs import get_training_configs


@pytest.mark.parametrize("recurrent_alias", ["lstm", "gru", "rnn", "transformer"])
def test_recurrent_alias_builds_mlp_config(recurrent_alias):
    """Recurrent alias builds mlp config."""
    cfg = get_training_configs(has_gpu=False, enabled_models=[recurrent_alias])
    assert (
        cfg.MLP_GENERAL_PARAMS is not None
    ), f"recurrent alias {recurrent_alias!r} needs the torch MLP config path (was None pre-fix -- literal 'recurrent' never matched)"


def test_mlp_alias_builds_mlp_config_and_ngb_does_not():
    """Mlp alias builds mlp config and ngb does not."""
    assert get_training_configs(has_gpu=False, enabled_models=["mlp"]).MLP_GENERAL_PARAMS is not None
    # ngb shares NeuralNetStrategy but has its own NGB path (no torch) -- must NOT pull in the heavy MLP config.
    assert get_training_configs(has_gpu=False, enabled_models=["ngb"]).MLP_GENERAL_PARAMS is None


def test_tree_only_suite_skips_mlp_config():
    """Tree only suite skips mlp config."""
    assert get_training_configs(has_gpu=False, enabled_models=["cb", "lgb"]).MLP_GENERAL_PARAMS is None


def test_cb_instance_keeps_cb_config_in_scope():
    """Cb instance keeps cb config in scope."""
    cb = pytest.importorskip("catboost")
    # has_gpu=False so no real GPU probe runs; the point is the CB config block is built for an instance-passed CB.
    cfg = get_training_configs(has_gpu=False, enabled_models=[cb.CatBoostClassifier()])
    assert cfg.CB_GENERAL_PARAMS is not None
