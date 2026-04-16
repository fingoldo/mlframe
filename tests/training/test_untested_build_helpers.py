"""Tests for internal _build_* / _initialize_* helpers in mlframe.training.core."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core import (
    _build_common_params_for_target,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _initialize_training_defaults,
)
from mlframe.training.configs import TrainingBehaviorConfig, TargetTypes


# ----- _initialize_training_defaults -----

def test_init_defaults_all_none():
    p, r, m = _initialize_training_defaults(None, None, None)
    assert p == {}
    assert r == []
    assert isinstance(m, dict)
    assert "n_workers" in m


def test_init_defaults_mutable_safety():
    # Calling twice with None must produce independent dicts (no shared mutable default)
    p1, r1, m1 = _initialize_training_defaults(None, None, None)
    p2, r2, m2 = _initialize_training_defaults(None, None, None)
    p1["x"] = 1
    r1.append("rf")
    m1["extra"] = True
    assert "x" not in p2
    assert "rf" not in r2
    assert "extra" not in m2


def test_init_defaults_preserves_user_values():
    user_p = {"a": 1}
    user_r = ["rf", "lgb"]
    user_m = {"n_workers": 99}
    p, r, m = _initialize_training_defaults(user_p, user_r, user_m)
    assert p is user_p
    assert r is user_r
    assert m is user_m


# ----- _build_common_params_for_target -----

def test_build_common_params_no_fairness_no_od():
    bc = TrainingBehaviorConfig()
    init = {"foo": 1, "train_target": [1, 2], "val_target": [3]}
    od_params, cur_bc = _build_common_params_for_target(
        init_common_params=init,
        trainset_features_stats={"m": 1},
        plot_file="/tmp/p",
        train_od_idx=None,
        val_od_idx=None,
        current_train_target=None,
        current_val_target=None,
        outlier_detector=None,
        behavior_config=bc,
        fairness_subgroups=None,
    )
    # train_target/val_target filtered out when no OD
    assert "train_target" not in od_params
    assert "val_target" not in od_params
    assert od_params["foo"] == 1
    assert od_params["trainset_features_stats"] == {"m": 1}
    assert od_params["plot_file"] == "/tmp/p"
    assert cur_bc is bc  # no copy when fairness is None


def test_build_common_params_with_fairness():
    bc = TrainingBehaviorConfig()
    fairness = {"group1": {"idx": [0, 1]}}
    _, cur_bc = _build_common_params_for_target(
        init_common_params={},
        trainset_features_stats=None,
        plot_file=None,
        train_od_idx=None,
        val_od_idx=None,
        current_train_target=None,
        current_val_target=None,
        outlier_detector=None,
        behavior_config=bc,
        fairness_subgroups=fairness,
    )
    # extra field set on copied config
    extra = getattr(cur_bc, "_precomputed_fairness_subgroups", None)
    if extra is None:
        extra = (cur_bc.model_extra or {}).get("_precomputed_fairness_subgroups")
    assert extra == fairness


def test_build_common_params_with_od_passes_targets():
    bc = TrainingBehaviorConfig()
    tr_tgt = np.array([0, 1, 0])
    va_tgt = np.array([1, 0])
    od_params, _ = _build_common_params_for_target(
        init_common_params={"train_target": "ignored"},
        trainset_features_stats=None,
        plot_file=None,
        train_od_idx=np.array([True, False, True]),
        val_od_idx=None,
        current_train_target=tr_tgt,
        current_val_target=va_tgt,
        outlier_detector=object(),  # truthy
        behavior_config=bc,
        fairness_subgroups=None,
    )
    assert od_params["train_target"] is tr_tgt
    assert od_params["val_target"] is va_tgt


# ----- _build_pre_pipelines -----

def test_build_pre_pipelines_ordinary_only():
    pipes, names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
    )
    assert pipes == [None]
    assert names == [""]


def test_build_pre_pipelines_rfecv_merge():
    pipes, names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=["cb_rfecv"],
        rfecv_models_params={"cb_rfecv": "FAKE_PIPELINE"},
        use_mrmr_fs=False,
        mrmr_kwargs={},
    )
    assert None in pipes
    assert "FAKE_PIPELINE" in pipes
    assert "cb_rfecv " in names


def test_build_pre_pipelines_unknown_rfecv_raises():
    with pytest.raises(ValueError, match="Unknown RFECV"):
        _build_pre_pipelines(
            use_ordinary_models=False,
            rfecv_models=["nope"],
            rfecv_models_params={"cb_rfecv": object()},
            use_mrmr_fs=False,
            mrmr_kwargs={},
        )


def test_build_pre_pipelines_mrmr_included():
    pipes, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs=dict(n_workers=1, verbose=0, fe_max_steps=0),
    )
    assert len(pipes) == 1
    assert any("MRMR" in n for n in names)


def test_build_pre_pipelines_custom():
    class DummyTrans:
        def fit(self, *a, **k): return self
        def transform(self, x): return x
    dummy = DummyTrans()
    pipes, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines={"my_pca": dummy},
    )
    assert dummy in pipes
    assert "my_pca " in names


# ----- _build_process_model_kwargs -----

def test_build_process_model_kwargs_minimal():
    kwargs = _build_process_model_kwargs(
        model_file="/m",
        model_name_with_weight="cb.0.5",
        model_file_name="cb.pkl",
        target_type=TargetTypes.REGRESSION,
        pre_pipeline=None,
        pre_pipeline_name="",
        cur_target_name="tgt",
        models={},
        model_params={"lr": 0.1},
        common_params={"a": 1},
        ens_models=None,
        trainset_features_stats=None,
        verbose=0,
        cached_dfs=None,
    )
    assert kwargs["model_file"] == "/m"
    assert kwargs["target_type"] == TargetTypes.REGRESSION
    assert "skip_preprocessing" not in kwargs
    assert "cached_train_df" not in kwargs


def test_build_process_model_kwargs_polars_pipeline_applied():
    kwargs = _build_process_model_kwargs(
        model_file="/m", model_name_with_weight="m", model_file_name="m.pkl",
        target_type=TargetTypes.REGRESSION, pre_pipeline=None, pre_pipeline_name="",
        cur_target_name="t", models={}, model_params={}, common_params={},
        ens_models=None, trainset_features_stats=None, verbose=0, cached_dfs=None,
        polars_pipeline_applied=True,
    )
    assert kwargs["skip_preprocessing"] is True


def test_build_process_model_kwargs_cached_dfs():
    tr, va, te = pd.DataFrame({"a": [1]}), None, pd.DataFrame({"a": [2]})
    kwargs = _build_process_model_kwargs(
        model_file="/m", model_name_with_weight="m", model_file_name="m.pkl",
        target_type=TargetTypes.REGRESSION, pre_pipeline=None, pre_pipeline_name="",
        cur_target_name="t", models={}, model_params={}, common_params={},
        ens_models=None, trainset_features_stats=None, verbose=0,
        cached_dfs=(tr, va, te),
    )
    assert kwargs["skip_pre_pipeline_transform"] is True
    assert kwargs["cached_train_df"] is tr
    assert kwargs["cached_val_df"] is None
    assert kwargs["cached_test_df"] is te


def test_build_process_model_kwargs_adds_model_category():
    common = {"existing": True}
    kwargs = _build_process_model_kwargs(
        model_file="/m", model_name_with_weight="m", model_file_name="m.pkl",
        target_type=TargetTypes.REGRESSION, pre_pipeline=None, pre_pipeline_name="",
        cur_target_name="t", models={}, model_params={}, common_params=common,
        ens_models=None, trainset_features_stats=None, verbose=0, cached_dfs=None,
        mlframe_model_name="cb",
    )
    assert kwargs["common_params"]["model_category"] == "cb"
    # Original not mutated
    assert "model_category" not in common
