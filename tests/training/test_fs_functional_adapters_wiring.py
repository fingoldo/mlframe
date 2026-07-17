"""Wiring + regression tests for the functional-utility FS selectors (ForwardSelect /
GreedyBackwardElimination / ZeroImportancePruning / CascadeSelect), wrapped by sklearn adapters in
``mlframe.feature_selection.functional_adapters`` and reachable from the suite via
``FeatureSelectionConfig.use_<sel>_fs`` + a ``_build_pre_pipelines`` branch (mirrors ACE/ShapProxiedFS).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest

from mlframe.training import FeatureSelectionConfig


def _build(**over):
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    base = dict(use_ordinary_models=False, rfecv_models=[], rfecv_models_params={}, use_mrmr_fs=False, mrmr_kwargs={})
    base.update(over)
    return _build_pre_pipelines(**base)


def _make_classification_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "s0": signal,
            "s1": signal * 0.9 + 0.1 * rng.standard_normal(n),
            "noise0": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
        }
    )
    y = pd.Series((signal > 0).astype(int))
    return df, y


# --------------------------------------------------------------------------- reachability


@pytest.mark.parametrize(
    "flag,names_key,kwargs_field",
    [
        ("use_forward_select_fs", "ForwardSelect ", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "GreedyBackwardElimination ", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "ZeroImportancePruning ", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "CascadeSelect ", "cascade_select_kwargs"),
    ],
)
def test_selector_reachable_from_suite(flag, names_key, kwargs_field):
    pps, names = _build(**{flag: True})
    assert names_key in names
    sel = pps[names.index(names_key)]
    assert getattr(sel, "_mlframe_selector_kind_") == names_key.strip()


def test_selector_kind_classifies_all_four():
    from mlframe.training.core._phase_train_one_target import _selector_kind

    for flag, names_key in [
        ("use_forward_select_fs", "ForwardSelect "),
        ("use_greedy_backward_elimination_fs", "GreedyBackwardElimination "),
        ("use_zero_importance_pruning_fs", "ZeroImportancePruning "),
        ("use_cascade_select_fs", "CascadeSelect "),
    ]:
        pps, names = _build(**{flag: True})
        assert _selector_kind(pps[names.index(names_key)]) == names_key.strip()


# --------------------------------------------------------------------------- master-flag gate + kwargs validation


@pytest.mark.parametrize(
    "flag,kwargs_field",
    [
        ("use_forward_select_fs", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "cascade_select_kwargs"),
    ],
)
def test_kwargs_master_flag_gate(flag, kwargs_field):
    # The 4 selector flags default to True (2026-07-12); the guard now only fires when the caller
    # explicitly turns the flag off while still supplying its kwargs.
    with pytest.raises(ValueError, match=f"{kwargs_field} supplied but {flag}"):
        FeatureSelectionConfig(**{flag: False, kwargs_field: {"cv": 3}})


@pytest.mark.parametrize(
    "flag,kwargs_field",
    [
        ("use_forward_select_fs", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "cascade_select_kwargs"),
    ],
)
def test_kwargs_rejects_unknown_key(flag, kwargs_field):
    with pytest.raises(ValueError, match="unknown key"):
        FeatureSelectionConfig(**{flag: True, kwargs_field: {"definitely_not_a_param": 1}})


# --------------------------------------------------------------------------- biz_value: genuinely runs the selector


def test_biz_forward_select_shrinks_noisy_frame():
    df, y = _make_classification_frame()
    pps, names = _build(use_forward_select_fs=True, forward_select_kwargs={"cv": 3, "min_improvement": 0.001})
    sel = pps[names.index("ForwardSelect ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) < df.shape[1]
    assert {"s0", "s1"} & set(kept), f"forward_select dropped both signal columns, kept {kept}"


def test_biz_greedy_backward_elimination_shrinks_noisy_frame():
    from sklearn.linear_model import LogisticRegression

    df, y = _make_classification_frame(n=400)
    pps, names = _build(
        use_greedy_backward_elimination_fs=True,
        greedy_backward_elimination_kwargs={"cv": None, "estimator": LogisticRegression(max_iter=1000)},
    )
    sel = pps[names.index("GreedyBackwardElimination ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) < df.shape[1], f"expected noise columns pruned, kept {kept}"


def test_biz_zero_importance_pruning_shrinks_noisy_frame():
    df, y = _make_classification_frame()
    pps, names = _build(use_zero_importance_pruning_fs=True, zero_importance_pruning_kwargs={})
    sel = pps[names.index("ZeroImportancePruning ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) <= df.shape[1]


def test_biz_cascade_select_shrinks_noisy_frame():
    df, y = _make_classification_frame(n=300)
    pps, names = _build(use_cascade_select_fs=True, cascade_select_kwargs={"n_boruta_iterations": 10, "cv": 3})
    sel = pps[names.index("CascadeSelect ")]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert len(kept) <= df.shape[1]
    assert hasattr(sel, "cascade_result_")


# --------------------------------------------------------------------------- regression: raw categorical column crashes the default tree estimator


def _make_classification_frame_with_categorical(n=200, seed=0):
    """Same signal/noise shape as ``_make_classification_frame`` plus one raw string categorical column, mirroring a
    CatBoost-native ``skip_categorical_encoding=True`` frame (categorical kept as object dtype, not pre-encoded)."""
    rng = np.random.default_rng(seed)
    df, y = _make_classification_frame(n=n, seed=seed)
    df["cat_0"] = rng.choice(["A", "B", "C"], size=n)
    return df, y


@pytest.mark.parametrize(
    "flag,kwargs_field,kwargs",
    [
        ("use_forward_select_fs", "forward_select_kwargs", {"cv": 3, "min_improvement": 0.001}),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs", {"cv": None}),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs", {}),
        ("use_cascade_select_fs", "cascade_select_kwargs", {"n_boruta_iterations": 10, "cv": 3}),
    ],
)
def test_selector_handles_raw_categorical_column_instead_of_crashing(flag, kwargs_field, kwargs):
    """The default ``_default_tree_estimator`` is a plain sklearn RandomForest, which raises "could not convert
    string to float" on ANY raw string column -- caught live via a fuzz combo where ``use_forward_select_fs``
    (default-ON in FeatureSelectionConfig) silently dropped a whole CatBoost model+weight combo from the suite
    because its categorical columns are kept as native strings (``skip_categorical_encoding=True``). Each selector
    must ordinal-encode non-numeric columns internally instead of crashing."""
    names_key = {
        "use_forward_select_fs": "ForwardSelect ",
        "use_greedy_backward_elimination_fs": "GreedyBackwardElimination ",
        "use_zero_importance_pruning_fs": "ZeroImportancePruning ",
        "use_cascade_select_fs": "CascadeSelect ",
    }[flag]
    df, y = _make_classification_frame_with_categorical()
    pps, names = _build(**{flag: True, kwargs_field: kwargs})
    sel = pps[names.index(names_key)]
    kept = list(sel.fit(df, y).transform(df).columns)
    assert set(kept) <= set(df.columns)


# --------------------------------------------------------------------------- regression: polars frame support (2026-07-12)
# Surfaced by the 2026-07-12 default-on flip: the suite's pre-pipeline slot is polars-native, but these
# selectors/adapters were only ever exercised with pandas while opt-in. Two distinct bugs found:
#   1) ``_FunctionalSelectorBase.transform`` fell through to ``np.asarray(X)[:, support]`` for any non-pandas
#      frame, silently dropping column names/dtypes -- downstream reporting (category_discriminability) then
#      crashed on a bare ndarray with no ``.columns``.
#   2) CascadeSelect's internal Boruta stage (``_boruta.py``) used pandas-only ``.apply(axis=0)``.


def test_transform_preserves_polars_frame_and_columns():
    import polars as pl

    df, y = _make_classification_frame()
    pl_df = pl.DataFrame(df)
    pps, names = _build(use_forward_select_fs=True, forward_select_kwargs={"cv": 3, "min_improvement": 0.001})
    sel = pps[names.index("ForwardSelect ")]
    sel.fit(pl_df, y.to_numpy())
    out = sel.transform(pl_df)
    assert isinstance(out, pl.DataFrame), f"expected a polars DataFrame back, got {type(out)}"
    assert list(out.columns) == sel.selected_features_


def test_biz_cascade_select_accepts_polars_frame():
    """CascadeSelect's internal Boruta stage used to crash on a polars frame (pandas-only ``.apply``)."""
    import polars as pl

    df, y = _make_classification_frame(n=300)
    pl_df = pl.DataFrame(df)
    pps, names = _build(use_cascade_select_fs=True, cascade_select_kwargs={"n_boruta_iterations": 10, "cv": 3})
    sel = pps[names.index("CascadeSelect ")]
    kept = list(sel.fit(pl_df, y.to_numpy()).transform(pl_df).columns)
    assert len(kept) <= pl_df.shape[1]


# --------------------------------------------------------------------------- regression: default is ON (2026-07-12)


def test_default_config_enables_all_four_new_flags():
    """``FeatureSelectionConfig`` defaults all four selectors ON (2026-07-12 default-flip); kwargs stay
    None (no forced overrides) until a caller opts into custom selector params."""
    cfg = FeatureSelectionConfig()
    assert cfg.use_forward_select_fs is True
    assert cfg.forward_select_kwargs is None
    assert cfg.use_greedy_backward_elimination_fs is True
    assert cfg.greedy_backward_elimination_kwargs is None
    assert cfg.use_zero_importance_pruning_fs is True
    assert cfg.zero_importance_pruning_kwargs is None
    assert cfg.use_cascade_select_fs is True
    assert cfg.cascade_select_kwargs is None


def test_build_pre_pipelines_opt_out_bit_identical_without_new_flags():
    """``_build_pre_pipelines`` itself still defaults every new selector kwarg to False (the config-level
    flip lives in ``FeatureSelectionConfig``, not this lower-level builder) -- explicitly passing all four
    flags False still reproduces the pre-existing pipeline list/names byte-for-byte, i.e. the opt-out path
    the config-level True default can still be overridden to."""
    pps_before, names_before = _build(use_mrmr_fs=True, mrmr_kwargs={})
    pps_after, names_after = _build(
        use_mrmr_fs=True,
        mrmr_kwargs={},
        use_forward_select_fs=False,
        use_greedy_backward_elimination_fs=False,
        use_zero_importance_pruning_fs=False,
        use_cascade_select_fs=False,
    )
    assert names_before == names_after
    assert len(pps_before) == len(pps_after)
