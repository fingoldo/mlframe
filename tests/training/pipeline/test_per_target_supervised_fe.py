"""Label-supervised composite FE is fitted per target when a suite has several targets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._per_target_supervised_fe import (
    PER_TARGET_KEY,
    apply_per_target_supervised_fe,
    foreign_columns,
    replay_per_target_supervised_fe,
    target_scoped_frames,
)


def _frame(n=3000, seed=0):
    """Four categorical columns and two binary targets, each driven by a different column pair."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: rng.integers(0, 6, n).astype(str) for k in "abcd"})
    pairs = {}
    for name, (u, v) in {"t_ab": ("a", "b"), "t_cd": ("c", "d")}.items():
        table = {(i, j): rng.integers(0, 2) for i in map(str, range(6)) for j in map(str, range(6))}
        pairs[name] = np.array([table[(x, y)] for x, y in zip(df[u], df[v])])
    return df, pairs


def _fit(df, pairs):
    """Run the per-target supervised FE on train/val/test splits of the frame and return its metadata and frames."""
    cfg = PreprocessingExtensionsConfig(categorical_group_concat_auto_enabled=True)
    targets = [("binary", name, y) for name, y in pairs.items()]
    md: dict = {}
    train, val, test = apply_per_target_supervised_fe(df.iloc[:2000], df.iloc[2000:2500], df.iloc[2500:], cfg, targets, np.arange(2000), None, None, None, None, md)
    return md, train, val, test


def test_each_target_gets_groups_learned_from_its_own_labels():
    """Each target's groups equal those learned from its labels alone, and differ between targets."""
    df, pairs = _frame()
    md, train, val, test = _fit(df, pairs)
    per = md[PER_TARGET_KEY]
    assert set(per) == {"binary/t_ab", "binary/t_cd"}
    from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe

    cfg = PreprocessingExtensionsConfig(categorical_group_concat_auto_enabled=True)
    for name, y in pairs.items():
        alone: dict = {}
        apply_categorical_composite_fe(df.iloc[:2000], None, None, cfg, y[:2000], alone)  # this target's labels, original columns only
        assert per[f"binary/{name}"]["state"]["categorical_group_concat_auto_groups"] == alone["categorical_group_concat_auto_groups"]
    assert per["binary/t_ab"]["state"]["categorical_group_concat_auto_groups"] != per["binary/t_cd"]["state"]["categorical_group_concat_auto_groups"]
    own_ab = list(per["binary/t_ab"]["renames"].values())
    assert own_ab and all(c.endswith("__target_binary_t_ab") for c in own_ab)
    assert set(train.columns) == set(val.columns) == set(test.columns)
    assert md["composite_fe_supervised_target"] == "per_target"


def test_replay_reproduces_every_targets_columns():
    """Replaying on raw test rows reproduces every per-target column the fit produced."""
    df, pairs = _frame()
    md, _, _, test = _fit(df, pairs)
    replayed = replay_per_target_supervised_fe(df.iloc[2500:][list("abcd")], md, None)
    new_cols = [c for c in test.columns if c not in "abcd"]
    assert new_cols and set(new_cols) <= set(replayed.columns)
    pd.testing.assert_frame_equal(replayed[new_cols].reset_index(drop=True), test[new_cols].reset_index(drop=True), check_dtype=False)


def test_a_targets_models_do_not_see_other_targets_columns_and_frames_are_restored():
    """Inside the scope a target's frames drop other targets' columns; on exit the originals return."""
    df, pairs = _frame()
    md, train, _, _ = _fit(df, pairs)
    foreign = foreign_columns(md, "binary", "t_ab")
    assert foreign and all(c.endswith("__target_binary_t_cd") for c in foreign)
    ctx = SimpleNamespace(metadata=md, train_df_pd=train, val_df_pd=None, test_df_pd=None, train_df_polars=None, val_df_polars=None,
                          test_df_polars=None, filtered_train_df=None, filtered_val_df=None, cat_features=list(train.columns))
    with target_scoped_frames(ctx, "binary", "t_ab"):
        assert not set(foreign) & set(ctx.train_df_pd.columns)
        assert not set(foreign) & set(ctx.cat_features)
    assert ctx.train_df_pd is train and set(foreign) <= set(ctx.cat_features)


def test_single_target_suite_has_nothing_to_scope():
    """With no per-target state there are no foreign columns and the frames are left alone."""
    assert foreign_columns({}, "binary", "y") == []
    ctx = SimpleNamespace(metadata={}, train_df_pd=pd.DataFrame({"x": [1]}), cat_features=["x"])
    before = ctx.train_df_pd
    with target_scoped_frames(ctx, "binary", "y"):
        assert ctx.train_df_pd is before


@pytest.mark.parametrize("n_targets", [2])
def test_powerset_concat_stays_one_shared_fit(n_targets):
    """The unsupervised powerset is not duplicated per target: per-target runs disable it."""
    df, pairs = _frame(n=600)
    cfg = PreprocessingExtensionsConfig(categorical_group_concat_auto_enabled=True, categorical_powerset_concat_enabled=True)
    md: dict = {}
    apply_per_target_supervised_fe(df, None, None, cfg, [("binary", n, y) for n, y in pairs.items()], None, None, None, None, None, md)
    assert md[PER_TARGET_KEY], "no per-target state was recorded, so the loop below would check nothing"
    for entry in md[PER_TARGET_KEY].values():
        assert "categorical_powerset_concat_columns" not in entry["state"]


def test_two_target_suite_trains_each_target_on_its_own_columns_and_predicts(tmp_path):
    """A two-target suite trains each target on its own columns and predicts end to end."""
    pytest.importorskip("catboost")
    from mlframe.training.configs import OutputConfig
    from mlframe.training.core import predict_mlframe_models_suite, train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df, pairs = _frame(n=1500, seed=3)
    df["num"] = np.random.default_rng(3).normal(size=len(df))
    for name, y in pairs.items():
        df[name] = y
    fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["t_ab", "t_cd"], use_recency_weighting=False)
    models, meta = train_mlframe_models_suite(
        df=df, target_name="pt", model_name="m", features_and_targets_extractor=fte, mlframe_models=["cb"],
        hyperparams_config={"iterations": 5, "cb_kwargs": {"task_type": "CPU", "verbose": 0}},
        use_ordinary_models=True, use_mlframe_ensembles=False, verbose=0,
        preprocessing_extensions=PreprocessingExtensionsConfig(categorical_group_concat_auto_enabled=True),
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models", save_charts=False, run_diagnostics=[]),
    )
    per = meta[PER_TARGET_KEY]
    assert len(per) == 2
    seen = {}
    for ttype, by_name in models.items():
        for tname, entries in by_name.items():
            for entry in entries:
                cols = set(getattr(entry, "columns", None) or [])
                seen[tname] = cols
                assert not cols & set(foreign_columns(meta, ttype, tname)), f"{tname} was trained on another target's columns"
    assert len(seen) == 2
    for key, entry in per.items():  # each target trained on its own per-target columns, not an empty set
        own = set(entry["renames"].values())
        assert own and own <= seen[key.split("/", 1)[1]], key
    out = predict_mlframe_models_suite(df.drop(columns=["t_ab", "t_cd"]).iloc[:200], models_path=str(tmp_path / "models" / "pt" / "m"), verbose=0)
    assert out is not None
