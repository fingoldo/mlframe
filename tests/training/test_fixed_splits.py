"""Exact, reusable splits: id membership recording, replay on a larger frame, date windows, validation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._fixed_splits import (
    SPLIT_IDS_FILENAME,
    has_pinned_splits,
    pinned_train_val_test_split,
    record_split_membership,
)
from mlframe.training._preprocessing_configs import TrainingSplitConfig
from mlframe.training.splitting import make_train_test_split

_SPLITTER_FIELDS = {"test_size", "val_size", "shuffle_val", "shuffle_test", "val_sequential_fraction", "test_sequential_fraction",
                    "trainset_aging_limit", "wholeday_splitting", "random_seed", "val_placement", "calib_size"}


def _frame(n=600, start="2024-01-01", id_offset=0, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"row_id": np.arange(id_offset, id_offset + n), "ts": ts, "x": rng.normal(size=n)})


def _kwargs(cfg):
    return {k: v for k, v in cfg.model_dump().items() if k in _SPLITTER_FIELDS}


def _fraction_split(df, cfg):
    return make_train_test_split(df=df, timestamps=df["ts"], return_calib=True, **_kwargs(cfg))


def _pinned(df, cfg, groups=None):
    return pinned_train_val_test_split(
        n_rows=len(df), row_ids=df["row_id"].to_numpy(), timestamps=df["ts"], split_config=cfg,
        stratify_y=None, groups=groups, splitter=make_train_test_split, splitter_kwargs=_kwargs(cfg),
    )


def _run1(tmp_path, cfg, df):
    tr, va, te, _, _, _, ca, _ = _fraction_split(df, cfg)
    entry = record_split_membership(row_ids=df["row_id"].to_numpy(), id_column="row_id", train_idx=tr, val_idx=va, test_idx=te,
                                    calib_idx=ca, timestamps=df["ts"], split_dir=str(tmp_path))
    ids = df["row_id"].to_numpy()
    return entry, {"train": set(ids[tr]), "val": set(ids[va]), "test": set(ids[te]), "calib": set(ids[ca])}


_USER_CFG = dict(shuffle_val=True, shuffle_test=False, test_size=0.1, val_size=0.1, wholeday_splitting=False)


def test_run1_writes_membership_file_and_metadata(tmp_path):
    df = _frame()
    entry, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    stored = pd.read_parquet(tmp_path / SPLIT_IDS_FILENAME)
    assert list(stored.columns) == ["row_id", "split"]
    for s in ("train", "val", "test"):
        assert set(stored.loc[stored["split"] == s, "row_id"]) == sets[s]
    assert len(stored) == len(df)
    assert entry["path"].endswith(SPLIT_IDS_FILENAME) and entry["counts"]["test"] == len(sets["test"])
    # Sequential val start is after every train row even though val has a random part drawn from the train period.
    assert pd.Timestamp(entry["holdout_starts"]["val"]) > df.loc[df["row_id"].isin(sets["train"]), "ts"].max()


def test_run2_larger_frame_reuses_test_and_val_exactly(tmp_path, caplog):
    df1 = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df1)
    history = _frame(n=300, start="2023-12-01", id_offset=100_000)
    df2 = pd.concat([history, df1], ignore_index=True).sample(frac=1.0, random_state=3).reset_index(drop=True)
    cfg2 = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), **_USER_CFG)
    assert has_pinned_splits(cfg2)
    with caplog.at_level(logging.INFO):
        tr, va, te, trd, vad, ted, ca, _, info = _pinned(df2, cfg2)
    ids = df2["row_id"].to_numpy()
    assert set(ids[te]) == sets["test"] and set(ids[va]) == sets["val"]
    # Train = every non-holdout row older than the sequential val start, incl. all new history; nothing newer leaks in.
    assert set(history["row_id"]) <= set(ids[tr])
    assert set(ids[tr]) == sets["train"] | set(history["row_id"])
    assert df2["ts"].iloc[tr].max() < df2["ts"].iloc[va].max()
    assert "pinned:file" in ted and "split pinned: test=file, val=file" in caplog.text


def test_run2_future_rows_excluded_from_train(tmp_path):
    df1 = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df1)
    newer = _frame(n=50, start="2024-01-24", id_offset=200_000)  # falls inside the old val/test period
    df2 = pd.concat([df1, newer], ignore_index=True)
    cfg2 = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), **_USER_CFG)
    tr, va, te, *_rest, info = _pinned(df2, cfg2)
    ids = df2["row_id"].to_numpy()
    cutoff = pd.Timestamp(info["cutoff"])
    n_new_after = int((newer["ts"] >= cutoff).sum())
    assert n_new_after > 0 and info["n_excluded_after_cutoff"] == n_new_after
    assert not (set(newer.loc[newer["ts"] >= cutoff, "row_id"]) & set(ids[tr]))
    # Opt-out keeps them in train.
    cfg3 = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), train_before_holdout=False, **_USER_CFG)
    tr3, *_ = _pinned(df2, cfg3)
    assert set(newer["row_id"]) <= set(ids[tr3])


def test_reuse_test_only_carves_val_from_rest(tmp_path):
    df1 = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df1)
    df2 = pd.concat([_frame(n=300, start="2023-12-01", id_offset=100_000), df1], ignore_index=True)
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), reuse_splits=("test",),
                              calib_size=0.05, **_USER_CFG)
    tr, va, te, _, _, _, ca, _, info = _pinned(df2, cfg)
    ids = df2["row_id"].to_numpy()
    assert set(ids[te]) == sets["test"]
    assert len(va) > 0 and not (set(ids[va]) & sets["test"])
    assert len(ca) > 0
    parts = [set(tr), set(va), set(te), set(ca)]
    assert sum(len(p) for p in parts) == len(set().union(*parts))  # disjoint
    assert info["sources"] == {"test": "file"}


def test_date_windows_select_exact_rows():
    df = _frame()
    cfg = TrainingSplitConfig(test_start="2024-01-22", test_end="2024-01-24", val_start="2024-01-20", val_end="2024-01-22",
                              wholeday_splitting=False)
    tr, va, te, _, vad, ted, *_ = _pinned(df, cfg)
    ts = df["ts"]
    assert np.array_equal(te, np.flatnonzero((ts >= "2024-01-22") & (ts < "2024-01-24")))
    assert np.array_equal(va, np.flatnonzero((ts >= "2024-01-20") & (ts < "2024-01-22")))
    assert np.array_equal(tr, np.flatnonzero(ts < "2024-01-20"))  # rows after test_end excluded
    assert "pinned:window" in ted


def test_window_and_ids_combined(tmp_path):
    df = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), reuse_splits=("test",),
                              val_start="2024-01-18", val_end="2024-01-20", wholeday_splitting=False)
    tr, va, te, *_ = _pinned(df, cfg)
    ids = df["row_id"].to_numpy()
    assert set(ids[te]) == sets["test"]
    ts = df["ts"]
    assert np.array_equal(va, np.flatnonzero((ts >= "2024-01-18") & (ts < "2024-01-20")))


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        (dict(test_start="2024-02-01", test_end="2024-01-01"), "must be >"),
        (dict(test_start="2024-01-10", val_start="2024-01-05", val_end="2024-01-12"), "overlaps"),
        (dict(split_ids_path="x.parquet"), "requires id_column"),
        (dict(id_column="row_id", split_ids_path="x.parquet", test_start="2024-01-01"), "defined twice"),
        (dict(reuse_splits=("test", "holdout")), "reuse_splits"),
        (dict(test_start="not a date"), "datetime-like"),
    ],
)
def test_config_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        TrainingSplitConfig(**kwargs)


def test_window_clash_with_file_rows_raises(tmp_path):
    df = _frame()
    _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), reuse_splits=("test",),
                              val_start="2024-01-20", wholeday_splitting=False)
    with pytest.raises(ValueError, match="already pinned"):
        _pinned(df, cfg)


def test_missing_ids_warn_per_split(tmp_path, caplog):
    df = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    dropped = sorted(sets["test"])[:7]
    df2 = df[~df["row_id"].isin(dropped)].reset_index(drop=True)
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), **_USER_CFG)
    with caplog.at_level(logging.WARNING):
        *_, info = _pinned(df2, cfg)
    assert info["missing_ids"] == {"test": 7}
    assert "7 of" in caplog.text


def test_no_ids_matched_raises(tmp_path):
    df = _frame()
    _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    df2 = df.assign(row_id=df["row_id"] + 10_000_000)
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), **_USER_CFG)
    with pytest.raises(ValueError, match="occur in the frame"):
        _pinned(df2, cfg)


def test_duplicate_or_null_ids_raise():
    from mlframe.training._fixed_splits import extract_row_ids

    df = _frame(n=20)
    df.loc[5, "row_id"] = 3
    with pytest.raises(ValueError, match="not unique"):
        extract_row_ids(df, "row_id")
    with pytest.raises(ValueError, match="not a column"):
        extract_row_ids(df, "nope")
    pl = pytest.importorskip("polars")
    with pytest.raises(ValueError, match="null"):
        extract_row_ids(pl.DataFrame({"row_id": [1, None, 3]}), "row_id")
    assert list(extract_row_ids(pl.from_pandas(_frame(n=5)), "row_id")) == [0, 1, 2, 3, 4]


def test_group_spanning_pinned_holdout_warns(tmp_path, caplog):
    df = _frame()
    _, sets = _run1(tmp_path, TrainingSplitConfig(id_column="row_id", **_USER_CFG), df)
    groups = (df["row_id"] % 10).to_numpy()  # every group spans every split
    cfg = TrainingSplitConfig(id_column="row_id", split_ids_path=str(tmp_path / SPLIT_IDS_FILENAME), **_USER_CFG)
    with caplog.at_level(logging.WARNING):
        tr, va, te, *_rest, info = _pinned(df, cfg, groups=groups)
    ids = df["row_id"].to_numpy()
    assert set(ids[te]) == sets["test"]  # rows not moved
    assert info["groups_spanning"]["test"] == 10 and "group(s) have rows in both train" in caplog.text


def test_default_config_is_not_pinned():
    assert not has_pinned_splits(TrainingSplitConfig())
    assert not has_pinned_splits(TrainingSplitConfig(id_column="row_id"))


# ---------------------------------------------------------------------------
# End-to-end: the suite trains/evaluates on the pinned test window and records membership.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
def test_e2e_suite_uses_pinned_test_and_writes_membership(tmp_path, frame_kind):
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import BaselineDiagnosticsConfig, DummyBaselinesConfig, OutputConfig, ReportingConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    n = 800
    rng = np.random.default_rng(5)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"row_id": np.arange(n) * 3 + 7, "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
                       "f0": x0, "f1": x1, "target": 2.0 * x0 - x1 + 0.1 * rng.normal(size=n)})
    ts = df["ts"]
    expected_test = set(df.loc[(ts >= "2024-01-30") & (ts < "2024-02-02"), "row_id"])
    frame = df if frame_kind == "pandas" else __import__("polars").from_pandas(df)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, ts_field="ts")
    models, metadata = train_mlframe_models_suite(
        df=frame,
        target_name="pin_e2e",
        model_name=f"pin_{frame_kind}",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        split_config=TrainingSplitConfig(id_column="row_id", test_start="2024-01-30", test_end="2024-02-02", val_size=0.1,
                                         wholeday_splitting=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    assert metadata["test_size"] == len(expected_test)
    mem = metadata["split_membership"]
    stored = pd.read_parquet(mem["path"])
    assert set(stored.loc[stored["split"] == "test", "row_id"]) == expected_test
    assert "row_id" not in metadata["columns"]
    entry = next(e for by_name in models.values() for es in by_name.values() if isinstance(es, list) for e in es)
    y_test = np.asarray(entry.test_target.values if hasattr(entry.test_target, "values") else entry.test_target).ravel()
    assert np.allclose(np.sort(y_test), np.sort(df.loc[df["row_id"].isin(expected_test), "target"].to_numpy()))


@pytest.mark.parametrize("drop_via", ["extractor_columns_to_drop", "preprocessing_drop_columns"])
def test_e2e_id_column_also_listed_for_dropping(tmp_path, drop_via):
    """The key may also be listed as a column to drop (the natural place for an id): it is still read before being
    dropped, the membership is recorded, it never becomes a feature, and predict accepts a frame that still has it."""
    from mlframe.training.core import predict_from_models, train_mlframe_models_suite
    from mlframe.training.configs import (BaselineDiagnosticsConfig, DummyBaselinesConfig, OutputConfig, PreprocessingConfig,
                                          ReportingConfig)
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    n = 800
    rng = np.random.default_rng(11)
    x0 = rng.normal(size=n)
    df = pd.DataFrame({"row_id": np.arange(n) * 5 + 1, "ts": pd.date_range("2024-01-01", periods=n, freq="h"),
                       "f0": x0, "target": 2.0 * x0 + 0.1 * rng.normal(size=n)})
    extra = {}
    if drop_via == "extractor_columns_to_drop":
        fte = SimpleFeaturesAndTargetsExtractor(ts_field="ts", regression_targets=["target"], columns_to_drop={"row_id"})
    else:
        fte = SimpleFeaturesAndTargetsExtractor(ts_field="ts", regression_targets=["target"])
        extra["preprocessing_config"] = PreprocessingConfig(drop_columns=["row_id"])
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="pin_drop",
        model_name=f"pin_{drop_via}",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        split_config=TrainingSplitConfig(id_column="row_id", test_size=0.1, val_size=0.1, wholeday_splitting=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
        **extra,
    )
    mem = metadata["split_membership"]
    stored = pd.read_parquet(mem["path"])
    assert set(stored["row_id"]) == set(df["row_id"]) and len(stored) == n
    assert mem["counts"]["test"] > 0 and mem["counts"]["val"] > 0
    assert "row_id" not in metadata["columns"]

    result = predict_from_models(df=df.head(50), models=models, metadata=metadata, features_and_targets_extractor=fte,
                                 return_probabilities=False, verbose=0)
    preds = next(iter(result["predictions"].values()))
    assert np.asarray(preds).shape[0] == 50
