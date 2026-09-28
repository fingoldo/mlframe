"""A target with missing labels under ``target_null_policy="drop_rows"`` trains as if its unlabelled rows never existed.

The reference is a second suite call on the frame reduced to that target's labelled rows, with val and test pinned to
the first call's rows (``split_ids_path``), so both calls train on the same rows: the models must predict the same test
values. Unsupervised steps fitted on the train rows would see the unlabelled ones in the first call (outlier detection,
the row-extreme scores standardised by train column statistics, the global scaler and imputer, composite discovery),
so they are off: there the two calls legitimately differ.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import (
    CompositeTargetDiscoveryConfig,
    PreprocessingExtensionsConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    TargetTypes,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

N_ROWS = 1500


def _frame(seed: int = 0) -> pd.DataFrame:
    """A frame with one fully labelled target (``y_full``) and one with ~30% missing labels (``y_part``)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_ROWS, 4))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["row_id"] = np.arange(N_ROWS)
    df["y_full"] = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.3, N_ROWS)
    y_part = X[:, 2] * 3 - X[:, 3] + rng.normal(0, 0.3, N_ROWS)
    y_part[rng.random(N_ROWS) < 0.3] = np.nan
    df["y_part"] = y_part
    return df


def _suite(df, targets, data_dir, common_init_params, split_config, models=("lgb",)):
    """Run the suite on ``df`` for ``targets`` with drop_rows missing-label handling and every unsupervised step off."""
    return train_mlframe_models_suite(
        df=df,
        target_name="t",
        model_name="m",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=list(targets), columns_to_drop={"row_id"}),
        mlframe_models=list(models),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        split_config=split_config,
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, target_null_policy="drop_rows"),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        preprocessing_extensions=PreprocessingExtensionsConfig(row_wise_extreme_columns_enabled=False),
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        reporting_config=common_init_params,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        verbose=0,
    )


def _test_metrics(models: dict, target: str) -> dict:
    """Per-model test-split metrics for one regression target, keyed by model name."""
    entries = models[TargetTypes.REGRESSION][target]
    entries = entries if isinstance(entries, list) else [entries]
    return {getattr(e, "model_name", str(i)): e.metrics["test"] for i, e in enumerate(entries)}


@pytest.mark.parametrize("models", [("lgb",), ("linear",)])
def test_a_target_with_missing_labels_trains_as_on_its_labelled_rows_alone(tmp_path, common_init_params, models):
    """A target with missing labels trains as if it had been fit on its labelled rows alone from the start: the reference is a second suite run on the frame pre-filtered to those rows, pinned to the same split."""
    df = _frame()
    dir_a, dir_b = str(tmp_path / "a"), str(tmp_path / "b")
    models_a, meta_a = _suite(df, ["y_full", "y_part"], dir_a, common_init_params, TrainingSplitConfig(id_column="row_id"), models)
    split_file = glob.glob(os.path.join(dir_a, "**", "split_ids.parquet"), recursive=True)
    assert len(split_file) == 1, split_file

    labelled = df[df["y_part"].notna()].drop(columns=["y_full"]).reset_index(drop=True)
    models_b, _ = _suite(labelled, ["y_part"], dir_b, common_init_params, TrainingSplitConfig(id_column="row_id", split_ids_path=split_file[0]), models)

    rows = meta_a["target_rows"]["regression/y_part"]
    assert rows["n_labelled"]["test_idx"] == int(df.loc[_split_rows(split_file[0], "test"), "y_part"].notna().sum())
    metrics_a, metrics_b = _test_metrics(models_a, "y_part"), _test_metrics(models_b, "y_part")
    assert metrics_a.keys() == metrics_b.keys()
    assert metrics_a, "no model was trained for y_part"
    # Scalar metrics only (some are nested diagnostics); RMSLE is NaN on a target with negatives.
    compared = [(name, metric, value) for name in metrics_a for metric, value in metrics_b[name].items() if isinstance(value, float)]
    assert compared, "no scalar test metric to compare"
    for name, metric, value in compared:
        assert metrics_a[name][metric] == pytest.approx(value, rel=1e-9, abs=1e-12, nan_ok=True), (name, metric)
    # The fully labelled target beside it trains on every row.
    assert "regression/y_full" not in meta_a["target_rows"]


def _split_rows(path: str, split: str) -> np.ndarray:
    """The row ids the suite assigned to one split, read back from its saved ``split_ids.parquet``."""
    from mlframe.training._fixed_splits import SPLIT_CODES

    stored = pd.read_parquet(path)
    codes = stored["split"]
    wanted = SPLIT_CODES[split] if pd.api.types.is_numeric_dtype(codes) else split
    return stored.loc[codes == wanted, "row_id"].to_numpy()


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_outlier_detection_and_missing_labels_work_together(tmp_path, common_init_params, monkeypatch, task):
    """Each target trains on the rows that are both labelled and kept by outlier detection; a fully labelled target beside
    it keeps every row outlier detection kept."""
    from sklearn.ensemble import IsolationForest

    from mlframe.training.configs import OutlierDetectionConfig
    from mlframe.training.core import _phase_runners as pr

    df = _frame(1)
    targets = ["y_full", "y_part"]
    if task == "classification":
        df["y_full"] = (df["y_full"] > 0).astype(float)
        df["y_part"] = np.where(df["y_part"].isna(), np.nan, (df["y_part"] > 0).astype(float))
    seen = {}
    real = pr._train_one_target

    def spy(ctx, target_type, targets_, name, values):
        """Record each target's filtered train/val/test row indices, then train as usual."""
        seen[name] = dict(train=np.asarray(ctx.filtered_train_idx), val=np.asarray(ctx.filtered_val_idx), test=np.asarray(ctx.test_idx))
        return real(ctx, target_type, targets_, name, values)

    monkeypatch.setattr(pr, "_train_one_target", spy)
    fte_kwargs = {"classification_targets": targets} if task == "classification" else {"regression_targets": targets}
    models, meta = train_mlframe_models_suite(
        df=df,
        target_name="t",
        model_name="m",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(columns_to_drop={"row_id"}, **fte_kwargs),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, target_null_policy="drop_rows"),
        outlier_detection_config=OutlierDetectionConfig(detector=IsolationForest(contamination=0.1, random_state=0)),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        reporting_config=common_init_params,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    labelled = df["y_part"].notna().to_numpy()
    full, part = seen["y_full"], seen["y_part"]
    assert full["train"].size < N_ROWS * 0.8, "outlier detection removed train rows"
    np.testing.assert_array_equal(part["train"], full["train"][labelled[full["train"]]], err_msg="labelled AND kept by OD, in order")
    np.testing.assert_array_equal(part["val"], full["val"][labelled[full["val"]]])
    np.testing.assert_array_equal(part["test"], full["test"][labelled[full["test"]]])
    record = meta["target_rows"][f"{'binary_classification' if task == 'classification' else 'regression'}/y_part"]
    assert record["n_labelled"]["filtered_train_idx"] == part["train"].size < record["n_labelled"]["train_idx"]
    tt = TargetTypes.BINARY_CLASSIFICATION if task == "classification" else TargetTypes.REGRESSION
    for name in targets:
        entries = models[tt][name]
        entry = entries[0] if isinstance(entries, list) else entries
        assert entry.metrics["test"], f"{name} was not scored"


def test_a_right_censored_target_trains_without_test_rows_and_logs_nothing_failing(tmp_path, common_init_params, caplog):
    """Labels missing for the newest rows only (outcomes not known yet): test, the newest block, has no labelled row.

    The target trains on train and val, its test is the suite's test_size=0 shape, its record says low_n, and nothing on
    the way logs a failure (dummy baselines used to report on the empty test; the fit used to predict on it and crash).
    """
    import logging

    df = _frame(2)
    df["ts"] = pd.date_range("2023-01-01", periods=N_ROWS, freq="h")
    df.loc[df.index >= int(N_ROWS * 0.75), "y_part"] = np.nan
    with caplog.at_level(logging.WARNING):
        models, meta = train_mlframe_models_suite(
            df=df,
            target_name="t",
            model_name="m",
            features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=["y_full", "y_part"], columns_to_drop={"row_id"}, ts_field="ts"),
            mlframe_models=["lgb"],
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, target_null_policy="drop_rows"),
            composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
            reporting_config=common_init_params,
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
            verbose=0,
        )
    record = meta["target_rows"]["regression/y_part"]
    assert record["n_labelled"]["test_idx"] == 0 and record["low_n"] is True
    entries = models[TargetTypes.REGRESSION]["y_part"]
    entry = entries[0] if isinstance(entries, list) else entries
    assert entry.metrics["val"] and not entry.metrics.get("test")
    # Not about this target: the host's commit headroom, and the default val placement mixing random in-period rows.
    unrelated = ("mlframe.training._commit_headroom", "mlframe.training.baselines._dummy_baseline_regression")
    bad = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name not in unrelated and any(w in r.getMessage().lower() for w in ("failed", "raised", "empty"))
    ]
    assert not bad, bad


def test_the_cross_target_ensemble_of_a_target_with_gaps_is_built_on_its_labelled_rows(tmp_path, common_init_params, monkeypatch):
    """Post-loop steps work per original target on split arguments; a target with gaps must get its labelled rows there."""
    from mlframe.training.core import _phase_composite_post as post
    from mlframe.training.core import _phase_composite_post_xt_ensemble as xt

    seen = {}
    real_build = xt._build_cross_target_ensemble_for_target

    def spy_build(**kwargs):
        """Record each original target's filtered train/val row indices, then build the ensemble as usual."""
        seen.setdefault("xt", {})[kwargs["_orig_tname"]] = (np.asarray(kwargs["filtered_train_idx"]), np.asarray(kwargs["filtered_val_idx"]))
        return real_build(**kwargs)

    monkeypatch.setattr(xt, "_build_cross_target_ensemble_for_target", spy_build)
    real_wrap = post._run_composite_target_wrapping

    def spy_wrap(**kwargs):
        """Record which targets the composite wrapping step received split args for, then wrap as usual."""
        seen["wrap_keys"] = set(kwargs.get("split_args_by_target") or {})
        return real_wrap(**kwargs)

    monkeypatch.setattr(post, "_run_composite_target_wrapping", spy_wrap)
    df = _frame(3)
    models, _meta = train_mlframe_models_suite(
        df=df,
        target_name="t",
        model_name="m",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=["y_full", "y_part"], columns_to_drop={"row_id"}),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, target_null_policy="drop_rows"),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=True),
        reporting_config=common_init_params,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    labelled = df["y_part"].notna().to_numpy()
    assert "y_part" in seen.get("xt", {}), "the cross-target ensemble step never ran for the target with gaps"
    train_rows, val_rows = seen["xt"]["y_part"]
    assert train_rows.size and labelled[train_rows].all() and labelled[val_rows].all()
    full_train, _ = seen["xt"]["y_full"]
    assert train_rows.size < full_train.size
    assert ("regression", "y_part") in seen["wrap_keys"] and ("regression", "y_full") not in seen["wrap_keys"]
    assert "_CT_ENSEMBLE__y_part" in models[TargetTypes.REGRESSION], "no ensemble was built for the target with gaps"
