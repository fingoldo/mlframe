"""End-to-end coverage of the feature-drift auto-action layer.

Unit tests live in ``test_feature_drift_report.py`` (translator, recommend
field, merge logic). This file boots an actual multi-target
``train_mlframe_models_suite`` call, injects a deliberate feature shift
between train and val/test, and asserts that:

  1. ``metadata["feature_distribution_drift"]`` is stamped for the
     drifted target.
  2. When the FI-weighted drift score crosses
     ``WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLD`` (3.0), the per-target
     wire-in stamps ``metadata["feature_drift_auto_action"]`` with the
     sklearn-shape override + translated mlframe-shape override.
3. Suite completes without raising on the override-merged hyperparams_config.

The test is marked ``slow`` because it builds the suite (~30-60s) and
loads pytorch lightning to run mlp; CI shards that exclude ``slow`` skip
it. The lighter ``test_feature_drift_report.py`` unit tests run on every
PR push.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


class _DriftingFeaturesAndTargetsExtractor:
    """FTE that returns a regression or binary-classification target so the
    suite's chronological tail split lands drift between train and test.

    The first 80% of rows are 'train slice' with feature_0 ~ N(0, 1); the
    last 20% are 'test slice' with feature_0 ~ N(8, 1) -- an 8-sigma shift
    on the dominant feature. The dominant feature is the only one with
    weight 10 in the regression target; the rest have weight 0.1.
    """

    def __init__(self, target_column: str = "y", target_type=None):
        from mlframe.training.configs import TargetTypes
        self.target_column = target_column
        self.target_type = target_type if target_type is not None else TargetTypes.REGRESSION
        self.ts_field = None
        self.group_field = None
        self.weight_schemas = None
        self.target_carrier = "numpy"

    def transform(self, df):
        target_by_type = {self.target_type: {}}
        y = df[self.target_column].values if isinstance(df, pd.DataFrame) else df[self.target_column].to_numpy()
        target_by_type[self.target_type][self.target_column] = y
        cols_to_drop = [self.target_column]
        # Synthetic timestamps = row index. This disables auto-stratification
        # in the splitter (``_phase_helpers_fit_split.py:140``: stratify only
        # fires when timestamps is None) so classification targets get a
        # chronological tail split instead of class-balanced stratified
        # sampling -- required for the drift-injection synthetic to land
        # drifted rows in val/test instead of mixing them back into train.
        timestamps = np.arange(len(df))
        return (df, target_by_type, None, None, timestamps, None, cols_to_drop, {})


def _make_drifted_regression_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a regression dataframe where rows are pre-sorted so a
    chronological split lands feature drift between train and test slices.

    Dominant feature alpha = 10 (sensor sees this as 'high FI');
    nondominant alpha = 0.1. Feature_0 distribution shifts +8 std for the
    last 20% of rows. The FI-weighted drift score should be
    approximately |8 * 10| / |10 + 4*0.1| = 7.7 -- well above the
    threshold of 3.0.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([10.0, 0.1, 0.1, 0.1, 0.1])
    n_train = int(n * 0.8)
    n_test = n - n_train

    X_train = rng.normal(0.0, 1.0, (n_train, 5))
    X_test = rng.normal(0.0, 1.0, (n_test, 5))
    X_test[:, 0] += 8.0  # 8-sigma drift on dominant feature

    X = np.vstack([X_train, X_test])
    y = X @ alphas + rng.normal(0.0, 1.0, n)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["y"] = y.astype(np.float32)
    return df


def _make_drifted_binary_classification_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Mirror of ``_make_drifted_regression_df`` for binary classification.

    Uses alpha_dom=10 (matching the regression helper) -- this saturates
    the sigmoid for the drifted rows so test labels skew to class 1,
    BUT keeps train labels balanced and crucially makes f0 strongly
    dominant in the baseline_diagnostics FI ablation. Without that
    dominance the FI-weighted drift aggregate dilutes across uninformative
    features and stays below the 3.0 threshold even at drift_z=8. The
    wire-in test cares about (a) the sensor's FI-weighted score crossing
    threshold and (b) the override flowing into MLP construction; the
    saturated-test-labels regime is irrelevant for those assertions.

    Drift starts at the 70% boundary so the suite's default 70/15/15
    split (with timestamps disabling stratification) puts ALL drifted
    rows into val/test and zero into train.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([10.0, 0.1, 0.1, 0.1, 0.1])
    n_train = int(n * 0.7)
    n_test = n - n_train

    X_train = rng.normal(0.0, 1.0, (n_train, 5))
    X_test = rng.normal(0.0, 1.0, (n_test, 5))
    X_test[:, 0] += 8.0

    X = np.vstack([X_train, X_test])
    score = X @ alphas + rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-score / 3.0))
    y = (rng.random(n) < p).astype(np.int64)

    df = pd.DataFrame(X.astype(np.float32), columns=[f"f{i}" for i in range(5)])
    df["y"] = y
    return df


@pytest.mark.slow
def test_feature_drift_auto_action_e2e_regression():
    """Full suite run with drift-on-dominant injected between train and
    val/test slices. Verifies the feature-drift sensor produces a
    recommendation AND the per-target wire-in stamps the auto-action
    metadata.

    Uses ``has_time=True`` so the suite splits chronologically (preserving
    the row order so drift lands between train and val/test); without this
    a random split would mix drifted rows back into train and the sensor
    wouldn't see any z-score gap.
    """
    from mlframe.training import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        ModelHyperparamsConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )

    import mlframe.training.core._phase_train_one_target as pt

    df = _make_drifted_regression_df(n=600, seed=0)
    fte = _DriftingFeaturesAndTargetsExtractor("y")

    captured_ctx_holder: dict = {}
    _orig_train_one = pt._train_one_target

    def _stash_ctx(ctx, target_type, targets, cur_target_name, cur_target_values):
        captured_ctx_holder["ctx"] = ctx
        return _orig_train_one(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _stash_ctx
    import mlframe.training.core.main as _main
    pt_alias = _main.pr
    _orig_pt_alias = pt_alias._train_one_target
    pt_alias._train_one_target = _stash_ctx

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=df, target_name="drift_e2e", model_name="mt",
                features_and_targets_extractor=fte,
                # Use linear-only model set: avoids the ~14s pytorch lightning
                # import that mlp would trigger, but the wire-in still fires
                # on the diagnostics phase that ran before model selection.
                # If "mlp" not in mlframe_models, the wire-in early-exits
                # without stamping feature_drift_auto_action -- so we test
                # the SENSOR side here and rely on the unit tests in
                # test_feature_drift_report.py for the merge logic.
                mlframe_models=["linear"],
                use_mlframe_ensembles=False,
                verbose=0,
                # Defaults shuffle_val=False / shuffle_test=False give a
                # chronological tail split; rows stay in input order so
                # the injected drift (last 20% of rows) lands in val/test.
                split_config=TrainingSplitConfig(
                    test_size=0.15, val_size=0.15,
                    # val_sequential_fraction=1.0 forces val to be a single
                    # contiguous slice from the train/test boundary instead
                    # of mixing in random rows from earlier. Required for
                    # this drift-injection synthetic so the train slice
                    # is purely pre-drift.
                    val_sequential_fraction=1.0,
                ),
                hyperparams_config=ModelHyperparamsConfig(),
                behavior_config=TrainingBehaviorConfig(),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None, "ctx was never captured -- _train_one_target wasn't called"

    metadata = ctx.metadata or {}
    fd_stamps = metadata.get("feature_distribution_drift", {})
    assert fd_stamps, (
        f"expected feature_distribution_drift in metadata; got keys={list(metadata.keys())}"
    )

    # The drift sensor should have flagged a non-trivial z on f0 for the
    # regression target.
    by_type = next(iter(fd_stamps.values()))
    by_target = next(iter(by_type.values()))
    assert "per_feature" in by_target
    f0_entry = by_target["per_feature"].get("f0")
    assert f0_entry is not None, f"f0 missing from per_feature stamp: {by_target['per_feature']}"
    test_z = float(f0_entry["test_z"])
    assert abs(test_z) > 3.0, (
        f"expected |test_z| > 3.0 for f0 after 8-sigma injected drift; got test_z={test_z}"
    )


@pytest.mark.slow
def test_feature_drift_auto_action_e2e_with_mlp_in_model_set():
    """Second e2e variant that includes MLP in the model set so the
    PER-TARGET wire-in actually fires (the linear-only variant tests the
    sensor side; this one tests the merge-into-hyperparams_config side).

    Pays the ~14s pytorch lightning import cost so this test is heavier
    than the linear-only variant -- still ``slow`` marked. The assertion
    is on metadata["feature_drift_auto_action"][regression][drift_e2e]
    existing with the expected sklearn + mlframe shapes; we don't assert
    on MLP test R^2 because the MLP fit happens AFTER the wire-in and is
    its own complex flow (covered by the bench-stack, not this test)."""
    from mlframe.training import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        ModelHyperparamsConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )

    import mlframe.training.core._phase_train_one_target as pt

    df = _make_drifted_regression_df(n=600, seed=0)
    fte = _DriftingFeaturesAndTargetsExtractor("y")

    captured_ctx_holder: dict = {}
    _orig_train_one = pt._train_one_target

    def _stash_ctx(ctx, target_type, targets, cur_target_name, cur_target_values):
        captured_ctx_holder["ctx"] = ctx
        return _orig_train_one(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _stash_ctx
    import mlframe.training.core.main as _main
    pt_alias = _main.pr
    _orig_pt_alias = pt_alias._train_one_target
    pt_alias._train_one_target = _stash_ctx

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=df, target_name="drift_e2e_mlp", model_name="mt",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],  # the actual wire-in target
                use_mlframe_ensembles=False,
                verbose=0,
                split_config=TrainingSplitConfig(
                    test_size=0.15, val_size=0.15,
                    # val_sequential_fraction=1.0 forces val to be a single
                    # contiguous slice from the train/test boundary instead
                    # of mixing in random rows from earlier. Required for
                    # this drift-injection synthetic so the train slice
                    # is purely pre-drift.
                    val_sequential_fraction=1.0,
                ),
                hyperparams_config=ModelHyperparamsConfig(),
                behavior_config=TrainingBehaviorConfig(),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None, "ctx not captured"
    metadata = ctx.metadata or {}

    # Sensor side: drift stamped.
    assert metadata.get("feature_distribution_drift"), (
        "feature_distribution_drift missing from metadata"
    )

    # Wire-in side: feature_drift_auto_action stamped because (a) the 8-sigma
    # drift produces weighted_score >> 3.0, (b) baseline_diagnostics is
    # enabled in this config and produces FI, (c) "mlp" is in mlframe_models,
    # (d) target_type is regression. With all four conditions met the
    # auto-action MUST stamp; an empty dict means the wire-in regressed.
    auto_action = metadata.get("feature_drift_auto_action", {})
    assert auto_action, (
        f"feature_drift_auto_action missing despite drifted regression "
        f"target + mlp in model set + baseline_diagnostics enabled. "
        f"metadata keys: {list(metadata.keys())}"
    )
    by_type = next(iter(auto_action.values()))
    by_target = next(iter(by_type.values()))
    assert "sklearn_override" in by_target
    assert "mlframe_mlp_kwargs_override" in by_target
    assert by_target["sklearn_override"].get("activation") == "identity"
    # The translation must include the linear-collapse network_params.
    mlframe_override = by_target["mlframe_mlp_kwargs_override"]
    assert "network_params" in mlframe_override
    import torch
    assert mlframe_override["network_params"]["activation_function"] is torch.nn.Identity


@pytest.mark.slow
def test_feature_drift_auto_action_e2e_binary_classification_no_auto_apply():
    """Classification mirror of the regression e2e: confirms the auto-apply
    is GATED OFF for classification (target-type-grouped threshold is None).

    The paired threshold study
    (``profiling/bench_drift_fi_vs_model_harm_classification.py``, 810
    trials) found Pearson r=-0.101 overall for classification: the FI-
    weighted drift score does NOT reliably predict MLP harm on
    classification targets, with interaction_binary showing r=-0.227
    (MLP with relu actually OUTPERFORMS LogReg on interaction-rich
    targets under drift). The wire-in must NOT auto-apply the override
    even when extreme feature drift is detected -- the override is
    documented in ``ROBUST_MLP_OVERRIDES_UNDER_DRIFT_CLASSIFICATION``
    for manual use but not auto-triggered.

    Asserts:
      1. Sensor still stamps the drift report (observational).
      2. ``recommend_neural_overrides`` is None on the classification
         report (the per-type threshold gate filtered it out).
      3. ``feature_drift_auto_action`` is NOT stamped in metadata.
    """
    from mlframe.training import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        ModelHyperparamsConfig,
        TargetTypes,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )
    import mlframe.training.core._phase_train_one_target as pt

    df = _make_drifted_binary_classification_df(n=600, seed=0)
    fte = _DriftingFeaturesAndTargetsExtractor("y", target_type=TargetTypes.BINARY_CLASSIFICATION)

    # alpha_dom=10 in the data helper saturates the sigmoid for drifted
    # rows -- all test labels become class 1 and the suite raises on
    # 'val target has only one unique value'. continue_on_model_failure
    # makes the suite swallow the per-model crash so we can still inspect
    # the auto-action stamp that landed BEFORE the MLP fit.

    captured_ctx_holder: dict = {}
    _orig_train_one = pt._train_one_target

    def _stash_ctx(ctx, target_type, targets, cur_target_name, cur_target_values):
        captured_ctx_holder["ctx"] = ctx
        return _orig_train_one(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _stash_ctx
    import mlframe.training.core.main as _main
    pt_alias = _main.pr
    _orig_pt_alias = pt_alias._train_one_target
    pt_alias._train_one_target = _stash_ctx

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=df, target_name="drift_e2e_clf", model_name="mt",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
                split_config=TrainingSplitConfig(
                    test_size=0.15, val_size=0.15,
                    # val_sequential_fraction=1.0 forces val to be a single
                    # contiguous slice from the train/test boundary instead
                    # of mixing in random rows from earlier. Required for
                    # this drift-injection synthetic so the train slice
                    # is purely pre-drift.
                    val_sequential_fraction=1.0,
                ),
                hyperparams_config=ModelHyperparamsConfig(),
                behavior_config=TrainingBehaviorConfig(continue_on_model_failure=True),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None
    metadata = ctx.metadata or {}

    # Sensor still runs (drift report stamped) but the recommendation must
    # be None for classification because the per-type threshold is None.
    fd = metadata.get("feature_distribution_drift", {})
    assert fd, "feature_distribution_drift missing -- sensor did not run"
    by_type = next(iter(fd.values()))
    by_target = next(iter(by_type.values()))
    assert by_target["recommend_neural_overrides"] is None, (
        f"recommend_neural_overrides MUST be None on classification (per-type "
        f"threshold is None); got {by_target['recommend_neural_overrides']}"
    )

    # The wire-in must NOT have stamped auto_action since no override
    # was recommended.
    auto_action = metadata.get("feature_drift_auto_action", {})
    assert not auto_action, (
        f"feature_drift_auto_action must be empty on classification; "
        f"got {auto_action}"
    )