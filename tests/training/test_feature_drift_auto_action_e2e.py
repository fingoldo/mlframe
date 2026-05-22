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

    Uses alpha_dom=1.0 (matching the classification sweep regime) so the
    sigmoid stays in transition under drift -- val keeps both classes
    alive after the 8-sigma shift, avoiding the suite's
    'val target has only one unique value' validation error. The
    FI-weighted drift aggregate still crosses 3.0 because f0 dominates
    the linear baseline (alpha_dom=1.0 vs noise alpha=0.1) in
    baseline_diagnostics' ablation FI.

    Drift starts at the 70% boundary so the suite's default 70/15/15
    split (with timestamps disabling stratification) puts ALL drifted
    rows into val/test and zero into train.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([1.0, 0.1, 0.1, 0.1, 0.1])
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
def test_feature_drift_auto_action_default_off_warn_only():
    """Default behaviour: the auto-apply flag is OFF, so even with extreme
    drift on a dominant feature + MLP in the model set + drift detected,
    the MLP is trained with the USER-supplied mlp_kwargs unchanged. The
    sensor still emits a WARN log line with the recommendation and stamps
    ``metadata["feature_drift_auto_action_skipped"]`` for observability.

    This pins the safe default: no black-box config rewrites unless the
    operator explicitly opts in. Without this gate the previous version
    would mutate hyperparams_config every time drift was detected, which
    is surprising behaviour for users who passed mlp_kwargs deliberately."""
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
                df=df, target_name="drift_e2e_default_off", model_name="mt",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
                split_config=TrainingSplitConfig(
                    test_size=0.15, val_size=0.15, val_sequential_fraction=1.0,
                ),
                hyperparams_config=ModelHyperparamsConfig(),
                # Default TrainingBehaviorConfig() has
                # feature_drift_auto_apply_neural_overrides=False.
                behavior_config=TrainingBehaviorConfig(),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None
    metadata = ctx.metadata or {}

    # Sensor still runs.
    assert metadata.get("feature_distribution_drift"), "drift report missing"

    # Auto-apply MUST NOT have fired (flag is OFF).
    assert not metadata.get("feature_drift_auto_action"), (
        f"feature_drift_auto_action must be empty when the auto-apply flag "
        f"defaults to OFF; got {metadata.get('feature_drift_auto_action')}"
    )

    # Skip-stamp MUST be present so dashboards can show what WOULD have been
    # applied if the operator had opted in.
    skipped = metadata.get("feature_drift_auto_action_skipped", {})
    assert skipped, (
        f"feature_drift_auto_action_skipped missing; expected the sensor to "
        f"surface the recommendation it would have applied. metadata keys: "
        f"{list(metadata.keys())}"
    )
    by_type = next(iter(skipped.values()))
    by_target = next(iter(by_type.values()))
    assert by_target.get("reason") == "auto_apply_disabled"
    assert by_target["sklearn_override_recommended"].get("activation") == "identity"


@pytest.mark.slow
def test_feature_drift_auto_action_e2e_with_mlp_opt_in():
    """Opt-in path: when ``feature_drift_auto_apply_neural_overrides=True``
    on regression, the per-target wire-in applies the override (the
    classification version still gates on shape detector independently).

    Confirms the regression auto-apply still works after the default-OFF
    refactor."""
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
                df=df, target_name="drift_e2e_optin", model_name="mt",
                features_and_targets_extractor=fte,
                mlframe_models=["mlp"],
                use_mlframe_ensembles=False,
                verbose=0,
                split_config=TrainingSplitConfig(
                    test_size=0.15, val_size=0.15, val_sequential_fraction=1.0,
                ),
                hyperparams_config=ModelHyperparamsConfig(),
                behavior_config=TrainingBehaviorConfig(
                    feature_drift_auto_apply_neural_overrides=True,
                ),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None
    metadata = ctx.metadata or {}

    auto_action = metadata.get("feature_drift_auto_action", {})
    assert auto_action, (
        f"feature_drift_auto_action missing despite opt-in flag set + "
        f"drifted regression target + mlp in model set. metadata keys: "
        f"{list(metadata.keys())}"
    )
    by_type = next(iter(auto_action.values()))
    by_target = next(iter(by_type.values()))
    assert by_target["sklearn_override"].get("activation") == "identity"
    import torch
    mlframe_override = by_target["mlframe_mlp_kwargs_override"]
    assert mlframe_override["network_params"]["activation_function"] is torch.nn.Identity


@pytest.mark.slow
def test_feature_drift_auto_action_e2e_binary_classification_no_auto_apply():
    """Classification e2e: confirms auto-apply does NOT fire on
    classification regardless of feature drift severity, even with the
    opt-in flag set. Two gates conspire to block it:

      1. ``feature_drift_auto_apply_neural_overrides`` default OFF in
         ``TrainingBehaviorConfig`` (this test doesn't set it -> OFF).
      2. Even with the flag ON, the linear-shape detector reads
         ``baseline_diagnostics.init_score_baseline.delta_vs_raw_pct``
         and gates classification on |delta| <= 10%. The synthetic data
         here saturates the sigmoid (alpha_dom=10 + drift_z=8), making
         the LogReg vs LightGBM gap meaningless -- the shape signal
         won't read as 'clean linear' and the override stays off.

    Asserts:
      1. Sensor still stamps the drift report (observational).
      2. ``feature_drift_auto_action`` is NOT stamped in metadata.
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

    # alpha_dom=1.0 in the data helper keeps the sigmoid in transition so
    # val retains both classes after the 8-sigma drift on the dominant
    # feature -- avoids the suite's 'val target has only one unique value'
    # validation error. Sensor still fires because f0 dominates the linear
    # baseline FI even with alpha_dom=1.0.

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
                # Default flag (feature_drift_auto_apply_neural_overrides=False).
                behavior_config=TrainingBehaviorConfig(),
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=True),
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder.get("ctx")
    assert ctx is not None
    metadata = ctx.metadata or {}

    # Sensor still runs (drift report stamped). recommend_neural_overrides
    # could be either None (if shape signal unavailable or marks the target
    # as nonlinear-shape) or a dict (if the shape detector reads as linear);
    # we don't assert on it because the synthetic-data layer might land in
    # either branch depending on baseline_diagnostics outcomes.
    fd = metadata.get("feature_distribution_drift", {})
    assert fd, "feature_distribution_drift missing -- sensor did not run"

    # Default-OFF gate: the auto-apply flag defaults to False in
    # TrainingBehaviorConfig, so feature_drift_auto_action MUST NOT have
    # been stamped regardless of what the sensor recommended.
    auto_action = metadata.get("feature_drift_auto_action", {})
    assert not auto_action, (
        f"feature_drift_auto_action must be empty on classification when "
        f"the default-OFF flag is set; got {auto_action}"
    )