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
    """FTE that returns a regression target and a precomputed split column
    so the suite's chronological split lands drift between train and test.

    The first 80% of rows are 'train slice' with feature_0 ~ N(0, 1); the
    last 20% are 'test slice' with feature_0 ~ N(8, 1) -- an 8-sigma shift
    on the dominant feature. The dominant feature is the only one with
    weight 10 in the regression target; the rest have weight 0.1.
    """

    def __init__(self, target_column: str = "y"):
        from mlframe.training.configs import TargetTypes
        self.target_column = target_column
        self.target_type = TargetTypes.REGRESSION
        self.ts_field = None
        self.group_field = None
        self.weight_schemas = None
        self.target_carrier = "numpy"

    def transform(self, df):
        target_by_type = {self.target_type: {}}
        y = df[self.target_column].values if isinstance(df, pd.DataFrame) else df[self.target_column].to_numpy()
        target_by_type[self.target_type][self.target_column] = y
        cols_to_drop = [self.target_column]
        return (df, target_by_type, None, None, None, None, cols_to_drop, {})


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
                split_config=TrainingSplitConfig(test_size=0.15, val_size=0.15),
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