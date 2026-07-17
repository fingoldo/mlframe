"""biz_val + regression coverage for the row-wise FE extensions wired into
``PreprocessingExtensionsConfig`` (row_wise_summary_stats / row_wise_top_k_extreme_columns).

Both flags default to ``True`` in ``PreprocessingExtensionsConfig`` AND
``_phase_helpers_fit_pipeline.py`` now auto-constructs a default
``PreprocessingExtensionsConfig()`` whenever a caller passes no
``preprocessing_extensions`` at all (the suite's own top-level default is
``None``) -- both pieces are required for the steps to be genuinely
default-ON rather than merely reachable via opt-in. See
``training/core/DEFAULTS_CHANGELOG.md``.
"""

from __future__ import annotations

import numpy as np


def _reg_frame(seed=7, n=300):
    """Builds a small polars regression frame with a linear target over three numeric features."""
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (2 * x0 - x1 + 0.5 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _run_suite(preprocessing_extensions, tmp_path):
    """Runs train_mlframe_models_suite with the given row-wise preprocessing extensions toggle."""
    from sklearn.linear_model import Ridge

    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        OutputConfig,
        TrainingBehaviorConfig,
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        ReportingConfig,
    )
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    custom = Ridge(alpha=0.7)

    kwargs = dict(
        df=_reg_frame(),
        target_name="rw",
        model_name="rw_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=[custom],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False),
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )
    if preprocessing_extensions is not _UNSET:
        kwargs["preprocessing_extensions"] = preprocessing_extensions

    models, _metadata = train_mlframe_models_suite(**kwargs)

    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    trained = [e[0] if isinstance(e, tuple) and e else e for e in trained]
    ridge_entries = [e for e in trained if isinstance(getattr(e, "model", None), Ridge)]
    assert ridge_entries, f"custom Ridge instance never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"
    fitted = ridge_entries[0].model
    return list(getattr(fitted, "feature_names_in_", []))


_UNSET = object()


def test_biz_val_row_wise_extensions_default_on_without_touching_config(tmp_path):
    """A caller that NEVER touches ``preprocessing_extensions`` (the suite's own top-level
    default) must still see row_summary_* / row_extreme_* columns in the model's actual
    fitted feature set -- proving the steps are truly default-ON, not merely opt-in-reachable."""
    cols = _run_suite(_UNSET, tmp_path)
    summary_cols = [c for c in cols if c.startswith("row_summary_")]
    extreme_cols = [c for c in cols if c.startswith("row_extreme_")]
    assert summary_cols, f"row_wise_summary_stats must fire by default; got columns={cols}"
    assert extreme_cols, f"row_wise_top_k_extreme_columns must fire by default; got columns={cols}"


def test_row_wise_extensions_explicit_false_disables_both(tmp_path):
    """Regression: explicitly setting both flags False must actually suppress the columns
    (proves the toggle works in both directions, not just that the default is on)."""
    from mlframe.training.configs import PreprocessingExtensionsConfig

    cols = _run_suite(
        PreprocessingExtensionsConfig(row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False),
        tmp_path,
    )
    assert not [c for c in cols if c.startswith("row_summary_")], f"row_summary_* leaked despite enabled=False; cols={cols}"
    assert not [c for c in cols if c.startswith("row_extreme_")], f"row_extreme_* leaked despite enabled=False; cols={cols}"
