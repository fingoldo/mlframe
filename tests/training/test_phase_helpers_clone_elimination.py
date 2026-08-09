"""Regression: ``_phase_fit_pipeline`` must NOT call ``train_df.clone()`` on the
polars-input path even when categorical encoding is enabled.

Pre-fix code (CONV-HIGH-1 gate) cloned the pre-pipeline polars frames whenever
``categorical_encoding is not None`` and ``not skip_categorical_encoding``,
claiming polars-ds ``Blueprint.ordinal_encode`` / ``one_hot_encode`` would
mutate the source. That claim is wrong:

* ``bp.ordinal_encode(...)`` / ``bp.one_hot_encode(...)`` return a NEW
  Blueprint; ``bp.materialize()`` produces a pipeline; ``pipeline.transform()``
  returns a NEW DataFrame. None of these mutate the input frame.
* Polars DataFrames are conceptually immutable through the public API; Arrow
  buffers are reference-counted (``Series.clone()`` is an Arc bump, not a copy)
  and operations like ``with_columns`` produce new frames.
* With Polars' global string cache (memory note: "polars 1.x global string
  cache"), interned dict entries grow monotonically -- existing codes for
  existing string values never shift. So aliasing the pre-encoding frame and
  letting the pipeline transform a separate post-encoding frame is safe; the
  ALIAS still references the original column codes.

Fix: replace the gated ``train_df.clone()`` with an unconditional alias on the
polars-input path. ``_phase_fit_pipeline`` returns ``train_df_polars_pre`` IS
the same object as the caller's input ``train_df``.
"""

from __future__ import annotations


import numpy as np
import pytest

pl = pytest.importorskip("polars")


def _make_toy_polars_df(n_rows: int = 50) -> "pl.DataFrame":
    """Minimal Polars frame with mixed string + numeric columns."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "num1": rng.normal(size=n_rows).astype(np.float64),
            "num2": rng.normal(size=n_rows).astype(np.float64),
            "cat1": rng.choice(["a", "b", "c"], size=n_rows),
            "target": rng.integers(0, 2, size=n_rows),
        }
    )


def test_phase_fit_pipeline_aliases_polars_pre_when_encoding_enabled(monkeypatch):
    """The polars-pre frame returned by ``_phase_fit_pipeline`` must be the
    SAME object as the input ``train_df`` (alias, not clone), even with
    ``categorical_encoding='ordinal'`` and ``skip_categorical_encoding=False``.
    Pre-fix: ``train_df.clone()`` produced a different frame object.
    """
    from mlframe.training.core import _phase_helpers as ph
    # _phase_fit_pipeline's own code resolves ``fit_and_transform_pipeline`` from ITS OWN module's
    # globals (Python's normal function-global-lookup rule) -- it is DEFINED in
    # ``_phase_helpers_fit_pipeline.py``, not ``_phase_helpers.py`` (which only re-exports
    # ``_phase_fit_pipeline`` itself for the call below). Patching ``ph.fit_and_transform_pipeline``
    # has zero effect on the real call site; the stub must be installed on the defining module (CI
    # caught this on shard 8/8 across all required Python versions: AttributeError reading
    # ph.fit_and_transform_pipeline, since ``_phase_helpers`` never even imports that name).
    from mlframe.training.core import _phase_helpers_fit_pipeline as ph_fit
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        PreprocessingConfig,
        FeatureTypesConfig,
        PreprocessingExtensionsConfig,
    )

    train_df = _make_toy_polars_df(50)
    val_df = _make_toy_polars_df(20)
    test_df = _make_toy_polars_df(20)

    # Capture the input frame identity BEFORE the phase runs. The phase may
    # rebind its local ``train_df`` (datetime decomposition, pipeline
    # transform); we care about the original object's identity vs. what's
    # returned in ``train_df_polars_pre``.
    input_train_id = id(train_df)
    input_val_id = id(val_df)
    input_test_id = id(test_df)

    # Stub ``fit_and_transform_pipeline`` to skip the heavy polars-ds pass.
    # We only care about the pre-clone-vs-alias decision in _phase_fit_pipeline.
    def _stub_fit_and_transform(train_df, val_df, test_df, **kwargs):
        # Return frames as-is, no pipeline, no cat_features.
        """Stub fit and transform."""
        return train_df, val_df, test_df, None, []

    monkeypatch.setattr(ph_fit, "fit_and_transform_pipeline", _stub_fit_and_transform)

    # Build configs that force the pre-fix clone branch to fire.
    pipeline_config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
    )
    preprocessing_config = PreprocessingConfig()
    feature_types_config = FeatureTypesConfig()

    metadata: dict = {}
    out = ph._phase_fit_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        mlframe_models=[
            "linear"
        ],  # non-polars-native to keep encoding on (the canonical alias; "lr" silently fell through to TreeModelStrategy + UserWarning, which IS polars-native -- the inverse of what the comment promises)
        pipeline_config=pipeline_config,
        preprocessing_config=preprocessing_config,
        feature_types_config=feature_types_config,
        # NOT None: _phase_fit_pipeline replaces preprocessing_extensions=None with a fresh
        # PreprocessingExtensionsConfig(), whose row_wise_summary_stats_enabled /
        # row_wise_extreme_columns_enabled default to True (documented, intentional). Those steps
        # compute extension columns on the pandas side and back-merge them into train_df_polars_pre
        # (a genuinely new, hstack'd frame) -- a REAL rebinding orthogonal to what this test checks
        # (the CONV-HIGH-1 clone-vs-alias decision). Explicitly disabling both keeps this test isolated
        # to that one decision, matching its own docstring's scope.
        preprocessing_extensions=PreprocessingExtensionsConfig(row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False),
        metadata=metadata,
        verbose=False,
    )

    # Per the return tuple shape at the end of _phase_fit_pipeline:
    # (train_df, val_df, test_df, pipeline, extensions_pipeline,
    #  cat_features, cat_features_polars,
    #  was_polars_input, all_models_polars_native, polars_pipeline_applied,
    #  train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
    #  pipeline_config, preprocessing_extensions, train_df_pandas_pre_meta)
    train_df_polars_pre = out[10]
    val_df_polars_pre = out[11]
    test_df_polars_pre = out[12]

    assert train_df_polars_pre is not None
    assert id(train_df_polars_pre) == input_train_id, (
        "train_df_polars_pre was cloned (different object id); the gate "
        "elimination requires alias semantics. Pre-fix code called "
        "train_df.clone() inside _phase_fit_pipeline; the new path must do "
        "`train_df_polars_pre = train_df` instead."
    )
    assert id(val_df_polars_pre) == input_val_id
    assert id(test_df_polars_pre) == input_test_id
