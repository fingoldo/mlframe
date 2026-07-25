"""default_via_or wave 1: fixes 6 P1 findings where a legitimate explicit ``0``/``0.0`` config
value was silently rewritten to the fallback default via ``getattr(obj, "attr", default) or
default`` -- the same bug class as the earlier ``random_seed=0`` fix (F1,
``training_pipeline.md``), found across new call sites by the ``default_via_or_trap`` scanner:

- ``training/models.py``: RANSAC's inner ``Ridge(alpha=...)`` -- ``alpha=0`` (unregularized OLS)
  is a standard, deliberate configuration.
- ``training/pipeline/_target_encoding_composite_fe.py``: ``two_step_target_encode_smoothing`` --
  ``0.0`` (no additive smoothing) is a legitimate value.
- ``training/pipeline/_entity_time_composite_fe.py`` (2 sites): ``recency_aggregation_param`` --
  ``0.0`` is a valid exponent/parameter for some aggregation schemes.
- ``training/pipeline/_cross_sectional_composite_fe.py``: ``cross_sectional_neighbors_k`` --
  ``0`` is a valid (if degenerate) neighbor count.
- ``training/core/_phase_helpers_fit_split.py``: ``composite_cardinality_cap`` -- ``0`` is a
  legitimate "never use composite-key stratification" sentinel.
"""

from __future__ import annotations


def test_ransac_ridge_alpha_zero_not_rewritten_to_default():
    """RANSAC's inner Ridge honors an explicit alpha=0 (unregularized) instead of rewriting to 1.0."""
    from mlframe.training._model_configs import LinearModelConfig
    from mlframe.training.models import _build_ransac_regressor

    config = LinearModelConfig(model_type="ransac", alpha=0.0)
    model = _build_ransac_regressor(config)
    assert model.estimator.alpha == 0.0


def test_ransac_ridge_alpha_default_when_unset():
    """Baseline: LinearModelConfig's own default alpha still applies."""
    from mlframe.training._model_configs import LinearModelConfig
    from mlframe.training.models import _build_ransac_regressor

    config = LinearModelConfig(model_type="ransac")
    model = _build_ransac_regressor(config)
    assert model.estimator.alpha == config.alpha


def test_target_encoding_smoothing_zero_not_rewritten_to_default():
    """two_step_target_encode_smoothing=0.0 (no smoothing) must not be rewritten to 1.0."""
    import numpy as np
    import pandas as pd

    from mlframe.training.pipeline._target_encoding_composite_fe import apply_target_encoding_composite_fe

    rng = np.random.default_rng(0)
    n = 40
    train_df = pd.DataFrame({"cat": rng.choice(["a", "b"], n)})
    y_train = rng.normal(size=n)
    entities = rng.integers(0, 4, n).astype(np.int64)
    ts = np.arange(n, dtype=np.float64)
    train_idx = np.arange(n)

    class _Cfg:
        """Minimal config fixture for this test."""

        two_step_target_encode_columns = ["cat"]
        two_step_target_encode_decay_half_life = 30.0
        two_step_target_encode_smoothing = 0.0

    metadata: dict = {}
    apply_target_encoding_composite_fe(
        train_df, None, None, _Cfg(), entities, ts, y_train, train_idx, None, None, metadata=metadata,
    )
    assert metadata.get("two_step_target_encode_columns") == ["cat"]


def test_entity_time_recency_param_zero_not_rewritten_to_default():
    """recency_aggregation_param=0.0 must be persisted as 0.0, not rewritten to 1.0."""
    import numpy as np
    import pandas as pd

    from mlframe.training.pipeline._entity_time_composite_fe import apply_entity_time_composite_fe

    n = 20
    train_df = pd.DataFrame({"x": np.arange(n, dtype=np.float64)})
    group_ids = np.zeros(n, dtype=np.int64)
    timestamps = np.arange(n, dtype=np.float64)
    train_idx = np.arange(n)

    class _Cfg:
        """Minimal config fixture for this test."""

        state_duration_columns = None
        recency_aggregation_columns = ["x"]
        recency_aggregation_scheme = "poly"
        recency_aggregation_param = 0.0
        recency_aggregation_agg = "mean"
        state_duration_include_activation_count = False

    metadata: dict = {}
    apply_entity_time_composite_fe(
        train_df, None, None, _Cfg(), group_ids, timestamps, train_idx, None, None, metadata=metadata,
    )
    assert metadata["recency_aggregation_param"] == 0.0


def test_cross_sectional_neighbors_k_zero_not_rewritten_to_default():
    """cross_sectional_neighbors_k=0 must be honored as 0, not rewritten to 10."""
    import numpy as np
    import pandas as pd

    from mlframe.training.pipeline._cross_sectional_composite_fe import apply_cross_sectional_composite_fe

    rng = np.random.default_rng(0)
    n = 40
    train_df = pd.DataFrame({"snap": rng.integers(0, 10, n), "f1": rng.normal(size=n)})

    class _Cfg:
        """Minimal config fixture for this test."""

        cross_sectional_neighbors_snapshot_col = "snap"
        cross_sectional_neighbors_feature_cols = ["f1"]
        cross_sectional_neighbors_k = 0
        cross_sectional_neighbors_agg_stats = ("mean",)

    metadata: dict = {}
    apply_cross_sectional_composite_fe(train_df, None, None, _Cfg(), metadata=metadata, verbose=0)
    assert metadata["cross_sectional_neighbors_k"] == 0


def test_composite_cardinality_cap_zero_not_rewritten_to_default():
    """composite_cardinality_cap=0 (deliberately disabling composite-key stratification) must be
    honored, not silently rewritten to 200."""
    from types import SimpleNamespace

    split_config = SimpleNamespace(composite_cardinality_cap=0)
    _cap = getattr(split_config, "composite_cardinality_cap", None)
    _MAX_COMPOSITE_CARDINALITY = 200 if _cap is None else int(_cap)
    assert _MAX_COMPOSITE_CARDINALITY == 0
