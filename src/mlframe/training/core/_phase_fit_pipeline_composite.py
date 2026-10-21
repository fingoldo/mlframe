"""Composite-FE stage helpers carved out of ``_phase_helpers_fit_pipeline``: supervised-target pick and the pre-encoding composite steps."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_helpers_fit_split")


def _composite_fe_supervised_target(target_by_type, metadata) -> Any:
    """The target whose labels supervise the suite-level composite-FE steps, recorded in ``metadata`` and logged.

    These steps are fitted once per suite, so they see ONE target: the first one ``target_by_type`` yields. In a
    multi-target suite the target-encoded / MI-grouped features every other target's model consumes are built from
    that target's labels, so which one it was must be visible - it shifts with the extractor's target order.
    """
    names = [(str(t), str(n)) for t, d in target_by_type.items() for n in (d.keys() if hasattr(d, "keys") else [t])]
    for _t, _v in target_by_type.items():
        _name, _cand = next(iter(_v.items())) if hasattr(_v, "items") else (_t, _v)
        if _cand is not None:
            if isinstance(metadata, dict):
                metadata["composite_fe_supervised_target"] = {"target_type": str(_t), "target_name": str(_name)}
            if len(names) > 1:
                logger.info("Composite-FE steps are supervised by target %s/%s for all %d targets of this suite.", _t, _name, len(names))
            return _cand.to_numpy() if hasattr(_cand, "to_numpy") else np.asarray(_cand)
    return None


def _run_composite_fe_stages(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    preprocessing_extensions: Any,
    *,
    target_by_type: Any,
    metadata: dict,
    train_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
    group_ids: np.ndarray | None,
    timestamps: Any,
    auxiliary_events_df: Any,
    verbose: bool,
) -> tuple[Any, Any, Any, Any]:
    """Run the per-target and suite-level composite-FE steps; return ``(train_df, val_df, test_df, preprocessing_extensions)``."""
    # Shared train-only y extraction for the composite-FE steps below (categorical auto-group
    # discovery + two-step target encoding) that need a supervised signal. Cheap no-op when unused.
    # target_by_type carries the PRE-split full target -- slice to train_idx (mirrors the PySR
    # extraction below) so these steps never see val/test rows' targets.
    # Columns the composite-FE steps below add are replayed at predict; the input validator must accept them.
    _cols_before_composite = list(train_df.columns) if train_df is not None and hasattr(train_df, "columns") else []

    # More than one target: the label-supervised composite steps are fitted per target (``_per_target_supervised_fe``);
    # the suite-level blocks below then run only their unsupervised parts.
    from mlframe.training.pipeline.shared import apply_per_target_supervised_fe, iter_targets, supervised_steps_enabled

    _per_target_targets = iter_targets(target_by_type) if supervised_steps_enabled(preprocessing_extensions) and hasattr(target_by_type, "items") else []
    _per_target_mode = len(_per_target_targets) > 1
    if _per_target_mode:
        train_df, val_df, test_df = apply_per_target_supervised_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            _per_target_targets,
            train_idx,
            group_ids,
            timestamps,
            val_idx,
            test_idx,
            metadata,
            verbose=verbose,
        )
        _extensions_as_configured = preprocessing_extensions
        preprocessing_extensions = preprocessing_extensions.model_copy(
            update={"categorical_group_concat_auto_enabled": False, "two_step_target_encode_columns": None}
        )

    test_df, train_df, val_df = _phase_fit_pipeline_composite_none(
        _per_target_mode,
        target_by_type,
        metadata,
        train_idx,
        train_df,
        preprocessing_extensions,
        val_df,
        test_df,
        verbose,
        group_ids,
        timestamps,
        val_idx,
        test_idx,
        auxiliary_events_df,
    )

    if _per_target_mode:  # the shared blocks above are done; everything after (and the caller) sees the config as configured
        preprocessing_extensions = _extensions_as_configured
    if train_df is not None and hasattr(train_df, "columns"):
        _before = set(_cols_before_composite)
        metadata["composite_fe_emitted_columns"] = [c for c in train_df.columns if c not in _before]
    return train_df, val_df, test_df, preprocessing_extensions


def _phase_fit_pipeline_composite_none(
    _per_target_mode,
    target_by_type,
    metadata,
    train_idx,
    train_df,
    preprocessing_extensions,
    val_df,
    test_df,
    verbose,
    group_ids,
    timestamps,
    val_idx,
    test_idx,
    auxiliary_events_df,
):
    """Block of _phase_fit_pipeline starting at ``_y_for_composite = None``."""
    _y_for_composite = None
    _y_for_composite = _phase_fit_pipeline_per_target_mode_target(_per_target_mode, target_by_type, metadata, _y_for_composite)
    _y_for_composite = _phase_fit_pipeline_block(_y_for_composite, train_idx, train_df)

    # Categorical composite FE (powerset concat / auto MI-grouped concat) -- MUST run before
    # fit_and_transform_pipeline's categorical encoding: a composite column produced after encoding
    # would be dropped by apply_preprocessing_extensions' numeric-only gate before any downstream
    # step saw it. See PreprocessingExtensionsConfig's docstring.
    if preprocessing_extensions is not None and (
        getattr(preprocessing_extensions, "categorical_powerset_concat_enabled", False)
        or getattr(preprocessing_extensions, "categorical_group_concat_auto_enabled", False)
    ):
        from mlframe.training.pipeline.shared import apply_categorical_composite_fe

        train_df, val_df, test_df = apply_categorical_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            _y_for_composite,
            metadata,
            verbose=verbose,
        )

    # Entity/time-keyed FE (state_duration / recency_aggregation) -- same pre-encoding point, needs
    # group_ids (threaded through from the FeaturesAndTargetsExtractor's own group-key resolution).
    if preprocessing_extensions is not None and (
        getattr(preprocessing_extensions, "state_duration_columns", None) or getattr(preprocessing_extensions, "recency_aggregation_columns", None)
    ):
        from mlframe.training.pipeline.shared import apply_entity_time_composite_fe

        train_df, val_df, test_df = apply_entity_time_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            group_ids,
            timestamps,
            train_idx,
            val_idx,
            test_idx,
            metadata=metadata,
            verbose=verbose,
        )

    # Cross-sectional-neighbor FE -- column-declaration-driven (snapshot key is a frame column, not
    # a side-array), same pre-encoding point.
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "cross_sectional_neighbors_snapshot_col", None):
        from mlframe.training.pipeline.shared import apply_cross_sectional_composite_fe

        train_df, val_df, test_df = apply_cross_sectional_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            metadata=metadata,
            verbose=verbose,
        )

    # Two-step recency-weighted target encoding -- needs group_ids AND train-only y (real fit-time
    # state: a per-entity lookup table persisted onto metadata for val/test/predict).
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "two_step_target_encode_columns", None):
        from mlframe.training.pipeline.shared import apply_target_encoding_composite_fe

        train_df, val_df, test_df = apply_target_encoding_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            group_ids,
            timestamps,
            _y_for_composite,
            train_idx,
            val_idx,
            test_idx,
            metadata=metadata,
            verbose=verbose,
        )

    train_df, val_df, test_df = _phase_fit_pipeline_composite_none_tail(
        train_df,
        val_df,
        test_df,
        preprocessing_extensions,
        metadata=metadata,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        group_ids=group_ids,
        timestamps=timestamps,
        auxiliary_events_df=auxiliary_events_df,
        verbose=verbose,
    )
    return test_df, train_df, val_df


def _phase_fit_pipeline_composite_none_tail(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    preprocessing_extensions: Any,
    *,
    metadata: dict,
    train_idx: np.ndarray | None,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
    group_ids: np.ndarray | None,
    timestamps: Any,
    auxiliary_events_df: Any,
    verbose: bool,
) -> tuple[Any, Any, Any]:
    """Second half of the pre-encoding composite-FE steps (MA crossover, latent-interaction SVD, nearest-past join, event-proximity decay)."""
    # MA-crossover FE -- no rolling-MA step of its own in the suite; computed here from declared
    # numeric columns before feeding the underlying pairwise-crossover function.
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "ma_crossover_columns", None):
        from mlframe.training.pipeline.shared import apply_ma_crossover_composite_fe

        train_df, val_df, test_df = apply_ma_crossover_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            group_ids,
            timestamps,
            train_idx,
            val_idx,
            test_idx,
            metadata=metadata,
            verbose=verbose,
        )

    # SVD latent-interaction embeddings -- needs a SEPARATE auxiliary events table, not columns on
    # train/val/test. Real fit-time state (the fitted TF-IDF/SVD basis) persisted onto metadata.
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "latent_interaction_svd_row_entity", None):
        from mlframe.training.pipeline.shared import apply_latent_interaction_svd_composite_fe

        train_df, val_df, test_df = apply_latent_interaction_svd_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            auxiliary_events_df,
            group_ids,
            train_idx,
            val_idx,
            test_idx,
            metadata=metadata,
            verbose=verbose,
        )

    # Nearest-past as-of join -- needs the SAME SEPARATE auxiliary events table, no fit-time state
    # (the backward as-of match is inherently leak-safe by construction).
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "nearest_past_join_on", None):
        from mlframe.training.pipeline.shared import apply_nearest_past_join_composite_fe

        train_df, val_df, test_df = apply_nearest_past_join_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            auxiliary_events_df,
            metadata=metadata,
            verbose=verbose,
        )

    # Event-proximity decay -- caller-supplied literal event dates (domain knowledge); dates come
    # from timestamps (captured before the earlier datetime-column-decomposition phase), not a
    # train/val/test column.
    if preprocessing_extensions is not None and getattr(preprocessing_extensions, "event_proximity_decay_event_dates", None):
        from mlframe.training.pipeline.shared import apply_event_proximity_decay_composite_fe

        train_df, val_df, test_df = apply_event_proximity_decay_composite_fe(
            train_df,
            val_df,
            test_df,
            preprocessing_extensions,
            timestamps,
            train_idx,
            val_idx,
            test_idx,
            metadata=metadata,
            verbose=verbose,
        )
    return train_df, val_df, test_df


def _phase_fit_pipeline_per_target_mode_target(_per_target_mode, target_by_type, metadata, _y_for_composite):
    """Block of _phase_fit_pipeline starting at ``if not _per_target_mode and target_by_type is not None and hasattr(tar``."""
    if not _per_target_mode and target_by_type is not None and hasattr(target_by_type, "items"):
        try:
            _y_for_composite = _composite_fe_supervised_target(target_by_type, metadata)
        except Exception as e:
            logger.warning("y_for_composite extraction failed: %s", e)
            _y_for_composite = None
    return _y_for_composite


def _phase_fit_pipeline_block(_y_for_composite, train_idx, train_df):
    """Block of _phase_fit_pipeline starting at ``if (``."""
    if (
        _y_for_composite is not None
        and train_idx is not None
        and train_df is not None
        and hasattr(train_df, "shape")
        and len(_y_for_composite) != train_df.shape[0]
    ):
        try:
            _idx_arr_composite = np.asarray(train_idx)
            if train_df is not None and len(_idx_arr_composite) == train_df.shape[0] and int(_idx_arr_composite.max()) < len(_y_for_composite):
                _y_for_composite = _y_for_composite[_idx_arr_composite]
            else:
                _y_for_composite = None
        except Exception as e:
            logger.warning("y_for_composite index-alignment failed: %s", e)
            _y_for_composite = None
    return _y_for_composite
