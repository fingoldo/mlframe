"""Replay of the suite-owned feature engineering on a predict frame, in training order."""

from __future__ import annotations

from typing import Any


def replay_composite_fe_chain(
    df: Any,
    metadata: dict,
    *,
    group_ids: Any,
    timestamps: Any,
    auxiliary_events_df: Any,
    verbose: Any,
) -> Any:
    """Replay the datetime decomposition and every composite FE family on *df*, in the order training applied them."""
    from .predict import _replay_suite_datetime_decomposition
    from mlframe.training.pipeline.shared import replay_categorical_composite_fe
    from mlframe.training.pipeline.shared import replay_entity_time_composite_fe
    from mlframe.training.pipeline.shared import replay_cross_sectional_composite_fe
    from mlframe.training.pipeline.shared import replay_target_encoding_composite_fe
    from mlframe.training.pipeline.shared import replay_per_target_supervised_fe
    from mlframe.training.pipeline.shared import replay_ma_crossover_composite_fe
    from mlframe.training.pipeline.shared import replay_latent_interaction_svd_composite_fe
    from mlframe.training.pipeline.shared import replay_nearest_past_join_composite_fe
    from mlframe.training.pipeline.shared import replay_event_proximity_decay_composite_fe

    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)
    df = replay_categorical_composite_fe(df, metadata, verbose=verbose)
    df = replay_entity_time_composite_fe(df, metadata, group_ids, timestamps, verbose=verbose)
    df = replay_cross_sectional_composite_fe(df, metadata, verbose=verbose)
    df = replay_target_encoding_composite_fe(df, metadata, group_ids, verbose=verbose)
    df = replay_per_target_supervised_fe(df, metadata, group_ids, verbose=verbose)
    df = replay_ma_crossover_composite_fe(df, metadata, group_ids, timestamps, verbose=verbose)
    df = replay_latent_interaction_svd_composite_fe(df, metadata, auxiliary_events_df, group_ids, verbose=verbose)
    df = replay_nearest_past_join_composite_fe(df, metadata, auxiliary_events_df, verbose=verbose)
    df = replay_event_proximity_decay_composite_fe(df, metadata, timestamps, verbose=verbose)
    return df
