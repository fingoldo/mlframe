"""Cross-package API of ``mlframe.training.pipeline``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.pipeline`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _PRE_PIPELINE_CACHE as PRE_PIPELINE_CACHE,
    _PRE_PIPELINE_CACHE_LOCK as PRE_PIPELINE_CACHE_LOCK,
    _PRE_PIPELINE_CACHE_MAX as PRE_PIPELINE_CACHE_MAX,
)
from ._categorical_composite_fe import (
    apply_categorical_composite_fe,
    replay_categorical_composite_fe,
)
from ._cross_sectional_composite_fe import (
    apply_cross_sectional_composite_fe,
    replay_cross_sectional_composite_fe,
)
from ._entity_time_composite_fe import (
    apply_entity_time_composite_fe,
    replay_entity_time_composite_fe,
)
from ._event_proximity_decay_composite_fe import (
    apply_event_proximity_decay_composite_fe,
    replay_event_proximity_decay_composite_fe,
)
from ._latent_interaction_svd_composite_fe import (
    apply_latent_interaction_svd_composite_fe,
    replay_latent_interaction_svd_composite_fe,
)
from ._ma_crossover_composite_fe import (
    apply_ma_crossover_composite_fe,
    replay_ma_crossover_composite_fe,
)
from ._nearest_past_join_composite_fe import (
    apply_nearest_past_join_composite_fe,
    replay_nearest_past_join_composite_fe,
)
from ._per_target_supervised_fe import (
    apply_per_target_supervised_fe,
    iter_targets,
    replay_per_target_supervised_fe,
    supervised_steps_enabled,
    target_scoped_frames,
)
from ._pipeline_cache import (
    _content_fingerprint_for_cache as content_fingerprint_for_cache,
    _full_target_content_hash as full_target_content_hash,
    _pipeline_signature_for_cache as pipeline_signature_for_cache,
    _pre_pipeline_cache_clear as pre_pipeline_cache_clear,
    _pre_pipeline_cache_get as pre_pipeline_cache_get,
    _pre_pipeline_cache_set as pre_pipeline_cache_set,
)
from ._pipeline_helpers import (
    _extract_feature_selector as extract_feature_selector,
    _is_fitted as is_fitted,
    _multilabel_target_to_1d_for_supervised_encoders as multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform as passthrough_cols_fit_transform,
    _prepare_test_split as prepare_test_split,
    _selector_output_columns as selector_output_columns,
)
from ._pipeline_helpers_apply import (
    _apply_pre_pipeline_transforms as apply_pre_pipeline_transforms,
)
from ._target_encoding_composite_fe import (
    apply_target_encoding_composite_fe,
    replay_target_encoding_composite_fe,
)
from . import (
    target_label_changed,
)

__all__ = [
    "target_label_changed",
    "PRE_PIPELINE_CACHE",
    "PRE_PIPELINE_CACHE_LOCK",
    "PRE_PIPELINE_CACHE_MAX",
    "apply_categorical_composite_fe",
    "apply_cross_sectional_composite_fe",
    "apply_entity_time_composite_fe",
    "apply_event_proximity_decay_composite_fe",
    "apply_latent_interaction_svd_composite_fe",
    "apply_ma_crossover_composite_fe",
    "apply_nearest_past_join_composite_fe",
    "apply_per_target_supervised_fe",
    "apply_pre_pipeline_transforms",
    "apply_target_encoding_composite_fe",
    "content_fingerprint_for_cache",
    "extract_feature_selector",
    "full_target_content_hash",
    "is_fitted",
    "iter_targets",
    "multilabel_target_to_1d_for_supervised_encoders",
    "passthrough_cols_fit_transform",
    "pipeline_signature_for_cache",
    "pre_pipeline_cache_clear",
    "pre_pipeline_cache_get",
    "pre_pipeline_cache_set",
    "prepare_test_split",
    "replay_categorical_composite_fe",
    "replay_cross_sectional_composite_fe",
    "replay_entity_time_composite_fe",
    "replay_event_proximity_decay_composite_fe",
    "replay_latent_interaction_svd_composite_fe",
    "replay_ma_crossover_composite_fe",
    "replay_nearest_past_join_composite_fe",
    "replay_per_target_supervised_fe",
    "replay_target_encoding_composite_fe",
    "selector_output_columns",
    "supervised_steps_enabled",
    "target_scoped_frames",
]
