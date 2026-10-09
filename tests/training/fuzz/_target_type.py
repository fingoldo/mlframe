"""Which ``TargetTypes`` member the features-and-targets extractor must be told for a fuzz combo.

``SimpleFeaturesAndTargetsExtractor(regression=...)`` is a boolean, so a combo whose ``target_type`` is ``quantile_regression`` or
``multi_target_regression`` was handed ``regression=False`` by every suite that wrote ``regression=(combo.target_type == "regression")``
and the extractor then treated the continuous column as a binary classification target. LightGBM's classifier encoder was fitted on
a thousand distinct float labels and the fit failed with "y contains previously unseen labels". Passing the exact target type removes
the guess.
"""

from __future__ import annotations

from typing import Any

from mlframe.training.configs import TargetTypes

_BY_NAME = {
    "regression": TargetTypes.REGRESSION,
    "quantile_regression": TargetTypes.QUANTILE_REGRESSION,
    "binary_classification": TargetTypes.BINARY_CLASSIFICATION,
    "multiclass_classification": TargetTypes.MULTICLASS_CLASSIFICATION,
    "multilabel_classification": TargetTypes.MULTILABEL_CLASSIFICATION,
    "learning_to_rank": TargetTypes.LEARNING_TO_RANK,
    "multi_target_regression": TargetTypes.MULTI_TARGET_REGRESSION,
}


def target_type_for_combo(combo: Any, target_col: str) -> TargetTypes:
    """The ``TargetTypes`` member for ``combo``, given the target column ``build_frame_for_combo`` actually emitted.

    A ``multi_target_regression`` combo whose models cannot all take a 2-D target is downgraded by the frame builder to a
    1-D ``target_reg`` column; at the data level that is plain regression, and the emitted column name is the authoritative signal.
    """
    name = combo.target_type
    if name == "multi_target_regression" and target_col != "target":
        name = "regression"
    return _BY_NAME[name]
