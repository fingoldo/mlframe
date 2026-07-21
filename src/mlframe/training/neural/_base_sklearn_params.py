"""sklearn ``get_params`` / ``set_params`` for ``PytorchLightningEstimator``.

Carved out of ``neural/base.py`` (monolith-split, sibling re-export
pattern per CLAUDE.md) to keep the facade under the 1k-LOC budget. These
are bound as methods on ``PytorchLightningEstimator`` in ``base.py`` -- they
take ``self`` as the first argument and behave identically to the inline
methods they replaced.
"""
from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_params(self, deep: bool = True) -> Dict[str, Any]:
    """Returns a dictionary of all parameters for scikit-learn compatibility.

    All __init__ parameters must be included for sklearn.base.clone() to work correctly.
    """
    # Enumerated from the real __init__ signature (not a hand-maintained key list) so a newly-added
    # constructor parameter can never silently go missing here again -- a hand-maintained list previously
    # drifted 10 params stale (use_ema, ema_params, label_smoothing, focal_loss_gamma, focal_loss_alpha,
    # capture_iteration_metrics, random_state, class_weight, use_learnable_cat_embeddings,
    # categorical_embed_dim), so sklearn.base.clone() (used by cross_val_score / GridSearchCV /
    # StackingClassifier / any Pipeline step) silently dropped those params back to their constructor
    # defaults on every clone.
    sig = inspect.signature(type(self).__init__)
    params: Dict[str, Any] = {}
    for name in sig.parameters:
        if name == "self":
            continue
        value = getattr(self, name)
        if deep and isinstance(value, dict):
            value = deepcopy(value)
        params[name] = value
    return params


def set_params(self, **params: Any):
    """Sets parameters for scikit-learn compatibility."""
    for key, value in params.items():
        if key in ("model_params", "datamodule_params"):
            setattr(self, key, deepcopy(value))  # deep copy nested dicts
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            logger.warning("Parameter %s not found in %s", key, self.__class__.__name__)
    return self
