"""sklearn ``get_params`` / ``set_params`` for ``PytorchLightningEstimator``.

Carved out of ``neural/base.py`` (monolith-split, sibling re-export
pattern per CLAUDE.md) to keep the facade under the 1k-LOC budget. These
are bound as methods on ``PytorchLightningEstimator`` in ``base.py`` -- they
take ``self`` as the first argument and behave identically to the inline
methods they replaced.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_params(self, deep: bool = True) -> Dict[str, Any]:
    """Returns a dictionary of all parameters for scikit-learn compatibility.

    All __init__ parameters must be included for sklearn.base.clone() to work correctly.
    """
    # Wave 26 P1 fix (2026-05-20): pre-fix ``trainer_params`` and
    # ``tune_params`` were returned by reference even with deep=True;
    # the four sibling param-dicts (model_params, network_params,
    # datamodule_params, swa_params) were correctly deepcopied. This
    # asymmetry was an oversight: sklearn's clone() calls
    # get_params(deep=True) and rebinds into a new instance. Any
    # downstream mutation of the clone's trainer_params (e.g. setting
    # a new logger) poisoned the original estimator that was still
    # being trained.
    params = {
        "model_class": self.model_class,
        "model_params": deepcopy(self.model_params) if deep else self.model_params,
        "network_params": deepcopy(self.network_params) if deep else self.network_params,
        "datamodule_class": self.datamodule_class,
        "datamodule_params": deepcopy(self.datamodule_params) if deep else self.datamodule_params,
        "trainer_params": deepcopy(self.trainer_params) if deep else self.trainer_params,
        "use_swa": self.use_swa,
        "swa_params": deepcopy(self.swa_params) if deep and self.swa_params else self.swa_params,
        "tune_params": deepcopy(self.tune_params) if deep and self.tune_params else self.tune_params,
        "tune_batch_size": self.tune_batch_size,
        "float32_matmul_precision": self.float32_matmul_precision,
        "early_stopping_rounds": self.early_stopping_rounds,
    }
    return params


def set_params(self, **params: Any):
    """Sets parameters for scikit-learn compatibility."""
    for key, value in params.items():
        if key in ("model_params", "datamodule_params"):
            setattr(self, key, deepcopy(value))  # deep copy nested dicts
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            logger.warning(f"Parameter {key} not found in {self.__class__.__name__}")
    return self
