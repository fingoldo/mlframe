"""
Training helper functions and callback classes.

This module contains helper utilities:
- parse_catboost_devices: GPU device parsing for CatBoost
- get_training_configs: Training configuration factory
- get_trainset_features_stats: Compute training set statistics (pandas)
- get_trainset_features_stats_polars: Compute training set statistics (polars)
- UniversalCallback: Base callback class for training monitoring
- LightGBMCallback, XGBoostCallback, CatBoostCallback: Model-specific callbacks
"""

from __future__ import annotations


import logging
import psutil
from dataclasses import dataclass
from timeit import default_timer as timer
from types import SimpleNamespace
from typing import Optional, Dict, List, Callable, Sequence, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

# NOTE: torch + mlframe.lightninglib are imported lazily inside `get_training_configs`
# (only needed for MLP configs). Top-level import cost ~2-3s — avoided for CB/LGB/XGB-only runs.
import lightgbm as lgb

import xgboost as xgb
from xgboost.callback import TrainingCallback

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from pyutilz.system import get_gpuinfo_gpu_info, tqdmu, get_own_memory_usage
from pyutilz.pythonlib import get_parent_func_args, store_params_in_object
from ._gpu_probe import CUDA_IS_AVAILABLE, LGB_GPU_AVAILABLE, XGB_GPU_AVAILABLE
from mlframe.metrics.core import (
    compute_probabilistic_multiclass_error,
    robust_mlperf_metric,
    ICE,
)

from .utils import get_numeric_columns, get_categorical_columns
# `_probe_xgb_gpu_support` / `_probe_lgb_gpu_support` re-export was dropped:
# they are private to `_gpu_probe.py` (only used to compute the module-level
# XGB_GPU_AVAILABLE / LGB_GPU_AVAILABLE booleans imported above), and no other
# module ever imported them via this re-export. Removing closes the "noqa F401
# on private name" anti-pattern flagged in the Wave-3 audit.
from ._classif_helpers import (  # noqa: E402,F401
    _canonical_predict_proba_shape,
    _predict_from_probs,
    _classif_objective_kwargs,
    _maybe_wrap_multilabel,
    _compute_chain_orders,
    _ChainEnsemble,
    _build_classifier_chain_ensemble,
)  # _build_classifier_chain_ensemble kept in helpers.py (line 86)
from .callbacks import (  # noqa: E402,F401
    UniversalCallback,
    LightGBMCallback,
    XGBoostCallback,
    CatBoostCallback,
)

logger = logging.getLogger(__name__)


# Constant - CUDA availability
try:
    from numba.cuda import is_available as is_cuda_available

    CUDA_IS_AVAILABLE = is_cuda_available()
except (ImportError, AttributeError, ModuleNotFoundError):
    CUDA_IS_AVAILABLE = False


# Per-library GPU support gating. ``CUDA_IS_AVAILABLE`` (numba probe)
# tells us only whether the system has a usable CUDA device; it does NOT
# tell us whether the installed XGBoost / LightGBM binaries were COMPILED
# with GPU support. On Windows the default ``pip install xgboost`` ships
# the CPU-only wheel, so passing ``device='cuda'`` triggers ``WARNING:
# Device is changed from GPU to CPU as we couldn't find any available GPU
# on the system`` per-fit (custom 3rdParty XGB build can also report
# ``xgb.build_info() == {... 'USE_CUDA': False ...}`` despite CUDA being
# available for CatBoost). Probe each binary's actual GPU support ONCE at
# module-import so the helpers never set device='cuda'

def parse_catboost_devices(devices: str, all_gpus: list = None) -> List[Dict]:
    """
    Parses a GPU devices string and returns a list of GPU info dicts
    corresponding to the specified device indices.

    Parameters
    ----------
    devices : str
        A string specifying device indices. Formats supported:
          - "0"             (single GPU)
          - "0:1:3"         (multiple GPUs)
          - "0-3"           (range of GPUs, inclusive)

    Returns
    -------
    list[dict]
        Filtered list of GPU info dictionaries.
    """

    if not all_gpus:
        all_gpus = get_gpuinfo_gpu_info()

    if not devices:
        return all_gpus

    # Parse the devices string
    device_indices = []
    try:
        if "-" in devices:  # range format
            parts = devices.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range format '{devices}'. Expected 'start-end' (e.g., '0-3')")
            start, end = parts
            start_int, end_int = int(start), int(end)
            if start_int > end_int:
                raise ValueError(f"Invalid range '{devices}': start ({start_int}) > end ({end_int})")
            device_indices = list(range(start_int, end_int + 1))
        elif ":" in devices:  # multiple specific GPUs
            device_indices = [int(x) for x in devices.split(":")]
        else:  # single GPU
            device_indices = [int(devices)]
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid device specification '{devices}'. Must contain integers only.") from e
        raise

    # Validate indices
    max_index = len(all_gpus) - 1
    invalid = [i for i in device_indices if i < 0 or i > max_index]
    if invalid:
        raise ValueError(f"Invalid GPU indices {invalid}. Available range: 0-{max_index}")

    # Filter GPU list
    filtered_gpus = [gpu for gpu in all_gpus if gpu["index"] in device_indices]
    return filtered_gpus


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Training Configuration Factory
# -----------------------------------------------------------------------------------------------------------------------------------------------------


# get_training_configs carved to ``_helpers_training_configs``; re-exported below.
from ._helpers_training_configs import get_training_configs  # noqa: F401



# -----------------------------------------------------------------------------------------------------------------------------------------------------
# CatBoost text-processing helper
# -----------------------------------------------------------------------------------------------------------------------------------------------------


# CB's default occurrence_lower_bound (CatBoost master, 2026). Words appearing
# in fewer rows than this are pruned from the TF-IDF dictionary. On small
# training sets the default is too aggressive and can either raise
# "Dictionary size is 0" or - in the C++ _train loop - HANG indefinitely
# while the empty dictionary stalls bag-of-words feature construction.
CB_DEFAULT_OCCURRENCE_LOWER_BOUND = 50

# Above this row count we keep CB's defaults; production-sized data needs
# the original 50-occurrence floor to keep dictionaries bounded. Below it
# we scale the floor down proportionally so RFECV inner CV folds and
# small-n training runs do not collapse the dictionary.
CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS = 1000

# Fraction of training rows used for the occurrence floor when scaling.
# 5% means: a word appearing in 5%+ of the fold survives the prune, even
# with as few as 40 rows (where it becomes 2 — the absolute minimum that
# still excludes truly singleton terms).
CB_TEXT_PROC_OCCURRENCE_FRACTION = 0.05
CB_TEXT_PROC_OCCURRENCE_FLOOR = 2


def compute_cb_text_processing(n_train_rows: int) -> Optional[dict]:
    """Return a CatBoost ``text_processing`` config that scales
    ``occurrence_lower_bound`` to the training set size, or ``None`` if
    CB's defaults are appropriate.

    CatBoost's default ``occurrence_lower_bound=50`` rejects any token
    that appears in fewer than 50 rows. On small training sets (RFECV
    inner CV folds, single-pass training with ``n_train_rows`` < ~1000),
    this default rejects the entire vocabulary and either:
      * raises ``"Dictionary size is 0"`` — handled by the existing
        fallback at ``trainer._train_model_with_fallback``; or
      * hangs inside CB's C++ ``_train`` loop waiting for an empty
        dictionary to materialise (fuzz c0056 / c0070).

    The fix is a row-proportional floor: a word that appears in at least
    5 % of the rows survives, clamped to >= 2 (so a single-occurrence
    word never builds a dictionary entry — that would inflate the
    artefact and cause zero generalisation).

    Args:
        n_train_rows: Number of rows in the *fit-time* training set. For
            RFECV this is the inner-fold train size, NOT the outer suite
            input size.

    Returns:
        ``text_processing`` dict suitable for ``CatBoost.set_params(
        text_processing=...)`` / ``CatBoost.__init__(text_processing=...)``,
        or ``None`` when defaults are fine (``n_train_rows`` >=
        ``CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS``, or invalid input).
    """
    if not isinstance(n_train_rows, int) or n_train_rows <= 0:
        return None
    if n_train_rows >= CB_TEXT_PROC_DEFAULT_THRESHOLD_ROWS:
        return None
    olb = max(
        CB_TEXT_PROC_OCCURRENCE_FLOOR,
        int(round(n_train_rows * CB_TEXT_PROC_OCCURRENCE_FRACTION)),
    )
    return {
        "tokenizers": [{"tokenizer_id": "Space", "delimiter": " "}],
        "dictionaries": [
            {
                "dictionary_id": "Word",
                "occurrence_lower_bound": str(olb),
                "max_dictionary_size": "50000",
                "gram_order": "1",
            }
        ],
        "feature_processing": {
            "default": [
                {
                    "tokenizers_names": ["Space"],
                    "dictionaries_names": ["Word"],
                    "feature_calcers": ["BoW"],
                }
            ]
        },
    }


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Wave 95 (2026-05-21): get_trainset_features_stats / _polars + the
# precompute bundle (TrainMlframeSuitePrecomputed, precompute_*) moved
# to sibling file _precompute.py to drop helpers.py below the 1k-line
# monolith threshold. Re-exported below so existing callers keep working.

from ._precompute import (  # noqa: F401, E402
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
    TrainMlframeSuitePrecomputed,
    precompute_composite_target_specs,
    precompute_dummy_baselines,
    precompute_trainset_features_stats,
    precompute_all,
)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Callback Classes for Training Monitoring
# -----------------------------------------------------------------------------------------------------------------------------------------------------

