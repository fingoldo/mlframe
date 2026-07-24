"""``mlframe.feature_selection.filters.mrmr`` -- the MRMR estimator package facade.

The ``MRMR`` class body moved verbatim into ``._mrmr_class`` (the irreducible single class, LOC-exempt); this
facade re-exports it and the FULL historical public surface so every ``from mlframe.feature_selection.filters
.mrmr import X`` (the ubiquitous ``import MRMR`` plus the in-package lazy ``from .mrmr import (...)`` /
``from ..mrmr import (...)`` blocks inside ``_mrmr_fit_impl`` / ``_mrmr_fe_step`` / ``_mrmr_validate_transform``,
and the histogram / fingerprint / hash / signature helpers re-imported from ``_mrmr_fingerprints``) keeps
resolving unchanged.

``MRMR.__module__`` is rewritten to this package path so pickle of a fitted MRMR resolves the class via the stable
facade (old pickles already reference this path), and the method bindings that used to live at the bottom of the
monolith (``MRMR._fit_impl`` / ``MRMR._run_fe_step`` / ``MRMR.partial_fit`` / ``MRMR.get_fe_report`` /
``MRMR._validate_*`` / ``MRMR._append_engineered``) run here in their original order so ``self.<method>(...)`` call
sites are untouched. ``_FIT_CACHE`` is a class attribute defined on ``MRMR`` in ``._mrmr_class`` -- the same class
object is re-exported here, so the process-wide fit cache, ``isinstance`` checks, and any monkey-patch setters all
operate on one identity.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import logging
import math
import os
import psutil
import textwrap
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations, islice
from os.path import exists
from timeit import default_timer as timer
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import numba
from numba import njit, jit
from numba.core import types
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, is_regressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

# Top-level helpers (histogram + fingerprint/hash + replay + chunker) live in
# ``_mrmr_fingerprints.py``; re-imported here so downstream callers continue to
# resolve the historical ``mrmr.<name>`` namespace.
from .._mrmr_fingerprints import (
    _astropy_histogram,
    histogram,
    _canonicalise_dtype_str,
    _mrmr_compute_y_fingerprint_sample,
    _mrmr_compute_x_fingerprint,
    _mrmr_y_corr_sample,
    _mrmr_y_corr,
    _hashable_params_signature,
    _content_array_signature,
    _target_to_numpy_values,
    _target_name_signature,
    _full_y_content_hash,
    _full_x_content_hash,
    _replay_fitted_state,
    _lazy_chunks,
    _MRMR_IDENTITY_FP_CACHE,
    _MRMR_IDENTITY_FP_LOCK,
    _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
)

from numpy.polynomial.hermite import hermval
from scipy import special as sp
from scipy.stats import mode

from catboost import CatBoostClassifier

from pyutilz.numbalib import (
    generate_combinations_recursive_njit,
    python_dict_2_numba_dict,
    set_numba_random_seed,
)
from pyutilz.parallel import mem_map_array, parallel_run, split_list_into_chunks
from pyutilz.pythonlib import (
    get_parent_func_args,
    sort_dict_by_value,
    store_params_in_object,
)
from pyutilz.system import tqdmu

from mlframe.core.arrays import arrayMinMax
from mlframe.feature_selection.wrappers import RFECV
from mlframe.metrics.core import compute_probabilistic_multiclass_error
from mlframe.utils.misc import set_random_seed

from .._internals import (
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from .._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from ..discretization import (
    categorize_dataset,
    discretize_array,
)
from ..feature_engineering import (
    UNIFIED_FE_SUBSAMPLE_N,
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from ..gpu import init_kernels, mi_direct_gpu
from ..info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from ..permutation import distribute_permutations, mi_direct, parallel_mi
from ..evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from ..fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from ..screen import postprocess_candidates, screen_predictors

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from ._mrmr_class import MRMR

# Pickle resolves a class via ``__module__`` + ``__qualname__``. The class object now lives in ``._mrmr_class``;
# stamp the package facade path here so the private submodule never leaks into a pickle. ``filters/__init__.py``
# subsequently rewrites it again to ``mlframe.feature_selection.filters`` -- the historical pickle-BC value, which
# Python always reaches first since parent packages import before submodules -- so the on-disk contract is
# unchanged by this split (old pickles keep loading against the same re-exported class object).
MRMR.__module__ = "mlframe.feature_selection.filters.mrmr"

# Bind the carved-out methods onto the class, preserving the original order. Each ``_*_func`` lives in a sibling
# module as a module-level function taking ``self`` first; the binding makes ``self.<method>(...)`` resolve to it.
from .._mrmr_fit_impl import _fit_impl as _fit_impl_func
MRMR._fit_impl = _fit_impl_func

from .._mrmr_fe_step import _run_fe_step as _run_fe_step_func
MRMR._run_fe_step = _run_fe_step_func

# Gate-A SIS front-screen application (I/O + column subsetting around the standalone ``sis_screen`` kernel),
# carved verbatim out of the class body into a filters-level sibling. ``self._apply_sis_screen(X, y)`` call site
# in ``fit`` is unchanged; selection is byte-for-byte identical.
from .._mrmr_sis_apply import _apply_sis_screen as _apply_sis_screen_func
MRMR._apply_sis_screen = _apply_sis_screen_func

from .._mrmr_validate_transform import (
    _validate_string_params as _validate_string_params_func,
    _validate_inputs as _validate_inputs_func,
    transform as _transform_func,
    _append_engineered as _append_engineered_func,
)
MRMR._validate_string_params = _validate_string_params_func
MRMR._validate_inputs = _validate_inputs_func
# ``transform`` is defined on the class body (as a thin delegator) so ``_SetOutputMixin.__init_subclass__`` wraps it
# correctly. Do NOT late-rebind ``MRMR.transform`` here -- that strips the wrapper and silently breaks
# ``set_output(transform='pandas')`` for direct ndarray-input calls.
MRMR._append_engineered = _append_engineered_func

from .._mrmr_partial_fit import partial_fit as _partial_fit_func
MRMR.partial_fit = _partial_fit_func

from .._mrmr_fe_provenance import get_fe_report as _get_fe_report_func
MRMR.get_fe_report = _get_fe_report_func

# W2 provenance self-audit accessor: surviving engineered recipe.kinds that
# resolved to ``engineered_unknown`` (deliberate ``factorize`` on a clean fit;
# anything else is an unregistered FE family).
from .._mrmr_fe_provenance import get_unlabeled_recipe_kinds as _get_unlabeled_recipe_kinds_func
MRMR.get_unlabeled_recipe_kinds = _get_unlabeled_recipe_kinds_func

# Per-gate FE rejection ledger accessor (the rejection side of get_fe_report).
from .._fe_rejection_ledger import get_fe_rejection_report as _get_fe_rejection_report_func
MRMR.get_fe_rejection_report = _get_fe_rejection_report_func

# One-call SELECTION-STABILITY / CONFIDENCE report (W3): per-feature selection-frequency +
# per-recipe survival-frequency, computed by REPLAY of the cheap MI screen on K bootstrap
# resamples of the stored binned screening matrix -- no MRMR refit (the #15 replay-not-refit trick).
from .._mrmr_stability_report import selection_stability_report as _selection_stability_report_func
MRMR.selection_stability_report = _selection_stability_report_func

# One-call human-readable explanation: assembles fe_provenance_ (survivors) + fe_rejection_ledger_
# (binding gate) + _fe_recommended_flags_ (Layer-99 chosen flags) into a one-screen narrative.
from .._mrmr_explain import explain_selection as _explain_selection_func
MRMR.explain_selection = _explain_selection_func

# ``set_params`` override bound directly onto the class -- not
# defined on a mixin -- because ``BaseEstimator`` (which already defines ``set_params``) sits BEFORE the
# config mixins in ``MRMR``'s MRO, so a mixin-level override would never be reached. See
# ``_mrmr_config_dataclasses.mrmr_set_params`` for the config-invalidation mechanism.
from ._mrmr_config_dataclasses import mrmr_set_params as _mrmr_set_params_func
MRMR.set_params = _mrmr_set_params_func

# Semi-supervised fit helper -- importable from the ``mrmr`` namespace so callers can
# ``from mlframe.feature_selection.filters.mrmr import fit_with_unlabeled`` without reaching into the sibling path.
from .._semi_supervised_fe import fit_with_unlabeled

__all__ = [
    "MRMR",
    "fit_with_unlabeled",
    "histogram",
    "_astropy_histogram",
    "_canonicalise_dtype_str",
    "_mrmr_compute_y_fingerprint_sample",
    "_mrmr_compute_x_fingerprint",
    "_mrmr_y_corr_sample",
    "_mrmr_y_corr",
    "_hashable_params_signature",
    "_content_array_signature",
    "_target_to_numpy_values",
    "_target_name_signature",
    "_full_y_content_hash",
    "_full_x_content_hash",
    "_replay_fitted_state",
    "_lazy_chunks",
    "_MRMR_IDENTITY_FP_CACHE",
    "_MRMR_IDENTITY_FP_LOCK",
    "_MRMR_BATCH_PRECOMPUTE_MIN_PAIRS",
    "RFECV",
    "CatBoostClassifier",
    "compute_probabilistic_multiclass_error",
    "categorize_dataset",
    "discretize_array",
    "check_prospective_fe_pairs",
    "compute_pairs_mis",
    "create_binary_transformations",
    "create_unary_transformations",
    "get_existing_feature_name",
    "get_new_feature_name",
    "screen_predictors",
    "postprocess_candidates",
    "parallel_run",
    "sort_dict_by_value",
    "tqdmu",
    "ENSURE_ARROW_DF_SUPPORT",
]
