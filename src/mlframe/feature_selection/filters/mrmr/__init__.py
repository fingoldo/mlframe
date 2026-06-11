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
from .._mrmr_fingerprints import (  # noqa: E402,F401
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
    _MRMR_BATCH_PRECOMPUTE_MAX_K,
    _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
)

from numpy.polynomial.hermite import hermval  # noqa: F401
from scipy import special as sp  # noqa: F401
from scipy.stats import mode  # noqa: F401

from catboost import CatBoostClassifier  # noqa: F401

from pyutilz.numbalib import (  # noqa: F401
    generate_combinations_recursive_njit,
    python_dict_2_numba_dict,
    set_numba_random_seed,
)
from pyutilz.parallel import mem_map_array, parallel_run, split_list_into_chunks  # noqa: F401
from pyutilz.pythonlib import (  # noqa: F401
    get_parent_func_args,
    sort_dict_by_value,
    store_params_in_object,
)
from pyutilz.system import tqdmu  # noqa: F401

from mlframe.core.arrays import arrayMinMax  # noqa: F401
from mlframe.feature_selection.wrappers import RFECV  # noqa: F401
from mlframe.metrics.core import compute_probabilistic_multiclass_error  # noqa: F401
from mlframe.utils.misc import set_random_seed  # noqa: F401

from .._internals import (  # noqa: F401
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from .._numba_utils import arr2str, count_cand_nbins, unpack_and_sort  # noqa: F401
from ..discretization import (  # noqa: F401
    categorize_dataset,
    discretize_array,
)
from ..feature_engineering import (  # noqa: F401
    FE_DEFAULT_SUBSAMPLE_N,
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from ..gpu import init_kernels, mi_direct_gpu  # noqa: F401
from ..info_theory import (  # noqa: F401
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from ..permutation import distribute_permutations, mi_direct, parallel_mi  # noqa: F401
from ..evaluation import (  # noqa: F401
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from ..fleuret import (  # noqa: F401
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from ..screen import postprocess_candidates, screen_predictors  # noqa: F401

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from ._mrmr_class import MRMR  # noqa: E402

# Pickle resolves a class via ``__module__`` + ``__qualname__``. The class object now lives in ``._mrmr_class``;
# stamp the package facade path here so the private submodule never leaks into a pickle. ``filters/__init__.py``
# subsequently rewrites it again to ``mlframe.feature_selection.filters`` -- the historical pickle-BC value, which
# Python always reaches first since parent packages import before submodules -- so the on-disk contract is
# unchanged by this split (old pickles keep loading against the same re-exported class object).
MRMR.__module__ = "mlframe.feature_selection.filters.mrmr"

# Bind the carved-out methods onto the class, preserving the original order. Each ``_*_func`` lives in a sibling
# module as a module-level function taking ``self`` first; the binding makes ``self.<method>(...)`` resolve to it.
from .._mrmr_fit_impl import _fit_impl as _fit_impl_func  # noqa: E402
MRMR._fit_impl = _fit_impl_func

from .._mrmr_fe_step import _run_fe_step as _run_fe_step_func  # noqa: E402
MRMR._run_fe_step = _run_fe_step_func

from .._mrmr_validate_transform import (  # noqa: E402
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

from .._mrmr_partial_fit import partial_fit as _partial_fit_func  # noqa: E402
MRMR.partial_fit = _partial_fit_func

from .._mrmr_fe_provenance import get_fe_report as _get_fe_report_func  # noqa: E402
MRMR.get_fe_report = _get_fe_report_func

# W2 provenance self-audit accessor: surviving engineered recipe.kinds that
# resolved to ``engineered_unknown`` (deliberate ``factorize`` on a clean fit;
# anything else is an unregistered FE family).
from .._mrmr_fe_provenance import get_unlabeled_recipe_kinds as _get_unlabeled_recipe_kinds_func  # noqa: E402
MRMR.get_unlabeled_recipe_kinds = _get_unlabeled_recipe_kinds_func

# Per-gate FE rejection ledger accessor (the rejection side of get_fe_report).
from .._fe_rejection_ledger import get_fe_rejection_report as _get_fe_rejection_report_func  # noqa: E402
MRMR.get_fe_rejection_report = _get_fe_rejection_report_func

# One-call human-readable explanation: assembles fe_provenance_ (survivors) + fe_rejection_ledger_
# (binding gate) + _fe_recommended_flags_ (Layer-99 chosen flags) into a one-screen narrative.
from .._mrmr_explain import explain_selection as _explain_selection_func  # noqa: E402
MRMR.explain_selection = _explain_selection_func

# Semi-supervised fit helper -- importable from the ``mrmr`` namespace so callers can
# ``from mlframe.feature_selection.filters.mrmr import fit_with_unlabeled`` without reaching into the sibling path.
from .._semi_supervised_fe import fit_with_unlabeled  # noqa: E402,F401

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
    "_MRMR_BATCH_PRECOMPUTE_MAX_K",
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
