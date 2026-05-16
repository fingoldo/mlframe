"""sklearn-compatible MRMR estimator.

Tolerates FE-engineered feature names in the post-fit support index map (routes synthetic names through
``_engineered_features_``). Includes an explicit input-validation contract in ``_validate_inputs`` and an
``__setstate__`` shim that injects defaults for newer kwargs / attributes so old joblib / cloudpickle pipelines
unpickle cleanly.
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
from typing import Sequence

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

from astropy.stats import histogram
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

from ._internals import (
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .discretization import (
    categorize_dataset,
    discretize_array,
)
from .feature_engineering import (
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from .gpu import init_kernels, mi_direct_gpu
from .info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from .permutation import distribute_permutations, mi_direct, parallel_mi
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .screen import postprocess_candidates, screen_predictors

logger = logging.getLogger(__name__)


def _hashable_params_signature(params: dict) -> tuple:
    """Build a hashable tuple-of-(key, value) signature from ``get_params`` output.

    Non-hashable values (numpy arrays, callables, dicts of unhashables) fall
    back to content-based hashing. Used by ``MRMR._FIT_CACHE`` so two clones
    with the same constructor parameters and (X, y) share fitted state.

    CACHE-Low-2: numpy arrays are content-hashed via
    ``(arr.tobytes(), arr.shape, str(arr.dtype))`` instead of ``repr(v)``;
    ``repr`` is array-summary on numpy >= 2 (e.g. ``array([1, ..., 4])``)
    so two arrays with identical content but different sizes / abbreviation
    behaviour produced different signatures before, defeating the cache.
    """
    items = []
    for k, v in sorted(params.items()):
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            # CACHE-Low-2: content-hash numpy arrays so a copy hashes equal
            # to the original. ``np.ndarray.tobytes`` + shape + dtype is the
            # cheapest exact fingerprint and works for all numpy versions.
            if isinstance(v, np.ndarray):
                try:
                    items.append(
                        (k, (v.tobytes(), v.shape, str(v.dtype)))
                    )
                    continue
                except Exception:
                    pass
            try:
                items.append((k, repr(v)))
            except Exception:
                items.append((k, id(v)))
    return tuple(items)


def _content_array_signature(arr) -> tuple:
    """Cheap O(1) content-based fingerprint of an array / DataFrame.

    Used as a cache key when ``id()`` is unreliable: the training suite copies X between model iterations,
    so semantically-equal X have distinct ``id(X)`` but identical content. Samples 10 evenly-spaced positions;
    column names (when available) are included so ``df`` vs ``df.rename(...)`` produce different keys (otherwise
    a fit on the renamed frame would replay the prior ``feature_names_in_`` and ``transform()`` would mis-select).

    Returns (shape, dtype_str, sampled_values_bytes, col_names); falls back to ``id(arr)`` on failure.
    """
    try:
        # Capture column names (DataFrames only) BEFORE unwrap so signature distinguishes df vs df.rename(...).
        col_names = None
        if hasattr(arr, "columns"):
            try:
                col_names = tuple(str(c) for c in arr.columns)
            except Exception:
                col_names = None
        # Unwrap to numpy
        if hasattr(arr, "to_numpy"):
            try:
                np_arr = arr.to_numpy()
            except Exception:
                return ("uncached", id(arr))
        elif hasattr(arr, "values"):
            np_arr = arr.values
        else:
            np_arr = arr
        if not hasattr(np_arr, "shape") or not hasattr(np_arr, "dtype"):
            return ("uncached", id(arr))
        shape = np_arr.shape
        dtype_str = str(np_arr.dtype)
        # Sample 10 positions: 0, 11%, 22%, ..., 100%.
        flat = np_arr.ravel()
        n = flat.size
        if n == 0:
            return (shape, dtype_str, b"", col_names)
        idx = [int(i * (n - 1) / 9) for i in range(10)] if n >= 10 else list(range(n))
        try:
            sampled = bytes(flat[idx].tobytes())
        except Exception:
            return ("uncached", id(arr))
        return (shape, dtype_str, sampled, col_names)
    except Exception:
        return ("uncached", id(arr))


def _target_to_numpy_values(y) -> np.ndarray:
    """Return a numpy view/array for sklearn-style targets.

    Pandas exposes both ``to_numpy`` and ``values``; Polars Series exposes ``to_numpy`` only. MRMR only needs
    the 1-D/2-D target vector for its temporary target columns, so normalize y directly without copying X.
    """
    if isinstance(y, np.ndarray):
        return y
    if hasattr(y, "to_numpy"):
        return y.to_numpy()
    if hasattr(y, "values"):
        return y.values
    return np.asarray(y)


def _target_name_signature(y) -> tuple:
    """Return a tuple of column names (or Series ``name``) for ``y`` if available.

    Two distinct targets with statistically-similar 10-cell samples (e.g. both balanced binary) must produce
    different cache keys; the target name (when the caller gave us a named Series / DataFrame) is the cheapest
    discriminator. Falls back to ``()`` when y is a bare ndarray.
    """
    try:
        if hasattr(y, "columns"):
            return tuple(str(c) for c in y.columns)
        name = getattr(y, "name", None)
        if name is not None:
            return (str(name),)
    except Exception:
        pass
    return ()


def _full_y_content_hash(y) -> str:
    """Return a hex digest covering the full content of ``y`` for cache-key disambiguation.

    The 10-sample ``_content_array_signature`` collides on targets whose sampled cells coincide (common for
    balanced binary classification). A full blake2b over y.tobytes() rules this out; the cost is O(len(y))
    bytes hashed, which is negligible next to the actual MRMR fit. Returns ``""`` on any conversion failure
    so the caller can choose to skip the cache.
    """
    try:
        arr = _target_to_numpy_values(y)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        # ascontiguousarray covers non-contiguous slices; tobytes itself would copy too but this is explicit.
        buf = np.ascontiguousarray(arr).tobytes()
        h = hashlib.blake2b(buf, digest_size=16)
        # Fold shape + dtype so reshape-only changes also bust the key.
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        return h.hexdigest()
    except Exception:
        return ""


# Constructor-parameter names of ``MRMR``. Populated lazily to avoid importing inspect at module load.
_MRMR_INIT_PARAM_NAMES: frozenset[str] | None = None


def _replay_fitted_state(target: MRMR, source: MRMR) -> int:
    """Copy fitted attributes from ``source`` onto ``target``, preserving target's constructor params intact.

    A cloned MRMR has its own constructor params + ``signature=None`` + no fitted state; we want to inherit
    ``support_``, ``_engineered_recipes_``, ``_cat_fe_cache_``, ``ranking_``, etc. WITHOUT overwriting any
    constructor params. Returns the number of attributes replayed.
    """
    global _MRMR_INIT_PARAM_NAMES
    if _MRMR_INIT_PARAM_NAMES is None:
        import inspect
        _MRMR_INIT_PARAM_NAMES = frozenset(
            p for p in inspect.signature(MRMR.__init__).parameters
            if p != "self"
        )
    n_replayed = 0
    for k, v in source.__dict__.items():
        if k in _MRMR_INIT_PARAM_NAMES:
            continue
        # Shallow assign; deep copy would defeat the cache (we WANT to share cached numpy arrays).
        target.__dict__[k] = v
        n_replayed += 1
    return n_replayed


def _lazy_chunks(iterable, chunk_size: int):
    """Lazily yield successive ``chunk_size``-sized lists from ``iterable``.

    Peak memory is O(chunk_size), not O(len(iterable)). Used by ``_run_fe_step``
    to avoid materialising all O(k^2) feature pairs at once on wide datasets
    (k=5000 features = 12.5M pairs, ~300 MB tuple-list before chunking).
    """
    if chunk_size < 1:
        raise ValueError(f"_lazy_chunks: chunk_size must be >= 1, got {chunk_size}")
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


class MRMR(BaseEstimator, TransformerMixin):
    """Finds subset of features having highest impact on target and least redundancy.

    Parameters
    ----------
        cv : int, cross-validation generator or an iterable, default=None

    Attributes
    ----------


    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ranking_ ?: narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    """

    # Process-wide cache of fitted state, keyed by (content_sig(X), content_sig(y), params_signature). When the
    # training suite iterates over models (clone()ing the pre-pipeline MRMR each time, stripping
    # ``_cat_fe_cache_``), subsequent fits on the same arrays hit this cache and skip the full cat-FE +
    # permutation work. LRU-bounded via ``OrderedDict`` + ``fit_cache_max`` (default 4) so long-lived workers
    # do not leak; ``MRMR._FIT_CACHE.clear()`` between suites still drains the lot. Cache hit: replay all
    # fitted attributes onto ``self`` and return early; constructor params are NEVER overwritten (the key
    # already includes the params signature, so a hit guarantees matching state).
    _FIT_CACHE: "OrderedDict[tuple, MRMR]" = OrderedDict()

    def __init__(
        self,
        # quantization
        quantization_method: str = "quantile",
        quantization_nbins: int = 10,
        quantization_dtype: object = np.int32,
        # NaN handling at discretization. "separate_bin" (default): assign a
        # dedicated post-max bin for NaN values per column, so MI estimators see
        # them as an honest category. "ffill_bfill": legacy forward/backward
        # fill (preserves temporal smoothness for time-series). "fillna_zero":
        # legacy pandas behaviour - mixes NaN into bin-0 with true-zero values,
        # which biases MI; only kept for reproducibility of pre-2026-05-15 runs.
        nan_strategy: str = "separate_bin",
        # factors
        factors_names_to_use: Sequence[str] = None,
        factors_to_use: Sequence[int] = None,
        # algorithm
        mrmr_relevance_algo: str = "fleuret",
        mrmr_redundancy_algo: str = "fleuret",
        reduce_gain_on_subelement_chosen: bool = True,
        # ``use_simple_mode=True`` skips the per-candidate conditional-MI redundancy check, which is fast but allows perfectly-correlated duplicate columns to BOTH be
        # selected (you'll see e.g. ``support_ = [x, 2*x]`` when both are informative). Set to ``False`` when feature deduplication matters more than wall-time.
        use_simple_mode: bool = True,
        run_additional_rfecv_minutes: bool = False,
        # performance
        extra_x_shuffling: bool = True,
        dtype=np.int32,
        random_seed: int = None,
        use_gpu: bool = False,
        n_workers: int = 1,
        # confidence
        min_occupancy: int = None,
        min_nonzero_confidence: float = 0.99,
        full_npermutations: int = 3,
        baseline_npermutations: int = 2,
        # stopping conditions
        # min_relevance_gain: absolute MI floor. In ``min_relevance_gain_mode='absolute'`` the screening stops when marginal gain < this value verbatim; in the default ``'relative_to_entropy'`` mode this value is IGNORED and the effective absolute floor is ``min_relevance_gain_frac * H(y)``. The absolute mode is dataset-blind -- 0.0001 is enormous on a low-entropy target (99/1 binary, H(y) ~= 0.056) and tiny on a high-entropy one (uniform 10-class, H(y) ~= 2.30), so the default switched to the relative formulation.
        min_relevance_gain: float = 0.0001,
        # Fraction of H(y) used as the effective absolute floor when ``min_relevance_gain_mode='relative_to_entropy'``. 0.001 = 0.1% of the target entropy.
        min_relevance_gain_frac: float = 0.001,
        # Resolution mode for ``min_relevance_gain``. ``'relative_to_entropy'`` scales the floor with H(y) so noisy features cannot pile up on low-entropy targets; ``'absolute'`` honours ``min_relevance_gain`` verbatim (legacy behaviour).
        min_relevance_gain_mode: str = "relative_to_entropy",
        max_consec_unconfirmed: int = 10,
        max_runtime_mins: float = None,
        interactions_min_order: int = 1,
        interactions_max_order: int = 1,
        interactions_order_reversed: bool = False,
        max_veteranes_interactions_order: int = 1,
        only_unknown_interactions: bool = False,
        # feature engineering settings
        fe_max_steps=1,
        fe_npermutations=0,
        fe_ntop_features=0,
        fe_unary_preset="minimal",
        fe_binary_preset="minimal",
        fe_max_pair_features: int = 1,
        fe_min_nonzero_confidence: float = 1.0,
        fe_min_pair_mi: float = 0.001,
        fe_min_pair_mi_prevalence: float = 1.05,  # transformations of what exactly pairs of factors we consider, at all. mi of entire pair must be at least that higher than the mi of its individual factors.
        fe_min_engineered_mi_prevalence: float = 0.98,  # mi of transformed pair must be at least that higher than the mi of the entire pair
        fe_good_to_best_feature_mi_threshold: float = 0.98,  # when multiple good transformations exist for the same factors pair.
        fe_max_external_validation_factors: int = 0,  # how many other factors to validate against
        fe_max_polynoms: int = 0,
        fe_print_best_mis_only: bool = True,
        fe_smart_polynom_iters: int = 0,
        fe_smart_polynom_optimization_steps: int = 1000,
        fe_min_polynom_degree: int = 3,
        fe_max_polynom_degree: int = 8,
        fe_min_polynom_coeff: float = -10.0,
        fe_max_polynom_coeff: float = 10.0,
        # verbosity and formatting
        verbose: bool | int = 0,
        ndigits: int = 5,
        parallel_kwargs: dict = None,
        # CV
        cv: object | int | None = 3,
        cv_shuffle: bool = False,
        # service
        random_state: int = None,
        n_jobs: int = -1,
        skip_retraining_on_same_shape: bool = True,
        # Cardinality cutoff for the confirmation step. ``None`` (default) computes
        # ``quantization_nbins ** interactions_max_order * 2`` (20 for the defaults). Pin to 50 for legacy behaviour.
        # Conservative default skips high-cardinality conditioning sets where permutation-based confirmation does not
        # converge in reasonable time.
        max_confirmation_cand_nbins: int = None,
        # When screening returns zero selected_vars, legacy code fell back to FE on ALL features; new default is
        # to skip FE (running FE on an empty screen typically just amplifies noise). Set True for legacy.
        fe_fallback_to_all: bool = False,
        # Pipeline-fatal fallback: when screening yields zero features (all MI ~= 0), the default leaves
        # ``support_`` empty and ``transform`` returns 0 columns, crashing any downstream estimator expecting >=1
        # feature. Set >= 1 to keep at least that many features by raw MI rank; chosen features are flagged via
        # ``self.fallback_used_``.
        min_features_fallback: int = 0,
        # Cat-FE (categorical feature interactions): single dataclass consolidating ~22 cat_fe_* knobs.
        # ``None`` = default CatFEConfig() with ``enable=True`` and conservative production settings (cat-FE
        # shows measurable wins; XOR biz_value test, 0 regressions). Restore legacy via CatFEConfig(enable=False).
        cat_fe_config=None,
        # Bound on the process-wide _FIT_CACHE. Strong refs hold every fitted MRMR; long-lived workers (web services, JupyterHub kernels) leaked memory unboundedly pre-2026-05-15. Default 4 covers a typical model suite (RFECV+MRMR x catboost+linear+mlp) without thrashing.
        fit_cache_max: int = 4,
        # hidden
        stop_file: str = "stop",
    ):

        # checks
        if n_jobs == -1:
            n_jobs = psutil.cpu_count(logical=False)

        if parallel_kwargs is None:
            parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES)

        # assert isinstance(estimator, (BaseEstimator,))

        # ``random_state`` was declared but never read internally (``random_seed`` is the only consumed alias).
        # Treat sklearn-style ``random_state`` as a fallback alias: if the user passed only ``random_state``,
        # promote it to ``random_seed`` so seeded behaviour does not silently disappear. Warn on conflicts.
        if random_state is not None:
            if random_seed is None:
                random_seed = random_state
            elif random_seed != random_state:
                warnings.warn(
                    "MRMR: both random_seed and random_state were set to different values; "
                    f"using random_seed={random_seed} and ignoring random_state={random_state}.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())
        self.signature = None

    # Allowed string values for the constructor params. Kept module-private rather
    # than a typing.Literal alias so the runtime check can produce a richer error
    # listing the valid options. fix audit row FS-P2-1.
    _VALID_QUANTIZATION_METHODS = ("quantile", "uniform")
    _VALID_NAN_STRATEGIES = ("separate_bin", "fillna_zero", "ffill_bfill", "propagate", "raise")
    _VALID_MRMR_RELEVANCE_ALGOS = ("fleuret", "pld")
    _VALID_MRMR_REDUNDANCY_ALGOS = ("fleuret", "pld_max", "pld_mean")
    _VALID_FE_UNARY_PRESETS = ("minimal", "medium", "maximal")
    _VALID_FE_BINARY_PRESETS = ("minimal", "medium", "maximal")

    def _validate_string_params(self):
        """Raise ValueError on bad constructor strings. Each branch lists the
        accepted values verbatim so the error message is actionable. fix audit
        row FS-P2-1."""
        _checks = (
            ("quantization_method", self._VALID_QUANTIZATION_METHODS),
            ("nan_strategy", self._VALID_NAN_STRATEGIES),
            ("mrmr_relevance_algo", self._VALID_MRMR_RELEVANCE_ALGOS),
            ("mrmr_redundancy_algo", self._VALID_MRMR_REDUNDANCY_ALGOS),
            ("fe_unary_preset", self._VALID_FE_UNARY_PRESETS),
            ("fe_binary_preset", self._VALID_FE_BINARY_PRESETS),
        )
        for _name, _valid in _checks:
            _val = getattr(self, _name, None)
            if _val is None:
                continue
            if not isinstance(_val, str):
                raise ValueError(
                    f"MRMR: {_name} must be a string; got {type(_val).__name__}={_val!r}. "
                    f"Valid values: {_valid}."
                )
            if _val not in _valid:
                raise ValueError(
                    f"MRMR: {_name}={_val!r} is not a recognised value. "
                    f"Valid values: {_valid}."
                )

    # Input validation contract: explicit guards for memory-exhaustion shapes, malformed dtypes,
    # +/-inf values, single-class y, and polars LazyFrame / Expr edge cases. Each guard raises ValueError or warns.
    # All-constant features are NOT rejected here: zero-variance columns survive validation and surface as MI=0
    # in the screening loop, which is the documented downstream behaviour.
    def _validate_inputs(self, X, y):
        # Validate string-valued constructor params on every fit. We intentionally
        # do NOT validate inside __init__ to preserve sklearn-style "no work in
        # __init__" semantics (clone() must not raise).
        self._validate_string_params()
        import warnings as _w
        n_rows = getattr(X, "shape", (None,))[0]
        if n_rows is not None:
            n_cols = X.shape[1] if len(X.shape) > 1 else 1
            if n_rows == 0:
                raise ValueError("MRMR.fit: empty input (n_rows=0)")
            if n_rows == 1:
                raise ValueError("MRMR.fit: cannot fit on a single row")
            if isinstance(n_cols, int):
                # MRMR's binned-frame working set is roughly ``n_rows * n_cols * 4`` bytes (int32 per cell). The previous absolute 1e9 cell ceiling rejected datasets that comfortably fit in RAM on a modern 128 GB+ host while letting through wide-but-not-as-wide frames on a tiny 16 GB box. Compare to ``psutil.virtual_memory().available * 0.5`` -- half of free RAM is the standard "safe working set" headroom for one stage of the pipeline.
                _footprint_bytes = n_rows * n_cols * 4
                try:
                    import psutil as _psutil
                    _available_bytes = int(_psutil.virtual_memory().available)
                except Exception:
                    _available_bytes = 0
                _headroom_bytes = _available_bytes // 2
                if _headroom_bytes > 0 and _footprint_bytes > _headroom_bytes:
                    raise ValueError(
                        f"MRMR.fit: refusing to allocate for n*p={n_rows * n_cols:_} "
                        f"(~{_footprint_bytes / 1e9:.2f} GB int32 working set) on a host with "
                        f"{_available_bytes / 1e9:.2f} GB available RAM; threshold is half of available "
                        f"(~{_headroom_bytes / 1e9:.2f} GB). Subsample, split the dataset, or free RAM "
                        "headroom before fitting."
                    )
        if self.quantization_nbins > 1000:
            raise ValueError(f"quantization_nbins={self.quantization_nbins} > 1000 likely OOMs")
        if self.interactions_max_order > 5:
            raise ValueError(f"interactions_max_order={self.interactions_max_order} > 5 explodes combinatorially")
        if getattr(self, "fe_max_steps", 0) > 20:
            raise ValueError(f"fe_max_steps={self.fe_max_steps} > 20 unlikely to converge")
        # Polars edge cases.
        try:
            import polars as _pl
            if isinstance(X, _pl.LazyFrame):
                _w.warn("MRMR.fit autocollecting LazyFrame; pass DataFrame to skip this copy.", stacklevel=3)
                X = X.collect()
            if hasattr(_pl, "Expr") and isinstance(X, _pl.Expr):
                raise ValueError("MRMR.fit cannot accept polars Expr; materialise via .select(...) first")
            if isinstance(X, _pl.DataFrame):
                # Polars struct columns are not supported.
                struct_cols = [name for name, dt in X.schema.items() if str(dt).startswith("Struct")]
                if struct_cols:
                    raise ValueError(f"MRMR.fit: polars Struct columns not supported: {struct_cols}")
        except ImportError:
            pass
        # Pandas: duplicate column names.
        if hasattr(X, "columns"):
            cols = list(X.columns)
            if len(cols) != len(set(cols)):
                from collections import Counter
                dups = [c for c, n in Counter(cols).items() if n > 1]
                raise ValueError(f"MRMR.fit: duplicate column names not supported: {dups}")
        # Numeric-column extraction for NaN / Inf validation. Object-dtype frames (numeric + cat/string mixed) used to slip through because the whole frame was
        # converted to object-dtype where dtype.kind != "f"; scan numeric columns explicitly instead.
        try:
            try:
                import polars as _pl
                if isinstance(X, _pl.DataFrame):
                    _num_cols = [n for n, d in X.schema.items() if d.is_numeric()]
                    if _num_cols:
                        _arr = X.select(_num_cols).to_numpy()
                elif hasattr(X, "select_dtypes"):
                    _arr = X.select_dtypes(include=["number"]).to_numpy()
                else:
                    _arr = X.to_numpy()
            except ImportError:
                _arr = X.to_numpy() if hasattr(X, "to_numpy") else None
            if _arr is not None and _arr.dtype.kind == "f":
                if np.isinf(_arr).any():
                    raise ValueError(
                        "MRMR.fit: input X contains +/-inf values. Replace or drop these rows before fitting; the discretization step produces undefined bins on inf."
                    )
                # NaN is allowed and routed through `self.nan_strategy` (default
                # "separate_bin": NaN rows get an honest dedicated bin instead of
                # being merged into bin-0 or imputed silently). transform()
                # preserves NaN in the returned X for downstream NaN-aware models
                # (catboost, lightgbm, xgboost histogram tree).
        except ValueError:
            raise  # re-raise our own ValueError
        except Exception:
            pass
        # All-same y: raise (symmetric with RFECV.fit's single-class y validation). Constant y has H(y)=0 so
        # every MI(X_j, y) = 0; the entire MRMR pipeline produces zero-information output.
        try:
            if len(np.unique(np.asarray(y))) == 1:
                raise ValueError(
                    "MRMR.fit: target y has only 1 unique value. H(y)=0 "
                    "so all features have MI(X_j, y)=0 by construction. "
                    "Drop or rebuild y before fitting."
                )
        except ValueError:
            raise  # re-raise our own ValueError
        except Exception:
            pass
        return X

    def _resolve_target_prefix(self) -> str:
        """Stable, seedable prefix for the temporary target columns injected during fit.

        Pre-fix code used ``str(np.random.random())[3:9]`` which (a) reseeded
        nothing but consumed from the process-global numpy RNG, breaking
        reproducibility across test orderings, and (b) produced a different
        prefix every call. With ``random_seed`` set, derive a deterministic 6-hex
        suffix from a local ``np.random.default_rng``; otherwise fall back to a
        process-stable but seedable PID+id(self)-based source so concurrent
        instances stay collision-resistant without touching global state.
        """
        if self.random_seed is not None:
            local_rng = np.random.default_rng(int(self.random_seed))
            tok = int(local_rng.integers(0, 2**24))
        else:
            tok = (os.getpid() ^ id(self)) & 0xFFFFFF
        return f"targ_{tok:06x}"

    def _coerce_target_dtype(self, vals: np.ndarray) -> np.ndarray:
        """Memory-saving int64 -> int16 downcast, guarded against silent truncation.

        Pre-fix path was unconditional: ``vals.dtype == np.int64`` triggered an
        ``astype(np.int16)`` regardless of value range, silently truncating any
        target outside [-32768, 32767]. New behaviour: downcast only when the
        observed range fits; otherwise keep int64 and warn at logger level so
        regression / multiclass-with-large-codes targets are preserved bit-exact.
        """
        if vals.dtype != np.int64:
            return vals
        vmin, vmax = vals.min(), vals.max()
        info = np.iinfo(np.int16)
        if vmin >= info.min and vmax <= info.max:
            if self.verbose:
                logger.info("Converted targets from int64 to int16.")
            return vals.astype(np.int16)
        if self.verbose:
            logger.warning(
                "MRMR: keeping int64 targets (range [%d, %d] exceeds int16 [%d, %d]); skipping memory-saving downcast.",
                int(vmin), int(vmax), info.min, info.max,
            )
        return vals

    def _rfecv_cv_kwargs(self) -> dict:
        """Forward ``self.cv`` / ``self.cv_shuffle`` into the inner RFECV call.

        These two MRMR constructor params used to be dead (zero callers read
        ``self.cv``); they're now threaded into the RFECV instance built for the
        post-screening ``run_additional_rfecv_minutes`` pass so users who pass
        ``cv=5`` actually get 5-fold there.
        """
        return {"cv": self.cv, "cv_shuffle": self.cv_shuffle}

    # Pickle BC: old MRMR pickles lacking newer attributes resurface with the legacy defaults injected.
    def __setstate__(self, state):
        defaults = {
            "max_confirmation_cand_nbins": 50,  # legacy default
            "fe_fallback_to_all": True,         # legacy default
            "_engineered_features_": [],
            # Recipes-based replay so transform() can recompute engineered features on test data. Old pickles
            # have no recipes (their engineered cols were never replayable); empty list reproduces the legacy
            # "engineered cols dropped from transform output" behaviour bit-exact.
            "_engineered_recipes_": [],
            # Cat-FE: ``None`` means disabled or never ran; injecting ``None`` makes getattr(...) no-op.
            "cat_fe_config": None,
            "_cat_fe_state_": None,
            "_cat_fe_cache_": None,  # streaming cache; None on legacy pickles
        }
        for k, v in defaults.items():
            state.setdefault(k, v)
        self.__dict__.update(state)

    def _maybe_resample_for_sample_weight(self, X, y, sample_weight: np.ndarray | None):
        """When ``sample_weight`` is provided AND not effectively uniform, draw n=len(X) rows with replacement
        using probabilities w_i / sum(w). The resampled empirical bincount approximates the weighted bincount
        (np.bincount(x, weights=w) up to MC noise), so MI relevance / redundancy estimated downstream from
        binned joint histograms becomes weight-aware without touching info_theory / screen internals.
        Returns (X, y) unchanged when sample_weight is None / all-equal (preserves byte-for-byte legacy path
        and lets the FS cache reuse a single fit across uniform-weight callers)."""
        if sample_weight is None:
            return X, y
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.ndim != 1:
            raise ValueError(f"MRMR.fit sample_weight must be 1-D, got shape {sw.shape}")
        n_rows = X.shape[0]
        if sw.shape[0] != n_rows:
            raise ValueError(f"MRMR.fit sample_weight length {sw.shape[0]} != n_rows {n_rows}")
        if not np.all(np.isfinite(sw)) or (sw < 0).any():
            raise ValueError("MRMR.fit sample_weight must be finite and non-negative")
        total = float(sw.sum())
        if total <= 0:
            raise ValueError("MRMR.fit sample_weight sums to zero")
        # Uniform -> nothing to do (preserves bit-exact legacy + cache reuse).
        if float(sw.max() - sw.min()) <= 1e-12 * max(1.0, abs(float(sw.mean()))):
            return X, y
        rng = np.random.default_rng(self.random_seed if self.random_seed is not None else 0)
        probs = sw / total
        idx = rng.choice(n_rows, size=n_rows, replace=True, p=probs)
        # iloc preserves dtypes / category metadata; works on pandas + polars (via take) + numpy.
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_rs = X[idx.tolist()] if hasattr(X, "__getitem__") else X.take(idx)
            elif isinstance(X, pd.DataFrame):
                X_rs = X.iloc[idx]
            else:
                X_rs = np.asarray(X)[idx]
        except ImportError:
            if isinstance(X, pd.DataFrame):
                X_rs = X.iloc[idx]
            else:
                X_rs = np.asarray(X)[idx]
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_rs = y.iloc[idx]
        else:
            y_rs = np.asarray(y)[idx]
        return X_rs, y_rs

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, sample_weight: np.ndarray | pd.Series | None = None, **fit_params):
        """Public ``fit`` wrapper. The body (``_fit_impl``) is run inside a try / finally so the
        temporary target columns injected into a caller-supplied pandas frame are always dropped,
        even if screening / cat-FE / discretization raises. Pre-fix code dropped only on success,
        leaking ``targ_*`` columns into the caller's DataFrame on failure paths.

        sample_weight: optional per-row weights. When provided and non-uniform, rows of (X, y) are
        resampled with replacement using probabilities proportional to sample_weight before screening;
        the resampled distribution converges to the weighted bincount target distribution as N grows,
        so MI relevance / redundancy (computed downstream via binned joint histograms) becomes
        weight-aware without modifying screen / info_theory internals. sample_weight=None retains the
        old code path byte-for-byte (regression sentry)."""
        self._pandas_frame_for_target_cleanup = None
        self._target_names_for_cleanup = None
        # Persist user-supplied weights so cached _cat_fe_state_ / FE replay can introspect; cache key
        # below already excludes weights so the FS-cache reuse contract stays intact when the suite
        # caller decides to gate weight-aware MRMR behind FeatureSelectionConfig.use_sample_weights_in_fs.
        self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        X, y = self._maybe_resample_for_sample_weight(X, y, self._fit_sample_weight_)
        try:
            return self._fit_impl(X, y, groups, **fit_params)
        finally:
            frame = getattr(self, "_pandas_frame_for_target_cleanup", None)
            names = getattr(self, "_target_names_for_cleanup", None)
            if frame is not None and names:
                # Drop only columns that actually exist (success path already removed them).
                present = [c for c in names if c in frame.columns]
                if present:
                    try:
                        frame.drop(columns=present, inplace=True)
                    except Exception:
                        pass
            self._pandas_frame_for_target_cleanup = None
            self._target_names_for_cleanup = None

    def _fit_impl(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, **fit_params):
        """We run N selections on data subsets, and pick only features that appear in all selections"""
        X = self._validate_inputs(X, y)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute inputs/outputs signature
        # ----------------------------------------------------------------------------------------------------------------------------

        signature = (X.shape, y.shape)
        if self.skip_retraining_on_same_shape:
            if signature == self.signature:
                if self.verbose:
                    logger.info("Skipping retraining on the same inputs signature %s", signature)
                return self

        # Process-wide ``_FIT_CACHE`` hit. After sklearn.base.clone() the cloned MRMR has no fitted state so
        # the signature==signature shortcut above never fires. Content-based key (id-based missed every hit
        # because the suite copies X between iterations -- different id() but identical content);
        # _content_array_signature returns shape+dtype+10 sampled values, cheap O(1) and statistically unique
        # enough to avoid false positives on real data. Falls through to full fit on any error or miss.
        _cache_key = None
        try:
            _params_sig = _hashable_params_signature(self.get_params(deep=False))
            _x_sig = _content_array_signature(X)
            _y_sig = _content_array_signature(y)
            # Two targets with statistically-similar 10-cell samples (e.g. balanced binary) collide on _y_sig
            # alone and replay one another's support_. Include the target name AND a full blake2b over y to
            # disambiguate. Empty hash => skip cache (don't risk a wrong replay).
            _y_name = _target_name_signature(y)
            _y_full_hash = _full_y_content_hash(y)
            if not _y_full_hash:
                _cache_key = None
            else:
                _cache_key = (_x_sig, _y_sig, _y_name, _y_full_hash, _params_sig)
        except Exception:
            _cache_key = None
        if _cache_key is not None and _cache_key in MRMR._FIT_CACHE:
            _cached = MRMR._FIT_CACHE[_cache_key]
            MRMR._FIT_CACHE.move_to_end(_cache_key)
            _replayed = _replay_fitted_state(self, _cached)
            if self.verbose:
                logger.info(
                    "MRMR.fit: _FIT_CACHE hit -- replayed %d fitted attrs "
                    "from prior fit, skipping cat-FE + permutation.",
                    _replayed,
                )
            return self

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        # Outer FE-loop runtime-budget guard. screen_predictors honours self.max_runtime_mins on its own; here we additionally
        # short-circuit between FE iterations so a long FE step that finished after the budget elapsed doesn't trigger another.
        start_time = timer()
        ran_out_of_time = False

        dtype = self.dtype

        parallel_kwargs = self.parallel_kwargs
        n_jobs = self.n_jobs
        verbose = self.verbose

        prefetch_factor = 4

        fe_max_steps = self.fe_max_steps
        fe_npermutations = self.fe_npermutations
        fe_unary_preset = self.fe_unary_preset
        fe_binary_preset = self.fe_binary_preset
        fe_max_pair_features = self.fe_max_pair_features

        fe_min_nonzero_confidence = self.fe_min_nonzero_confidence
        fe_min_pair_mi = self.fe_min_pair_mi
        fe_min_pair_mi_prevalence = self.fe_min_pair_mi_prevalence
        fe_min_engineered_mi_prevalence = self.fe_min_engineered_mi_prevalence
        fe_good_to_best_feature_mi_threshold = self.fe_good_to_best_feature_mi_threshold
        fe_max_external_validation_factors = self.fe_max_external_validation_factors
        fe_max_polynoms = self.fe_max_polynoms
        fe_print_best_mis_only = self.fe_print_best_mis_only
        fe_smart_polynom_iters = self.fe_smart_polynom_iters
        fe_smart_polynom_optimization_steps = self.fe_smart_polynom_optimization_steps
        fe_min_polynom_degree = self.fe_min_polynom_degree
        fe_max_polynom_degree = self.fe_max_polynom_degree
        fe_min_polynom_coeff = self.fe_min_polynom_coeff
        fe_max_polynom_coeff = self.fe_max_polynom_coeff

        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        self.feature_names_in_ = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # ---------------------------------------------------------------------------------------------------------------
        # Temporarily inject targets
        # ---------------------------------------------------------------------------------------------------------------

        target_prefix = self._resolve_target_prefix()
        y_shape = y.shape
        if len(y_shape) == 2:
            y_shape = y_shape[1]
        else:
            y_shape = 1
        target_names = [target_prefix + "_" + str(i) for i in range(y_shape)]

        vals = _target_to_numpy_values(y)
        vals = self._coerce_target_dtype(vals)

        # Native Polars support -- no `.to_pandas()` copy. Production frames are 100+ GB; full materialization
        # would OOM. Use Polars-native ops when the input is pl.DataFrame.
        try:
            import polars as pl  # local alias; safe even if pl is already imported module-scope
            _is_polars_input = isinstance(X, pl.DataFrame)
        except ImportError:
            _is_polars_input = False

        # Track the caller-visible pandas frame so the ``finally`` below can always drop the injected target columns even if
        # ``fit`` raises mid-way (e.g. categorize_dataset / screen_predictors / cat-FE step). Pre-fix code dropped only on
        # the happy path, so a raised exception left ``targ_*`` columns on the caller's frame; downstream pipelines then
        # baked them into ``feature_names_in_`` and crashed on ``transform``.
        _caller_pandas_frame = None
        if _is_polars_input:
            # Polars is immutable; with_columns returns a new frame sharing buffers with X -- no data copy.
            target_series = [pl.Series(name, vals[:, i] if vals.ndim == 2 else vals) for i, name in enumerate(target_names)]
            X = X.with_columns(target_series)
        else:
            # Multilabel target (N, K): pass through unchanged so each column maps to its target_names entry.
            # Previous .reshape(-1, 1) only worked for 1-D y; crashed on multilabel with "Must have equal len keys
            # and value when setting with an ndarray".
            _caller_pandas_frame = X
            if vals.ndim == 2:
                X.loc[:, target_names] = vals
            else:
                X.loc[:, target_names] = vals.reshape(-1, 1)
            # Register cleanup with the public ``fit`` wrapper so any later raise still strips ``targ_*``.
            self._pandas_frame_for_target_cleanup = _caller_pandas_frame
            self._target_names_for_cleanup = list(target_names)

        # ---------------------------------------------------------------------------------------------------------------
        # Discretize continuous data
        # ---------------------------------------------------------------------------------------------------------------

        logger.info("categorizing dataset...")
        # NaN handling is delegated to `categorize_dataset` via
        # `missing_strategy`. The legacy ffill/bfill path was a temporal-fill
        # workaround that injected fake signal correlated with the row's
        # neighbours; the default "separate_bin" treats NaN as an honest
        # category (its own bin per column), which an MI estimator handles
        # correctly with no special-casing on the receiving side.
        if self.nan_strategy in ("ffill_bfill",):
            # Legacy path retained for reproducibility of pre-2026-05-15 runs.
            if _is_polars_input:
                _x_for_cat = X.fill_null(strategy="forward").fill_null(strategy="backward")
            else:
                _x_for_cat = X.ffill().bfill()
            _strategy_for_categorize = "fillna_zero"  # any residual NaN -> 0 (legacy)
        else:
            _x_for_cat = X
            _strategy_for_categorize = self.nan_strategy
        data, cols, nbins = categorize_dataset(
            df=_x_for_cat,
            method=self.quantization_method,
            n_bins=self.quantization_nbins,
            dtype=self.quantization_dtype,
            missing_strategy=_strategy_for_categorize,
        )
        logger.info("categorized.")

        target_indices = np.array([cols.index(col) for col in target_names], dtype=np.int64)

        # ---------------------------------------------------------------------------------------------------------------
        # Core
        # ---------------------------------------------------------------------------------------------------------------

        if _is_polars_input:
            # Polars schema-driven detection; mirrors categorize_dataset's _is_pl_cat.
            import polars as _pl
            _CAT_DTYPES_FOR_VARS = {_pl.Utf8, _pl.String, _pl.Categorical, _pl.Boolean}
            categorical_vars_names = [
                name for name, dt in X.schema.items()
                if dt in _CAT_DTYPES_FOR_VARS
                or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))
            ]
        else:
            categorical_vars_names = X.head().select_dtypes(include=("category", "object", "bool")).columns.values.tolist()
        categorical_vars = [cols.index(col) for col in categorical_vars_names]

        if fe_max_steps > 0:
            unary_transformations = create_unary_transformations(preset=fe_unary_preset)
            binary_transformations = create_binary_transformations(preset=fe_binary_preset)
            if fe_max_polynoms:
                # Generated polynomial coefficients are appended directly to unary_transformations under "poly_<coef>" keys;
                # no separate registry is needed.
                for _ in range(fe_max_polynoms):
                    length = np.random.randint(3, 9)
                    coef = np.empty(shape=length, dtype=np.float32)
                    for i in range(length):
                        coef[i] = np.random.normal((1.0 if i == 1 else 0.0), scale=0.05)

                    unary_transformations["poly_" + str(coef)] = coef

            if verbose > 2:
                logger.info("nunary_transformations: %s", f"{len(unary_transformations):_}")
                logger.info("nbinary_transformations: %s", f"{len(binary_transformations):_}")

            engineered_features = set()
            checked_pairs = set()
        # engineered_recipes (name -> EngineeredRecipe) is initialised unconditionally; the splitter at the bottom
        # of fit() looks it up regardless of fe_max_steps. Stays empty when FE is disabled.
        engineered_recipes: dict = {}

        # Cat-FE step (categorical interaction generator). Runs once before the screening loop when
        # ``cat_fe_config.enable=True``; augments data/cols/nbins with ordinal-encoded columns capturing pair
        # (and future k-way) synergies. Engineered cols enter screening as atomic 1-way features.
        cat_fe_cfg = getattr(self, "cat_fe_config", None)
        self._cat_fe_state_ = None
        # ``None`` means "use default CatFEConfig()" which has enable=True. Pass CatFEConfig(enable=False) for legacy.
        if cat_fe_cfg is None:
            from .cat_fe_state import CatFEConfig as _CatFEConfig
            cat_fe_cfg = _CatFEConfig()
        if cat_fe_cfg.enable and len(categorical_vars) >= 2:
            from .cat_interactions import run_cat_interaction_step
            from .info_theory import merge_vars as _merge_vars_for_cat_fe

            # Pre-compute classes_y / freqs_y for cat-FE (avoids re-binning the target inside every kernel call).
            _classes_y, _freqs_y, _ = _merge_vars_for_cat_fe(
                factors_data=data, vars_indices=target_indices,
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            _classes_y_safe = _classes_y.copy()

            # Pull cached cat-FE state from prior fit (if any).
            _prev_cache = getattr(self, "_cat_fe_cache_", None)
            _n_cols_before_cat_fe = data.shape[1]
            data, cols, nbins, cat_fe_state = run_cat_interaction_step(
                data=data, cols=cols, nbins=nbins,
                target_indices=target_indices,
                classes_y=_classes_y, classes_y_safe=_classes_y_safe,
                freqs_y=_freqs_y,
                categorical_vars=categorical_vars,
                cfg=cat_fe_cfg,
                streaming_cache=_prev_cache,
                dtype=dtype, verbose=verbose,
            )
            self._cat_fe_state_ = cat_fe_state
            # Register engineered cat features as categorical_vars so the downstream numeric-FE step excludes them
            # from numeric_vars_to_consider; without this, k-way cat engineered cols enter prospective_pairs and
            # check_prospective_fe_pairs hits KeyError reading them from X (which lacks engineered cols).
            # Engineered cat cols are appended at the end of data/cols at positions [_n_cols_before_cat_fe..].
            _n_cat_fe_added = data.shape[1] - _n_cols_before_cat_fe
            if _n_cat_fe_added > 0:
                categorical_vars = list(categorical_vars) + list(
                    range(_n_cols_before_cat_fe, data.shape[1])
                )
            # Persist cache for next fit() call
            if cat_fe_state.streaming_cache_out:
                self._cat_fe_cache_ = cat_fe_state.streaming_cache_out
            # Cat-FE recipes feed the same engineered_recipes dict numeric FE uses; the fit-end splitter copies
            # any recipe whose engineered name appears in selected_vars_names into ``self._engineered_recipes_``.
            for r in cat_fe_state.recipes:
                engineered_recipes[r.name] = r
            if verbose and cat_fe_state.recipes:
                logger.info(
                    "MRMR cat-FE produced %d engineered feature(s); "
                    "data extended from %d to %d cols.",
                    len(cat_fe_state.recipes),
                    data.shape[1] - len(cat_fe_state.recipes),
                    data.shape[1],
                )

        # Resolve effective ``min_relevance_gain`` against the target entropy. ``'relative_to_entropy'`` mode uses ``min_relevance_gain_frac * H(y)`` so the stop floor scales with how much information the target actually carries; ``'absolute'`` mode retains the legacy verbatim value. The target is already discretized into bins (``data[:, target_indices[0]]`` with bin count ``nbins[target_indices[0]]``); ``np.bincount`` + Shannon entropy in nats matches the screen_predictors estimator family.
        if self.min_relevance_gain_mode not in ("absolute", "relative_to_entropy"):
            raise ValueError(
                f"MRMR.min_relevance_gain_mode={self.min_relevance_gain_mode!r} must be 'absolute' or 'relative_to_entropy'."
            )
        if self.min_relevance_gain_mode == "relative_to_entropy":
            _target_col_idx = int(target_indices[0])
            _y_bins = data[:, _target_col_idx]
            _y_nbins = int(nbins[_target_col_idx])
            _y_counts = np.bincount(_y_bins, minlength=_y_nbins).astype(np.float64)
            _y_total = float(_y_counts.sum())
            if _y_total > 0:
                _p = _y_counts[_y_counts > 0] / _y_total
                _h_y_nats = float(-(_p * np.log(_p)).sum())
            else:
                _h_y_nats = 0.0
            _effective_min_relevance_gain = float(self.min_relevance_gain_frac) * _h_y_nats
            if verbose:
                logger.info(
                    "MRMR min_relevance_gain resolution: mode=relative_to_entropy, H(y)=%.4f nats, frac=%.4g, effective floor=%.6g (legacy absolute would have been %.6g).",
                    _h_y_nats, self.min_relevance_gain_frac, _effective_min_relevance_gain, self.min_relevance_gain,
                )
        else:
            _effective_min_relevance_gain = float(self.min_relevance_gain)

        num_fs_steps = 0
        while True:
            n_recommended_features = 0
            times_spent = defaultdict(float)
            selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y = (
                screen_predictors(
                    factors_data=data,
                    y=target_indices,
                    factors_nbins=nbins,
                    factors_names=cols,
                    factors_names_to_use=self.factors_names_to_use,
                    factors_to_use=self.factors_to_use,
                    # algorithm
                    mrmr_relevance_algo=self.mrmr_relevance_algo,
                    mrmr_redundancy_algo=self.mrmr_redundancy_algo,
                    reduce_gain_on_subelement_chosen=self.reduce_gain_on_subelement_chosen,
                    use_simple_mode=self.use_simple_mode,
                    # performance
                    extra_x_shuffling=self.extra_x_shuffling,
                    dtype=self.dtype,
                    random_seed=self.random_seed,
                    use_gpu=self.use_gpu,
                    n_workers=self.n_workers,
                    # confidence
                    min_occupancy=self.min_occupancy,
                    min_nonzero_confidence=self.min_nonzero_confidence,
                    full_npermutations=self.full_npermutations,
                    baseline_npermutations=self.baseline_npermutations,
                    # stopping conditions
                    min_relevance_gain=_effective_min_relevance_gain,
                    max_consec_unconfirmed=self.max_consec_unconfirmed,
                    max_runtime_mins=self.max_runtime_mins,
                    interactions_min_order=self.interactions_min_order,
                    interactions_max_order=self.interactions_max_order,
                    interactions_order_reversed=self.interactions_order_reversed,
                    max_veteranes_interactions_order=self.max_veteranes_interactions_order,
                    only_unknown_interactions=self.only_unknown_interactions,
                    # Resolve effective max_confirmation_cand_nbins: user-pinned wins, else formula default.
                    max_confirmation_cand_nbins=(
                        self.max_confirmation_cand_nbins
                        if self.max_confirmation_cand_nbins is not None
                        else self.quantization_nbins ** self.interactions_max_order * 2
                    ),
                    # FE-on-empty-screen fallback flag (consumed by MRMR.fit).
                    fe_fallback_to_all=self.fe_fallback_to_all,
                    # verbosity and formatting
                    verbose=self.verbose,
                    ndigits=self.ndigits,
                    parallel_kwargs=self.parallel_kwargs,
                    stop_file=self.stop_file,
                    # engineered_lineage from cat-FE step (None when cat-FE didn't run); screen uses it to skip
                    # redundant (orig_parent, engineered_col) k-way candidates.
                    engineered_lineage=(
                        self._cat_fe_state_.lineage
                        if getattr(self, "_cat_fe_state_", None) is not None
                        and self._cat_fe_state_.lineage
                        else None
                    ),
                )
            )

            if fe_max_steps == 0 or num_fs_steps >= fe_max_steps:
                break

            if self.max_runtime_mins is not None:
                elapsed_min = (timer() - start_time) / 60.0
                if elapsed_min >= self.max_runtime_mins:
                    ran_out_of_time = True
                    if verbose:
                        logger.info("MRMR.fit: runtime budget %.1f min exceeded at FE step %d; stopping.", self.max_runtime_mins, num_fs_steps)
                    break

            # Feature engineering iteration delegated to ``_run_fe_step`` (testable / experiment-friendly outside
            # the screening loop). Returns updated state + n_recommended_features; zero breaks the outer loop.
            fe_result = self._run_fe_step(
                data=data, cols=cols, nbins=nbins, X=X,
                target_names=target_names, target_indices=target_indices,
                selected_vars=selected_vars,
                categorical_vars=categorical_vars,
                classes_y=classes_y, classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
                unary_transformations=unary_transformations,
                binary_transformations=binary_transformations,
                engineered_features=engineered_features,
                engineered_recipes=engineered_recipes,
                checked_pairs=checked_pairs,
                times_spent=times_spent,
                num_fs_steps=num_fs_steps,
                n_jobs=n_jobs, prefetch_factor=prefetch_factor,
                parallel_kwargs=parallel_kwargs,
                _is_polars_input=_is_polars_input,
                verbose=verbose,
                fe_max_steps=fe_max_steps,
                fe_npermutations=fe_npermutations,
                fe_max_pair_features=fe_max_pair_features,
                fe_print_best_mis_only=fe_print_best_mis_only,
                fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
                fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
                fe_max_external_validation_factors=fe_max_external_validation_factors,
                fe_min_pair_mi=fe_min_pair_mi,
                fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                fe_smart_polynom_iters=fe_smart_polynom_iters,
                fe_smart_polynom_optimization_steps=fe_smart_polynom_optimization_steps,
                fe_min_polynom_degree=fe_min_polynom_degree,
                fe_max_polynom_degree=fe_max_polynom_degree,
                fe_min_polynom_coeff=fe_min_polynom_coeff,
                fe_max_polynom_coeff=fe_max_polynom_coeff,
                fe_unary_preset=fe_unary_preset,
                fe_binary_preset=fe_binary_preset,
            )
            if fe_result is None:
                break  # FE skip: empty screening + fe_fallback_to_all=False
            data, cols, nbins, X, selected_vars, n_recommended_features = fe_result

            if n_recommended_features == 0:
                break

            num_fs_steps += 1
            if num_fs_steps >= fe_max_steps:
                break  # uncomment to avoid recheck of single-rounded FE

        if verbose > 2:
            logger.info("time spent by binary func: %s", sort_dict_by_value(times_spent))
        # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

        # ---------------------------------------------------------------------------------------------------------------
        # Drop temporary targets
        # ---------------------------------------------------------------------------------------------------------------

        # Fuzz-caught: previous ``X = X.drop(columns=target_names)`` returned a new DataFrame and only rebound the
        # local; for pandas input (where X.loc[:, target_names] = ... mutated the caller's frame), the caller's
        # X was left with the injected ``targ_<id>`` columns, which leaked into downstream sklearn pipeline
        # (imputer/scaler recorded them in feature_names_in_ and raised on transform). Fix: drop in place (pandas)
        # or rebind (polars -- immutable, caller's X was never mutated).
        if _is_polars_input:
            X = X.drop(target_names)  # no-copy lazy op; caller's X untouched
        else:
            X.drop(columns=target_names, inplace=True)  # restores caller's original schema

        # ---------------------------------------------------------------------------------------------------------------
        # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
        # ---------------------------------------------------------------------------------------------------------------

        selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]
        # Tolerate FE-engineered names: screening output may include synthetic feature names not in
        # feature_names_in_; record them in self._engineered_features_ instead of raising on the .index() lookup.
        # Also surface matching EngineeredRecipe (built during _run_fe_step) so transform() can replay each
        # engineered column on test data. An engineered name without a recipe (e.g. higher-order interaction
        # whose parents are themselves engineered) is recorded by name only and dropped from transform output.
        self._engineered_features_ = []
        self._engineered_recipes_ = []
        original_indices = []
        engineered_without_recipe = []
        for col in selected_vars_names:
            if col in self.feature_names_in_:
                original_indices.append(self.feature_names_in_.index(col))
            else:
                self._engineered_features_.append(col)
                recipe = engineered_recipes.get(col)
                if recipe is not None:
                    self._engineered_recipes_.append(recipe)
                else:
                    engineered_without_recipe.append(col)
        if engineered_without_recipe and verbose:
            # Happens with fe_max_steps>1 when a higher-order interaction's parents are themselves engineered features. The recipe replay path can only
            # reconstruct 1-deep engineering; deeper nests are recorded in self._engineered_features_ but DROPPED from transform output. Surface the cost.
            logger.warning(
                "MRMR.fit: %d engineered feature(s) selected without replayable recipe (nested-engineered parents at fe_max_steps=%d); they will be DROPPED from transform output: %s",
                len(engineered_without_recipe), self.fe_max_steps, engineered_without_recipe[:8],
            )
        # ``selected_vars`` is downstream re-bound to the integer indices of the RAW columns only; engineered features are appended in transform() via
        # ``_append_engineered`` using ``self._engineered_recipes_``. This split mirrors the on-disk contract: support_ indexes feature_names_in_; engineered output
        # columns come from the recipes list. n_features_ counts BOTH (see assignment below).
        selected_vars = original_indices

        # ---------------------------------------------------------------------------------------------------------------
        # additional_rfecv run
        # ---------------------------------------------------------------------------------------------------------------

        if self.run_additional_rfecv_minutes:
            """On the factors discarded by MRMR, let's run RFECV to see if any of them participate in interactions"""
            n_unexplored = X.shape[1] - len(selected_vars)
            if n_unexplored > 0:
                if verbose:
                    logger.info(
                        f"Running RFECV for {self.run_additional_rfecv_minutes} minute(s) over {n_unexplored:_} feature(s) discarded by MRMR to extract interactions..."
                    )

                from mlframe.training import get_training_configs

                configs = get_training_configs(has_time=True)

                params = configs.COMMON_RFECV_PARAMS.copy()
                params["max_runtime_mins"] = self.run_additional_rfecv_minutes
                # Wire MRMR.cv / cv_shuffle into the additional RFECV pass; pre-fix they were dead constructor params.
                # ``params`` may already carry ``cv`` from configs.COMMON_RFECV_PARAMS; MRMR's explicit setting wins.
                params.update(self._rfecv_cv_kwargs())

                # Classifier-vs-regressor detection. Preference order:
                #   1) Explicit ``target_type`` attribute on self (set by the caller / harness).
                #   2) Honest dtype + cardinality heuristic: float dtype is regression by
                #      construction (zero-inflated targets like ``[0]*900 + [1.7, 2.4, ...]``
                #      satisfy the legacy ratio>100 but are NOT classification). Integer
                #      dtype with ratio>100 AND small absolute cardinality (<=64 unique
                #      values) is classification. Everything else is regression.
                # Pre-fix, the regression else-branch silently skipped the
                # additional-RFECV pass entirely, so regression callers got no benefit
                # from run_additional_rfecv_minutes. The dtype guard prevents misclassifying
                # zero-inflated float targets. fix audit row FS-L-2.
                _explicit_tt = getattr(self, "target_type", None)
                if _explicit_tt is not None:
                    _tt_str = str(_explicit_tt).lower()
                    _is_classification = "classif" in _tt_str or _tt_str in ("binary", "multiclass", "multilabel")
                else:
                    _y_arr = np.asarray(y)
                    _n_unique = len(np.unique(_y_arr))
                    _ratio = len(_y_arr) / max(1, _n_unique)
                    _is_float = _y_arr.dtype.kind == "f"
                    _is_classification = (not _is_float) and _ratio > 100 and _n_unique <= 64
                    if _ratio > 100 and _is_float:
                        logger.warning(
                            "MRMR.run_additional_rfecv: target is float dtype with %d unique values; "
                            "treating as regression despite samples/unique ratio %.1f>100. Pass "
                            "target_type='classification' explicitly to override.",
                            _n_unique, _ratio,
                        )
                temp_columns = list(set(X.columns) - set(X.columns[selected_vars]))

                if _is_classification:
                    cb_num_rfecv = RFECV(
                        estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        scoring=make_scorer(
                            score_func=compute_probabilistic_multiclass_error, response_method='predict_proba', greater_is_better=False
                        ),
                        **params,
                    )
                else:
                    # Regression branch: CatBoostRegressor with the same shared params; default scoring lets
                    # RFECV pick from the estimator (negative-MSE-like). Keeping the import local avoids
                    # paying the CatBoostRegressor import cost when only classification is exercised.
                    from catboost import CatBoostRegressor
                    cb_num_rfecv = RFECV(
                        estimator=CatBoostRegressor(**configs.CB_REGR),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        **params,
                    )
                cb_num_rfecv.fit(X[temp_columns], y)

                if cb_num_rfecv.n_features_ > 0:
                    new_features = np.array(temp_columns)[cb_num_rfecv.support_]
                    if verbose:
                        logger.info("RFECV selected %d additional feature(s): %s", cb_num_rfecv.n_features_, new_features)
                    for feature in new_features:
                        selected_vars.append(self.feature_names_in_.index(feature))
                else:
                    if verbose:
                        logger.info("RFECV selected no additional features.")

        # ---------------------------------------------------------------------------------------------------------------
        # Assign support
        # ---------------------------------------------------------------------------------------------------------------

        self.support_ = np.array(selected_vars)
        self.fallback_used_ = False
        # n_features_ reports the column count produced by transform() = raw selected + engineered (replayable via _engineered_recipes_). Higher-order
        # engineered features without a replayable recipe were already warned about above and are NOT counted (they don't appear in transform output).
        n_engineered_out = len(self._engineered_recipes_)
        if selected_vars or n_engineered_out:
            self.n_features_ = len(selected_vars) + n_engineered_out
        else:
            self.n_features_ = 0
            # Empty support_ fallback: rank by raw MI(X_j, y) so downstream pipelines don't crash on 0-feature
            # transform output. Only triggers when min_features_fallback >= 1 (off by default).
            _min_fb = int(getattr(self, "min_features_fallback", 0) or 0)
            if _min_fb >= 1 and self.n_features_in_ > 0:
                try:
                    # Rank by cached confident MI with the target; take top-K. cached_MIs may not be populated;
                    # re-compute from the original frame as a last resort.
                    _raw_mi = []
                    for _i in range(self.n_features_in_):
                        _key = (_i, -1)  # (feature_idx, target_idx=-1 is target)
                        _mi = self.cached_MIs.get(_key, 0.0) if hasattr(self, "cached_MIs") else 0.0
                        _raw_mi.append((_i, float(_mi)))
                    # Sort by MI desc; pick top-K
                    _raw_mi.sort(key=lambda kv: kv[1], reverse=True)
                    _topk = [i for i, _ in _raw_mi[:_min_fb]]
                    if _topk:
                        self.support_ = np.array(_topk)
                        self.n_features_ = len(_topk)
                        self.fallback_used_ = True
                        logger.warning(
                            "MRMR: screening returned 0 features; falling back "
                            "to top-%d by raw MI(X_j, y). Set "
                            "min_features_fallback=0 to disable. "
                            "fallback_used_=True is set on the estimator.",
                            self.n_features_,
                        )
                except Exception as _exc:
                    logger.warning(
                        "MRMR fallback to top-K MI failed: %s. Returning empty support_.",
                        _exc,
                    )

        # ---------------------------------------------------------------------------------------------------------------
        # Report FS results
        # ---------------------------------------------------------------------------------------------------------------

        if verbose:
            predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
            predictors_str = textwrap.shorten(predictors_str, width=300)
            logger.info("MRMR+ selected %d out of %d features: %s", self.n_features_, self.n_features_in_, predictors_str)

        self.signature = signature
        self.ran_out_of_time_ = ran_out_of_time

        # Store self in process-wide cache so cloned MRMR instances fit on the same (X, y) arrays can replay
        # this fitted state instead of re-running cat-FE + permutation. Bound the LRU by ``fit_cache_max``;
        # the default (4) covers a typical model suite without thrashing and long-lived workers no longer leak.
        if _cache_key is not None:
            MRMR._FIT_CACHE[_cache_key] = self
            MRMR._FIT_CACHE.move_to_end(_cache_key)
            _cap = int(getattr(self, "fit_cache_max", 4) or 4)
            while len(MRMR._FIT_CACHE) > _cap:
                MRMR._FIT_CACHE.popitem(last=False)
        return self


    def _run_fe_step(
        self,
        *,
        # Mutable state from MRMR.fit (returned updated as tuple).
        data, cols, nbins, X,
        target_names, target_indices,
        selected_vars, categorical_vars,
        classes_y, classes_y_safe, freqs_y,
        cached_MIs, cached_confident_MIs,
        unary_transformations, binary_transformations,
        engineered_features, checked_pairs,
        # Parallel dict (name -> EngineeredRecipe) populated as new columns are added to data / cols so
        # transform() can replay them on test data. Mutated in place; MRMR.fit reads it after the FE loop
        # and copies surviving recipes into self._engineered_recipes_. ``None`` skips recipe construction.
        engineered_recipes=None,
        times_spent,
        num_fs_steps,
        # Service.
        n_jobs, prefetch_factor, parallel_kwargs,
        _is_polars_input, verbose,
        # FE config (frozen per fit).
        fe_max_steps, fe_npermutations, fe_max_pair_features,
        fe_print_best_mis_only, fe_min_nonzero_confidence,
        fe_min_engineered_mi_prevalence,
        fe_good_to_best_feature_mi_threshold,
        fe_max_external_validation_factors,
        fe_min_pair_mi, fe_min_pair_mi_prevalence,
        fe_smart_polynom_iters, fe_smart_polynom_optimization_steps,
        fe_min_polynom_degree, fe_max_polynom_degree,
        fe_min_polynom_coeff, fe_max_polynom_coeff,
        # Preset-name snapshot so recipes can rebuild the correct registry at replay time. Default "minimal"
        # matches MRMR.__init__ defaults; callers that override via self.fe_unary_preset / self.fe_binary_preset
        # get the actual values threaded through by fit().
        fe_unary_preset: str = "minimal",
        fe_binary_preset: str = "minimal",
    ):
        """One Feature Engineering iteration. Extracted from ``MRMR.fit`` for testability and FE experimentation.

        Returns ``None`` if the FE step should not run (empty-screen + ``fe_fallback_to_all=False``); otherwise
        ``(data, cols, nbins, X, selected_vars, n_recommended_features)``. ``n_recommended_features == 0`` signals
        the outer loop to stop. Private; external callers should use ``MRMR.fit()`` or ``MRMR.fit_transform()``.
        """
        if verbose:
            logger.info("MRMR+ selected %d out of %d features before the Feature Engineering step.", len(selected_vars), self.n_features_in_)

        if len(selected_vars) == 0:
            if self.fe_fallback_to_all:
                logger.info("Proceeding with all features though (fe_fallback_to_all=True).")
                selected_vars = np.array([cols.index(col) for col in cols if col not in target_names])
            else:
                logger.info("Skipping Feature Engineering (screening returned 0 features and fe_fallback_to_all=False).")
                return None

        if _is_polars_input:
            import polars as pl  # noqa: F401  -- pl is used in the polars dispatch branches below

        n_recommended_features = 0
        if verbose >= 2:
            logger.info("Computing prospective FE pairs...")

        if self.fe_ntop_features:
            numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
        else:
            numeric_vars_to_consider = selected_vars

        numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

        # Honor factors_to_use / factors_names_to_use in the FE step too; intersect the FE pool with the user's
        # restriction so the contract matches the screening step.
        if self.factors_to_use is not None:
            numeric_vars_to_consider = numeric_vars_to_consider & set(self.factors_to_use)
        if self.factors_names_to_use is not None:
            allowed = {cols.index(n) for n in self.factors_names_to_use if n in cols}
            numeric_vars_to_consider = numeric_vars_to_consider & allowed

        # `combinations(...)` is consumed lazily by tqdmu (small path) or by
        # `_lazy_chunks` (large path). Pair count is closed-form, avoiding
        # `list(combinations(...))` materialisation (O(k^2) tuples, ~300 MB at
        # k=5000) before chunking even starts.
        _k = len(numeric_vars_to_consider)
        n_pairs = (_k * (_k - 1)) // 2

        if verbose:
            logger.info("Feature Engineering: Computing MIs of %d most prospective feature pairs...", n_pairs)

        if _k < 50:
            compute_pairs_mis(
                all_pairs=tqdmu(
                    combinations(numeric_vars_to_consider, 2),
                    total=n_pairs,
                    desc="getting pairs MIs",
                    leave=False,
                    mininterval=5,
                ),
                data=data,
                target_indices=target_indices,
                nbins=nbins,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                fe_npermutations=fe_npermutations,
                cached_confident_MIs=cached_confident_MIs,
                cached_MIs=cached_MIs,
                fe_min_pair_mi=fe_min_pair_mi,
                fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
            )
        else:
            chunk_size = max(1, n_pairs // (n_jobs * prefetch_factor))
            dicts = parallel_run(
                [
                    delayed(compute_pairs_mis)(
                        all_pairs=chunk,
                        data=data,
                        target_indices=target_indices,
                        nbins=nbins,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                        fe_npermutations=fe_npermutations,
                        cached_confident_MIs=cached_confident_MIs,
                        cached_MIs=cached_MIs,
                        fe_min_pair_mi=fe_min_pair_mi,
                        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
                    )
                    for chunk in _lazy_chunks(combinations(numeric_vars_to_consider, 2), chunk_size)
                ],
                n_jobs=n_jobs,
                **parallel_kwargs,
            )
            for next_dict in dicts:
                cached_MIs.update(next_dict)

        # ---------------------------------------------------------------------------------------------------------------
        # For every pair of factors (A,B), select ones having MI((A,B),Y)>MI(A,Y)+MI(B,Y). Such ones must posess more special connection!
        # ---------------------------------------------------------------------------------------------------------------

        vars_usage_counter = defaultdict(int)
        prospective_pairs = {}
        for raw_vars_pair, pair_mi in sort_dict_by_value(cached_MIs).items():
            if len(raw_vars_pair) == 2:
                if raw_vars_pair in checked_pairs:
                    continue
                if raw_vars_pair[0] in numeric_vars_to_consider and raw_vars_pair[1] in numeric_vars_to_consider:
                    ind_elems_mi_sum = cached_MIs[(raw_vars_pair[0],)] + cached_MIs[(raw_vars_pair[1],)]
                    # Guard against ZeroDivisionError: when both individual features have zero MI with target
                    # (canonical 3-way XOR case: MI(x_i, y) = 0 for all i but the joint signal exists), any positive pair_mi
                    # qualifies as infinite uplift -- keep the pair.
                    if ind_elems_mi_sum <= 0:
                        if pair_mi > 0:
                            uplift = float("inf")
                            if verbose >= 2:
                                logger.info(
                                    f"Factors pair {raw_vars_pair} has zero individual MI but pair_mi={pair_mi:.4f} -- canonical hidden-pair case (e.g. XOR), keeping for FE"
                                )
                            prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                            for var in raw_vars_pair:
                                vars_usage_counter[var] += 1
                        continue
                    if pair_mi > ind_elems_mi_sum * fe_min_pair_mi_prevalence:
                        uplift = pair_mi / ind_elems_mi_sum
                        if verbose >= 2:
                            logger.info(
                                f"Factors pair {raw_vars_pair} will be considered for Feature Engineering, {ind_elems_mi_sum:.4f}->{pair_mi:.4f}, rat={uplift:.2f}"
                            )
                        prospective_pairs[(raw_vars_pair, pair_mi)] = vars_usage_counter[raw_vars_pair[0]] + vars_usage_counter[raw_vars_pair[1]]
                        for var in raw_vars_pair:
                            vars_usage_counter[var] += 1

        # Now need to sort prospective_pairs by the uplift, to check most promising pairs within the time budget.
        # Also need to sort them by their members usage frequency+members ids sum. this way, their splitting will benefit more from caching.
        prospective_pairs = sort_dict_by_value(prospective_pairs, reverse=True)

        if fe_smart_polynom_iters:
            # Orthogonal-polynomial pair FE: Chebyshev default basis (empirically robust); tight coef range [-2, 2],
            # fixed degree per study, L2 regularisation, identity-baseline filter. Override basis via
            # ``self.fe_polynomial_basis``. See feature_selection.filters.hermite_fe and bench_polynomial_bases.
            from .hermite_fe import optimise_hermite_pair

            for (raw_vars_pair, _pair_mi), _uplift in prospective_pairs.items():
                if _is_polars_input:
                    vals_a = X[:, raw_vars_pair[0]].to_numpy()
                    vals_b = X[:, raw_vars_pair[1]].to_numpy()
                else:
                    vals_a = X.iloc[:, raw_vars_pair[0]].values
                    vals_b = X.iloc[:, raw_vars_pair[1]].values

                # Skip if any column is constant after extraction.
                if np.std(vals_a) < 1e-12 or np.std(vals_b) < 1e-12:
                    continue

                # Iterations correspond to study restarts: run fe_smart_polynom_iters full optimise_hermite_pair
                # calls with different seeds and pick the global best.
                best_res = None
                fe_basis = getattr(self, "fe_polynomial_basis", "chebyshev")
                fe_mi_est = getattr(self, "fe_mi_estimator", "plugin")
                fe_optimizer = getattr(self, "fe_optimizer", "cma")
                fe_warm_start = getattr(self, "fe_warm_start", True)
                fe_multi_fidelity = getattr(self, "fe_multi_fidelity", True)
                for seed_offset in range(fe_smart_polynom_iters):
                    res = optimise_hermite_pair(
                        x_a=vals_a, x_b=vals_b, y=classes_y,
                        discrete_target=True,  # mRMR target is always classification-class-coded
                        max_degree=fe_max_polynom_degree,
                        min_degree=fe_min_polynom_degree,
                        n_trials=fe_smart_polynom_optimization_steps,
                        coef_range=(fe_min_polynom_coeff, fe_max_polynom_coeff),
                        l2_penalty=getattr(self, "fe_hermite_l2_penalty", 0.05),
                        n_neighbors=None,  # auto by n
                        seed=42 + seed_offset,
                        sweep_degrees=True,
                        basis=fe_basis,
                        mi_estimator=fe_mi_est,
                        optimizer=fe_optimizer,
                        warm_start=fe_warm_start,
                        multi_fidelity=fe_multi_fidelity,
                    )
                    if res is not None and (best_res is None or res.mi > best_res.mi):
                        best_res = res

                if best_res is not None and verbose:
                    logger.info(
                        "Polynomial-pair FE (%s): pair=%s baseline_mi=%.4f best_mi=%.4f uplift=%.2fx "
                        "degree=%d bf=%s |c|2=(%.2f, %.2f)",
                        best_res.basis, raw_vars_pair, best_res.baseline_mi, best_res.mi,
                        best_res.uplift, best_res.degree_a, best_res.bin_func_name,
                        np.linalg.norm(best_res.coef_a), np.linalg.norm(best_res.coef_b),
                    )
                # Integration of best_res into data/cols/engineered_features happens via the regular
                # check_prospective_fe_pairs pipeline below; we log the gain here for diagnostics only. A tighter
                # integration that injects best_res.transform(...) directly into data is future work.

        else:
            original_cols = {i: self.feature_names_in_.index(col) for i, col in enumerate(cols) if col in self.feature_names_in_}
            if verbose >= 1:
                logger.info("Checking %d most prospective_pairs for feature engineering...", len(prospective_pairs))
            if len(X) < 50_000 or len(prospective_pairs) < 2:
                prospective_additions = check_prospective_fe_pairs(
                    prospective_pairs,
                    X,
                    unary_transformations,
                    binary_transformations,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    num_fs_steps,
                    cols,
                    original_cols,
                    fe_max_steps,
                    fe_npermutations,
                    fe_max_pair_features,
                    fe_print_best_mis_only,
                    fe_min_nonzero_confidence,
                    fe_min_engineered_mi_prevalence,
                    fe_good_to_best_feature_mi_threshold,
                    fe_max_external_validation_factors,
                    numeric_vars_to_consider,
                    self.quantization_nbins,
                    self.quantization_method,
                    self.quantization_dtype,
                    times_spent,
                    verbose,
                )
            else:

                prospective_additions = {}
                desired_nitems = max(1, len(prospective_pairs) // (n_jobs * prefetch_factor))

                jobs_list = []

                nitems = 0
                cur_dict = {}
                for key, value in prospective_pairs.items():
                    nitems += 1
                    cur_dict[key] = value
                    if nitems >= desired_nitems:
                        jobs_list.append(cur_dict)
                        nitems = 0
                        cur_dict = {}
                if cur_dict:
                    jobs_list.append(cur_dict)

                if verbose:
                    logger.info(
                        f"Using {desired_nitems:_} items per thread for checking {len(prospective_pairs):_} prospective_pairs with gain>{fe_min_pair_mi_prevalence:.2f}."
                    )

                dicts = parallel_run(
                    [
                        delayed(check_prospective_fe_pairs)(
                            chunk,
                            X,
                            unary_transformations,
                            binary_transformations,
                            classes_y,
                            classes_y_safe,
                            freqs_y,
                            num_fs_steps,
                            cols,
                            original_cols,
                            fe_max_steps,
                            fe_npermutations,
                            fe_max_pair_features,
                            fe_print_best_mis_only,
                            fe_min_nonzero_confidence,
                            fe_min_engineered_mi_prevalence,
                            fe_good_to_best_feature_mi_threshold,
                            fe_max_external_validation_factors,
                            numeric_vars_to_consider,
                            self.quantization_nbins,
                            self.quantization_method,
                            self.quantization_dtype,
                            times_spent,
                            verbose,
                        )
                        for chunk in jobs_list
                    ],
                    # max_nbytes=0,
                    n_jobs=n_jobs,
                    **parallel_kwargs,
                )
                for next_dict in dicts:
                    prospective_additions.update(next_dict)

            for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, new_nbins, messages) in prospective_additions.items():
                if this_pair_features:
                    engineered_features.update(this_pair_features)
                    if verbose:
                        for mes in messages:
                            logger.info(mes)
                        # logger.info(f"Features {new_cols} are recommended to use as new features!")
                    if fe_max_steps > 1:
                        new_vals = np.empty(shape=(len(X), len(this_pair_features)), dtype=self.quantization_dtype)
                        for j in range(len(this_pair_features)):
                            new_vals[:, j] = discretize_array(
                                arr=transformed_vals[:, j],
                                n_bins=self.quantization_nbins,
                                method=self.quantization_method,
                                dtype=self.quantization_dtype,
                            )
                        data = np.append(data, new_vals, axis=1)
                        # ``nbins`` is a numpy.ndarray (returned by categorize_dataset), so plain ``+`` does
                        # element-wise addition / broadcasting, not concatenation. Use np.concatenate so nbins
                        # grows in lockstep with data.shape[1] (otherwise screen_predictors trips its
                        # targets_data.shape[1] == len(targets_nbins) assertion when engineered cols feed back).
                        nbins = np.concatenate([
                            np.asarray(nbins),
                            np.asarray(new_nbins, dtype=nbins.dtype),
                        ])
                        cols = cols + new_cols
                        if _is_polars_input:
                            # Polars is immutable: with_columns returns a new frame sharing buffers; caller's X untouched.
                            _series_to_add = [
                                pl.Series(col, transformed_vals[:, j])
                                for j, col in enumerate(new_cols)
                            ]
                            X = X.with_columns(_series_to_add)
                        else:
                            for col in new_cols:
                                X[col] = transformed_vals[:, j]

                        # Build EngineeredRecipe for each newly-appended column so transform() can replay it.
                        # Only runs when columns were added (fe_max_steps > 1). Best-effort: parents that are
                        # themselves engineered (higher-order interaction) are skipped (nested replay is future work).
                        if engineered_recipes is not None:
                            from .engineered_recipes import build_unary_binary_recipe
                            for config, _j in this_pair_features:
                                # config = (transformations_pair, bin_func_name, i)
                                # transformations_pair = ((var_a_idx, unary_a_name),
                                #                        (var_b_idx, unary_b_name))
                                transformations_pair, bin_func_name, _ = config
                                (var_a_idx, unary_a_name) = transformations_pair[0]
                                (var_b_idx, unary_b_name) = transformations_pair[1]
                                # Map cols-index -> feature_names_in_-name. If a parent is itself engineered,
                                # cols[var] is not in feature_names_in_; skip with a warning rather than produce
                                # an unreplayable recipe.
                                src_a_name_raw = cols[var_a_idx]
                                src_b_name_raw = cols[var_b_idx]
                                if (
                                    src_a_name_raw not in self.feature_names_in_
                                    or src_b_name_raw not in self.feature_names_in_
                                ):
                                    if verbose:
                                        logger.info(
                                            "Skipping recipe construction for nested "
                                            "engineered feature '%s' (parents %s, %s "
                                            "are not in feature_names_in_); higher-"
                                            "order replay is future work.",
                                            get_new_feature_name(config, cols),
                                            src_a_name_raw, src_b_name_raw,
                                        )
                                    continue
                                eng_name = get_new_feature_name(config, cols)
                                engineered_recipes[eng_name] = build_unary_binary_recipe(
                                    name=eng_name,
                                    src_a_name=src_a_name_raw,
                                    src_b_name=src_b_name_raw,
                                    unary_a_name=unary_a_name,
                                    unary_b_name=unary_b_name,
                                    binary_name=bin_func_name,
                                    unary_preset=fe_unary_preset,
                                    binary_preset=fe_binary_preset,
                                    quantization_nbins=self.quantization_nbins,
                                    quantization_method=self.quantization_method,
                                    quantization_dtype=self.quantization_dtype,
                                )

                    n_recommended_features += len(this_pair_features)

                # TODO: handle factors_to_use / factors_names_to_use threading here.
                checked_pairs.add(raw_vars_pair)

        return data, cols, nbins, X, selected_vars, n_recommended_features


    def get_feature_names_out(self, input_features=None):
        """sklearn-1.x transformer protocol. Returns the names of selected features as an ndarray of str,
        matching transform() output cols. When ``self._engineered_recipes_`` is non-empty, their names are
        appended AFTER the base-feature names; order matches transform() output column order.
        """
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This MRMR instance is not fitted yet. Call 'fit' before "
                "using 'get_feature_names_out'."
            )
        support = self.support_
        engineered_names = [r.name for r in getattr(self, "_engineered_recipes_", [])]
        if len(support) == 0 and not engineered_names:
            return np.array([], dtype=object)
        # MRMR's support_ is integer indices into feature_names_in_
        if len(support) > 0 and isinstance(support[0], (bool, np.bool_)):
            base_names = [n for n, s in zip(self.feature_names_in_, support) if s]
        else:
            base_names = [self.feature_names_in_[i] for i in support]
        return np.asarray(base_names + engineered_names, dtype=object)

    def transform(self, X, y=None):
        # Unfitted -> NotFittedError (sklearn-canonical); previously returned X unchanged, masking config bugs.
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This MRMR instance is not fitted yet. Call 'fit' before "
                "using 'transform'."
            )
        support = self.support_
        recipes = getattr(self, "_engineered_recipes_", [])

        # Fast-path: when MRMR selected every input column AND produced zero engineered recipes, transform()
        # is the identity. Return X unchanged to avoid a full-copy X[selected_cols] and to let the caller detect
        # the no-op (checked via ``_mlframe_identity_equivalent`` downstream).
        if not recipes and hasattr(X, "shape"):
            _support_arr = np.asarray(support)
            if len(_support_arr) > 0 and isinstance(_support_arr.flat[0], (bool, np.bool_)):
                _n_selected = int(np.count_nonzero(_support_arr))
            else:
                _n_selected = len(_support_arr)
            if _n_selected == X.shape[1]:
                return X

        # Empty-base-support: if no base AND no engineered recipes, return legacy empty output. Recipes but no
        # base falls through and only the engineered cols come out.
        if len(support) == 0 and not recipes:
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, []]
            else:
                return X[:, np.array([], dtype=np.intp)]

        if isinstance(X, pd.DataFrame):
            if ENSURE_ARROW_DF_SUPPORT:
                # Use column names to support Arrow-backed DataFrames (from polars zero-copy conversion).
                # Arrow-backed DFs don't support .iloc[:, integer_array] reliably.
                selected_cols = [self.feature_names_in_[i] for i in support]
                # Fuzz-caught: in a multi-model suite where a fitted MRMR is reused across models, val_df passed
                # to transform can have a different column set than the train_df MRMR fit on (e.g. after
                # _filter_categorical_features narrowed the frame). Detect column drift explicitly so we raise
                # with actionable context instead of an unhelpful KeyError.
                missing = [c for c in selected_cols if c not in X.columns]
                if missing:
                    # Raise on column drift (symmetric with RFECV.transform); silent intersection masked
                    # downstream column-set bugs. Callers wanting degradation can catch and intersect themselves.
                    raise RuntimeError(
                        f"MRMR.transform: {len(missing)}/{len(selected_cols)} "
                        f"selected columns missing from input X ({missing[:8]}). "
                        f"The fitted support_ no longer matches the input's "
                        f"physical columns; an upstream step (constant-col "
                        f"removal / imputer drop / OD filter) is mutating the "
                        f"column set BETWEEN fit and transform. Investigate."
                    )
                base_out = X[selected_cols]
            else:
                base_out = X.iloc[:, support]
            return self._append_engineered(base_out, X, recipes)
        else:
            base_out = X[:, support]
            out = self._append_engineered(base_out, X, recipes)
            # When X is polars and Pipeline has set_output(transform="pandas"), sklearn's PandasAdapter calls
            # pd.DataFrame(out, ...) which (unlike polars' own .to_pandas()) does NOT preserve Arrow-backed
            # dtypes: pl.Enum / pl.Categorical collapse to object; pl.Float32 turns to object-of-strings
            # ('1.23' as text), and downstream HGB/XGB/SimpleImputer raise "could not convert string to float".
            # Convert ourselves via polars' Arrow-preserving .to_pandas() whenever the consumer expects pandas.
            try:
                from sklearn.utils._set_output import _get_output_config
                _cfg = _get_output_config("transform", estimator=self)
                _want_pandas = (_cfg.get("dense") or "default") == "pandas"
            except Exception:
                _want_pandas = False
            if _want_pandas:
                try:
                    import polars as _pl
                    if isinstance(out, _pl.DataFrame):
                        out = out.to_pandas()
                except ImportError:
                    pass
            return out

    def _append_engineered(self, base_out, X, recipes):
        """Append engineered-recipe columns onto ``base_out``.

        Inputs:
        - ``base_out``: DataFrame / ndarray already restricted to the base
          ``support_`` columns. Caller's dtype is preserved.
        - ``X``: full input frame (DataFrame, ndarray, or polars). Recipe
          replay reads source columns from here BY NAME, so X must
          contain at least every ``recipe.src_names`` entry.
        - ``recipes``: list of EngineeredRecipe to replay.

        Behaviour:
        - Returns ``base_out`` unchanged when ``recipes`` is empty (legacy
          path, zero overhead).
        - For pandas / polars input, engineered cols are appended as
          named columns.
        - For ndarray input, engineered cols are stacked as additional
          numeric columns; column names are not preserved (caller is
          expected to use ``get_feature_names_out`` for naming).
        """
        if not recipes:
            return base_out

        # Lazy import keeps import-time cost off MRMR users who never engage FE.
        from .engineered_recipes import apply_recipe

        # K-way recipes ship a chained-lookup payload (extras ``chain_lookups`` / ``chain_nuniqs``) so they
        # replay on test data alongside pair recipes. The only filter is the legacy ``requires_refit_for_replay``
        # flag retained for OLD pickles that pre-date the chain payload.
        replayable = [
            r for r in recipes
            if r.extra.get("chain_lookups") is not None
            or not r.extra.get("requires_refit_for_replay")
        ]
        if len(replayable) < len(recipes) and self.verbose:
            logger.info(
                "MRMR.transform: skipping %d legacy k-way recipe(s) "
                "without chained-lookup payload (pre-D3 pickle). Re-fit "
                "to materialise the chain.",
                len(recipes) - len(replayable),
            )
        if not replayable:
            return base_out
        recipes = replayable
        engineered_cols = [apply_recipe(r, X) for r in recipes]
        if isinstance(base_out, pd.DataFrame):
            # ``copy=False`` would risk mutating caller's view (base_out is a view into pandas X). Build a narrow
            # new frame: engineered cols are fresh ndarrays anyway, only base cols share buffers with X.
            engineered_df = pd.DataFrame(
                {r.name: col for r, col in zip(recipes, engineered_cols)},
                index=base_out.index,
            )
            return pd.concat([base_out, engineered_df], axis=1)

        # Try polars first if available (avoid hard import).
        try:
            import polars as _pl
            if isinstance(base_out, _pl.DataFrame):
                return base_out.with_columns([
                    _pl.Series(r.name, col) for r, col in zip(recipes, engineered_cols)
                ])
        except ImportError:
            pass

        # ndarray fallback: hstack engineered cols. Names are lost but row order matches get_feature_names_out.
        engineered_arr = np.column_stack(engineered_cols) if engineered_cols else np.empty((base_out.shape[0], 0))
        if base_out.size == 0:
            return engineered_arr.astype(base_out.dtype, copy=False)
        return np.hstack([base_out, engineered_arr.astype(base_out.dtype, copy=False)])


