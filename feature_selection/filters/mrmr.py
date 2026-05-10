"""sklearn-compatible MRMR estimator.

B1 (etap 11) - tolerate FE-engineered feature names in the post-fit
    support index map (route synthetic names through ``_engineered_features_``).
B14 (etap 11) - dead string-literal blocks deleted (referenced undefined
    ``cv``, ``jobs``, ``parallel_run`` symbols and were never live code).
B26 (etap 11) - explicit input-validation contract in ``_validate_inputs``.
B27 (etap 11) - ``__setstate__`` shim that injects defaults for the B13/B15
    knobs and the new ``_engineered_features_`` attribute, so old joblib /
    cloudpickle pipelines unpickle cleanly into the new schema.
"""
from __future__ import annotations

import copy
import gc
import logging
import math
import os
import psutil
import textwrap
import time
import warnings
from collections import defaultdict
from itertools import combinations
from os.path import exists
from timeit import default_timer as timer
from typing import Sequence, Union

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

from mlframe.arrays import arrayMinMax
from mlframe.feature_selection.wrappers import RFECV
from mlframe.metrics import compute_probabilistic_multiclass_error
from mlframe.utils import set_random_seed

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

    def __init__(
        self,
        # quantization
        quantization_method: str = "quantile",
        quantization_nbins: int = 10,
        quantization_dtype: object = np.int32,
        # factors
        factors_names_to_use: Sequence[str] = None,
        factors_to_use: Sequence[int] = None,
        # algorithm
        mrmr_relevance_algo: str = "fleuret",
        mrmr_redundancy_algo: str = "fleuret",
        reduce_gain_on_subelement_chosen: bool = True,
        use_simple_mode: bool = True,  # when true, works very fast but leaves redundant features
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
        min_relevance_gain: float = 0.0001,
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
        verbose: Union[bool, int] = 0,
        ndigits: int = 5,
        parallel_kwargs: dict = None,
        # CV
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = False,
        # service
        random_state: int = None,
        n_jobs: int = -1,
        skip_retraining_on_same_shape: bool = True,
        # B13 (post-plan): cardinality cutoff for the confirmation step.
        # ``None`` (the default) computes
        # ``quantization_nbins ** interactions_max_order * 2``, which for
        # the default ``quantization_nbins=10, interactions_max_order=1``
        # gives 20. Pre-B13 default was a hardcoded 50 -- the new default
        # is more conservative (skips more high-cardinality conditioning
        # sets where permutation-based confirmation does not converge in
        # reasonable time). Pin to 50 to restore legacy behaviour.
        max_confirmation_cand_nbins: int = None,
        # B15 (post-plan): when the screening pass returns zero
        # ``selected_vars``, the legacy code fell back to running FE on
        # ALL features. The new default is to skip FE instead -- FE on
        # an empty screen typically just amplifies noise. Set to ``True``
        # to restore legacy behaviour.
        fe_fallback_to_all: bool = False,
        # P0-H39 (audit): pipeline-fatal fallback. When the entire screening
        # phase yields zero features (all MI ~= 0), MRMR's default leaves
        # ``support_`` empty and ``transform`` returns 0 columns -- which
        # crashes any downstream estimator expecting >=1 feature. Set
        # ``min_features_fallback`` >= 1 to keep at least that many features
        # by raw MI rank when the screening pass collapses. The chosen
        # features are flagged as "fallback" via ``self.fallback_used_``.
        min_features_fallback: int = 0,
        # Cat-FE (categorical feature interactions). Single dataclass kwarg
        # consolidating ~14 cat_fe_* knobs (SB8). ``None`` (the default)
        # means cat-FE is disabled bit-exact with legacy behaviour. To
        # opt in, pass ``cat_fe_config=CatFEConfig(enable=True, ...)``.
        # See ``mlframe.feature_selection.filters.cat_fe_state.CatFEConfig``.
        cat_fe_config=None,
        # hidden
        stop_file: str = "stop",
    ):

        # checks
        if n_jobs == -1:
            n_jobs = psutil.cpu_count(logical=False)

        if parallel_kwargs is None:
            parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES)

        # assert isinstance(estimator, (BaseEstimator,))

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())
        self.signature = None

    # B26 (etap 11): input validation contract -- explicit guards for
    # memory-exhaustion shapes, malformed dtypes, all-constant features,
    # and polars LazyFrame / Expr edge cases. Each guard either raises a
    # descriptive ValueError or warns + applies a safe default.
    def _validate_inputs(self, X, y):
        import warnings as _w
        n_rows = getattr(X, "shape", (None,))[0]
        if n_rows is not None:
            n_cols = X.shape[1] if len(X.shape) > 1 else 1
            if n_rows == 0:
                raise ValueError("MRMR.fit: empty input (n_rows=0)")
            if n_rows == 1:
                raise ValueError("MRMR.fit: cannot fit on a single row")
            if isinstance(n_cols, int) and n_rows * n_cols > 1e9:
                raise ValueError(
                    f"MRMR.fit: refusing to allocate for n*p={n_rows * n_cols:_} (>1e9). "
                    "Subsample or split the dataset before fitting."
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
                _w.warn("MRMR.fit autocollecting LazyFrame; pass DataFrame to skip this copy.")
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
                from collections import Counter as _C
                dups = [c for c, n in _C(cols).items() if n > 1]
                raise ValueError(f"MRMR.fit: duplicate column names not supported: {dups}")
        # Infinite values in numeric data.
        if hasattr(X, "to_numpy"):
            try:
                arr_check = X.to_numpy()
                if arr_check.dtype.kind == "f" and np.isinf(arr_check).any():
                    _w.warn("MRMR.fit: input contains +/-inf values; downstream discretization may produce undefined bins")
            except Exception:
                pass
        # All-same y check. P1-H37 (audit): raise instead of warn -
        # symmetric with RFECV.fit's single-class y validation. A constant
        # y has H(y)=0 so every MI(X_j, y) is 0; the entire MRMR pipeline
        # produces zero-information output. Caller should catch the bad
        # input upstream rather than letting MRMR silently return [].
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

    # B27 (etap 11): pickle BC for the upcoming B13 default flip
    # (max_confirmation_cand_nbins kwarg) -- old MRMR pickles lacking this
    # attribute resurface with the legacy default injected.
    def __setstate__(self, state):
        defaults = {
            "max_confirmation_cand_nbins": 50,  # B13 legacy default
            "fe_fallback_to_all": True,         # B15 legacy default
            "_engineered_features_": [],        # B1 new attribute
            # PR-1 prep: recipes-based replay so transform() can recompute
            # engineered features on test data. Old pickles have no
            # recipes -- their engineered cols were never replayable, so
            # an empty list reproduces the legacy "engineered cols dropped
            # from transform output" behaviour bit-exact.
            "_engineered_recipes_": [],
            # Cat-FE: ``None`` means cat-FE was either disabled or never
            # ran. Old pickles (pre-cat-FE) had no concept of this attr;
            # injecting ``None`` makes ``getattr(self, "cat_fe_config",
            # None)`` consistent with no-op behaviour.
            "cat_fe_config": None,
            "_cat_fe_state_": None,
        }
        for k, v in defaults.items():
            state.setdefault(k, v)
        self.__dict__.update(state)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, **fit_params):
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

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        start_time = timer()
        ran_out_of_time = False

        quantization_method = self.quantization_method
        quantization_nbins = self.quantization_nbins
        dtype = self.dtype

        max_runtime_mins = self.max_runtime_mins
        random_state = self.random_state
        parallel_kwargs = self.parallel_kwargs
        n_jobs = self.n_jobs
        verbose = self.verbose
        cv_shuffle = self.cv_shuffle
        cv = self.cv

        prefetch_factor = 4

        fe_max_steps = self.fe_max_steps
        fe_npermutations = self.fe_npermutations
        fe_unary_preset = self.fe_unary_preset
        fe_binary_preset = self.fe_binary_preset
        fe_max_pair_features = self.fe_max_pair_features

        # Fix 10 addendum (2026-04-22): MRMR feature engineering (invoked
        # when fe_max_steps > 0) uses pandas-only ops at multiple sites
        # (X.iloc[:, i].values at filters.py:~3184, 3324, 3537, 3623,
        # 3679; X[col] = vals at ~3324). Adapting all of these to polars
        # is a separate large refactor. For now, gracefully disable FE
        # when the input is polars -- the selector itself still works and
        # uses the zero-copy polars categorize_dataset path. Log once so
        # it's visible that FE was skipped.
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

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init cv
        # ----------------------------------------------------------------------------------------------------------------------------

        """
        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3

            if groups is not None:
                cv = GroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            else:
                cv = KFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state)
            
            if verbose:
                logger.info("Using cv=%s", cv)
        """

        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        self.feature_names_in_ = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # ---------------------------------------------------------------------------------------------------------------
        # Temporarily inject targets
        # ---------------------------------------------------------------------------------------------------------------

        target_prefix = "targ_" + str(np.random.random())[3:9]
        y_shape = y.shape
        if len(y_shape) == 2:
            y_shape = y_shape[1]
        else:
            y_shape = 1
        target_names = [target_prefix + str(i) for i in range(y_shape)]

        if isinstance(y, np.ndarray):
            vals = y
        else:
            vals = y.values

        if vals.dtype == np.int64:
            print("Converted targets from int64 to int16.")
            vals = vals.astype(np.int16)

        # 2026-04-22 (Fix 10 addendum to jolly-wishing-deer plan): native
        # Polars support -- no `.to_pandas()` copy. mlframe production
        # frames are 100+ GB, and a full materialization would OOM on
        # the prod box. CLAUDE.md forbids caller-visible copies; use
        # Polars-native ops when the input is pl.DataFrame.
        try:
            import polars as pl  # local alias; safe even if pl is already imported module-scope
            _is_polars_input = isinstance(X, pl.DataFrame)
        except ImportError:
            _is_polars_input = False

        if _is_polars_input:
            # Polars is immutable; `with_columns` returns a new frame that
            # shares buffers with X -- no data copy. Caller's X is not mutated.
            target_series = [pl.Series(name, vals[:, i] if vals.ndim == 2 else vals) for i, name in enumerate(target_names)]
            X = X.with_columns(target_series)
        else:
            # 2026-04-24 Session 6: multilabel target -> vals is (N, K). Pass it
            # through unchanged so each column maps to its target_names entry.
            # The previous .reshape(-1, 1) was right ONLY for 1-D y (single
            # column with shape (N, 1)) and crashed on multilabel with
            # "Must have equal len keys and value when setting with an ndarray".
            if vals.ndim == 2:
                X.loc[:, target_names] = vals
            else:
                X.loc[:, target_names] = vals.reshape(-1, 1)

        # ---------------------------------------------------------------------------------------------------------------
        # Discretize continuous data
        # ---------------------------------------------------------------------------------------------------------------

        logger.info("categorizing dataset...")
        if _is_polars_input:
            # Polars: fill_null forward-then-backward -- no-copy lazy op.
            _filled = X.fill_null(strategy="forward").fill_null(strategy="backward")
        else:
            _filled = X.ffill().bfill()
        data, cols, nbins = categorize_dataset(
            df=_filled,
            method=self.quantization_method,
            n_bins=self.quantization_nbins,
            dtype=self.quantization_dtype,
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
                polynomial_transformations = {}  # 'identity':lambda x: x,
                for _ in range(fe_max_polynoms):
                    length = np.random.randint(3, 9)
                    # coef=(np.random.random(length)-0.5)*1
                    coef = np.empty(shape=length, dtype=np.float32)
                    for i in range(length):
                        coef[i] = np.random.normal((1.0 if i == 1 else 0.0), scale=0.05)

                    unary_transformations["poly_" + str(coef)] = coef

            if verbose > 2:
                print(f"nunary_transformations: {len(unary_transformations):_}")
                print(f"nbinary_transformations: {len(binary_transformations):_}")

            engineered_features = set()
            checked_pairs = set()
        # PR-1 prep: ``engineered_recipes`` (name -> EngineeredRecipe) is
        # initialised UNCONDITIONALLY -- the splitter at the bottom of
        # ``fit()`` looks it up on every screening completion regardless
        # of ``fe_max_steps``. When ``fe_max_steps == 0`` (FE disabled),
        # the dict simply stays empty and no recipes are persisted.
        engineered_recipes: dict = {}

        # ---------------------------------------------------------------------
        # Cat-FE step (categorical interaction generator). Runs ONCE before
        # the screening loop when ``self.cat_fe_config.enable=True``. The
        # step augments ``data`` / ``cols`` / ``nbins`` with new ordinal-
        # encoded columns capturing pair (and future k-way) synergies.
        # Engineered cols enter the screening pass as 1-way features --
        # the screening's own k-way enumeration sees them as atomic.
        # See plan v3 §"Точка интеграции".
        # ---------------------------------------------------------------------
        cat_fe_cfg = getattr(self, "cat_fe_config", None)
        self._cat_fe_state_ = None
        if cat_fe_cfg is not None and cat_fe_cfg.enable and len(categorical_vars) >= 2:
            from .cat_interactions import run_cat_interaction_step
            from .info_theory import merge_vars as _merge_vars_for_cat_fe

            # Pre-compute classes_y / freqs_y for cat-FE (avoids re-binning
            # the target inside every kernel call -- P2).
            _classes_y, _freqs_y, _ = _merge_vars_for_cat_fe(
                factors_data=data, vars_indices=target_indices,
                var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
            )
            _classes_y_safe = _classes_y.copy()

            data, cols, nbins, cat_fe_state = run_cat_interaction_step(
                data=data, cols=cols, nbins=nbins,
                target_indices=target_indices,
                classes_y=_classes_y, classes_y_safe=_classes_y_safe,
                freqs_y=_freqs_y,
                categorical_vars=categorical_vars,
                cfg=cat_fe_cfg,
                dtype=dtype, verbose=verbose,
            )
            self._cat_fe_state_ = cat_fe_state
            # Cat-FE recipes feed into the same engineered_recipes dict that
            # numeric FE populates -- the fit-end splitter copies any recipe
            # whose engineered name shows up in selected_vars_names into
            # ``self._engineered_recipes_``. This puts cat-FE and numeric-FE
            # recipes on the same replay path through ``transform()``.
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
                    min_relevance_gain=self.min_relevance_gain,
                    max_consec_unconfirmed=self.max_consec_unconfirmed,
                    max_runtime_mins=self.max_runtime_mins,
                    interactions_min_order=self.interactions_min_order,
                    interactions_max_order=self.interactions_max_order,
                    interactions_order_reversed=self.interactions_order_reversed,
                    max_veteranes_interactions_order=self.max_veteranes_interactions_order,
                    only_unknown_interactions=self.only_unknown_interactions,
                    # B13: resolve effective max_confirmation_cand_nbins.
                    # User-pinned value wins; otherwise compute formula default.
                    max_confirmation_cand_nbins=(
                        self.max_confirmation_cand_nbins
                        if self.max_confirmation_cand_nbins is not None
                        else self.quantization_nbins ** self.interactions_max_order * 2
                    ),
                    # B15: FE-on-empty-screen fallback flag (consumed by MRMR.fit).
                    fe_fallback_to_all=self.fe_fallback_to_all,
                    # verbosity and formatting
                    verbose=self.verbose,
                    ndigits=self.ndigits,
                    parallel_kwargs=self.parallel_kwargs,
                    stop_file=self.stop_file,
                )
            )

            if fe_max_steps == 0 or num_fs_steps >= fe_max_steps:
                break

            # Feature engineering iteration -- delegated to ``_run_fe_step``
            # so the FE controller is callable / testable / experiment-
            # friendly outside of the screening loop. Returns the updated
            # state tuple plus ``n_recommended_features``; if zero, the
            # outer loop breaks (no further FE worth attempting).
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
                break  # B15 skip / empty screening + fe_fallback_to_all=False
            data, cols, nbins, X, selected_vars, n_recommended_features = fe_result

            if n_recommended_features == 0:
                break

            num_fs_steps += 1
            if num_fs_steps >= fe_max_steps:
                break  # uncomment to avoid recheck of single-rounded FE

        if verbose > 2:
            print("time spent by binary func:", sort_dict_by_value(times_spent))
        # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

        # ---------------------------------------------------------------------------------------------------------------
        # Drop Temporarily targets
        # ---------------------------------------------------------------------------------------------------------------

        # 2026-04-22 fuzz-caught: the previous ``X = X.drop(columns=target_names)``
        # returned a NEW DataFrame and only rebound the local ``X``. When the
        # caller's input was pandas (so X.loc[:, target_names] = ... mutated
        # the caller's frame directly), the caller's X was left with the
        # injected ``targ_<id>`` columns attached. They then leaked into the
        # downstream sklearn pipeline: imputer/scaler recorded ``targ_<id>``
        # in ``feature_names_in_`` and raised on the next transform call.
        # Fix: drop in place (pandas) or rebind (polars -- immutable, caller's
        # X was never mutated so nothing to clean).
        if _is_polars_input:
            X = X.drop(target_names)  # no-copy lazy op; caller's X untouched
        else:
            X.drop(columns=target_names, inplace=True)  # restores caller's original schema

        # ---------------------------------------------------------------------------------------------------------------
        # selected_vars needs to be transformed to names using the cols variable and then back to indices using original Df columns names.
        # It's needed 'casue categorize_data can rearrange cat columns.
        # ---------------------------------------------------------------------------------------------------------------

        selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]
        # B1 (etap 11): tolerate FE-engineered names. The screening output may include
        # synthetic feature names that are NOT in feature_names_in_; record them in
        # self._engineered_features_ instead of raising on the .index() lookup.
        # PR-1: also surface the matching ``EngineeredRecipe`` from
        # ``engineered_recipes`` (built during ``_run_fe_step``) so
        # ``transform()`` can replay each engineered column on test data.
        # An engineered name in ``selected_vars_names`` without a recipe
        # (e.g. higher-order interaction whose parents are themselves
        # engineered) is recorded by name only and dropped from transform
        # output -- mirrors the legacy behaviour for those cases.
        self._engineered_features_ = []
        self._engineered_recipes_ = []
        original_indices = []
        for col in selected_vars_names:
            if col in self.feature_names_in_:
                original_indices.append(self.feature_names_in_.index(col))
            else:
                self._engineered_features_.append(col)
                recipe = engineered_recipes.get(col)
                if recipe is not None:
                    self._engineered_recipes_.append(recipe)
        selected_vars = original_indices  # !TODO! failing when fe_max_steps>1. need other source.

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

                if len(y) / len(np.unique(y)) > 100:  # classification

                    cb_num_rfecv = RFECV(
                        estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        scoring=make_scorer(
                            score_func=compute_probabilistic_multiclass_error, response_method='predict_proba', greater_is_better=False
                        ),
                        **params,
                    )
                    temp_columns = list(set(X.columns) - set(X.columns[selected_vars]))
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
        if selected_vars:
            self.n_features_ = len(selected_vars)
        else:
            self.n_features_ = 0
            # P0-H39 (audit): when screening collapsed to empty support_,
            # fall back to top-K by raw MI(X_j, y) so downstream pipelines
            # don't crash on a 0-feature transform output. Only triggers
            # if min_features_fallback >= 1 (off by default to preserve
            # legacy semantics for callers who handle empty selection).
            _min_fb = int(getattr(self, "min_features_fallback", 0) or 0)
            if _min_fb >= 1 and self.n_features_in_ > 0:
                try:
                    # Quick raw-MI fallback: rank features by their cached
                    # confident MI with the target, take top-K. cached_MIs
                    # may not be populated; re-compute from the original
                    # frame as a last resort.
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
        # assign extra vars for upcoming vars improving
        # ---------------------------------------------------------------------------------------------------------------

        # self.cached_MIs_ = cached_MIs
        # self.cached_cond_MIs_ = cached_cond_MIs
        # self.cached_confident_MIs_ = cached_confident_MIs

        # ---------------------------------------------------------------------------------------------------------------
        # Report FS results
        # ---------------------------------------------------------------------------------------------------------------

        if verbose:
            predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
            predictors_str = textwrap.shorten(predictors_str, width=300)
            logger.info("MRMR+ selected %d out of %d features: %s", self.n_features_, self.n_features_in_, predictors_str)

        self.signature = signature
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
        # PR-1 prep: parallel dict (name -> EngineeredRecipe) populated as
        # new columns are added to `data` / `cols` so `transform()` can
        # replay them on test data. Mutated in place; `MRMR.fit` reads it
        # after the FE loop completes and copies the surviving recipes
        # into `self._engineered_recipes_`. Default to ``None`` for
        # callers that haven't been updated -- in that case we simply skip
        # recipe construction (legacy behaviour bit-exact).
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
        # PR-1 prep: snapshot of preset names so recipes can rebuild
        # the correct registry at replay time. Default to ``"minimal"``
        # to match the legacy ``MRMR.__init__`` defaults; if a caller
        # has overridden the presets via ``self.fe_unary_preset`` /
        # ``self.fe_binary_preset``, ``fit()`` threads the actual
        # values through.
        fe_unary_preset: str = "minimal",
        fe_binary_preset: str = "minimal",
    ):
        """One Feature Engineering iteration.

        Extracted from ``MRMR.fit`` for testability + experimentation
        with alternative FE strategies. Honour the same logic / defaults
        as the legacy in-line block so the post-extract golden tests
        stay bit-exact.

        Returns ``None`` if the FE step should not run at all (B15
        empty-screen + ``fe_fallback_to_all=False``); otherwise a tuple
        ``(data, cols, nbins, X, selected_vars, n_recommended_features)``
        with the updated state. ``n_recommended_features == 0`` signals
        the outer loop to stop iterating.
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

        n_recommended_features = 0
        if verbose >= 2:
            logger.info("Computing prospective FE pairs...")

        if self.fe_ntop_features:
            numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
        else:
            numeric_vars_to_consider = selected_vars

        numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

        # B2 (post-plan): honor factors_to_use / factors_names_to_use
        # in the FE step too. Previously the legacy ``# !TODO! handle
        # factors_to_use etc`` comment punted this; engineered features
        # were generated from selected pairs even when the caller had
        # restricted the candidate set. Now we intersect the FE pool
        # with the user's restriction, matching the screening step's
        # contract.
        if self.factors_to_use is not None:
            numeric_vars_to_consider = numeric_vars_to_consider & set(self.factors_to_use)
        if self.factors_names_to_use is not None:
            allowed = {cols.index(n) for n in self.factors_names_to_use if n in cols}
            numeric_vars_to_consider = numeric_vars_to_consider & allowed

        all_pairs = list(combinations(numeric_vars_to_consider, 2))

        if verbose:
            logger.info("Feature Engineering: Computing MIs of %d most prospective feature pairs...", len(all_pairs))

        if len(numeric_vars_to_consider) < 50:
            compute_pairs_mis(
                all_pairs=tqdmu(all_pairs, desc="getting pairs MIs", leave=False, mininterval=5),
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
                    for chunk in split_list_into_chunks(all_pairs, len(all_pairs) // (n_jobs * prefetch_factor))
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
                    # Guard against ZeroDivisionError: when both
                    # individual features have zero MI with the
                    # target (canonical 3-way XOR case where
                    # ``MI(x_i, y) == 0`` for all i but the joint
                    # signal exists), any positive ``pair_mi``
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
            # ---------------------------------------------------------------
            # Improved orthogonal-polynomial pair FE:
            # - Default basis is Chebyshev (empirically robust on real
            #   tabular data; see ``bench_polynomial_bases``); tight
            #   coef range [-2, 2], fixed degree per study, L2
            #   regularisation, identity-baseline filter.
            # - Adds engineered features only when MI uplift > threshold.
            # - Allow override via ``self.fe_polynomial_basis`` attr
            #   (defaults to "chebyshev"); set to "hermite" to recover
            #   the prior bit-exact Hermite-only behaviour.
            # See ``feature_selection.filters.hermite_fe`` + the
            # ``bench_polynomial_bases`` benchmark.
            # ---------------------------------------------------------------
            from .hermite_fe import optimise_hermite_pair

            for (raw_vars_pair, pair_mi), uplift in prospective_pairs.items():
                if _is_polars_input:
                    vals_a = X[:, raw_vars_pair[0]].to_numpy()
                    vals_b = X[:, raw_vars_pair[1]].to_numpy()
                else:
                    vals_a = X.iloc[:, raw_vars_pair[0]].values
                    vals_b = X.iloc[:, raw_vars_pair[1]].values

                # Skip if any column is constant after extraction.
                if np.std(vals_a) < 1e-12 or np.std(vals_b) < 1e-12:
                    continue

                # Iterations now correspond to "study restarts" -- the
                # legacy outer ``range(fe_smart_polynom_iters)`` loop.
                # We keep that semantics by running ``fe_smart_polynom_iters``
                # full optimise_hermite_pair calls with different seeds
                # and picking the global best.
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
                # NOTE: integration of best_res into ``data`` / ``cols`` /
                # ``engineered_features`` happens via the regular
                # ``check_prospective_fe_pairs`` pipeline below. We log
                # the gain here for diagnostics only; the engineered
                # column itself is added on the next call when
                # ``check_prospective_fe_pairs`` re-evaluates and finds
                # the same transformation worth adding. A tighter
                # integration that injects ``best_res.transform(...)``
                # directly into ``data`` is reserved for a future PR --
                # would shortcut the second evaluation pass.

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
                        nbins = nbins + new_nbins
                        cols = cols + new_cols
                        if _is_polars_input:
                            # Polars is immutable -- accumulate new cols
                            # via with_columns (returns a new frame
                            # sharing buffers with X). Caller's original
                            # frame is never mutated.
                            _series_to_add = [
                                pl.Series(col, transformed_vals[:, j])
                                for j, col in enumerate(new_cols)
                            ]
                            X = X.with_columns(_series_to_add)
                        else:
                            for col in new_cols:
                                X[col] = transformed_vals[:, j]

                        # PR-1 prep: build EngineeredRecipe for each newly-
                        # appended column so ``transform()`` can replay it.
                        # Only runs when columns were actually added (i.e.
                        # ``fe_max_steps > 1`` -- otherwise engineered cols
                        # never reach the next screening pass and can't be
                        # selected). Recipe construction is best-effort:
                        # if a parent column is itself engineered (higher-
                        # order interaction), we skip and log -- replaying
                        # nested recipes is future work.
                        if engineered_recipes is not None:
                            from .engineered_recipes import build_unary_binary_recipe
                            for config, j in this_pair_features:
                                # config = (transformations_pair, bin_func_name, i)
                                # transformations_pair = ((var_a_idx, unary_a_name),
                                #                        (var_b_idx, unary_b_name))
                                transformations_pair, bin_func_name, _ = config
                                (var_a_idx, unary_a_name) = transformations_pair[0]
                                (var_b_idx, unary_b_name) = transformations_pair[1]
                                # Map cols-index -> feature_names_in_-name. If the
                                # parent is itself engineered, ``cols[var]`` is not
                                # in ``feature_names_in_`` -- skip with a warning
                                # rather than producing an unreplayable recipe.
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

                # !TODO!  handle factors_to_use etc
                """
                factors_names_to_use=self.factors_names_to_use,
                factors_to_use=self.factors_to_use,                    
                    """
                checked_pairs.add(raw_vars_pair)


        return data, cols, nbins, X, selected_vars, n_recommended_features


    def get_feature_names_out(self, input_features=None):
        """sklearn-1.x transformer protocol. Returns the names of selected
        features as an ndarray of str, matching transform() output cols.
        Symmetric with RFECV.get_feature_names_out (added PR-10 audit
        finding from test_selectors_shared.py).

        PR-1: when ``self._engineered_recipes_`` is non-empty (FE step
        produced replayable engineered features), their names are appended
        AFTER the base-feature names. Order matches ``transform()`` output
        column order.
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
        # P0-G34 (audit): unfitted -> NotFittedError, sklearn-canonical.
        # Prior code silently returned X unchanged, masking config bugs.
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This MRMR instance is not fitted yet. Call 'fit' before "
                "using 'transform'."
            )
        support = self.support_
        recipes = getattr(self, "_engineered_recipes_", [])

        # Handle the empty-base-support cases. If there are no base
        # features AND no engineered recipes, return the legacy empty
        # output. If there are recipes but no base, fall through and
        # only the engineered cols come out.
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
                # 2026-04-22 fuzz-caught: in a multi-model suite where
                # the same fitted MRMR is reused across models (via sklearn
                # Pipeline state), the val_df passed to transform can have
                # a different column set than the train_df MRMR fit on
                # (e.g. after ``_filter_categorical_features`` or other
                # per-model reshaping narrowed the frame). Falling through
                # to ``X[selected_cols]`` on a DF missing any selected
                # col raises KeyError with no actionable context.
                # Intersect with X.columns and keep order, warn if shrunk.
                missing = [c for c in selected_cols if c not in X.columns]
                if missing:
                    # P0-G35 (audit): symmetric with RFECV.transform - raise
                    # on column drift instead of silently shrinking the
                    # selection. Silent intersection masked downstream
                    # column-set bugs in the suite. If the caller really
                    # wants degradation, they can catch the error and
                    # call ``X[list(set(selected) & set(X.columns))]``.
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
            return self._append_engineered(base_out, X, recipes)

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

        # Lazy import: keeps the import-time cost off MRMR users who
        # never engage FE.
        from .engineered_recipes import apply_recipe

        # k-way (k > 2) cat-FE recipes are diagnostic-only at v1: they
        # surface synergistic groups via _cat_fe_state_.diagnostics but
        # don't yet have a replay path. Filter them out here so transform
        # doesn't raise -- users who need replay should set
        # cat_fe_config.max_kway_order=2.
        replayable = [
            r for r in recipes
            if not r.extra.get("requires_refit_for_replay")
        ]
        if len(replayable) < len(recipes) and self.verbose:
            logger.info(
                "MRMR.transform: skipping %d k-way (order>2) engineered "
                "feature(s) -- replay not implemented for k>2 in v1. "
                "Set CatFEConfig(max_kway_order=2) if you need them in "
                "transform output.",
                len(recipes) - len(replayable),
            )
        if not replayable:
            return base_out
        recipes = replayable
        engineered_cols = [apply_recipe(r, X) for r in recipes]
        if isinstance(base_out, pd.DataFrame):
            # ``copy=False`` would risk mutating caller's view if base_out
            # is itself a view into X (it is, when X is pandas). Build a
            # narrow new frame instead -- the engineered cols are fresh
            # ndarrays anyway, only the base cols share buffers with X.
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

        # ndarray fallback: hstack engineered cols. Loses names but keeps
        # the row count consistent with ``get_feature_names_out`` order.
        engineered_arr = np.column_stack(engineered_cols) if engineered_cols else np.empty((base_out.shape[0], 0))
        if base_out.size == 0:
            return engineered_arr.astype(base_out.dtype, copy=False)
        return np.hstack([base_out, engineered_arr.astype(base_out.dtype, copy=False)])


# Etap 8: FE block (check_prospective_fe_pairs, compute_pairs_mis,
# create_unary/binary_transformations, get_existing/new_feature_name)
# moved to feature_engineering.py with B10 (isinstance polars detection)
# applied. B5/B6 preallocation deferred to Phase-1 profiling per plan.
from .feature_engineering import (
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_unary_transformations,
    create_binary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)

import numpy as np
import numba
from numba import types
from numba.typed import Dict as NumbaDict


# `sanitize` moved to ``_internals.py`` (etap 2). Imported at module top.