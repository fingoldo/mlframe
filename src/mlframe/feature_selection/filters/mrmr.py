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

# Top-level helpers (histogram + fingerprint/hash + replay + chunker) live in
# ``_mrmr_fingerprints.py``; re-imported below so the parent module and
# downstream callers continue to resolve historical names.
from ._mrmr_fingerprints import (  # noqa: E402,F401
    _astropy_histogram,
    histogram,
    _canonicalise_dtype_str,
    _mrmr_compute_y_fingerprint_sample,
    _mrmr_compute_x_fingerprint,
    _hashable_params_signature,
    _content_array_signature,
    _target_to_numpy_values,
    _target_name_signature,
    _full_y_content_hash,
    _replay_fitted_state,
    _lazy_chunks,
    _MRMR_IDENTITY_FP_CACHE,
    _MRMR_IDENTITY_FP_LOCK,
    _MRMR_BATCH_PRECOMPUTE_MAX_K,
    _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
)

# Bulk of the in-package imports that the ``MRMR`` class body relies on. The
# sibling fingerprint module also imports them so it can stand alone; Python
# caches module imports so the duplication has zero runtime cost.
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

from ._internals import (  # noqa: F401
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort  # noqa: F401
from .discretization import (  # noqa: F401
    categorize_dataset,
    discretize_array,
)
from .feature_engineering import (  # noqa: F401
    FE_DEFAULT_SUBSAMPLE_N,
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from .gpu import init_kernels, mi_direct_gpu  # noqa: F401
from .info_theory import (  # noqa: F401
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from .permutation import distribute_permutations, mi_direct, parallel_mi  # noqa: F401
from .evaluation import (  # noqa: F401
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import (  # noqa: F401
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .screen import postprocess_candidates, screen_predictors  # noqa: F401

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

    # Process-wide cache of fitted state, keyed by (content_sig(X), content_sig(y), params_signature). When the
    # training suite iterates over models (clone()ing the pre-pipeline MRMR each time, stripping
    # ``_cat_fe_cache_``), subsequent fits on the same arrays hit this cache and skip the full cat-FE +
    # permutation work. LRU-bounded via ``OrderedDict`` + ``fit_cache_max`` (default 4) so long-lived workers
    # do not leak; ``MRMR._FIT_CACHE.clear()`` between suites still drains the lot. Cache hit: replay all
    # fitted attributes onto ``self`` and return early; constructor params are NEVER overwritten (the key
    # already includes the params signature, so a hit guarantees matching state).
    _FIT_CACHE: "OrderedDict[tuple, MRMR]" = OrderedDict()

    @classmethod
    def clear_fit_cache(cls) -> int:
        """Drain the process-wide MRMR fit cache. Returns the entry count that was dropped. Call between
        suites (model retraining boundary, JupyterHub kernel reuse, web-service request boundary) when
        long-lived workers must release fitted-MRMR memory. Without this, the cache holds up to
        ``fit_cache_max`` (default 4) full MRMR instances per process for as long as the process lives."""
        n = len(cls._FIT_CACHE)
        cls._FIT_CACHE.clear()
        return n

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
        # ``None`` (legacy default) triggers process-stable but seedable random_state derivation
        # downstream (see ``_resolve_target_prefix``: uses pid ^ id(self) instead of touching the
        # numpy global RNG). For bit-exact reproducibility across runs / mlflow hash stability, pass
        # an explicit integer seed.
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
        # 2026-05-18 audit-fixes flip #1 (``fe_npermutations`` 0->3):
        # pre-fix value 0 combined with ``fe_min_nonzero_confidence=1.0``
        # made the FE confidence gate STRUCTURALLY UNREACHABLE (confidence
        # = ``1 - failures/npermutations`` is undefined at npermutations=0).
        # Features with weak individual MI were silently dropped BEFORE
        # the polynom-FE block could evaluate them as a pair, even when
        # the pair carried genuine interaction signal. Flipping to 3
        # aligns FE permutation count with screening-side
        # ``full_npermutations=3``; cost ~3% FE wall time.
        fe_npermutations=3,
        fe_ntop_features=0,
        fe_unary_preset="minimal",
        fe_binary_preset="minimal",
        # 2026-05-18 audit-fixes flip #2 (``fe_max_pair_features`` 1->10):
        # pre-fix only ONE pair per FE step was evaluated. On a dataset
        # with 50 features (1225 candidate pairs ranked by prevalence-
        # passing pair-MI) only the top-ranked pair was promoted to
        # transformation evaluation. Multi-interaction problems (3+
        # independent interacting pairs) lost 2/3 of the signal. 10 is a
        # measure-first compromise: per-pair compute is cheap (<200ms on
        # n=200k with default unary/binary presets), 10 covers most
        # practical pair-interaction structures, AND the gates further
        # downstream (``fe_min_engineered_mi_prevalence``) filter the
        # eventual injection set.
        fe_max_pair_features: int = 10,
        # 2026-05-18 audit-fixes flip #1 (``fe_min_nonzero_confidence``
        # 1.0->0.99): pre-fix 1.0 required EVERY permutation to clear the
        # null-hypothesis test exactly, making the gate unreachable at any
        # noise level. 0.99 matches the screening-side
        # ``min_nonzero_confidence`` default so both stages apply equally
        # strict statistical rigor without the unreachable-gate trap.
        fe_min_nonzero_confidence: float = 0.99,
        fe_min_pair_mi: float = 0.001,
        fe_min_pair_mi_prevalence: float = 1.05,  # transformations of what exactly pairs of factors we consider, at all. mi of entire pair must be at least that higher than the mi of its individual factors.
        fe_min_engineered_mi_prevalence: float = 0.98,  # mi of transformed pair must be at least that higher than the mi of the entire pair
        fe_good_to_best_feature_mi_threshold: float = 0.98,  # when multiple good transformations exist for the same factors pair.
        fe_max_external_validation_factors: int = 0,  # how many other factors to validate against
        fe_max_polynoms: int = 0,
        fe_print_best_mis_only: bool = True,
        # 2026-05-18 audit-fixes HIGH#4 default-flip evaluation result:
        # ``profiling/bench_polynom_fe_default_flip.py`` measured on three
        # canonical "polynom-FE should help" scenarios (XOR, saddle,
        # symmetric-linear-plus-interaction): 0 / 3 cleared the >= 20%
        # downstream-LightGBM AUC lift bar; 3 / 3 showed no-harm
        # (|delta| <= 1%).
        #
        # Diagnostic: on the symmetric-linear-plus-interaction scenario
        # screening keeps ALL 4 features (support_size=4) AND polynom-FE
        # evaluates 5 pairs - the pipeline is healthy. The null result
        # is because the downstream evaluator (LightGBM) already
        # discovers multiplicative interactions natively via tree
        # splits, making polynom-FE engineered columns redundant.
        # Polynom-FE's value would be on LINEAR downstream models
        # (Ridge / Lasso / MLP without interaction layers) where pair
        # interactions must be explicitly engineered.
        #
        # Decision: keep default 0. Users with tree-based downstreams
        # rarely need polynom-FE; users with linear downstreams should
        # opt in with a positive value after measuring on their data.
        fe_smart_polynom_iters: int = 0,
        fe_smart_polynom_optimization_steps: int = 1000,
        # 2026-05-18: subsample inside the CMA-ES / Optuna inner search to
        # bound per-pair MI compute on production-size frames. cProfile on
        # n=500k showed ``_eval_coef_pair`` + ``_plugin_mi_classif_njit``
        # dominate (32% + 22% of fit time); each MI call scales linearly
        # with n. At n=4M with default config (10 restarts x 200 trials
        # x C(25,2)=300 pairs) per-pair cost projects to ~5 min, ~25
        # hours serial.
        #
        # Default 200_000 (raised from 100_000 on 2026-05-18 after the
        # n=1M bench showed 100k could lose 1 hermite feature on
        # marginal seeds while 200k kept it). The FINAL injected column
        # is still computed from FULL source so no train-time precision
        # is lost.
        #
        # Set to ``None`` / 0 / negative to disable (use full data).
        # 2026-05-21: unified with check_prospective_fe_pairs via the shared
        # FE_DEFAULT_SUBSAMPLE_N constant; both FE entry points now scale their
        # MI-sweep buffer with the same knob. Re-tune in feature_engineering.py
        # to land both sites consistently.
        fe_smart_polynom_subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
        # 2026-05-21 (CRITICAL #2): subsample rows for check_prospective_fe_pairs's
        # MI sweep. The hoisted shared scratch buffer scales linearly with n; on
        # n=4M with the medium preset it lands at ~17.6 GiB and crashes the suite.
        # Bench (bench_fe_pair_subsample_accuracy.py): jaccard=1.0 vs full-n at
        # 50k+, 0.88 at 5k. Default 200_000 matches fe_smart_polynom_subsample_n
        # for cross-block consistency. 0 = use full data (legacy).
        fe_check_pairs_subsample_n: int = FE_DEFAULT_SUBSAMPLE_N,
        # 2026-05-18 audit-fixes flip #3 (``fe_min_polynom_degree`` 3->1):
        # pre-fix the Hermite/Chebyshev optimiser was locked to a minimum
        # cubic basis. Degree-1 (linear product, the XOR / multiplicative
        # interaction case) and degree-2 (saddle / circle / quadratic
        # response) were structurally excluded. The optimiser then forced
        # simple interactions into overfit-prone cubic+ representations,
        # wasting Optuna budget AND injecting columns with higher
        # variance than necessary. ``min_degree=1`` lets the optimiser
        # discover the actual signal degree; the test in
        # ``test_biz_cma_es_finds_xor_optimum`` already verified the
        # optimiser converges on degree=2 for XOR when range is open.
        fe_min_polynom_degree: int = 1,
        fe_max_polynom_degree: int = 8,
        fe_min_polynom_coeff: float = -10.0,
        fe_max_polynom_coeff: float = 10.0,
        # 2026-05-22: explicit __init__ params for the fe-* knobs that the
        # polynom-pair FE inner search consults via getattr(self, ...).
        # Pre-fix these were accessible by setting them as attributes after
        # construction (``mrmr.fe_optimizer = 'cma_batch'``) but the
        # ``FeatureSelectionConfig.mrmr_kwargs`` validator rejected them as
        # unknown because they weren't in this signature. Adding here lets
        # users pass them through ``mrmr_kwargs={'fe_optimizer': ...}`` via
        # the suite config.
        fe_hermite_l2_penalty: float = 0.05,
        fe_polynomial_basis: str = "chebyshev",
        fe_mi_estimator: str = "plugin",
        # 2026-05-22: cma_batch is the new default (20.58x faster than
        # optuna, within_1%=1.00 vs all other optimizers on a 12-pair bench).
        # See profiling/bench_polynom_optimizers.py.
        fe_optimizer: str = "cma_batch",
        fe_warm_start: bool = True,
        fe_multi_fidelity: bool = True,
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
        # Pipeline-fatal fallback: when screening yields zero features (all MI ~= 0), the default
        # ``min_features_fallback=1`` keeps the single highest-MI column so ``support_`` is never empty -- empty
        # support causes downstream estimators to crash with a 0-column transform output. Set to 0 explicitly to
        # restore the legacy "let the pipeline fail loudly" semantics. Chosen features are flagged via
        # ``self.fallback_used_``.
        min_features_fallback: int = 1,
        # Cat-FE (categorical feature interactions): single dataclass consolidating ~22 cat_fe_* knobs.
        # ``None`` = default CatFEConfig() with ``enable=True`` and conservative production settings (cat-FE
        # shows measurable wins; XOR biz_value test, 0 regressions). Restore legacy via CatFEConfig(enable=False).
        cat_fe_config=None,
        # Bound on the process-wide _FIT_CACHE. Strong refs hold every fitted MRMR; long-lived workers (web services, JupyterHub kernels) leaked memory unboundedly pre-2026-05-15. Default 4 covers a typical model suite (RFECV+MRMR x catboost+linear+mlp) without thrashing.
        fit_cache_max: int = 4,
        # 2026-05-18 #5: adaptive FE threshold relaxation. When the first-pass
        # FE produces 0 engineered features (typically because pair-level MI
        # is near the individual-MI sum on heavily-correlated features and
        # the engineered candidate cannot clear the strict
        # ``fe_min_engineered_mi_prevalence=0.98`` gate), retry ONCE with
        # thresholds scaled by ``fe_adaptive_relax_factor``. Default True
        # (Accuracy/perf over legacy) -- the retry adds at most ~10% to FE
        # wall time and skips the expensive Hermite Optuna re-run because
        # those results are already cached / injected from the first pass.
        # Set False to restore the historical "0 features = give up" path.
        fe_adaptive_threshold_relax: bool = True,
        # T2#12 2026-05-18 default 0.9 rationale: the strict gates sit at
        # ``fe_min_engineered_mi_prevalence=0.98`` and
        # ``fe_min_pair_mi_prevalence=1.05``. A retry factor of 0.9
        # brings those to 0.882 and 0.945 respectively - just under the
        # baseline-MI sum, where many tightly-correlated engineered
        # features land. Smaller factors (0.7-0.8) flood the FE pool
        # with low-uplift engineered cols and slow downstream MRMR
        # screening with no measurable gain. Larger (>=0.95) are
        # indistinguishable from no retry. Tune only after observing
        # the per-pass engineered count in your data.
        fe_adaptive_relax_factor: float = 0.9,
        # When ``mrmr_skip_when_prior_was_identity=True``, this controls whether
        # legitimately distinct targets on the same X must produce a separate cache
        # slot. With True (default) the cache key adds a y-fingerprint sample so
        # target A's identity result cannot poison target B; the per-call cost is a
        # ~5us blake2b over a 1000-element y sample. False reverts to X-only keying
        # (the user's original TVT scenario where 2 composites on same X both
        # returned identity); safe only when the operator can guarantee that
        # identity-on-target-A implies identity-on-target-B.
        mrmr_identity_cache_include_y: bool = True,
        # 2026-05-18 #2: cross-target identity cache. When True and a prior fit on the SAME X-fingerprint produced an identity result (all input columns selected, zero engineered features), subsequent calls with a different y short-circuit the entire FE pipeline. Production TVT log showed 88 min of MRMR work that produced identity output, then ANOTHER MRMR call on the same X for a composite target -- second call would also be 88 min wasted.
        #
        # Default flipped False -> True 2026-05-18 (Accuracy/perf over legacy): on multi-target suites the second MRMR call on the same X usually hits the cache and saves the full FE pipeline run-time. The conservative case (prior identity result was wrong for the new target) is rare in practice because composite-target y values are highly correlated with the raw y -- if MRMR found nothing on raw y, it almost never finds something on the residual.
        mrmr_skip_when_prior_was_identity: bool = True,
        # hidden
        stop_file: str = "stop",
    ):

        # checks
        if n_jobs == -1:
            n_jobs = psutil.cpu_count(logical=False)

        if parallel_kwargs is None:
            # backend="threading": joblib uses ThreadPoolExecutor in the same
            # process instead of the default loky ProcessPoolExecutor. Data
            # arrays are shared in memory (zero copy, no memmap_folder, no
            # paging-file pressure); numba kernels in mi_direct /
            # parallel_mi_prange / compute_mi_from_classes already release the
            # GIL so threads run truly parallel on CPU cores. Pre-fix iter-371
            # 1M cb multiclass + MRMR + binary=medium: 3 joblib loky workers
            # each memmap'd the 1M-row dataset (3GB RAM total), then Windows
            # WinError 1455 (paging file too small) cascaded into MemoryError +
            # dangling joblib_memmapping_folder temp files that the resource
            # tracker could not clean up.
            # ``max_nbytes`` is silently ignored by joblib's threading backend (it's a memmap-spill threshold for loky workers only). Kept here as a no-op for symmetry with the loky branch above; documented so a future reader doesn't misread it as a live tuning knob.
            parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES, backend="threading")

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

    # ``_validate_string_params`` + ``_validate_inputs`` are implemented
    # in ``_mrmr_validate_transform.py`` and bound onto this class at the
    # bottom of this module.

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
        # When random_seed is None the user wants ENTROPY-seeded randomness for the resample;
        # the prior fallback hardcoded ``0`` which deterministically returned the same draw on
        # every call -- two A/B comparisons expecting independent random resamples got identical
        # samples. ``np.random.default_rng(None)`` pulls fresh entropy from the OS.
        rng = np.random.default_rng(self.random_seed)
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
        old code path byte-for-byte (regression sentry).

        groups: ACCEPTED FOR API COMPAT BUT NOT CONSUMED. MRMR's MI estimator treats each row
        independently; group-stratified MI / group-resample permutations would require modifying
        screen_predictors and info_theory.merge_vars. Until that's implemented, a non-None groups
        argument emits a UserWarning so callers wrapping MRMR with GroupKFold know they need to
        precompute per-group MI themselves. The signature is retained for symmetry with sklearn's
        SelectorMixin.fit and to let sklearn Pipeline routing forward the kwarg without TypeError.

        Wrapper / _fit_impl forwarding asymmetry: ``sample_weight`` is CONSUMED at this wrapper level (via ``_maybe_resample_for_sample_weight`` before the ``_fit_impl`` call); ``groups`` is FORWARDED into ``_fit_impl`` which then silently drops them. A future refactor moving ``groups`` consumption into ``_fit_impl`` must also remove or downgrade the wrapper-level warning, otherwise the two ends would emit duplicate / contradictory messages.

        2026-05-18 #2: cross-target identity cache. When a prior fit on the SAME X (same columns + same dtypes) produced an identity result (all input columns selected + zero engineered features), subsequent calls with a different y short-circuit the 80+ min FE pipeline and return identity-equivalent output. Opt-in via ``mrmr_skip_when_prior_was_identity=True``."""
        if groups is not None:
            warnings.warn(
                "MRMR.fit received groups but the current implementation does NOT consume them; "
                "MI is estimated per-row. For grouped MI estimation, wrap MRMR with a per-group "
                "selector and aggregate manually. Pass groups=None to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        self._pandas_frame_for_target_cleanup = None
        self._target_names_for_cleanup = None

        # 2026-05-18 #2 cross-target identity cache.
        _identity_skip = bool(getattr(self, "mrmr_skip_when_prior_was_identity", False))
        _include_y = bool(getattr(self, "mrmr_identity_cache_include_y", False))
        # Suite caller (train_mlframe_models_suite) can inject a ctx-scoped dict here via
        # ``_mlframe_identity_cache_override_`` so cache lifetime is bounded by the suite
        # rather than the process. When absent, fall back to the module-level dict for
        # cross-suite reuse (CI matrices opt in via mrmr_identity_cache_scope="process").
        _cache_dict = getattr(self, "_mlframe_identity_cache_override_", None)
        if _cache_dict is None:
            _cache_dict = _MRMR_IDENTITY_FP_CACHE
        _x_fp = None
        if _identity_skip:
            _x_fp = _mrmr_compute_x_fingerprint(X)
            if _include_y:
                # T3#18: stricter cache key -- include y-fingerprint so legitimately distinct targets on same X get separate slots.
                _x_fp = _x_fp + "_yfp_" + _mrmr_compute_y_fingerprint_sample(y)
            with _MRMR_IDENTITY_FP_LOCK:
                _prior_was_identity = _cache_dict.get(_x_fp)
            if _prior_was_identity is True:
                logger.info(
                    "[MRMR] cross-target identity cache HIT for X fingerprint=%s -- "
                    "prior fit returned identity, skipping ~minute(s) of FE pipeline.",
                    _x_fp,
                )
                self._fit_identity_shortcut(X)
                self._fit_sample_weight_ = None
                return self

        # Persist user-supplied weights so cached _cat_fe_state_ / FE replay can introspect; cache key
        # below already excludes weights so the FS-cache reuse contract stays intact when the suite
        # caller decides to gate weight-aware MRMR behind FeatureSelectionConfig.use_sample_weights_in_fs.
        self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        X, y = self._maybe_resample_for_sample_weight(X, y, self._fit_sample_weight_)
        try:
            result = self._fit_impl(X, y, groups, **fit_params)
            # Stash X-fingerprint -> identity-bool in cross-target cache so a SUBSEQUENT fit (different y, same X) can early-skip the FE pipeline.
            if _identity_skip and _x_fp is not None:
                try:
                    _is_id = (
                        getattr(self, "support_", None) is not None
                        and len(self.support_) == X.shape[1]
                        and len(getattr(self, "_engineered_features_", []) or []) == 0
                    )
                    with _MRMR_IDENTITY_FP_LOCK:
                        _cache_dict[_x_fp] = bool(_is_id)
                    if _is_id:
                        logger.info(
                            "[MRMR] cross-target identity cache STORED for X fingerprint=%s "
                            "(no features dropped, no engineered features); subsequent "
                            "fits on this X will short-circuit.",
                            _x_fp,
                        )
                except Exception:
                    pass
            return result
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

    def _fit_identity_shortcut(self, X) -> None:
        """Populate the fit-result attributes as if MRMR returned the input X unchanged.

        Used by the cross-target identity cache (2026-05-18 #2): when a previous fit on the SAME X returned identity (all input columns selected, zero engineered features), subsequent calls with a different y can skip the entire FE pipeline since the only y-dependent thing -- the selected feature subset -- is forced to "all input columns".
        """
        n_cols = X.shape[1] if X.ndim > 1 else 1
        self.support_ = np.arange(n_cols, dtype=np.int64)
        self.feature_names_in_ = (
            X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
            if hasattr(X, "columns") else [f"f{i}" for i in range(n_cols)]
        )
        self._engineered_features_ = []
        self._engineered_recipes_ = {}
        self.n_features_in_ = int(n_cols)
        self.fallback_used_ = False
        # Mark for transform() to know we're in shortcut state. Some downstream code looks at .signature; safe-default to a stable string.
        self.signature = f"_mrmr_identity_shortcut_n{n_cols}"

    # ``_fit_impl`` is implemented in ``_mrmr_fit_impl.py`` and bound onto
    # this class at the bottom of this module.



    # ``_run_fe_step`` is implemented in ``_mrmr_fe_step.py`` and bound
    # onto this class at the bottom of this module.

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

    # ``transform`` + ``_append_engineered`` are implemented in
    # ``_mrmr_validate_transform.py`` and bound onto this class at the
    # bottom of this module.





# Bind the carved-out method onto the class. _fit_impl lives in
# _mrmr_fit_impl.py as a module-level function taking self as the
# first argument; the binding here makes MRMR._fit_impl resolve to that
# function so self._fit_impl(...) call sites keep working unchanged.
from ._mrmr_fit_impl import _fit_impl as _fit_impl_func  # noqa: E402
MRMR._fit_impl = _fit_impl_func

from ._mrmr_fe_step import _run_fe_step as _run_fe_step_func  # noqa: E402
MRMR._run_fe_step = _run_fe_step_func


from ._mrmr_validate_transform import (  # noqa: E402
    _validate_string_params as _validate_string_params_func,
    _validate_inputs as _validate_inputs_func,
    transform as _transform_func,
    _append_engineered as _append_engineered_func,
)
MRMR._validate_string_params = _validate_string_params_func
MRMR._validate_inputs = _validate_inputs_func
MRMR.transform = _transform_func
MRMR._append_engineered = _append_engineered_func
