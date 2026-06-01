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
    _full_x_content_hash,
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

    Notes
    -----
    ``cluster_aggregate_enable`` (ON by default; gated so it is a no-op without genuine clusters) turns on clustered-feature aggregation: correlated
    "reflection" features (noisy copies of a hidden factor ``z``) are combined into one denoised
    aggregate (``mean_z`` / ``mean_inv_var`` / ``pca_pc1`` / ``factor_score`` / ``median``) that recovers
    ``z`` better than any single reflection. Adopted only if it beats the best member's MI with the
    target. Helps capacity-limited / linear downstreams, redundant sensor data, interpretability, and
    tight feature budgets; for tree/GBM downstreams expect no-harm rather than a lift (trees already
    average reflections via splits). ``augment`` adds the aggregate; ``replace`` also drops the members.
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
        # 2026-05-29 Wave 7: per-feature adaptive bin chooser. Default
        # ``'mdlp'`` (Fayyad-Irani 1993, with njit-accelerated kernel) is the
        # honest combined-ranking winner of the F1 leaderboard
        # (``|err vs truth| + noise_floor``: MDLP 0.107, Sturges 0.135,
        # quantile10 0.139, OptimalJoint 0.167, FD 0.175). MDLP is the only
        # strategy with a TRUE zero no-signal floor, which directly improves
        # MRMR's relevance gate against false-positive feature picks. Pass
        # ``nbins_strategy=None`` to restore the pre-2026-05-29 fixed
        # ``quantization_nbins`` quantile behaviour.
        # The MRMR hot path stays exclusively on the plug-in MI njit kernel
        # chain (mi_direct / fleuret / permutation); alternative MI estimator
        # families (KSG, neural, copula, aggregators) live in their own
        # modules for ad-hoc / benchmark use only and are explicitly NOT
        # wired into MRMR.fit().
        nbins_strategy: str = "mdlp",
        nbins_strategy_kwargs: dict = None,
        # 2026-05-30 Wave 8 — 10 new research-grade opt-in knobs (sibling modules):
        # F13 Chao-Shen entropy correction (Pawluszek-Filipiak 2025).
        #   'none' (default) | 'miller_madow' | 'chao_shen'
        mi_correction: str = "none",
        # A1 JMIM redundancy aggregator (Bennasar 2015). Alternative to Fleuret CMIM
        # ``min_k I(X_k; Y | Z_j)``; JMIM uses ``min_j I(X_k, X_j; Y)`` which
        # preserves synergy that CMIM rejects on multi-collinear groups.
        #   None (legacy) | 'jmim'
        redundancy_aggregator: str = None,
        # A3 MRwMR-BUR unique-relevance bonus (Gao 2022). Additive bonus on the
        # MRMR score for features whose marginal-y relevance cannot be explained
        # by any already-selected feature.
        bur_lambda: float = 0.0,
        # A2 RelaxMRMR 3-D MI redundancy (Vinh 2016). Adds ``I(X; Z_i; Z_j | Y)``
        # interaction term. Cost is O(|S|^2) 3-D plug-in MIs per candidate.
        relaxmrmr_alpha: float = 0.0,
        # C8 CMI-permutation stopping criterion (Yu & Príncipe 2019). Replaces
        # the ``min_relevance_gain_frac`` threshold with a permutation null test.
        cmi_perm_stop: bool = False,
        cmi_perm_n_permutations: int = 100,
        cmi_perm_alpha: float = 0.05,
        # C9 UAED universal elbow detector (Llorente 2023). Auto-pick subset
        # size from the CMI-gain curve when ``n_features=None``.
        uaed_auto_size: bool = False,
        # D10 Conditional Permutation Test (Berrett 2020). Permutes X CONDITIONAL
        # on Z preserving X|Z; valid p-values under arbitrary confounding.
        cpt_test: bool = False,
        cpt_n_permutations: int = 200,
        # E11 Cluster Stability Selection (Faletto-Bien 2022). Opt-in via
        # ``stability_selection_method='cluster'``. Default 'classic' is the
        # existing Meinshausen-Buhlmann + Shah-Samworth Wave-4 path.
        # E12 Complementary Pairs Stability (Shah-Samworth 2013) accessible
        # via ``stability_selection_method='complementary_pairs'``.
        stability_selection_method: str = "classic",
        stability_selection_corr_threshold: float = 0.8,
        stability_n_bootstrap: int = 50,
        stability_pi_threshold: float = 0.6,
        # F14 PID decomposition (Williams-Beer + Ince I_ccs). When enabled,
        # synergistic features bypass the standard redundancy gate.
        pid_synergy_bonus: float = 0.0,
        # 2026-05-28: MI normalization knob to combat the cardinality bias.
        # Raw I(X; Y) is bounded by min(H(X), H(Y)); high-cardinality features
        # (zip codes / hash IDs / 50-bin continuous) get inflated relevance.
        # Symmetric Uncertainty SU(X,Y) := 2*I/(H(X)+H(Y)) normalises to [0,1]
        # and removes the bias (Witten-Frank-Hall 2011).
        #   'none' (default): legacy raw MI scoring (bit-for-bit identical to pre-2026-05-28).
        #   'su'            : Symmetric Uncertainty on both unconditional + conditional steps.
        # Default 'none' preserves the regression sentry; flip to 'su' for mixed
        # cat-cardinality data (different binning per feature / target-encoded
        # cats at different K). See _info_theory.symmetric_uncertainty.
        mi_normalization: str = "none",
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
        # 2026-05-30: diminishing-returns gate. Stops greedy selection once the
        # current candidate's gain drops below this fraction of the FIRST
        # selected feature's gain. Catches "trailing noise" leakage on
        # imbalanced y / large n where tiny-but-statistically-positive gains
        # clear the absolute floor (Layer 13 finding: noise gain 0.0004 vs
        # signal gain 0.0176 - both cleared min_relevance_gain_frac * H(y) at
        # 1% imbalance, but noise is only 2.5% of signal). 0.0 disables; 0.05
        # = stop once gain falls below 5% of first gain. Applies from the
        # SECOND selected feature onward (the first feature is the anchor).
        min_relevance_gain_relative_to_first: float = 0.05,
        # 2026-05-30: Miller-Madow MI bias correction at the selection gate.
        # Plug-in MI overestimates by ~(|X|-1)*(|Y|-1)/(2n) for high-card
        # X (Miller 1955, Paninski 2003). On 1200-level user_id at n=2500
        # with binary y the bias is ~0.24 nats - enough to make pure noise
        # outrank real numeric signal (Layer 10 seed=101 hijack). True =
        # subtract MM bias from gains at the floor comparison; does NOT
        # mutate the raw mrmr_gains_ attr (downstream sees raw plug-in MI).
        cardinality_bias_correction: bool = True,
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
        # (the original scenario where 2 composites on same X both
        # returned identity); safe only when the operator can guarantee that
        # identity-on-target-A implies identity-on-target-B.
        mrmr_identity_cache_include_y: bool = True,
        # Cross-target identity cache. When True and a prior fit on the SAME X-fingerprint produced an identity result (all input columns selected, zero engineered features), subsequent calls with a different y short-circuit the entire FE pipeline. A prod log showed 88 min of MRMR work that produced identity output, then ANOTHER MRMR call on the same X for a composite target -- second call would also be 88 min wasted.
        #
        # Default True (accuracy/perf over legacy): on multi-target suites the second MRMR call on the same X usually hits the cache and saves the full FE pipeline run-time. The conservative case (prior identity result was wrong for the new target) is rare in practice because composite-target y values are highly correlated with the raw y -- if MRMR found nothing on raw y, it almost never finds something on the residual.
        mrmr_skip_when_prior_was_identity: bool = True,
        # When True, ``fit(groups=...)`` raises ``NotImplementedError`` instead of emitting the warn-only "MRMR does not consume groups" UserWarning. Use this in production pipelines where silently
        # ignoring groups would mask a real correctness gap (cross-group leakage in MI estimation on panel / user-session / sliding-window data). Default False keeps the legacy warn behaviour for
        # ad-hoc callers who already know the limitation and want MI computed per-row anyway.
        strict_groups: bool = False,
        # Friend-graph post-analysis: after screening, render the selected set as a
        # node-link diagram (node=feature sized by entropy, edge=pairwise MI, arrow=ADC
        # direction) and classify each feature green (unique) / red (suspected redundant
        # sink) / yellow (middling). Diagnostic by default; the fitted graph is exposed as
        # ``self.friend_graph_`` and summarized into the suite's feature_selection_report.
        build_friend_graph: bool = True,
        # When True, drop red (suspected-sink) features from ``support_`` after the graph is
        # built, protecting the neighbor that carries each removed feature's unique target info
        # so cause and effect are never dropped together. Off by default -- changes the selected set.
        friend_graph_prune: bool = False,
        # Guard on the O(k^2) edge pass: above this many selected features the graph keeps
        # node stats only and skips edges (warns). Raise for large opt-in graphs.
        friend_graph_max_nodes: int = 200,
        # Edge kept only when I(X_a; X_b) exceeds this absolute floor (nats).
        friend_graph_mi_eps: float = 1e-6,
        # ...and exceeds ``friend_graph_edge_significance`` times the finite-sample MI bias
        # ``(na-1)(nb-1)/(2n)`` expected under independence -- suppresses spurious edges.
        friend_graph_edge_significance: float = 3.0,
        # A feature is a sink candidate when it is connected to at least this many others
        # (graph degree); it is flagged red only if its neighbors then carry more unique
        # target info than its own relevance (scaled by ``friend_graph_unique_ratio``).
        friend_graph_garbage_min_degree: int = 3,
        friend_graph_unique_ratio: float = 1.0,
        # A feature is green (unique knowledge) when connected to at most this many others.
        friend_graph_unique_max_degree: int = 1,
        # Clustered-feature aggregation: when several correlated "reflection" features are noisy copies
        # of one hidden factor z, build a DENOISED aggregate (noise ~ sigma^2/k) instead of keeping one
        # and dropping the rest. ENABLED by default -- discovery is gated (min |corr| + PC1 unidimensionality
        # + a strict MI gate requiring the aggregate to beat the best single member), so it only fires on
        # genuine correlated-reflection clusters and is a no-op otherwise. "augment" (default) ADDS the
        # aggregate while keeping existing features (members additionally drop as redundant only under
        # use_simple_mode=False); "replace" substitutes the cluster with its aggregate. Helps
        # capacity-limited/linear downstreams, sensor data, interpretability; for tree/GBM downstreams
        # expect no-harm (trees already average reflections via splits).
        cluster_aggregate_enable: bool = True,
        # 2026-05-30 Wave 8 — default flipped from 'augment' to 'replace'.
        # When a denoised aggregate beats its member MIs (gain threshold per
        # ``cluster_aggregate_min_gain``), 'replace' drops the raw members from
        # the final selection AND from candidate consideration so they cannot
        # be re-picked downstream. This eliminates the duplicate-vote effect
        # (raw + aggregate both surviving) that the 'augment' mode silently
        # allowed; per Agent-B critique 2026-05-28, the augment behaviour was
        # the most common production-confusion point. Set
        # ``cluster_aggregate_mode='augment'`` to restore pre-Wave-8 behaviour
        # (raw + aggregate both kept).
        cluster_aggregate_mode: str = "replace",  # "augment" | "replace"
        # Aggregator menu (best gated method per cluster is kept): mean_z (default), mean_inv_var
        # (hetero noise), median (robust), pca_pc1 (hetero loadings), factor_score (Bartlett 1-factor).
        cluster_aggregate_methods: tuple = ("mean_z",),
        # Adopt only if MI(aggregate; y) >= this * max member MI (the denoising claim is "strictly
        # beats the best single member"; also what makes the no-harm cases reject).
        cluster_aggregate_mi_prevalence: float = 1.0,
        # Cluster a feature only if its marginal relevance clears this LOW floor (excludes pure noise;
        # well below the selection threshold so all-weak reflection clusters are still captured).
        cluster_aggregate_min_member_relevance: float = 0.0,
        cluster_aggregate_min_cluster_size: int = 3,
        cluster_aggregate_max_cluster_size: int = 12,
        # Min |corr| between members (continuous) to count as one reflection cluster.
        cluster_aggregate_corr_threshold: float = 0.6,
        # Min PC1 variance fraction for the cluster to be unidimensional (rejects multi-factor /
        # partial-shared+distinct clusters that averaging would blur).
        cluster_aggregate_homogeneity_tau: float = 0.6,
        # O(m^2) cost guard on the relevance-floored candidate pool.
        cluster_aggregate_max_candidates: int = 200,
        # 2026-05-30 Wave 9 — Dynamic Cluster Discovery (DCD).
        # Organic in-greedy-loop cluster discovery using ONLY MI/SU distances
        # (no Pearson — captures non-linear deps like XOR). After each
        # selection, prune the Pool by ``SU(x, just_selected) > tau_cluster``;
        # when cluster reaches threshold, swap raw anchor with PC1 aggregate if
        # ``I(rep ; y | Selected − anchor) > anchor_rel * (1 + swap_gain_threshold)``.
        # Pre-impl gate (bench_dcd_pair_su_scaling) confirmed 0.003× cost vs
        # full pairwise SU at p=10000.
        # 2026-05-30 Wave 9.1: dcd_enable=True by DEFAULT. Layer-6 biz_value
        # showed DCD is the documented MRMR mechanism for production
        # redundancy-control (near-duplicate decoys, collinear clusters,
        # synergistic groups). Pre-Wave-9 the default was False to preserve
        # bit-stability with pre-Wave-9 fits; that's no longer the priority
        # vs giving every user cluster-aware selection out of the box. The
        # 0.003x overhead is negligible. Users wanting pre-Wave-9 behaviour
        # opt out via dcd_enable=False.
        dcd_enable: bool = True,
        # Layer 47 (2026-05-31): ``dcd_tau_cluster`` accepts ``'auto'`` to
        # opt into the per-fit bimodality-detection calibration sweep
        # (``make_dcd_state`` samples 100 random feature pairs, fits a
        # coarse histogram, and picks tau at the valley between the two
        # SU modes; falls back to 0.7 when the distribution is unimodal).
        # Numeric values keep the legacy fixed-tau behaviour bit-identical.
        dcd_tau_cluster=0.7,
        dcd_distance: str = "su",
        # Layer 47 (2026-05-31): knobs for the auto-tau calibration sweep.
        # ``dcd_tau_calibration_n_pairs`` is the number of random feature pairs
        # sampled for the bimodality histogram; ``dcd_tau_calibration_seed``
        # makes the random sample deterministic per fit.
        dcd_tau_calibration_n_pairs: int = 100,
        dcd_tau_calibration_seed: int = 0,
        # 2026-05-31 Layer 42: default kept at 4 pending downstream fix.
        # Lowering to 2 (member count beyond anchor) makes the canonical
        # 3-feature redundancy cluster (anchor + 2 dups) actually trigger
        # the PC1 swap that previously was effectively gated OFF, but the
        # post-swap aggregate is not currently wired into
        # ``_engineered_recipes_`` / ``support_`` (commit_swap is called
        # with ``engineered_recipes=None`` at _screen_predictors.py L718),
        # so the swap silently drops the anchor from the output. Until the
        # aggregate->recipe propagation lands, threshold=2 net-shrinks
        # ``support_`` on real data. Layer 42 instead exposes the new
        # validated lower bound (>=1) + documents the kt-tuned 2-step opt-in:
        # ``dcd_cluster_size_threshold=2`` + a registered recipe pathway
        # (next layer). Pin =2 explicitly to enable the new gate.
        dcd_cluster_size_threshold: int = 4,
        dcd_swap_gain_threshold: float = 0.05,
        # 2026-05-31 Layer 43 (PART B): default flipped to ``"auto"``. When set
        # to ``"auto"``, ``evaluate_swap_candidate`` runs a K-fold (n_folds=5)
        # OOF conditional-MI scoring over the three linear-combiner methods
        # (``mean_z``, ``mean_inv_var``, ``pca_pc1``) and picks the winner per
        # cluster. The winning method name is persisted in the recipe ``extra``
        # and the ``swap_log`` entry. Replay (``_apply_cluster_aggregate``)
        # uses the chosen method bit-identically (no y at transform time).
        # Pinning a specific method (``"pca_pc1"`` etc.) keeps the legacy
        # single-method path; pinning ``"auto"`` is the strict superset of the
        # legacy default since the auto path includes ``pca_pc1`` as a
        # candidate and picks it whenever it dominates.
        dcd_swap_method: str = "auto",
        dcd_pairwise_cache_max: int = 50_000,
        dcd_min_cluster_size: int = 2,
        dcd_max_cluster_size: int = 12,
        # 2026-05-30 Wave 9.1 iter 3: permutation-null gate on swap
        # acceptance. With ``full_npermutations > 0`` AND ``dcd_enable=True``,
        # ``evaluate_swap_candidate`` shuffles the PC1 rep B times, builds
        # the null distribution of conditional MI, and only accepts the swap
        # when both the deterministic gain gate AND ``perm_p_value < swap_alpha``
        # are satisfied. Prior to this fix the swap was a pure point-MI
        # comparison and accepted spurious aggregates on noisy / small-n
        # data because PC1 (continuous, re-binned with finer granularity than
        # the raw anchor) is upward-biased.
        dcd_swap_alpha: float = 0.05,
        # ``dcd_postoc_compose=True`` keeps the post-hoc cluster_aggregate
        # active alongside DCD. Default False auto-suppresses it (DCD
        # already processed clusters during screening; running again would
        # double-aggregate).
        dcd_postoc_compose: bool = False,
        # 2026-05-31 Layer 23 — Hybrid orthogonal-polynomial + MI-greedy FE
        # auto-wired into the fit pipeline (sibling module:
        # ``_orthogonal_univariate_fe.hybrid_orth_mi_fe`` /
        # ``hybrid_orth_mi_pair_fe``). Default OFF -- legacy behaviour is
        # byte-identical when ``fe_hybrid_orth_enable=False``. When True, the
        # hybrid FE runs ONCE before screening: it generates
        # ``basis_n(preprocess(X[c]))`` columns for each n in ``fe_hybrid_orth_degrees``
        # and ranks by MI uplift vs the raw source baseline; the top-K winners
        # are appended to X and screened as ordinary numeric columns. With
        # ``fe_hybrid_orth_pair_enable=True`` (the default when the master is
        # on) the bilinear ``basis_a(z_i) * basis_b(z_j)`` cross-basis stage
        # also fires, capturing the XOR / saddle / circle pair targets.
        #
        # Stored as ``EngineeredRecipe`` objects of kinds ``"orth_univariate"`` /
        # ``"orth_pair_cross"``; the recipe is closed-form in the source column
        # values alone (no y reference at recipe-build time), so
        # ``MRMR.transform`` replays each engineered column on test data
        # without any leakage risk.
        fe_hybrid_orth_enable: bool = False,
        fe_hybrid_orth_degrees: tuple = (2, 3),
        fe_hybrid_orth_basis: str = "auto",
        # Combined cap on appended columns (univariate + pair). Top-K is
        # applied separately to each stage by the underlying hybrid pipeline;
        # this is the per-stage budget. Default 5 = at most 5 univariate
        # winners + at most 5 pair winners when pair_enable=True.
        fe_hybrid_orth_top_k: int = 5,
        fe_hybrid_orth_pair_enable: bool = True,
        fe_hybrid_orth_pair_max_degree: int = 2,
        # 2026-05-31 Layer 56 — TRI-PRODUCT cross-basis FE (sibling module
        # ``_orthogonal_triplet_fe``). Captures genuine 3-way interactions
        # like ``y = sign(x_i * x_j * x_k)`` (3-way XOR) or
        # ``y = sign(price * quantity * count - threshold)`` that no pair
        # term can resolve (3-way XOR has zero marginal pair MI).
        #
        # Default OFF -- combinatorial enumeration O(p^3 * deg^3) is too
        # aggressive to enable silently. When master + triplet_enable are
        # ON, the triplet stage runs AFTER the pair stage on the SAME
        # input frame X (raw sources), uses ``triplet_seed_k`` top-MI raw
        # columns to bound the candidate count to C(seed_k, 3) * deg^3,
        # and appends ``top_count`` MI-uplift winners as
        # ``orth_triplet_cross`` recipes. Replay reads only X.
        #
        # ``triplet_max_degree=1`` default emits exactly one cell per
        # triplet (``He_1*He_1*He_1`` -- the dominant 3-way signal).
        # Bump to 2 only if your domain has known cubic-in-each-leg
        # 3-way interactions; otherwise the deg-1 cell carries every
        # multiplicative 3-way target the literature pins.
        fe_hybrid_orth_triplet_enable: bool = False,
        fe_hybrid_orth_triplet_max_degree: int = 1,
        fe_hybrid_orth_triplet_seed_k: int = 4,
        fe_hybrid_orth_triplet_top_count: int = 2,
        # 2026-05-31 Layer 57 — ADAPTIVE PER-COLUMN DEGREE selection
        # (sibling module ``_orthogonal_adaptive_degree_fe``). Independent
        # opt-in (does NOT require fe_hybrid_orth_enable). When enabled,
        # for each source column we evaluate every degree in
        # ``fe_hybrid_orth_adaptive_degree_range`` and emit ONLY the
        # argmax-MI degree (if it clears ``min_uplift`` over raw).
        #
        # Default OFF preserves Layer 21's fixed-degree sweep
        # ``fe_hybrid_orth_degrees=(2,3)`` byte-for-byte. Recipes emit as
        # ``orth_univariate`` (no new kind -- the recipe already carries
        # ``(basis, degree)`` per column, the only change is the value
        # is the per-column argmax instead of a sweep). Replay reads X
        # only, no y, leakage-free by construction.
        fe_hybrid_orth_adaptive_degree_enable: bool = False,
        fe_hybrid_orth_adaptive_degree_range: tuple = (1, 2, 3, 4, 5, 6),
        fe_hybrid_orth_adaptive_degree_min_uplift: float = 1.05,
        # 2026-05-31 Layer 58 — CONDITIONAL BASIS ROUTING (sibling module
        # ``_orthogonal_routing_fe``). Independent opt-in (does NOT require
        # fe_hybrid_orth_enable). When enabled, for each source column we
        # try every (pre_transform, basis, degree) cell in the cartesian
        # product over ``PRE_TRANSFORM_NAMES`` x candidate bases x
        # ``fe_hybrid_orth_conditional_routing_degrees``, keep the per-column
        # MI argmax, then global top-K by uplift. The ``min_uplift`` default
        # is tighter (1.10) than Layer 21/57's 1.05 because the candidate
        # pool is 4x larger per column so the noise tail is fatter.
        #
        # Default OFF preserves byte-for-byte legacy behaviour. Recipes
        # emit as ``orth_univariate`` with ``extra["pre_transform"]``
        # carrying the chosen transform tag. Replay reads X only, no y,
        # leakage-free by construction.
        fe_hybrid_orth_conditional_routing_enable: bool = False,
        fe_hybrid_orth_conditional_routing_top_k: int = 5,
        fe_hybrid_orth_conditional_routing_min_uplift: float = 1.10,
        fe_hybrid_orth_conditional_routing_degrees: tuple = (2, 3),
        # 2026-05-31 Layer 59 — DIFF-BASIS FE for highly-correlated source
        # pairs (sibling module ``_orthogonal_diff_basis_fe``). Independent
        # opt-in (does NOT require fe_hybrid_orth_enable). When enabled, the
        # auto-pair detector flags every pair with |Pearson corr| >= the
        # threshold, computes the residual ``X[a] - X[b]``, and evaluates
        # ``basis_d(preprocess(diff))`` for each degree. Top-K winners
        # appended; recipe kind ``"orth_diff_basis"``; replay reads X only,
        # no y. Default OFF preserves legacy pickle byte-equivalence.
        fe_hybrid_orth_diff_basis_enable: bool = False,
        fe_hybrid_orth_diff_basis_corr_threshold: float = 0.7,
        fe_hybrid_orth_diff_basis_degrees: tuple = (1, 2, 3),
        fe_hybrid_orth_diff_basis_top_k: int = 3,
        # 2026-05-31 Layer 61 — PER-CLUSTER SHARED-BASIS FE (sibling module
        # ``_orthogonal_cluster_basis_fe``). Independent opt-in (does NOT
        # require fe_hybrid_orth_enable). When enabled, the internal
        # correlation-based cluster detector finds connected components of
        # the |Pearson corr| >= corr_threshold graph, reduces each cluster
        # to one column via ``aggregator`` (mean_z / median_z / pc1), then
        # evaluates ``basis_d(preprocess(aggregate))`` for each requested
        # degree. Top-K winners appended; recipe kind
        # ``"orth_cluster_basis"``; replay reads X only, no y. Default OFF
        # preserves legacy pickle byte-equivalence.
        fe_hybrid_orth_cluster_basis_enable: bool = False,
        fe_hybrid_orth_cluster_basis_aggregator: str = "mean_z",
        fe_hybrid_orth_cluster_basis_degrees: tuple = (2, 3),
        fe_hybrid_orth_cluster_basis_top_k: int = 3,
        # 2026-05-31 Layer 62 — BOOTSTRAP-STABLE MI ranking for the hybrid
        # orth-poly FE (sibling module ``_orthogonal_bootstrap_mi_fe``).
        # Independent opt-in (does NOT require fe_hybrid_orth_enable).
        # When enabled, the same per-source univariate basis columns Layer
        # 21 generates are scored by the lower-confidence-bound of MI
        # uplift across ``n_boot`` bootstrap subsamples (drawn with
        # replacement at ``sample_fraction``) instead of a single point
        # estimate. Candidates with a high MEAN MI but a long right tail
        # get a large std and a small LCB; stable signals ride through.
        # Selection-stability win: borderline noise-driven flukes that
        # the point-estimate ranking admits in 1 of N runs are filtered
        # out. Recipes use ``orth_univariate`` kind -- the engineered
        # VALUES are bit-equal to Layer 21, only the selection rule
        # differs -- so replay is shared infrastructure.
        # Default OFF preserves legacy pickle byte-equivalence.
        fe_hybrid_orth_bootstrap_enable: bool = False,
        fe_hybrid_orth_bootstrap_n_boot: int = 10,
        fe_hybrid_orth_bootstrap_sample_fraction: float = 0.8,
        # 2026-05-31 Layer 63 — THREE-GATE + K-fold OOF MI ranking for the
        # hybrid orth-poly FE (sibling module ``_orthogonal_three_gate_mi_fe``).
        # Independent opt-in (does NOT require fe_hybrid_orth_enable).
        # When enabled, the per-source univariate basis columns Layer 21
        # generates are scored by OOF MI (K-fold held-out estimate using
        # train-fitted bin edges) and admitted by THREE gates rather than
        # two: (1) relative uplift_oof >= min_uplift, (2) absolute OOF
        # engineered_mi >= MAD floor, (3) CMI(candidate; y | current
        # support) >= cmi_min. Gate 3 catches the "duplicate signal"
        # failure mode that two-gate selection misses: once x__He2 is in
        # support, a second basis like x__T2 has near-identical marginal
        # MI but negligible CMI given x__He2, so it is correctly dropped.
        # Engineered VALUES bit-equal to Layer 21 -- only selection rule
        # changes -- so recipes use ``orth_univariate`` kind and replay
        # is shared infrastructure.
        # Default OFF preserves legacy pickle byte-equivalence.
        fe_hybrid_orth_three_gate_enable: bool = False,
        fe_hybrid_orth_three_gate_n_folds: int = 5,
        fe_hybrid_orth_three_gate_cmi_min: float = 0.001,
        # 2026-05-31 Layer 65 — KSG / k-NN MI ranking for hybrid orth-poly FE
        # (sibling module ``_orthogonal_ksg_mi_fe``). Independent opt-in
        # (does NOT require fe_hybrid_orth_enable). Layer 21 ranks by the
        # plug-in quantile-binned MI estimator (fast, but discretises smooth
        # continuous structure away); Layer 65 swaps it for the Kraskov-
        # Stoegbauer-Grassberger k-NN MI estimator via sklearn's
        # ``mutual_info_classif`` (Ross 2014 mixed-KSG for discrete y).
        # The KSG estimator is asymptotically unbiased on continuous data
        # and recovers smooth signals (e.g. a He_3 cubic ripple that
        # binning erases). Engineered VALUES are bit-equal to Layer 21 so
        # recipes reuse the ``orth_univariate`` kind and replay is shared
        # infrastructure. Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_ksg_enable: bool = False,
        fe_hybrid_orth_ksg_n_neighbors: int = 3,
        # KSG-specific selection thresholds. KSG MI values are smaller than
        # plug-in's on the same signal (KSG is less biased upward), so the
        # uplift gate floor that Layer 21 calibrated for plug-in (1.05) is
        # too strict here -- KSG's k-NN already captures non-monotone
        # structure in raw x1, depressing the per-engineered uplift below
        # 1.05 even when the engineered column is genuinely useful. The
        # 0.95 floor admits engineered columns whose MI is within 5 % of
        # the raw source's MI, which is the smallest difference the
        # k-NN estimator can resolve at typical sample sizes.
        fe_hybrid_orth_ksg_min_uplift: float = 0.95,
        fe_hybrid_orth_ksg_min_abs_mi_frac: float = 0.05,
        # 2026-06-01 Layer 66 — COPULA-MI ranking for hybrid orth-poly FE
        # (sibling module ``_orthogonal_copula_mi_fe``). Independent opt-in
        # (does NOT require fe_hybrid_orth_enable). Layer 21 ranks by the
        # plug-in quantile-binned MI estimator on RAW values -- on heavy-
        # tailed or skewed marginals the qcut bin edges pile extreme-value
        # observations into a single bin and hide genuine dependence inside
        # the bulk. Layer 66 rank-transforms each variable to a uniform on
        # ``(0, 1)`` (Sklar's theorem: the copula carries all dependence
        # structure independently of the marginals), then estimates MI on
        # the uniform pair via equal-width binning + Miller-Madow bias
        # correction. The resulting MI is INVARIANT under any strictly-
        # monotone transform of either variable -- exactly the property
        # heavy-tail / log-scale signals need to be scored fairly.
        # Engineered VALUES are bit-equal to Layer 21 so recipes reuse the
        # ``orth_univariate`` kind and replay is shared infrastructure.
        # Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_copula_enable: bool = False,
        fe_hybrid_orth_copula_n_bins: int = 20,
        # 2026-06-01 Layer 67 — DISTANCE-CORRELATION ranking for hybrid orth-
        # poly FE (sibling module ``_orthogonal_dcor_fe``). Independent opt-in
        # (does NOT require fe_hybrid_orth_enable). Layer 21 / 65 / 66 are
        # all MI estimators (differing in how they estimate it); Layer 67 is
        # the Szekely-Rizzo distance correlation -- a NON-MI dependence
        # measure with the universal ``dCor == 0 iff independent`` guarantee
        # that Pearson lacks. Excels on non-monotone / non-functional /
        # oscillatory dependencies where MI estimators converge slowly.
        # Naive dCor is O(n^2) memory; subsamples at n=500 keep the per-
        # pair distance matrices at 2 MB each. Engineered VALUES are bit-
        # equal to Layer 21 so recipes reuse the ``orth_univariate`` kind
        # and replay is shared infrastructure. Default OFF preserves pickle
        # byte-equivalence.
        fe_hybrid_orth_dcor_enable: bool = False,
        fe_hybrid_orth_dcor_n_sample: int = 500,
        # 2026-06-01 Layer 71 — HSIC (Hilbert-Schmidt Independence Criterion)
        # ranking for the hybrid orth-poly FE (sibling module
        # ``_orthogonal_hsic_fe``). Independent opt-in (does NOT require
        # fe_hybrid_orth_enable). Like Layer 67 dCor, HSIC is a NON-MI
        # dependence measure with the universal ``HSIC == 0 iff independent``
        # guarantee under a CHARACTERISTIC kernel (Gaussian RBF). Operates
        # at a kernel-chosen length SCALE via the median-heuristic bandwidth;
        # complementary to dCor (which has no scale parameter) on sharp
        # local non-linearities and high-frequency oscillation. Naive HSIC
        # is O(n^2) memory; the working sample is capped at n=500 via
        # deterministic random subsample. Engineered VALUES bit-equal to
        # Layer 21 -> recipes reuse the ``orth_univariate`` kind. Default
        # OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_hsic_enable: bool = False,
        fe_hybrid_orth_hsic_kernel: str = "rbf",
        fe_hybrid_orth_hsic_n_sample: int = 500,
        # 2026-06-01 Layer 72 — JMIM (Joint Mutual Information Maximisation,
        # Bennasar 2015) redundancy-aware ranking for hybrid orth-poly FE
        # (sibling module ``_orthogonal_jmim_fe``). Independent opt-in (does
        # NOT require fe_hybrid_orth_enable). Layers 21 / 65 / 66 / 67 / 71
        # rank by MARGINAL dependence with y; Layer 72 ranks by the WORST-
        # CASE joint MI against the already-selected support:
        # ``J(X_k) = min over X_j in S of I((X_k, X_j); Y)`` (Bennasar 2015,
        # Eq. 5). The min over S enforces non-redundancy column-by-column,
        # so a candidate that is informative jointly with ONE support
        # member but redundant with ANOTHER cannot hide behind the strong
        # interaction. Engineered VALUES are bit-equal to Layer 21 ->
        # recipes reuse the ``orth_univariate`` kind. Default OFF preserves
        # pickle byte-equivalence.
        fe_hybrid_orth_jmim_enable: bool = False,
        fe_hybrid_orth_jmim_n_bins: int = 10,
        # 2026-06-01 Layer 73 — Total Correlation (Watanabe 1960) multivariate-
        # redundancy ranking for hybrid orth-poly FE (sibling module
        # ``_orthogonal_total_correlation_fe``). Independent opt-in (does NOT
        # require fe_hybrid_orth_enable). Layers 21 / 65 / 66 / 67 / 71 rank
        # by MARGINAL dependence with y; Layer 72 (JMIM) ranks by the worst
        # PAIRWISE joint MI with the support. Layer 73 ranks by the FULL-
        # ORDER joint shared information ``TC(Z) = sum H(Z_i) - H(Z)``
        # contribution: ``delta_tc = TC([support, c, y]) - TC([support, y])``.
        # Catches higher-order redundancy (e.g. XOR-style three-variable
        # parity) that every pairwise scorer misses. Engineered VALUES are
        # bit-equal to Layer 21 -> recipes reuse the ``orth_univariate``
        # kind. Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_tc_enable: bool = False,
        fe_hybrid_orth_tc_n_bins: int = 10,
        # 2026-06-01 Layer 74 — CMIM (Conditional Mutual Information
        # Maximisation, Fleuret 2004) redundancy-aware ranking for hybrid
        # orth-poly FE (sibling module ``_orthogonal_cmim_fe``). Independent
        # opt-in (does NOT require fe_hybrid_orth_enable). Companion to
        # Layer 72 (JMIM): JMIM scores ``min_j I((X_k, X_j); Y)`` (joint
        # MI -- rewards complementarity); CMIM scores
        # ``min_j I(X_k; Y | X_j)`` (conditional MI -- penalises
        # redundancy). On heavily-DUPLICATING candidate pools (near-copies
        # of one strong predictor) CMIM is the empirical winner; on
        # heavily-INTERACTING pools JMIM wins. Engineered VALUES are bit-
        # equal to Layer 21 -> recipes reuse the ``orth_univariate`` kind.
        # Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_cmim_enable: bool = False,
        fe_hybrid_orth_cmim_n_bins: int = 10,
        # 2026-06-01 Layer 68 — PER-COLUMN SCORER AUTO-SELECTION across the
        # full Layer 21 / 65 / 66 / 67 family (sibling module
        # ``_orthogonal_scorer_auto_fe``). Independent opt-in (does NOT
        # require fe_hybrid_orth_enable). Each scorer wins on a different
        # signal family (plug-in: discrete-binned; KSG: smooth continuous;
        # copula: heavy-tailed; dCor: non-monotone) -- on heterogeneous
        # frames the single-scorer opt-ins of Layers 65 / 66 / 67 are
        # wrong on SOME columns no matter which one the user picks. Layer
        # 68 runs all four under a small bootstrap budget, picks the
        # per-column scorer with the highest LOWER CONFIDENCE BOUND
        # (mean - 1.96 * std) across ``n_boot`` resamples, and uses ITS
        # score for the ranking + selection. Engineered VALUES are bit-
        # equal to Layer 21 so recipes reuse the ``orth_univariate`` kind
        # and replay is shared infrastructure. Default OFF preserves
        # pickle byte-equivalence.
        fe_hybrid_orth_auto_scorer_enable: bool = False,
        fe_hybrid_orth_auto_scorer_n_boot: int = 5,
        # 2026-06-01 Layer 69 — ENSEMBLE-OF-SCORERS rank-fusion for hybrid
        # orth-poly FE. Sibling of Layer 68: instead of picking ONE scorer
        # per column via bootstrap LCB, aggregate per-scorer rankings via
        # mean_rank / borda_count / reciprocal_rank fusion and select by
        # the consensus rank. The ensemble wins when bootstrap-LCB noise
        # makes the per-column winner unstable across seeds -- rank fusion
        # smooths over the instability because a column ranked high by
        # ANY of the participating scorers keeps a high consensus rank
        # even if no individual scorer wins the LCB tournament on every
        # seed. Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_ensemble_enable: bool = False,
        fe_hybrid_orth_ensemble_aggregator: str = "mean_rank",
        # 2026-06-01 Layer 71: HSIC added to the ensemble default pool;
        # callers that previously pinned the 4-tuple keep the old
        # behaviour, the default now leverages all five scorers.
        fe_hybrid_orth_ensemble_scorers: tuple = (
            "plug_in", "ksg", "copula", "dcor", "hsic",
        ),
        # 2026-06-01 Layer 76 — META-SCORER auto-selection that LEARNS
        # from cheap signal characteristics (sibling module
        # ``_orthogonal_meta_scorer_fe``). Independent opt-in (does NOT
        # require fe_hybrid_orth_enable). Layer 68 (per-column bootstrap
        # LCB) and Layer 69 (rank-fusion ensemble) run ALL scorers and
        # let a meta-criterion pick; Layer 76 instead spends a small
        # fixed budget on cheap fingerprints (skew, kurtosis, n_unique,
        # mean abs Pearson, dCor proxy via Spearman) and a deterministic
        # 5-rule cascade distilled from the L75 empirical matrix to
        # PREDICT which scorer will win, then runs ONLY that scorer. The
        # wall-clock saving is roughly n_scorers - 1 vs L68/L69. Engineered
        # VALUES bit-equal to Layer 21 -> recipes reuse the
        # ``orth_univariate`` kind. Set ``fe_hybrid_orth_meta_force_scorer``
        # to override the rule cascade and pin a specific scorer
        # (one of "plug_in"/"ksg"/"copula"/"dcor"/"hsic"/"jmim"/"cmim"/"tc").
        # Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_meta_enable: bool = False,
        fe_hybrid_orth_meta_force_scorer: Optional[str] = None,
        # 2026-05-31 Layer 32 — extra (non-polynomial) basis FE: B-spline +
        # Fourier. Complementary to the orth-poly path: spline catches sharp
        # local non-linearities (threshold rules ``y = sign(x - tau)``);
        # Fourier catches periodic patterns (``y = sign(sin(2*pi*x))``).
        # Empty tuple (default) keeps the legacy behaviour byte-identical.
        # When non-empty AND ``fe_hybrid_orth_enable=True``, the extra-basis
        # stage runs after the polynomial stages and appends its own top-K
        # MI-uplift winners. Recipes (``orth_spline`` / ``orth_fourier``)
        # are closed-form in the source column alone -- replay reads X
        # only, no y leakage.
        fe_hybrid_orth_extra_bases: tuple = (),
        fe_hybrid_orth_fourier_freqs: tuple = (1.0, 2.0),
        fe_hybrid_orth_spline_knots: int = 5,
        # 2026-05-31 Layer 26 — generic MI-greedy FE constructor (sibling
        # to the orthogonal-polynomial one). Default OFF -- legacy
        # behaviour is byte-identical when ``fe_mi_greedy_enable=False``.
        # When True, the MI-greedy FE runs ONCE before screening (after
        # the hybrid orth stage when both are enabled): it enumerates
        # generic unary / binary transforms (log_abs, sqrt_abs, square,
        # cube, reciprocal_safe, tanh, expm1_clip, abs / add, sub, mul,
        # div_safe, max, min, abs_diff, ratio_log) over the top-N source
        # columns by raw MI, ranks the candidates by MI uplift, and
        # appends the top-K winners to X. Recipes of kind
        # ``"mi_greedy_transform"`` carry transform name + src cols only
        # (no y), so transform() replay is leakage-free.
        fe_mi_greedy_enable: bool = False,
        fe_mi_greedy_top_k: int = 5,
        fe_mi_greedy_seed_cols_count: int = 5,
        fe_mi_greedy_include_unary: bool = True,
        fe_mi_greedy_include_binary: bool = True,
        # 2026-05-31 Layer 60 — CMI-greedy FE constructor (sibling to the
        # marginal-MI greedy one above). Default OFF -- legacy behaviour
        # is byte-identical when ``fe_mi_greedy_cmi_enable=False``. When
        # True, the same candidate transform library used by Layer 26 is
        # ranked by ``CMI(candidate; y | currently-selected-support)``
        # instead of marginal ``MI(candidate; y)``, so duplicate-signal
        # transforms (``log_abs(x)`` AND ``square(x)`` both monotone in
        # ``|x|``) are naturally suppressed -- once one of the family is
        # in the support, the others' CMI collapses near zero and they
        # are never picked. Recipes reuse kind ``"mi_greedy_transform"``
        # so transform-time replay is shared infrastructure.
        fe_mi_greedy_cmi_enable: bool = False,
        fe_mi_greedy_cmi_top_k: int = 5,
        fe_mi_greedy_cmi_seed_cols_count: int = 4,
        fe_mi_greedy_cmi_min_gain: float = 0.005,
        # 2026-05-31 Layer 33 — K-fold target encoding for raw categorical
        # columns. Default OFF -- legacy behaviour is byte-identical when
        # ``fe_kfold_te_enable=False``. When True, after the hybrid + MI-
        # greedy stages run, every column in ``fe_kfold_te_cols`` (or
        # auto-detected categoricals with cardinality in [5, 500] when the
        # tuple is empty) is target-encoded with K-fold OOF discipline and
        # the encoded ``{col}__te`` column is appended to X. The recipe
        # (``kfold_target_encoded``) stores the FULL-data per-category
        # mean for deterministic replay -- no y is referenced at transform.
        fe_kfold_te_enable: bool = False,
        fe_kfold_te_cols: tuple = (),
        fe_kfold_te_folds: int = 5,
        fe_kfold_te_smoothing: float = 10.0,
        # 2026-05-31 Layer 34 — COUNT + FREQUENCY ENCODING for high-
        # cardinality categoricals, plus CATEGORICAL x NUMERIC INTERACTION
        # via OOF target-mean residual. Default OFF -- legacy behaviour
        # is byte-identical when all three master switches are False.
        # Each encoded column is appended via its own recipe kind
        # (``count_encoded`` / ``frequency_encoded`` / ``cat_num_residual``);
        # replay is a pure function of X (no y reference at transform).
        # ``fe_count_encoding_cols`` / ``fe_frequency_encoding_cols`` reuse
        # the same auto-detection (object / categorical / string dtype with
        # cardinality in [5, 500]) as Layer 33 when left as empty tuple.
        # ``fe_cat_num_interaction_cat_cols`` x ``fe_cat_num_interaction_num_cols``
        # is the explicit Cartesian product (no auto-detect; the choice of
        # which numeric column to condition on which categorical column is
        # domain-specific).
        fe_count_encoding_enable: bool = False,
        fe_count_encoding_cols: tuple = (),
        fe_frequency_encoding_enable: bool = False,
        fe_frequency_encoding_cols: tuple = (),
        fe_cat_num_interaction_enable: bool = False,
        fe_cat_num_interaction_cat_cols: tuple = (),
        fe_cat_num_interaction_num_cols: tuple = (),
        fe_cat_num_interaction_folds: int = 5,
        fe_cat_num_interaction_smoothing: float = 10.0,
        # 2026-05-31 Layer 37 — MISSINGNESS-AWARE FE. Default OFF; legacy
        # behaviour is byte-identical when all three master switches stay
        # False. Layer 7's ``nan_strategy='separate_bin'`` already handles
        # MNAR at the binning level inside the MI estimator; Layer 37
        # COMPLEMENTS that by EXPOSING missingness as standalone engineered
        # features the downstream model can consume directly.
        # * ``missing_indicator``: per-source ``is_missing__{col}`` binary
        #   column. When ``fe_missingness_indicator_cols`` is empty AND the
        #   master switch is ON, auto-detect picks columns with NaN rate
        #   in [1%, 99%].
        # * ``missingness_count``: per-row count of NaNs across a column
        #   subset (auto-detected if empty).
        # * ``missingness_pattern``: per-row label of the top-K most
        #   frequent missingness patterns at fit; unseen patterns at
        #   transform map to the "other" bucket.
        # Recipes (``missing_indicator`` / ``missingness_count`` /
        # ``missingness_pattern``) replay as pure functions of X.
        fe_missingness_indicator_enable: bool = False,
        fe_missingness_indicator_cols: tuple = (),
        fe_missingness_count_enable: bool = False,
        fe_missingness_pattern_enable: bool = False,
        fe_missingness_pattern_top_k: int = 5,
        # 2026-05-31 Layer 38 — CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE.
        # Default OFF; legacy behaviour byte-identical when all three master
        # switches stay False. Each is appended via its own recipe kind
        # (``pairwise_ratio`` / ``grouped_delta`` / ``lagged_diff``); replay is
        # a pure function of X (no y reference at transform).
        # * ``pairwise_ratio``: ``ratio__{a}__{b}`` (safe division floored at
        #   ``eps``); pairs whose Pearson |corr| with either source is > 0.99
        #   are rejected (no info gain). Set ``fe_pairwise_log_ratio_enable``
        #   instead to emit ``log1p(|a|+eps) - log1p(|b|+eps)`` (handles
        #   negative values gracefully).
        # * ``grouped_delta``: ``x - mean(x | group)`` AND per-group z-score.
        #   The recipe stores per-group mean/std at fit; unseen groups at
        #   replay fall back to the train global stats.
        # * ``lagged_diff``: sort by ``time_col`` then compute ``x - x.shift(p)``
        #   for each p in ``periods``.
        fe_pairwise_ratio_enable: bool = False,
        fe_pairwise_ratio_cols: tuple = (),
        fe_pairwise_ratio_eps: float = 1e-9,
        fe_pairwise_log_ratio_enable: bool = False,
        fe_pairwise_log_ratio_cols: tuple = (),
        fe_grouped_delta_enable: bool = False,
        fe_grouped_delta_group_col: str = None,
        fe_grouped_delta_num_cols: tuple = (),
        fe_lagged_diff_enable: bool = False,
        fe_lagged_diff_time_col: str = None,
        fe_lagged_diff_value_cols: tuple = (),
        fe_lagged_diff_periods: tuple = (1, 2),
        # Artifact retention for cross-selector reuse. When True, after fit() the
        # estimator carries ``su_to_target_``, ``mi_to_target_``, ``cached_MIs``,
        # and (when ``retain_bins=True``) per-column binned arrays so a downstream
        # selector (e.g. ShapProxiedFS(precomputed=mrmr.export_artifacts()))
        # can skip its own univariate pre-screen. Default False keeps the legacy
        # memory footprint byte-identical.
        retain_artifacts: bool = False,
        retain_bins: bool = True,
        # 2026-05-31 Layer 53 — incremental / streaming refit support via
        # ``partial_fit(X_new, y_new)``. Default-OFF byte-identical with
        # legacy fit(): the partial_fit method is opt-in only. Knobs:
        #   partial_fit_decay : float in [0, 1]. 0 = no decay (concatenate
        #     all historic batches), 1 = full re-weight on the new batch
        #     (effectively re-fit on new only). Intermediate values weight
        #     recent rows more heavily via per-row sample weights at the
        #     resample stage. Implemented by upsampling the new batch by
        #     ``ceil(1 / max(1-decay, eps))`` against the historic buffer
        #     when calling the underlying fit(); preserves the bit-exact
        #     legacy weighted-resample contract.
        #   partial_fit_min_recompute : int. Minimum number of new rows
        #     observed since the last full refit before partial_fit triggers
        #     a recompute. Smaller updates are buffered until the threshold
        #     is reached. Defaults to 100 to amortise screening cost across
        #     small streaming batches.
        #   partial_fit_window : int or None. Rolling window in rows. When
        #     not None and the buffered (X, y) exceeds this length, the
        #     oldest rows are dropped before the next refit. None disables
        #     the rolling window (cumulative growth).
        partial_fit_decay: float = 0.0,
        partial_fit_min_recompute: int = 100,
        partial_fit_window: int = None,
        # hidden
        stop_file: str = "stop",
        # iter79: content-addressable disk cache for the per-column adaptive bin-edge stage. ``None``
        # (default) disables. When set, ``per_feature_edges`` caches each column's edge array keyed
        # by (column-summary, method, kwargs, y-summary-when-supervised); re-fits on the same X+y
        # skip the per-column edge-builder. Most useful for hyperparam sweeps and ablations where
        # the binning input recurs verbatim across runs. See ``mlframe.utils.disk_cache``.
        cache_dir: str = None,
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
    # 2026-05-29 Wave 7: adaptive per-feature bin-edge chooser.
    # MRMR's MI computation stays exclusively on the integer-bin plug-in path
    # (see bench_adaptive_nbins / bench_adaptive_nbins_mega). Alternative MI
    # estimators (KSG, MINE, InfoNet, MIST, fastMI, aggregators) are
    # intentionally NOT routed into the MRMR hot loop.
    _VALID_NBINS_STRATEGIES = (
        None,
        "auto", "sturges", "freedman_diaconis", "fd", "qs",
        "knuth", "blocks",  # demoted to research-only with AccuracyWarning
        "mdlp", "fayyad_irani", "optimal_joint", "cv",
        "mah", "mah_sci", "sci", "marx",  # Marx 2021 SCI-guided adaptive
    )
    # 2026-05-30 Wave 8 opt-in validation sets.
    _VALID_MI_CORRECTIONS = ("none", "miller_madow", "chao_shen")
    _VALID_REDUNDANCY_AGGREGATORS = (None, "jmim")
    _VALID_STABILITY_SELECTION_METHODS = (
        "classic", "cluster", "complementary_pairs",
    )
    # 2026-05-29: per mega-bench v3 Knuth (MI_mean 0.342, weak on uniform),
    # Bayesian Blocks (MI_mean 0.272, weakest overall), and MAH/SCI
    # (MI_mean 0.168, catastrophic on noisy continuous signals due to
    # over-aggressive SCI-greedy bin merging that collapses to ~2 bins)
    # are demoted from the recommended option set. They remain selectable
    # for research / reproduction work but emit an ``AccuracyWarning`` so
    # downstream callers can opt-in explicitly.
    _DEMOTED_NBINS_STRATEGIES = (
        "knuth", "blocks",
        "mah", "mah_sci", "sci", "marx",
    )
    _VALID_FE_UNARY_PRESETS = ("minimal", "medium", "maximal")
    _VALID_FE_BINARY_PRESETS = ("minimal", "medium", "maximal")
    _VALID_CLUSTER_AGGREGATE_MODES = ("augment", "replace")
    # Layer 44 (2026-05-31): the cluster_aggregate method allow-list expands
    # to include the four new combiners (``pca_pc2``, ``median_z``,
    # ``signed_max_abs``, ``signed_l2_sum``) so direct ``cluster_aggregate_methods``
    # API users can pin them individually, not just reach them via DCD ``auto``.
    _VALID_CLUSTER_AGGREGATE_METHODS = (
        "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score",
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    )
    # 2026-05-30 Wave 9 — DCD validation constants. swap_methods alias the
    # cluster_aggregate methods (Critic2/E fix: no duplicate constant).
    # Layer 46 (2026-05-31): ``"auto"`` runs SU and VI in parallel per pair
    # and returns the tighter redundancy score (``max(SU, VI_sim)``). Catches
    # both linear-friendly duplicates (SU strong) and non-linear functional
    # equivalences like y = f(x^2) (VI strong, SU silent).
    _VALID_DCD_DISTANCES = ("su", "vi", "sotoca_pla", "auto")
    # Layer 44: DCD ``dcd_swap_method`` accepts the same expanded combiner set
    # so users can pin a single new method instead of relying on ``auto``.
    _VALID_DCD_SWAP_METHODS = (
        "auto", "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score",
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    )

    # ``_validate_string_params`` + ``_validate_inputs`` are implemented
    # in ``_mrmr_validate_transform.py`` and bound onto this class at the
    # bottom of this module.

    # 2026-05-30 Wave 8 — opt-in stability-selection outer-loop wrapper.
    # Routes to Faletto-Bien 2022 Cluster Stability Selection or
    # Shah-Samworth 2013 Complementary Pairs Stability when
    # ``stability_selection_method != 'classic'``. The classic path falls
    # through to the legacy ``self.fit`` body.
    def _stability_outer_fit(self, X, y, **fit_kwargs):
        method = getattr(self, "stability_selection_method", "classic")
        if method == "classic":
            return None  # fall through to legacy fit
        from ._stability_cluster import (
            cluster_stability_selection,
            complementary_pairs_stability,
        )
        import pandas as pd
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = (
            y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        ).ravel()
        feature_names = (
            list(X.columns) if hasattr(X, "columns")
            else [f"f{i}" for i in range(X_arr.shape[1])]
        )

        def _inner_selector(X_sub, y_sub):
            X_sub_df = pd.DataFrame(X_sub, columns=feature_names)
            y_sub_s = pd.Series(y_sub, name="y")
            # Use a fresh sibling instance with classic method to avoid
            # recursion AND drop bootstrap-incompatible settings.
            sub = type(self)(
                **{
                    **{k: v for k, v in self.get_params().items()
                       if k not in (
                           "stability_selection_method",
                           "stability_selection_corr_threshold",
                           "uaed_auto_size",
                           "cmi_perm_stop",
                       )},
                    "stability_selection_method": "classic",
                    "verbose": 0,
                }
            )
            sub.fit(X_sub_df, y_sub_s)
            if not hasattr(sub, "support_") or sub.support_ is None:
                return np.asarray([], dtype=np.int64)
            return np.asarray(sub.support_, dtype=np.int64)

        if method == "cluster":
            corr_thr = float(
                getattr(self, "stability_selection_corr_threshold", 0.8)
            )
            sel, freq, info = cluster_stability_selection(
                X_arr, y_arr, _inner_selector,
                n_bootstrap=int(getattr(self, "stability_n_bootstrap", 50)),
                pi_threshold=float(
                    getattr(self, "stability_pi_threshold", 0.6)
                ),
                corr_threshold=corr_thr,
                rng_seed=int(self.random_seed or 0),
            )
        elif method == "complementary_pairs":
            sel, freq, info = complementary_pairs_stability(
                X_arr, y_arr, _inner_selector,
                n_pairs=int(getattr(self, "stability_n_bootstrap", 50)),
                pi_threshold=float(
                    getattr(self, "stability_pi_threshold", 0.6)
                ),
                rng_seed=int(self.random_seed or 0),
            )
        else:
            raise ValueError(f"unknown stability_selection_method={method!r}")

        # Persist the standard MRMR public-API attributes from the chosen set.
        self.support_ = np.asarray(sel, dtype=np.int64)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        self.n_features_ = int(self.support_.size)
        self.stability_freq_ = freq
        self.stability_info_ = info
        return self

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
            "strict_groups": False,  # added 2026-05-25; legacy pickles default to warn-only behaviour
            # Friend-graph post-analysis (added 2026-05-27). Legacy pickles refit with the
            # current defaults; ``friend_graph_`` itself is a fitted attribute, not seeded here.
            "build_friend_graph": True,
            "friend_graph_prune": False,
            "friend_graph_max_nodes": 200,
            "friend_graph_mi_eps": 1e-6,
            "friend_graph_edge_significance": 3.0,
            "friend_graph_garbage_min_degree": 3,
            "friend_graph_unique_ratio": 1.0,
            "friend_graph_unique_max_degree": 1,
            # Clustered-feature aggregation (added 2026-05-27). Off by default; legacy pickles refit
            # with these defaults.
            "cluster_aggregate_enable": True,
            "cluster_aggregate_mode": "augment",
            "cluster_aggregate_methods": ("mean_z",),
            "cluster_aggregate_mi_prevalence": 1.0,
            "cluster_aggregate_min_member_relevance": 0.0,
            "cluster_aggregate_min_cluster_size": 3,
            "cluster_aggregate_max_cluster_size": 12,
            "cluster_aggregate_corr_threshold": 0.6,
            "cluster_aggregate_homogeneity_tau": 0.6,
            "cluster_aggregate_max_candidates": 200,
            # 2026-05-31 Layer 23 — hybrid orthogonal-poly FE auto-wire.
            # Defaults preserve legacy behaviour: master switch OFF, so old
            # pickles unpickle to "hybrid FE disabled".
            "fe_hybrid_orth_enable": False,
            "fe_hybrid_orth_degrees": (2, 3),
            "fe_hybrid_orth_basis": "auto",
            "fe_hybrid_orth_top_k": 5,
            "fe_hybrid_orth_pair_enable": True,
            "fe_hybrid_orth_pair_max_degree": 2,
            # 2026-05-31 Layer 56 — triplet cross-basis FE defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_triplet_enable": False,
            "fe_hybrid_orth_triplet_max_degree": 1,
            "fe_hybrid_orth_triplet_seed_k": 4,
            "fe_hybrid_orth_triplet_top_count": 2,
            # 2026-05-31 Layer 57 — adaptive per-column degree defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_adaptive_degree_enable": False,
            "fe_hybrid_orth_adaptive_degree_range": (1, 2, 3, 4, 5, 6),
            "fe_hybrid_orth_adaptive_degree_min_uplift": 1.05,
            # 2026-05-31 Layer 58 — conditional basis routing FE defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_conditional_routing_enable": False,
            "fe_hybrid_orth_conditional_routing_top_k": 5,
            "fe_hybrid_orth_conditional_routing_min_uplift": 1.10,
            "fe_hybrid_orth_conditional_routing_degrees": (2, 3),
            # 2026-05-31 Layer 59 — diff-basis FE defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_diff_basis_enable": False,
            "fe_hybrid_orth_diff_basis_corr_threshold": 0.7,
            "fe_hybrid_orth_diff_basis_degrees": (1, 2, 3),
            "fe_hybrid_orth_diff_basis_top_k": 3,
            # 2026-05-31 Layer 61 — per-cluster shared-basis FE defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_cluster_basis_enable": False,
            "fe_hybrid_orth_cluster_basis_aggregator": "mean_z",
            "fe_hybrid_orth_cluster_basis_degrees": (2, 3),
            "fe_hybrid_orth_cluster_basis_top_k": 3,
            # 2026-05-31 Layer 62 — bootstrap-stable MI ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_bootstrap_enable": False,
            "fe_hybrid_orth_bootstrap_n_boot": 10,
            "fe_hybrid_orth_bootstrap_sample_fraction": 0.8,
            # 2026-05-31 Layer 63 — three-gate + K-fold OOF MI defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_three_gate_enable": False,
            "fe_hybrid_orth_three_gate_n_folds": 5,
            "fe_hybrid_orth_three_gate_cmi_min": 0.001,
            # 2026-05-31 Layer 65 — KSG / k-NN MI ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_ksg_enable": False,
            "fe_hybrid_orth_ksg_n_neighbors": 3,
            "fe_hybrid_orth_ksg_min_uplift": 0.95,
            "fe_hybrid_orth_ksg_min_abs_mi_frac": 0.05,
            # 2026-06-01 Layer 66 — copula-MI ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_copula_enable": False,
            "fe_hybrid_orth_copula_n_bins": 20,
            # 2026-06-01 Layer 67 — distance-correlation ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_dcor_enable": False,
            "fe_hybrid_orth_dcor_n_sample": 500,
            # 2026-06-01 Layer 71 — HSIC ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_hsic_enable": False,
            "fe_hybrid_orth_hsic_kernel": "rbf",
            "fe_hybrid_orth_hsic_n_sample": 500,
            # 2026-06-01 Layer 72 — JMIM (Bennasar 2015) defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_jmim_enable": False,
            "fe_hybrid_orth_jmim_n_bins": 10,
            # 2026-06-01 Layer 73 — TC (Watanabe 1960) ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_tc_enable": False,
            "fe_hybrid_orth_tc_n_bins": 10,
            # 2026-06-01 Layer 74 — CMIM (Fleuret 2004) ranking defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_cmim_enable": False,
            "fe_hybrid_orth_cmim_n_bins": 10,
            # 2026-06-01 Layer 68 — per-column scorer auto-selection defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_auto_scorer_enable": False,
            "fe_hybrid_orth_auto_scorer_n_boot": 5,
            # 2026-06-01 Layer 69 — ensemble rank-fusion defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_ensemble_enable": False,
            "fe_hybrid_orth_ensemble_aggregator": "mean_rank",
            "fe_hybrid_orth_ensemble_scorers": (
                "plug_in", "ksg", "copula", "dcor", "hsic",
            ),
            # 2026-06-01 Layer 76 — meta-scorer auto-selection defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_hybrid_orth_meta_enable": False,
            "fe_hybrid_orth_meta_force_scorer": None,
            # 2026-05-31 Layer 32 — extra-basis (spline / fourier) defaults.
            "fe_hybrid_orth_extra_bases": (),
            "fe_hybrid_orth_fourier_freqs": (1.0, 2.0),
            "fe_hybrid_orth_spline_knots": 5,
            # Fitted attribute (list of engineered names from hybrid stage);
            # legacy pickles default to empty list.
            "hybrid_orth_features_": [],
            # 2026-05-31 Layer 26 — MI-greedy FE constructor. Defaults
            # preserve legacy behaviour: master switch OFF.
            "fe_mi_greedy_enable": False,
            "fe_mi_greedy_top_k": 5,
            "fe_mi_greedy_seed_cols_count": 5,
            "fe_mi_greedy_include_unary": True,
            "fe_mi_greedy_include_binary": True,
            "mi_greedy_features_": [],
            # 2026-05-31 Layer 60 — CMI-greedy FE constructor. Defaults
            # preserve legacy behaviour: master switch OFF.
            "fe_mi_greedy_cmi_enable": False,
            "fe_mi_greedy_cmi_top_k": 5,
            "fe_mi_greedy_cmi_seed_cols_count": 4,
            "fe_mi_greedy_cmi_min_gain": 0.005,
            # 2026-05-31 Layer 33 — K-fold target-encoding FE defaults.
            # Master switch OFF preserves legacy pickle byte-equivalence.
            "fe_kfold_te_enable": False,
            "fe_kfold_te_cols": (),
            "fe_kfold_te_folds": 5,
            "fe_kfold_te_smoothing": 10.0,
            "kfold_te_features_": [],
            # 2026-05-31 Layer 34 — count / frequency / cat x num residual.
            # Master switches OFF preserve legacy pickle byte-equivalence.
            "fe_count_encoding_enable": False,
            "fe_count_encoding_cols": (),
            "fe_frequency_encoding_enable": False,
            "fe_frequency_encoding_cols": (),
            "fe_cat_num_interaction_enable": False,
            "fe_cat_num_interaction_cat_cols": (),
            "fe_cat_num_interaction_num_cols": (),
            "fe_cat_num_interaction_folds": 5,
            "fe_cat_num_interaction_smoothing": 10.0,
            "count_encoding_features_": [],
            "frequency_encoding_features_": [],
            "cat_num_interaction_features_": [],
            # 2026-05-31 Layer 53 — partial_fit / streaming refit.
            # Legacy pickles default OFF (decay 0, threshold 100, no window);
            # fitted-state buffers default to None until partial_fit is called.
            "partial_fit_decay": 0.0,
            "partial_fit_min_recompute": 100,
            "partial_fit_window": None,
            "_partial_fit_X_buffer_": None,
            "_partial_fit_y_buffer_": None,
            "_partial_fit_n_seen_": 0,
            "_partial_fit_n_since_refit_": 0,
            # 2026-05-31 Layer 54 — FE provenance tracking.
            # Legacy pickles default to ``None`` and the empty predictor log;
            # the next fit() repopulates from the live greedy run.
            "fe_provenance_": None,
            "_predictors_log_": (),
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
            if getattr(self, "strict_groups", False):
                raise NotImplementedError(
                    "MRMR.fit received groups but the current implementation does NOT consume them and "
                    "strict_groups=True was set. Either implement grouped MI estimation by wrapping MRMR "
                    "with a per-group selector and aggregating manually, set strict_groups=False to "
                    "accept the warn-only fallback, or pass groups=None."
                )
            warnings.warn(
                "MRMR.fit received groups but the current implementation does NOT consume them; "
                "MI is estimated per-row. For grouped MI estimation, wrap MRMR with a per-group "
                "selector and aggregate manually. Pass groups=None to silence this warning, or set "
                "strict_groups=True to raise instead.",
                UserWarning,
                stacklevel=2,
            )
        self._pandas_frame_for_target_cleanup = None
        self._target_names_for_cleanup = None

        # 2026-05-30 Wave 8 — Stability-selection outer-loop short-circuit.
        # When ``stability_selection_method`` is 'cluster' or
        # 'complementary_pairs', delegate to the bootstrap aggregator before
        # the legacy single-fit body executes.
        _stab_method = getattr(self, "stability_selection_method", "classic")
        if _stab_method != "classic":
            try:
                _stab_result = self._stability_outer_fit(
                    X, y, groups=groups, sample_weight=sample_weight,
                    **fit_params,
                )
            except Exception as _exc:
                warnings.warn(
                    f"MRMR stability_selection_method={_stab_method!r} outer-loop "
                    f"raised {type(_exc).__name__}: {_exc}. Falling back to classic fit.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                if _stab_result is not None:
                    return _stab_result

        # TODO B (2026-05-28): reject NaN/Inf in y at fit entry, matching the
        # sibling selectors (RFECV / ShapProxiedFS). Pre-fix MRMR let NaN flow
        # into the MI scorer and silently degraded relevance numbers. The
        # shared selector-contract test suite caught this asymmetry. Skip the
        # check on object-dtype y (categorical labels) where np.isnan would
        # raise; numeric / float / int paths get validated.
        _y_check = np.asarray(y)
        if _y_check.dtype.kind in "fc":
            _n_nan = int(np.isnan(_y_check).sum())
            _n_inf = int(np.isinf(_y_check).sum())
            if _n_nan or _n_inf:
                raise ValueError(
                    f"MRMR.fit: y contains {_n_nan} NaN and {_n_inf} +/-inf values. "
                    f"MI estimation silently degrades on NaN; drop or impute these rows "
                    f"before fitting."
                )

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

        # 2026-05-28: activate thread-local SU normalization when mi_normalization='su'.
        # The toggle is read by evaluation.py / Fleuret loops at the scoring site so
        # raw conditional_mi (and cached entropy numbers) stay legacy-bit-stable for
        # the default ``mi_normalization='none'`` path. Restored in finally so a
        # crashing _fit_impl can't leak SU mode into subsequent fits.
        from .info_theory import set_su_normalization, set_jmim_aggregator, set_bur_lambda
        _mi_norm = getattr(self, "mi_normalization", "none")
        if _mi_norm not in ("none", "su"):
            raise ValueError(
                f"MRMR.mi_normalization must be 'none' or 'su'; got {_mi_norm!r}."
            )
        _prev_su = _mi_norm == "su"
        set_su_normalization(_prev_su)
        # 2026-05-30 Wave 8 — activate JMIM aggregator + BUR weight thread-locals.
        # Both default OFF (redundancy_aggregator=None, bur_lambda=0.0) so the
        # legacy Fleuret path stays bit-stable.
        _jmim_on = getattr(self, "redundancy_aggregator", None) == "jmim"
        _bur_lambda = float(getattr(self, "bur_lambda", 0.0) or 0.0)
        set_jmim_aggregator(_jmim_on)
        set_bur_lambda(_bur_lambda)
        # 2026-05-30 Wave 9 — activate DCD thread-local. The DCDState dataclass
        # is constructed inside ``_screen_predictors`` (passed via dcd_config
        # kwarg) — joblib-safe; the thread-local is only the read-only branch
        # toggle. Reset in finally.
        from ._dynamic_cluster_discovery import set_dcd_active as _set_dcd_active
        _dcd_on = bool(getattr(self, "dcd_enable", False))
        _set_dcd_active(_dcd_on)
        # Critic1/H-3 fix: when DCD active and dcd_postoc_compose=False, suppress
        # the post-hoc cluster_aggregate FE-step (else double-aggregation). Save
        # and restore the original flag to keep the constructor-arg semantics
        # bit-stable across fits.
        _orig_cluster_aggregate_enable = bool(
            getattr(self, "cluster_aggregate_enable", True)
        )
        _dcd_suppress_postoc = _dcd_on and not bool(
            getattr(self, "dcd_postoc_compose", False)
        )
        if _dcd_suppress_postoc:
            self.cluster_aggregate_enable = False
        try:
            result = self._fit_impl(X, y, groups, **fit_params)
            try:
                from mlframe.training.provenance import record_provenance as _record_provenance
                _n_rows = int(X.shape[0]) if hasattr(X, "shape") else None
                # 2026-05-30 Wave 9.1 fix (loop iter 6): read ``random_seed``
                # (the documented public API) instead of ``random_state``. The
                # ctor at mrmr.py:641 promotes ``random_state -> random_seed``
                # but NOT the reverse, so when the user passed the documented
                # ``random_seed=42`` API directly, ``self.random_state`` stayed
                # at its default ``None`` and the provenance trail recorded
                # ``seed=None`` even though the actual kernel seed was 42.
                # Reading ``random_seed`` works for both APIs because the ctor
                # promotion guarantees it's populated from either source.
                _seed_resolved = getattr(self, "random_seed", None)
                if _seed_resolved is None:
                    _seed_resolved = getattr(self, "random_state", None)
                _seed_for_provenance = (
                    int(_seed_resolved) if _seed_resolved is not None else None
                )
                _record_provenance(
                    getattr(self, "_provenance_sink_", None),
                    "mrmr",
                    source="train_only",
                    n_rows=_n_rows,
                    seed=_seed_for_provenance,
                    extra={"n_features_in": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None},
                )
                self.provenance_ = {
                    "step": "mrmr",
                    "source": "train_only",
                    "n_rows": _n_rows,
                    "seed": _seed_for_provenance,
                }
            except Exception:
                pass
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
            # Layer 54 (2026-05-31): populate ``fe_provenance_`` from
            # the sibling module so users can audit which engineered
            # columns landed in support_, why (origin + mechanism
            # details) and what each contributed in the greedy gain
            # ledger. Pure metadata; never mutates the selection result.
            from ._mrmr_fe_provenance import populate_fe_provenance as _pop_prov
            _pop_prov(self)
            return result
        finally:
            # 2026-05-28: restore SU thread-local to its pre-fit state (always False
            # outside of a MRMR fit -- nested fits are not supported by the simple
            # toggle, but neither is the cache layer they would share).
            try:
                set_su_normalization(False)
            except Exception:
                pass
            # 2026-05-30 Wave 8 — reset JMIM + BUR thread-locals.
            try:
                set_jmim_aggregator(False)
            except Exception:
                pass
            try:
                set_bur_lambda(0.0)
            except Exception:
                pass
            # 2026-05-30 Wave 9 — reset DCD thread-local and restore
            # cluster_aggregate_enable to its constructor value (Critic2 fix:
            # missing reset in v1 plan).
            try:
                _set_dcd_active(False)
            except Exception:
                pass
            try:
                self.cluster_aggregate_enable = _orig_cluster_aggregate_enable
            except Exception:
                pass
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

    def export_artifacts(self) -> dict:
        """Return the in-fit reusable intermediates as a dict for downstream selectors.

        Requires the estimator to have been constructed with
        ``retain_artifacts=True`` (off by default to preserve the legacy memory
        footprint) and to have been fitted. The returned dict carries
        Symmetric Uncertainty + direct MI vectors against y, plus -- when
        ``retain_bins=True`` -- the per-column binned arrays. Schema is defined
        in ``_mrmr_artifacts._ARTIFACT_SCHEMA``; consumers MUST tolerate missing
        optional keys for forward compat.

        The canonical consumer is ``ShapProxiedFS(precomputed=...)``: passing
        ``mrmr.export_artifacts()`` lets that selector skip its own univariate
        F-statistic pre-screen and rank by MRMR's SU(X_j, y) instead. The
        selected subset is unchanged for SU-vs-F-ranking-equivalent regimes;
        the win is wall-clock + a more cardinality-honest ranking on mixed-
        cardinality data.

        Raises
        ------
        ValueError
            If ``self.retain_artifacts`` is False (artifacts were not captured).
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        """
        if not getattr(self, "retain_artifacts", False):
            raise ValueError(
                "MRMR.export_artifacts() requires retain_artifacts=True at construction time. "
                "Re-construct as MRMR(retain_artifacts=True, ...) and fit before exporting."
            )
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["support_"])
        artifacts = getattr(self, "_artifacts_", None)
        if not artifacts:
            # retain_artifacts=True was set but the in-fit capture did not
            # populate the dict -- likely a fit() path that bypassed
            # _fit_impl (identity shortcut, FIT_CACHE hit on a pre-Wave-66
            # cached instance, stability-selection outer loop). Surface a
            # clear error so the caller can adjust the pipeline rather than
            # silently consuming an empty dict.
            raise ValueError(
                "MRMR.export_artifacts(): retain_artifacts=True but the in-fit capture did "
                "not populate self._artifacts_. The fit may have hit the identity-shortcut "
                "cache or a pre-retain_artifacts cached instance; refit with "
                "MRMR._FIT_CACHE.clear() and mrmr_skip_when_prior_was_identity=False."
            )
        return artifacts

    def _fit_identity_shortcut(self, X) -> None:
        """Populate the fit-result attributes as if MRMR returned the input X unchanged.

        Used by the cross-target identity cache (2026-05-18 #2): when a previous fit on the SAME X returned identity (all input columns selected, zero engineered features), subsequent calls with a different y can skip the entire FE pipeline since the only y-dependent thing -- the selected feature subset -- is forced to "all input columns".
        """
        n_cols = X.shape[1] if X.ndim > 1 else 1
        self.support_ = np.arange(n_cols, dtype=np.int64)
        # 2026-05-30 Wave 9.1 fix (loop iter 35): the prior expression
        # ``X.columns.tolist() if hasattr(X.columns, "tolist") else
        # list(X.columns) if hasattr(X, "columns") else [...]`` was a
        # mis-parenthesised ternary. Python parses it as
        # ``A if B1 else (C if B2 else E)``, evaluating ``B1 =
        # hasattr(X.columns, "tolist")`` BEFORE the outer ``B2 =
        # hasattr(X, "columns")`` guard. The inner ``X.columns`` access
        # raised AttributeError on ndarray X, so the identity-shortcut
        # cache-hit path (opt-in via ``mrmr_skip_when_prior_was_identity``)
        # crashed on every ndarray fit instead of short-circuiting.
        if hasattr(X, "columns"):
            _cols = X.columns
            self.feature_names_in_ = (
                _cols.tolist() if hasattr(_cols, "tolist") else list(_cols)
            )
        else:
            self.feature_names_in_ = [f"f{i}" for i in range(n_cols)]
        self._engineered_features_ = []
        self._engineered_recipes_ = {}
        self.n_features_in_ = int(n_cols)
        self.n_features_ = int(n_cols)
        self.fallback_used_ = False
        # 2026-05-30 Wave 9.1: set DCD/diagnostic fitted attrs to safe
        # defaults so the identity shortcut produces a
        # fitted-state-complete estimator (matches full-fit attribute
        # surface). Without these the cache-replay tests and
        # downstream consumers that introspect ``sel.dcd_`` /
        # ``sel.mrmr_gains_`` /``sel.friend_graph_`` /
        # ``sel.cluster_aggregate_`` blow up on the shortcut path.
        self.dcd_ = None
        # Layer 41 (2026-05-31): identity-shortcut path must also expose the
        # ``cluster_members_`` attribute (None when DCD was disabled or did
        # not run) so introspection code paths don't AttributeError.
        self.cluster_members_ = None
        # Layer 48 (2026-05-31): hierarchical post-hoc cluster map. Empty
        # dict default (matches "DCD ran but found no super-structure" --
        # meaningfully different from None, which would mean DCD disabled).
        # Identity shortcut bypasses DCD entirely, so the empty default is
        # the correct attribute-complete marker.
        self.cluster_hierarchy_ = {}
        self.mrmr_gains_ = np.array([], dtype=np.float64)
        self.friend_graph_ = None
        self.cluster_aggregate_ = None
        self.ran_out_of_time_ = False
        self.provenance_ = None
        self._feature_names_in_synthesized_ = not hasattr(X, "columns")
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

        Per the sklearn protocol (BaseEstimator._check_feature_names):
        - When ``input_features`` is None: use ``feature_names_in_``.
        - When ``input_features`` is provided AND fit-time saw real names
          (DataFrame input): the two MUST match or a ``ValueError`` is raised.
          This is the Pipeline column-drift detection contract.
        - When ``input_features`` is provided AND fit-time was an ndarray:
          synthesized ``feature_N`` placeholders are opaque; honour the
          caller's names. This lets Pipelines that name columns downstream
          (e.g. ColumnTransformer + array math + name re-injection) propagate.

        2026-05-30 Wave 9.1 fix (loop iter 12): pre-fix the ``input_features``
        argument was accepted but silently ignored on every code path, so:
        (a) Pipeline column-drift detection was bypassed - mismatched
        ``input_features`` produced fit-time names with no warning;
        (b) After ndarray fit, user-supplied ``input_features`` were dropped
        and synthesized ``feature_N`` placeholders propagated to downstream
        consumers.
        """
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This MRMR instance is not fitted yet. Call 'fit' before "
                "using 'get_feature_names_out'."
            )
        # Resolve effective fit-time feature names. If ``input_features`` was
        # provided, validate against the saved ``feature_names_in_`` (sklearn
        # column-drift protocol) - but only when fit saw real names. The
        # ndarray-fit path synthesises ``feature_N`` placeholders which the
        # caller can override.
        if input_features is not None:
            in_names = np.asarray(input_features, dtype=object)
            saved = np.asarray(self.feature_names_in_, dtype=object)
            # 2026-05-30 Wave 9.1 fix (loop iter 27): use the
            # ``_feature_names_in_synthesized_`` sentinel set at fit
            # time instead of the brittle ``startswith("feature_")``
            # heuristic. The heuristic misclassified legitimate
            # DataFrame columns the user happened to name
            # ``feature_<n>`` (very common pattern after
            # ``pd.DataFrame(arr)`` + rename) and silently bypassed
            # the sklearn column-drift contract -
            # ``get_feature_names_out(['totally_wrong_A','B','C'])``
            # returned ``['totally_wrong_A']`` instead of raising.
            # Back-compat fallback for unpickled pre-iter-27 estimators
            # without the sentinel: require an EXACT regex match
            # (anchored, ``feature_\d+$``) AND count parity, not just
            # ``startswith``.
            synthesized = getattr(self, "_feature_names_in_synthesized_", None)
            if synthesized is None:
                import re as _re
                _placeholder = _re.compile(r"^feature_\d+$")
                synthesized = all(
                    _placeholder.match(str(n)) is not None for n in saved
                )
            if not synthesized:
                if (len(in_names) != len(saved)
                        or not np.array_equal(in_names, saved)):
                    raise ValueError(
                        f"input_features is not equal to feature_names_in_. "
                        f"Got {list(in_names)[:8]}, expected "
                        f"{list(saved)[:8]}."
                    )
                fni = saved
            else:
                # ndarray-fit case: caller's names take precedence.
                if len(in_names) != len(saved):
                    raise ValueError(
                        f"input_features length ({len(in_names)}) does not "
                        f"match the number of features seen at fit "
                        f"({len(saved)})."
                    )
                fni = in_names
        else:
            fni = np.asarray(self.feature_names_in_, dtype=object)
        support = self.support_
        engineered_names = [r.name for r in getattr(self, "_engineered_recipes_", [])]
        if len(support) == 0 and not engineered_names:
            return np.array([], dtype=object)
        if len(support) > 0 and isinstance(support[0], (bool, np.bool_)):
            base_names = [n for n, s in zip(fni, support) if s]
        else:
            base_names = [fni[i] for i in support]
        return np.asarray(list(base_names) + engineered_names, dtype=object)

    # 2026-05-30 Wave 9.1 fix (loop iter 43): explicit
    # ``__sklearn_is_fitted__`` and ``get_support`` so sklearn's
    # ``check_is_fitted`` / ``SelectorMixin`` consumers behave
    # correctly.
    # Pre-fix the class declared only ``BaseEstimator, TransformerMixin``
    # with no ``__sklearn_is_fitted__``, so ``check_is_fitted`` fell
    # back to a heuristic scanning for ANY trailing-underscore attr.
    # ``_mrmr_fit_impl`` sets ``feature_names_in_`` / ``n_features_in_``
    # ~700 lines BEFORE ``support_`` (line 942), so a fit() that
    # crashed mid-screen left a half-fit instance that
    # ``check_is_fitted`` accepted but ``transform`` then refused with
    # ``NotFittedError`` - confusing for any downstream gate that used
    # the canonical check.
    # Also added ``get_support`` to honour the SelectorMixin contract
    # downstream consumers (sklearn Pipeline, RFECV, monitoring hooks)
    # expect.
    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "support_") and hasattr(self, "feature_names_in_")

    def get_support(self, indices: bool = False):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)
        mask = np.zeros(int(self.n_features_in_), dtype=bool)
        _supp = np.asarray(self.support_, dtype=np.intp)
        if _supp.size:
            mask[_supp] = True
        return np.where(mask)[0] if indices else mask

    # ``transform`` + ``_append_engineered`` are implemented in
    # ``_mrmr_validate_transform.py`` and bound onto this class at the
    # bottom of this module.
    def transform(self, X, y=None):
        """sklearn-1.x transformer protocol. Delegates to the
        implementation in ``_mrmr_validate_transform.py``.

        2026-05-30 Wave 9.1 fix (loop iter 34): defined directly on the
        class body (rather than late-bound at module bottom) so
        ``_SetOutputMixin.__init_subclass__`` actually wraps it. Pre-fix
        the bottom-of-module ``MRMR.transform = _transform_func`` rebind
        nuked the wrapper that ``__init_subclass__`` had attached
        during class definition, silently making
        ``MRMR.set_output(transform='pandas')`` a no-op when transform
        was called directly with ndarray input (the canonical sklearn
        contract requires a DataFrame).
        """
        from ._mrmr_validate_transform import transform as _t
        return _t(self, X, y)





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
# 2026-05-30 Wave 9.1 fix (loop iter 34): ``transform`` is now defined
# on the class body above (as a thin delegator) so
# ``_SetOutputMixin.__init_subclass__`` wraps it correctly. Do NOT
# late-rebind ``MRMR.transform`` here - that would replay the original
# bug by stripping the wrapper.
MRMR._append_engineered = _append_engineered_func

# Layer 53 (2026-05-31): incremental / streaming refit. ``partial_fit``
# lives in the sibling module so the parent stays under the 1.8k LOC budget;
# binding here keeps the public sklearn-style API on the class surface.
from ._mrmr_partial_fit import partial_fit as _partial_fit_func  # noqa: E402
MRMR.partial_fit = _partial_fit_func

# Layer 54 (2026-05-31): FE provenance report binding. The DataFrame is
# populated inside ``fit`` (see _mrmr_fe_provenance.compute_fe_provenance);
# ``get_fe_report`` is a thin renderer bound here so the public surface
# stays on the class without forcing the parent module to take a new
# heavyweight import at load time.
from ._mrmr_fe_provenance import get_fe_report as _get_fe_report_func  # noqa: E402
MRMR.get_fe_report = _get_fe_report_func
