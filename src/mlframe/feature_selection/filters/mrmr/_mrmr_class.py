"""sklearn-compatible MRMR estimator -- the irreducible class body.

The ``MRMR`` class moved here verbatim from the former ``mrmr.py`` monolith; the ``mrmr`` package facade
(``__init__.py``) re-exports it (and sets ``MRMR.__module__`` to the package path so pickle resolves), runs the
method bindings, and re-exports the full historical public surface. ``MRMR.__module__`` is rewritten by the facade
so do not rely on this submodule's path for pickle.

Tolerates FE-engineered feature names in the post-fit support index map (routes synthetic names through
``_engineered_features_``). Includes an explicit input-validation contract in ``_validate_inputs`` and an
``__setstate__`` shim that injects defaults for newer kwargs / attributes so old joblib / cloudpickle pipelines
unpickle cleanly.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
import warnings
from collections import OrderedDict
from typing import Any, Callable, ClassVar, Iterable, NoReturn, Optional, Sequence, cast

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import BaseCrossValidator

# Top-level helpers (histogram + fingerprint/hash + replay + chunker) live in
# ``_mrmr_fingerprints.py``; re-imported below so the parent module and
# downstream callers continue to resolve historical names.
# Pure-data constants carved out of this module (no class refs -> safe top-level import):
# the constructor-param validation allow-lists and the legacy-pickle default-injection roster.
from ._mrmr_param_constants import (
    _VALID_QUANTIZATION_METHODS,
    _VALID_NAN_STRATEGIES,
    _VALID_MRMR_RELEVANCE_ALGOS,
    _VALID_MRMR_REDUNDANCY_ALGOS,
    _VALID_NBINS_STRATEGIES,
    _VALID_MI_CORRECTIONS,
    _VALID_REDUNDANCY_AGGREGATORS,
    _VALID_STABILITY_SELECTION_METHODS,
    _DEMOTED_NBINS_STRATEGIES,
    _VALID_FE_UNARY_PRESETS,
    _VALID_FE_BINARY_PRESETS,
    _VALID_CLUSTER_AGGREGATE_MODES,
    _VALID_CLUSTER_AGGREGATE_METHODS,
    _VALID_DCD_DISTANCES,
    _VALID_DCD_SWAP_METHODS,
    _VALID_RFECV_SELECTION_RULES,
    _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS,
)
from ._mrmr_setstate_defaults import build_setstate_defaults
from ._mrmr_config_dataclasses import (
    FastSearchConfig,
    StabilitySelectionConfig,
    SynergyRedundancyConfig,
    GroupAwareConfig,
    DCDConfig,
    HybridOrthConfig,
    apply_mrmr_config_objects,
)

from .._mrmr_fingerprints import (
    _mrmr_compute_y_fingerprint_sample,
    _mrmr_compute_x_fingerprint,
    _mrmr_y_corr_sample,
    _mrmr_y_corr,
    _hashable_params_signature,
    _MRMR_IDENTITY_FP_CACHE,
    _MRMR_IDENTITY_FP_LOCK,
)

from pyutilz.pythonlib import (
    get_parent_func_args,
    store_params_in_object,
)

from mlframe.utils.misc import hygienic_fit

from ..feature_engineering import UNIFIED_FE_SUBSAMPLE_N
from ..screen import _preserve_global_numpy_rng_state

# Hoisted from scattered ``fit()``-body local imports:
# each of these previously re-ran its ``from ... import ...`` statement on every single
# ``fit()`` call. The per-call cost of a cached ``sys.modules`` lookup is sub-microsecond,
# but with 15+ such sites executing every fit it adds up to real, pointlessly-repeated work,
# and it obscures the module's actual dependency surface. None of these targets import back
# from ``mrmr``/``feature_selection`` (verified: no circular-import risk) and none eagerly
# import an optional heavy dependency (cupy/torch imports inside these modules are already
# themselves lazy, function-local) at module scope, so hoisting is safe in a GPU-less/minimal
# environment. ``import polars as _pl`` at the polars-Struct-column check stays local: it is
# genuinely conditional on an optional dependency (a caller who never passes polars input
# should never pay for -- or require -- a polars import).
from .._param_accuracy_warnings import warn_accuracy_suboptimal_params
from .._mrmr_degenerate import audit_degenerate_columns
from .._meta_fe_recommender import recommend_fe_flags_by_rules
from .._synergy_detector import detect_synergy
from .._dynamic_cluster_discovery import set_dcd_active as _set_dcd_active
from .._fe_gpu_strict import set_auto_fit_n as _set_auto_fit_n, clear_auto_fit_n as _clear_auto_fit_n
from .._mrmr_fe_provenance import populate_fe_provenance as _pop_prov
from .._fe_rejection_ledger import populate_fe_rejection_ledger as _pop_rej
from .._fe_family_timing import log_fe_family_summary as _log_fe_wall
from .._fourier_detect_cap import clear_fourier_detect_cap, set_fourier_detect_cap
from .._mrmr_validate_transform import transform as _mrmr_transform_impl
from ..info_theory._state_and_dispatch import set_group_mi as _set_group_mi
from ..info_theory._group_mi import prepare_group_segments as _prepare_group_segments
from ..info_theory import (
    set_su_normalization, set_jmim_aggregator, set_bur_lambda, set_mi_miller_madow,
    set_mi_chao_shen, use_mi_chao_shen,
    set_relaxmrmr_alpha, set_pid_synergy_bonus, set_cmi_perm_stop, set_cpt_test,
    use_su_normalization, use_jmim_aggregator, get_bur_lambda, use_mi_miller_madow,
    get_relaxmrmr_alpha, get_pid_synergy_bonus, get_cmi_perm_stop, get_cpt_test,
)
from mlframe.training.provenance import record_provenance as _record_provenance

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Sentinel distinguishing "constructor attribute was not overridden during fit" from a stored value of None.
_UNSET = object()


def _mrmr_y_is_multioutput(y) -> bool:
    """True when y carries >=2 target columns (2D DataFrame / 2D ndarray); a Series / 1D array / single-column frame is single-target."""
    if isinstance(y, pd.DataFrame):
        return bool(y.shape[1] >= 2)
    if str(type(y).__module__).startswith("polars") and type(y).__name__ == "DataFrame":
        return bool(y.shape[1] >= 2)
    try:
        arr = np.asarray(y)
    except Exception as exc:
        logger.debug("mrmr: multi-target detection np.asarray(y) failed; treating as single-target: %r", exc, exc_info=True)
        return False
    return arr.ndim >= 2 and arr.shape[-1] >= 2


def _safe_restore(action: Callable[[], Any], description: str) -> None:
    """Run a single ``fit()``-finally restore action, swallowing any exception it raises into a debug
    log line instead of letting a failed restore mask the fit's real outcome (success or a genuine
    error) or abort the remaining restores. Replaces 11 near-identical
    ``try: ... except Exception as e: logger.debug("suppressed in _mrmr_class.py:<N>: %s", e)`` blocks
    that each hardcoded their own source line number -- numbers which drifted out of sync with the
    actual line on every subsequent edit, silently giving false diagnostics forever once stale.
    ``description`` is a short human-readable label instead, which cannot go stale the same way."""
    try:
        action()
    except Exception as exc:  # nosec B110 - a restore-step failure must never break/mask the fit's real outcome
        logger.debug("mrmr: fit()-finally restore failed (%s): %r", description, exc, exc_info=True)


from ._mrmr_class_shared import _mrmr_y_columns  # noqa: F401 -- re-exported for callers importing from here

from ._mrmr_class_config import _MRMRConfigMixin
from ._mrmr_class_transform import _MRMRTransformMixin
from ._mrmr_class_fit_helpers import _MRMRFitHelpersMixin

# SelectorMixin ADDED purely for the isinstance(x, SelectorMixin) contract and its
# ``inverse_transform``/``get_feature_names_out`` conveniences -- MRMR's OWN ``transform()`` (defined directly on
# this class body, see its own docstring below) always wins regardless of MRO since an own-class-body method
# beats any inherited one, so ``transform()`` still returns the FE-engineered columns (not SelectorMixin's
# mask-only slice). ``get_feature_names_out``/``get_support`` (on ``_MRMRTransformMixin``) still win over
# SelectorMixin's versions via MRO ordering below.
# MRO ordering is load-bearing: SelectorMixin itself subclasses TransformerMixin, so (a) SelectorMixin MUST precede
# TransformerMixin in this tuple (else C3 linearization raises TypeError at class-definition time), and (b)
# ``_MRMRTransformMixin`` MUST precede SelectorMixin so its get_feature_names_out()/get_support() resolve first
# via MRO -- confirmed by test_mrmr_selectormixin_mro.py pinning both facts.
# Pickle schema version, stamped by ``MRMR.__getstate__`` and
# checked by ``MRMR.__setstate__``. Bump only when a pickle-relevant change lands that the
# legacy-injection roster (``_mrmr_setstate_defaults.py``) can't fully paper over by itself -- e.g. a
# param RENAME (not just added/removed) or a change to what a stored value MEANS. Purely additive
# params (new ctor default, picked up automatically by the fresh-instance catch-all in
# ``__setstate__``) do not need a bump. This is a coarse downgrade DETECTOR, not a migration engine: an
# older installed mlframe loading a newer pickle still can't understand a newer schema either way, but
# the version mismatch lets it WARN instead of silently misbehaving.
_MRMR_SCHEMA_VERSION = 1


class MRMR(BaseEstimator, _MRMRTransformMixin, SelectorMixin, TransformerMixin, _MRMRConfigMixin, _MRMRFitHelpersMixin):
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

    degenerate_columns_ : dict
        Diagnostic surface populated at ``fit``: ``{column -> reason}`` for every
        pathological input column detected by a cheap O(p) scan, where reason is one of
        ``"all_nan"`` / ``"constant"`` / ``"duplicate_of:<col>"`` / ``"collinear_with:<col>"``.
        PURELY DIAGNOSTIC -- it does not remove columns or change which features are
        selected (the relevance + conditional-MI redundancy gates already handle
        degenerate columns); it mirrors the sibling selectors' diagnostic attributes so a
        downstream report / UI can see what the frame contained.

    Notes
    -----
    ``cluster_aggregate_enable`` (ON by default; gated so it is a no-op without genuine clusters) turns on clustered-feature aggregation: correlated
    "reflection" features (noisy copies of a hidden factor ``z``) are combined into one denoised
    aggregate (``mean_z`` / ``mean_inv_var`` / ``pca_pc1`` / ``factor_score`` / ``median``) that recovers
    ``z`` better than any single reflection. Adopted only if it beats the best member's MI with the
    target. Helps capacity-limited / linear downstreams, redundant sensor data, interpretability, and
    tight feature budgets; for tree/GBM downstreams expect no-harm rather than a lift (trees already
    average reflections via splits). ``augment`` adds the aggregate; ``replace`` also drops the members.

    Caching
    -------
    A previously-fitted ``MRMR`` instance's internal ndarray
    fitted attributes may become READ-ONLY as a side effect of a LATER, unrelated ``.fit()`` call
    elsewhere in the process. This happens when that later call hits the process-wide ``_FIT_CACHE``
    (identical params + content as this instance's own fit) and replays from this instance: the replay
    logic freezes (``flags.writeable = False``) this instance's own large ndarray attributes in place so
    they can be safely shared rather than copied. A previously fully-mutable, already-returned instance
    can therefore start raising ``ValueError: assignment destination is read-only`` on
    ``instance.some_ndarray_attr[i] = x`` with no code change on the caller's part. Set ``fit_cache_max=0``
    to opt out of the shared cache entirely if in-place mutation of fitted attributes is required.
    average reflections via splits). ``augment`` adds the aggregate; ``replace`` also drops the members.
    """

    # Set dynamically (to None in _mrmr_class_fit_helpers.py's identity-shortcut path, to a dict in the
    # provenance-recording block below) -- annotated here so mypy sees the full attribute contract.
    provenance_: dict[str, Any] | None

    # Process-wide cache of fitted state, keyed by (content_sig(X), content_sig(y), params_signature). When the
    # training suite iterates over models (clone()ing the pre-pipeline MRMR each time, stripping
    # ``_cat_fe_cache_``), subsequent fits on the same arrays hit this cache and skip the full cat-FE +
    # permutation work. LRU-bounded via ``OrderedDict`` + ``fit_cache_max`` (default 4) so long-lived workers
    # do not leak; ``MRMR._FIT_CACHE.clear()`` between suites still drains the lot. Cache hit: replay all
    # fitted attributes onto ``self`` and return early; constructor params are NEVER overwritten (the key
    # already includes the params signature, so a hit guarantees matching state).
    _FIT_CACHE: "ClassVar[OrderedDict[tuple, MRMR]]" = OrderedDict()  # noqa: RUF012 -- intentional shared class-level LRU cache, not a per-instance mutable-default bug


    # Private, non-BaseEstimator instance flag: when set True by a
    # caller BEFORE ``fit()`` (e.g. the stability-selection outer loop's throwaway bootstrap-replicate
    # sub-fits), ``fit()`` skips storing this instance's own entry in the process-wide ``_FIT_CACHE`` --
    # for a guaranteed-future-miss fit (a different row-subsample every call) that would only evict a
    # legitimately-reusable entry belonging to an unrelated concurrent caller. Not a constructor param
    # (must never appear in ``get_params()``/``clone()``); declared here at class scope only so mypy
    # resolves the attribute.
    _skip_fit_cache: bool = False

    # Fast-search sub-knob overrides applied for the duration of a fit when ``fe_fast_search=True``.
    # Each entry is (attr, fast_value). The override is applied ONLY when the current attr value still
    # equals its package default (so an explicit user value always wins). ``fe_check_pairs_subsample_n``
    # is resolved separately via kernel_tuning_cache (HW/size aware) -- see ``_apply_fast_search_profile``.
    _FAST_SEARCH_OVERRIDES = (
        ("fe_max_steps", 1),
        ("fe_pair_prewarp_enable", False),
        ("fe_stability_vote_enable", False),
        ("fe_escalation_underdelivery_enable", False),
    )

    # DEFAULT screen-subsample for the MI/FE candidate search, resolved (HW/size-aware) via the
    # kernel_tuning_cache. This is the FEATURE-RECOVERY default (distinct from the bit-stability
    # ``_fast_search_default_subsample_n`` 90k fallback): the FE MI-sweep / polynom-pair / conditional-gate
    # DETECTION are all rank-stable under row subsampling and the FINAL survivor columns are replayed at
    # FULL n (the recipe), so subsampling only the SCREEN cannot lose train-time precision -- it can only
    # move a borderline MI tie. The FE-screen accuracy bench (bench_fe_pair_subsample_accuracy.py) measured
    # survivor jaccard 1.0 / winner-match 5/5 vs the full-n screen at n_eff>=25_000, i.e. the 25k screen
    # reproduces the 200k-default survivor set EXACTLY while cutting the MI-sweep buffer ~8x. The canonical
    # n=100k additive-regression fit drops 168.8s -> ~75s and still recovers the a**2/b and log(c)*sin(d)
    # compounds. NEVER hardcode a per-HW threshold (mlframe is shared infra): a quiet/large-RAM box can
    # record a larger ``subsample_n`` under the ``mrmr_default_screen_n`` cache key and override the floor.
    # UNIFIED: the screen subsample is the SAME single knob as the FE pair-search / fast-preset
    # (``feature_engineering.UNIFIED_FE_SUBSAMPLE_N``) -- one source of truth, KTC-tuned per host under the
    # ``mrmr_default_screen_n`` cache key (``_default_screen_subsample_n``). >25k validated floor (jaccard 1.0
    # vs full-n screen), headroom for the gate-detection MI band.
    _DEFAULT_SCREEN_SUBSAMPLE_N = UNIFIED_FE_SUBSAMPLE_N

    def __init__(
        self,
        # quantization
        quantization_method: str = "quantile",
        quantization_nbins: int = 10,  # [ACCURACY-CAVEAT] <5 is too coarse for the plug-in MI; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        quantization_dtype: type = np.int32,
        # Cap categorical column cardinality: any categorical with more than this many distinct codes folds its rare tail
        # into one "other" bucket (top-(cap-1) by frequency kept). None (default) = uncapped. A high-cardinality categorical
        # has sparse contingency cells so its plug-in MI/CMI is unreliable regardless (the analytic null guards on >=5
        # expected/cell); capping DENSIFIES the cells (better MI) AND lets the whole codes matrix stay a narrow int -- set
        # to <=127 to keep the compact-codes storage int8 (4x smaller) even when legitimate high-card categoricals exist.
        max_categorical_cardinality: int | None = None,
        # per-feature adaptive bin chooser. Default
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
        # 2026-07-19: ``nbins_strategy='mdlp'`` now runs significance-gated ("validated")
        # splitting by DEFAULT instead of the classic in-sample MDL threshold + depth cap --
        # measured real accuracy win on held-out RMSE (see supervised_binning.py's
        # ``mdlp_bin_edges`` docstring) at a 20-80x per-column cost. Pass
        # ``nbins_strategy_kwargs={"mdlp_fast_mode": True}`` to opt back into the cheap
        # classic path for a specific run where wall-time matters more (e.g. a quick
        # exploratory fit); other recognised keys: ``mdlp_alpha``, ``mdlp_n_permutations``,
        # ``mdlp_bonferroni``, ``mdlp_max_y_classes``, ``mdlp_backend``, ``mdlp_scaled_min_split``.
        nbins_strategy_kwargs: dict | None = None,
        # Shared per-column bin-count ceiling applied to every adaptive strategy whose own formula
        # has no natural upper bound (knuth, bayesian_blocks, freedman_diaconis) -- see
        # ``_adaptive_nbins.MAX_ADAPTIVE_NBINS``. 256 matches MDLP's own implicit ceiling
        # (max_depth=8 -> 2**8 leaves), so every strategy answers "how many bins can one column
        # produce" the same way by default. Lower it (e.g. 64) to bound downstream pairwise-MI cost
        # further on very wide / high-row-count real data; per-method overrides
        # (``nbins_strategy_kwargs={"knuth_m_max_cap": ..., "bb_m_max_cap": ...}``) still win when set.
        max_adaptive_nbins: int = 256,
        # Large-n REGRESSION adaptive quantization gate. On a large-n regression target the supervised MDLP per-feature binning
        # under-resolves a heavy-tailed continuous y: the 180-cell large-n MRMR campaign (reg n=100k, 15 seeds) measured a 15/15
        # paired win for fixed 20-bin quantile over MDLP -- holdout R2 0.597 vs 0.481 (+0.116, std 0.0025) and F1 0.909 vs 0.667
        # (+0.242, std 0.0). The same fixed-20 path LOST at reg n=20k (holdout -0.143) and at classification (clf n=20k holdout
        # -0.052; clf n=100k exact tie), so the win is regime-specific, not a blanket flip. This knob gates the campaign-winner
        # config (nbins_strategy=None, quantization_nbins=20) ON exactly where it wins: detected regression AND n_rows >= the
        # threshold below, ONLY when the user left both quantization params at their constructor defaults (explicit user values
        # are never overridden). Set to False to keep MDLP everywhere (pre-2026-06-18 behaviour); set the threshold to retune.
        adaptive_nbins_large_n_reg: bool = True,
        adaptive_nbins_large_n_reg_threshold: int = 50_000,
        adaptive_nbins_large_n_reg_nbins: int = 20,
        # 10 new research-grade opt-in knobs (sibling modules):
        # F13 Chao-Shen entropy correction (Pawluszek-Filipiak 2025).
        #   'none' (default) | 'miller_madow' | 'chao_shen'
        mi_correction: str = "none",
        # A1 JMIM redundancy aggregator (Bennasar 2015). Alternative to Fleuret CMIM
        # ``min_k I(X_k; Y | Z_j)``; JMIM uses ``min_j I(X_k, X_j; Y)`` which
        # preserves synergy that CMIM rejects on multi-collinear groups.
        #   None (legacy) | 'jmim' | 'auto' (data-dependent: a cheap pre-fit synergy probe routes to JMIM
        #   only when the data is synergistic, else stays plain Fleuret -- so the additive-regime
        #   over-selection that keeps 'jmim' opt-in is avoided. See _synergy_detector.detect_synergy.)
        redundancy_aggregator: str | None = None,
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
        # D10 Conditional Permutation Test (Berrett, Wang, Barber, Samworth 2020). Permutes the candidate WITHIN each
        # already-selected-feature stratum (preserving X | selected), giving valid p-values under arbitrary confounding
        # by the selected set. Complements cmi_perm_stop above: dropped candidate iff p >= 0.05. See evaluation.py's
        # per-candidate scoring gate for the implementation (``_conditional_permutation.conditional_permutation_test``).
        cpt_test: bool = False,
        cpt_n_permutations: int = 200,
        # E11 Cluster Stability Selection (Faletto-Bien 2022). Opt-in via
        # ``stability_selection_method='cluster'``. Default 'classic' is the
        # existing Meinshausen-Buhlmann + Shah-Samworth path.
        # E12 Complementary Pairs Stability (Shah-Samworth 2013) accessible
        # via ``stability_selection_method='complementary_pairs'``.
        stability_selection_method: str = "classic",
        stability_selection_corr_threshold: float = 0.8,
        stability_n_bootstrap: int = 50,
        stability_pi_threshold: float = 0.6,
        # F14 PID decomposition (Williams-Beer + Ince I_ccs). When enabled,
        # synergistic features bypass the standard redundancy gate.
        pid_synergy_bonus: float = 0.0,
        # MI normalization knob to combat the cardinality bias.
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
        factors_names_to_use: Sequence[str] | None = None,
        factors_to_use: Sequence[int] | None = None,
        # algorithm
        mrmr_relevance_algo: str = "fleuret",
        mrmr_redundancy_algo: str = "fleuret",
        reduce_gain_on_subelement_chosen: bool = True,
        # ``use_simple_mode=True`` skips the per-candidate conditional-MI redundancy check: fast, but selects redundant near-duplicate columns (e.g. both ``x`` and
        # ``2*x``, or a raw column AND an engineered feature that subsumes it). Default flipped True->False. Conditional-MI (Fleuret) redundancy IS the
        # point of MRMR; with it ON the selector returns a COMPACT, deduplicated set and -- once FE is in the loop -- prefers the engineered combination over its
        # redundant raw parents. VERIFIED on an additive target ``y = sign(x0+x1+x2+noise)``: full mode returns ``{x1, add(x0,x2), ...}`` (the engineered sum captures
        # the additive signal) at downstream LogReg AUC 0.992 == the all-raw baseline, with FEWER features. (An earlier read of this as "drops signal" was a metric
        # artifact: the raw-index ``signal_overlap`` test does not credit engineered features.) Costs ~2x wall-time vs simple mode; set ``use_simple_mode=True`` to opt
        # back into the faster dedup-free path on very wide feature sets.
        use_simple_mode: bool = False,
        run_additional_rfecv_minutes: bool = False,
        # Selection rule for the additional-RFECV rescue pass (run_additional_rfecv_minutes>0). The discarded pool is mostly noise plus a few
        # synergy-only features (interaction operands with ~zero marginal relevance). RFECV's recall-oriented default rule ('one_se_max') keeps
        # the LARGEST subset within 1 SE, which on noise-robust boosters re-admits ~the entire discarded pool -- undoing MRMR's parsimony and
        # re-injecting noise. 'one_se_min' keeps the SMALLEST subset within 1 SE, so the rescue re-adds only features that genuinely lift CV.
        additional_rfecv_selection_rule: str = "one_se_min",
        # Extra kwargs merged into (and overriding) the rescue RFECV's params, e.g. {"max_refits": 30, "n_features_selection_rule": "argmax"}.
        additional_rfecv_kwargs: dict | None = None,
        # performance
        extra_x_shuffling: bool = True,
        dtype: type = np.int32,
        # DEPRECATED alias for ``random_state`` -- kept for backward compatibility only;
        # prefer ``random_state``. ``None`` (legacy default) triggers process-stable but seedable
        # random_state derivation downstream (see ``_resolve_target_prefix``: uses pid ^ id(self)
        # instead of touching the numpy global RNG). For bit-exact reproducibility across runs / mlflow
        # hash stability, pass an explicit integer seed via ``random_state``.
        random_seed: int | None = None,
        use_gpu: bool = False,
        # Candidate-MI evaluation parallelism for the screen_predictors greedy loop (joblib
        # backend="threading" pool over evaluate_candidates / the Fleuret permutation-confirmation
        # step). Independent of ``n_jobs`` below (which drives CPU sub-helpers -- permutation-null MI,
        # wide-frame nbins edges -- each self-gated separately; the pair-search FE stage forces serial
        # under GPU-strict regardless of either knob). No sklearn-familiar ``-1``-style auto-resolve
        # (unlike ``n_jobs``): default 1 = SERIAL. See ``n_jobs``'s own docstring for the split
        # rationale.
        n_workers: int = 1,
        # confidence
        min_occupancy: int | None = None,
        min_nonzero_confidence: float = 0.99,
        full_npermutations: int = 3,
        baseline_npermutations: int = 2,
        # sample-size-aware Fleuret confirmation. With the
        # default ``use_simple_mode=False`` (full Fleuret conditional-MI
        # redundancy) the conditional permutation-confidence gate OVER-REJECTS
        # on small-n / high-cardinality data: the (X, Y, Z) conditioning joint
        # is severely undersampled (e.g. sklearn diabetes n=442, s5 10-bin ->
        # ~0.4 rows/cell), so the shuffled-y NULL conditional MI is ~= the REAL
        # conditional MI and every genuine feature after the first is rejected
        # -> premature stop (diabetes: 1 feature / R2=0.20 vs 9-feat simple-mode
        # R2=0.39). When the conditioning joint has fewer than this many rows
        # per OCCUPIED cell the conditional test is unreliable, so the
        # confirmation falls back to a MARGINAL-MI permutation test (the
        # X-marginal joint, ~|X| cells, is far better sampled). Dedup is
        # preserved by the relevance-minus-redundancy gain term (independent of
        # this gate); pure noise is still rejected (its marginal permutation
        # test rejects it too). Set to 0.0 to always use the strict conditional
        # test (legacy behaviour). Default 5.0 (ON), tuned on diabetes: at 5
        # rows/cell the plug-in CMI bias dominates the estimate.
        fe_confirm_undersample_rows_per_cell: float = 5.0,  # [ACCURACY-CAVEAT] 0.0 (legacy strict) under-selects small-n; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        # stopping conditions
        # min_relevance_gain: absolute MI floor. In ``min_relevance_gain_mode='absolute'`` the screening stops when marginal gain < this value verbatim; in the default ``'relative_to_entropy'`` mode this value is IGNORED and the effective absolute floor is ``min_relevance_gain_frac * H(y)``. The absolute mode is dataset-blind -- 0.0001 is enormous on a low-entropy target (99/1 binary, H(y) ~= 0.056) and tiny on a high-entropy one (uniform 10-class, H(y) ~= 2.30), so the default switched to the relative formulation.
        min_relevance_gain: float = 0.0001,
        # Fraction of H(y) used as the effective absolute floor when ``min_relevance_gain_mode='relative_to_entropy'``. 0.001 = 0.1% of the target entropy.
        min_relevance_gain_frac: float = 0.001,
        # Resolution mode for ``min_relevance_gain``. ``'relative_to_entropy'`` scales the floor with H(y) so noisy features cannot pile up on low-entropy targets; ``'absolute'`` honours ``min_relevance_gain`` verbatim (legacy behaviour).
        min_relevance_gain_mode: str = "relative_to_entropy",
        # diminishing-returns gate. Stops greedy selection once the
        # current candidate's gain drops below this fraction of the FIRST
        # selected feature's gain. Catches "trailing noise" leakage on
        # imbalanced y / large n where tiny-but-statistically-positive gains
        # clear the absolute floor (Layer 13 finding: noise gain 0.0004 vs
        # signal gain 0.0176 - both cleared min_relevance_gain_frac * H(y) at
        # 1% imbalance, but noise is only 2.5% of signal). 0.0 disables; 0.05
        # = stop once gain falls below 5% of first gain. Applies from the
        # SECOND selected feature onward (the first feature is the anchor).
        min_relevance_gain_relative_to_first: float = 0.05,
        # Miller-Madow MI bias correction at the selection gate.
        # Plug-in MI overestimates by ~(|X|-1)*(|Y|-1)/(2n) for high-card
        # X (Miller 1955, Paninski 2003). On 1200-level user_id at n=2500
        # with binary y the bias is ~0.24 nats - enough to make pure noise
        # outrank real numeric signal (Layer 10 seed=101 hijack). True =
        # subtract MM bias from gains at the floor comparison; does NOT
        # mutate the raw mrmr_gains_ attr (downstream sees raw plug-in MI).
        cardinality_bias_correction: bool = True,
        max_consec_unconfirmed: int = 10,
        max_runtime_mins: float | None = None,
        interactions_min_order: int = 1,
        interactions_max_order: int = 1,
        interactions_order_reversed: bool = False,
        max_veteranes_interactions_order: int = 1,
        only_unknown_interactions: bool = False,
        # FAST-SEARCH MASTER TOGGLE (2026-06-14, default ON). Trades the EXHAUSTIVE FE search tail
        # for a substantially faster fit that still recovers the signal. When True AND the user has
        # NOT explicitly overridden a sub-knob (the value still equals its package default), fit()
        # temporarily applies a fast profile and restores every knob in ``finally`` (so clone /
        # pickle / repeated-fit constructor-arg semantics stay stable, exactly like ``fe_auto``):
        #   * fe_max_steps 2 -> 1: skip the second augmented-pool pass whose ONLY product is FUSING
        #     two already-found half-composites into one column (cosmetic for a linear/tree model --
        #     the two separate halves give the identical fit). The dominant single lever.
        #   * fe_pair_prewarp_enable True -> False: drop the per-operand learned 1-D pre-warp sweep.
        #   * fe_stability_vote_enable / fe_escalation_underdelivery_enable True -> False: skip the
        #     post-discovery cross-fold confirmation + under-delivery escalation passes.
        #   * fe_check_pairs_subsample_n -> kernel_tuning_cache-resolved screen-n (HW/size aware; the
        #     MI/CMI screen is rank-stable under subsampling, so survivor identities are preserved and
        #     the FINAL survivor columns are still replayed at FULL n). Falls back to a safe default
        #     when no cached tuning exists.
        # On the two canonical n=100k interaction synthetics (y=a**2/b + log(c)*sin(d) and its warped
        # variant) this lands each fit < 60s (from ~130s / ~100s warm) with both interactions recovered
        # and Ridge-holdout MAE within tolerance. BUT it is SELECTION-ALTERING and trades search
        # exhaustiveness for speed: dropping the step-2 fusion + stability-vote + escalation passes lets
        # EXTRA over-materialized columns through (spurious cross-group gate_mask / cross-signal / rint
        # composites alongside the genuine div(sqr(a),neg(b)) + mul(log(c),sin(d))). The exhaustive search
        # (default) instead returns the clean FUSED single composite. DEFAULT FALSE: the
        # exhaustive search's clean, minimal selection is the right default; opt IN to ``fe_fast_search=
        # True`` when fit speed matters more than a tidy support set. (The fast path's over-materialization
        # is a known gap -- the junk-pruning passes that run under the exhaustive search do not yet all
        # run under fe_max_steps=1; until they do, fast trades cleanliness for speed.)
        fe_fast_search: bool = False,
        # feature engineering settings
        # MULTI-STEP FE DEFAULT 1 -> 2 (2026-06-10, user request). At step k>1 the operand
        # pool also carries the engineered columns selected by the prior step (capped by
        # ``fe_max_engineered_operands``), so the pair search builds COMPOSITES of two
        # engineered features -- e.g. the additive ``add(div(sqr(a),neg(b)),mul(log(c),sin(d)))``
        # that captures ~the entire deterministic signal of ``y = a**2/b + log(c)*sin(d)`` in a
        # SINGLE feature, where step-1 (fe_max_steps=1) recovers only the two separate halves.
        # An n-dependence probe (n=30k/50k/100k, multiple seeds) confirmed the clean composite
        # is recovered at every n with no spurious cross-mix, and the strict pin test
        # (test_mrmr_fe_composite_feedforward.py) guards it. Cost: one extra augmented-pool FE
        # pass; ``fe_max_engineered_operands`` (default 8) bounds the O(k^2) pair blow-up.
        # Set ``fe_max_steps=1`` to restore the single-step behaviour (no composites).
        fe_max_steps=2,
        # after the FE step appends engineered columns, run ONE more
        # screening pass over the AUGMENTED pool (raw + engineered) so the
        # engineered columns -- which are already quantised bin-code columns --
        # are selected by the SAME greedy relevance-minus-redundancy machinery as
        # raw features, rather than promoted into the result by fiat. This (a)
        # drops engineered features that are redundant given an already-selected
        # one (e.g. 1/b-d^2 whose conditional MI given a^2/b is ~0.03), and
        # (b) records a real mrmr_gain / support_rank for every engineered column
        # in fe_provenance_. Default ON; set False to keep the legacy
        # promote-by-fiat behaviour (engineered cols appended unconditionally,
        # no redundancy check, no gain). The re-screen does not run FE again, so
        # there is no unbounded recursion and no extra engineered columns appear.
        fe_reselect_after_engineering: bool = True,
        # Raw-retention sample-size scope (2026-06-08 regression fix). The post-FE
        # raw-retention pass re-adds a screening-confirmed raw feature the re-selection
        # dropped, to recover a genuine weak raw signal an engineered feature absorbed as
        # a redundant near-duplicate. That override is a SMALL-N device (validated on
        # n=500/2000/3000 fixtures where the conditional-MI redundancy estimate is noisy).
        # At large n the re-selection's conditional-MI redundancy term is reliable, so a raw
        # operand it drops in favour of a surviving MULTI-parent engineered child (e.g. raw
        # ``a,c,d`` vs ``div(sqr(a),abs(b))`` / ``mul(log(c),sin(d))`` for
        # ``y=a**2/b+log(c)*sin(d)``) is genuinely redundant and must STAY dropped -- the
        # blanket re-add padded support_ with redundant raw columns (support_rank -1, no gain)
        # and regressed the canonical selection. Above this row count, raw-retention defers to
        # the re-selection for raw columns that ARE operands of a surviving engineered feature;
        # raws absorbed by an UNRELATED engineered feature keep the protective re-add at any n.
        fe_raw_retention_max_n: int = 20000,
        # FE rejection-ledger record cap. ``fe_rejection_ledger_`` records every FE candidate
        # rejected during the fit for post-hoc diagnosis (which gate rejected it, at what margin). Capped
        # to bound memory on pathological wide-p fits; raise on very-wide-p (hundreds of thousands of
        # columns) fits if full-ledger post-hoc diagnosis matters more than the extra ~200-400 bytes/record.
        # None -> module default (currently 500_000; see ``_fe_rejection_ledger.FE_REJECTION_LEDGER_CAP``).
        fe_rejection_ledger_cap: Optional[int] = None,
        # RAW-VS-ENGINEERED CONDITIONAL-REDUNDANCY DROP. After all
        # retention / augmentation passes, prune any selected RAW operand that is
        # conditionally redundant given a surviving engineered feature built from it
        # (e.g. raw ``a, b`` beside ``div(neg(a),sqrt(b))`` for ``y=(a**2)/b``, which the
        # ratio fully determines). Uses the debiased excess-CMI test (the S5 idea applied
        # to raw-vs-engineered) so the verdict is n-INVARIANT and never drops a raw that
        # carries genuine independent signal. ON by default; set False to restore the
        # pre-fix behaviour (the small-n protective retention re-adds subsumed operands).
        fe_drop_redundant_raw_operands: bool = True,
        # Raw-vs-engineered redundancy POLICY. "drop" (default): minimal-set behaviour -- prune raw
        # operands a surviving engineered feature subsumes (the I4b invariant; right for tree downstreams
        # and minimal-redundancy selection). "emit_both": ALSO keep the SIGNAL-bearing raw operands of a
        # selected engineered feature (a linear downstream needs the raw even when a nonlinear child
        # info-subsumes it; noise operands are gated out by a marginal-significance test so FS still
        # rejects noise). The keep-vs-drop verdict is model-class-dependent and statistically
        # indistinguishable per-raw, so it is a caller policy, not an inferred discriminator.
        redundancy_policy: str = "drop",
        # TAU for the redundancy drop: a raw operand must retain >= this scale-free
        # fraction of the weakest consuming engineered survivor's own debiased excess CMI
        # to be judged a genuine independent term (else it is dropped as redundant). 0.15
        # mirrors the S5 engineered-vs-engineered gate; validated across n=1000..50000.
        fe_raw_redundancy_retain_frac: float = 0.15,
        # CROSS-FOLD RECIPE STABILITY VOTING. After the
        # expensive FE search has selected its survivors on the FULL data, REPLAY each
        # surviving numeric-pair (``unary_binary``) recipe -- leak-safe, the recipe is
        # frozen, only the rows change -- on K held-out folds and recompute its uplift
        # gate per fold; admit the recipe only if it clears the gate in >= ceil(quorum*K)
        # folds. A near-FREE consensus layer OVER the existing gates (no refit -- only K
        # plug-in-MI replays per recipe): it kills recipes that won on a fold-specific
        # quirk of the full-data split, complementing the order-2/order-3 maxT floors
        # (which kill chance-MAX candidates WITHIN a fold). ON by default -- it cuts
        # fold-specific NOISE survivors with no measured loss of genuine signal recovery
        # at negligible cost. Self-gates to a no-op below 2 unary_binary survivors / k<2 /
        # tiny n. Set False to byte-reproduce the pre-vote support.
        fe_stability_vote_enable: bool = True,
        # Number of held-out folds for the stability vote (>= 2; below 2 the vote is a
        # structural no-op). 5 mirrors the backlog spec -- enough folds that a genuine
        # recipe clears the quorum comfortably while a single-fold-quirk winner fails.
        # Accepts ``"auto"`` (hardcoded-threshold bench follow-up, 2026-06-13): a GUARDED
        # n-floored K that equals 5 for n >= 500 and only drops to 2-4 for genuinely tiny n
        # that cannot keep ~100 rows/fold across 5 folds. An explicit int (incl. the default
        # 5) is honoured verbatim -- byte-identical to the pre-2026-06-13 behaviour. The bench
        # showed LOWERING K degrades the vote (k=3 lost the F2 feature), so "auto" never raises
        # the noise by going below 5 on data that can sustain it.
        fe_stability_vote_k: "int | str" = 5,
        # Fraction of folds a recipe must clear (pass bar = ceil(quorum*K) folds). 0.6 ->
        # "in at least 3 of 5 folds" (the Meinshausen-Buhlmann-style support threshold).
        # Higher = stricter (drops more as fold-specific); 0 disables the vote.
        fe_stability_vote_quorum: float = 0.6,
        # SUCCESSIVE-HALVING / RUNG-SCHEDULE FE-search budget.
        # ON by default. Routes the expensive per-pair operator search
        # (``check_prospective_fe_pairs`` -- all unary x binary transforms / CMA-ES /
        # full discretize / prewarp, ~4-50s per pair) via a CHEAP rung-0 SCREEN: rank the
        # gate-surviving prospective pairs by their JOINT MI ``pair_mi`` (a monotone-ish
        # proxy of the operator-search outcome that the pair-MI gate ALREADY computed, so
        # the screen is FREE) and run the expensive search only on the top fraction. GATES
        # UNCHANGED -- this changes WHERE the compute goes, not admission, and generalises
        # the existing ``fe_synergy_max_pairs`` per-pair budget to the whole pool. Measured
        # 1.7-2.2x at keep_frac=0.5 / up to 11x at keep_frac=0.25 (n=5000, p=40, canonical
        # fixture + noise) with NO genuine signal pair dropped across 5 seeds (the relative
        # pair_mi floor below protects a moderate-MI genuine winner from the fractional
        # cut). Self-gates to a no-op below ``fe_rung_min_pairs`` pairs / all-zero pair_mi.
        # Set False to byte-reproduce the flat top-K sweep.
        fe_rung_schedule_enable: bool = True,
        # Rung-0 keep fraction. None (default) routes per (n_rows, n_pairs) through the
        # per-host ``kernel_tuning_cache`` (the iron rule -- never hardcode one threshold
        # across hardware / data shapes), with a measurement-backed fallback (0.34 large
        # pool / 0.50 moderate / 1.0 small). A float in (0, 1] forces that fraction. The
        # relative pair_mi floor below is applied REGARDLESS of this fraction, so a small
        # fraction never drops a genuine moderate-MI winner.
        fe_rung_keep_frac: float | None = None,
        # ALWAYS keep a prospective pair whose ``pair_mi >= fe_rung_rel_floor * max_pair_mi``
        # regardless of its rank, so a moderate-MI genuine winner survives the fractional
        # rung. 0.40 was the binding no-drop value in the benchmark (protects a pair at
        # 0.45*max while still cutting (c,noise) spurious survivors at 0.17*max).
        fe_rung_rel_floor: float = 0.40,
        # Below this many prospective pairs the rung screen is a structural no-op
        # (byte-identical flat sweep): a handful of pairs is already cheap to search fully.
        fe_rung_min_pairs: int = 6,
        # SUFFICIENT-SUMMARY EARLY-STOP -- DEFAULT-ON. The user's
        # "compare-to-theoretical-max" idea, realised cheaply via a Data-Processing-Inequality
        # (DPI) residual test. After each MRMR feature SELECTION (once per fit/screen pass, NOT
        # per candidate pair), fit a CHEAP ridge of y on the SMALL selected set
        # ``E_hat[y|selected]`` (1-5 cols -- engineered features linearise the signal so a linear
        # fit captures E[y|selected]; the design is tiny because the SELECTED set is small) and
        # form the residual ``r = y - E_hat``. If ``MI(r; x_j) <= the maxT permutation null`` for
        # EVERY raw feature AND the residual is small relative to y (``Var(r)/Var(y) <=
        # fe_sufficient_summary_residual_frac``, the H(y)-relative size guard), STOP the FE search:
        # by the DPI any future engineered candidate is a function of the raws, so it cannot have
        # more MI with r than the raws do -> the selection has reached I(observables; y), the
        # theoretical max, and the remaining search is provably pointless. The final selection is
        # UNCHANGED (verified byte-identical with early-stop on vs off on genuine multi-signal
        # fixtures) -- this only skips work that could find nothing. CONSERVATIVE: fires only when
        # BOTH the variance guard AND the all-raws maxT test pass, so it never stops while a
        # genuine second signal (incl. a NONLINEAR leftover the linear E_hat underfits, caught by
        # MI(r; raw)) is still discoverable. Reuses the SHIPPED MI kernels + maxT permutation null
        # (``pooled_permutation_null_gain_floor``). Set False to disable. See
        # ``_fe_sufficient_summary.py``.
        fe_sufficient_summary_early_stop: bool = True,
        fe_sufficient_summary_residual_frac: float = 0.25,
        fe_sufficient_summary_maxt_permutations: int = 25,
        fe_sufficient_summary_maxt_quantile: float = 0.95,
        fe_sufficient_summary_ridge_alpha: float = 1e-3,
        # AUTO-ESCALATION to the richer SHIPPED bases (2026-06-10, backlog idea B) --
        # DEFAULT-ON. When a prospective pair PASSED the pair-MI prescreen (joint-MI
        # ratio gate + order-2 maxT floor) but the unary/binary operator search admitted
        # NOTHING for it, the legacy behaviour was only the "FE produced 0 engineered
        # features despite N pair(s) passing the pair-MI gate" WARNING -- detected
        # signal, silently abandoned. With this ON the FE step ESCALATES those pairs to
        # the two richer shipped basis families and lets the EXISTING gates decide
        # ("escalation proposes, gates decide"): (1) the signal-adaptive ORTH-POLY pair
        # warp -- the rank-1 ALS per-operand fit re-run at ``fe_escalation_poly_degree``
        # across all four shipped bases (chebyshev/hermite/legendre/laguerre), best
        # basis by held-out reconstruction |corr|; (2) DEMODULATED adaptive-frequency
        # FOURIER/CHIRP warps -- for a multiplicative pair ``y ~ g(a)*b`` the shipped
        # held-out multitone detector is run on ``(z01(a), t = y_c * zscore(b))``
        # (E[t|a] ~ g(a)), locking an INNER frequency (``sin(3.7*a)*b``) no library
        # unary can express; the chirp axis covers growing-frequency inners. Candidates
        # must clear the order-2 maxT floor on the Miller-Madow-debiased MI scale, a
        # marginal-permutation floor, and the S5 conditional-MI redundancy gate vs the
        # admitted engineered support -- a pure-noise pair that slipped the prescreen
        # admits NOTHING (measured 0/N on noise controls). Structurally a no-op when
        # every surviving pair already produced an admitted column (the common case).
        # Recipes are standard ``unary_binary`` + ``prewarp`` specs (the Fourier mix is
        # the closed-form ``fourier_adaptive`` spec), so transform()/pickle/stability-
        # vote treat escalated features exactly like default-prewarp pair features.
        # See ``_fe_auto_escalation.py`` + tests/feature_selection/test_fe_auto_escalation.py.
        fe_auto_escalation_enable: bool = True,
        # Escalate at most this many prescreen-surviving zero-admission pairs per FE
        # step (strongest joint MI first) to bound the proposer cost.
        fe_escalation_max_pairs: int = 8,
        # Below this many rows escalation is skipped entirely: the S5 gate falls back
        # to marginal admission under ~500 rows and the adaptive-frequency detector is
        # n-gated at 800 inside, so small-n escalation would lose its hard noise gates.
        fe_escalation_min_rows: int = 500,
        # Held-out floor shared by the proposers: the ALS rank-1 reconstruction must
        # track y on the stride-validation slice with |corr| >= this, and the Fourier
        # detector's held-out periodogram floor is max(this, 0.30) (the shipped robust
        # anti-chance-peak floor).
        fe_escalation_min_val_corr: float = 0.15,
        # ALS warp degree for the escalated orth-poly proposer. Higher than the default
        # prewarp's 4 on purpose: escalation only runs where that default failed.
        fe_escalation_poly_degree: int = 6,
        # Max validated frequencies per demodulated Fourier/chirp warp (multitone cap).
        fe_escalation_fourier_max_freqs: int = 3,
        # Keep at most this many candidates per escalated pair (by debiased MI) before
        # the floors + S5 gate, so the gate pool stays small.
        fe_escalation_max_candidates_per_pair: int = 3,
        # PAIR-NESS margin for the escalated orth-poly proposer: the rank-1 PAIR
        # reconstruction's held-out |corr| must beat the best SINGLE-operand 1-D warp's
        # held-out |corr| by this factor, else the "pair" is a wrapped univariate trend
        # (a genuine-marginal x noise cross-mix the ALS collapses to ~constant on the
        # noise side) and is not proposed -- the univariate stages own that signal.
        # 1.15 mirrors fe_synergy_min_prevalence; genuine product terms measure >= 1.5.
        fe_escalation_pairness_margin: float = 1.15,
        # UNDERDELIVERY trigger: also escalate a pair whose unary/binary
        # search DID admit a column when the best admitted capture leaves SIGNIFICANT
        # conditional pair MI on the table -- leftover CMI(joint(a,b) codes; y | best
        # admitted column's codes) above its conditional-permutation null floor AND a
        # debiased excess >= ``fe_escalation_underdelivery_excess_frac`` of the captured
        # MI. Catches the ``y=sin(3.7a)*b`` envelope capture ``mul(sin(a),qubed(b))``
        # the marginal-uplift fallback admits while most of the detected signal stays
        # unexpressed (an MI-ratio bar vs the prescreen pair_mi CANNOT separate that
        # case: the junk capture measures ratio 1.20, above genuine captures' own
        # scale, because the 2-D prescreen joint MI under-estimates pair information).
        # A false trigger is safe (escalation only PROPOSES; the full gates -- incl.
        # the S5 CMI gate conditioned on the pair's own admitted column -- decide), so
        # the trigger runs cheap: stride-subsampled rows + an 8-permutation null.
        # Escalated proposers for such pairs fit the BINNED-MEAN RESIDUAL of the
        # target given the existing capture (see ``run_fe_auto_escalation``), so only
        # genuinely-missing signal can be proposed -- a remap of the existing capture
        # finds ~no residual correlation and dies at the held-out floors.
        # ``_self_ratio`` is the DISCRETISATION-RESIDUAL control: even a functionally
        # COMPLETE capture leaves leftover CMI in the 2-D joint (its nbins quantile
        # code is coarse), so the joint's leftover must also exceed this multiple of
        # the capture's OWN finer-binning refinement CMI(capture@2*nbins; y |
        # capture@nbins). Measured: complete captures 0.70-2.44, the sin-fixture
        # envelope junk capture 14.6 -- 3.0 separates with margin on both sides.
        fe_escalation_underdelivery_enable: bool = True,
        fe_escalation_underdelivery_excess_frac: float = 0.05,
        fe_escalation_underdelivery_self_ratio: float = 3.0,
        # PREVALENCE-FAILED SYNERGY RESCUE (2026-06-12, F2 a**2/b miss, default-ON). A
        # synergy pair (>=1 bootstrap-added unselected operand) whose JOINT MI cleared the
        # order-2 maxT permutation floor but missed the stricter ``fe_synergy_min_prevalence``
        # RAW-MI ratio bar is handed to the auto-escalation as a failed pair. The raw-MI
        # prevalence ratio structurally UNDER-estimates a smooth, non-bilinear ratio
        # interaction: the user's genuine ``a**2/b`` term scores ratio ~1.11 (joint MI 0.028
        # vs marginal-sum 0.025) -- far below the 1.5 synergy bar -- so it was dropped before
        # ANY FE/escalation ran, and the output carried no (a,b) feature at all (F2 downstream
        # R^2 capped at 0.947 vs the 0.997 the feature reaches). LOWERING the raw-MI bar is
        # bench-rejected (injects optimisation-inflated noise products); instead the rescue
        # re-tests each pair with the LEAK-SAFE held-out rank-1 ALS pair-vs-single |corr|
        # margin inside ``_propose_poly`` (the genuine (a,b) scores pair/single ratio 1.24 vs
        # cross-mix/noise ~1.0), and the proposed candidate then faces the FULL admission
        # gates (order-2 maxT on MM-debiased MI + marginal-permutation floor + S5 CMI
        # redundancy). Only synergy pairs that cleared the maxT null are eligible (a
        # pure-chance noise pair never reaches escalation); a false rescue only PROPOSES.
        # Set False to restore the pre-fix behaviour (synergy prevalence is a hard drop).
        fe_synergy_prevalence_rescue_enable: bool = True,
        # ESCALATION FEATURES ARE TERMINAL in the composite feed-forward (2026-06-12, F2
        # rescue). An ``esc_*`` orth-poly / adaptive-Fourier escalation feature already
        # captures a genuine richer-basis interaction; feeding it back as an operand for a
        # FURTHER pair composite fuses two independent additive target terms into one ratio
        # whose joint MI tops the greedy ranking, so MRMR then drops the clean raw
        # predictors -- measured on F2: the standalone esc_poly(a,b) is a +0.05 downstream
        # R^2 win, but the fed-forward nested ``div(log(esc_poly(a,b)),exp(...))`` regresses
        # it. Default OFF (escalation features stay selected but never seed composites); set
        # True to restore the unrestricted feed-forward.
        fe_escalation_feedforward_enable: bool = False,
        # ``fe_npermutations`` default 0->3:
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
        # ENGINEERED-OPERAND FEED-FORWARD CAP. At FE step k>1 the
        # operand pool also carries the engineered columns selected by the prior
        # step(s), so the pair search can build COMPOSITES of two engineered
        # features -- e.g. the additive ``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))``
        # that captures ~the entire deterministic signal of
        # ``y = a**2/b + log(c)*sin(d)``. To bound the O(k^2) pair blow-up (engineered
        # cols accumulate across steps), only the top-K engineered operands BY THEIR
        # SCREENING MI WITH THE TARGET are fed back each step; the rest still reach
        # ``support_`` as selected features, they just don't seed further composites.
        # 8 covers the realistic "a handful of strong engineered factors recombine"
        # case while keeping the extra pair count small. 0 disables the feed-forward
        # (raw-only operands, the pre-2026-06-08 behaviour); a negative value means
        # "no cap" (feed back every selected engineered operand).
        fe_max_engineered_operands: int = 8,
        fe_unary_preset="medium",
        fe_binary_preset="minimal",
        # ``fe_max_pair_features`` default 1->10:
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
        # ``fe_min_nonzero_confidence`` default 1.0->0.99:
        # pre-fix 1.0 required EVERY permutation to clear the
        # null-hypothesis test exactly, making the gate unreachable at any
        # noise level. 0.99 matches the screening-side
        # ``min_nonzero_confidence`` default so both stages apply equally
        # strict statistical rigor without the unreachable-gate trap.
        fe_min_nonzero_confidence: float = 0.99,
        # This floor only gates the LEGACY per-pair ``mi_direct`` permutation-test path inside
        # ``compute_pairs_mis`` (feature_engineering.py). Since the finding-#21 fix removed
        # ``_MRMR_BATCH_PRECOMPUTE_MAX_K``, ``dispatch_batch_pair_mi_chunked`` unconditionally
        # pre-fills ``cached_MIs`` for every C(k,2) pair at n_pairs>=8 via a fast batched kernel BEFORE
        # this legacy sweep runs, and ``compute_pairs_mis`` skips its expensive ``mi_direct`` call
        # whenever a pair is already cached -- so this floor being weak no longer bounds real compute
        # cost for the common case (see ``test_pair_mi_legacy_sweep_cache_starved.py``). It still
        # matters for the n_pairs<8 / batch-precompute-failure fallback.
        fe_min_pair_mi: float = 0.001,
        # mi of entire pair must be at least that higher than the mi of its individual factors, to consider the pair at all.
        # Accepts ``"auto"`` (hardcoded-threshold conversion, 2026-06-13): a GUARDED data-derived mode
        # that keeps the 1.05 ratio bar but applies it to the MILLER-MADOW-DEBIASED pair MI (the
        # analytic finite-sample joint-MI bias subtracted), so the gate can only TIGHTEN -- it drops
        # the best-of-pool finite-sample-noise pairs a fixed 1.05 admits (bench: bilinear archetype
        # 0.207 -> 0.092) while a genuine high-signal pair (joint MI >> bias) is untouched. The bias
        # is analytic (no extra shuffles). An explicit float (incl. the default 1.05) is honoured
        # verbatim -> byte-identical to the pre-conversion behaviour.
        fe_min_pair_mi_prevalence: "float | str" = 1.05,
        # default-flip 0.98 -> 0.90: a 1-D engineered column summarising
        # a 2-D pair-joint structurally cannot retain 98% of the (finite-sample-bias-
        # inflated) 2-D joint MI. On the canonical fixture y=a**2/b + f/5 + log(c)*sin(d)
        # the genuine features mul(sqr(a),reciproc(b)) [rat~0.95] and
        # mul(log(c),sin(d)) [rat~1.01] cleared 0.90 but not 0.98, so the default fit
        # found ZERO engineered cols. 0.90 keeps genuine 1-D summaries of real 2-D
        # interactions while still rejecting noise (which lands well below the pair MI).
        # 2026-06-08 regression fix: restored to 0.90 (the documented value above). A
        # campaign tightening to 0.97 (commit 855c2568, to cut optimisation-inflated
        # noise-FE) rejected GENUINE engineered features on weaker-signal canonical targets
        # (e.g. ``y=0.2*a**2/b + log(c*2)*sin(d/3)`` produced 0 engineered, with the runtime
        # warning literally suggesting a 0.90 retry), contradicting the rationale comment
        # just above which still described 0.90. The noise-FE the 0.97 raise targeted is now
        # rejected HW-robustly by the two-tier marginal-uplift joint-recovery gate (see
        # ``_FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO`` in _feature_engineering_pairs), so the
        # joint-prevalence floor no longer has to carry that load and can return to the value
        # that admits genuine 1-D summaries of real 2-D interactions.
        fe_min_engineered_mi_prevalence: float = 0.90,  # mi of transformed pair must be at least that higher than the mi of the entire pair
        # MILLER-MADOW DEBIAS of the joint-prevalence ratio gate (2026-06-09, + #4).
        # The gate ``best_mi / pair_mi > fe_min_engineered_mi_prevalence`` compares a 1-D
        # engineered MI (over ~``quantization_nbins`` bins) against a 2-D joint MI (over
        # ~``nbins^2`` bins). Both are RAW plug-in MIs whose positive finite-sample bias is
        # ``(k_x-1)(k_y-1)/2n``; the JOINT denominator carries ~``nbins``x more bias, so the
        # raw ratio is structurally depressed below 1.0 at small/moderate n. When True the
        # Miller-Madow MI-bias term is subtracted from BOTH sides before the ratio (occupied
        # bin counts, #4) with a denominator-positivity guard, and the order-2 maxT floor is
        # MM-debiased CONSISTENTLY (IRON RULE, see ``_permutation_null`` / ``compute_pair_maxt_floor``).
        #
        # bench-rejected as a DEFAULT; kept OPT-IN (default False). The ISOLATED
        # ratio fix is real -- on the He2(a)*b fixture the raw ratio is 0.555 / 0.841 / 1.003
        # at n=500 / 2000 / 8000 and the MM(occupied-K) ratio is 1.99 / 1.15 / 1.10 (crosses
        # the 0.90 bar at small n where raw fails; n=8000 raw~=corrected => large-n untouched),
        # and the pure-NOISE frame stays below the bar both raw AND MM-corrected (0.27 / 0.28 /
        # 0.56 -- no isolated FP). BUT it does NOT translate to an END-TO-END win and it ADMITS
        # NOISE on the realistic weak target: (1) on the CLEAN He2(a)*b end-to-end the existing
        # marginal-uplift FALLBACK gate already recovers the genuine (a,b) pair 5/5 at n=500 and
        # n=2000 with MM OFF, so MM adds 0 incremental recovery; (2) on the user's WEAK F2
        # (``0.2*a**2/b + log(c)*sin(d)``, 10 seeds uniform n=20000) MM REGRESSES genuine_ab
        # 10/10 -> 8/10, does NOT recover genuine_cd (0/10 -> 0/10), and nearly TRIPLES cross-mix
        # admission 3/10 -> 9/10 seeds (n_cross 6 -> 11): the MM over-correction (ratio -> ~2.0)
        # makes the prevalence gate uniformly too permissive, so cross-signal artefacts like
        # ``sub(reciproc(b),sin(d))`` / ``add(reciproc(b),log(d))`` clear the relaxed ratio.
        # This is the IRON-RULE failure mode and matches the prior-session II-routing rejection
        # on the SAME target (the cross-mix has HIGHER interaction info than the genuine pair, so
        # NO ratio threshold separates them). The maxT-floor co-update is correct + green (6/6
        # order-2 biz tests, noise 100% at/below the MM floor) -- the reject is the RATIO gate
        # relaxation, not the floor. True enables the full mechanism for re-bench on other data.
        fe_mm_debias_prevalence: bool = False,
        # ENGINEERED-FEATURE ACCEPTANCE STRATEGY (strategy S5, 2026-06-08).
        # ``"conditional_mi"`` (default): the PRINCIPLED, constant-free gate. After the
        # per-pair search has chosen one best engineered column per pair, a greedy CMI-MRMR
        # runs over the surviving pool -- a candidate is admitted iff its CONDITIONAL MI with
        # y GIVEN the already-admitted ENGINEERED features clears BOTH a conditional-
        # permutation floor (significance) AND a scale-free fraction
        # (``fe_engineered_cmi_retain_frac``) of the weakest admitted feature's CMI (the
        # order-of-magnitude redundancy separator). This rejects a redundant engineered
        # column whose y-information is wholly carried by the admitted features -- WITHOUT a
        # hand-tuned per-dataset ratio constant -- while keeping every genuine column that
        # carries a private interaction term. The per-pair ``fe_min_engineered_mi_prevalence``
        # ratio still acts as the cheap upstream pre-screen; the CMI gate is the principled
        # FINAL redundancy filter. Validated 10/10 vs four failing approaches across 16
        # (seed, formula) cells.  ``"prevalence_ratio"``: legacy/compat -- skip the CMI gate
        # and let the per-pair ``fe_min_engineered_mi_prevalence`` ratio alone decide (the
        # exact pre-S5 behaviour), kept for fallback and byte-reproduction of old fits.
        fe_acceptance: str = "conditional_mi",
        # TAU for the ``conditional_mi`` acceptance: a candidate must RETAIN at least this
        # fraction of the WEAKEST already-admitted engineered feature's conditional MI.
        # SCALE-FREE -- it is a fraction of an in-data CMI quantity, NOT an MI-nats constant.
        # Measured robust window [0.084, 1.0) across 16 (seed, formula) cells (a redundant
        # feature never exceeds 8.3% of the weakest genuine one); 0.15 sits in the middle
        # with ~2x margin both sides. Larger = stricter (drops more as redundant).
        fe_engineered_cmi_retain_frac: float = 0.15,
        # Strong-significance escape margin for the ``conditional_mi`` acceptance. A candidate
        # whose observed CMI clears its OWN conditional-permutation floor by at least this
        # MULTIPLICATIVE margin is admitted even when its debiased excess is below the relative
        # (TAU) bar. Prevents the FALSE REJECT of a genuinely complementary but individually
        # WEAKER feature (independent of the admitted support, but adding an order-of-magnitude
        # less information than a strong incumbent): with one strong seed the TAU bar becomes a
        # large absolute threshold and would otherwise drop such a feature as "redundant".
        # A truly redundant feature's CMI collapses to ~its floor (cmi/floor ~1) so the escape
        # opens no false-ADMIT path. Set <= 1 to disable (pure two-leg behaviour). 3.0 sits with
        # ~2x margin on both sides of the measured redundant (<=1.4x) vs genuine (>=20x) gap.
        fe_engineered_cmi_significance_escape_margin: float = 3.0,
        # COST GUARD for the ``conditional_mi`` acceptance gate. The greedy gate is
        # O(K^2) in the engineered-candidate count K (each remaining candidate is
        # re-scored against the admitted support in EACH greedy round, and each
        # scoring runs a 25-permutation within-stratum conditional-permutation null).
        # On a wide FE candidate pool (the synergy bootstrap / GBM seeder can surface
        # dozens-to-hundreds of survivors) this blows up unbounded (~2.0x per doubling
        # of K). When the surviving pool exceeds this cap the gate PRE-RANKS by
        # marginal MI and keeps only the top-M before the greedy -- bounding the cost
        # to O(M^2). Safe for the redundancy decision: the gate already admits in
        # marginal-MI order and a redundant remap shares its genuine sibling's
        # marginal MI, so every genuine driver's representative is retained while only
        # the deep-tail redundant remaps (rejected anyway) are dropped pre-greedy.
        # ``<= 0`` disables the cap (unbounded greedy).
        fe_engineered_cmi_max_candidates: int = 64,
        fe_good_to_best_feature_mi_threshold: float = 0.98,  # when multiple good transformations exist for the same factors pair.
        fe_max_external_validation_factors: int = 0,  # how many other factors to validate against
        fe_max_polynoms: int = 0,
        fe_print_best_mis_only: bool = True,
        # default-flip evaluation result:
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
        # subsample inside the CMA-ES / Optuna inner search to
        # bound per-pair MI compute on production-size frames. cProfile on
        # n=500k showed ``_eval_coef_pair`` + ``_plugin_mi_classif_njit``
        # dominate (32% + 22% of fit time); each MI call scales linearly
        # with n. At n=4M with default config (10 restarts x 200 trials
        # x C(25,2)=300 pairs) per-pair cost projects to ~5 min, ~25
        # hours serial.
        #
        # Default 200_000 (raised from 100_000 after the
        # n=1M bench showed 100k could lose 1 hermite feature on
        # marginal seeds while 200k kept it). The FINAL injected column
        # is still computed from FULL source so no train-time precision
        # is lost.
        #
        # Set to ``None`` / 0 / negative to disable (use full data).
        # unified with check_prospective_fe_pairs via the shared
        # UNIFIED_FE_SUBSAMPLE_N constant; both FE entry points now scale their
        # MI-sweep buffer with the same knob. Re-tune in feature_engineering.py
        # to land both sites consistently.
        fe_smart_polynom_subsample_n: int = UNIFIED_FE_SUBSAMPLE_N,
        # Subsample rows for check_prospective_fe_pairs's
        # MI sweep. The hoisted shared scratch buffer scales linearly with n; on
        # n=4M with the medium preset it lands at ~17.6 GiB and crashes the suite.
        # Bench (bench_fe_pair_subsample_accuracy.py): jaccard=1.0 vs full-n at
        # 50k+, 0.88 at 5k. Default is the unified UNIFIED_FE_SUBSAMPLE_N (30k),
        # shared with fe_smart_polynom_subsample_n for cross-block consistency.
        # 0 = use full data (legacy). 30_000 is a VALIDATED value -- survivor
        # jaccard 1.0 / winner-match 5/5 vs the full-n screen (see
        # UNIFIED_FE_SUBSAMPLE_N in feature_engineering.py). 10_000 is the
        # marginal floor; do NOT set below 25_000.
        fe_check_pairs_subsample_n: int = UNIFIED_FE_SUBSAMPLE_N,
        # STRATIFIED FE SUBSAMPLE (R1, 2026-06-18). The FE MI-sweep / pure-form-retention /
        # polynom-pair subsamplers above draw rows with a PLAIN uniform ``rng.choice`` -- no
        # class balance for classification, no y-quantile coverage for regression. On a small
        # rare-class fraction (uniform can drop ALL rows of a 1% class) or a heavy-tailed
        # regression target (uniform under-represents the tails) the sampled MI / linear-usability
        # screen is computed on a distribution that has lost the very structure FE is meant to
        # recover. ``stratified_subsample_idx`` replaces the uniform draw with a per-class
        # proportional (>=1/class) draw for classification and a 10-quantile-bin proportional draw
        # for regression (preserves tails), falling back to uniform on degenerate y.
        #
        # Tri-state knob (DEFAULT True = always stratify):
        #   * True  (DEFAULT) -> ALWAYS stratified. Stratified draws preserve rare classes (>=1, often
        #     >=2, of each class) and the regression y-tails (proportional across 10 quantile bins),
        #     so the FE MI / linear-usability screen always sees the structure FE is meant to recover.
        #   * False -> always plain uniform: forces the byte-identical LEGACY draw at every n / class
        #     mix (use only for exact replay of pre-R1 runs).
        #   * None  -> AUTO (accepted but no longer the default): OFF on the common path, ON only when
        #     a uniform draw would plausibly lose target structure (small min-class fraction /
        #     heavy-tailed regression). See ``_resolve_fe_subsample_stratify``.
        # Default ON: the rare-class / heavy-tail protection always applies; only an explicit False
        # restores the legacy uniform path.
        fe_subsample_stratify: bool | None = True,
        # ``fe_min_polynom_degree`` default 3->1:
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
        # default-flip 8 -> 6: degree 8 inflated the joint coefficient
        # search to ~18 dims (9 per operand) for no measured recovery benefit on
        # the pre-distortion fixtures, while degree 6 (14 dims) recovers every
        # case the ALS warm-start can reach. Measured at n=4000, cma_batch,
        # 15 restarts x 300 steps: F-POLY |corr-to-true-signal| 0.97 (deg 4, 6,
        # and 8 identical -- a cubic*quadratic product is a degree-4 object),
        # F-OSC sin(a**2)*b 0.95 at deg 6 (deg 4 only reaches 0.005 -- the
        # oscillation needs the degree-5/6 Chebyshev terms), and a pure-noise
        # control engineers ZERO columns at every degree. Degree 6 is therefore
        # the smallest default that recovers BOTH the polynomial and oscillatory
        # pre-distortion regimes; raise to 8+ only for very high-frequency
        # targets (scale n_trials proportionally). See
        # tests/feature_selection/test_biz_value_mrmr_pre_distortion.py.
        fe_max_polynom_degree: int = 6,
        fe_min_polynom_coeff: float = -10.0,
        fe_max_polynom_coeff: float = 10.0,
        # explicit __init__ params for the fe-* knobs that the
        # polynom-pair FE inner search consults via getattr(self, ...).
        # Pre-fix these were accessible by setting them as attributes after
        # construction (``mrmr.fe_optimizer = 'cma_batch'``) but the
        # ``FeatureSelectionConfig.mrmr_kwargs`` validator rejected them as
        # unknown because they weren't in this signature. Adding here lets
        # users pass them through ``mrmr_kwargs={'fe_optimizer': ...}`` via
        # the suite config.
        #
        # ``fe_hermite_l2_penalty`` is the coefficient-magnitude regulariser
        # weight. The penalty SEMANTICS changed from the raw
        # ``lambda * ||c||^2`` (which grew without bound and crushed genuinely
        # high-MI / high-coefficient solutions -- e.g. the separable Chebyshev
        # reconstruction of a non-monotone pre-distortion product, whose true
        # coefficients have ||c||^2 ~ 86 so the raw penalty ~4.3 dwarfed the MI
        # peak ~1.5) to a SCALE-SATURATING form ``lambda * ||c||^2 / (||c||^2 +
        # saturation)`` that rises toward a constant ``lambda`` ceiling instead.
        # The value 0.05 is unchanged but now harmless to large-coefficient
        # solutions while still regularising pure noise (tiny ||c||^2 pays
        # ~full lambda). Set 0 to disable entirely; the saturation scale is
        # ``hermite_fe._L2_PENALTY_SATURATION_DEFAULT`` (1.0).
        fe_hermite_l2_penalty: float = 0.05,
        fe_polynomial_basis: str = "chebyshev",
        # PER-OPERAND PRE-WARP for the elementary unary/binary pair
        # search. Default OFF -> byte-identical legacy path. When True, BEFORE the
        # unary x unary x binary combination search the engine fits, per raw
        # operand, one learned 1-D orthogonal-polynomial warp ``f(x)`` of the
        # target (``hermite_fe.fit_operand_prewarp`` -- the shared 1-D sibling of
        # the orthogonal-poly path's ALS warm start) and adds it as an extra
        # ``prewarp`` pseudo-unary alongside ``identity/sqr/log/...``. This lets
        # the unary/binary path represent a WITHIN-OPERAND non-monotone distortion
        # (e.g. ``a**3 - 2a``) that no single library unary can express, so a
        # target ``F3(F1(a), F2(b))`` with a non-monotone inner ``F1`` becomes
        # recoverable through the cheap function search WITHOUT the full
        # orthogonal-poly CMA optimiser. The pre-warp is still gated by the
        # existing MI-prevalence / external-validation machinery (so it never
        # fabricates a feature on noise / linear data), and its fitted coeffs are
        # stored in the EngineeredRecipe for leak-safe, y-free replay at
        # transform() time. Orthogonal to ``fe_smart_polynom_iters`` /
        # ``fe_hybrid_orth_enable`` (works with both off).
        # default flipped False->True. Unlike the orthogonal-poly
        # path (``fe_smart_polynom_iters``, OFF by default because its CMA/Optuna
        # search is ~60s/fit), the pre-warp is a single rank-1 ALS least-squares
        # solve (~5ms/pair, cProfile-confirmed negligible) AND is uplift-gated +
        # noise/linear-clean, so it is a free accuracy win: ON by default per the
        # "accuracy-improving mechanisms default-on" policy. Set False to disable.
        fe_pair_prewarp_enable: bool = True,
        fe_pair_prewarp_basis: str = "chebyshev",
        fe_pair_prewarp_max_degree: int = 4,
        # PER-OPERAND MEDIAN GATE for the elementary unary/binary
        # pair search. Default OFF -> byte-identical legacy path. When True, the
        # unary/binary search gains, per raw operand, a ``gate_med`` pseudo-unary
        # ``(x > train_median_x).astype(float)`` alongside ``identity/sqr/log/...``.
        # Combined with the existing ``mul`` binary it expresses the median-gated
        # operators the bilinear product cannot: ``(a > median_a) * b`` (gated_med,
        # via ``mul(gate_med(a), b)``) and the conjunction
        # ``(a > median_a) & (b > median_b)`` (thr_and_med, via
        # ``mul(gate_med(a), gate_med(b))``). The median ADAPTS the split to each
        # operand's distribution, so it recovers the gate on shifted / skewed
        # operands where a fixed threshold-0 gate is useless (measured skew-bench:
        # gated_med +0.0355 / thr_and_med +0.0435 downstream-AUC d_mean vs raw,
        # beating products +0.022/+0.020 and threshold-0 +0.009/+0.0001). The
        # fitted state is ONE float per operand (the TRAIN median) -- it cannot
        # overfit, so (unlike the prewarp) no held-out validation is needed; the
        # gate still passes every existing MI-prevalence / external-validation
        # acceptance gate (it competes on equal footing in the per-pair MI sweep
        # and wins only where the conditional form beats the library). The median
        # is stored in the EngineeredRecipe for leak-safe, y-free closed-form
        # replay at transform() time. Orthogonal to every other FE knob.
        # Default OFF (opt-in): the rich-ops cost finding says do not tax the
        # default minimal path; opt-in is correct.
        fe_gate_med_enable: bool = False,
        # Minimum (best-prewarp-MI / best-nonprewarp-MI) ratio for the prewarp
        # alternative-acceptance path past the joint-MI-prevalence gate. The
        # prewarp feature must beat the elementary library by this factor to be
        # admitted -- directed + noise-safe (uplift ~1.0x on linear/noise data).
        fe_pair_prewarp_uplift_threshold: float = 1.20,
        fe_mi_estimator: str = "plugin",
        # cma_batch is the new default (20.58x faster than
        # optuna, within_1%=1.00 vs all other optimizers on a 12-pair bench).
        # See profiling/bench_polynom_optimizers.py.
        fe_optimizer: str = "cupy_kernel",
        fe_warm_start: bool = True,
        fe_multi_fidelity: bool = True,
        # verbosity and formatting
        verbose: bool | int = 0,
        ndigits: int = 5,
        parallel_kwargs: dict | None = None,
        # CV
        cv: int | BaseCrossValidator | Iterable | None = 3,
        cv_shuffle: bool = False,
        # service
        # Canonical seed parameter (sklearn's name). See ``random_seed`` above for the
        # deprecated alias and ``_effective_random_seed`` for the resolution order.
        random_state: int | None = None,
        # sklearn-familiar auto-resolving parallelism knob (``-1`` -> physical cpu_count, see
        # ``__init__``'s resolution below). Drives CPU sub-helpers ONLY (permutation-null MI batches,
        # wide-frame nbins edge computation, the FE pair-search joblib fan-out in _run_fe_step) -- it
        # does NOT parallelize screen_predictors' candidate-MI evaluation loop, which is gated
        # separately by ``n_workers`` above (default 1 = serial). Easy to misread ``n_jobs=-1`` as
        # "the whole fit runs on all cores"; it does not.
        n_jobs: int = -1,
        # Skip the full re-fit when a process-cache hit replays a prior fit on the SAME content. The cache invalidates on
        # (a) X content change, (b) y / TARGET content change, AND (c) ANY selector-parameter change (set_params or direct
        # attribute assignment alike; params are re-read at every fit call) -- so it never replays a stale fit for a changed
        # target or changed settings. Both layers honour this: the in-object identity skip and the process-wide _FIT_CACHE.
        skip_retraining_on_same_content: bool = True,
        # Cardinality cutoff for the confirmation step. ``None`` (default) computes
        # ``quantization_nbins ** interactions_max_order * 2`` (20 for the defaults). Pin to 50 for legacy behaviour.
        # Conservative default skips high-cardinality conditioning sets where permutation-based confirmation does not
        # converge in reasonable time.
        max_confirmation_cand_nbins: int | None = None,
        # When screening returns zero selected_vars, legacy code fell back to FE on ALL features; new default is
        # to skip FE (running FE on an empty screen typically just amplifies noise). Set True for legacy.
        fe_fallback_to_all: bool = False,
        # Pipeline-fatal fallback: when screening yields zero features (all MI ~= 0), the default
        # ``min_features_fallback=1`` keeps the single highest-MI column so ``support_`` is never empty -- empty
        # support causes downstream estimators to crash with a 0-column transform output. Set to 0 explicitly to
        # restore the legacy "let the pipeline fail loudly" semantics. Chosen features are flagged via
        # ``self.fallback_used_``.
        min_features_fallback: int = 1,  # [ACCURACY-CAVEAT] 0 removes the never-empty floor (support_ can be empty); see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        # SIS FRONT GATE (Gate A of the p=100k cascade -- see tests/feature_selection/MRMR_100K_SCALING_DESIGN.md
        # and filters/_mrmr_sis_screen.py). When the input width p reaches this threshold, a single chunked
        # O(p*n) screen (fused marginal-MI + 2nd-moment interaction propensity) cuts the pool to a few thousand
        # data-derived survivors BEFORE the super-linear MRMR/Fleuret/FE machinery runs. This is the
        # fastest-default DISPATCH knob, not a user opt-in: below the threshold today's path runs unchanged;
        # at/above it the gate makes wide frames feasible. Set to 0 / None to disable the gate entirely.
        sis_screen_threshold: int = 20000,
        # SIS survivor REDUNDANCY DEDUP (reuse audit RU-1): after the gate picks survivors, collapse near-
        # duplicate columns (|Pearson|>=this, blocked corr_clusters, keep the highest-fused rep) BEFORE the
        # O(k*p*n) Fleuret CMI loop. Selection-neutral (MRMR's CMI would reject the copies anyway); the win is
        # data-dependent (~1% on independent survivors, up to the redundant fraction on correlated families).
        # Set 0 to disable. Only genuine near-linear duplicates merge; pure-interaction operands never do.
        sis_dedup_corr_thr: float = 0.92,
        # Cat-FE (categorical feature interactions): single dataclass consolidating ~22 cat_fe_* knobs.
        # ``None`` = default CatFEConfig() with ``enable=True`` and conservative production settings (cat-FE
        # shows measurable wins; XOR biz_value test, 0 regressions). Restore legacy via CatFEConfig(enable=False).
        cat_fe_config=None,
        # Bound on the process-wide _FIT_CACHE. Strong refs hold every fitted MRMR; long-lived workers (web services, JupyterHub kernels) leaked memory unboundedly pre-2026-05-15. Default 4 covers a typical model suite (RFECV+MRMR x catboost+linear+mlp) without thrashing.
        fit_cache_max: int = 4,
        # #5: adaptive FE threshold relaxation. When the first-pass
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
        # default 0.9 rationale: the strict gates sit at
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
        # Y-correlation gate on the cross-target identity short-circuit. The "prior identity implies new-target
        # identity" assumption only holds when the new target is correlated with the one that produced the
        # cached identity result. When a candidate hit is found, |corr(new_y_sample, prior_y_sample)| is measured
        # against this threshold; below it, the short-circuit is REFUSED and a full fit runs (the cached entry is
        # left intact for genuinely-correlated future targets). ``mrmr_identity_cache_ycorr_threshold=0.0``
        # disables the gate (legacy: any X-fingerprint hit short-circuits). Default 0.5 (bench-set): a moderate
        # correlation floor that admits composite/residual targets (highly correlated with raw y) while refusing
        # an unrelated target on the same X. The threshold is benched in _benchmarks/bench_identity_cache_ycorr.py.
        mrmr_identity_cache_ycorr_threshold: float = 0.5,
        # When True (the default), ``fit(groups=...)`` raises ``NotImplementedError`` instead of emitting the warn-only "MRMR does not consume groups" UserWarning -- matching
        # ``sample_weight``, which is ALWAYS consumed rather than silently dropped; passing ``groups=`` without ``group_aware_mi=True`` is equally a correctness gap (cross-group leakage in MI
        # estimation on panel / user-session / sliding-window data) and should not silently degrade. Set ``strict_groups=False`` to opt back into the legacy warn-only group-naive fallback for
        # ad-hoc callers who already know the limitation and want MI computed per-row anyway.
        strict_groups: bool = True,
        # Group-aware relevance MI. When True and ``fit(groups=...)`` is supplied, MRMR ranks features by the per-group estimator
        # ``I(X;Y|G) = Σ_g w_g·MM(I_g(X;Y))`` instead of the global ``MI(X;Y)``, so a feature predictive only through
        # between-group LEVEL differences (high global MI, ~0 within-group -- leakage that will not generalise to unseen groups)
        # is demoted, while a genuine within-group signal (even one whose sign flips across groups) is retained. Binning stays
        # global (edges comparable across groups); per-group Miller-Madow debias corrects the small-``n_g`` plug-in bias.
        # OFF by default (opt-in) until a real run confirms the selected-feature set improves out-of-group generalisation.
        group_aware_mi: bool = False,
        # Aggregation of the per-group MI: "size" -> ``w_g = n_g/N`` (the plug-in ``I(X;Y|G)``); "equal" -> equal-weight mean over
        # groups clearing ``group_mi_min_rows`` (so one huge group does not dominate). Only consulted when group_aware_mi=True.
        group_mi_aggregate: str = "size",
        # Groups with fewer than this many finite rows are skipped in the per-group MI aggregate (too few rows for a stable MM estimate).
        group_mi_min_rows: int = 20,
        # Friend-graph post-analysis: after screening, render the selected set as a
        # node-link diagram (node=feature sized by entropy, edge=pairwise MI, arrow=ADC
        # direction) and classify each feature green (unique) / red (suspected redundant
        # sink) / yellow (middling). Diagnostic by default; the fitted graph is exposed as
        # ``self.friend_graph_`` and summarized into the suite's feature_selection_report.
        # OFF by default: the build imports networkx + runs an O(k^2) edge pass + a force-directed
        # spring_layout -- pure diagnostic-display cost on the fit hot path (it dominated a small-data fit profile,
        # almost entirely the one-time ``import networkx``). Turn on for the diagnostic graph; ``friend_graph_prune``
        # auto-builds it regardless (the prune/cluster step requires the graph).
        build_friend_graph: bool = False,
        # When True, drop red (suspected-sink) features from ``support_`` after the graph is
        # built, protecting the neighbor that carries each removed feature's unique target info
        # so cause and effect are never dropped together. Off by default -- changes the selected set.
        friend_graph_prune: bool = False,
        # Guard on the O(k^2) edge pass: above this many selected features the graph keeps
        # node stats only and skips edges (warns). Raise for large opt-in graphs.
        friend_graph_max_nodes: int = 200,
        # GPU acceleration of the friend-graph O(k^2) pairwise-MI edge pass + the k node
        # entropy/relevance stats (the diagnostic build + the prune/cluster path). None
        # (default) -> per-host kernel_tuning dispatch keyed by (k, n), CPU fallback when
        # no GPU / not chosen; "cpu" forces the legacy CPU pass; "cupy"/"cuda" force a GPU
        # backend. BIT-IDENTICAL (GPU does only integer counting; entropy + every keep/drop
        # decision stay on the bit-exact CPU path).
        friend_graph_gpu_backend: Optional[str] = None,
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
        # default flipped from 'augment' to 'replace'.
        # When a denoised aggregate beats its member MIs (gain threshold per
        # ``cluster_aggregate_mi_prevalence``), 'replace' drops the raw members from
        # the final selection AND from candidate consideration so they cannot
        # be re-picked downstream. This eliminates the duplicate-vote effect
        # (raw + aggregate both surviving) that the 'augment' mode silently
        # allowed; the augment behaviour was the most common
        # production-confusion point. Set
        # ``cluster_aggregate_mode='augment'`` to restore the legacy behaviour
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
        # Dynamic Cluster Discovery (DCD).
        # Organic in-greedy-loop cluster discovery using ONLY MI/SU distances
        # (no Pearson — captures non-linear deps like XOR). After each
        # selection, prune the Pool by ``SU(x, just_selected) > tau_cluster``;
        # when cluster reaches threshold, swap raw anchor with PC1 aggregate if
        # ``I(rep ; y | Selected − anchor) > anchor_rel * (1 + swap_gain_threshold)``.
        # Pre-impl gate (bench_dcd_pair_su_scaling) confirmed 0.003× cost vs
        # full pairwise SU at p=10000.
        # 1: dcd_enable=True by DEFAULT. Layer-6 biz_value
        # showed DCD is the documented MRMR mechanism for production
        # redundancy-control (near-duplicate decoys, collinear clusters,
        # synergistic groups). The default was previously False to preserve
        # bit-stability with legacy fits; that's no longer the priority
        # vs giving every user cluster-aware selection out of the box. The
        # 0.003x overhead is negligible. Users wanting the legacy behaviour
        # opt out via dcd_enable=False.
        dcd_enable: bool = True,  # [ACCURACY-CAVEAT] False disables denoised cluster-aggregate; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        # ``dcd_tau_cluster`` accepts ``'auto'`` to
        # opt into the per-fit bimodality-detection calibration sweep
        # (``make_dcd_state`` samples 100 random feature pairs, fits a
        # coarse histogram, and picks tau at the valley between the two
        # SU modes; falls back to 0.7 when the distribution is unimodal).
        # Numeric values keep the legacy fixed-tau behaviour bit-identical.
        dcd_tau_cluster=0.7,
        dcd_distance: str = "su",
        # knobs for the auto-tau calibration sweep.
        # ``dcd_tau_calibration_n_pairs`` is the number of random feature pairs
        # sampled for the bimodality histogram; ``dcd_tau_calibration_seed``
        # makes the random sample deterministic per fit.
        dcd_tau_calibration_n_pairs: int = 100,
        dcd_tau_calibration_seed: int = 0,
        # Cluster must reach this many members (beyond the anchor) before the
        # PC1/aggregate swap is even evaluated. The Layer-42 blocker that kept
        # this at 4 (commit_swap called with engineered_recipes=None, so a swap
        # net-shrank support_) is RESOLVED -- recipe propagation landed in
        # (_mrmr_fit_impl threads engineered_recipes into screen).
        # bench-attempt-rejected (bench_dcd_cluster_size_threshold):
        # lowering the DEFAULT to 2 was benchmarked after the swap-null fix and
        # gives NO actionable win -- swaps fire rarely on small clusters (the
        # swap GATE, not this threshold, is the binding constraint) and mean OOS
        # AUC moved +0.0009 (noise). Keep 4 as the default; pin =2 to opt in.
        dcd_cluster_size_threshold: int = 4,
        dcd_swap_gain_threshold: float = 0.05,
        # (PART B): default flipped to ``"auto"``. When set
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
        # 1 iter 3: permutation-null gate on swap
        # acceptance. With ``full_npermutations > 0`` AND ``dcd_enable=True``,
        # ``evaluate_swap_candidate`` shuffles the PC1 rep B times, builds
        # the null distribution of conditional MI, and only accepts the swap
        # when both the deterministic gain gate AND ``perm_p_value < swap_alpha``
        # are satisfied. Prior to this fix the swap was a pure point-MI
        # comparison and accepted spurious aggregates on noisy / small-n
        # data because PC1 (continuous, re-binned with finer granularity than
        # the raw anchor) is upward-biased.
        dcd_swap_alpha: float = 0.05,
        # (audit dcd-core-1 / dcd-swap-null-1/2): the swap
        # permutation null's draw count, decoupled from ``full_npermutations``
        # (the screening confidence, default 3) which only acts as the on/off
        # switch. Reusing full_npermutations=3 made the null un-passable
        # (min-p (0+1)/(3+1)=0.25 >> swap_alpha 0.05), so EVERY DCD swap
        # (aggregate + member) was silently rejected and the supervised swap
        # subsystem was dead on the default path. 199 gives min-p 0.005;
        # evaluate_swap_candidate also auto-raises B to ceil(1/swap_alpha).
        dcd_swap_npermutations: int = 199,
        # Monotone-warp linear-usability tie-break. When two mutually-redundant candidates are strictly-monotone twins (e.g. f and g=exp(4f), rank-identical so binned MI/SU tie), the cluster-pruning
        # gate keeps exactly one and prunes the other -- otherwise decided by column order alone. ON (default, per the enable-correct-by-default policy) biases that already-forced choice toward the
        # more linearly-usable leg (raw f over its exp-warp g): trees are indifferent but a linear downstream recovers the signal f carried. Detects twins via RAW rank-corr >= ``dcd_warp_twin_rank_corr``
        # (NOT coarse-binned codes) and requires a linear-usability margin (|corr(col, rank col)|) of ``dcd_warp_linear_margin``; one leg is kept either way so this can never empty support_ nor add an
        # unvalidated column, and any non-twin / non-tie pair stays byte-identical to the order-decided default. Opt out with ``warp_tiebreak_prefer_linear=False`` for legacy column-order behaviour.
        warp_tiebreak_prefer_linear: bool = True,
        warp_twin_rank_corr: float = 0.99,
        warp_linear_margin: float = 0.05,
        # ``dcd_postoc_compose=True`` keeps the post-hoc cluster_aggregate
        # active alongside DCD. Default False auto-suppresses it (DCD
        # already processed clusters during screening; running again would
        # double-aggregate).
        dcd_postoc_compose: bool = False,
        # Hybrid orthogonal-polynomial + MI-greedy FE
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
        fe_hybrid_orth_enable: bool = True,  # DEFAULT ON: the orth-FE hybrid DECISIONS now run on the FE row-subsample (fe_decide_on_subsample / inline subsample-replay), so the family is affordable by default; was opt-in only because the full-n decision was too costly.
        # univariate-basis FE, DEFAULT ON. Runs ONLY the
        # orthogonal-basis univariate stage (``a__T2`` ~ a**2, ``a__T3`` ~ a**3,
        # ...), which closes the single-variable-nonlinearity gap the pair-FE
        # path structurally cannot reach (no pairing of two columns makes a clean
        # a**2 out of one column; on a symmetric domain raw ``a`` is
        # uninformative about a**2). Uplift-gated, so it is near-no-op when there
        # is no univariate nonlinearity. The heavier pair-CROSS-basis stage stays
        # behind ``fe_hybrid_orth_enable``. Set False for the legacy pair-only FE.
        fe_univariate_basis_enable: bool = True,
        # ACCURACY GATE, DEFAULT ON. After every FE stage, drop engineered columns that add no held-out downstream uplift over their raw source (a Fourier/Hermite/chirp of a monotone / MNAR / leak column that would otherwise evict the raw signal from support_). Opt out for legacy byte-stability / benchmarks.
        fe_accuracy_gate: bool = True,  # [ACCURACY-CAVEAT] False lets sub-bar engineered candidates in; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        # FOURIER univariate basis, DEFAULT ON. The orthogonal-poly
        # univariate basis (``a__T2`` / ``a__He2`` ...) recovers polynomial
        # nonlinearities but a degree<=4 polynomial CANNOT express a full-period
        # (or higher-frequency) sinusoid, so a pure oscillatory univariate signal
        # (``y = sin(d)``, ``sin(2d)``, ``sin(a**2)`` ...) was only partially
        # recovered (measured ``sin(d)`` |corr| ~0.77 via the poly approx). The
        # extra-basis Fourier stage (sin/cos at ``fe_hybrid_orth_fourier_freqs``)
        # closes that. It runs in the DEFAULT univariate path (no longer requiring
        # the heavy ``fe_hybrid_orth_enable`` master switch), uplift+multiple-
        # comparison gated (the same ``sqrt(2 ln n_cands)`` extreme-value threshold
        # the poly basis uses) so it is near-no-op when there is no oscillation.
        # Set False for poly-only univariate FE.
        fe_univariate_fourier_enable: bool = True,
        # ADAPTIVE-FREQUENCY Fourier, DEFAULT ON. The fixed Fourier
        # grid (``fe_hybrid_orth_fourier_freqs``, default {1, 2}) only covers a
        # couple of z-space frequencies; an ARBITRARY-period oscillation
        # (``y = sin(3.7*x)``, ``sin(5.3*x)``, ``sin(6.8*x)``) lands at a NON-
        # integer z-space frequency and is missed by the fixed grid (recovered
        # at |corr| 0.02-0.23 only). When True, each numeric column's dominant
        # z-space frequency is DETECTED via a coarse periodogram-power sweep +
        # local refine, CONFIRMED on a held-out stride slice, and (if it clears
        # ``fe_univariate_fourier_adaptive_min_val_corr``) ADDED to that column's
        # Fourier frequencies. The emitted sin/cos are tagged adaptive and
        # PROTECTED past the MRMR screen (a single leg has low marginal MI --
        # phase -- so the screen would otherwise drop the validated pair).
        # Detection is N-GATED at >= 800 MI rows: at small n a chance frequency
        # over a held-out slice false-positives, so smaller frames skip it
        # entirely (no adaptive column added). Set False for fixed-grid-only.
        fe_univariate_fourier_adaptive: bool = True,
        # Held-out validation effective-|corr| floor the detected frequency's
        # sin+cos support must clear on the held-out slice to be admitted.
        # 0.15 admits genuine arbitrary-period oscillations while rejecting
        # noise (whose held-out periodogram power collapses below the floor).
        fe_univariate_fourier_adaptive_min_val_corr: float = 0.15,
        # ADAPTIVE-CHIRP Fourier, DEFAULT ON. The linear adaptive
        # detector above (and the fixed grid) emit Fourier on the LINEAR argument
        # and therefore cannot represent an oscillation whose frequency GROWS with
        # the argument -- a "chirp" ``y ~ sin(2*pi*f*z**2)``. Over a bounded
        # support a SLOW chirp is already spanned by the linear MULTITONE basis,
        # but a FAST chirp sweeps an instantaneous-frequency band WIDER than the
        # 6-tone deflation basis can cover and the linear path collapses (Phase-0
        # bench: linear R^2 0.07-0.53, and NOTHING at f>=4, vs chirp 0.88). When
        # True, the SAME held-out-validated detector is run on the QUADRATIC-
        # ARGUMENT warp ``u = sign(z)*z**2`` (z standardised on the column), which
        # makes the chirp STATIONARY in u; the emitted ``__qsin``/``__qcos`` legs
        # are tagged adaptive=True and PROTECTED past the screen + EXEMPT from the
        # Spearman dedup, exactly like the linear adaptive legs. ADDITIVE: on a
        # plain linear target the chirp legs are harmless (the screen's held-out
        # gate keeps them only when they genuinely help; combined recovery never
        # falls below linear-only). N-GATED at >= 800 MI rows like the linear
        # path, so small-n frames are byte-identical chirp-on vs off. Set False
        # for the linear-argument-only Fourier path.
        fe_univariate_fourier_chirp: bool = True,
        # Held-out validation effective-|corr| floor for the chirp detector
        # (same semantics + default as the linear adaptive floor above).
        fe_univariate_fourier_chirp_min_val_corr: float = 0.15,
        # Column-count cap on the adaptive/chirp DETECTOR only. Profiled as the single largest default-ON
        # pre-FE cost on a wide-p fit (~34% of the pre-categorize wall at p=420 -- the SAME p=423 scale the
        # MRMR audit's motivating production fit ran at) -- the held-out frequency sweep is roughly linear
        # in column count regardless of row count, unlike most of this pipeline which is already
        # row-subsampled. Default 100 (2026-07-10, corrective-mechanism-on-by-default): bounds cost on
        # wide-p fits while preserving full legacy behaviour (no cap in practice) below it, since the vast
        # majority of tabular problems sit under 100 raw columns; the biz_val tests for this cap use p<=8 so
        # are unaffected. Columns beyond the cap still get the cheap fixed-grid Fourier basis, only the
        # expensive adaptive/chirp detection is capped. ``None`` restores the pre-2026-07-10 unlimited behaviour.
        fe_univariate_fourier_adaptive_max_cols: Optional[int] = 100,
        # HINGE / piecewise-linear change-point basis.
        # DEFAULT ON. Captures a SLOPE CHANGE at a data-dependent
        # threshold ``y = a*x + b*max(x - tau, 0)`` (pricing tiers, dose-response,
        # saturation) -- a signal shape NOTHING in the catalog captures:
        # ``numeric_rounding`` is piecewise-CONSTANT (wrong form), the cubic
        # B-spline rounds off a sharp kink at its FIXED quantile knots, and an
        # orthogonal polynomial needs a high degree + rings (Gibbs) around the
        # kink. The breakpoint ``tau`` is detected by scanning inner-quantile
        # candidate cuts for the max drop in a 2-segment continuous linear-fit SSE
        # (a slope-aware stump), then HELD-OUT-validated on the ``%3`` stride
        # slice (the 2-segment fit must beat plain linear OOS R^2 by
        # ``fe_hinge_min_heldout_r2_uplift``) -- so a chance breakpoint / pure
        # noise admits NO hinge column.
        #
        # RETENTION (the reason this can be default-ON correctly): a single relu
        # leg is MONOTONE in x, hence MI-INVARIANT by the DPI and near-collinear
        # with raw x, so the greedy MI screen DROPS it as redundant -- exactly as
        # it drops a single adaptive-Fourier leg. Its value is downstream LINEAR
        # usability (the SECOND SLOPE ``[1, x, relu(x-tau)]`` hands a linear model),
        # NOT marginal MI. The leg is therefore admitted on its held-out
        # incremental linear-R^2 over raw x in the FE stage, and the support-
        # finalisation HINGE-PROTECTION block re-adds any leg the MI screen drops
        # whose raw SOURCE survived (self-limiting: a hinge on a never-selected
        # noise / smooth / linear column is left out). So a genuine slope-change
        # column keeps its leg in support_ on the DEFAULT path while neutral data
        # adds zero hinge columns -- no generate-then-drop waste, no spurious cols.
        #
        # COST (the legitimate concern behind the old opt-in): the full per-column
        # scan is O(n * n_cuts) lstsq solves (~2.2 ms/col). Default-on is kept
        # cheap by a 3-cut SSE PRE-CHECK (``_hinge_slope_change_plausible``) that
        # short-circuits the 24-cut scan for any column without a plausible slope
        # change (the common case on wide data) -- ~8x fewer solves on a no-kink
        # column, so wide / large-p fits are not bloated (measured p=50 n=4000:
        # the default-on wall add is small -- see bench in the FE backlog doc).
        # Recipes (``hinge_basis``) store only ``{tau, side}`` -- NO y -- so
        # transform replay is the pure function ``np.maximum(x-tau,0)``, leak-free.
        # Set False to disable hinge engineering entirely.
        fe_hinge_enable: bool = True,
        fe_hinge_top_k: int = 5,
        fe_hinge_max_breakpoints: int = 2,
        # Emit a step indicator ``1[x > tau]`` alongside the relu legs. Default
        # False -- the relu legs already span the continuous piecewise-linear
        # family; the discontinuous indicator overlaps numeric_rounding.
        fe_hinge_emit_indicator: bool = False,
        # Held-out R^2-uplift floor: the 2-segment hinge fit must beat the
        # 1-segment (plain linear) fit by at least this much OOS R^2 on the %3
        # stride slice for the breakpoint to be admitted. 0.02 leaves a wide
        # margin between the noise control (uplift ~0) and a genuine slope change
        # (uplift ~0.3+).
        fe_hinge_min_heldout_r2_uplift: float = 0.02,
        # SYNERGY BOOTSTRAP, DEFAULT ON (cap-gated). Pure-synergy
        # interactions (``y = a*d``, ``sign(a)*sign(d)``, ``log(c)*sin(d)`` ...)
        # carry ~ZERO MARGINAL MI on each factor (``E[y|a]=E[y|d]=0`` by symmetry),
        # so the greedy screen never selects either factor and the pair therefore
        # never enters the FE pool (which is the SELECTED set). The pair-MI
        # prospective screen ALREADY keeps zero-individual-MI pairs whose JOINT MI
        # is positive (the canonical XOR branch); it simply never SEES those pairs.
        # When the raw numeric feature count is <= this cap, the FE step augments
        # the pair pool with the UNSELECTED raw numeric columns so an all-pairs
        # joint-MI sweep (O(p^2), cheap at small p; bounded by the batch pair-MI
        # path's MAX_K=200) screens the synergy pairs. Set 0 to disable.
        # raised 60 -> 250. The old 60 SKIPPED the bootstrap on any
        # frame wider than 60 cols, so moderate-width frames (e.g. 220 cols) never
        # got their interaction products engineered. MEASURED: enabling it on a
        # 220-col frame lifts standalone MRMR downstream AUC +0.045 (and the hybrid
        # +0.030); it is a no-op below 60 (already ran). ABOVE the cap, the wide-frame
        # interaction-propensity pre-rank (``fe_synergy_prerank``, below) now picks the
        # top ``cap`` columns to sweep instead of skipping the bootstrap outright.
        # The downstream cost is bounded -- the synergy SWEEP is O(p^2)
        # joint-MI but only the top ``fe_synergy_max_pairs`` (16) pairs proceed to
        # the expensive per-pair search, and the sweep uses the GPU/batch pair-MI
        # path. 250 is the cost/benefit sweet spot: it covers moderate-p frames while
        # still skipping very-wide ones (e.g. 500+) where the O(p^2) sweep is the
        # cost wall AND the interactions are typically not pairwise-bilinear anyway.
        # On a VERY large n with p in (60, 250] the sweep is heavier -- lower this
        # cap (or set 0) if FE wall-time on such a frame matters.
        fe_synergy_screen_max_features: int = 250,
        # WIDE-FRAME INTERACTION-PROPENSITY PRE-RANK. When the raw numeric count EXCEEDS
        # ``fe_synergy_screen_max_features``, choose WHICH ``cap`` columns enter the O(p^2) synergy sweep by an
        # interaction-propensity score |corr(x^2,y)|+|corr(x,y^2)| instead of skipping the bootstrap. Marginal MI
        # is the wrong ranking (a pure-interaction operand has ~0 marginal MI by construction); higher moments
        # leak even when the linear marginal is flat. Bench (test_fe_interaction_prerank, 2026-06-18): recovers
        # planted zero-marginal operands into the top-250 at recall ~0.88 (realistic leakage L=0.1) vs marginal-MI
        # 0.68 / random 0.12, at O(p*n) ~5s for p=10k. NO-OP below the cap (the bootstrap already adds every
        # column). IRREDUCIBLE: a perfectly-balanced zero-higher-moment interaction (L=0) is invisible to any O(p)
        # score and still requires the full exhaustive sweep -- the pre-rank does not claim it. Default ON; set
        # False to restore the legacy skip-past-cap behaviour (engineer nothing on wide zero-marginal frames).
        fe_synergy_prerank: bool = True,
        # SECOND FUNNEL STAGE -- GPU-EXHAUSTIVE SYNERGY SWEEP. The pre-rank above is an O(p)
        # propensity score; a PERFECTLY BALANCED (L=0) interaction (balanced XOR / sign product whose every
        # univariate higher moment vs y is zero) is INVISIBLE to it, so neither operand enters the kept cap.
        # Only the EXHAUSTIVE C(p,2) joint-MI sweep recovers such a pair (the measured CUDA kernel ranks a
        # planted balanced XOR pair #0 of 50M with joint MI = ln2). ``fe_synergy_exhaustive``:
        #   * "auto" (default): ESCALATE to the full exhaustive C(p,2) sweep WHEN it is affordable -- a CUDA GPU
        #     is available AND the predicted wall-time is <= ``fe_synergy_exhaustive_max_seconds`` -- otherwise
        #     fall back to the pre-rank + capped sweep. So the DEFAULT gets the COMPLETE result (incl. the
        #     balanced L=0 case) for free at small/moderate p, and only wide frames where exhaustive would blow
        #     the budget use the cheap O(p) pre-rank (which still recovers any LEAKY interaction at ~0.88 recall;
        #     only the measure-zero perfectly-balanced case is then missed). This is why "auto" is not merely a
        #     slower pre-rank: it is exhaustive-when-cheap, pre-rank-when-not.
        #   * "force"/True: run the exhaustive sweep whenever a CUDA GPU is available, IGNORING the budget (the
        #     user explicitly wants completeness and accepts the wall-time).
        #   * "never"/False: always the pre-rank + capped sweep (guaranteed fast; never pays for the GPU sweep).
        # The exhaustive sweep bypasses the cap, the pre-rank, AND the n*p^2 cost gate, and reuses the existing
        # ``batch_pair_mi_cuda`` kernel (no new kernel). Throughput (CUDA pairs/s) is measured-and-cached per
        # host + (n, p) via pyutilz.performance.kernel_tuning (NOT hardcoded; ~5e4 pairs/s is only the cold-cache
        # fallback). No CUDA GPU -> "auto"/"force" both fall back to the pre-rank (CPU exhaustive is too slow).
        fe_synergy_exhaustive: str = "auto",
        # OPTIONAL override (seconds) for the "auto" exhaustive-escalation budget. By DEFAULT (None) the budget
        # is MRMR's own ``max_runtime_mins`` * 60; if max_runtime_mins is ALSO unset, the budget is UNLIMITED --
        # "auto" then escalates to the exhaustive sweep regardless of p (the user did not ask to bound wall-time).
        # Set this (or max_runtime_mins) to bound the worst-case FE wall-time: at the bench ~5e4 CUDA pairs/s,
        # p=2000 -> ~38s, p=5000 -> ~241s, p=10000 -> ~1004s. "force" ignores the budget entirely.
        fe_synergy_exhaustive_max_seconds: float | None = None,
        # N-AWARE COST GATE on the synergy bootstrap's all-pairs joint-MI sweep (O(p^2) pairs x O(n) each). The
        # feature cap above does NOT bound wall-time -- a wide-but-not-too-wide frame at large n blows up
        # super-linearly (measured p=200: n=5k +108%, n=20k +300%, n=100k >24min). The bootstrap fires only when
        # n * p^2 <= this budget. 5e8 fires on the measured WINS (hard_synth n=5000 p=220 -> 2.4e8) but SKIPS the
        # large-n blow-ups (n=100k p=200 -> 4e9). Set to float("inf") to disable the cost gate (cap-only behaviour).
        fe_synergy_max_sweep_cost: float = 5e8,
        # Budget on the number of SYNERGY pairs (>=1 operand is a bootstrap-added
        # unselected column) that proceed to the EXPENSIVE per-pair unary/binary/
        # prewarp search. On a signal-drowned-in-noise frame many noise pairs clear
        # the uplift gate by finite-sample chance; without a budget every one would
        # be searched, blowing up FE wall-time. The top ``fe_synergy_max_pairs``
        # synergy pairs by JOINT MI are kept; the rest dropped. Selected-selected
        # pairs are unaffected (count bounded by the small selected set).
        fe_synergy_max_pairs: int = 16,
        # STRICTER joint-MI uplift threshold for SYNERGY pairs than the regular
        # ``fe_min_pair_mi_prevalence`` (1.05). A synergy pair's operands are
        # UNSELECTED -- usually noise -- so adding one as a 2nd joint dimension
        # inflates the finite-sample joint-MI estimate by ~5-15% (more bins =>
        # more positive bias), which would clear the lenient 1.05 gate and inject
        # a spurious feature. Genuine synergy (XOR / sign / bilinear) has joint MI
        # FAR above the marginal sum, so a high bar keeps the real interactions while
        # rejecting bias-only noise pairs. Applies ONLY to synergy pairs;
        # selected-selected pairs keep ``fe_min_pair_mi_prevalence``.
        # raised 1.15 -> 1.5 (with ``fe_min_engineered_mi_prevalence`` 0.90 -> 0.97):
        # a round-3 FE-quality bench + the mlframe recovery suite confirmed the
        # tighter pair was a WIN -- it HALVES the engineered set and cuts spurious
        # noise-products (the optimisation-inflated noise-FE that survive the looser
        # gates, e.g. ``div(log(noise_2),neg(noise_3))``; layer49 noise-containing
        # cols 5 -> 1) for +~0.005 downstream AUC, while genuine synergy (XOR /
        # sign / bilinear, uplift >> 1.5) and every univariate/pair recovery
        # contract are UNCHANGED (142-test recovery+core sweep green; layer49
        # support-bound now met). Held-out-CV validation of every engineered
        # feature (the more surgical fix) was validated in principle -- noise-FE
        # MI collapses to 12-36% on a held-out slice vs genuine's 90-104% -- but
        # needs train-based FE selection (deep rewrite) for marginal gain over this
        # tighter-prevalence cut, so it is deferred.
        # Accepts ``"auto"`` (hardcoded-threshold conversion #3, 2026-06-13): keeps the 1.5 synergy bar
        # but applies it to the MILLER-MADOW-DEBIASED joint MI (the same guarded mechanism as
        # ``fe_min_pair_mi_prevalence="auto"``), so a synergy pair is admitted only when its DEBIASED
        # joint MI clears 1.5x the marginal sum -- tightening against finite-sample-bias noise a fixed
        # 1.5 on the RAW MI admits. An explicit float (incl. the 1.5 default) is honoured verbatim.
        fe_synergy_min_prevalence: "float | str" = 1.5,
        # DATA-DRIVEN PAIR PREVALENCE: the hardcoded ``fe_*_min_prevalence``
        # ratio bars over the MM-debiased joint MI under-admit ASYMMETRIC interactions
        # whose one operand has a strong marginal (the joint's analytic bias subtraction
        # exceeds the marginals', dropping the ratio below the bar even when the OTHER
        # operand adds genuine conditional signal -- e.g. F2's ``log(2c)*sin(d/3)``: the
        # (c,d) MM-ratio is ~1.03 < 1.05, yet ``CMI(d; y | c)`` clears its within-stratum
        # permutation null by +0.085 while the noise pair (c,e) sits ON the null). When the
        # ratio bar fails but the pair cleared the order-2 maxT floor, MRMR re-decides with
        # a CONDITIONAL-PERMUTATION NULL (the S5 gate's primitive): condition the weaker
        # operand on the stronger and admit iff its observed CMI clears the null quantile
        # floor by ``fe_pair_perm_null_excess_frac`` of the anchor's marginal MI. The
        # permutation cancels the finite-sample bias by construction, so genuine asymmetric
        # interactions are admitted WITHOUT lowering the bar for noise. Set False to restore
        # the hardcoded-ratio-only screen. DEFAULT OFF -- MEASURED to NOT help: on F2 the
        # CMI null correctly admits the genuine (c,d) pair (oracle feature mul(log2c,sin(d/3))
        # nearly HALVES closed-form-linear MAE, 0.092 -> 0.050), but CMI cannot separate it
        # from an additive cross-mix (a,c), so admitting it ALSO admits cross-mix pairs whose
        # FE composites HURT: closed-form-linear MAE went 0.092 (master) -> 0.097 (fe1, cross-mix
        # esc_poly) -> 0.868 (fe2, step-2 fusion destroys the clean a**2/b). The unary/binary +
        # escalation search never builds the clean log*sin product even when (c,d) is admitted.
        # Kept as an opt-in research knob; a clean (c,d) win needs a NEW separable warp-product
        # proposer (terminal, no fusion), not this admission relaxation.
        fe_pair_perm_null_admission_enable: bool = False,
        fe_pair_perm_null_excess_frac: float = 0.05,
        # TAIL-CONCENTRATED USABILITY ADMISSION. Under heavy operand outliers a genuine ratio
        # (a**2/b) becomes TAIL-CONCENTRATED: its rank-MI collapses (bulk Spearman ~0, signal only in the 5%
        # outlier tail) so the (a,b) pair fails BOTH the joint-MI prevalence and the order-2 maxT gates in
        # ``score_prospective_pairs`` -- even though the ratio carries strong LINEAR usability (|corr(continuous
        # y)| 0.986 for the true form vs 0.371 for the spurious rank-MI winner; corr is outlier-inflated, which
        # is exactly right here). Binning cannot recover it (it clips the outlier tail carrying the a**2
        # magnitude), so this credits a rank-MI-REJECTED pair when the max |Pearson corr(continuous y)| over a
        # small scale/sign-robust bivariate form dictionary of the RAW operands clears
        # ``fe_pair_usability_admission_min_corr`` AND beats the best single-operand form by
        # ``fe_pair_usability_admission_pairness_margin`` (the pairness discriminator: dividing by the TRUE
        # denominator improves corr, dividing by an unrelated operand only adds noise -- so cross-mix / noise
        # pairs and the 'e' operand are rejected). The SAME detector + knobs also credit tail concentration at
        # the two DOWNSTREAM rank-MI gates the tail-concentrated form fails (winner-selection
        # ``_select_single_best`` and the engineered-MI joint-prevalence gate in ``_pairs_score``): when the
        # rank-MI form leader DISAGREES with the |corr(y)| leader beyond the Miller-Madow tie band, the
        # |corr|-best engineered form is promoted as the winner and admitted. Finally, when a RANK-AWARE
        # tail-concentrated pair is present in the FIRST FE sweep's pool (``fe_pair_usability_admission_rank_frac``
        # gates the rank-collapse leg), the first sweep's pair-MI prevalence bar is relaxed to the SAME value the
        # adaptive-threshold retry uses (``max(1.001, bar * fe_adaptive_relax_factor)``) so a co-signal half
        # whose joint MI barely exceeds its marginal sum (F2 (c,d)) builds in the SAME sweep and C2 additive
        # fusion can fuse the two halves. The disagreement + rank-collapse requirements make all of this a
        # strict no-op on the 4 passing F2 profiles + canonical fixtures (there the ratio is BOTH the rank-MI
        # and |corr| leader, rank and linear AGREE -> nothing fires, selection is byte-identical). Default ON.
        # Set False for the legacy rank-MI-only paths (byte-identical everywhere).
        fe_pair_usability_admission_enable: bool = True,
        fe_pair_usability_admission_min_corr: float = 0.6,
        fe_pair_usability_admission_pairness_margin: float = 1.05,
        # RANK-COLLAPSE leg of the FIRST-SWEEP prevalence-relaxation pre-scan: a pair relaxes the bar only when
        # its linear-best raw form's RANK (Spearman-style) association with y is <= this fraction of its linear
        # |corr| -- the tail-concentration signature (linear survives, rank collapses). Balanced data (rank and
        # linear agree) never clears it, so canonical / the 4 passing profiles keep the strict bar.
        fe_pair_usability_admission_rank_frac: float = 0.7,
        # Survivor-strength gate for the raw-operand TAIL-CONCENTRATION subsumption DROP (in drop_redundant_raw_operands):
        # only drop a rank-collapsed raw when the subsuming selected survivor is a NEAR-COMPLETE continuous proxy for y,
        # |corr(continuous y)| >= this. Distinct from the upstream admission gate (fe_pair_usability_admission_min_corr):
        # admission decides whether a tail-concentrated engineered pair is BUILT at all, this decides whether a raw operand
        # is DROPPED as subsumed by a survivor. A weak proxy (survivor |corr(y)| ~0.67) still leaves TREE-recoverable signal
        # the linear-only no-harm reasoning of the drop leg misses, so the drop is only safe when the survivor ~= y. Default
        # 0.85 separates a near-complete proxy (with_outliers F2 survivor ~0.99, drop safe) from a weak one (heavytail ~0.67,
        # drop harmful); the earlier binned-CMI legs still drop genuinely-subsumed operands below this via a different path.
        fe_raw_tail_subsume_min_corr: float = 0.85,
        # Cost guard: max candidate pairs the first-sweep tail-concentration pre-scan inspects (early-exits on
        # the first tail-concentrated pair). Bounds the O(pairs) x O(n) |corr| scan on wide pools. 0 = no cap.
        fe_pair_usability_prescan_max_pairs: int = 256,
        # PAIRNESS-ROUTED PREVALENCE RESCUE: route a SELECTED-SELECTED prevalence-
        # failing maxT-clearing pair to the auto-escalation second-chance (held-out ALS pairness
        # test, which CAN separate a multiplicative interaction from an additive cross-mix).
        # DEFAULT OFF -- MEASURED to be a NO-OP on F2 (the escalation does not fire / produces no
        # (c,d) candidate for the weak rescue pair, so the output is identical to master). Opt-in
        # research knob.
        fe_prevalence_rescue_all_pairs: bool = False,
        # MULTI-CANDIDATE DIVERSE EMISSION: per raw pair the unary/binary search
        # emits only the SINGLE max-target-MI engineered form. MI is a RANK statistic blind to
        # LINEAR usability, so the MI-winner can be a tree-friendly monotone warp a linear model
        # cannot use, while a lower-MI form is the linearly-aligned one (F2: the MI-winner
        # ``sub(exp(c),cbrt(d))`` helps a linear downstream ~0 while the lower-MI
        # ``mul(log(c),sin(d))`` cuts MAE 0.092->0.063). With ``fe_multi_emit_max_per_pair > 1``
        # the search ALSO emits the next DISTINCT forms (greedy by target MI, skipping any whose
        # continuous values correlate above ``fe_multi_emit_diversity_corr`` with an
        # already-emitted column, down to ``fe_multi_emit_mi_floor`` x best_mi); the downstream
        # MRMR redundancy gate prunes residual overlap. Purely additive (never emits fewer than
        # the single-best path); ==1 is byte-identical to the legacy one-per-pair behaviour.
        # DEFAULT 1 (off) pending a downstream bench: measured on F2 it emits the diverse forms
        # but the cross-pair MRMR greedy still selects high-MI cross-mix over the linearly-usable
        # form, so it does not on its own reach the (c,d) goal and it adds candidate-buffer cost;
        # ship default>1 only once a multi-task bench shows a downstream win.
        fe_multi_emit_max_per_pair: int = 1,
        fe_multi_emit_mi_floor: float = 0.5,
        fe_multi_emit_diversity_corr: float = 0.90,
        # ORDER-2 Westfall-Young maxT permutation-null floor on the
        # PROSPECTIVE-PAIR JOINT MI. The FE step ranks O(p^2) candidate pairs by
        # JOINT MI(x_i, x_j; y); at high p the MAX joint MI over PURE-NOISE pairs
        # is a positive order statistic that grows with the pool size -- the SAME
        # best-of-p selection bias the order-1 screening floor rejects, now at
        # order 2. The per-pair prevalence gates above are PER-PAIR and do NOT
        # account for max-over-pool selection, so a wide noise matrix still
        # surfaces "synergistic-looking" noise pairs. This floor shuffles the
        # discretised target K times, takes the per-shuffle MAX joint MI over the
        # candidate pool via the SAME batched estimator the screen scores
        # ``pair_mi`` with, and floors prospective-pair selection at the q-th
        # quantile of those maxes -- a genuine synergy pair clears it, the
        # best-of-p noise does not. Applied IN ADDITION to the prevalence gates,
        # computed ONCE per FE step. SELF-GATING: below ``fe_pair_maxt_min_pairs``
        # candidate pairs the floor is 0.0 (no-op => byte-identical narrow pools).
        # Set ``fe_pair_maxt_null_permutations=0`` to disable. DEFAULT-ON (mirrors
        # the order-1 ``screen_predictors`` floor). See ``_permutation_null.py``.
        fe_pair_maxt_null_permutations: int = 25,
        fe_pair_maxt_null_quantile: float = 0.95,
        fe_pair_maxt_min_pairs: int = 30,
        # SIGNED INTERACTION-INFORMATION (co-information) routing on the prospective
        # pairs (backlog idea #8). The prevalence ratio gate + order-2 maxT floor
        # admit a pair whose JOINT MI beats the marginal sum, but that fires for BOTH
        # genuine synergy (a,b JOINTLY carry y: a**2/b, log(c)*sin(d), XOR) AND for an
        # ADDITIVE cross-mix where a feeds one independent term of y and b a DIFFERENT
        # one (the user's weak-F2 surrogate ``add(invqubed(a), invsqrt(c))`` mixing an
        # (a,b)-term operand with a (c,d)-term operand -- a and c do NOT interact in y).
        # Signed ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)`` separates them: >0 genuine
        # synergy, ~0 additive (no interaction), <0 redundancy. All three terms are
        # ALREADY computed by the gate (``cached_MIs`` marginals + ``pair_mi`` joint), so
        # II is a near-free signed re-read. Each term is Miller-Madow corrected on its own
        # cardinality before differencing (the JOINT term has nbins_a*nbins_b bins, ~nbins x
        # the marginal bias, so an un-corrected difference would manufacture a positive II
        # out of finite-sample inflation). A deterministic/low-noise SUM still yields a
        # SMALL positive "completion" II, so positive II is FLOORED by a permutation null on
        # the per-shuffle MAX II (same maxT machinery as the joint-MI floor): a genuine
        # multiplicative synergy sits far above it, the additive cross-mix below. Pairs that
        # already passed the gate AND are speculative (synergy-bootstrap-added operand) AND
        # route ADDITIVE (II <= floor) are DEMOTED out of the per-pair FE search so no
        # cross-mix surrogate is built; positive-II -> product/cross-basis, negative-II ->
        # cluster-aggregate-eligible (tags surfaced in ``fe_interaction_routes_``). This
        # changes RANKING/ROUTING only -- the maxT floor + ratio gate stay as the detection
        # guards (iron rule (d)). SELF-GATING: floor==0.0 (narrow pool / disabled) => every
        # pair kept (byte-stable). Set ``fe_ii_routing_null_permutations=0`` to disable.
        #
        # bench-rejected as a DEFAULT for the user's weak-F2 cross-mix --
        # DEFAULT-OFF. The mechanism is correct and DOES cleanly separate STRONG synergy from
        # additive/redundancy (synthetic n=3000: synergy II=+0.55 vs additive II=+0.03 below the
        # null floor vs redundancy II=-1.10; unit tests pin this). But on the user's WEAK F2
        # (``0.2*a**2/b + f/5 + log(c*2)*sin(d/3)``, coefficient 0.2, unobserved f, 6-8 supervised
        # bins) it does NOT reduce the cross-mix: measured per-pair II on the 3 cross-mix seeds
        # {0,6,8} at n=20000 shows the CROSS-MIX pair (b,c) has the HIGHEST II on EVERY seed
        # (+0.0132/+0.0135/+0.0139) -- ABOVE the genuine (a,b) a**2/b pair (+0.0114/+0.0120/
        # +0.0132). The y-shuffle null floor sits at ~0.0007, two orders of magnitude BELOW every
        # real pair, so all route ``synergy`` and nothing is demoted. Because the cross-mix II
        # EXCEEDS a genuine-interaction II in this regime, NO II threshold (absolute, relative, or
        # null-floored) can demote the cross-mix without also demoting the genuine pair: under
        # coarse binning + weak signal the "completion synergy" of two weakly-informative columns
        # is information-theoretically as large as the genuine 2-way signal. F2 10-seed result was
        # NEUTRAL (cross_mix 3/10 -> 3/10, genuine_ab 10/10 -> 10/10, genuine_cd 0/10 -> 0/10).
        # Kept as an OPT-IN (``fe_ii_routing_enable=True``) -- it is a sound co-information router
        # for strong-signal / large-n / fine-binned pools (see ``_interaction_information.py`` +
        # ``test_interaction_information_routing.py``); it just cannot beat the F2 detection floor,
        # which is the same too-weak-signal floor the backlog flags (log*sin recovers 0/10).
        fe_ii_routing_enable: bool = False,
        fe_ii_routing_null_permutations: int = 25,
        fe_ii_routing_null_quantile: float = 0.95,
        fe_ii_routing_min_pairs: int = 30,
        # SURROGATE-GBM SPLIT-CO-OCCURRENCE INTERACTION SEEDER (backlog idea #6). The
        # univariate-MI ``seed_count`` that the prospective-pair sweep / triplet FE pick
        # source columns by is BLIND to pure synergy: a zero-marginal interaction operand
        # (``y = sign(x_a*x_b*x_c) + noise`` -- every marginal MI ~= 0) is never ranked
        # top-N, so the pair is never enumerated and the triple never seeded -> the needle
        # is MISSED. This proposer fits ONE shallow LightGBM (~150 depth-4 trees) on the
        # discretised matrix, walks every root-to-leaf path, and tallies a depth-discounted
        # split-gain co-occurrence weight for every co-splitting PAIR + TRIPLE: a zero-marginal
        # operand still appears as a split partner CONDITIONED on its co-splitter, so
        # co-occurrence ranks the true interaction members at the top. Top-K pairs feed the
        # prospective pool (BYPASSING seed_count); top-K triples feed the order-3-floored
        # triplet FE. Cost O(n*trees*depth) -- INDEPENDENT of p^2/p^3 (the large-p scaling
        # lever). SELF-GATE: emits NOTHING unless the surrogate's OOF score beats a permuted-y
        # baseline (pure noise -> tie -> no seeds -> pool not polluted). The order-2/order-3
        # maxT floors then gate every emitted candidate (proposer GENERATES, floors GATE).
        #
        # OPT-IN default (``fe_gbm_seeder_enable=False``) per the iron rule's "ship behind a
        # flag if cost-risky on small p": a LightGBM fit + a permuted-y refit per FE step is a
        # fixed cost that is NOT worth paying on the narrow tabular pools where seed_count
        # already sees every operand; it pays off on LARGE-p frames with zero-marginal
        # interactions. ``fe_gbm_seeder_min_features`` self-routes it OFF below a pool width
        # where seed_count is not the blocker; raise the flag to enable. Needs lightgbm.
        fe_gbm_seeder_enable: bool = False,
        fe_gbm_seeder_min_features: int = 30,
        fe_gbm_seeder_top_k_pairs: int = 12,
        fe_gbm_seeder_top_k_triples: int = 8,
        fe_gbm_seeder_n_estimators: int = 300,
        fe_gbm_seeder_max_depth: int = 4,
        fe_gbm_seeder_self_gate_margin: float = 0.0,
        # SELF-GATE as a PERMUTATION SIGNIFICANCE TEST (not a single-split point comparison):
        # ``self_gate_reps`` real + ``reps`` permuted-y OOF splits; the real-mean must sit
        # ``self_gate_min_z`` sigma above the permuted-null distribution. A single split is
        # too noisy (~+/-0.02 acc) and false-positives on a wide noise pool; the z-test makes
        # pure noise reliably FAIL. The surrogate trains on the CANDIDATE-ONLY submatrix (the
        # target column is excluded -- training on the full screening matrix, which contains
        # the discretised target, would leak a perfect OOF).
        fe_gbm_seeder_self_gate_reps: int = 5,
        fe_gbm_seeder_self_gate_min_z: float = 2.0,
        # GRADIENT-INTERACTION (MIXED SECOND PARTIALS) SEEDER. Fits one smooth
        # differentiable RFF+ridge surrogate on a row sample and proposes the operands of pairs
        # (a, b) whose ``E[(d2f/dxa dxb)^2]`` is large -- the calculus definition of a non-additive
        # interaction (a sum ``g(a)+h(b)`` has mixed partial == 0). Targets SMOOTH/ROTATED
        # interactions (a*b hyperbolic saddles, sin(a)*b). Same pool plug point as #6: proposes
        # operands, the maxT floor + CMI/prevalence gates DECIDE. Self-gated by an OOF-R2 vs
        # permuted-y check, a GAM additive-residual baseline (additive targets emit 0), and a
        # permutation null on the max mixed-partial energy.
        #
        # OPT-IN default (``fe_gradient_interaction_enable=False``). bench-reject (2026-06-10,
        # ``_gradient_interaction_seeder.py``): on the prescribed cheap-validation fixture
        # ``y=sin(x5)*x31+noise`` (n=2000, p=60) the gradient detector ranks the (5,31) saddle #1
        # and proposes exactly that pair, BUT the #6 GBM split-co-occurrence seeder ALSO ranks
        # (5,31) #1 -- modern boosting represents a smooth 2-way product over N(0,1) fine, so the
        # two are equally good there, NOT complementary, and the GBM does not under-rank it. The
        # full self-gated proposer (OOF gate + 12-shuffle null) is ~8 s at n=2000/p=60 -- too heavy
        # to default-on. Noise control HOLDS (pure-noise and additive both -> 0 proposals). Routed
        # OFF outside its [min_p, max_p] size regime by ``_route_gradient_seeder`` (thresholds via
        # kernel_tuning_cache). Needs sklearn. Raise the flag to enable.
        fe_gradient_interaction_enable: bool = False,
        # ORDER-3 Westfall-Young maxT permutation-null floor on the candidate-TRIPLE pool
        # (backlog idea #7 -- the MANDATORY rail for any 3-way proposer, ships in the same
        # change as the GBM seeder which opens 3-way). The triplet/quadruplet FE modules lack
        # an order-MATCHED floor; opening 3-way generation WILL surface chance-max noise triples
        # (a wide noise matrix's MAX 3-D joint MI is a positive order statistic growing with the
        # pool -- best-of-pool selection bias at order 3, STRONGER than order 2 because the 3-way
        # joint cardinality inflates the plug-in MI further). This floor shuffles the target K
        # times, takes the per-shuffle MAX 3-way joint MI over the proposed-triple pool via the
        # SAME batched dense-renumber estimator (``batch_triple_mi_prange``, cardinality <= n),
        # and floors at the q-quantile. Gates every GBM-seeded triple. SELF-GATING + mirrors the
        # order-2 knobs; ``=0`` disables. See ``_permutation_null.py``.
        fe_triple_maxt_null_permutations: int = 25,
        fe_triple_maxt_null_quantile: float = 0.95,
        fe_triple_maxt_min_triples: int = 4,
        fe_hybrid_orth_degrees: tuple = (2, 3),
        fe_hybrid_orth_basis: str = "auto",
        # Combined cap on appended columns (univariate + pair). Top-K is
        # applied separately to each stage by the underlying hybrid pipeline;
        # this is the per-stage budget. Default 5 = at most 5 univariate
        # winners + at most 5 pair winners when pair_enable=True.
        fe_hybrid_orth_top_k: int = 5,
        fe_hybrid_orth_pair_enable: bool = True,
        fe_hybrid_orth_pair_max_degree: int = 2,
        # TRI-PRODUCT cross-basis FE (sibling module
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
        # DEFAULT ON: the seed_k cap keeps it bounded (C(seed_k,3) triplets, a handful of
        # candidates regardless of p), and the replay P0 (per-leg preprocess refit) is now fixed. Bench:
        # on a genuine 3-way a*b*c target the linear downstream goes 0.094 -> 0.049 (the floor) because
        # it finally gets the a*b*c feature, with NO harm on an additive target and ~negligible fit cost.
        # Set False to restore the pre-2026-06-13 behaviour (no triplet-cross candidates).
        fe_hybrid_orth_triplet_enable: bool = True,
        fe_hybrid_orth_triplet_max_degree: int = 1,
        fe_hybrid_orth_triplet_seed_k: int = 4,
        fe_hybrid_orth_triplet_top_count: int = 2,
        # QUADRUPLET (4-way) cross-basis FE (sibling
        # module ``_orthogonal_quadruplet_fe``). Captures genuine 4-way
        # interactions: ``y = sign(x_1 * x_2 * x_3 * x_4)`` (4-way XOR
        # where every triplet marginal MI is zero by symmetry) and
        # ``revenue = price * qty * count * discount``.
        #
        # Default OFF -- combinatorial enumeration O(p^4 * deg^4) needs
        # the seed_k cap to stay bounded. With seed_k=4 we get C(4,4)=1
        # quadruplet * deg^4 cells; seed_k=5 yields C(5,4)=5 quadruplets
        # * deg^4 cells (~80 candidates at deg=2). Recipe kind
        # ``orth_quadruplet_cross``; replay reads X only, no y.
        # DEFAULT ON: bounded by seed_k (C(seed_k,4) quadruplets -- 1 at seed_k=4), replay
        # P0 fixed, captures genuine 4-way interactions a linear model cannot otherwise form. Set False
        # to restore the pre-2026-06-13 behaviour.
        fe_hybrid_orth_quadruplet_enable: bool = True,
        fe_hybrid_orth_quadruplet_max_degree: int = 1,
        fe_hybrid_orth_quadruplet_seed_k: int = 4,
        fe_hybrid_orth_quadruplet_top_count: int = 2,
        # ADAPTIVE-ARITY cross-basis FE (sibling
        # module ``_orthogonal_adaptive_arity_fe``). Tries arity 2/3/4
        # per seed tuple and emits ONLY the winning arity per maximal
        # signal set, so the caller does not have to pick arity by hand.
        # Independent opt-in (does NOT require fe_hybrid_orth_enable).
        # Combinatorial cost O(sum_{k=2..A} C(seed_k, k) * deg^k); with
        # defaults A=3, seed_k=4, deg=1 the candidate count is C(4,2)+
        # C(4,3)=10. Recipes route to the per-arity Layer 22 / 56 / 77
        # builders -- no new recipe kind.
        fe_hybrid_orth_adaptive_arity_enable: bool = False,
        fe_hybrid_orth_adaptive_arity_max_arity: int = 3,
        fe_hybrid_orth_adaptive_arity_max_degree: int = 1,
        fe_hybrid_orth_adaptive_arity_seed_k: int = 4,
        fe_hybrid_orth_adaptive_arity_top_count: int = 3,
        # FE-FAMILY COMPUTE BUDGETING (gt_07, sibling module ``filters/_fe_family_budget.py``).
        # When True, scales the triplet/quadruplet/adaptive-arity seed_k/top_count quotas by a
        # per-family budget fraction persisted across fits (keyed by a dataset fingerprint -- a
        # different dataset always starts from an equal-split budget, never carries over another
        # dataset's learned fractions), and after fit reallocates that budget proportional to each
        # family's realized-importance-credit / wall-cost ROI (with a mandatory floor + exploration
        # reserve so no family is ever permanently starved -- see ``reallocate_budgets``'s
        # docstring). Additive credit approximation (not full family-Shapley): each surviving
        # engineered column's ``mrmr_gain`` is attributed to its recipe kind's family; a
        # ``credit="loo"`` (leave-one-family-out) upgrade is specced for future work when families'
        # outputs are strongly redundant (additive credit double-counts shared value; LOO would not).
        # Default "auto" (opt-in-once-then-remembered, not opt-in-every-call): "auto" probes the
        # persisted-budget cache for THIS dataset's fingerprint before the fit and behaves exactly like
        # ``False`` if nothing is cached there -- it can never silently start learning on a dataset that
        # has never once been fit with ``fe_budget_learning=True`` explicitly, since that persisted cache
        # entry is the only way the probe finds anything. Once a caller has opted in at least once for a
        # given dataset, later "auto" fits on the SAME dataset (e.g. a daily retraining job re-running
        # this exact MRMR config) resume learning without the caller re-passing ``True`` every call.
        # Pass ``True``/``False`` explicitly to bypass the probe entirely (unconditionally on/off).
        fe_budget_learning: bool | str = "auto",
        fe_budget_kwargs: Optional[dict] = None,
        # SEMI-SUPERVISED basis-preprocess fitting
        # (sibling module ``_semi_supervised_fe``). Independent opt-in (does
        # NOT require fe_hybrid_orth_enable). When True AND the user invokes
        # ``fit_with_unlabeled(mrmr, X_labeled, y, X_unlabeled)``, the L21 /
        # L22 / L56 / L77 / L78 orth-poly basis preprocess fits (z-score /
        # min-max / shift params) consume the concatenated
        # ``X_labeled + X_unlabeled`` pool per column instead of labeled-only.
        # MI scoring still consumes the LABELED y only. y is never read by
        # the augmentation code path, so leakage is impossible by construction.
        # Default OFF preserves byte-equivalent legacy behaviour (the
        # thread-local pool stays empty and consumers fall back to the
        # labeled-only fit). When the flag is on but the wrapper is not used
        # (plain ``mrmr.fit(X, y)``), behaviour is also byte-identical.
        fe_semi_supervised_enable: bool = False,
        # LASSO-BASED PRE-SELECTION (sibling module
        # ``_orthogonal_lasso_fe``). Independent opt-in (does NOT require
        # fe_hybrid_orth_enable). Layers 21 / 65-74 score candidates via MI /
        # dependence metrics (greedy non-parametric); Lasso (L1) is the dual
        # parametric approach -- fit a single linear model on
        # ``[raw_X, engineered_X]`` and treat |coef| as the per-column score.
        # Complements MI on LINEAR-additive signals where the L1 coefficient
        # path is the natural shrinkage operator. Underperforms MI on
        # non-monotone targets (e.g. ``y = cos(x)``): a linear model has zero
        # Pearson correlation there, so |coef|=0 and the column drops out --
        # this is the expected behaviour, not a regression.
        #
        # Recipes still use ``orth_univariate`` kind (engineered VALUES are
        # bit-equal to Layer 21 -- only the selection metric changes), replay
        # reads X only, no y. Default OFF preserves legacy pickle byte
        # equivalence.
        fe_hybrid_orth_lasso_enable: bool = False,
        fe_hybrid_orth_lasso_alpha: float = 0.01,
        # ELASTIC NET (L1 + L2) PRE-SELECTION (sibling
        # module ``_orthogonal_elasticnet_fe``). Independent opt-in (does NOT
        # require fe_hybrid_orth_enable). Lasso (Layer 81) arbitrarily picks
        # one of a correlated candidate pair and zeroes the rest; the L2
        # penalty in Elastic Net shares coefficient mass among correlated
        # columns ("grouping effect", Zou & Hastie 2005), so a correlated
        # pair survives or drops together rather than being arbitrarily
        # split. ``l1_ratio=1.0`` reproduces Lasso; ``l1_ratio=0.0``
        # reproduces Ridge; default 0.5 splits the penalty evenly.
        #
        # Recipes still use ``orth_univariate`` kind (engineered VALUES are
        # bit-equal to Layer 21 -- only the selection metric changes), replay
        # reads X only, no y. Default OFF preserves legacy pickle byte
        # equivalence.
        fe_hybrid_orth_elasticnet_enable: bool = False,
        fe_hybrid_orth_elasticnet_alpha: float = 0.01,
        fe_hybrid_orth_elasticnet_l1_ratio: float = 0.5,
        # ADAPTIVE PER-COLUMN DEGREE selection
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
        # CONDITIONAL BASIS ROUTING (sibling module
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
        # DIFF-BASIS FE for highly-correlated source
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
        # PER-CLUSTER SHARED-BASIS FE (sibling module
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
        # BOOTSTRAP-STABLE MI ranking for the hybrid
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
        # THREE-GATE + K-fold OOF MI ranking for the
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
        # KSG / k-NN MI ranking for hybrid orth-poly FE
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
        # COPULA-MI ranking for hybrid orth-poly FE
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
        # DISTANCE-CORRELATION ranking for hybrid orth-
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
        # HSIC (Hilbert-Schmidt Independence Criterion)
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
        # > recipes reuse the ``orth_univariate`` kind. Default
        # OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_hsic_enable: bool = False,
        fe_hybrid_orth_hsic_kernel: str = "rbf",
        fe_hybrid_orth_hsic_n_sample: int = 500,
        # JMIM (Joint Mutual Information Maximisation,
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
        # Total Correlation (Watanabe 1960) multivariate-
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
        # CMIM (Conditional Mutual Information
        # Maximisation, Fleuret 2004) redundancy-aware ranking for hybrid
        # orth-poly FE (sibling module ``_orthogonal_cmim_fe``). Independent
        # opt-in (does NOT require fe_hybrid_orth_enable). Companion to
        # (JMIM): JMIM scores ``min_j I((X_k, X_j); Y)`` (joint
        # MI -- rewards complementarity); CMIM scores
        # ``min_j I(X_k; Y | X_j)`` (conditional MI -- penalises
        # redundancy). On heavily-DUPLICATING candidate pools (near-copies
        # of one strong predictor) CMIM is the empirical winner; on
        # heavily-INTERACTING pools JMIM wins. Engineered VALUES are bit-
        # equal to Layer 21 -> recipes reuse the ``orth_univariate`` kind.
        # Default OFF preserves pickle byte-equivalence.
        fe_hybrid_orth_cmim_enable: bool = False,
        fe_hybrid_orth_cmim_n_bins: int = 10,
        # PER-COLUMN SCORER AUTO-SELECTION across the
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
        # The per-column LCB ratio-to-own-raw-baseline exploding on weak discrete marginals (letting HSIC
        # always win) was fixed via additive headroom normalization: (lcb - raw_max) / scale, not a ratio --
        # default=False is no longer gated on a correctness bug. A quick 5-seed AUC benchmark (mixed linear +
        # non-monotone-cos + high-frequency-sin synthetic) came back NEUTRAL: identical
        # selection and holdout AUC on vs off, because the default plug-in-MI scorer over the
        # existing Chebyshev degree-2/3 basis already fully captured that signal, leaving no
        # headroom for the auto-scorer pool to demonstrate an edge. Not evidence either way; a real
        # decision needs a fixture where the default scorer genuinely under-detects (e.g. a signal
        # right at the edge of what degree-2/3 Chebyshev can fit) before flipping this default,
        # given the added n_boot x 9-scorer bootstrap cost per engineered column.
        fe_hybrid_orth_auto_scorer_enable: bool = False,
        fe_hybrid_orth_auto_scorer_n_boot: int = 5,
        # ENSEMBLE-OF-SCORERS rank-fusion for hybrid
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
        # HSIC added to the ensemble default pool;
        # callers that previously pinned the 4-tuple keep the old
        # behaviour, the default now leverages all five scorers.
        fe_hybrid_orth_ensemble_scorers: tuple = (
            "plug_in", "ksg", "copula", "dcor", "hsic",
        ),
        # META-SCORER auto-selection that LEARNS
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
        # DEFAULT-SCORER ROUTING flag for the Layer 21
        # univariate hybrid orth-poly basis-selection stage. Routes the
        # ``fe_hybrid_orth_enable=True`` univariate dispatch through one of
        # the Layer 65 / 66 / 67 / 71 / 72 / 73 / 74 / 76 / 81 / 82 / 68 / 69
        # scorers instead of the plug-in MI default. Recipes still emit as
        # ``orth_univariate`` (engineered VALUES are bit-identical -- only
        # the SCORING / SELECTION differs), replay reads X only, no y, so
        # the replay path is shared with Layer 21. Layer 83's 7-dataset
        # showdown placed CMIM (5/7 wins) on top of the empirical leader-
        # board, so callers that want to pick the empirically-best scorer
        # in one knob can set ``"cmim"`` here without having to know which
        # specific master flag to toggle. ``"plug_in"`` (default) keeps
        # behaviour byte-identical to pre-L85 master. Accepted values:
        # ``"plug_in"`` / ``"cmim"`` / ``"jmim"`` / ``"tc"`` / ``"ksg"`` /
        # ``"copula"`` / ``"dcor"`` / ``"hsic"`` / ``"auto"`` (Layer 68
        # bootstrap LCB per-column) / ``"ensemble"`` (Layer 69 rank-fusion)
        # / ``"meta"`` (Layer 76 cascade) / ``"lasso"`` (Layer 81) /
        # ``"elasticnet"`` (Layer 82) / ``"auto_oracle"`` (Layer 100).
        # ``"auto_oracle"`` UNIFIES the two prior scorer-selection paths:
        # it uses the L76 cold-start cascade as its prior, the L68 bake-off
        # (run once via ``OracleScorerSelector.benchmark_all_scorers``) as
        # the populator, and the Param-Oracle to LEARN the best scorer per
        # dataset fingerprint over time -- recommending the learned-best
        # scorer once a fingerprint bucket has confident history and
        # falling back to the L76 rules until then. NOTE: when this flag is non-default,
        # the pair stage (Layer 22) is skipped because the alternate
        # scorers operate on the univariate stage only -- callers needing
        # both should keep ``"plug_in"`` and toggle the per-stage opt-in
        # flags individually.
        fe_hybrid_orth_default_scorer: str = "plug_in",
        # extra (non-polynomial) basis FE: B-spline +
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
        # POWER-arguments for the Fourier basis: build sin/cos on x**p for each p.
        # p=2 captures even-argument CHIRPS (``sin(a**2)`` etc.) that a Fourier on the
        # linear argument cannot. Self-contained replayable recipe (raw -> power ->
        # Fourier). (1,) for linear-argument-only.
        fe_hybrid_orth_fourier_powers: tuple = (1, 2),
        fe_hybrid_orth_spline_knots: int = 5,
        # generic MI-greedy FE constructor (sibling
        # to the orthogonal-polynomial one). Default OFF -- legacy
        # behaviour is byte-identical when ``fe_mi_greedy_enable=False``.
        # When True, the MI-greedy FE runs ONCE before screening (after
        # the hybrid orth stage when both are enabled): it enumerates
        # generic unary / binary transforms (log_abs, sqrt_abs, square,
        # cube, reciprocal_safe, tanh, expm1_clip, abs / add, sub, mul,
        # div, max, min, abs_diff, ratio_log) over the top-N source
        # columns by raw MI, ranks the candidates by MI uplift, and
        # appends the top-K winners to X. Recipes of kind
        # ``"mi_greedy_transform"`` carry transform name + src cols only
        # (no y), so transform() replay is leakage-free.
        fe_mi_greedy_enable: bool = False,
        fe_mi_greedy_top_k: int = 5,
        fe_mi_greedy_seed_cols_count: int = 5,
        fe_mi_greedy_include_unary: bool = True,
        fe_mi_greedy_include_binary: bool = True,
        # CMI-greedy FE constructor (sibling to the
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
        # K-fold target encoding for raw categorical
        # columns. Default OFF -- legacy behaviour is byte-identical when
        # ``fe_kfold_te_enable=False``. When True, after the hybrid + MI-
        # greedy stages run, every column in ``fe_kfold_te_cols`` (or
        # auto-detected categoricals with cardinality in [5, 500] when the
        # tuple is empty) is target-encoded with K-fold OOF discipline and
        # the encoded ``{col}__te`` column is appended to X. The recipe
        # (``kfold_target_encoded``) stores the FULL-data per-category
        # mean for deterministic replay -- no y is referenced at transform.
        fe_kfold_te_enable: bool = True,
        fe_kfold_te_cols: tuple = (),
        fe_kfold_te_folds: int = 5,
        fe_kfold_te_smoothing: float = 10.0,
        # Per-category target STATISTICS to emit. Each becomes a separate leak-safe ``{col}__te_{stat}`` column
        # (``mean`` keeps the historical ``{col}__te`` name). ``std`` / ``skew`` / ``kurt`` carry the within-
        # category spread / asymmetry / tailedness of y -- signal the mean cannot express when the category
        # MODULATES a raw feature (heteroscedastic / varying-slope targets): measured +0.04..+0.09 OOS R^2 in
        # those regimes. DEFAULT = full panel: harmless elsewhere -- for a binary target std/skew/
        # kurt are deterministic functions of the mean, so the MI screen drops them as redundant. Pass
        # ``("mean",)`` to restore the lean single-stat encoder. See ``_target_encoding_fe.TE_SUPPORTED_STATS``
        # and ``_benchmarks/bench_multistat_cell_encoding``.
        fe_kfold_te_stats: tuple = ("mean", "std", "skew", "kurt"),
        # GROUPED AGGREGATION OVER QUANTILE-BINNED NUMERIC CELLS. Default OFF. When True, each
        # eligible NaN-free numeric column is quantile-binned (UNSUPERVISED -- no y-leakage; equal-frequency
        # cells -> uniform per-cell sample size for stable higher moments) into a group key, and the per-cell
        # mean/std/skew/kurt of every OTHER numeric column become leak-safe ``binagg_{stat}(...)`` features. The
        # within-cell SPREAD / SHAPE carries signal the cell mean cannot when the target is heteroscedastic /
        # the cell modulates a feature (measured +0.9 OOS R2 on a sigma(cell) target where the cell mean is
        # ~constant). Bin count = ``min(fe_binned_numeric_agg_nbins, moment_stability_cap)`` ties resolution to
        # the highest requested moment (Freedman-Diaconis is bench-REJECTED: it over-bins at large n); high-order
        # moments whose per-cell sample floor cannot be met are auto-dropped rather than coarsening everything.
        # Edges stored per group column for leak-safe replay. See ``filters._binned_numeric_agg_fe``.
        # DEFAULT OFF, consistent with every sibling FE master switch (kfold_te / count_encoded / grouped_agg /
        # cat_num_residual ...): the within-cell spread/shape is a measured lift on heteroscedastic / cell-
        # modulates-feature targets but is SITUATIONAL, and ON-by-default injects up to max_pairs*stats columns
        # into every fit's screening -- a default-ON flip displaced raw categoricals from the selection on the
        # kitchen-sink fixture (test_biz_value_mrmr_layer39), confirming broad selection perturbation on shared
        # infra. Opt in with ``fe_binned_numeric_agg_enable=True`` where the target's cell-conditional spread matters.
        fe_binned_numeric_agg_enable: bool = True,
        fe_binned_numeric_agg_stats: tuple = ("mean", "std", "skew", "kurt"),
        fe_binned_numeric_agg_nbins: int = 10,
        fe_binned_numeric_agg_max_pairs: int = 64,
        # Redundancy gate (default ON): keep a ``binagg_*`` column only when it adds conditional information about y
        # BEYOND its own source columns (``CMI(col; y | group_col, agg_col) >= min_cmi_gain``). Without it the Tier-1
        # MI floor admits binned aggregates that merely re-encode raw signal (e.g. on a linearly-separable target the
        # raw source already explains y), bloating the candidate pool with redundant columns. Set False for the
        # pre-gate behaviour.
        fe_binned_numeric_agg_redundancy_gate: bool = True,
        fe_binned_numeric_agg_min_cmi_gain: float = 0.005,
        # COUNT + FREQUENCY ENCODING for high-
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
        # MISSINGNESS-AWARE FE. Default OFF; legacy
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
        # CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE.
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
        fe_grouped_delta_group_col: str | None = None,
        fe_grouped_delta_num_cols: tuple = (),
        fe_lagged_diff_enable: bool = False,
        fe_lagged_diff_time_col: str | None = None,
        fe_lagged_diff_value_cols: tuple = (),
        fe_lagged_diff_periods: tuple = (1, 2),
        # TWO-TIER IT GATES on the four recipe-emitting
        # FE mechanisms that otherwise emit columns with NO relevance gate
        # (L33 k-fold target encoding, L34 count/freq/cat-num residual, L37
        # missingness, L38 ratio/grouped-delta/lagged-diff). Both default OFF
        # so legacy behaviour is byte-identical.
        # * ``fe_local_mi_gate`` (Tier 1): after each of the four mechanisms
        #   generates its candidate columns, drop any whose marginal
        #   ``MI(col; y)`` is below the RAW-baseline noise floor (median +
        #   3.5*MAD of the raw columns' MI distribution -- anchored on raw, not
        #   the engineered pool, per the Layer 90 lesson) and keep the top
        #   ``fe_local_mi_gate_top_k`` survivors. Bounds combinatorial pool
        #   growth (50 cat cols -> <= top_k count-encoded columns).
        #   DEFAULT FLIPPED to True. The gate is a pure
        #   corrective: it only ever drops freshly-engineered columns whose
        #   marginal MI(col; y) is below the RAW-baseline noise floor and keeps
        #   the top ``fe_local_mi_gate_top_k`` survivors. It can never drop a
        #   raw input feature, never reduces real signal (a predictive
        #   engineered column clears the raw floor by construction), and it is a
        #   strict NO-OP unless one of the four recipe-emitting FE mechanisms
        #   (L33/L34/L37/L38) is also enabled. Enabling by default therefore
        #   shrinks the engineered candidate pool (speed + downstream-MRMR
        #   precision) for every user who turns an FE mechanism on without
        #   reading this docstring, with no accuracy downside. Pin False to
        #   restore the pre-Layer-97 un-gated (full-pool) behaviour.
        # * ``fe_unified_second_pass_gate`` (Tier 2): a single greedy CMI pass
        #   over ALL engineered columns (from every mechanism) conditioned on
        #   the running support (seeded from the top raw-MI columns). Drops a
        #   column when ``CMI(col; y | already-selected) < min_gain`` -- catches
        #   CROSS-mechanism redundancy a per-mechanism gate cannot see
        #   (``count(cat_a)`` vs ``freq(cat_a)`` are affine; only one is kept).
        fe_local_mi_gate: bool = True,
        fe_local_mi_gate_top_k: int = 20,
        fe_unified_second_pass_gate: bool = False,
        fe_unified_second_pass_max_keep: int | None = None,
        fe_unified_second_pass_min_gain: float = 0.005,
        # grouped multi-stat aggregator with CMI gate.
        # NVIDIA cuDF Kaggle-Grandmaster technique #1: per-group statistics of
        # a continuous column broadcast back to rows, plus z-within-group and
        # ratio-to-group residuals. Each survivor is CMI-gated against the raw
        # support and uplift-gated against the source num_col's marginal MI.
        # Default OFF -> byte-identical legacy path. ``group_cols`` /
        # ``num_cols`` empty => auto-detect (int-as-cat / continuous).
        fe_grouped_agg_enable: bool = False,
        fe_grouped_agg_stats: tuple = ("mean", "std", "min", "max", "nunique", "skew", "median"),
        fe_grouped_agg_group_cols: tuple = (),
        fe_grouped_agg_num_cols: tuple = (),
        fe_grouped_agg_top_k: int = 10,
        # COMPOSITE (multi-column) GROUP-KEY aggregates,
        # the multi-col extension of Layer 87. Real aggregations key on more
        # than one column (groupby([region, month]) / groupby([store,
        # category])); the interaction at the composite level often carries
        # signal that neither single-column group exposes. Each composite key
        # is factorized into one integer-coded group and run through the Layer
        # 87 per-group stat / z / ratio machinery; each survivor is CMI-gated
        # against the raw support and uplift-gated against the source num_col
        # marginal MI. Composite keys whose distinct-cell count exceeds
        # 0.5*n are refused (Layer 29 guard). Default OFF -> byte-identical
        # legacy path. ``key_sets`` empty => auto-detect r-combinations (up to
        # ``max_arity``) of detected group columns that clear the guard.
        fe_composite_group_agg_enable: bool = False,
        fe_composite_group_agg_key_sets: tuple = (),
        fe_composite_group_agg_max_arity: int = 2,
        fe_composite_group_agg_stats: tuple = ("mean", "std", "count"),
        fe_composite_group_agg_num_cols: tuple = (),
        fe_composite_group_agg_top_k: int = 10,
        # per-group histogram + quantile FE with
        # target-aware edges. NVIDIA cuDF Kaggle-Grandmaster technique #2:
        # percentile-rank-within-group (empirical CDF position of x in its
        # group) + per-group IQR / p90-p10 spread, optionally a target-aware
        # supervised per-group bin index (OOF-fit MDLP edges maximising
        # I(bin; y)). Each survivor MI-gated against the source num_col
        # marginal MI. Default OFF -> byte-identical legacy path. ``group_cols``
        # / ``num_cols`` empty => auto-detect (int-as-cat / continuous).
        fe_grouped_quantile_enable: bool = False,
        fe_grouped_quantile_quantiles: tuple = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
        fe_grouped_quantile_target_aware: bool = False,
        fe_grouped_quantile_n_bins: int = 5,
        fe_grouped_quantile_top_k: int = 8,
        fe_grouped_quantile_group_cols: tuple = (),
        fe_grouped_quantile_num_cols: tuple = (),
        # cat x cat synergy cross with interaction-
        # information pre-filter. NVIDIA cuDF Kaggle-Grandmaster technique #3:
        # combine two categorical columns into hash(cat_i || cat_j) then
        # target-encode it. The IT enhancement pre-filters pairs by interaction
        # information II(cat_i, cat_j; y) = I(cat_i, cat_j; y) - I(cat_i; y) -
        # I(cat_j; y); only synergistic pairs (II > threshold, e.g. XOR) are
        # materialised. High-cardinality crosses (> 0.5*n distinct cells, the
        # pre-screen) route through K-fold OOF target encoding (Layer
        # 33) instead of a raw integer code. Default OFF -> byte-identical
        # legacy path. ``cat_cols`` empty => auto-detect categoricals.
        fe_cat_pair_enable: bool = False,
        fe_cat_pair_min_interaction_info: float = 0.001,
        fe_cat_pair_cat_cols: tuple = (),
        fe_cat_pair_top_k: int = 5,
        # cat x cat x cat TRIPLE synergy cross via beam
        # search. Extends the Layer 89 pairwise interaction-information cross to
        # the THIRD order: II3(a,b,c;y) = I(a,b,c;y) - [I(a,b;y)+I(a,c;y)+
        # I(b,c;y)] + [I(a;y)+I(b;y)+I(c;y)] (co-information). Positive II3 =
        # genuine three-way synergy NO pair or single explains (the parity target
        # y = a XOR b XOR c, where every pairwise II ~ 0 yet the triple is fully
        # predictive). Beam search seeds from the top-K synergistic PAIRS (Layer
        # 89) and extends each by the best third cat -> <= beam_width * p triples
        # evaluated instead of C(p,3). High-cardinality crosses (> 0.5*n distinct
        # cells, Layer 29 pre-screen) route through K-fold OOF target encoding
        # (Layer 33). Default OFF -> byte-identical legacy path. ``cat_cols``
        # empty => auto-detect categoricals.
        fe_cat_triple_enable: bool = False,
        fe_cat_triple_min_interaction_info: float = 0.001,
        fe_cat_triple_cat_cols: tuple = (),
        fe_cat_triple_beam_width: int = 3,
        fe_cat_triple_top_k: int = 3,
        # NUMERIC DECOMPOSITION FE. NVIDIA cuDF Kaggle-
        # Grandmaster technique #4: multi-precision rounding (round(x/p)*p for
        # p in precisions) + decimal-digit extraction (floor(x*10^k) mod 10).
        # Captures price-anchored step-function targets (rounding buckets) and
        # cents-digit / encoded-id-substructure signals (digit extraction) that
        # any monotone transform of raw x is blind to. The IT enhancement gates
        # each candidate by Layer 62 bootstrap-stable MI (lower CB): a
        # decomposition is kept only when its MI lower bound clears the raw
        # column's noise band, so smooth-target frames (where rounding is lossy
        # raw x and digits are pure noise) emit nothing. Default OFF ->
        # byte-identical legacy path. ``cols`` empty => all numeric columns.
        fe_numeric_decompose_enable: bool = False,
        fe_numeric_decompose_precisions: tuple = (1, 0.1, 0.01, 0.001),
        fe_numeric_decompose_digits: tuple = (0, 1, 2),
        fe_numeric_decompose_n_boot: int = 10,
        fe_numeric_decompose_top_k: int = 5,
        # PART A — PERIODIC / MODULAR decomposition FE
        # (extends Layer 90). For each (col, period) emit x mod period plus its
        # sin/cos phase encoding sin/cos(2*pi*(x mod period)/period). Captures
        # cyclic signals (hour-of-day, day-of-week, sensor cycles) that any
        # monotone transform of raw x is blind to; the sin/cos pair gives cyclic
        # continuity (phase 0 and period-eps are neighbours on the unit circle).
        # Each candidate gated by Layer 62 bootstrap-stable MI (lower CB), which
        # doubles as auto-period detection: the correct period's residue carries
        # the signal and survives, wrong periods scramble it into noise and fall
        # below the floor. Default OFF -> byte-identical legacy path. ``cols``
        # empty => all numeric columns.
        fe_modular_enable: bool = False,
        fe_modular_periods: tuple = (7, 12, 24, 30, 365),
        fe_modular_top_k: int = 6,
        # PAIRWISE / N-WAY MODULAR FE (extends the single-column modular path above).
        # Detects a target that is an integer MODULUS of a COMBINATION of columns -- (a+b) mod m,
        # (a*b) mod m, n-way parity (a+b+c) mod 2, or a single column's hidden non-calendar period --
        # which smooth bases (poly / Fourier) cannot fit (a sawtooth residue needs unbounded harmonics).
        # Cheap-first / escalate: a coarse modular scan gated by a permutation-null (only escalates after
        # the cheap gate responds, so a non-modular frame costs ~the scan and emits nothing), then a fine
        # modulus refine. Each responded detection becomes a frozen recipe (combine cols, mod m) replayed
        # leak-free at predict. DEFAULT-ON: the permutation-null gate makes it self-limiting -- on non-modular
        # data the cheap scan responds to nothing and emits zero columns, so the only cost is the bounded scan.
        # Wide-frame validation (_benchmarks/bench_pairwise_modular_wideframe.py) measured: zero false-positive
        # injection on pure-noise + smooth frames at p=30 over 3 seeds, real (a+b) mod 7 still caught amid 25
        # noise columns, and added wall-time within the budget guard (~+1.6s / ~+6% at p=30; ~free above 30).
        # Opt out with fe_pairwise_modular_enable=False for byte-identical legacy / replay.
        # BUDGET GUARD (cost is O(p) self-scan + budgeted pairs/triples; see bench_pairwise_modular_cost.py +
        # bench_pairwise_modular_wideframe.py): max_int_cols=30 skips the whole sweep above 30 integer-eligible
        # columns (above 30 the added cost is near-zero by construction); max_triple_cols=20 drops the C(p,3)
        # triple sweep above 20 cols (pairs-only). Both skips are logged, never silent. Triples stay ON within
        # the tighter triple budget because the bench shows the triple overhead is flat (~0.13s, capped by max_triples).
        #
        # DISCRETE STRUCTURAL FE OPERATORS — master switch. The four operators below (pairwise-modular, integer-lattice,
        # row-argmax, conditional-gate) each detect a target whose signal lives in a discrete / non-smooth structure that
        # the smooth + arithmetic basis catalog cannot express in one column: modular periods/parity, gcd/lcm grid alignment,
        # ordinal row-comparisons, and data-dependent regime switches respectively. All four are cheap-first (a bounded scan
        # gated by a best-existing-op margin + a permutation null, so a frame without that structure injects ZERO columns),
        # leak-free + bit-identical at predict (frozen recipes replayed as pure functions of X), budget-guarded on wide frames,
        # and work on classification AND regression (continuous y is quantile-binned for the class-MI floor; 2D y is skipped).
        # All default ON (measured downstream model lift, e.g. gcd +0.087 held-out AUC where a tree cannot form gcd(a,b)).
        # Set fe_discrete_structural_operators_enable=False to disable ALL FOUR at once (pure classical FE), regardless of the
        # individual flags; leave it True (default) and the per-operator fe_*_enable flags govern individually.
        fe_discrete_structural_operators_enable: bool = True,  # [ACCURACY-CAVEAT] False disables ALL FOUR discrete-structural FE families; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        fe_pairwise_modular_enable: bool = True,  # [ACCURACY-CAVEAT] False drops modular/parity/period interactions; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        fe_pairwise_modular_top_k: int = 4,
        fe_pairwise_modular_max_int_cols: int = 30,
        fe_pairwise_modular_max_triple_cols: int = 20,
        # PAIRWISE INTEGER-LATTICE FE (sibling of pairwise-modular). Detects a target that is a function of a hidden
        # COMMON DIVISOR (gcd(a,b) -- shared factor / grid alignment), its dual lcm(a,b), or a bit-level co-occurrence
        # of integer codes (a & b), which smooth bases + the existing arithmetic/modular ops cannot express (gcd is
        # number-theoretic, non-smooth, non-monotone in either argument). XOR is EXCLUDED as redundant with the modular
        # residue operator (measured lift ~0.09); only the three measured-distinct ops (gcd / lcm / bitwise_and) ship.
        # Cheap-first pairs-only scan gated by a dual test (the engineered column's MI must beat BOTH operands' raw MI
        # by a margin AND a 12-permutation null upper band), so a non-lattice frame injects nothing. Each responded
        # detection becomes a frozen recipe (cast both operands to int, take gcd/lcm/and) replayed leak-free at predict.
        # DEFAULT-ON (measured-safe): bench_integer_lattice_fe measured 6.97x MI lift on a gcd-shared-factor target with
        # 0 false-positive on smooth/noise controls; bench_integer_lattice_wideframe confirmed 0 FP at p=30 over 3 seeds,
        # signal still caught amid 25 noise int cols, and bounded added wall-time (pairs-only, cheaper than modular).
        # Opt out with fe_integer_lattice_enable=False for byte-identical legacy / replay. BUDGET GUARD: max_int_cols=30
        # skips the whole sweep above 30 integer-eligible columns (logged, never silent). No triple budget -- gcd/lcm/AND
        # are binary, so the sweep is pairs-only (O(C(p,2))), no n-way analogue.
        fe_integer_lattice_enable: bool = True,
        fe_integer_lattice_top_k: int = 4,
        fe_integer_lattice_max_int_cols: int = 30,
        # ROW-ARGMAX FE (frontier pass 2). Emits, for a column TRIPLE (a, b, c), the integer index 0/1/2 of the row-maximum --
        # an ordinal/comparison pattern the MI/linear path cannot read off marginals or pairwise diffs (a single shipped column
        # never equals the 3-way argmax code; measured +0.55 single-column MI lift over the best shipped op). ZERO free params,
        # detector-clean (negative lift on smooth/noise/ordinary-interaction controls; 0 FP at p=30 over 3 seeds in
        # bench_conditional_gate_wideframe), leak-free deterministic replay (np.argmax over the stacked source columns). Default ON;
        # opt out with fe_row_argmax_enable=False for byte-identical legacy/replay. BUDGET GUARD: max_cols=30 skips the whole
        # C(p,3) sweep above 30 eligible columns (logged, never silent).
        fe_row_argmax_enable: bool = True,  # [ACCURACY-CAVEAT] False drops the row-argmax interaction; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        fe_row_argmax_top_k: int = 4,
        fe_row_argmax_max_cols: int = 30,
        # CONDITIONAL-GATE FE (frontier pass 2). Emits a regime switch c>tau ? a : b (select) and a masked interaction 1[c>tau]*a
        # (mask): two raw features routed/masked by a THIRD column's data-dependent threshold tau (found by a ~17-point quantile
        # scan, FROZEN in the recipe for exact replay). No shipped op expresses a hard value-selecting switch (conditional_residual
        # is a residual, the hinge basis is univariate, a*c is a smooth surface); measured +0.55 select / +0.31 mask MI lift over
        # the best shipped op. HARDENED detector: the engineered MI must beat the BEST-EXISTING-OP MI on the same operands (max over
        # raw/product/ratio/diff/min/max), NOT just the raw single-operand floor -- this removes the prototype's false positives on
        # smooth/ordinary_mul (bench_conditional_gate_wideframe: 0 FP at p=30 over 3 seeds with the hardened floor).
        # DEFAULT ON (opt out via fe_conditional_gate_enable=False for byte-identical legacy/replay). The prior O(p^3) wide-frame cost
        # blow-up that forced this OFF is gone: the candidate set is RELEVANCE-PRUNED before the sweep. Operands a,b come from the
        # top-fe_conditional_gate_k_operand columns by raw MI vs y (the value the switch routes carries marginal relevance); the gate
        # column c comes from the top-fe_conditional_gate_k_gate by a CONDITIONAL-DIVERGENCE rank (how much a c>median split changes the
        # operand->y MI), because a pure regime switch's gate column can be marginally independent of y (raw-MI ranks it last). The
        # sweep is then O(k_operand^2 * k_gate * tau-scan), FLAT in p -- measured ~flat (~+10-15% of a full fit at n=20000, comparable
        # to modular/lattice/argmax) instead of +50%..+291%. k_operand=10 / k_gate=8 keep all 3 gate-synthetic seeds caught amid 25
        # noise; 0 FP at p=30 over 3 seeds (smooth/ordinary_mul/random) holds. BUDGET GUARD: max_cols=200 defense-in-depth outer cap.
        fe_conditional_gate_enable: bool = True,  # [ACCURACY-CAVEAT] False drops the conditional-gate interaction; see _param_accuracy_warnings.ACCURACY_SUBOPTIMAL
        fe_conditional_gate_top_k: int = 4,
        fe_conditional_gate_max_cols: int = 200,
        fe_conditional_gate_k_gate: int = 8,
        fe_conditional_gate_k_operand: int = 10,
        # PART B — PER-GROUP DISTRIBUTION-DISTANCE FE
        # (extends Layer 88). For each (group, num) emit how far the row's GROUP
        # distribution sits from the GLOBAL one: group-level z
        # ((group_mean-global_mean)/global_std), per-group KL divergence, and
        # per-group Wasserstein-1 distance, broadcast to rows. A group-anomaly
        # detector (rows in atypical groups flagged), orthogonal to Layer 88's
        # within-group percentile rank. Each survivor MI-gated against the source
        # num_col marginal MI. Default OFF -> byte-identical legacy path.
        # ``group_cols`` / ``num_cols`` empty => auto-detect.
        fe_group_distance_enable: bool = False,
        fe_group_distance_top_k: int = 6,
        fe_group_distance_group_cols: tuple = (),
        fe_group_distance_num_cols: tuple = (),
        # THREE new recipe-based FE families. All default
        # OFF -> byte-identical legacy path.
        #   A) rare-category indicator + frequency-band: a category being RARE is
        #      itself predictive (rare merchant = fraud risk). Emits is_rare_{col}
        #      + freq_band_{col}; MI-gated against the raw-baseline floor.
        #   B) NUM x NUM conditional residual x_i - E[x_i | bin(x_j)]: conditional
        #      anomaly (income high FOR this age bracket). Cardinality-bounded by
        #      top raw-MI columns; MI-gated.
        #   C) RankGauss (rank-Gaussianisation): rank -> normal quantile. Monotone
        #      so MI-invariant by the data-processing inequality -> NOT MI-gated;
        #      the value is DOWNSTREAM (linear / NN). Pool bounded by raw MI.
        # ``*_cols`` empty => auto-detect.
        fe_rare_category_enable: bool = False,
        fe_rare_category_cols: tuple = (),
        fe_rare_category_threshold: float = 0.01,
        fe_rare_category_top_k: int = 10,
        fe_conditional_residual_enable: bool = False,
        fe_conditional_residual_cols: tuple = (),
        fe_conditional_residual_n_bins: int = 10,
        fe_conditional_residual_top_k: int = 10,
        fe_conditional_residual_max_pair_cols: int = 6,
        # FAMILY D -- CONDITIONAL DISPERSION / 2nd-moment.
        # Bin x_j; per bin store conditional STD of x_i (Family B stores the mean);
        # emit the conditional z-score |z|=|(x_i-mu_hat_bin)/sigma_hat_bin| and the
        # squared anomaly z^2. Models conditional SCALE (volatility / dispersion
        # regimes) -- the gap Family B's conditional MEAN leaves. DEFAULT-ON: it is
        # MI-gateable (|z| is a NON-monotone fold -> genuine MI on heteroscedastic
        # targets, unlike the MI-invariant hinge/isotonic) and SELF-LIMITING (a
        # dual-uplift gate admits a column only when its MI beats BOTH raw x_i AND
        # the |mean-residual| Family-B sibling; on homoscedastic data |z| is a
        # scaled |residual| -> no uplift -> dropped; on the canonical pair-FE
        # fixture the noise floor + dual-uplift admit 0, so it does NOT perturb
        # genuine-feature recovery). Leak-safe replay (kind ``conditional_dispersion``)
        # stores x_j edges + per-bin (mu_hat, sigma_hat); replay is closed-form.
        fe_conditional_dispersion_enable: bool = True,
        fe_conditional_dispersion_cols: tuple = (),
        fe_conditional_dispersion_n_bins: int = 10,
        fe_conditional_dispersion_top_k: int = 10,
        fe_conditional_dispersion_max_pair_cols: int = 6,
        # CONDITIONAL QUANTILE-RANK: 4th member of the
        # conditional-dispersion family (grouped_agg mean/std -> composite_group_agg ->
        # conditional-dispersion z-score/|z| -> conditional quantile-rank). Bin x_j; emit
        # q(row) = empirical_rank(x_i within bin(x_j)) -- the row's TRUE within-bin percentile,
        # not a z-score. On a heavy-tailed/skewed conditional distribution a z-score badly
        # misrepresents "how extreme" a row is (mean/std is not a sufficient statistic for a
        # skewed shape); quantile-rank resolves it directly. MI-gateable and designed to be
        # SELF-LIMITING the same way the sibling conditional_dispersion family is (on a
        # homoscedastic, non-skewed conditional distribution quantile-rank is a near-monotone
        # reparametrization of the raw column and should clear no uplift). DEFAULT OFF for now,
        # unlike the sibling: conditional_dispersion earned its default-ON only after the existing
        # regression/biz_value/fuzz-combo suite (which already treats fe_conditional_dispersion_
        # enable as a toggle axis) validated it did not perturb genuine-feature recovery anywhere
        # -- this sibling has not yet been run through that same validation, so default OFF avoids
        # an unvalidated interaction with dozens of existing fixtures. Flip once validated the same
        # way. Leak-safe replay (kind ``conditional_quantile_rank``) stores x_j quantile edges + the
        # per-bin sorted x_i reference values; replay is closed-form searchsorted, no y.
        fe_conditional_quantile_rank_enable: bool = False,
        fe_conditional_quantile_rank_cols: tuple = (),
        fe_conditional_quantile_rank_n_bins: int = 10,
        fe_conditional_quantile_rank_top_k: int = 10,
        fe_conditional_quantile_rank_max_pair_cols: int = 6,
        # Bandt-Pompe ordinal-pattern K-fold target
        # encoding. Default OFF for the same reason as its sibling above -- brand-new, not yet
        # validated against the existing fuzz-combo/regression suite. Leak-safe replay (kind
        # ``ordinal_pattern_te``) recomputes the perm_id fresh from the raw K source columns and
        # looks up a frozen TE lookup table; no y reference is captured in the recipe.
        fe_ordinal_pattern_enable: bool = False,
        fe_ordinal_pattern_cols: tuple = (),
        fe_ordinal_pattern_k: int = 3,
        fe_ordinal_pattern_max_cols_for_tuples: int = 5,
        fe_ordinal_pattern_n_folds: int = 5,
        fe_ordinal_pattern_smoothing: float = 10.0,
        fe_ordinal_pattern_top_k: int = 5,
        # Random Fourier Features (random kitchen sinks)
        # joint kernel-approximation block. Default OFF -- brand-new, not yet validated against
        # the existing fuzz-combo/regression suite. Leak-safe replay (kind ``random_fourier``)
        # stores the frozen W-column/phase/bandwidth; no y reference is captured in the recipe.
        fe_random_fourier_enable: bool = False,
        fe_random_fourier_cols: tuple = (),
        fe_random_fourier_m: int = 64,
        fe_random_fourier_bandwidth: Optional[float] = None,
        fe_random_fourier_max_cols_for_block: int = 8,
        fe_random_fourier_top_k: int = 8,
        # Sliced Inverse Regression (SIR) oblique-direction
        # projection feature. Default OFF -- brand-new, not yet validated against the existing
        # fuzz-combo/regression suite. Leak-safe replay (kind ``sir_direction``) stores the frozen
        # centering x_mean + direction vector; y's effect is already baked into the frozen
        # direction at fit time, so the recipe itself carries no y reference.
        fe_sir_direction_enable: bool = False,
        fe_sir_direction_cols: tuple = (),
        fe_sir_direction_n_slices: int = 10,
        fe_sir_direction_n_directions: int = 2,
        fe_sir_direction_max_cols_for_block: int = 8,
        fe_sir_direction_top_k: int = 2,
        # Local Outlier Factor / k-NN local density-ratio
        # feature. Default OFF -- brand-new, not yet validated against the existing fuzz-combo/
        # regression suite. Leak-safe replay (kind ``lof_score``) stores a BOUNDED reference
        # sample (``fe_lof_max_ref`` rows, never the whole fit frame -- RAM discipline) + its
        # precomputed density internals; no y reference is captured in the recipe.
        fe_lof_enable: bool = False,
        fe_lof_cols: tuple = (),
        fe_lof_k: int = 20,
        fe_lof_max_ref: int = 2000,
        fe_lof_max_cols_for_block: int = 8,
        fe_lof_top_k: int = 1,
        # Multivariate Mahalanobis / Gaussian-copula joint
        # density anomaly score. Default OFF -- brand-new, not yet validated against the existing
        # fuzz-combo/regression suite. Leak-safe replay (kind ``mahalanobis_density``) stores the
        # frozen Ledoit-Wolf mu/Sigma_inv; no y reference is captured in the recipe.
        fe_mahalanobis_density_enable: bool = False,
        fe_mahalanobis_density_cols: tuple = (),
        fe_mahalanobis_density_max_cols_for_block: int = 20,
        fe_mahalanobis_density_top_k: int = 1,
        # HAAR WAVELET / localized multiresolution basis.
        # A NEW operator for LOCALIZED bump / multiscale piecewise structure the
        # catalog cannot capture: y jumps only inside a narrow sub-window of x, or
        # has step/contrast structure at several scales at once. The catalog has
        # the WRONG form -- Fourier is GLOBAL (Gibbs-rings a bump), spline knots
        # are FIXED quantiles (a bump between knots is smoothed away), rounding is
        # global. On x's support emit a SMALL dyadic set of Haar indicators
        # psi_{j,k} (+1 left half / -1 right half of a dyadic interval, 0 outside),
        # scales j=0..3. DEFAULT-ON: each leg is held-out-scale-selected (a
        # noise-aware MAD floor over candidate legs' held-out MIs + a max-legs cap
        # bound the candidate explosion) and then admitted on its held-out
        # INCREMENTAL MI over raw x AND a complementarity guard (it must beat a
        # SMOOTH location-refinement of x), so it is SELF-LIMITING: on a localized
        # step/bump the leg sharpens y beyond raw x -> admitted; on a SMOOTH (sin /
        # monotone) column the smooth refinement wins -> 0 legs (Fourier owns that
        # regime, complementary not redundant); on pure noise the held-out floor
        # rejects all -> 0 legs. Non-monotone leg -> MI-VISIBLE (unlike the
        # monotone hinge), so the gate is MI-based (conditioned on raw x). Leak-safe
        # replay (kind ``orth_wavelet``) stores (lo, span) + dyadic (j, k); replay
        # is the closed-form indicator, no y. Structurally like orth_spline.
        fe_wavelet_enable: bool = True,
        # Column-count cap on the wavelet held-out scale-selection. Profiled as the second-largest
        # default-ON pre-FE cost on a wide-p fit (~26% of the pre-categorize wall at p=420 -- the SAME
        # p=423 scale the MRMR audit's motivating production fit ran at), roughly linear in column count.
        # Unlike the Fourier adaptive cap, there is no cheap fallback here -- columns beyond the cap get NO
        # wavelet legs at all, so this default is deliberately more generous than the Fourier cap's floor
        # would suggest. Default 100 (2026-07-10, corrective-mechanism-on-by-default): bounds cost on
        # wide-p fits while preserving full legacy behaviour below it; the biz_val tests for this cap use
        # p<=8 so are unaffected. ``None`` restores the pre-2026-07-10 unlimited behaviour.
        fe_wavelet_max_cols: Optional[int] = 100,
        fe_wavelet_cols: tuple = (),
        fe_wavelet_max_scale: int = 3,
        fe_wavelet_max_legs: int = 6,
        fe_wavelet_top_k: int = 8,
        # RankGauss = rank -> Phi^-1(empirical CDF); leak-safe (TRAIN sorted values,
        # replay via searchsorted + extreme-clip). Stays default-OFF on purpose.
        # bench-rejected flipping it default-ON / adding a duplicate
        # qrank/qnorm univariate operator: it is REDUNDANT with the cubic-B-spline
        # quantile-knot block (the spline path is itself opt-in via
        # ``fe_hybrid_orth_enable`` / extra_bases, NOT default-on) -- with the spline
        # enabled, |corr(qnorm(x), y)| matches the spline-linear extract to ~3
        # decimals on heavy-tailed monotone x (exp/lognorm/pareto reg ~0.950); the
        # lone pareto-reg OOS lift was a fixed-alpha Ridge artifact (RidgeCV on the
        # spline block closes it). RankGauss DOES beat RAW x for a linear/NN
        # downstream (its existing biz-value test), so keep the opt-in knob -- just
        # don't make it default. (D:/Temp/item6_rank_transform_findings.md; the earlier
        # 05f062d7 commit msg's "default-on spline" wording was inaccurate -- spline is opt-in.)
        fe_rankgauss_enable: bool = False,
        fe_rankgauss_cols: tuple = (),
        fe_rankgauss_top_k: int = 10,
        # TEMPORAL LEAK-SAFE GROUPED AGGREGATIONS. Layer
        # 87 grouped aggregates compute the per-group statistic over the WHOLE
        # train fold; for time-series / transaction data that peeks at a row's
        # own future (the per-entity mean includes the entity's LATER rows),
        # inflating train-CV and collapsing the forward holdout. This layer
        # keys aggregations on a TIME column and only ever sees the strict past:
        # expanding stat over rows before the current one, rolling time-window
        # stat, and lag features. Each survivor MI-gated against y. Recipes
        # store the FIT-time per-entity sorted history so test rows compute
        # their stat against TRAIN history only -- no test-future peeking, no
        # train-label leak. Default OFF -> byte-identical legacy path.
        fe_temporal_agg_enable: bool = False,
        fe_temporal_agg_entity_cols: tuple = (),
        fe_temporal_agg_value_cols: tuple = (),
        fe_temporal_agg_time_col: str | None = None,
        fe_temporal_agg_stats: tuple = ("mean", "std", "count"),
        fe_temporal_agg_windows: tuple = (),
        fe_temporal_agg_lags: tuple = (1,),
        fe_temporal_agg_top_k: int = 10,
        # Artifact retention for cross-selector reuse. When True, after fit() the
        # estimator carries ``su_to_target_``, ``mi_to_target_``, ``cached_MIs``,
        # and (when ``retain_bins=True``) per-column binned arrays so a downstream
        # selector (e.g. ShapProxiedFS(precomputed=mrmr.export_artifacts()))
        # can skip its own univariate pre-screen. Default False keeps the legacy
        # memory footprint byte-identical.
        retain_artifacts: bool = False,
        retain_bins: bool = True,
        # incremental / streaming refit support via
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
        partial_fit_window: int | None = None,
        # META FE-RECOMMENDER "1-knob" mode. Default OFF
        # -> byte-identical legacy path (individual fe_*_enable defaults
        # untouched). When True, fit() fingerprints (X, y) BEFORE the FE stages
        # run and asks the Layer-99 rule recommender (built on the Layer-98
        # Param-Oracle) which master FE generators match the data shape, then
        # flips exactly those fe_*_enable flags ON for this fit -- so a user who
        # sets only ``fe_auto=True`` gets the int-as-cat -> grouped-agg,
        # cats -> count/freq/cat-pair, time+entity -> temporal, NaNs ->
        # missingness, heavy-tail -> hybrid-orth mapping automatically, without
        # reading 50 docstrings. Flags the user explicitly set to True are never
        # turned OFF (auto only ADDS recommended generators). The original flag
        # values are restored after fit so the constructor-arg semantics stay
        # stable across fits and pickling/clone round-trips.
        fe_auto: bool = False,
        # hidden
        stop_file: str = "stop",
        # content-addressable disk cache for the per-column adaptive bin-edge stage. ``None``
        # (default) disables. When set, ``per_feature_edges`` caches each column's edge array keyed
        # by (column-summary, method, kwargs, y-summary-when-supervised); re-fits on the same X+y
        # skip the per-column edge-builder. Most useful for hyperparam sweeps and ablations where
        # the binning input recurs verbatim across runs. See ``mlframe.utils.disk_cache``.
        cache_dir: str | None = None,
        # EMBEDDING / FREE-TEXT PASSTHROUGH (default ON). Columns whose cells are embedding vectors (list/ndarray) or long free-text cannot be MI-discretised; when
        # True they are detected at fit, EXCLUDED from the MI screen / FE candidate set, and PASSED THROUGH to the transform output unchanged so a downstream
        # learnable-embedding network (PyTorch-Lightning MLP / recurrent) and its ``_encode_emb_text_fit`` boundary encoder consume them end-to-end. Set False for
        # the legacy behaviour (such columns reach the discretiser and crash / mis-bin). ``*_detect_embeddings`` / ``*_detect_text`` independently gate the two kinds.
        embedding_passthrough: bool = True,
        embedding_passthrough_detect_embeddings: bool = True,
        embedding_passthrough_detect_text: bool = True,
        # USABILITY-AWARE MULTI-LIST SELECTION. After the pure-MI fit
        # (``support_``, byte-identical to today) optionally run a SECOND selection pass tuned for a
        # LINEAR / additive downstream. MI is rank-based and blind to linear usability, so the
        # pure-MI list can carry raw operands (c, d) without the engineered interaction (c*d) a
        # linear model needs; the usability pass re-selects from a fresh pool with a relevance that
        # blends MI with the held-out partial corr of each candidate with the RESIDUAL after the
        # selected features (see ``_usability_lists`` / ``_usability_aware_selection``). It exposes
        # ``support_linear_`` (w->1), ``support_universal_`` (blend) and ``support_nonlinear_``
        # (alias of ``support_``), each replayable via ``transform_usability(X, which=...)``.
        # OFF by default: the CV-MAE forward selection it runs costs seconds-to-minutes and must not
        # be charged to every fit; the suite turns it on and routes linear models to the linear list.
        usability_aware_lists: bool = False,
        usability_w_linear: float = 0.85,
        usability_w_universal: float = 0.5,
        usability_feature_dtype: type = np.float32,
        usability_max_base_features: int = 16,
        usability_pool_kwargs: dict | None = None,
        usability_greedy_kwargs: dict | None = None,
        # Multi-output (2D y: multilabel / multi-target regression). MRMR's greedy partial-gain machinery merges the target columns into one
        # joint target, and that merged target makes the lazy confirmation step drop the 2nd genuine feature even when the per-column MI is high.
        # 'union' (default): fit one single-target selector per output column (the 1D path, which is correct) and UNION the selected raw columns.
        # 'intersect': keep only columns selected for EVERY output column. None / 'joint': legacy merged-target behaviour (byte-identical to pre-2026-06-20).
        multioutput_strategy: Optional[str] = "union",
        # Nested config-dataclass alternative to the individual flat kwargs above. Purely ADDITIVE: every flat kwarg above keeps working
        # unchanged forever (this migration's blast radius -- ~50+ existing call sites -- rules out a
        # breaking rename). When a config IS passed, its fields override that cluster's flat defaults;
        # mirrors the existing ``cat_fe_config`` precedent (``cat_fe_state.CatFEConfig``). See
        # ``_mrmr_config_dataclasses.py`` for field definitions / pydantic validation.
        fast_search_config: Optional[FastSearchConfig] = None,
        stability_config: Optional[StabilitySelectionConfig] = None,
        synergy_config: Optional[SynergyRedundancyConfig] = None,
        group_aware_config: Optional[GroupAwareConfig] = None,
        dcd_config: Optional[DCDConfig] = None,
        hybrid_orth_config: Optional[HybridOrthConfig] = None,
    ):

        # assert isinstance(estimator, (BaseEstimator,))

        # sklearn contract: ``__init__`` MUST store every constructor argument UNMODIFIED so ``get_params``
        # round-trips and ``clone`` is a true copy. ``random_state``/``random_seed`` reconciliation is
        # therefore resolved LAZILY at fit time (``_effective_random_seed``), NOT here -- mutating the
        # locals before ``store_params_in_object`` made ``get_params`` echo the promoted value
        # (``random_state`` showing ``random_seed``) and re-emit the deprecation warning on every
        # ``clone`` of even a default-constructed estimator. The only thing done at construction time
        # is to WARN when the user actually passed a conflicting / deprecated argument.
        # ``n_jobs=-1``/``parallel_kwargs=None`` used to be resolved to concrete values HERE, before
        # ``store_params_in_object`` -- the exact bug this comment block already warned against for
        # ``random_state``/``random_seed``. That meant
        # ``self.n_jobs``/``self.parallel_kwargs`` never actually held the constructor's sentinel value:
        # ``get_params()['n_jobs']`` permanently showed a resolved core COUNT (not ``-1``), a pickled/
        # cloned estimator carried the ORIGINAL machine's core count forever instead of re-resolving on
        # the machine that unpickles/clones it, and ``set_params(**mrmr.get_params())`` on a fresh
        # instance reproduced a different effective value on different hardware. Resolution now happens
        # LAZILY at the point of use via ``_effective_n_jobs()``/``_effective_parallel_kwargs()``,
        # mirroring ``_effective_random_seed()``.
        # ``random_state`` (sklearn's name) is canonical; ``random_seed`` is a deprecated alias kept
        # for backward compatibility -- see ``_effective_random_seed``.
        if random_state is not None and random_seed is not None and random_seed != random_state:
            # This is a conflicting-VALUES notice (which of two
            # explicitly-passed args wins), not a pure API-deprecation notice -- DeprecationWarning is
            # filtered by default in many contexts (plain ``python script.py``, non-``__main__`` code),
            # so a genuinely actionable "you set two conflicting values" warning could go unseen. UserWarning
            # is not filtered by default. The pure "random_seed is deprecated" notice below (neither
            # conflicts, just an alias in use) stays DeprecationWarning.
            warnings.warn(
                "MRMR: both random_seed (deprecated) and random_state were set to different "
                f"values; using random_state={random_state} and ignoring random_seed={random_seed}. "
                "Prefer random_state.",
                UserWarning,
                stacklevel=2,
            )
        elif random_seed is not None and random_state is None:
            warnings.warn(
                "MRMR: random_seed is deprecated, use random_state instead (sklearn's naming).",
                DeprecationWarning,
                stacklevel=2,
            )

        # save params
        # postfix="" is required here: every attribute read in this class (hundreds of getattr(self, "x", default)
        # call sites, plus sklearn's get_params/set_params/clone) expects plain self.x, not pyutilz's own
        # store_params_in_object(postfix="_param_") default -- must be passed explicitly, not inherited.
        store_params_in_object(obj=self, params=get_parent_func_args(), postfix="")
        self.signature: tuple | str | None = None
        # Nested config-dataclass overrides: apply
        # AFTER store_params_in_object so a passed config wins over its cluster's individual flat
        # kwargs, which are already set on self by this point. self.fast_search_config etc. (the raw
        # config objects, possibly None) are themselves stored by store_params_in_object above like
        # any other ctor param -- this call only expands a non-None config's fields onto the matching
        # flat attrs; it does not touch self.fast_search_config etc. themselves.
        apply_mrmr_config_objects(
            self,
            fast_search_config=fast_search_config,
            stability_config=stability_config,
            synergy_config=synergy_config,
            group_aware_config=group_aware_config,
            dcd_config=dcd_config,
            hybrid_orth_config=hybrid_orth_config,
        )

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        # ``n_workers`` (candidate-MI evaluation parallelism; default 1 = SERIAL, the fast path) is hidden by sklearn's
        # repr because it equals its default, while ``n_jobs`` (still shown as its raw stored value, e.g. ``-1`` --
        # resolved lazily via ``_effective_n_jobs()``, not at construction) is easily misread as
        # "MI runs on n_jobs threads". n_jobs only drives CPU sub-helpers (permutation-null MI, wide-frame nbins edges),
        # each self-gated (pair-search forces serial under GPU-strict). Surface n_workers so the parallelism is unambiguous.
        # This textually patches BaseEstimator.__repr__'s output
        # on an untested assumption about its trailing format (a literal ")"-ending string) rather than a
        # documented public contract. Wrapped defensively so a future sklearn internals change that
        # breaks that assumption degrades to the PLAIN super().__repr__() (still correct, just missing
        # the extra n_workers= annotation) instead of raising out of a routine repr() call.
        r: str = super().__repr__(N_CHAR_MAX=N_CHAR_MAX)
        try:
            if "n_workers=" not in r and r.endswith(")"):
                _inner = r[:-1]
                _sep = "" if _inner.endswith("(") else ", "
                r = f"{_inner}{_sep}n_workers={getattr(self, 'n_workers', 1)})"
        except Exception as exc:
            logger.debug("mrmr: __repr__ n_workers annotation skipped (%r); falling back to the plain BaseEstimator repr.", exc)
        return r

    # Constructor-param validation allow-lists. Carved VERBATIM into the leaf module
    # ``_mrmr_param_constants.py`` (no class refs -> no cycle) and re-bound here as class
    # attributes so ``self._VALID_*`` resolution stays byte-identical (the validator in
    # ``_mrmr_validate_transform`` reads them off the instance). Kept module-private rather
    # than a typing.Literal alias so the runtime check can produce a richer error listing
    # the valid options (fix audit row FS-P2-1). See that sibling for the per-constant notes.
    _VALID_QUANTIZATION_METHODS = _VALID_QUANTIZATION_METHODS
    _VALID_NAN_STRATEGIES = _VALID_NAN_STRATEGIES
    _VALID_MRMR_RELEVANCE_ALGOS = _VALID_MRMR_RELEVANCE_ALGOS
    _VALID_MRMR_REDUNDANCY_ALGOS = _VALID_MRMR_REDUNDANCY_ALGOS
    _VALID_NBINS_STRATEGIES = _VALID_NBINS_STRATEGIES
    _VALID_MI_CORRECTIONS = _VALID_MI_CORRECTIONS
    _VALID_REDUNDANCY_AGGREGATORS = _VALID_REDUNDANCY_AGGREGATORS
    _VALID_STABILITY_SELECTION_METHODS = _VALID_STABILITY_SELECTION_METHODS
    _DEMOTED_NBINS_STRATEGIES = _DEMOTED_NBINS_STRATEGIES
    _VALID_FE_UNARY_PRESETS = _VALID_FE_UNARY_PRESETS
    _VALID_FE_BINARY_PRESETS = _VALID_FE_BINARY_PRESETS
    _VALID_CLUSTER_AGGREGATE_MODES = _VALID_CLUSTER_AGGREGATE_MODES
    _VALID_CLUSTER_AGGREGATE_METHODS = _VALID_CLUSTER_AGGREGATE_METHODS
    _VALID_DCD_DISTANCES = _VALID_DCD_DISTANCES
    _VALID_DCD_SWAP_METHODS = _VALID_DCD_SWAP_METHODS
    _VALID_RFECV_SELECTION_RULES = _VALID_RFECV_SELECTION_RULES
    _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS = _VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS

    # ``_validate_string_params`` + ``_validate_inputs`` are implemented
    # in ``_mrmr_validate_transform.py`` and bound onto this class at the
    # bottom of this module.

    # Keys whose ``__setstate__`` legacy-injection value INTENTIONALLY differs from the
    # live constructor default, to keep OLD pickles byte-identical on reload (master FE
    # switches default ON for new fits but must stay OFF / at the old contract when an
    # attribute-less legacy pickle is resurrected). These are NOT sourced from
    # ``_ctor_defaults()``; every other shared key IS, so it cannot drift.
    _SETSTATE_LEGACY_OVERRIDES = frozenset({
        "max_confirmation_cand_nbins",          # legacy 50; ctor None (adaptive)
        "fe_fallback_to_all",                   # legacy True; ctor False
        "mrmr_identity_cache_ycorr_threshold",  # legacy 0.0 (gate off); ctor 0.5
        "fe_pair_prewarp_enable",               # legacy OFF; ctor ON
        "fe_hybrid_orth_enable",                # legacy OFF; ctor ON
        "fe_hybrid_orth_triplet_enable",        # legacy OFF; ctor ON
        "fe_hybrid_orth_quadruplet_enable",     # legacy OFF; ctor ON
        "fe_kfold_te_enable",                   # legacy OFF; ctor ON
        "fe_conditional_dispersion_enable",     # legacy OFF; ctor ON
        "fe_wavelet_enable",                    # legacy OFF; ctor ON
    })

    @property
    def _fit_reentrancy_lock(self) -> threading.Lock:
        """Per-instance, lazily-created lock guarding against concurrent ``fit()`` calls on the SAME
        object. Not a real constructor param / fitted attribute -- stored
        under a private ``__dict__`` key and excluded from pickling via ``__getstate__`` below (a
        ``threading.Lock`` is not picklable); a fresh lock is lazily recreated after unpickle on first use."""
        lock = self.__dict__.get("_fit_reentrancy_lock_")
        if lock is None:
            lock = threading.Lock()
            self.__dict__["_fit_reentrancy_lock_"] = lock
        return lock

    def __getstate__(self):
        """Strip the non-picklable lazy re-entrancy lock (see ``_fit_reentrancy_lock`` above) before
        pickling; everything else follows ``BaseEstimator``'s default ``__dict__`` snapshot. Stamps
        ``_mrmr_schema_version``: before this, pickle
        compatibility was inferred PURELY from which ctor-param keys were absent from ``state`` (the
        ``_SETSTATE_LEGACY_DEFAULTS``/``_SETSTATE_LEGACY_OVERRIDES`` roster in
        ``_mrmr_setstate_defaults.py``) -- correct for an OLDER pickle loaded by NEWER code (every
        legacy key really is just "missing"), but silent for the inverse: a NEWER pickle (produced by a
        future mlframe) loaded by an OLDER installed mlframe (a deploy rollback) carries keys the older
        ``__init__`` never resolves/validates, and ``self.__dict__.update(state)`` sets them with no
        error. The version number itself doesn't prevent that (older code still can't understand a
        newer schema), but it lets ``__setstate__`` detect and WARN on the mismatch instead of silently
        misbehaving -- see the check in ``__setstate__``."""
        state = self.__dict__.copy()
        state.pop("_fit_reentrancy_lock_", None)
        state["_mrmr_schema_version"] = _MRMR_SCHEMA_VERSION
        return state

    def __setstate__(self, state):
        # MUST stay on the MRMR class body (not a mixin): it OVERRIDES BaseEstimator.__setstate__, so any
        # mixin placed after BaseEstimator in the MRO would be shadowed and this legacy-default injection
        # would silently never run on unpickle.
        # Downgrade detection: a pickle stamped with a NEWER schema version than this
        # installed mlframe understands is a real hazard the legacy-injection roster below cannot help
        # with (it only knows how to fill in what's MISSING, not what a newer/renamed key MEANS) -- warn
        # so the silent-misbehavior risk is at least visible, then proceed with the same best-effort
        # legacy-roster injection (there is no better fallback available). A pickle with no stamp at all
        # (pre-finding-#2) or an OLDER/EQUAL version is the normal, fully-supported case: no warning.
        _pickled_version = state.get("_mrmr_schema_version")
        if _pickled_version is not None and _pickled_version > _MRMR_SCHEMA_VERSION:
            warnings.warn(
                f"MRMR: unpickling a pickle with schema version {_pickled_version}, newer than this "
                f"installed mlframe's {_MRMR_SCHEMA_VERSION} (a deploy rollback / downgrade scenario). "
                "Any renamed/reinterpreted parameter this older version doesn't recognize will be set "
                "on the instance verbatim with no validation. Upgrade mlframe to match, or re-fit "
                "instead of unpickling.",
                UserWarning,
                stacklevel=2,
            )
        # Legacy-injection roster carved VERBATIM into ``_mrmr_setstate_defaults.py``;
        # ``build_setstate_defaults()`` returns a fresh deep copy each call so no two
        # unpickled instances alias a mutable default (the literal dict was re-executed
        # per call before; the deep copy preserves that exactly).
        defaults = build_setstate_defaults()
        # D5: source every shared ctor-param default from the SINGLE source of
        # truth (the constructor signature) so a setstate literal can never silently drift from
        # the ctor default. Documented legacy-pickle overrides (above) are exempt; setstate-only
        # keys (fitted attrs / legacy-only params not on the ctor) keep their explicit literals.
        _ctor = self._ctor_defaults()
        for k in list(defaults.keys()):
            # Only RE-SOURCE keys this dict already injects (preserve the exact legacy-injection
            # roster); never widen it by adding ctor params that setstate never injected.
            if k not in _ctor or k in self._SETSTATE_LEGACY_OVERRIDES:
                continue
            v = _ctor[k]
            # deep-copy mutable ctor defaults so unpickled instances never share a default object.
            defaults[k] = copy.deepcopy(v) if isinstance(v, (list, dict, set)) else v
        for k, v in defaults.items():
            state.setdefault(k, v)
        # P0 pickle BC: the hand-maintained roster above enumerates only a subset of ctor params, so a pickle
        # produced before ANY other ctor param existed re-surfaces without it -- and the fit path reads many
        # via bare ``self.<param>`` (e.g. ``self.dtype``), raising AttributeError before any work. Inject every
        # remaining ctor default the roster did not cover (roster keys + LEGACY_OVERRIDES already set above keep
        # their possibly-legacy-divergent values; the keys here are never overwritten once present in state).
        # Source the value from a FRESHLY-CONSTRUCTED instance, not the raw signature default, so any param
        # ``__init__`` DOES still resolve at construction time matches a fresh MRMR exactly -- a resurrected
        # legacy pickle then behaves identically to a new one (no ctor-vs-legacy drift). ``n_jobs``/
        # ``parallel_kwargs`` are no longer resolved at construction time (they are stored raw,
        # like every other ctor param, and resolved lazily via ``_effective_n_jobs()``/
        # ``_effective_parallel_kwargs()``), so for those two this now trivially matches the raw signature
        # default too -- the fresh-instance source stays authoritative for any FUTURE param with real
        # construction-time resolution.
        # Cached per class per process (see ``_resolve_fresh_instance_defaults``) so this pays the full ~300-param
        # constructor cost at most once per process, not on every single unpickle.
        _fresh_dict = type(self)._resolve_fresh_instance_defaults()
        for k, v in _ctor.items():
            if k in state:
                continue
            fv = _fresh_dict.get(k, v) if _fresh_dict is not None else v
            state[k] = copy.deepcopy(fv) if isinstance(fv, (list, dict, set)) else fv
        self.__dict__.update(state)

    @hygienic_fit
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | Any,
        y: pd.DataFrame | pd.Series | np.ndarray | Any,
        groups: pd.Series | np.ndarray | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        **fit_params,
    ):
        """Thin outer wrapper around ``_fit_body``: tracks the
        process-wide in-flight-fit count (gates the GPU circuit-breaker re-arm to the 0->1 transition
        so a concurrently-running fit's tripped breaker is never clobbered by another fit starting)
        and enforces the documented "no concurrent fit() on the SAME instance" contract via a
        non-blocking re-entrancy guard, before delegating to the real body. See ``_fit_body`` for the
        actual docstring/contract."""
        if not self._fit_reentrancy_lock.acquire(blocking=False):
            raise RuntimeError(
                "MRMR.fit() called concurrently on the SAME instance from two threads. Concurrent "
                "fit() on one MRMR object is not supported (sklearn convention: estimators are not "
                "thread-safe for concurrent fit) -- use a separate clone() per thread instead."
            )
        try:
            self._enter_active_fit_scope()
            try:
                return self._fit_body(X, y, groups=groups, sample_weight=sample_weight, **fit_params)
            finally:
                self._exit_active_fit_scope()
        finally:
            self._fit_reentrancy_lock.release()

    def _fit_body(
        self,
        X: pd.DataFrame | np.ndarray | Any,
        y: pd.DataFrame | pd.Series | np.ndarray | Any,
        groups: pd.Series | np.ndarray | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        **fit_params,
    ):
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

        Cross-target identity cache. When a prior fit on the SAME X (same columns + same dtypes) produced an identity result (all input columns selected + zero engineered features), subsequent calls with a different y short-circuit the 80+ min FE pipeline and return identity-equivalent output. Opt-in via ``mrmr_skip_when_prior_was_identity=True``."""
        # groups contract check and polars validate+bridge each moved
        # verbatim to a named helper on _MRMRFitHelpersMixin (see their docstrings for the original
        # rationale) -- zero behavior change, pure extraction. The GPU-breaker re-arm now happens in the
        # outer ``fit()`` wrapper's ``_enter_active_fit_scope()``, gated to
        # the 0->1 in-flight-fit transition instead of running unconditionally on every call here.
        #
        # Pre-override ctor-params snapshot (bug found while testing the re-entrancy guard): the in-object "identical refit -> skip" signature
        # (``_fit_impl_core.py``'s ``_self_params_sig``) used to be computed from ``self.get_params()``
        # READ INSIDE ``_fit_impl`` -- i.e. AFTER this method's OWN below-here overrides (cluster_
        # aggregate_enable, fast-search profile, default-screen-subsample, etc.) had already flipped
        # several ctor-param-named attributes to a TRANSIENT mid-fit value. The signature got stored
        # with that transient value baked in, but the ``finally`` block restores the true ctor value
        # before returning -- so a SUBSEQUENT identical fit()'s freshly-read params (post-restore) could
        # NEVER match the stored signature for any config whose override actually fires (e.g. the
        # DEFAULT ``cluster_aggregate_enable=True``), permanently defeating the same-content skip
        # optimization for the common case and forcing a full re-fit (with a different tie-break outcome
        # on near-equal-gain features) on every "identical" refit. Snapshotting the STABLE, pre-override
        # params here and threading it into ``_fit_impl`` (which prefers it over a fresh
        # ``self.get_params()`` read) fixes the signature to always reflect the user-visible ctor state.
        self._pre_fit_ctor_params_snapshot_ = self.get_params(deep=True)
        # gt_07 FE-family budget: load a persisted per-family budget (keyed by dataset fingerprint)
        # and scale the triplet/quadruplet/adaptive-arity seed_k/top_count quotas by it -- a lower
        # budget fraction means fewer candidates proposed for that family this fit. Snapshot the
        # ORIGINAL ctor values first (restored in the ``finally`` block below) so a crashing fit
        # never leaves the instance with a scaled-down quota baked in for a later ``get_params()``.
        _fe_budget_quota_snapshot: dict[str, int] = {}
        _fe_budget_setting = getattr(self, "fe_budget_learning", False)
        _fe_budget_learning_effective = bool(_fe_budget_setting)
        _fe_loaded_budgets: Optional[dict[str, float]] = None
        if isinstance(_fe_budget_setting, str):
            if _fe_budget_setting != "auto":
                raise ValueError(f"fe_budget_learning must be True, False, or 'auto'; got {_fe_budget_setting!r}")
            try:
                from .._fe_family_budget import dataset_fingerprint as _fe_budget_fp, load_budgets as _fe_load_budgets

                _fe_budget_cols = list(X.columns) if hasattr(X, "columns") else [str(i) for i in range(np.asarray(X).shape[1])]
                self._fe_budget_fingerprint_ = _fe_budget_fp(len(_fe_budget_cols), _fe_budget_cols)
                _fe_loaded_budgets = _fe_load_budgets(fingerprint=self._fe_budget_fingerprint_)
                # "auto" fires ONLY when a previous explicit opt-in already persisted a budget for this
                # exact dataset fingerprint -- otherwise it is a strict no-op, identical to False.
                _fe_budget_learning_effective = _fe_loaded_budgets is not None
            except Exception as _fe_budget_probe_exc:
                logger.warning("[MRMR] fe_budget_learning='auto': cache probe failed (%s); treating as disabled this fit.", _fe_budget_probe_exc)
                _fe_budget_learning_effective = False
        if _fe_budget_learning_effective:
            try:
                from .._fe_family_budget import dataset_fingerprint as _fe_budget_fp, load_budgets as _fe_load_budgets

                _fe_budget_cols = list(X.columns) if hasattr(X, "columns") else [str(i) for i in range(np.asarray(X).shape[1])]
                self._fe_budget_fingerprint_ = _fe_budget_fp(len(_fe_budget_cols), _fe_budget_cols)
                if _fe_loaded_budgets is None:
                    _fe_loaded_budgets = _fe_load_budgets(fingerprint=self._fe_budget_fingerprint_)
                # ``_FE_FAMILY_WALL`` (``_fe_family_timing.py``) is PROCESS-GLOBAL by design (nested
                # fits / composite-discovery passes accumulate into it so a whole-run summary via
                # ``log_fe_family_summary()`` reflects the whole suite) -- it is never reset here
                # (that would break other, unrelated consumers of the same ledger). Instead, snapshot
                # it now so the post-fit block below can compute THIS FIT's own wall-time delta
                # (post-fit snapshot minus this one), not the process-cumulative total across every
                # fit since process start (which would make credit/wall ROI increasingly wrong the
                # longer a training service has been running -- verified: without this delta, ROI
                # after several fits in the same process was dominated by stale history and the
                # learned budget stopped changing at all).
                from .._fe_family_timing import get_fe_family_wall as _fe_get_wall_pre

                self._fe_budget_wall_pre_fit_ = _fe_get_wall_pre()
                _fe_budget_quota_attrs = {
                    "triplet": ("fe_hybrid_orth_triplet_seed_k", "fe_hybrid_orth_triplet_top_count"),
                    "quadruplet": ("fe_hybrid_orth_quadruplet_seed_k", "fe_hybrid_orth_quadruplet_top_count"),
                    "adaptive_arity": ("fe_hybrid_orth_adaptive_arity_seed_k",),
                }
                _fe_equal_share = 1.0 / max(len(_fe_budget_quota_attrs), 1)
                # Stash the base budget used THIS fit (loaded, or equal-split when nothing was
                # persisted yet) so the post-fit reallocation block below compounds on top of it
                # instead of restarting from equal-split every fit (which would make learning never
                # accumulate across successive fits).
                self._fe_budget_prev_ = dict(_fe_loaded_budgets) if _fe_loaded_budgets else {f: _fe_equal_share for f in _fe_budget_quota_attrs}
                if _fe_loaded_budgets:
                    for _fam, _attrs in _fe_budget_quota_attrs.items():
                        _fam_fraction = _fe_loaded_budgets.get(_fam, _fe_equal_share)
                        _fam_scale = _fam_fraction / _fe_equal_share  # 1.0 at equal-split; <1 shrinks, >1 grows
                        for _attr in _attrs:
                            _orig_val = int(getattr(self, _attr, 0) or 0)
                            _fe_budget_quota_snapshot[_attr] = _orig_val
                            setattr(self, _attr, max(1, round(_orig_val * _fam_scale)))
                    logger.info("[MRMR] fe_budget_learning: applied loaded budgets %s (fingerprint=%s).", _fe_loaded_budgets, self._fe_budget_fingerprint_)
            except Exception as _fe_budget_exc:
                logger.warning("[MRMR] fe_budget_learning: pre-fit budget load/scale failed (%s); proceeding with unscaled quotas.", _fe_budget_exc)
        self._check_groups_contract(groups)
        self._pandas_frame_for_target_cleanup = None
        self._target_names_for_cleanup = None
        X, y = self._validate_and_bridge_polars_input(X, y)

        # ACCURACY-CAVEAT WARNING. Surface (once) any parameter value that is valid but KNOWN to degrade
        # selection accuracy (an explicit opt-out of an on-by-default accuracy mechanism, or a numeric
        # knob pinned to a documented-bad setting). Silent on a default config. See
        # ``_param_accuracy_warnings.ACCURACY_SUBOPTIMAL`` -- the single source of truth for the
        # ``# [ACCURACY-CAVEAT]`` markers on the flagged constructor parameters.
        try:
            warn_accuracy_suboptimal_params(self)
        except Exception as exc:
            logger.debug("mrmr: accuracy-suboptimal param warning failed (diagnostic only): %r", exc, exc_info=True)  # a diagnostic warning must never break a fit

        # Stability-selection outer-loop short-circuit.
        # When ``stability_selection_method`` is 'cluster' or
        # 'complementary_pairs', delegate to the bootstrap aggregator before
        # the legacy single-fit body executes.
        _stab_method = getattr(self, "stability_selection_method", "classic")
        if _stab_method not in ("classic", "cluster", "complementary_pairs"):
            raise ValueError(f"stability_selection_method must be 'classic', 'cluster', or 'complementary_pairs'; got {_stab_method!r}.")
        if _stab_method != "classic":
            try:
                _stab_result = self._stability_outer_fit(
                    X, y, groups=groups, sample_weight=sample_weight,
                    **fit_params,
                )
            except Exception as _exc:
                warnings.warn(
                    f"MRMR stability_selection_method={_stab_method!r} outer-loop raised {type(_exc).__name__}: {_exc}. "
                    "Falling back to classic fit. This is usually a data-shape issue (too few rows for "
                    "stability_n_bootstrap subsamples, or a column that can't enter the correlation "
                    "clustering step) rather than a bug -- check n_rows / stability_n_bootstrap if it recurs.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                if _stab_result is not None:
                    return _stab_result

        # Reject NaN/Inf in y at fit entry, matching the sibling selectors
        # (RFECV / ShapProxiedFS): NaN flowing into the MI scorer silently
        # degrades relevance numbers. Skip the check on object-dtype y
        # (categorical labels) where np.isnan would raise; numeric / float /
        # int paths get validated.
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

        # Multi-output (2D y) opt-in. MRMR's merged-target greedy under-selects the 2nd genuine feature on a 2D y (the lazy confirmation step
        # drops it even though per-column MI is high), so fit one single-target selector per output column (the correct 1D path) and aggregate.
        _mo_strategy = getattr(self, "multioutput_strategy", "union")
        if _mo_strategy not in (None, "joint", "union", "intersect"):
            raise ValueError(f"multioutput_strategy must be None, 'joint', 'union', or 'intersect'; got {_mo_strategy!r}.")
        # 09_error_messages_ux.md: 'joint' is intentionally EQUIVALENT to None here (both fall through to
        # the legacy merged-target path below, per the ctor docstring: "None / 'joint': legacy
        # merged-target behaviour, byte-identical to pre-2026-06-20") -- not a validation-accepts-but-
        # runtime-ignores gap. Only 'union'/'intersect' route through the per-column multioutput path.
        if _mo_strategy in ("union", "intersect") and _mrmr_y_is_multioutput(y):
            return self._fit_multioutput(X, y, groups, sample_weight, _mo_strategy, fit_params)

        # DEGENERATE-COLUMN DIAGNOSTIC SURFACE. Cheap O(p) scan of the INPUT frame
        # for pathological columns (all-NaN / constant / exact-duplicate / perfectly-
        # collinear) recorded into ``degenerate_columns_`` (column -> reason). PURELY
        # DIAGNOSTIC: the existing relevance gate already drops all-NaN/constant columns
        # (MI ~ 0) and the conditional-MI redundancy gate already drops duplicate /
        # collinear columns, byte-identically -- this scan does NOT remove columns or
        # alter the selection. It mirrors the sibling selectors' diagnostic attributes
        # so a downstream report / UI can SEE what the frame contained. Wrapped so a
        # diagnostic failure can never break a fit that would otherwise succeed.
        try:
            self.degenerate_columns_ = audit_degenerate_columns(X)
        except Exception as exc:
            logger.debug("mrmr: degenerate-column audit failed (diagnostic only): %r", exc, exc_info=True)
            self.degenerate_columns_ = {}

        # #2 cross-target identity cache.
        _identity_skip = bool(getattr(self, "mrmr_skip_when_prior_was_identity", False))
        _include_y = bool(getattr(self, "mrmr_identity_cache_include_y", False))
        # Suite caller (train_mlframe_models_suite) can inject a ctx-scoped dict here via
        # ``_mlframe_identity_cache_override_`` so cache lifetime is bounded by the suite
        # rather than the process. When absent, fall back to the module-level dict for
        # cross-suite reuse (CI matrices opt in via mrmr_identity_cache_scope="process").
        # Entries are legacy bool OR (is_id, prior_y_sample) tuples (see the store below); the module-level
        # dict is declared dict[str, bool] for its legacy shape, so widen the local view to match actual usage.
        _cache_dict: Optional[dict] = getattr(self, "_mlframe_identity_cache_override_", None)
        if _cache_dict is None:
            _cache_dict = cast(dict, _MRMR_IDENTITY_FP_CACHE)
        _x_fp = None
        if _identity_skip:
            _x_fp = _mrmr_compute_x_fingerprint(X)
            if _include_y:
                # T3#18: stricter cache key -- include y-fingerprint so legitimately distinct targets on same X get separate slots.
                _x_fp = _x_fp + "_yfp_" + _mrmr_compute_y_fingerprint_sample(y)
            with _MRMR_IDENTITY_FP_LOCK:
                _prior_entry = _cache_dict.get(_x_fp)
            # Entry is either a legacy bool or the (is_id, prior_y_sample) tuple stored below.
            if isinstance(_prior_entry, tuple):
                _prior_was_identity, _prior_y_sample = _prior_entry
            else:
                _prior_was_identity, _prior_y_sample = _prior_entry, None
            # A refit of this SAME instance on the exact (X, y) pair that produced this cache entry
            # is not a cross-target reuse -- it's a self-refit, which _fit_impl's own signature /
            # _FIT_CACHE shortcuts already handle precisely (replaying the TRUE fitted support_ order
            # and mrmr_gains_). Taking the coarse identity-shortcut here instead would silently replace
            # the real MI-ranked selection order with raw arange(n_cols) and zero out mrmr_gains_/
            # provenance_ -- caught live via a bit-identical-refit regression test.
            _is_self_refit = _x_fp == getattr(self, "_own_last_identity_fp_", None)
            if _prior_was_identity is True and not _is_self_refit:
                _ycorr_thr = float(getattr(self, "mrmr_identity_cache_ycorr_threshold", 0.0) or 0.0)
                _ycorr_ok = True
                _measured_corr = None
                if _ycorr_thr > 0.0:
                    if _prior_y_sample is not None:
                        _measured_corr = _mrmr_y_corr(_mrmr_y_corr_sample(y), _prior_y_sample)
                        # NaN corr (constant sample / mismatched length) is treated as "cannot confirm" -> refuse.
                        _ycorr_ok = _measured_corr is not None and abs(_measured_corr) >= _ycorr_thr
                    else:
                        # The user asked for the y-correlation safety gate (thr > 0) but the cached entry is the
                        # legacy bool format with no prior y-sample to check against -- we cannot confirm the new
                        # target matches the one that produced the cached identity selection. Refuse the shortcut
                        # and run a full fit rather than emit a selection that never saw this y.
                        _ycorr_ok = False
                if _ycorr_ok:
                    logger.info(
                        "[MRMR] cross-target identity cache HIT for X fingerprint=%s (y-corr=%s, thr=%.3g) -- "
                        "prior fit returned identity, skipping ~minute(s) of FE pipeline.",
                        _x_fp, ("%.3f" % _measured_corr) if _measured_corr is not None else "n/a", _ycorr_thr,
                    )
                    self._fit_identity_shortcut(X)
                    self._fit_sample_weight_ = None
                    self._identity_cache_ycorr_ = _measured_corr
                    return self
                logger.info(
                    "[MRMR] cross-target identity cache candidate REFUSED for X fingerprint=%s: "
                    "|y-corr|=%.3f < threshold %.3g; running a full fit for this distinct target.",
                    _x_fp, abs(_measured_corr) if _measured_corr is not None else float("nan"), _ycorr_thr,
                )

        # Persist user-supplied weights so cached _cat_fe_state_ / FE replay can introspect; cache key
        # below already excludes weights so the FS-cache reuse contract stays intact when the suite
        # caller decides to gate weight-aware MRMR behind FeatureSelectionConfig.use_sample_weights_in_fs.
        self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        X, y = self._maybe_resample_for_sample_weight(X, y, self._fit_sample_weight_)

        # INPUT-MUTATION ISOLATION (P1, 2026-06-11): ``_fit_impl`` injects temporary
        # ``targ_*`` columns into the working pandas frame (X.loc[:, target_names] = ...)
        # AND the FE pipeline appends engineered columns in place (X[name]=..., pd.concat
        # rebinds, hinge/cat-FE generators). The targ_* injection is reversed in the finally
        # below, but the engineered FE columns were NEVER removed -- so a caller-supplied
        # DataFrame came back with ['a','b'] -> ['a','b','a__relu_gt...',...] permanently
        # appended. This breaks the sklearn fit-must-not-mutate-input contract and, worse,
        # silently corrupts a frame REUSED across fits: the 2nd fit's FE builds on the 1st
        # fit's leaked columns -> different post-FE X content -> the _FIT_CACHE / replay
        # signature misses (wrong selection / no cache hit). Surfaced by
        # test_replay_fitted_state_isolation.py (A then B fit on the SAME X: A's leaked FE
        # cols made B's content hash differ -> no cache replay -> arrays not shared/frozen).
        #
        # FIX AT THE BOUNDARY (not in the FE-step machinery, which is being actively re-split):
        # operate on an INTERNAL copy so every downstream append lands on our copy and the
        # caller's frame is never touched. Polars input is immutable (with_columns returns a
        # new frame), numpy arrays carry no column index -- neither is mutated by our code, so
        # only pandas needs isolating.
        #
        # ALWAYS a SHALLOW copy (deep=False), regardless of pandas' Copy-on-Write setting
        # a deep copy is a real O(n*p) alloc+memcpy of the
        # whole (possibly SIS-reduced) frame, on EVERY fit, and pandas < 3.0's default IS
        # CoW-off -- so this was not a rare edge case, it was the common path for most installed
        # pandas versions. The deep-copy branch existed only as a defensive fallback for "a
        # shallow copy could write through an existing-cell mutation" -- but every internal
        # mutation site (this boundary's own ``targ_*`` injection at ``X.loc[:, target_names] =
        # ...``, and every FE stage: ``X[name] = ...``, ``pd.concat`` rebinds, hinge/cat-FE
        # generators) only ever ADDS a NEW column key, never overwrites an EXISTING column's
        # cell values in place. A pandas ``DataFrame.copy(deep=False)`` shares the existing
        # columns' underlying arrays with the caller's frame, but assigning a NEW column key
        # allocates a fresh block on the COPY's own BlockManager -- it cannot write through to
        # the original's shared arrays regardless of CoW, because no code path here ever
        # mutates an EXISTING column's values in place. Verified empirically (not just by
        # code-reading) with CoW forced off: fit an FE-heavy MRMR on a plain pandas frame,
        # confirm the caller's original frame is byte-identical after fit (columns + values +
        # dtypes), across test_mrmr_input_not_mutated.py, test_replay_fitted_state_isolation.py,
        # and the broader fe/ suite -- all pass under a forced-shallow copy. Copy ONCE here, not
        # per FE step.
        # SIS FRONT GATE (Gate A) dispatch. When the frame is at/above ``sis_screen_threshold`` columns, run
        # the chunked O(p*n) screen (filters/_mrmr_sis_screen.sis_screen) to cut the pool to a few thousand
        # data-derived survivors BEFORE the super-linear MRMR machinery. Fastest-default dispatch, not opt-in.
        # Subsetting X to survivor columns (pandas/polars/numpy) keeps the rest of fit unchanged; the screen
        # is best-effort -- any failure falls through to the full path.
        try:
            _sis_thr = int(getattr(self, "sis_screen_threshold", 0) or 0)
            _p_in = int(X.shape[1]) if hasattr(X, "shape") and getattr(X, "ndim", 1) > 1 else 0
            if _sis_thr and _p_in >= _sis_thr:
                X = self._apply_sis_screen(X, y)
        except Exception as _sis_exc:
            warnings.warn(
                f"MRMR SIS front gate raised {type(_sis_exc).__name__}: {_sis_exc}; falling back to the "
                "full-width MRMR path (safe -- the screen is a fast-path optimisation, not a correctness "
                "requirement; the fit still runs, just without the pre-screen speedup).",
                UserWarning,
                stacklevel=2,
            )

        if isinstance(X, pd.DataFrame):
            X = X.copy(deep=False)

        # fe_auto "1-knob" mode. BEFORE the FE stages run,
        # ask the rule recommender which master FE generators match this (X, y)
        # data shape and flip exactly those fe_*_enable flags ON for this fit
        # (auto only ADDS; a flag the user already set True is left True). The
        # ORIGINAL values are captured here and restored in the finally block so
        # constructor-arg semantics stay stable across fits / pickling / clone.
        _fe_auto_restore: dict = {}
        if bool(getattr(self, "fe_auto", False)):
            try:
                _rec_flags = recommend_fe_flags_by_rules(X, y)
                for _flag, _on in _rec_flags.items():
                    if _on and not bool(getattr(self, _flag, False)):
                        _fe_auto_restore[_flag] = getattr(self, _flag, False)
                        setattr(self, _flag, True)
                # Persist the recommender's CHOSEN flags as a fit-only attribute so
                # ``explain_selection()`` can narrate WHICH fe_* generators Layer-99 turned
                # on for this fit. Pure metadata; the live flags are still restored in the
                # finally block, so constructor-arg semantics stay stable.
                self._fe_recommended_flags_ = dict(sorted(_rec_flags.items()))
                if _fe_auto_restore:
                    # 09_error_messages_ux.md: fe_auto=True is a behavior-ALTERING opt-in (it turns on FE
                    # generators the caller didn't explicitly request) -- logger.info alone is invisible
                    # to a plain-script caller with default logging. Pair it with the same guaranteed-
                    # visible warnings.warn channel the module already uses for "your setting was
                    # silently overridden" situations (groups, group_aware_mi).
                    logger.info(
                        "[MRMR] fe_auto=True -> enabled FE generators for this fit: %s",
                        sorted(_fe_auto_restore),
                    )
                    warnings.warn(
                        f"MRMR.fit: fe_auto=True enabled these FE generator(s) for this fit: "
                        f"{sorted(_fe_auto_restore)}. Pass fe_auto=False or set the fe_*_enable flags "
                        "explicitly to silence this warning.",
                        UserWarning,
                        stacklevel=2,
                    )
            except Exception as _exc:
                warnings.warn(
                    f"MRMR fe_auto: rule recommender raised {type(_exc).__name__}: {_exc}. Proceeding "
                    "with the explicitly-set fe_*_enable flags (safe -- fe_auto is a convenience "
                    "recommender, not a correctness requirement).",
                    UserWarning,
                    stacklevel=2,
                )
                _fe_auto_restore = {}

        # activate thread-local SU normalization when mi_normalization='su'.
        # The toggle is read by evaluation.py / Fleuret loops at the scoring site so
        # raw conditional_mi (and cached entropy numbers) stay legacy-bit-stable for
        # the default ``mi_normalization='none'`` path. Restored in finally so a
        # crashing _fit_impl can't leak SU mode into subsequent fits.
        # Snapshot the MI thread-locals at fit ENTRY so the ``finally`` can restore the caller's values
        # instead of hardcoded literals: a nested / outer MRMR fit (the worker path already does this in
        # _evaluation_driver.py) must not have its toggles clobbered to False/0.0 when an inner fit exits.
        _toggles_snapshot = (
            use_su_normalization(), use_jmim_aggregator(), get_bur_lambda(),
            use_mi_miller_madow(), get_relaxmrmr_alpha(), get_pid_synergy_bonus(),
            get_cmi_perm_stop(), get_cpt_test(), use_mi_chao_shen(),
        )
        def _restore_toggles_snapshot_and_raise(exc: BaseException) -> NoReturn:
            """Restore the MI-correction thread-locals to their fit-entry snapshot then re-raise ``exc``.

            The validation checks below can fire AFTER some of these thread-locals have already been
            activated but BEFORE the protective try/finally further down starts -- without this, a raised
            ValueError here would leave the corrupted thread-local state active for every subsequent,
            unrelated fit on this thread.
            """
            _su0e, _jmim0e, _bur0e, _mm0e, _relax0e, _pid0e, _cmi0e, _cpt0e, _cs0e = _toggles_snapshot
            _safe_restore(lambda: set_su_normalization(_su0e), "SU normalization thread-local (activation-block exception)")
            _safe_restore(lambda: set_jmim_aggregator(_jmim0e), "JMIM aggregator thread-local (activation-block exception)")
            _safe_restore(lambda: set_bur_lambda(_bur0e), "BUR lambda thread-local (activation-block exception)")
            _safe_restore(lambda: set_mi_miller_madow(_mm0e), "Miller-Madow thread-local (activation-block exception)")
            _safe_restore(lambda: set_mi_chao_shen(_cs0e), "Chao-Shen thread-local (activation-block exception)")
            # Currently dormant (relaxmrmr_alpha/pid_synergy_bonus/cmi_perm_stop/cpt_test are only
            # activated further below, strictly after both call sites of this helper) -- restored anyway
            # so a future reordering that moves those activations earlier, or a new raise point added
            # between them and the protective try/finally, cannot silently reintroduce the same
            # thread-local-leak bug class this helper exists to prevent.
            _safe_restore(lambda: set_relaxmrmr_alpha(_relax0e), "RelaxMRMR alpha thread-local (activation-block exception)")
            _safe_restore(lambda: set_pid_synergy_bonus(_pid0e), "PID synergy-bonus thread-local (activation-block exception)")
            _safe_restore(lambda: set_cmi_perm_stop(_cmi0e[0], _cmi0e[1], _cmi0e[2]), "CMI-perm-stop thread-local (activation-block exception)")
            _safe_restore(lambda: set_cpt_test(_cpt0e[0], _cpt0e[1]), "CPT-test thread-local (activation-block exception)")
            raise exc

        _mi_norm = getattr(self, "mi_normalization", "none")
        if _mi_norm not in ("none", "su"):
            raise ValueError(f"MRMR.mi_normalization must be 'none' or 'su'; got {_mi_norm!r}.")
        _prev_su = _mi_norm == "su"
        set_su_normalization(_prev_su)
        # activate JMIM aggregator + BUR weight thread-locals.
        # Both default OFF (redundancy_aggregator=None, bur_lambda=0.0) so the
        # legacy Fleuret path stays bit-stable.
        _redundancy_agg = getattr(self, "redundancy_aggregator", None)
        if _redundancy_agg not in (None, "jmim", "auto"):
            # A typo (e.g. 'JMIM', 'jimm') would otherwise silently fall through to plain Fleuret with no signal
            # that the requested aggregator was ignored -- fail loudly instead.
            _restore_toggles_snapshot_and_raise(ValueError(f"redundancy_aggregator must be one of None, 'jmim', 'auto'; got {_redundancy_agg!r}."))
        if _redundancy_agg == "auto":
            # Data-dependent gate: run a cheap pre-fit synergy probe on (X, y). Route to JMIM only when the
            # data is synergistic (XOR / sign-product pairs whose joint >> marginals); else stay plain
            # Fleuret so the additive-regime over-selection that keeps 'jmim' opt-in cannot regress. The
            # detector threshold is a multiple of a label-permuted null scale read from kernel_tuning_cache
            # (data-derived, no hardcoded magic). Decision recorded on a fit-only attr for explain/logging.
            _jmim_on = False
            try:
                _Xarr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
                _yarr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                _jmim_on, _syn_info = detect_synergy(_Xarr, _yarr, random_seed=int(self._effective_random_seed() or 0))
                self._synergy_auto_decision_ = {"jmim_engaged": bool(_jmim_on), "detector_failed": False, **_syn_info}
                logger.info("[MRMR] redundancy_aggregator='auto' -> synergy detector: %s", self._synergy_auto_decision_)
            except Exception as _exc:
                warnings.warn(
                    f"MRMR redundancy_aggregator='auto': synergy detector raised {type(_exc).__name__}: {_exc}. "
                    "Falling back to plain Fleuret (safe -- the detector is a data-dependent gate, not a "
                    "correctness requirement; set redundancy_aggregator='jmim' to force JMIM regardless).",
                    UserWarning,
                    stacklevel=2,
                )
                # ``detector_failed`` is the explicit, documented way to tell "detector crashed"
                # apart from "detector ran and judged the data non-synergistic" -- both otherwise
                # look identical (jmim_engaged=False) to a caller who only checks that one key.
                self._synergy_auto_decision_ = {"jmim_engaged": False, "detector_failed": True, "error": str(_exc)}
        else:
            _jmim_on = _redundancy_agg == "jmim"
        _bur_lambda = float(getattr(self, "bur_lambda", 0.0) or 0.0)
        set_jmim_aggregator(_jmim_on)
        set_bur_lambda(_bur_lambda)
        # Miller-Madow / Chao-Shen relevance-MI bias correction. Both subtract/re-estimate away the plug-in
        # estimator's finite-sample bias from the OBSERVED relevance so high-cardinality noise no longer
        # out-ranks low-cardinality true signal at small n. Default 'none' keeps the legacy plug-in
        # estimator bit-exact. Reset in the finally. Chao-Shen
        # was previously an accepted-but-silently-ignored value (degraded to plug-in with a warning); it is
        # now fully wired into both the observed-relevance and permutation-null paths, mirroring
        # Miller-Madow's wiring exactly (see compute_relevance_score / mi_or_su_from_classes).
        _mi_corr = getattr(self, "mi_correction", "none")
        _mm_on = _mi_corr == "miller_madow"
        _cs_on = _mi_corr == "chao_shen"
        set_mi_miller_madow(_mm_on)
        set_mi_chao_shen(_cs_on)
        # Group-aware relevance MI: per-group I(X;Y|G) so a between-group-level feature (high global MI, ~0 within-group)
        # is demoted. Row resampling under non-uniform sample_weight reshuffles X but not groups, so restrict to the
        # no-resample case (sample_weight is None); otherwise disable with a warning rather than mis-assign rows.
        _gmi_payload = None
        if getattr(self, "group_aware_mi", False) and groups is not None:
            _g_arr = np.asarray(groups)
            _n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
            # 09_error_messages_ux.md: a groups-length mismatch is almost certainly a caller bug (wrong
            # array passed, stale groups from a differently-shaped prior call), not a "gracefully degrade
            # and move on" situation -- raise instead of silently disabling group-aware MI for the fit.
            if _g_arr.shape[0] != _n_rows:
                _restore_toggles_snapshot_and_raise(ValueError(f"MRMR.fit: groups length {_g_arr.shape[0]} != X rows {_n_rows}; groups must have one entry per row of X."))
            if sample_weight is not None:
                # 09_error_messages_ux.md: this is functionally identical to the ``groups``-ignored
                # situation (line ~3014's ``warnings.warn(UserWarning)``) -- an on-by-request feature
                # silently disabled for the fit -- so it uses the SAME guaranteed-visible channel instead
                # of a logger.warning a plain-script user with default logging would never see.
                warnings.warn(
                    "MRMR.fit: group_aware_mi disabled this fit because sample_weight is non-uniform "
                    "(resampling rows would misalign them against groups). Pass sample_weight=None or "
                    "group_aware_mi=False to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
                logger.warning("[MRMR] group_aware_mi disabled this fit: non-uniform sample_weight resamples rows and would misalign groups; pass sample_weight=None or group_aware_mi=False.")
            else:
                _si, _off = _prepare_group_segments(_g_arr)
                _size_weighted = getattr(self, "group_mi_aggregate", "size") == "size"
                _gmi_payload = (_si, _off, int(getattr(self, "group_mi_min_rows", 20)), _size_weighted)
                self.groups_ignored_ = False
        _set_group_mi(_gmi_payload)
        # Research-knob thread-locals (RelaxMRMR 3-D redundancy / PID synergy bonus / CMI permutation early-stop). All default OFF (alpha=0 / bonus=0 / stop=False) so the
        # legacy Fleuret per-candidate score is byte-identical; reset in the finally. Read in evaluation.py and forwarded to joblib workers like the SU/JMIM/BUR toggles.
        set_relaxmrmr_alpha(float(getattr(self, "relaxmrmr_alpha", 0.0) or 0.0))
        set_pid_synergy_bonus(float(getattr(self, "pid_synergy_bonus", 0.0) or 0.0))
        set_cmi_perm_stop(
            bool(getattr(self, "cmi_perm_stop", False)),
            float(getattr(self, "cmi_perm_alpha", 0.05)),
            int(getattr(self, "cmi_perm_n_permutations", 100)),
        )
        set_cpt_test(
            bool(getattr(self, "cpt_test", False)),
            int(getattr(self, "cpt_n_permutations", 200)),
        )
        # activate DCD thread-local. The DCDState dataclass
        # is constructed inside ``_screen_predictors`` (passed via dcd_config
        # kwarg) — joblib-safe; the thread-local is only the read-only branch
        # toggle. Reset in finally.
        _dcd_on = bool(getattr(self, "dcd_enable", False))
        _set_dcd_active(_dcd_on)
        # Critic1/H-3 fix: when DCD active and dcd_postoc_compose=False, suppress
        # the post-hoc cluster_aggregate FE-step (else double-aggregation). Save
        # and restore the original flag to keep the constructor-arg semantics
        # bit-stable across fits.
        _orig_cluster_aggregate_enable = bool(getattr(self, "cluster_aggregate_enable", True))
        _dcd_suppress_postoc = _dcd_on and not bool(getattr(self, "dcd_postoc_compose", False))
        if _dcd_suppress_postoc:
            self.cluster_aggregate_enable = False
        # FAST-SEARCH PROFILE. Apply the fast FE-search overrides for the duration of
        # this fit, recording each pre-fit value so the ``finally`` restores constructor-arg
        # semantics (clone / pickle / repeated-fit stability). Only knobs the user left at their
        # package default are overridden -- an explicit user value always wins. See the
        # ``fe_fast_search`` docstring in __init__ for the rationale + measured wins.
        # DEFAULT SCREEN SUBSAMPLE. Apply the feature-recovery screen-subsample for large n on
        # EVERY fit (not just fe_fast_search): the FE MI-sweep / polynom-pair / conditional-gate DETECTION
        # are rank-stable under subsampling and the survivors replay at full n, so the default MRMR() fit
        # can screen on ~30k rows at n=100k (168.8s -> ~75s, both compounds still recovered). Only knobs at
        # their package default are shrunk; restored in ``finally``. n below the screen size is a no-op.
        _default_screen_saved: dict = {}
        try:
            _n_rows_for_screen = int(X.shape[0]) if hasattr(X, "shape") else None
            if _n_rows_for_screen is not None:
                _default_screen_saved = self._apply_default_screen_subsample(_n_rows_for_screen)
        except Exception as exc:
            logger.debug("mrmr: default screen-subsample application failed; screening at full n: %r", exc, exc_info=True)
            _default_screen_saved = {}
        _fast_search_saved: dict = {}
        if bool(getattr(self, "fe_fast_search", False)):
            try:
                _fast_search_saved = self._apply_fast_search_profile()
            except Exception as exc:
                logger.debug("mrmr: fast-search profile application failed; using unmodified knobs: %r", exc, exc_info=True)
                _fast_search_saved = {}
        # LAZY ctor-alias reconciliation (sklearn ``get_params`` stays byte-identical to what the user
        # passed). The constructor no longer promotes ``random_state`` -> ``random_seed``; that is
        # resolved HERE and the EFFECTIVE value is written onto the public attr for the fit duration so
        # every reader (this module + _fit_impl_core's skip check + the cross-file ``self.random_seed``
        # uses) sees it, then the original stored value is restored in ``finally`` (saved == _UNSET means
        # "not overridden, leave alone").
        _eff_seed = self._effective_random_seed()
        _orig_random_seed = _UNSET
        if _eff_seed != getattr(self, "random_seed", None):
            _orig_random_seed = getattr(self, "random_seed", None)
            self.random_seed = _eff_seed
        # PICKLE-ONLY migration (NOT a ctor alias -- the ctor no longer accepts
        # ``skip_retraining_on_same_shape`` at all): an already-pickled MRMR predating
        # the content/shape rename can still carry the old attribute verbatim in its ``__dict__``
        # (``__setstate__`` never removes it), so a genuinely-legacy saved model's explicit
        # True/False choice is still honoured here rather than silently reset to the current default.
        # A freshly-constructed instance never has this attribute, so this is a no-op for it.
        _orig_skip_content = _UNSET
        _skip_shape = getattr(self, "skip_retraining_on_same_shape", None)
        if _skip_shape is not None:
            _eff_skip = bool(_skip_shape)
            if _eff_skip != getattr(self, "skip_retraining_on_same_content", True):
                _orig_skip_content = getattr(self, "skip_retraining_on_same_content", True)
                self.skip_retraining_on_same_content = _eff_skip
        # _fit_impl's large-n regression adaptive-quantization
        # gate (adaptive_nbins_large_n_reg) permanently overwrote self.nbins_strategy/self.quantization_nbins
        # in place with no restore anywhere -- breaking the sklearn clone()/get_params() round-trip contract
        # and permanently freezing a config the gate's own campaign data says LOSES at smaller n on any
        # subsequent .fit() call on the same instance. Snapshot unconditionally here (mirrors _orig_random_seed
        # / _orig_skip_content above) and restore in the finally block below regardless of whether the gate fired.
        _orig_nbins_strategy = getattr(self, "nbins_strategy", None)
        _orig_quantization_nbins = getattr(self, "quantization_nbins", None)
        try:
            # GLOBAL-RNG CONTAINMENT + SEED DETERMINISM: a fit consumes process-global
            # ``np.random`` in places no per-call Generator covers (cat-confirm permutation shuffles,
            # the FE families' global shuffles, etc.). Two failures followed: (a) an UNSEEDED fit
            # advanced the caller's MT19937 -> a second fit in the same process drifted (run-order
            # flakiness under the xdist suite); (b) even a SEEDED fit (``random_seed`` set) was NON-
            # deterministic in those global-RNG parts because nothing reseeded the process RNG, so the
            # selection drifted run-to-run (e.g. the 5-class layer16 LogReg-macro-F1 gate flaked).
            # Scope the WHOLE fit: when ``random_seed`` is set the block reseeds numpy/numba/cupy to it
            # (the fit becomes reproducible) AND restores the caller's state on exit; ``None`` (no seed
            # requested) restores only. A seeded fit is now deterministic + leak-free; an unseeded fit
            # is leak-free with unchanged (entropy) behaviour.
            # Record the fit's row/column counts so the AUTO (unset MLFRAME_FE_GPU_STRICT) size-gated STRICT
            # default can engage GPU-resident FE on large-n fits (selection-equivalent to CPU by ~50k, ~2.5x
            # faster) OR on a wide-but-under-the-row-threshold fit whose total (n, p) work already clears the
            # same floor a per-call dispatch would need (2026-07-11 fix -- the row-only gate ignored column
            # count entirely), and stay on the exact CPU path otherwise. Cleared in finally so it never leaks
            # to a later fit.
            _fit_shape = getattr(X, "shape", None)
            _set_auto_fit_n(
                int(_fit_shape[0]) if _fit_shape is not None else None,
                int(_fit_shape[1]) if _fit_shape is not None and len(_fit_shape) > 1 else None,
            )
            try:
                with _preserve_global_numpy_rng_state(self._effective_random_seed()):
                    result = self._fit_impl(X, y, groups, **fit_params)
            finally:
                _clear_auto_fit_n()
            try:
                _n_rows = int(X.shape[0]) if hasattr(X, "shape") else None
                # ``_effective_random_seed`` resolves both the canonical ``random_state`` and the
                # deprecated ``random_seed`` alias, whichever is set.
                _seed_resolved = self._effective_random_seed()
                _seed_for_provenance = int(_seed_resolved) if _seed_resolved is not None else None
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
            except Exception as exc:
                logger.debug("mrmr: provenance_ metadata build failed (diagnostic only): %r", exc, exc_info=True)
            # Stash X-fingerprint -> identity-bool in cross-target cache so a SUBSEQUENT fit (different y, same X) can early-skip the FE pipeline.
            if _identity_skip and _x_fp is not None:
                try:
                    _is_id = (
                        getattr(self, "support_", None) is not None
                        and len(self.support_) == X.shape[1]
                        and len(getattr(self, "_engineered_features_", []) or []) == 0
                    )
                    # Store (is_id, y_sample) so a later candidate hit can apply the y-correlation gate
                    # (A1-06): the "prior identity implies new identity" assumption only holds for a target
                    # correlated with the one that produced this entry. y_sample is a deterministic strided
                    # numeric sample (cheap; never the full y). The value stays back-compatible: legacy bool
                    # readers still see truthiness, and the read path below handles both bool and tuple.
                    with _MRMR_IDENTITY_FP_LOCK:
                        _cache_dict[_x_fp] = (bool(_is_id), _mrmr_y_corr_sample(y) if _is_id else None)
                    if _is_id:
                        # Remember which (X, y) fingerprint THIS instance's own real fit produced, so a
                        # later self-refit on the identical pair defers to _fit_impl's precise same-instance
                        # shortcuts instead of the coarse cross-target identity-shortcut above.
                        self._own_last_identity_fp_ = _x_fp
                        logger.info(
                            "[MRMR] cross-target identity cache STORED for X fingerprint=%s "
                            "(no features dropped, no engineered features); subsequent "
                            "fits on this X will short-circuit.",
                            _x_fp,
                        )
                except Exception as exc:
                    logger.debug("mrmr: cross-target identity-cache store failed (optimisation only): %r", exc, exc_info=True)
            # populate ``fe_provenance_`` from
            # the sibling module so users can audit which engineered
            # columns landed in support_, why (origin + mechanism
            # details) and what each contributed in the greedy gain
            # ledger. Pure metadata; never mutates the selection result.
            _pop_prov(self)
            # populate ``fe_rejection_ledger_`` -- the rejection side of the provenance
            # surface: one row per FE candidate a gate dropped (which gate + the margin).
            # Pure metadata built from the records ``_run_fe_step`` accumulated; never
            # mutates the selection result.
            _pop_rej(self)
            # gt_07 FE-family budget: compute this fit's per-family credit/ROI from fe_provenance_ +
            # the wall-time ledger, reallocate next fit's budget, and persist it under the dataset
            # fingerprint computed above. Report is populated whenever the flag is on, independent of
            # whether the pre-fit quota scaling above succeeded (visibility is half the feature's
            # value even before enforcement, per the plan). Never allowed to affect the selection
            # result itself -- wrapped defensively, same posture as provenance population above.
            # Reuses the SAME resolved flag the pre-fit block computed above ("auto" only persists /
            # reallocates when it actually fired this fit -- a probe that found nothing cached must not
            # start writing a NEW cache entry, or "auto" would silently flip itself on forever after one
            # fit, defeating the whole "opt-in-once" contract).
            if _fe_budget_learning_effective:
                try:
                    from .._fe_family_budget import (
                        family_credit as _fe_family_credit,
                        family_roi as _fe_family_roi,
                        persist_budgets as _fe_persist_budgets,
                        reallocate_budgets as _fe_reallocate_budgets,
                    )
                    from .._fe_family_timing import get_fe_family_wall as _fe_get_wall

                    _fe_wall_post_fit = _fe_get_wall()
                    _fe_wall_pre_fit = getattr(self, "_fe_budget_wall_pre_fit_", None) or {}
                    # This fit's OWN wall delta, not the process-cumulative total (see the pre-fit
                    # snapshot comment above for why the ledger itself is never reset).
                    _fe_wall_snapshot = {
                        _fam: (
                            _post[0] - _fe_wall_pre_fit.get(_fam, (0.0, 0))[0],
                            _post[1] - _fe_wall_pre_fit.get(_fam, (0.0, 0))[1],
                        )
                        for _fam, _post in _fe_wall_post_fit.items()
                    }
                    _fe_credit = _fe_family_credit(getattr(self, "fe_provenance_", None))
                    _fe_roi = _fe_family_roi(_fe_credit, _fe_wall_snapshot)
                    _fe_budget_kwargs = dict(getattr(self, "fe_budget_kwargs", None) or {})
                    _fe_tracked_families = ("triplet", "quadruplet", "adaptive_arity")
                    _fe_equal_share = 1.0 / len(_fe_tracked_families)
                    # Compound on top of the budget actually used THIS fit (loaded pre-fit, or
                    # equal-split when nothing was persisted yet) -- restarting from equal-split every
                    # fit would make learning never accumulate across successive fits.
                    _fe_prev_budgets = dict(getattr(self, "_fe_budget_prev_", None) or {f: _fe_equal_share for f in _fe_tracked_families})
                    for _fam in _fe_tracked_families:
                        _fe_prev_budgets.setdefault(_fam, _fe_equal_share)
                    _fe_budgets_before = dict(_fe_prev_budgets)
                    _fe_budgets_after = _fe_reallocate_budgets(_fe_roi, base_budget=_fe_prev_budgets, **_fe_budget_kwargs)
                    _fe_persist_budgets(_fe_budgets_after, fingerprint=getattr(self, "_fe_budget_fingerprint_", None))
                    self.fe_family_budget_ = dict(
                        wall=_fe_wall_snapshot,
                        credit=_fe_credit,
                        roi=_fe_roi,
                        budgets_before=_fe_budgets_before,
                        budgets_after=_fe_budgets_after,
                    )
                except Exception as _fe_budget_post_exc:
                    logger.warning("[MRMR] fe_budget_learning: post-fit credit/reallocate/persist failed (%s); no budget update this fit.", _fe_budget_post_exc)
            _log_fe_wall()
            self._print_fit_summary()
            return result
        finally:
            # Restore the MI thread-locals to the values they held at fit ENTRY (snapshot above), not to
            # hardcoded literals: an inner fit must leave an outer fit's toggles intact. Mirrors the
            # _prev_* restore in _evaluation_driver.py's worker path.
            _su0, _jmim0, _bur0, _mm0, _relax0, _pid0, _cmi0, _cpt0, _cs0 = _toggles_snapshot

            def _restore_synergy_bonuses() -> None:
                """Restore RelaxMRMR/PID/CMI-perm/CPT thread-locals to their fit-entry snapshot."""
                set_relaxmrmr_alpha(_relax0)
                set_pid_synergy_bonus(_pid0)
                set_cmi_perm_stop(_cmi0[0], _cmi0[1], _cmi0[2])
                set_cpt_test(_cpt0[0], _cpt0[1])

            def _make_fe_budget_restorer(_a: str, _v: int) -> Callable[[], None]:
                """Bind (attr, value) at definition time so the restore closure isn't a late-binding loop-variable trap."""

                def _restore() -> None:
                    """Restore the bound attribute to its bound original value."""
                    setattr(self, _a, _v)

                return _restore

            for _attr, _orig_val in _fe_budget_quota_snapshot.items():
                _safe_restore(_make_fe_budget_restorer(_attr, _orig_val), f"fe_budget_learning quota override ({_attr})")
            _safe_restore(lambda: set_su_normalization(_su0), "SU normalization thread-local")
            _safe_restore(lambda: set_jmim_aggregator(_jmim0), "JMIM aggregator thread-local")
            _safe_restore(lambda: set_bur_lambda(_bur0), "BUR lambda thread-local")
            _safe_restore(lambda: set_mi_miller_madow(_mm0), "Miller-Madow thread-local")
            _safe_restore(lambda: set_mi_chao_shen(_cs0), "Chao-Shen thread-local")
            _safe_restore(lambda: _set_group_mi(None), "group-aware MI thread-local")
            _safe_restore(_restore_synergy_bonuses, "RelaxMRMR/PID/CMI-perm/CPT synergy thread-locals")
            # reset DCD thread-local and restore cluster_aggregate_enable to its constructor value
            # (Critic2 fix: missing reset in v1 plan).
            _safe_restore(lambda: _set_dcd_active(False), "DCD active thread-local")
            _safe_restore(lambda: setattr(self, "cluster_aggregate_enable", _orig_cluster_aggregate_enable), "cluster_aggregate_enable")
            _safe_restore(lambda: self.__dict__.pop("_pre_fit_ctor_params_snapshot_", None), "pre-fit ctor-params snapshot")
            # Restore the lazily-reconciled ctor aliases so ``get_params`` / ``clone`` see the unmodified
            # user-supplied values (sklearn round-trip contract). ``_UNSET`` => never overridden.
            if _orig_random_seed is not _UNSET:
                # cast: narrowed by the is-not-_UNSET sentinel check above; mypy can't track object-identity narrowing.
                self.random_seed = cast(Optional[int], _orig_random_seed)
            if _orig_skip_content is not _UNSET:
                self.skip_retraining_on_same_content = cast(bool, _orig_skip_content)
            # FIT_IMPL_A-1 fix: restore the adaptive_nbins_large_n_reg gate's in-place overwrite of
            # nbins_strategy/quantization_nbins so clone()/get_params()/a subsequent .fit() on this same
            # instance see the constructor's original values, not whatever the gate last computed.
            _safe_restore(lambda: setattr(self, "nbins_strategy", _orig_nbins_strategy), "nbins_strategy (adaptive_nbins_large_n_reg gate)")
            _safe_restore(lambda: setattr(self, "quantization_nbins", _orig_quantization_nbins), "quantization_nbins (adaptive_nbins_large_n_reg gate)")
            # restore the fast-search profile overrides (constructor-arg stability).
            # Restore the default screen-subsample knobs to their pre-fit (constructor) values so
            # clone / pickle / repeated-fit see unchanged constructor-arg semantics.
            if _default_screen_saved:

                def _make_default_screen_restorer(_k: str, _v: Any) -> Callable[[], None]:
                    """Bind (_k, _v) into a niladic restorer so the loop variables are captured by value, not by reference."""
                    return lambda: setattr(self, _k, _v)

                for _k, _v in _default_screen_saved.items():
                    _safe_restore(_make_default_screen_restorer(_k, _v), f"default-screen-subsample knob {_k!r}")
            if _fast_search_saved:

                def _restore_fast_search_knob(_k: str, _v: Any) -> None:
                    """Restore one saved fast-search knob (plain attr, Fourier-detect cap, or an os.environ entry)."""
                    if _k == "__fdcap__":
                        if _v is None:
                            clear_fourier_detect_cap()
                        else:
                            set_fourier_detect_cap(_v)
                    elif _k.startswith("__env__"):
                        _envk = _k[len("__env__") :]
                        if _v is None:
                            os.environ.pop(_envk, None)
                        else:
                            os.environ[_envk] = _v
                    else:
                        setattr(self, _k, _v)

                def _make_fast_search_restorer(_k: str, _v: Any) -> Callable[[], None]:
                    """Bind (_k, _v) into a niladic restorer so the loop variables are captured by value, not by reference."""
                    return lambda: _restore_fast_search_knob(_k, _v)

                for _k, _v in _fast_search_saved.items():
                    _safe_restore(_make_fast_search_restorer(_k, _v), f"fast-search knob {_k!r}")

            def _restore_fe_auto_flags() -> None:
                """Restore every fe_*_enable flag fe_auto flipped ON back to its pre-fit value."""
                for _flag, _orig in _fe_auto_restore.items():
                    setattr(self, _flag, _orig)

            # restore any fe_*_enable flags fe_auto flipped ON, so the
            # constructor-arg semantics are stable across fits / clone / pickle.
            _safe_restore(_restore_fe_auto_flags, "fe_auto-flipped fe_*_enable flags")
            frame = getattr(self, "_pandas_frame_for_target_cleanup", None)
            names = getattr(self, "_target_names_for_cleanup", None)
            if frame is not None and names:
                # Drop only columns that actually exist (success path already removed them).
                present = [c for c in names if c in frame.columns]
                if present:
                    # Restore the caller's frame in place (it stored this exact object for cleanup); option_context
                    # silences the conservative SettingWithCopy heuristic on a possibly-viewed frame, no copy.
                    def _drop_target_cleanup_columns() -> None:
                        """Drop the temporary targ_* columns this fit injected into the caller's own frame."""
                        with pd.option_context("mode.chained_assignment", None):
                            frame.drop(columns=present, inplace=True)  # noqa: PD002 -- must mutate the caller's stored frame OBJECT in place by identity (see the enclosing comment); rebinding a local would silently not clean up the caller's actual frame

                    _safe_restore(_drop_target_cleanup_columns, "target-cleanup column drop on caller frame")
            self._pandas_frame_for_target_cleanup = None
            self._target_names_for_cleanup = None

            def _refresh_signature_params_post_restore() -> None:
                """Re-stamp ``self.signature``'s params component from a LIVE ``get_params()`` read taken
                AFTER every restore above has completed (bug found while testing the re-entrancy
                guard). ``_fit_impl`` (``_fit_impl_core.py``) already
                does an analogous "refresh with post-fit values before storing" step so a param genuinely
                normalised IN PLACE during the fit (e.g. RFECV's ``scoring`` resolution) still matches the
                NEXT fit's freshly-read params -- but that refresh runs BEFORE this method's OWN transient
                overrides (cluster_aggregate_enable, fast-search profile, default-screen-subsample, ...)
                are restored, so it captured their TRANSIENT mid-fit values, permanently breaking the
                same-content-skip match for the common default config. This second, later refresh runs
                after every override above is undone, so it reflects the true, stable, post-fit-and-
                restore state -- exactly what the NEXT fit's pre-override snapshot will read."""
                _sig = getattr(self, "signature", None)
                if _sig is None:
                    return
                self.signature = (*_sig[:-1], _hashable_params_signature(self.get_params(deep=True)))

            _safe_restore(_refresh_signature_params_post_restore, "post-restore signature params refresh")

    # ``_fit_impl`` is implemented in ``_mrmr_fit_impl.py`` and bound onto
    # this class at the bottom of this module.

    # ``_run_fe_step`` is implemented in ``_mrmr_fe_step.py`` and bound
    # onto this class at the bottom of this module.

    # ``_append_engineered`` is implemented in ``_mrmr_validate_transform.py`` and bound onto
    # this class at the bottom of this module. ``transform`` itself, unlike its siblings above,
    # is NOT late-bound the same way -- see its own docstring immediately below for why.
    def transform(self, X, y=None):
        """sklearn-1.x transformer protocol. Delegates to the implementation in
        ``_mrmr_validate_transform.py``, but is defined directly on this class body (rather than
        late-bound at module bottom like ``_fit_impl``/``_run_fe_step``/``_append_engineered``)
        so ``_SetOutputMixin.__init_subclass__`` actually wraps it. Pre-fix, the bottom-of-module
        ``MRMR.transform = _transform_func`` rebind nuked the wrapper ``__init_subclass__`` had
        attached during class definition, silently making ``MRMR.set_output(transform='pandas')``
        a no-op when transform was called directly with ndarray input (the canonical sklearn
        contract requires a DataFrame)."""
        out = _mrmr_transform_impl(self, X, y)
        if getattr(self, "usability_aware_lists", False) and (getattr(self, "support_linear_", None) or getattr(self, "support_universal_", None)):
            out = self._append_usability_union(out, X)
        return out
