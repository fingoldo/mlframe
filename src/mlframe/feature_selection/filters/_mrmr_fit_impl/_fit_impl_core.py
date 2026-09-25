"""``MRMR._fit_impl`` main fit body for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. ``_fit_impl`` is bound back onto the ``MRMR`` class at the
parent's module bottom, so call sites that invoke ``self._fit_impl(...)``
continue to work unchanged.

Heavy lifting: signature/cache key build, content-hash short-circuit,
sub-sample loop, FE-step orchestration, MI ranking and the per-fold
selection. Many helpers (logger, signature hashing, target coercion)
live in the parent and are imported lazily inside this body to avoid the
``mrmr -> _mrmr_fit_impl -> mrmr`` import cycle.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from timeit import default_timer as timer


import numpy as np

import pandas as pd
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Guards every read-then-mutate sequence on the process-wide ``MRMR._FIT_CACHE`` (lookup + ``move_to_end`` on
# a hit; ``__setitem__`` + ``move_to_end`` + LRU/byte-cap ``popitem`` on store). Concurrent fits - multi-target
# discovery, joblib-threading callers, web-service workers - otherwise race ``popitem``/``__setitem__``/
# ``move_to_end`` on the same OrderedDict and can raise KeyError or evict the wrong entry. RLock so a wrapped
# region may safely re-enter. The companion ``_MRMR_IDENTITY_FP_CACHE`` already had its own lock; this closes the
# same gap for the fit cache. Exposed on the ``MRMR`` class (idempotently, inside the fit body) as
# ``_FIT_CACHE_LOCK`` so any other holder of the cache can take the same canonical lock.
_MRMR_FIT_CACHE_LOCK = threading.RLock()


# _pgn_raw_budget re-exported from the ``_assign_support`` sub-split (Tier F) for backward-compat
# direct imports (e.g. tests/feature_selection/filters/test_mrmr_pgn_engineered_budget.py) that
# import it from this module's own namespace.
from ._assign_support import _pgn_raw_budget  # noqa: F401 - re-exported facade name, imported directly by tests/test_mrmr_pgn_engineered_budget.py

# Above this many bytes of nullable-column data, densify masked columns one-per-``assign`` instead of all at once
# so peak extra RAM stays ~one float64 column rather than ~2x the whole nullable subset (100GB-frame safe).
_NULLABLE_DENSIFY_EAGER_MAX_BYTES = 2 * 1024**3


def _gains_paired_with_output(self, predictors_log) -> "np.ndarray | None":
    """Greedy gains in ``get_feature_names_out()`` order: raw support by column index, then advertised engineered recipes by name.

    A raw column the screen picked on its own has a single-index log entry; an engineered column the screen scored is logged under its
    recipe name. Anything without an entry (a raw re-added by a retention pass, an engineered column appended after the screen) gets 0.0.
    Returns ``None`` when ``support_`` is unavailable, so the caller keeps its positional fallback.
    """
    support = getattr(self, "support_", None)
    if support is None:
        return None
    support = np.asarray(support)
    if support.dtype == bool:
        support = np.flatnonzero(support)
    by_index: dict = {}
    by_name: dict = {}
    for entry in predictors_log:
        indices = tuple(entry.get("indices", ()) or ())
        gain = float(entry.get("gain", 0.0))
        if len(indices) == 1:
            by_index.setdefault(int(indices[0]), gain)
        name = entry.get("name")
        if name is not None:
            by_name.setdefault(str(name), gain)
    raw = [by_index.get(int(i), 0.0) for i in support.tolist()]
    recipes = getattr(self, "_engineered_recipes_", None) or []
    advertised = [r for r in recipes if r.extra.get("chain_lookups") is not None or not r.extra.get("requires_refit_for_replay")]
    engineered = [by_name.get(str(r.name), 0.0) for r in advertised]
    return np.asarray(raw + engineered, dtype=np.float64)


def _align_mrmr_gains(self) -> None:
    """Trim/pad ``self.mrmr_gains_`` to exactly ``self.n_features_`` (the ``len(mrmr_gains_) == n_features_``
    public contract). ``mrmr_gains_`` is the greedy log; the final feature count diverges (shorter on a
    degenerate/redundancy/cap/UAED trim, longer when FE/retention appended features the greedy never scored).
    Must be called as the VERY LAST fit step, after every ``n_features_`` mutation - including the group-aware
    final demotion, which drops zero-within-group engineered recipes and lowers ``n_features_`` but does not
    touch ``mrmr_gains_``. Idempotent (byte-identical when already aligned). Best-effort."""
    try:
        _g = getattr(self, "mrmr_gains_", None)
        _nf_final = int(getattr(self, "n_features_", 0) or 0)
        _log = getattr(self, "_predictors_log_", None) or ()
        _repaired = _gains_paired_with_output(self, _log) if _log else None
        if _repaired is not None:
            if _repaired.shape[0] > _nf_final:
                _repaired = _repaired[:_nf_final]
            elif _repaired.shape[0] < _nf_final:
                _repaired = np.concatenate([_repaired, np.zeros(_nf_final - _repaired.shape[0], dtype=np.float64)])
            self.mrmr_gains_ = _repaired
        elif _g is not None and _nf_final >= 0 and _g.shape[0] != _nf_final:
            if _g.shape[0] > _nf_final:
                self.mrmr_gains_ = _g[:_nf_final]
            else:
                self.mrmr_gains_ = np.concatenate([_g, np.zeros(_nf_final - _g.shape[0], dtype=np.float64)])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("mrmr: mrmr_gains_ finalisation failed: %r", e, exc_info=True)

"""MRMR._fit_impl main fit body.

The irreducible single function _fit_impl (bound onto the MRMR class
at the mrmr package facade) lives here verbatim. It is LOC-budget exempt:
one giant function cannot be split without distorting the fit control flow.
Its many lazy in-body from ..X import ... imports break the
mrmr -> _mrmr_fit_impl -> mrmr cycle; the small free helpers it calls live
in the sibling _helpers.py.
"""


from ._helpers import _mrmr_cache_bytes_total

def _fit_impl(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, **fit_params):
    """We run N selections on data subsets, and pick only features that appear in all selections"""
    # Lazy import: ``.mrmr`` re-imports this module at its module bottom for
    # method binding -> any top-level ``from .mrmr import ...`` here would
    # create a hard import cycle that ``tests/test_meta/test_no_import_cycles.py``
    # flags. Python's module cache makes repeat imports cheap.
    from ..mrmr import (
        MRMR,
        _hashable_params_signature,
        numeric_column_names,
        screen_predictors,
        sort_dict_by_value,
    )
    # Publish the canonical fit-cache lock on the class so any other holder of ``_FIT_CACHE`` shares it. Idempotent:
    # only set on first fit, never re-bound (re-binding would split the lock identity under concurrent fits).
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._state import ROUTED_RECIPE_FAMILIES, FEParams, FERecipes

    recipes = FERecipes()  # per-family FE recipe registries of this fit (see _fit_impl_stages._state)
    fe = FEParams()  # FE parameters resolved for this fit
    if getattr(MRMR, "_FIT_CACHE_LOCK", None) is None:
        MRMR._FIT_CACHE_LOCK = _MRMR_FIT_CACHE_LOCK
    # include_numeric NaN guard: snapshot raw NaN/inf-bearing NUMERIC columns at the VERY START of fit, before
    # _validate_inputs / categorize / any GPU-discretisation path can impute X. include_numeric must skip a column
    # the user supplied with NaN - its quantile-edge transform replay has no NaN bin, so a NaN test value would
    # silently clip to the top bin (train/serve skew). Captured here so a downstream in-place impute (e.g. the GPU
    # categorize path that is active when the harness sets CUDA_PATH) cannot erase the NaN before the candidate
    # scan and defeat the guard.
    _include_numeric_input_nan_cols = set()
    # Hoisted ONCE (y is never reassigned in _fit_impl): the as-numpy target was re-materialised
    # 53x across the FE/screen stages. Same array (read-only consumers); behavior-preserving.
    _y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    # Per-column boolean NaN mask snapshot at fit entry, before any in-place impute (the include_numeric / binned_numeric_agg cat-FE path GPU-categorizes
    # and imputes X in place when CUDA_PATH is set). The missingness-FE family (is_missing__/missingness_count/missingness_pattern) derives its signal
    # from where the input was NaN; it runs AFTER that impute, so it must read this snapshot, not the live (now-finite) X, or the signal is silently erased.
    _fit_entry_nan_mask = {}
    # Both consumers of this snapshot are opt-in and default OFF (missingness-FE family below, and cat-FE's
    # include_numeric branch far downstream): skip the per-column float64-cast + isfinite scan entirely when
    # neither will ever read it, rather than paying it on every fit regardless. Mirrors each consumer's own gate
    # exactly, so this can never diverge into a false skip.
    _cat_fe_cfg_probe = getattr(self, "cat_fe_config", None)
    _will_use_include_numeric_nan_guard = bool(_cat_fe_cfg_probe is not None and getattr(_cat_fe_cfg_probe, "enable", True) and getattr(_cat_fe_cfg_probe, "include_numeric", False))
    _will_use_missingness_fe = (
        bool(getattr(self, "fe_missingness_indicator_enable", False))
        or bool(getattr(self, "fe_missingness_count_enable", False))
        or bool(getattr(self, "fe_missingness_pattern_enable", False))
    )
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._prelude import _record_fit_entry_nan_mask
    _record_fit_entry_nan_mask(X, _will_use_include_numeric_nan_guard, _will_use_missingness_fe, _include_numeric_input_nan_cols, _fit_entry_nan_mask)
    X = self._validate_inputs(X, y)

    # Large-n regression adaptive-quantization gate. The 180-cell campaign showed fixed 20-bin quantile beats MDLP 15/15 on reg n=100k
    # (holdout +0.116 / F1 +0.242) but LOSES at reg n=20k and on classification, so it is gated to the detected (regression AND n>=threshold)
    # regime, and only when the user left both quantization params at defaults. getattr defaults keep this replay-safe on pre-flip pickles.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._prelude import _apply_large_n_regression_nbins_gate
    _apply_large_n_regression_nbins_gate(self, X, y)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute inputs/outputs signature
    # ----------------------------------------------------------------------------------------------------------------------------

    # Shape-only signature was too loose: un-cloned MRMR fit on target A, then re-fit on target B with
    # identical (n_rows, n_cols) shape, replayed A's support_ verbatim. Fold the y content hash in.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._fit_cache import _fit_signature
    _x_hash_for_sig, _y_hash_for_sig, signature = _fit_signature(self, y, X)
    from ._fit_impl_stages._fit_cache import _fit_cache_key, _replay_from_fit_cache, _same_inputs_skip

    if _same_inputs_skip(self, signature, _x_hash_for_sig):
        return self

    # Process-wide ``_FIT_CACHE`` hit. After sklearn.base.clone() the cloned MRMR has no fitted state so
    # the signature==signature shortcut above never fires. Content-based key (id-based missed every hit
    # because the suite copies X between iterations - different id() but identical content);
    # _content_array_signature returns shape+dtype+10 sampled values, cheap O(1) and statistically unique
    # enough to avoid false positives on real data. Falls through to full fit on any error or miss.
    _cache_key = _fit_cache_key(self, X, y, _y_hash_for_sig, groups)
    if _replay_from_fit_cache(self, _cache_key):
        return self

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    # Outer FE-loop runtime-budget guard. screen_predictors honours self.max_runtime_mins on its own; here we additionally
    # short-circuit between FE iterations so a long FE step that finished after the budget elapsed doesn't trigger another.
    start_time = timer()
    ran_out_of_time = False

    # Carry an absolute deadline to the OPTIONAL enrichment FE generators (orth / extra-basis / pair-cross,
    # plus hermite / wavelet / hinge / binned_numeric_agg / pairwise_modular / conditional_gate /
    # cat_interactions / target_encoding) so a single wide-frame enrichment pass that starts before the
    # budget is spent still aborts its per-column / per-pair loop at the deadline instead of running tens
    # of seconds past a tiny max_runtime_mins. Enrichment-only: the core screen / greedy MI is never gated,
    # so an aborted pass still leaves a usable partial selection. Cleared in MRMR.fit's finally (the outer
    # call-site boundary, in _mrmr_class.py) - NOT here, since this function has no single exit point.
    from .._fe_deadline import set_fe_deadline as _set_fe_deadline
    _set_fe_deadline((start_time + self.max_runtime_mins * 60.0) if self.max_runtime_mins is not None else None)

    def _fe_budget_ok() -> bool:
        """Between-FE-step wall-clock gate: True while remaining time under ``max_runtime_mins`` is unspent (or unset)."""
        # Pre-FE univariate generators (extra-basis, wavelet, dispersion, ...) run once before the FE loop and the
        # between-step guard below cannot bound a single long stage; gate each heavy default-ON stage on the remaining
        # wall-clock so an oversized fit handed a small max_runtime_mins aborts within a small multiple of the budget.
        if self.max_runtime_mins is None:
            return True
        return bool((timer() - start_time) / 60.0 < self.max_runtime_mins)

    def _fe_family_on(flag: str, default: bool = False) -> bool:
        """True iff the family's own ``fe_*_enable`` flag is set AND this fit has an FE budget.

        ``fe_max_steps=0`` is the "no feature engineering at all" contract, and it is unconditional: a family flag can only ENABLE a family
        within that budget, never buy its way past it. The budget is read from ``self`` (not the local), so the helper is safe anywhere in the fit.

        Previously only the hybrid-orth / univariate-basis pair honoured this; every other family fired at
        ``fe_max_steps=0``, which made "no FE" mean "no FE except the ~30 default-ON families" and silently
        engineered columns into fits that had explicitly asked for none. The wall-clock budget is part of the same question: ``_fe_budget_ok``
        used to be consulted at 4 of the ~35 cascade stages, so a spent ``max_runtime_mins`` still let the other ~31 families start.
        """
        return bool(getattr(self, flag, default)) and int(getattr(self, "fe_max_steps", 0) or 0) > 0 and _fe_budget_ok()

    dtype = self.dtype

    parallel_kwargs = self._effective_parallel_kwargs()
    n_jobs = self._effective_n_jobs()
    verbose = self.verbose

    prefetch_factor = 4

    fe.max_steps = self.fe_max_steps
    fe.npermutations = self.fe_npermutations
    fe.unary_preset = self.fe_unary_preset
    fe.binary_preset = self.fe_binary_preset
    fe.max_pair_features = self.fe_max_pair_features

    fe.min_nonzero_confidence = self.fe_min_nonzero_confidence
    fe.min_pair_mi = self.fe_min_pair_mi
    fe.min_pair_mi_prevalence = self.fe_min_pair_mi_prevalence
    fe.min_engineered_mi_prevalence = self.fe_min_engineered_mi_prevalence
    fe.good_to_best_feature_mi_threshold = self.fe_good_to_best_feature_mi_threshold
    fe.max_external_validation_factors = self.fe_max_external_validation_factors
    fe.max_polynoms = self.fe_max_polynoms
    fe.print_best_mis_only = self.fe_print_best_mis_only
    fe.smart_polynom_iters = self.fe_smart_polynom_iters
    fe.smart_polynom_optimization_steps = self.fe_smart_polynom_optimization_steps
    fe.min_polynom_degree = self.fe_min_polynom_degree
    fe.max_polynom_degree = self.fe_max_polynom_degree
    fe.min_polynom_coeff = self.fe_min_polynom_coeff
    fe.max_polynom_coeff = self.fe_max_polynom_coeff

    # Convert numpy array to DataFrame if needed
    # Record a sentinel
    # ``self._feature_names_in_synthesized_`` so ``get_feature_names_out``
    # can distinguish ndarray-fit synthesized placeholders from
    # legitimate DataFrame columns the user happened to name
    # ``feature_<int>``. Pre-fix the detection used
    # ``str(n).startswith("feature_")`` heuristically, which
    # misclassified real columns and silently bypassed the sklearn
    # column-drift contract for any user whose DataFrame happened to
    # use that naming (very common after ``pd.DataFrame(arr)`` + rename).
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _prepare_input_frame
    X = _prepare_input_frame(self, X, verbose)

    from ._fe_stage_cascade_early_a import _fe_stage_cascade_early_a
    from ._fe_stage_cascade_early_b import _fe_stage_cascade_early_b

    # Recipe registries for every FE family from Layer 23 through Layer 104: each family owns exactly
    # one dict here, mutated in place by whichever stage (inline below, or one of the _fe_stage_cascade_*
    # siblings) actually runs that family -- declared once, up front, so every consumer downstream (the
    # end-of-fit ``engineered_recipes.update(...)`` remap) sees the SAME dict objects regardless of which
    # sibling module mutated them. Passing a dict into a sibling and having it mutate entries in place is
    # safe (no return needed); a REASSIGNMENT inside a sibling would NOT propagate back -- confirmed via a
    # systematic check that none of these are ever reassigned (only ``[key] = value`` mutated) in the two
    # early-cascade siblings below.
    recipes.hybrid_orth: dict = {}
    recipes.mi_greedy: dict = {}
    recipes.kfold_te: dict = {}
    recipes.binned_agg: dict = {}
    recipes.count_enc: dict = {}
    recipes.freq_enc: dict = {}
    recipes.cat_num: dict = {}
    recipes.miss_ind: dict = {}
    recipes.miss_cnt: dict = {}
    recipes.miss_pat: dict = {}
    recipes.ratio: dict = {}
    recipes.log_ratio: dict = {}
    recipes.grouped_delta: dict = {}
    recipes.lagged_diff: dict = {}
    recipes.cat_pair: dict = {}
    recipes.cat_triple: dict = {}
    recipes.numeric_decompose: dict = {}
    recipes.temporal_agg: dict = {}
    recipes.modular: dict = {}
    recipes.pairwise_modular: dict = {}
    recipes.integer_lattice: dict = {}
    recipes.row_argmax: dict = {}
    recipes.conditional_gate: dict = {}
    recipes.group_distance: dict = {}
    recipes.rare_category: dict = {}
    recipes.conditional_residual: dict = {}
    recipes.conditional_dispersion: dict = {}
    recipes.conditional_quantile_rank: dict = {}
    recipes.ordinal_pattern: dict = {}
    recipes.random_fourier: dict = {}
    recipes.sir_direction: dict = {}
    recipes.lof: dict = {}
    recipes.mahalanobis_density: dict = {}
    recipes.wavelet: dict = {}
    recipes.rankgauss: dict = {}
    recipes.grouped_agg: dict = {}
    recipes.composite_group_agg: dict = {}
    recipes.grouped_quantile: dict = {}

    X, _raw_input_cols_pre_fe, recipes.hinge_deferred_values, recipes.hinge_deferred = _fe_stage_cascade_early_a(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe.max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok,
        _hybrid_orth_pre_recipes=recipes.hybrid_orth, _mi_greedy_pre_recipes=recipes.mi_greedy,
    )
    X = _fe_stage_cascade_early_b(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe.max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fit_entry_nan_mask=_fit_entry_nan_mask, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _kfold_te_pre_recipes=recipes.kfold_te, _binned_agg_pre_recipes=recipes.binned_agg,
        _count_enc_pre_recipes=recipes.count_enc, _freq_enc_pre_recipes=recipes.freq_enc,
        _cat_num_pre_recipes=recipes.cat_num,
        _miss_ind_pre_recipes=recipes.miss_ind, _miss_cnt_pre_recipes=recipes.miss_cnt,
        _miss_pat_pre_recipes=recipes.miss_pat,
        _ratio_pre_recipes=recipes.ratio, _log_ratio_pre_recipes=recipes.log_ratio,
        _grouped_delta_pre_recipes=recipes.grouped_delta, _lagged_diff_pre_recipes=recipes.lagged_diff,
    )
    from ._fe_stage_cascade_mid_a import _fe_stage_cascade_mid_a

    X = _fe_stage_cascade_mid_a(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe.max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _cat_pair_pre_recipes=recipes.cat_pair, _cat_triple_pre_recipes=recipes.cat_triple,
        _composite_group_agg_pre_recipes=recipes.composite_group_agg,
        _conditional_gate_pre_recipes=recipes.conditional_gate,
        _grouped_agg_pre_recipes=recipes.grouped_agg,
        _grouped_quantile_pre_recipes=recipes.grouped_quantile,
        _integer_lattice_pre_recipes=recipes.integer_lattice,
        _modular_pre_recipes=recipes.modular,
        _numeric_decompose_pre_recipes=recipes.numeric_decompose,
        _pairwise_modular_pre_recipes=recipes.pairwise_modular,
        _row_argmax_pre_recipes=recipes.row_argmax,
    )
    from ._fe_stage_cascade_mid_b import _fe_stage_cascade_mid_b

    X = _fe_stage_cascade_mid_b(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe.max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _group_distance_pre_recipes=recipes.group_distance,
        _rare_category_pre_recipes=recipes.rare_category,
        _conditional_residual_pre_recipes=recipes.conditional_residual,
        _conditional_dispersion_pre_recipes=recipes.conditional_dispersion,
        _conditional_quantile_rank_pre_recipes=recipes.conditional_quantile_rank,
        _ordinal_pattern_pre_recipes=recipes.ordinal_pattern,
        _random_fourier_pre_recipes=recipes.random_fourier,
        _sir_direction_pre_recipes=recipes.sir_direction,
        _lof_pre_recipes=recipes.lof,
        _mahalanobis_density_pre_recipes=recipes.mahalanobis_density,
        _wavelet_pre_recipes=recipes.wavelet,
        _rankgauss_pre_recipes=recipes.rankgauss,
    )
    # Layer 92: temporal leak-safe grouped aggregations. Carved
    # verbatim into the sibling ``_fe_stage_temporal_agg`` (Tier E partial
    # split); the helper threads self + ``_y_np`` / ``verbose`` /
    # ``_temporal_agg_pre_recipes`` explicitly, mutates self + the recipes dict
    # in place, and RETURNS the (possibly replaced) working ``X`` frame.
    from ._fe_stage_temporal_agg import _fe_stage_temporal_agg
    X = _fe_stage_temporal_agg(self, X, _y_np, verbose, recipes.temporal_agg, _fe_family_on=_fe_family_on)

    # ACCURACY GATE (2026-06-04, default ON via ``fe_accuracy_gate``). The MI-uplift gates inside the FE generators are fooled by plug-in MI's bias inflation: a Fourier / chirp / Hermite transform of a strong RAW signal earns an inflated MI estimate and out-ranks (then evicts) the raw column even when it adds NO real predictive value. The adaptive-Fourier PROTECTION block at support-finalisation then force-readds those hijackers past the MRMR screen, so they survive into support_ AND leak into ``hybrid_orth_features_`` / ``_adaptive_fourier_features_`` even when a genuine raw signal (or its is_missing__ MNAR indicator) carries the information. This gate runs a held-out multivariate linear-probe uplift check per engineered column against its raw source: a column that adds no held-out uplift over its source - or whose source is >2%-missing (MNAR fail-closed, the signal lives in the NaN pattern the probe cannot see) - is dropped here so it can neither evict the raw signal nor leak into the roster. Only orth_* engineered columns with a single resolvable raw source are gated; the is_missing__ / missingness_* indicators are exempt by construction (their recipes live in ``_miss_*_pre_recipes``, never ``_hybrid_orth_pre_recipes``, so they are never routed here). y is read only at fit; transform replays the survivors without y. Best-effort: any failure falls back to keeping the column.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_gates import _gate_engineered_accuracy
    X = _gate_engineered_accuracy(self, X, recipes, _y_np, verbose)

    # Layer 27: cross-stage engineered-column dedup. Hybrid and
    # MI-greedy stages run independently; on signals like ``y = sign(x^2 - 1)``
    # hybrid emits ``x__He2`` and MI-greedy emits ``square(x)`` / ``abs(x)`` /
    # ``sqrt_abs(x)`` / ``log_abs(x)`` - all are monotone-in-|x| encodings
    # of the SAME signal (Pearson |corr| ~ 0.99+ on rank-correlated MI binning).
    # MRMR's CMI gate can't tell them apart well enough to prune; the
    # combined support inflates with 4-5 near-identical columns. The cheap
    # cure is a pre-MRMR dedup pass against the engineered cousins: keep the
    # first appended occurrence, drop everything correlating >= 0.999 with an
    # already-kept engineered column. Raw input columns are never deduped
    # here - that's MRMR's job and removing raw cols would change the
    # ``feature_names_in_`` contract.
    # Order-preserving dedup BEFORE we walk the list: when the same
    # engineered name is emitted by both the hybrid_orth and the
    # mi_greedy stages (e.g. both produce ``square(x1)`` under a
    # signal-driven recipe), ``X[name]`` selects a 2-column DataFrame
    # rather than a Series and the downstream ``.rank()`` call
    # explodes with ``Data must be 1-dimensional``. The dedup also
    # short-circuits the inner O(K^2) pairwise rank-correlation loop
    # for the trivial perfect-name-match case.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_dedup import _dedup_engineered_across_stages
    X = _dedup_engineered_across_stages(self, _y_np, X, recipes, verbose)

    # Layer 91: Tier-2 UNIFIED SECOND-PASS CMI GATE. The Layer 27
    # dedup above is UNSUPERVISED (Spearman rank-corr between engineered cousins)
    # and so cannot see cross-mechanism redundancy that only manifests
    # conditional on y - e.g. ``count(cat_a)`` and ``freq(cat_a)`` ARE caught by
    # Spearman (identical rank order), but ``count(cat_a)`` vs a target-encoding
    # of cat_a that carries the same y-signal through a different bin pattern is
    # NOT. This gate runs a single greedy CMI selection over ALL engineered
    # columns (every mechanism) conditioned on the running support seeded from
    # the top raw-MI columns, keeping only columns that add new information about
    # y on top of raw + earlier-selected engineered columns. Default OFF (byte-
    # identical legacy path). y is read only here at fit; transform replays the
    # surviving recipes without y.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._engineered_gates import _second_pass_cmi_gate
    X = _second_pass_cmi_gate(self, X, _y_np, recipes, verbose)

    # Layer 23: feature_names_in_ MUST exclude hybrid-appended columns so
    # the end-of-fit ``selected_vars_names`` lookup routes hybrid names
    # into ``_engineered_features_`` / ``_engineered_recipes_`` instead of
    # the raw-feature ``original_indices`` path. transform() then replays
    # hybrid columns from recipes and the sklearn ``n_features_in_``
    # contract still matches the user-facing input width.
    # Layer 26: also exclude MI-greedy-appended columns - same routing
    # contract: they're engineered, not raw input.
    _hybrid_names_set = set(self.hybrid_orth_features_ or [])
    _mi_greedy_names_set = set(self.mi_greedy_features_ or [])
    _engineered_names_set = _hybrid_names_set | _mi_greedy_names_set
    _all_cols = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
    # Defense in depth (Layer 64 finding 2026-05-31): if any FE stage
    # accidentally appended a column under a name already present in
    # X (e.g. two recipe families converging on the same canonical
    # ``square(x1)`` label, or a stage re-emitting an input column it
    # picked up from a previous stage), pandas downstream raises
    # ``cannot reindex on an axis with duplicate labels`` when
    # ``X.loc[:, target_names] = vals`` runs the target injection.
    # Drop in-place: keep the FIRST occurrence (which is the original
    # raw input column or the first stage's emission), drop later
    # duplicate-named columns, and prune the engineered roster of any
    # name that was effectively shadowed so the recipe ledger stays
    # consistent with the column actually surviving in X.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _finalise_feature_names_in
    X, _fni = _finalise_feature_names_in(self, X, _all_cols, _engineered_names_set, verbose)
    # ndarray (not list) to match sklearn's own feature_names_in_ contract (BaseEstimator._check_feature_names) and
    # every other MRMR fit-path assignment (_mrmr_class_fit_helpers.py); a plain list here was the one straggler
    # that made ``==`` comparisons against a list/array ambiguous for callers expecting the canonical type.
    self.feature_names_in_ = np.asarray(_fni, dtype=object)
    self.n_features_in_ = len(self.feature_names_in_)

    # FE AUTO-ESCALATION fitting target: a RANK transform of the raw
    # numeric y, stashed for the escalation proposers' corr-based warp fits. The FE
    # step's ``classes_y`` are LABEL codes from the internal target quantisation
    # (NOT guaranteed ordinal/monotone in y - measured 37 unordered codes on a
    # heavy-tailed regression y), which destroys a Pearson-corr-validated ALS /
    # periodogram fit; the rank of y is monotone-equivalent to y, heavy-tail-robust,
    # and exactly as leak-safe (a fit-time supervised target; every emitted recipe
    # stays a closed-form function of x). Deleted at fit end (transient, keeps the
    # pickle slim). Non-numeric / multi-output y -> None (escalation falls back to
    # ``classes_y`` codes).
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _stash_fe_targets
    _stash_fe_targets(self, _y_np, X)

    # ---------------------------------------------------------------------------------------------------------------
    # Temporarily inject targets
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _inject_targets
    X, _is_polars_input, target_names = _inject_targets(self, y, X)

    # ---------------------------------------------------------------------------------------------------------------
    # MEM: force a GC pass before discretizing (PEAK-RSS bound). This used to also explicitly ``del``
    # ~50 per-family FE intermediate DataFrames (``X_t``/``X_q``/``X_te``/...) that used to live as
    # ``_fit_impl`` locals; every one of those families has since been carved into a cascade sibling
    # (_fe_stage_cascade_early_a.py, _fe_stage_cascade_early_b.py, _fe_stage_cascade_mid_a.py,
    # _fe_stage_cascade_mid_b.py, _hybrid_orth_family_variants.py) across four Tier F waves, so each
    # intermediate is now genuinely function-local to its own sibling and reclaimed automatically on
    # that function's return -- the explicit per-name ``del`` block is fully retired (ruff F821 on
    # each remaining bare ``del X_t``-style name, confirmed unbound anywhere in this scope, is what
    # caught each wave's leftover dead entries). The GC pass itself is still worth keeping: many
    # families above build full-width DataFrames as local variables INSIDE their own sibling
    # function, which Python's refcounting already frees on return, but an explicit collect here
    # still helps against any reference cycle (e.g. a DataFrame's own internal caches) before the
    # ``categorize_dataset`` peak.
    import gc as _gc
    _gc.collect()

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
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _discretize_inputs
    _nbins_strategy, _x_for_cat, cols, data, nbins = _discretize_inputs(self, _is_polars_input, X, target_names)

    # Which columns the cardinality pre-screen may judge by bin count. Under a SUPERVISED nbins strategy (the
    # default MDLP) a numeric column earns bins for explaining the target, so its bin count is a signal-strength
    # measure, not a cardinality one -- feeding it to a "too many levels" guard drops the best feature. Only
    # genuinely categorical columns, whose bins ARE their levels, stay eligible. With no supervised strategy every
    # bin count is unsupervised again and the guard applies to all of them, as before.
    _numeric_names = numeric_column_names(_x_for_cat) if _nbins_strategy else set()

    # 2026-07-11 perf: speculatively pre-warm the polynom-pair-FE loky pool here, AFTER categorization (not
    # before it). ``run_polynom_pair_fe`` is otherwise the pool's first user in a typical fit (the sibling CPU
    # pair-MI-sweep pool in ``_step_pairmi.py`` only engages when the GPU MI path fails), so production pays a
    # full cold-start (16 fresh worker processes each re-importing mlframe/numba/cupy, measured 28.1s cold vs
    # 0.7s warm) synchronously inside the polynom-pair-FE phase. Placed HERE, not at fit-entry: an earlier
    # placement (right after ``fe_smart_polynom_iters``/``n_jobs`` are read, well before categorization) was
    # measured to REGRESS a full 100k-row production run by ~223s wall-clock - categorization is itself
    # CPU-active (not idle), so the pre-warm's 16 concurrent worker-process spawns contended with it (round-12
    # A/B: the categorization gap grew 85.3s -> 153.1s, with exactly 16 new
    # "NumbaPerformanceWarning: Grid size 1" lines appearing in that window, matching n_jobs=16 workers
    # bootstrapping). The GPU pair-MI screening dispatched right after this point is genuinely GPU-bound
    # (blocks on ``.get()``/``copy_to_host``), leaving CPU free for the pre-warm to actually overlap with idle
    # time instead of stealing it. See ``maybe_prewarm_polynom_loky_pool``'s docstring for the pool-reuse
    # mechanics and the ``idle_worker_timeout`` fix (the pool must survive until ``run_polynom_pair_fe`` uses
    # it, several minutes later).
    from .._joblib_safe import maybe_prewarm_polynom_loky_pool

    maybe_prewarm_polynom_loky_pool(fe.smart_polynom_iters, n_jobs)

    # ``cols`` is a list; per-name ``cols.index`` is an O(len(cols)) scan, so resolving every target /
    # categorical name that way is O(C*P). Build a name->index map once and reuse it for both lookups.
    _name_to_idx = {c: i for i, c in enumerate(cols)}

    target_indices = np.array([_name_to_idx[col] for col in target_names], dtype=np.int64)

    # TARGET REBIN GUARD. The adaptive per-column ``nbins_strategy``
    # (default ``"mdlp"`` since Wave 7) is meant for FEATURE columns; applied to the
    # injected TARGET column it is SELF-REFERENTIAL (MDLP bins y supervised on y) and
    # on a heavy-tailed continuous y it produces a DEGENERATE encoding - measured on
    # the F2 fixture (y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3), n=20000): 37 bins with
    # 83.7% of all rows collapsed into ONE bin (vs the clean 10 x 2000 equal-frequency
    # legacy quantile bins). Every downstream MI/CMI - screening, pair gates, FE
    # acceptance - is computed AGAINST these target codes, so the bulk of the signal
    # becomes invisible (the genuine (c,d) term's measured CMI drops ~6x). Re-bin the
    # CONTINUOUS target columns (raw unique count > quantization_nbins; classification
    # labels are left untouched) with the legacy ``quantization_method`` /
    # ``quantization_nbins`` equal-frequency quantile path. No-op when
    # ``nbins_strategy`` is None (legacy fits already bin the target this way).
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _rebin_continuous_targets
    _rebin_continuous_targets(self, _nbins_strategy, target_indices, cols, _x_for_cat, nbins, data, verbose)

    # COMPACT CODES STORAGE. ``data`` holds per-column BIN INDICES (0..nbins-1 + a NaN bin / -1 sentinel), never JOINT
    # ids, so it fits the smallest int that spans its actual code range - int8 for the common nbins<=~127 case, int16
    # for a high-cardinality categorical. The base (n, p) matrix at scale (e.g. 795k x 496) drops 4x / 2x vs the legacy
    # int32. Selection-EQUIVALENT: the code VALUES are unchanged, and every consumer (merge_vars, the GPU path) reads
    # this storage and casts UP to int32 for JOINT math, so deep joints (nbins^order) never overflow. Engineered-code
    # appends downstream re-narrow to this dtype (``_append_codes``). Range-checked directly (one min/max pass) rather
    # than trusting nbins semantics. Opt out: MLFRAME_MRMR_COMPACT_CODES=0.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _compact_code_storage
    data = _compact_code_storage(data)

    # ---------------------------------------------------------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _categorical_var_names
    categorical_vars_names = _categorical_var_names(_is_polars_input, X)
    categorical_vars = [_name_to_idx[col] for col in categorical_vars_names]

    if fe.max_steps > 0:
        from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._transformations import _build_fe_transformations
        binary_transformations, checked_pairs, engineered_features, unary_transformations = _build_fe_transformations(self, fe, verbose)
    # engineered_recipes (name -> EngineeredRecipe) is initialised unconditionally; the splitter at the bottom
    # of fit() looks it up regardless of fe_max_steps. Stays empty when FE is disabled.
    engineered_recipes: dict = {}
    # PER-GATE FE REJECTION LEDGER (additive, 2026-06-11): the per-fit raw-record list is reset
    # near fit-start (above, before any FE stage records) so it accumulates the gate drops of
    # EVERY FE stage this fit - the recipe-FE families + cluster-basis (which record before this
    # point) AND the pair-search ``_run_fe_step`` loop below. fe_rejection_ledger_ is built from
    # it at fit-end. Stays empty when FE produced no rejected candidates.
    # Seed engineered_recipes with every recipe family built above (before the screening loop): the end-of-fit remap routes any
    # selected_vars_name matching a key here into _engineered_recipes_. Order is ROUTED_RECIPE_FAMILIES (later families win a key clash).
    for _family in ROUTED_RECIPE_FAMILIES:
        engineered_recipes.update(getattr(recipes, _family))
    # Reset per fit so a re-fit on the same instance doesn't carry stale cluster-aggregate state.
    self._cluster_aggregate_removals_ = []
    self.cluster_aggregate_ = []  # fitted summary (per-aggregate records) -> meta_info report

    # Cat-FE step (categorical interaction generator). Runs once before the screening loop when
    # ``cat_fe_config.enable=True``; augments data/cols/nbins with ordinal-encoded columns capturing pair
    # (and future k-way) synergies. Engineered cols enter screening as atomic 1-way features.
    cat_fe_cfg = getattr(self, "cat_fe_config", None)
    self._cat_fe_state_ = None
    # ``None`` means "use default CatFEConfig()" which has enable=True. Pass CatFEConfig(enable=False) for legacy.
    if cat_fe_cfg is None:
        from ..cat_fe_state import CatFEConfig as _CatFEConfig
        cat_fe_cfg = _CatFEConfig()
    # include_numeric: collect raw numeric feature values (keyed by data-column index) so the cat-FE step can
    # quantile-bin them into the candidate pool. Extracted from the ORIGINAL ``X`` (NaN visible) - NOT the
    # ffill'd ``_x_for_cat`` - so a NaN-bearing column is correctly skipped at fit (v1 has no NaN bin in the
    # quantile-edge replay) and fit/transform stay consistent (both read the user's raw frame).
    _num_raw_values = None
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._cat_fe import _collect_numeric_raw_values
    _num_raw_values = _collect_numeric_raw_values(self, cat_fe_cfg, categorical_vars, target_indices, cols, _include_numeric_input_nan_cols, X, _num_raw_values)
    _cat_fe_pool_size = len(categorical_vars) + (len(_num_raw_values) if _num_raw_values else 0)
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._cat_fe import _run_categorical_fe
    categorical_vars, cols, data, nbins = _run_categorical_fe(self, cat_fe_cfg, _cat_fe_pool_size, data, target_indices, nbins, dtype, cols, categorical_vars, _num_raw_values, verbose, _is_polars_input, _x_for_cat, engineered_recipes)

    # Resolve effective ``min_relevance_gain`` against the target entropy. ``'relative_to_entropy'`` mode uses ``min_relevance_gain_frac * H(y)`` so the stop floor scales with how much information the target actually carries; ``'absolute'`` mode retains the legacy verbatim value. The target is already discretized into bins (``data[:, target_indices[0]]`` with bin count ``nbins[target_indices[0]]``); ``np.bincount`` + Shannon entropy in nats matches the screen_predictors estimator family.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._prelude import _resolve_min_relevance_gain
    _effective_min_relevance_gain = _resolve_min_relevance_gain(self, target_indices, data, nbins, verbose)
    self._effective_min_relevance_gain_ = _effective_min_relevance_gain

    num_fs_steps = 0
    # Tracks whether the post-FE confirming re-screen has run, so it
    # fires at most once (see the fe_reselect_after_engineering block below). The
    # re-screen re-selects from the augmented pool (raw + engineered) using the
    # estimator's own use_simple_mode (now defaulting to False = full Fleuret
    # conditional-MI redundancy), which is what drops engineered columns redundant
    # given an already-selected one and records a real gain for every survivor.
    _did_confirm_rescreen = False
    # Carries the DCDState from the prior screen pass into the post-FE
    # confirm-rescreen so cluster discovery (anchor graph, pruned mask,
    # swap_log) accumulates instead of being rebuilt empty each iteration.
    _persisted_dcd_state = None
    # Carries the prior round's relevance/redundancy caches into the next screen_predictors() call
    # - see screen_predictors' ``seed_caches`` docstring. Mirrors the DCD-state
    # threading immediately above; before this fix each round rebuilt all 4 caches from scratch, fully
    # rescoring every raw column's relevance/entropy/conditional-MI even though those values cannot
    # change round-to-round (the data they're computed from is stable; only new columns get appended).
    _persisted_screen_caches = None
    # Cross-round cache for the maxT permutation-null gain floor (2026-07-09 fix; see
    # compute_fdr_gain_floor's ``maxt_floor_cache`` docstring) - a plain dict, mutated in place by
    # every screen_predictors() call this fit, so a raw-pool floor computed in round 1 is not
    # recomputed identically in round 2/3.
    _persisted_maxt_floor_cache: dict = {}
    # Carries the warmed joblib worker pool (n_workers>1 only) into the next screen_predictors() call
    # - see screen_predictors' ``seed_workers_pool`` docstring. ``None`` at n_workers<=1
    # (no pool built) or before round 1.
    _persisted_workers_pool = None
    # Declared BEFORE the loop so per-binary-func timing accumulates across ALL
    # screen/FE rounds of this fit, not just the last one - previously reset to empty at the top of
    # every iteration, so the end-of-fit log only ever reflected the FINAL round's timing even though
    # this loop typically runs 2-3 rounds per fit (raw screen, FE step(s), confirm-rescreen).
    from ._fit_impl_stages._screen_args import _cat_fe_lineage, _dcd_config, _effective_max_confirmation_cand_nbins, _screen_factors_names_to_use

    from ._fit_impl_stages._screen_round import _adaptive_relax_retry, _fe_step_params, _runtime_budget_exhausted, _sufficient_summary_reached
    times_spent: defaultdict = defaultdict(float)
    while True:
        n_recommended_features = 0
        # Resolve the fit's ONE shared row draw BEFORE the screen so the order-1 relevance sweep + FDR
        # floor score on it (screen is the first consumer -> caches the draw -> the FE step reuses the
        # SAME rows). None at small n -> full-n screen, unchanged.
        try:
            from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
            _screen_shared_idx = _get_shared_fe_subsample_idx(self, np.asarray(data[:, int(target_indices[0])]), len(data))
        except Exception as _sub_exc:
            # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
            log_throttle(logger, "mrmr_fit_shared_fe_subsample_failed", logging.WARNING, "mrmr: shared FE subsample resolution failed; screening at FULL n: %r", _sub_exc, exc_info=True)
            _screen_shared_idx = None
        (
            selected_vars,
            predictors,
            _any_influencing,
            entropy_cache,
            cached_MIs,
            cached_confident_MIs,
            cached_cond_MIs,
            classes_y,
            classes_y_safe,
            freqs_y,
            _dcd_state,
            _persisted_workers_pool,
        ) = screen_predictors(
            # Recomputed from the CURRENT ``cols``: the cat-interaction FE step rebinds data/cols/nbins with
            # engineered columns, and a crossed categorical can carry real cardinality, so it must stay
            # eligible for the ceiling rather than inherit an exemption computed before it existed.
            raw_cardinality_cols=(None if not _nbins_strategy else {c for c in cols if c not in _numeric_names}),
            factors_data=data,
            y=target_indices,  # type: ignore[arg-type]
            subsample_idx=_screen_shared_idx,
            factors_nbins=nbins,
            factors_names=cols,
            # Layer 23: when hybrid orth FE appended columns, extend the
            # candidate pool to include them so they reach the screening
            # gates. When the caller did not pin factors_names_to_use,
            # screen_predictors uses every column from ``cols`` so the
            # hybrid cols are naturally included.
            factors_names_to_use=_screen_factors_names_to_use(self),
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
            # 2026-06-02 RC2 — sample-size-aware Fleuret confirmation threshold.
            fe_confirm_undersample_rows_per_cell=float(getattr(self, "fe_confirm_undersample_rows_per_cell", 5.0) or 0.0),
            # stopping conditions
            min_relevance_gain=_effective_min_relevance_gain,
            min_relevance_gain_relative_to_first=float(getattr(self, "min_relevance_gain_relative_to_first", 0.0)),
            cardinality_bias_correction=bool(getattr(self, "cardinality_bias_correction", True)),
            max_consec_unconfirmed=self.max_consec_unconfirmed,
            max_runtime_mins=self.max_runtime_mins,
            interactions_min_order=self.interactions_min_order,
            interactions_max_order=self.interactions_max_order,
            interactions_order_reversed=self.interactions_order_reversed,
            max_veteranes_interactions_order=self.max_veteranes_interactions_order,
            only_unknown_interactions=self.only_unknown_interactions,
            # Resolve effective max_confirmation_cand_nbins: user-pinned wins, else formula default.
            max_confirmation_cand_nbins=_effective_max_confirmation_cand_nbins(self),
            # FE-on-empty-screen fallback flag (consumed by MRMR.fit).
            fe_fallback_to_all=self.fe_fallback_to_all,
            # verbosity and formatting
            verbose=self.verbose,
            ndigits=self.ndigits,
            parallel_kwargs=self._effective_parallel_kwargs(),
            stop_file=self.stop_file,
            # engineered_lineage from cat-FE step (None when cat-FE didn't run); screen uses it to skip
            # redundant (orig_parent, engineered_col) k-way candidates.
            engineered_lineage=_cat_fe_lineage(self),
            # 2026-05-30 Wave 9 — DCD config forward. Built only when
            # ``dcd_enable=True`` (per Critic1/F: passed as kwargs, NOT
            # via thread-local, for joblib parallel-backend safety).
            dcd_config=_dcd_config(self, X),
            # 2026-05-31 Layer 43 (PART A) — thread the local
            # engineered_recipes dict into screen so DCD's commit_swap can
            # register the PC1 aggregate as a replayable EngineeredRecipe.
            # Pre-fix the dict was inaccessible from screen and the swap
            # silently dropped the aggregate from ``_engineered_recipes_``.
            engineered_recipes=engineered_recipes,
            # 2026-06-02 — directed-FE tie-break: pass the snapshot of the
            # ORIGINAL user input columns (taken before any FE stage appended
            # engineered intermediates). screen_predictors uses it to mark
            # any candidate whose name is not in this set as engineered and,
            # on a near-tie in selection gain, prefer the engineered transform
            # over its raw parent (e.g. x1__He2 over x1 for an even-symmetric
            # target). Applies in BOTH the first screen and the post-FE
            # confirming re-screen (this same call runs in the while-loop).
            raw_feature_names=_raw_input_cols_pre_fe,
            # Thread the prior pass's DCDState so cluster discovery
            # accumulates across the confirm-rescreen (the matrix only
            # grows; raw indices are stable). Without this the rescreen
            # rebuilds an empty state and the published dcd_ summary loses
            # the screen-1 dup cluster (n_pruned/cluster_anchors reset).
            existing_dcd_state=_persisted_dcd_state,
            seed_caches=_persisted_screen_caches,
            seed_maxt_floor_cache=_persisted_maxt_floor_cache,
            seed_workers_pool=_persisted_workers_pool,
        )
        if _dcd_state is not None:
            _persisted_dcd_state = _dcd_state
        _persisted_screen_caches = (entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs)
        # 2026-05-30 Wave 9 — stash DCD summary on the estimator for the
        # public ``dcd_`` attribute (None when DCD was disabled).
        from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._screen_round import _attach_dcd_results
        cols, data, nbins = _attach_dcd_results(self, _dcd_state, X, data, cols, nbins)

        # MEMORY: prune fit-time ``_engineered_continuous_`` scratch for engineered columns that did not
        # survive THIS round's screen - see ``_prune_engineered_continuous_store`` docstring for why this
        # is safe (the FE operand pool only widens beyond ``selected_vars`` on the very first FE step,
        # before any engineered column exists). No-op when the store is empty/absent.
        if getattr(self, "_engineered_continuous_", None):
            from ._helpers import _prune_engineered_continuous_store
            _prune_engineered_continuous_store(self, cols, selected_vars)

        if fe.max_steps == 0 or num_fs_steps >= fe.max_steps:
            break

        if _runtime_budget_exhausted(self, start_time, num_fs_steps, verbose):
            ran_out_of_time = True
            break

        # SUFFICIENT-SUMMARY EARLY-STOP. The user's
        # "compare-to-theoretical-max" idea via a DPI residual test. Once the
        # current selection already captures all the information the observables
        # carry about y - i.e. the residual r = y - E_hat[y|selected] is pure
        # noise w.r.t. EVERY raw feature (all raws at the maxT permutation null)
        # AND small relative to y (Var(r)/Var(y) guard) - any future engineered
        # candidate is, by the Data-Processing Inequality, a function of the raws
        # and CANNOT have more MI with r than the raws do, so the remaining FE
        # search is provably pointless. Skip it. This NEVER changes the final
        # selection (it only skips work that could find nothing - with it OFF the
        # loop would run the remaining steps and engineer nothing new); verified
        # byte-identical on genuine multi-signal fixtures. CONSERVATIVE: stops only
        # when BOTH guards pass, so a genuine unfound second signal (incl. a
        # NONLINEAR leftover the linear E_hat underfits, caught by MI(r; raw))
        # blocks the stop. ``self.sufficient_summary_`` surfaces the verdict.
        if _sufficient_summary_reached(self, data, nbins, cols, selected_vars, target_indices, X, y, num_fs_steps, verbose):
            break

        # Feature engineering iteration delegated to ``_run_fe_step`` (testable / experiment-friendly outside
        # the screening loop). Returns updated state + n_recommended_features; zero breaks the outer loop.
        self._fe_steps_executed_ += 1
        _step_kw = dict(
            target_names=target_names, target_indices=target_indices,
            categorical_vars=categorical_vars,
            classes_y=classes_y, classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
            unary_transformations=unary_transformations,
            binary_transformations=binary_transformations,
            engineered_features=engineered_features,
            engineered_recipes=engineered_recipes,
            times_spent=times_spent,
            num_fs_steps=num_fs_steps,
            n_jobs=n_jobs, prefetch_factor=prefetch_factor,
            parallel_kwargs=parallel_kwargs,
            _is_polars_input=_is_polars_input,
            verbose=verbose,
            **_fe_step_params(fe),
        )
        fe_result = self._run_fe_step(data=data, cols=cols, nbins=nbins, X=X, selected_vars=selected_vars, checked_pairs=checked_pairs, **_step_kw)
        if fe_result is None:
            break  # FE skip: empty screening + fe_fallback_to_all=False
        data, cols, nbins, X, selected_vars, n_recommended_features = fe_result

        # Adaptive threshold relaxation: when the very first FE step engineers nothing, retry it ONCE with relaxed thresholds.
        if n_recommended_features == 0 and fe.max_steps > 0 and num_fs_steps == 0:
            fe_result_retry = _adaptive_relax_retry(self, fe, _step_kw, data, cols, nbins, X, selected_vars, verbose)
            if fe_result_retry is not None:
                data, cols, nbins, X, selected_vars, n_recommended_features = fe_result_retry

        if n_recommended_features == 0:
            break

        num_fs_steps += 1
        if num_fs_steps >= fe.max_steps:
            # CONFIRM-RESCREEN: the FE step appended engineered
            # columns and (legacy) promoted them into ``selected_vars`` BY FIAT,
            # bypassing redundancy filtering + gain accounting. Instead of
            # breaking here, loop ONCE more so the top-of-loop ``screen_predictors``
            # re-selects from the AUGMENTED pool. The engineered columns are
            # already quantised bin-code columns in ``data``/``cols``/``nbins``,
            # so MRMR treats them as ordinary candidates: a redundant engineered
            # feature (e.g. ``1/b - d**2`` whose conditional MI given an
            # already-selected ``a**2/b`` is ~0.03) is dropped by the Fleuret
            # redundancy term, and every surviving column - raw OR engineered -
            # earns a real ``mrmr_gain`` / ``support_rank``. The next iteration
            # hits the ``num_fs_steps >= fe_max_steps`` break at the TOP of the
            # loop BEFORE the FE step, so FE never runs again - no
            # unbounded recursion, no new engineered columns.
            if getattr(self, "fe_reselect_after_engineering", True) and n_recommended_features > 0 and not _did_confirm_rescreen:
                _did_confirm_rescreen = True
                continue
            break  # uncomment to avoid recheck of single-rounded FE

    # ENGINEERED-OPERAND FEED-FORWARD: the continuous engineered-value
    # store is FIT-TIME SCRATCH (full-length float64 arrays of training data) used
    # only to feed engineered operands into the next FE step's pair search. Drop it
    # once the FE loop is done so it never bloats the fitted estimator or breaks
    # pickle (the replayable composite carries only its parent recipes, never these
    # arrays). No-op when the attr was never created (no engineered columns).
    # SNAPSHOT FIRST: the raw-vs-engineered conditional-redundancy drop
    # below needs the CONTINUOUS engineered values to bin the engineered survivor
    # finely (the ``data`` matrix holds only the lossy ~10-code screening bins, which
    # leave a fully-subsumed denominator operand a spurious residual CMI). Snapshot
    # into a LOCAL (never an attr -> stays out of the pickled estimator) so the del
    # below still keeps the fitted object lean.
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._screen_round import _take_engineered_continuous_store
    _eng_continuous_snapshot = _take_engineered_continuous_store(self)

    # Surfaced at verbose>=1 (2026-07-09; was gated behind verbose>2, an unrealistically high bar that
    # left this cumulative-per-operator timing breakdown effectively invisible to normal production runs).
    if verbose and times_spent:
        logger.info("MRMR FE time spent by binary func (cumulative across all rounds): %s", sort_dict_by_value(times_spent))
    # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

    # ---------------------------------------------------------------------------------------------------------------
    # Drop temporary targets
    # ---------------------------------------------------------------------------------------------------------------

    # Fuzz-caught: previous ``X = X.drop(columns=target_names)`` returned a new DataFrame and only rebound the
    # local; for pandas input (where X.loc[:, target_names] = ... mutated the caller's frame), the caller's
    # X was left with the injected ``targ_<id>`` columns, which leaked into downstream sklearn pipeline
    # (imputer/scaler recorded them in feature_names_in_ and raised on transform). Fix: drop in place (pandas)
    # or rebind (polars - immutable, caller's X was never mutated).
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _drop_temporary_targets
    X, selected_vars = _drop_temporary_targets(_is_polars_input, X, target_names, _dcd_state, selected_vars)

    # ---------------------------------------------------------------------------------------------------------------
    # Friend-graph post-analysis (diagnostic; optional pruning). Built here, while ``selected_vars``,
    # ``data``, ``nbins`` and ``target_indices`` are all still in cols-space, BEFORE the remap below
    # rebinds ``selected_vars`` to original-frame indices. When pruning is enabled the pruned cols-space
    # list flows through that same remap into ``support_``. Never allowed to break fit - guarded.
    # ---------------------------------------------------------------------------------------------------------------
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _friend_graph_analysis
    cols, data, nbins, selected_vars = _friend_graph_analysis(self, X, classes_y, cols, data, nbins, target_indices, y, verbose, cached_MIs, engineered_recipes, _eng_continuous_snapshot, selected_vars, _effective_min_relevance_gain, recipes, _persisted_dcd_state, _y_np, _fe_family_on)
    # ---------------------------------------------------------------------------------------------------------------
    # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _map_selected_to_original_indices
    _fni_idx, selected_vars = _map_selected_to_original_indices(self, cols, selected_vars, verbose, engineered_recipes)

    # ---------------------------------------------------------------------------------------------------------------
    # additional_rfecv run
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _run_additional_rfecv
    _run_additional_rfecv(self, X, selected_vars, verbose, y, categorical_vars_names, _fni_idx)

    # ---------------------------------------------------------------------------------------------------------------
    # Assign support
    # ---------------------------------------------------------------------------------------------------------------

    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._post_selection import _assign_support
    _assign_support(self, X, classes_y, cols, data, nbins, target_indices, y, verbose, fe, cached_MIs, engineered_recipes, predictors, _eng_continuous_snapshot, selected_vars)

    # ---------------------------------------------------------------------------------------------------------------
    # Report FS results
    # ---------------------------------------------------------------------------------------------------------------
    from ._finalise import _finalise_fs_results

    return _finalise_fs_results(
        self,
        MRMR=MRMR,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        predictors=predictors,
        start_time=start_time,
        verbose=verbose,
        cache_key=_cache_key,
        signature=signature,
        ran_out_of_time=ran_out_of_time,
        hashable_params_signature=_hashable_params_signature,
        mrmr_cache_bytes_total=_mrmr_cache_bytes_total,
        align_mrmr_gains=_align_mrmr_gains,
        fit_cache_lock=_MRMR_FIT_CACHE_LOCK,
    )
