"""Configuration, default-resolution, seeding, and target-dtype helpers carved out of the MRMR class body.

Pure move from ``_mrmr_class`` into a mixin. ``MRMR`` inherits ``_MRMRConfigMixin`` (with ``BaseEstimator`` /
``TransformerMixin`` kept first in the MRO), so every ``self`` / ``cls`` reference resolves against the concrete
``MRMR`` instance -- the class attributes that stay on ``MRMR`` (``_FIT_CACHE`` / ``_FAST_SEARCH_OVERRIDES`` /
``_DEFAULT_SCREEN_SUBSAMPLE_N`` / ``__init__``) resolve via the MRO.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


class _MRMRConfigMixin:
    """Config / defaults / seed / target-dtype helpers for :class:`MRMR` (see module docstring).

    The class attributes below are declared here (not assigned) so mypy resolves them on ``cls``/``self``;
    the concrete values live on ``MRMR`` (the sibling class this mixin is combined with via multiple
    inheritance), which sets ``_FIT_CACHE`` / ``_FAST_SEARCH_OVERRIDES`` / ``_DEFAULT_SCREEN_SUBSAMPLE_N`` as
    class attributes and ``random_seed`` / ``verbose`` / ``cv`` / ``cv_shuffle`` as constructor params.
    """

    _FIT_CACHE: Any
    _FAST_SEARCH_OVERRIDES: Any
    _DEFAULT_SCREEN_SUBSAMPLE_N: ClassVar[int]
    random_seed: Any
    random_state: Any
    verbose: Any
    cv: Any
    cv_shuffle: Any

    @classmethod
    def clear_fit_cache(cls) -> int:
        """Drain the process-wide MRMR fit cache. Returns the entry count that was dropped. Call between
        suites (model retraining boundary, JupyterHub kernel reuse, web-service request boundary) when
        long-lived workers must release fitted-MRMR memory. Without this, the cache holds up to
        ``fit_cache_max`` (default 4) full MRMR instances per process for as long as the process lives."""
        n = len(cls._FIT_CACHE)
        cls._FIT_CACHE.clear()
        return n

    @classmethod
    def _fast_search_default_subsample_n(cls) -> int:
        """Screen subsample size for the fast FE pair search, resolved via the kernel_tuning_cache when a
        tuned entry exists for this host, else a safe HW-agnostic fallback. The MI/CMI pair screen is
        rank-stable under subsampling and the FINAL survivor columns are replayed at full n, so this only
        bounds the screen cost. Never hardcode per-HW thresholds (mlframe is shared infra); the cache lets
        a quiet/large-RAM box record a better value. Fallback 90_000: the smallest screen-n that kept the
        warped (c,d) interaction's selection bit-stable on the n=100k synthetics (75k/85k flipped the
        (a,b) composite at the MI tie; 90k did not)."""
        _fallback = 90_000
        try:
            from .._kernel_tuning import get_kernel_tuning_cache

            _cache = get_kernel_tuning_cache()
            if _cache is not None:
                tuned = _cache.lookup("mrmr_fast_search_screen_n")
                if tuned:
                    _v = int(tuned.get("subsample_n", 0) or 0)
                    if _v > 0:
                        return _v
        except Exception as exc:
            logger.debug("mrmr: fast_search screen_n KTC lookup failed; using fallback %d: %r", _fallback, exc, exc_info=True)
        return _fallback

    @classmethod
    def _default_screen_subsample_n(cls) -> int:
        """Feature-recovery default screen-subsample size, kernel_tuning_cache-resolved when a tuned entry
        exists for this host, else the HW-agnostic ``_DEFAULT_SCREEN_SUBSAMPLE_N`` floor. See the class
        attribute docstring for the rank-stability rationale + measured wins."""
        _fallback = int(cls._DEFAULT_SCREEN_SUBSAMPLE_N)
        try:
            from .._kernel_tuning import get_kernel_tuning_cache

            _cache = get_kernel_tuning_cache()
            if _cache is not None:
                tuned = _cache.lookup("mrmr_default_screen_n")
                if tuned:
                    _v = int(tuned.get("subsample_n", 0) or 0)
                    if _v > 0:
                        return _v
        except Exception as exc:
            logger.debug("mrmr: default screen_n KTC lookup failed; using fallback %d: %r", _fallback, exc, exc_info=True)
        return _fallback

    def _apply_default_screen_subsample(self, n_rows: int) -> dict:
        """Shrink the FE/MI screen subsamplers to the feature-recovery default for large n, returning
        {attr: pre_fit_value} to restore in ``finally``. Applies UNCONDITIONALLY (not gated on
        ``fe_fast_search``) so the default ``MRMR()`` fit subsamples the SCREEN at large n. Honours user
        intent: a knob is only touched when it is still at its package default, and only SHRUNK (never
        raised). No-op when ``n_rows`` is below the resolved screen size (small-n behaviour is unchanged --
        the subsamplers treat subsample_n>=n as full-n). The selected columns are replayed at full n."""
        import inspect

        saved: dict = {}
        try:
            _defaults = {p.name: p.default for p in inspect.signature(type(self).__init__).parameters.values() if p.default is not inspect._empty}
        except Exception as exc:
            logger.debug("mrmr: ctor-default introspection failed in _apply_default_screen_subsample; leaving knobs unchanged: %r", exc, exc_info=True)
            return saved
        _screen_n = self._default_screen_subsample_n()
        # Below the screen size there is nothing to subsample -- leave the knobs at their (full-n) default
        # so small-n fits are byte-identical to legacy.
        if not (isinstance(n_rows, int) and n_rows > _screen_n):
            return saved
        for _attr in ("fe_check_pairs_subsample_n", "fe_smart_polynom_subsample_n"):
            if _attr not in _defaults:
                continue
            _cur = getattr(self, _attr, None)
            # Only when the user left it at the package default (explicit user value always wins).
            if _cur != _defaults[_attr]:
                continue
            # Only SHRINK: a default of 0/None means "full-n"; treat that as +inf so we still shrink it.
            _cur_eff = int(_cur) if (isinstance(_cur, int) and _cur > 0) else n_rows
            if _screen_n < _cur_eff:
                saved[_attr] = _cur
                setattr(self, _attr, _screen_n)
        return saved

    def _apply_fast_search_profile(self) -> dict:
        """Override the fast-search sub-knobs for this fit, returning {attr: pre_fit_value} to restore in
        ``finally``. Only knobs still at their package default are touched (explicit user value wins). See
        the ``fe_fast_search`` __init__ docstring for the rationale and measured wins."""
        import inspect

        saved: dict = {}
        try:
            _defaults = {p.name: p.default for p in inspect.signature(type(self).__init__).parameters.values() if p.default is not inspect._empty}
        except Exception as exc:
            logger.debug("mrmr: ctor-default introspection failed in _apply_fast_search_profile; treating all knobs as user-set: %r", exc, exc_info=True)
            _defaults = {}

        def _override(attr, fast_value):
            """Set ``attr`` to ``fast_value`` unless the caller already customized it away from the package default."""
            cur = getattr(self, attr, None)
            # Only override when the user left it at the package default (or the attr is absent).
            if attr in _defaults and cur != _defaults[attr]:
                return
            saved[attr] = cur
            setattr(self, attr, fast_value)

        for _attr, _val in self._FAST_SEARCH_OVERRIDES:
            _override(_attr, _val)
        # Subsample is HW/size-dependent -> resolve via kernel_tuning_cache. Only shrink it (never raise a
        # user who already set a smaller screen-n); apply when still at the package default.
        _ss_default = _defaults.get("fe_check_pairs_subsample_n", None)
        _ss_cur = getattr(self, "fe_check_pairs_subsample_n", None)
        if _ss_default is None or _ss_cur == _ss_default:
            _fast_ss = self._fast_search_default_subsample_n()
            if _ss_cur is None or _fast_ss < int(_ss_cur):
                saved["fe_check_pairs_subsample_n"] = _ss_cur
                self.fe_check_pairs_subsample_n = _fast_ss
        # UNIFIED detection subsample (2026-06-17): tie the per-family DETECTION caps that read env
        # (currently the Fourier frequency detection) to the same fast-search subsample, so EVERY
        # family's detection runs on the small sample while values/recipes still replay full-n. Saved
        # under an ``__env__`` sentinel key; the fit's restore loop resets os.environ. Only SHRINK
        # (never raise a user-set smaller cap). At large n this caps the Fourier z_tr (else ~200k) to
        # the fast-search subsample -- bit-safe (detection-only; the recipe replays sin(2*pi*f*x) full-n).
        _fast_ss2 = getattr(self, "fe_check_pairs_subsample_n", None)
        if _fast_ss2:
            from .._fourier_detect_cap import get_fourier_detect_max_n, peek_fourier_detect_cap, set_fourier_detect_cap
            if int(_fast_ss2) < get_fourier_detect_max_n():
                # Thread-local (not os.environ) so concurrent fits do not race the cap; save the prior thread-local value (None if unset) for nested-fit-safe restore.
                saved["__fdcap__"] = peek_fourier_detect_cap()
                set_fourier_detect_cap(int(_fast_ss2))
        return saved

    @classmethod
    def recommend_default_scorer(cls) -> str:
        """Return the empirically-best ``fe_hybrid_orth_default_scorer`` value.

        's 7-dataset x 10-mechanism showdown placed CMIM (Layer 74)
        first on real sklearn data: 5/7 dataset wins on top-AUC of the
        downstream LogReg over the marginal-MI baseline, including all
        three high-redundancy fixtures where the conditional-MI
        redundancy filter dominates. JMIM (Layer 72) is the next-best on
        2/7 (the heavily-interacting pools), and the plug-in default is
        last on every redundant fixture. Callers that do not know which
        scorer to pick should default to the return value of this method.

        accelerated JMIM (~2.3x) and TC (~5.0x)
        via batched quantile binning + invariant support-side joint
        precompute; the perf improvement does NOT change the L83 AUC
        leaderboard (CMIM still wins 5/7) because the scorer math is
        bit-equivalent to the pre-opt path (rtol=1e-9). The recommended
        default therefore stays ``"cmim"`` -- L86 just makes the runner-
        up scorers cheap enough to evaluate inside an outer
        cross-validation without budget pain.

        Returns
        -------
        str
            ``"cmim"`` -- the Layer 83 leaderboard winner.
        """
        return "cmim"

    def _effective_random_seed(self):
        """Resolve the seed actually used by fit from the two stored constructor aliases.

        ``random_seed`` is the canonical name; ``random_state`` is the sklearn-style fallback. The
        constructor stores BOTH unmodified (sklearn ``get_params`` contract), so the promotion that used to
        happen in ``__init__`` is done here instead: ``random_seed`` wins when set, otherwise ``random_state``
        fills in. Returns ``None`` when neither is set (entropy-seeded behaviour preserved)."""
        seed = getattr(self, "random_seed", None)
        if seed is None:
            seed = getattr(self, "random_state", None)
        return seed

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

    @classmethod
    def _ctor_defaults(cls) -> dict:
        """Single source of truth for every constructor-parameter default.

        Read straight off ``__init__``'s signature so a ctor default can never
        silently diverge from a hand-written copy elsewhere (the D5 drift hazard:
        ``__setstate__`` injected a literal default that drifted from the ctor --
        e.g. ``cluster_aggregate_mode``). ``__setstate__`` overlays these onto its
        legacy-injection dict for every ctor-param key EXCEPT the documented
        legacy-pickle overrides below.
        """
        sig = inspect.signature(cls.__init__)
        return {name: param.default for name, param in sig.parameters.items() if param.default is not inspect.Parameter.empty}

    @classmethod
    def recommend_enabled_fe(cls, X=None, y=None) -> dict:
        """Classify every ``fe_*`` / ``dcd_*`` ctor flag by flip-safety AND
        recommend which FE generators to enable for a given ``(X, y)``.

        The Layer-99 rule recommender (``_meta_fe_recommender``) inspects ``X``
        dtypes / cardinalities / NaN rates / time+entity structure and turns on
        the master FE flags whose data-shape preconditions are met (e.g.
        ``fe_grouped_agg_enable`` only when an int-as-cat group column exists).
        When ``X`` is None it returns the static flip-safety classification only
        (``recommended_enable`` empty), so callers / tests can introspect the
        policy without supplying data. This is the cold-start path; the
        Param-Oracle-backed learned layer (``MetaFERecommender``) is optional and
        improves on these rules over time.

        Flip-safety taxonomy
        --------------------
        FLIP_SAFE
            Pure corrective / strict-improvement mechanisms: enabling cannot reduce
            accuracy and cannot materially slow a user who does not need them (no-op
            unless a paired master switch is on). Already flipped to default-True.
            * ``fe_local_mi_gate`` (Layer 91 Tier 1) — drops only sub-noise engineered
              columns, keeps top-k; no-op unless an L33/L34/L37/L38 FE mechanism is on.

        ALREADY_DEFAULT
            Corrective mechanisms shipped default-True in earlier layers; listed for
            completeness so the recommender never double-recommends them.
            * ``dcd_enable`` (dynamic cluster discovery, 0.003x overhead)
            * ``cardinality_bias_correction`` (Miller-Madow gate bias correction)
            * ``min_relevance_gain_relative_to_first`` (diminishing-returns gate)
            * ``cluster_aggregate_enable`` / ``cluster_aggregate_mode='replace'``
            * ``mrmr_skip_when_prior_was_identity`` / ``fe_adaptive_threshold_relax``
            * ``build_friend_graph`` (diagnostic only)

        FLIP_RISKY
            FE GENERATORS (add compute, help only on specific data shapes) and SCORER
            choices (domain-specific). Stay opt-in OR behind a cheap auto-detect; the
            L98 recommender is what will turn the data-shape-matched subset on.
            * generators: ``fe_count_encoding_enable``, ``fe_frequency_encoding_enable``,
              ``fe_cat_num_interaction_enable``, ``fe_missingness_enable``,
              ``fe_ratio_enable``, ``fe_grouped_delta_enable``, ``fe_lagged_diff_enable``,
              ``fe_grouped_agg_enable``, ``fe_composite_group_agg_enable``,
              ``fe_grouped_quantile_enable``, ``fe_cat_pair_enable``,
              ``fe_cat_triple_enable``, ``fe_hybrid_orth_enable`` (+ triplet / quadruplet
              / adaptive-arity / adaptive-degree / routing / lasso / elasticnet /
              semi-supervised variants), ``fe_smart_polynom_iters``, ``fe_max_polynoms``.
            * unified second-pass: ``fe_unified_second_pass_gate`` — a real CMI pass with
              ``min_gain`` cost that CAN drop columns; opt-in, not a pure corrective.
            * scorer / estimator: ``mrmr_relevance_algo`` / ``mrmr_redundancy_algo``,
              ``mi_correction`` (chao_shen), ``redundancy_aggregator`` (jmim),
              ``relaxmrmr_alpha``, ``pid_synergy_bonus``, ``bur_lambda``,
              ``cmi_perm_stop``, ``cpt_test``, ``uaed_auto_size``, ``mi_normalization``.

        Returns
        -------
        dict
            ``{"flip_safe": [...], "already_default": [...], "flip_risky": [...],
            "recommended_enable": [...]}``. ``recommended_enable`` is the
            data-driven subset of master FE flags the Layer-99 rule recommender
            turns ON for the supplied ``(X, y)`` data shape (empty when ``X`` is
            None or no precondition fires).
        """
        flip_safe = ["fe_local_mi_gate"]
        already_default = [
            "dcd_enable",
            "cardinality_bias_correction",
            "min_relevance_gain_relative_to_first",
            "cluster_aggregate_enable",
            "cluster_aggregate_mode",
            "mrmr_skip_when_prior_was_identity",
            "fe_adaptive_threshold_relax",
            "build_friend_graph",
        ]
        flip_risky = [
            "fe_count_encoding_enable", "fe_frequency_encoding_enable",
            "fe_cat_num_interaction_enable", "fe_missingness_enable",
            "fe_ratio_enable", "fe_grouped_delta_enable", "fe_lagged_diff_enable",
            "fe_grouped_agg_enable", "fe_composite_group_agg_enable",
            "fe_grouped_quantile_enable", "fe_cat_pair_enable", "fe_cat_triple_enable",
            "fe_hybrid_orth_enable", "fe_hybrid_orth_triplet_enable",
            "fe_hybrid_orth_quadruplet_enable", "fe_hybrid_orth_adaptive_arity_enable",
            "fe_hybrid_orth_adaptive_degree_enable", "fe_smart_polynom_iters",
            "fe_max_polynoms", "fe_unified_second_pass_gate",
            "mi_correction", "redundancy_aggregator", "relaxmrmr_alpha",
            "pid_synergy_bonus", "bur_lambda", "cmi_perm_stop", "cpt_test",
            "uaed_auto_size", "mi_normalization",
        ]
        # Layer-99: data-driven recommendation. When (X, y) is supplied, the
        # rule recommender picks the master FE flags whose data-shape
        # preconditions are met; cold-start (no Param-Oracle history needed).
        recommended_enable: list = []
        if X is not None:
            try:
                from .._meta_fe_recommender import recommend_fe_flags_by_rules
                rec = recommend_fe_flags_by_rules(X, y)
                recommended_enable = sorted(f for f, on in rec.items() if on)
            except Exception as _exc:  # never let recommendation break introspection
                logger.warning("recommend_enabled_fe: rule recommender failed: %s", _exc)
                recommended_enable = []
        return {
            "flip_safe": flip_safe,
            "already_default": already_default,
            "flip_risky": flip_risky,
            "recommended_enable": recommended_enable,
        }
