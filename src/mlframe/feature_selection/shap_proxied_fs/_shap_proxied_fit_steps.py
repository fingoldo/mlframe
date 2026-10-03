"""Fit-stage helper methods carved out of ``_shap_proxied_fit`` to keep that module under its size budget."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import _resolve_adaptive_n_anchors

logger = logging.getLogger("mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit")


class ShapProxiedFitStepsMixin:
    """Stage helpers of ``ShapProxiedFitMixin.fit``; every method reads only its arguments and ``self``."""

    # fitted / per-fit state assigned in ``fit`` itself
    n_features_in_: int
    feature_names_in_: Any
    _su_seeded_pairs_orig: Any
    # Constructor state lives on the concrete ``ShapProxiedFS`` class (see its ``__init__``); these
    # annotations declare the contract so mypy can type-check this mixin's methods on ``self``.
    model: Any
    classification: bool
    metric: Optional[str]
    optimizer: str
    out_of_fold: bool
    n_splits: int
    n_models: int
    min_features: int
    max_features: Optional[int]
    top_n: int
    holdout_size: float
    revalidate: bool
    n_revalidation_models: int
    lambda_stab: float
    parsimony_tol: float
    min_selected_ratio: float
    trust_guard: bool
    n_anchors: "int | str"
    fidelity_floor: Optional[float]
    spearman_floor: Optional[float]
    run_importance_ablation: bool
    use_bias_corrector: bool
    active_learning: bool
    active_learning_budget: Optional[int]
    config_jitter: bool
    uncertainty_penalty: float
    interaction_aware: bool
    max_interaction_features: int
    interaction_proxy_top_k: int
    su_seeded_interactions: bool
    su_seeded_top_k: int
    su_seeded_n_bins: int
    su_seeded_max_screen_cols: int
    su_seeded_snr_z: float
    su_seeded_snr_null_quantile: float
    su_seeded_snr_abs_floor: float
    su_seeded_n_permutations: int
    residual_passes: int
    residual_merge: str
    residual_lambda: float
    residual_top_k: Optional[int]
    residual_exclude_top: int
    beam_width: int
    brute_force_max_features: int
    adaptive_prescreen_by_stability: bool
    use_gpu: bool
    prefilter_top: Optional[int]
    prefilter_method: str
    prefilter_n_estimators: Optional[int]
    oof_shap_n_estimators: Optional[int]
    prefilter_stage1_keep: Optional[int]
    prefilter_univariate_batch_size: Optional[int]
    shap_prefilter_enabled: bool
    shap_prefilter_top: Optional[int]
    shap_prefilter_safety_factor: int
    shap_prefilter_min_features: int
    shap_aware_stage1_keep: bool
    shap_aware_stage1_cushion: int
    shap_aware_stage1_floor: int
    prescreen_top: Optional[int]
    prescreen_ranking: str
    banzhaf_n_coalitions: int
    within_cluster_refine: bool
    refine_n_estimators: Optional[int]
    refine_mode: str
    core_n_coalitions: int
    core_drop_threshold: float
    core_nucleolus: bool
    refine_ucb_enabled: bool
    refine_ucb_min_eval_size: Optional[int]
    refine_ucb_slack: Optional[float]
    refine_ucb_stdev_multiplier: float
    revalidation_n_estimators: Optional[int]
    revalidation_ucb_enabled: bool
    revalidation_ucb_min_eval_size: Optional[int]
    revalidation_ucb_slack: Optional[float]
    revalidation_adaptive_n_models: bool
    trust_guard_n_estimators: Optional[int]
    trust_guard_stratified_anchors: bool
    trust_guard_uniform_tail_frac: float
    trust_guard_cardinality_dist: str
    trust_guard_zipf_alpha: float
    trust_guard_fidelity_weights: tuple
    trust_guard_metric: str
    n_jobs: int
    inner_n_jobs_cap: bool
    random_state: int
    verbose: bool
    tqdm: bool
    precomputed: Optional[dict]
    cat_features: Optional[list]
    cache_dir: Optional[str]
    _rng: np.random.Generator
    _split_col_batch: int
    _deferred_holdout: Optional[tuple]
    # Provided by ``ShapProxiedMethodsMixin`` (the concrete class inherits both).
    _resolve_booster_kind: Callable[[], str]
    _su_screen_enabled: Callable[[], bool]
    _su_screen_snr_z: Callable[[], float]
    _resolve_optimizer: Callable[..., str]
    _resolve_revalidation_mmr_jaccard_threshold: Callable[[int], Optional[float]]
    _resolve_revalidation_ucb_stdev_multiplier: Callable[[int], float]
    _mmr_filter_by_jaccard: Callable[..., "list[int]"]
    _to_pandas: Callable[..., pd.DataFrame]
    _coerce_target: Callable[..., np.ndarray]

    def _fit_step1_effective_prefilter_top(self, effective_prefilter_top, n_features, X_search, report, _stage, _precomputed_aligned, model_template, y_search, X_cols, working_cols):
        """Step 1 of fit: lines starting at ``if effective_prefilter_top is not None and n_features > effective_pref``."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_precomputed import (
            su_to_prefilter_keep,
        )

        if effective_prefilter_top is not None and n_features > effective_prefilter_top:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import _default_stage1_keep, prefilter_columns, resolve_prefilter_method

            # iter33 SHAP-aware stage-A tightening: when ``shap_prefilter`` shrinks
            # ``effective_prefilter_top`` far below the legacy 2000 default, the two_stage prefilter's
            # stage-B booster fit on 2000 columns is the dominant wall-clock cost. Pre-resolve the
            # stage-A survivor count to ``max(floor, effective_prefilter_top * cushion)`` so the
            # booster fits on ~3x fewer columns at the same tree budget. Two protections:
            #   1) User-pinned ``prefilter_stage1_keep`` always wins (lever is a default-only tighten).
            #   2) Lever is gated on (a) ``shap_aware_stage1_keep=True``, (b)
            #      ``shap_prefilter_enabled=True``, (c) the resolved prefilter method == two_stage
            #      (only two_stage reads ``stage1_keep``; other paths ignore it).
            effective_stage1_keep = self.prefilter_stage1_keep
            if effective_stage1_keep is None and self.shap_aware_stage1_keep and self.shap_prefilter_enabled:
                _resolved = resolve_prefilter_method(self.prefilter_method, n_features=n_features, n_rows=int(X_search.shape[0]))
                if _resolved == "two_stage":
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_shap_prefilter import resolve_shap_aware_stage1_keep

                    effective_stage1_keep = resolve_shap_aware_stage1_keep(
                        effective_prefilter_top=int(effective_prefilter_top),
                        stage1_cushion=self.shap_aware_stage1_cushion,
                        stage1_floor=self.shap_aware_stage1_floor,
                        default_stage1_keep=_default_stage1_keep(n_features))
                    if "shap_prefilter" in report and isinstance(report["shap_prefilter"], dict):
                        report["shap_prefilter"]["stage1_keep_tightened"] = int(effective_stage1_keep)
                        report["shap_prefilter"]["stage1_keep_default"] = int(_default_stage1_keep(n_features))

            with _stage("prefilter"):
                if _precomputed_aligned is not None and "su_to_target" in _precomputed_aligned:
                    # iter66: replace the booster / F-statistic prefilter with
                    # the SU(X_j, y) ranking the MRMR screen already computed.
                    # Skips the cloned-booster fit / chunked f_classif pass
                    # entirely; the ordering is more cardinality-honest than
                    # the F-statistic for mixed-cardinality features
                    # (Witten-Frank-Hall 2011).
                    working_cols = su_to_prefilter_keep(
                        _precomputed_aligned, keep_top=int(effective_prefilter_top),
                    )
                    pf_info = {
                        "method": "precomputed_su",
                        "kept": int(working_cols.size),
                        "source": "MRMR.export_artifacts",
                    }
                else:
                    working_cols, pf_info = prefilter_columns(
                        model_template, X_search, y_search, method=self.prefilter_method,
                        prefilter_top=effective_prefilter_top, classification=self.classification,
                        n_features=n_features, n_estimators_cap=self.prefilter_n_estimators,
                        stage1_keep=effective_stage1_keep,
                        univariate_batch_size=self.prefilter_univariate_batch_size)
                # su_seeded_interactions RESCUE (#5b, OPT-IN): the native-importance / F-statistic
                # prefilter ranks by MARGINAL signal, so a PURE-interaction operand pair (op_a, op_b
                # with ~0 marginal - the exact regime the additive proxy is blind to) sits in the
                # tail the prefilter drops, and the downstream search can never pair them. Run the
                # CHEAP pairwise-SU synergy screen on the FULL pre-prefilter X_search HERE (operands
                # still present), SNR-gate it, and RESCUE the surviving operand columns into
                # working_cols so they flow through clustering / SHAP / search as normal columns. The
                # screen is O(P)+O(K) (no O(P^2) tensor) and NO-OPs cleanly (rescues nothing) when the
                # gate clears no pair. The kept pairs (by original column name) drive the post-search
                # sparse interaction objective. Skipped when the prefilter kept everything already.
                if self._su_screen_enabled() and len(working_cols) < n_features:
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import su_synergy_screen

                    with _stage("su_seeded_interactions"):
                        # ISOLATED rng (NOT self._rng): the screen's permutation-null shuffles must not
                        # advance the selector's shared RNG stream, or a no-op (gate clears nothing)
                        # would still perturb the downstream stochastic revalidation and silently
                        # change the additive default's selection. A fixed offset keeps it
                        # deterministic + reproducible while leaving self._rng byte-untouched.
                        _su_rng = np.random.default_rng(int(self.random_state) + 7919)
                        _kept_pairs_orig, _su_prefilter_info = su_synergy_screen(
                            X_search, y_search,
                            n_bins=self.su_seeded_n_bins,
                            top_k=self.su_seeded_top_k,
                            max_screen_cols=self.su_seeded_max_screen_cols,
                            snr_z=self._su_screen_snr_z(),
                            snr_null_quantile=self.su_seeded_snr_null_quantile,
                            snr_abs_floor=self.su_seeded_snr_abs_floor,
                            n_permutations=self.su_seeded_n_permutations,
                            importance=None, rng=_su_rng)
                        # X_search columns at this point are the FULL original names (prefilter slice
                        # below has not run yet); map operand names -> original feature indices.
                        _name_to_orig = {str(c): i for i, c in enumerate(X_cols)}
                        _rescue_orig: set[int] = set()
                        self._su_seeded_pairs_orig = []
                        for _syn, _jsu, _ca, _cb in _kept_pairs_orig:
                            if str(_ca) in _name_to_orig and str(_cb) in _name_to_orig:
                                _rescue_orig.add(_name_to_orig[str(_ca)])
                                _rescue_orig.add(_name_to_orig[str(_cb)])
                                self._su_seeded_pairs_orig.append((float(_syn), str(_ca), str(_cb)))
                        if _rescue_orig:
                            _wc_set = set(int(c) for c in working_cols)
                            _added = sorted(_rescue_orig - _wc_set)
                            if _added:
                                working_cols = np.sort(np.concatenate([np.asarray(working_cols, dtype=np.int64), np.asarray(_added, dtype=np.int64)]))
                        self._su_seeded_screen_info = dict(_su_prefilter_info)
                        self._su_seeded_screen_info["n_rescued_orig"] = len(_rescue_orig)
                if len(working_cols) < n_features:
                    X_search = X_search.iloc[:, working_cols]
                report["prefilter"] = pf_info
        return X_search, working_cols

    @staticmethod
    def _fit_step2_want_var_want(want_var, want_per_fold_phi, shap_out):
        """Step 2 of fit: lines starting at ``if want_var and want_per_fold_phi:``."""
        if want_var and want_per_fold_phi:
            phi, base, y_phi, phi_var, per_fold_phi_mean = shap_out
        elif want_var:
            phi, base, y_phi, phi_var = shap_out
            per_fold_phi_mean = None
        elif want_per_fold_phi:
            phi, base, y_phi, per_fold_phi_mean = shap_out
            phi_var = None
        else:
            phi, base, y_phi = shap_out
            phi_var = None
            per_fold_phi_mean = None
        return base, per_fold_phi_mean, phi, phi_var, y_phi

    def _fit_step3_prescreen_top_none(self, prescreen_top, n_proxy, _stage, residual_blend_importance, phi, base, y_phi, _su_rescue_proxy_idx, residual_rescue_proxy_idx, phi_var, unit_to_members, report, proxy_cols_kept):
        """Step 3 of fit: lines starting at ``if prescreen_top is not None and prescreen_top < n_proxy:``."""
        if prescreen_top is not None and prescreen_top < n_proxy:
            with _stage("prescreen"):
                from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import noise_floor_rescue_keep_set

                # residual_merge="blend" ranks by phi1+lambda*phi2 (aligned; excluded columns get 0
                # contribution) so residual-boosted weak features sort higher into top_keep - the
                # proxy loss consumed by the search below always stays raw phi1, never this vector.
                prescreen_ranking = str(getattr(self, "prescreen_ranking", "mean_abs_phi") or "mean_abs_phi").lower()
                banzhaf_stderr_max: Optional[float] = None
                if prescreen_ranking == "banzhaf" and residual_blend_importance is None:
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_banzhaf import banzhaf_msr

                    beta, banzhaf_info = banzhaf_msr(
                        phi, base, y_phi, classification=self.classification, metric=self.metric,
                        n_coalitions=int(self.banzhaf_n_coalitions), rng=self._rng,
                    )
                    # Shift nonnegative: the downstream noise-floor rescue's tail-quantile math assumes
                    # a nonnegative importance vector (see ``noise_floor_rescue_keep_set``); beta itself
                    # can be negative for harmful/noise features, so shifting by its min preserves the
                    # RANKING (a monotone translation) while keeping the rescue math intact.
                    importance = beta - beta.min()
                    banzhaf_stderr_max = float(np.max(banzhaf_info["beta_stderr"])) if len(beta) else None
                else:
                    importance = np.abs(phi).mean(axis=0) if residual_blend_importance is None else residual_blend_importance
                top_keep = np.argsort(-importance)[:prescreen_top]
                # Noise-floor rescue (bug fix, iter-2026-07-14): a flat top-K cut by mean|phi| alone
                # silently drops any real weak-signal column ranked below K whenever the frame has
                # more than K non-noise proxy columns - confirmed on a real wide/clustered fit
                # (112 post-clustering units, cap=28): weak/interaction-operand recall collapsed to
                # 0.0 while strong-signal recall stayed 1.0. The rescue widens the keep set to also
                # cover any column clearing a noise floor derived from the FULL importance vector's
                # tail, mirroring the same fix already shipped for the knee-ladder cap (see
                # ``noise_floor_rescue_keep_set`` - shared primitive, not duplicated logic).
                top_keep_set = set(int(i) for i in top_keep)
                rescued_set = noise_floor_rescue_keep_set(importance, top_keep) - top_keep_set
                keep_set = top_keep_set | rescued_set
                # Rescue su_seeded synergistic operands the marginal-|phi| ranking would discard.
                keep_set |= {int(i) for i in _su_rescue_proxy_idx if 0 <= int(i) < n_proxy}
                # Fourth union member: residual_merge="rescue" columns (top-k by mean|phi2|) - the
                # phi2-attributed regime the pass-1 additive proxy under-credits.
                keep_set |= {int(i) for i in residual_rescue_proxy_idx if 0 <= int(i) < n_proxy}
                keep = np.sort(np.fromiter(keep_set, dtype=np.int64))
                phi = np.ascontiguousarray(phi[:, keep])
                proxy_cols_kept = keep
                if phi_var is not None:
                    phi_var = np.ascontiguousarray(phi_var[:, keep])
                if unit_to_members is not None:
                    unit_to_members = [unit_to_members[i] for i in keep]
                else:
                    unit_to_members = [np.array([int(i)], dtype=np.int64) for i in keep]
                report["prescreen"] = dict(
                    kept=len(keep), of=int(n_proxy), su_rescued=len(_su_rescue_proxy_idx),
                    noise_floor_rescued=len(rescued_set), residual_rescued=len(residual_rescue_proxy_idx),
                    ranking=prescreen_ranking,
                )
                if banzhaf_stderr_max is not None:
                    report["prescreen"]["banzhaf_stderr_max"] = banzhaf_stderr_max
        return phi, phi_var, proxy_cols_kept, unit_to_members

    def _fit_step4_proxy_trust_diagnostic(self, report, working_cols, phi, unit_to_members, _stage, base, y_phi, model_template, X_search, X_hold, y_hold, _effective_min_card, rv):
        """Step 4 of fit: lines starting at ``if self.trust_guard:``."""
        if self.trust_guard:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import proxy_trust_guard

            # Stratified-anchor prior (opt-in via ``trust_guard_stratified_anchors``): when the
            # prefilter cached an F-score vector (two_stage / univariate paths), aggregate it into
            # UNIT space so the trust-guard sampler over-samples quality columns instead of drowning
            # in the noise tail. F-scores live in ORIGINAL column space (length n_features);
            # unit_to_members[u] -> WORKING-frame positions (post-prefilter); working_cols maps
            # working -> original. Per unit, take the MEAN F across its members (proxy for "is this
            # unit anchored by informative columns?"); singletons reduce to the member's own F.
            # Falls through to None (uniform sampler) when the prefilter didn't cache F-scores
            # (model / fast_model / gpu_model paths) OR the opt-in is OFF (the default; see
            # iter14-bench-attempt-rejected note in ``__init__``: the lever didn't pay at width=6000
            # because the post-two_stage cohort is already noise-filtered, so concentrating anchors
            # further compresses the spread the Spearman signal needs). Always-safe: misalignment is
            # detected inside ``proxy_trust_guard`` and degrades to uniform with a warning.
            unit_f_scores = None
            if self.trust_guard_stratified_anchors:
                from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import get_cached_f_scores

                f_scores_orig = get_cached_f_scores(report.get("prefilter"))
                if f_scores_orig is not None:
                    try:
                        f_working = np.asarray(f_scores_orig, dtype=np.float64)[np.asarray(working_cols)]
                    except (IndexError, TypeError):
                        f_working = None
                    if f_working is not None:
                        n_units = phi.shape[1]
                        if unit_to_members is None:
                            if f_working.shape[0] == n_units:
                                unit_f_scores = f_working
                        else:
                            if all(int(m) < f_working.shape[0] for u in unit_to_members for m in u):
                                unit_f_scores = np.array([float(np.mean(f_working[np.asarray(u, dtype=np.int64)])) for u in unit_to_members], dtype=np.float64)
            # iter18: resolve fidelity_floor / spearman_floor (deprecated alias). Supplying both
            # at the facade level is an error; supplying only the legacy name emits a
            # DeprecationWarning and copies the value through. ``None`` is the "unset" sentinel for
            # BOTH floors, so the both-set conflict is detected by ``fidelity_floor is not None``
            # (an explicit ``fidelity_floor=0.5`` is no longer mistaken for the default). The unset
            # ``fidelity_floor`` resolves to the effective 0.5 default.
            effective_floor = self.fidelity_floor if self.fidelity_floor is not None else 0.5
            if self.spearman_floor is not None:
                import warnings
                if self.fidelity_floor is not None:
                    raise ValueError("ShapProxiedFS: set either `fidelity_floor` (new name) or `spearman_floor` " "(deprecated alias), not both.")
                warnings.warn(
                    "`ShapProxiedFS(spearman_floor=...)` is deprecated since iter18; use "
                    "`fidelity_floor=...` (same semantics). The kwarg name was inherited from the "
                    "iter15 raw-Spearman gate but the gate has been the composite "
                    "`proxy_fidelity_score` since iter16.",
                    DeprecationWarning, stacklevel=2,
                )
                effective_floor = self.spearman_floor
            # Resolve the adaptive anchor budget. ``"auto"`` self-tunes from the RAW input width
            # ``n_features_in_`` (NOT the post-prefilter phi width): the sparsity problem the lever
            # targets is "p>> raw features -> 30 anchors thinly cover the space the proxy was asked to
            # rank". A literal int pins the legacy fixed count.
            if isinstance(self.n_anchors, str) and self.n_anchors.lower() == "auto":
                resolved_n_anchors = _resolve_adaptive_n_anchors(int(self.n_features_in_))
            else:
                resolved_n_anchors = int(self.n_anchors)
            report["trust_n_anchors"] = dict(
                resolved=int(resolved_n_anchors), raw_width=int(self.n_features_in_),
                search_width=int(phi.shape[1]),
                mode=("auto" if isinstance(self.n_anchors, str) else "fixed"))
            with _stage("trust_guard"):
                report["trust"] = proxy_trust_guard(
                    phi, base, y_phi, model_template, X_search, X_hold, y_hold,
                    n_anchors=resolved_n_anchors, rng=self._rng, min_card=_effective_min_card,
                    max_card=self.max_features, fidelity_floor=effective_floor,
                    n_estimators_cap=self.trust_guard_n_estimators,
                    unit_f_scores=unit_f_scores,
                    anchor_uniform_tail_frac=self.trust_guard_uniform_tail_frac,
                    cardinality_dist=str(self.trust_guard_cardinality_dist).lower(),
                    zipf_alpha=self.trust_guard_zipf_alpha,
                    fidelity_weights=(float(self.trust_guard_fidelity_weights[0]),
                                       float(self.trust_guard_fidelity_weights[1])),
                    trustworthy_metric=str(self.trust_guard_metric).lower(), **rv)

    def _fit_step5_penalty_focuses_retrain(self, candidates, report, phi, phi_var, n_features):
        """Step 5 of fit: lines starting at ``score = np.array([c[0] for c in candidates], dtype=np.float64) # raw p``."""
        score = np.array([c[0] for c in candidates], dtype=np.float64)  # raw proxy loss
        if self.use_bias_corrector and self.trust_guard and report.get("trust", {}).get("_corrector_data"):
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy_many

            cd = report["trust"]["_corrector_data"]
            corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
            if not corrector.fallback:
                cards = np.array([len(c[1]) for c in candidates], dtype=np.float64)
                redund = subset_redundancy_many(phi, [c[1] for c in candidates])
                score = corrector.predict(score, cards, redund)
                report["bias_corrector"] = dict(applied=True, n_anchors=len(cd["proxy"]))
        if self.uncertainty_penalty > 0 and phi_var is not None:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import subset_uncertainty_many

            unc = subset_uncertainty_many(phi_var, [c[1] for c in candidates])
            score = score + self.uncertainty_penalty * unc
            report["uncertainty"] = dict(applied=True, penalty=float(self.uncertainty_penalty))
        order = np.argsort(score, kind="stable")
        candidates = [candidates[i] for i in order]
        score = score[order]  # keep aligned with candidates for downstream UCB consumption

        # MMR de-duplication of the corrector-sorted candidate list BEFORE revalidate (iter50).
        # At wide regimes (n_features>=20000) top_n=20 candidates are near-duplicate unions of the
        # same SHAP-aware stage-B picks; UCB short-circuits the proxy-loss tail but still pays
        # per-batch dispatch on the redundant subsets. Greedy keep-if-Jaccard-distance>tau in
        # corrector-sorted order; dropped candidates are corrector-equivalent to a retained one
        # and would not pass the parsimony band as a meaningful improvement. Disabled by default
        # below width 20000.
        mmr_tau = self._resolve_revalidation_mmr_jaccard_threshold(n_features)
        if mmr_tau is not None and self.revalidate and len(candidates) > 1:
            kept_idx = self._mmr_filter_by_jaccard(candidates, float(mmr_tau))
            n_total = len(candidates)
            if len(kept_idx) < n_total:
                candidates = [candidates[i] for i in kept_idx]
                score = score[np.asarray(kept_idx, dtype=np.int64)]
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau), n_kept=len(kept_idx), n_total=int(n_total))
            else:
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau), n_kept=int(n_total), n_total=int(n_total))
        return candidates, score

    def _fit_step6_proposal_generator_seeding(self, unit_to_members, candidates, report, _budget_exhausted, _budget_max_mins, _budget_stop_file, _stage, model_template, X_search, y_search, X_hold, y_hold, phi, rv, n_features, score, honest_cache):
        """Step 6 of fit: lines starting at ``def _cand_names(idx):``."""
        def _cand_names(idx):
            """Expand a candidate subset (unit indices, or raw column indices when no clustering was applied) to sorted feature names."""
            if unit_to_members is not None:
                cols = sorted({int(c) for u in idx for c in unit_to_members[int(u)]})
            else:
                cols = sorted(int(i) for i in idx)
            return [str(self.feature_names_in_[i]) for i in cols]

        report["candidates"] = [dict(proxy_loss=float(lo), features=_cand_names(c)) for lo, c in candidates[: self.top_n]]

        # Honest re-validation of the top-N on the disjoint holdout (active-learning variant when the
        # corrector anchors are available, else the static top-N retrain).
        _reval_budget_skip = self.revalidate and _budget_exhausted()
        if _reval_budget_skip:
            report["budget_skipped"] = dict(phase="revalidation", max_runtime_mins=_budget_max_mins, stop_file=_budget_stop_file)
            if self.verbose:
                logger.info("ShapProxiedFS: budget/stop reached; skipping honest revalidation, finalising with proxy-best subset.")
        if self.revalidate and not _reval_budget_skip:
            cdata = report.get("trust", {}).get("_corrector_data")
            with _stage("revalidation"):
                if self.active_learning and cdata:
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import active_learning_revalidate

                    # `or self.top_n` here would additionally clobber a legitimately-set 0 (a valid
                    # way to request zero active-learning evaluations) with a nonzero default; only
                    # widen on a genuine None/unset value.
                    budget = self.active_learning_budget if self.active_learning_budget is not None else self.top_n
                    best_idx, ranked, n_eval = active_learning_revalidate(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        corrector_data=cdata, phi=phi, budget=budget, n_models=self.n_revalidation_models,
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n],
                                                  active_learning=dict(n_evaluated=n_eval, budget=budget))
                else:
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import revalidate_top_n

                    best_idx, ranked, baseline = revalidate_top_n(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        n_models=self.n_revalidation_models, lambda_stab=self.lambda_stab,
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators,
                        ucb_enabled=self.revalidation_ucb_enabled,
                        ucb_min_eval_size=self.revalidation_ucb_min_eval_size,
                        ucb_slack=self.revalidation_ucb_slack,
                        ucb_stdev_multiplier=self._resolve_revalidation_ucb_stdev_multiplier(n_features),
                        adaptive_n_models=self.revalidation_adaptive_n_models,
                        candidate_score=score, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n], random_baseline=baseline)
        else:
            best_idx = tuple(candidates[0][1])

        # Importance-top-k ablation (unique-value gate vs plain SHAP importance).
        if self.run_importance_ablation and best_idx and not _budget_exhausted():
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import importance_topk_ablation

            with _stage("importance_ablation"):
                report["importance_ablation"] = importance_topk_ablation(
                    phi, best_idx, model_template, X_search, y_search, X_hold, y_hold,
                    classification=self.classification, metric=self.metric, unit_to_members=unit_to_members,
                    cache=honest_cache, disk_cache_dir=self.cache_dir)
        return best_idx

    @staticmethod
    def _fit_step7_expand_best_proxy(unit_to_members, best_idx):
        """Step 7 of fit: lines starting at ``if unit_to_members is not None:``."""
        if unit_to_members is not None:
            member_cols = sorted({int(c) for u in best_idx for c in unit_to_members[int(u)]})
        else:
            member_cols = sorted(int(i) for i in best_idx)
        return member_cols

    def _fit_step8_expand_best_proxy(self, unit_to_members, member_cols, _budget_exhausted, _stage, best_idx, residual_protected_working_cols, model_template, X_search, y_search, X_hold, y_hold, honest_cache, phi, base, y_phi, report):
        """Step 8 of fit: lines starting at ``if self.within_cluster_refine and unit_to_members is not None and len(``."""
        if self.within_cluster_refine and unit_to_members is not None and len(member_cols) > 1 and not _budget_exhausted():
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import resolve_metric
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import (
                _honest_loss, _open_disk_cache, within_cluster_refine,
            )

            with _stage("within_cluster_refine"):
                # Pass the per-unit member lists so refine can collapse each cluster to a single
                # representative in ONE parallel batch (O(sum k_c) trials) instead of legacy
                # O(k^2) greedy drops. unit_to_members is in proxy-unit space; each chosen unit
                # contributes one group of member columns.
                member_groups = [[int(c) for c in unit_to_members[int(u)]] for u in best_idx]
                # Residual-rescued columns (residual_merge="rescue") survived the prescreen cut; the
                # empirical trace showed prescreen survival is NOT sufficient - this greedy
                # parsimony_tol pruner re-drops them unless explicitly protected (gt_09 sec 3.4).
                _residual_protected = residual_protected_working_cols & set(member_cols) or None

                _legacy_refine_memo: list = []

                def _run_legacy_refine():
                    """Legacy greedy parsimony_tol refine; core's honest-gate baseline and fallback, so memoised."""
                    if not _legacy_refine_memo:
                        _legacy_refine_memo.append(within_cluster_refine(
                            member_cols, model_template, X_search, y_search, X_hold, y_hold,
                            classification=self.classification, metric=self.metric,
                            parsimony_tol=self.parsimony_tol, n_jobs=self.n_jobs, cache=honest_cache,
                            member_groups=member_groups, refine_n_estimators=self.refine_n_estimators,
                            ucb_enabled=self.refine_ucb_enabled,
                            ucb_min_eval_size=self.refine_ucb_min_eval_size,
                            ucb_slack=self.refine_ucb_slack,
                            ucb_stdev_multiplier=self.refine_ucb_stdev_multiplier,
                            inner_n_jobs_cap=self.inner_n_jobs_cap,
                            disk_cache_dir=self.cache_dir,
                            protected_cols=_residual_protected))
                    return list(_legacy_refine_memo[0])

                _refine_mode_effective = self.refine_mode
                _core_disk_cache = None
                _base_honest_loss_for_auto: Optional[float] = None
                if _refine_mode_effective in ("core", "auto"):
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_heuristics import _Evaluator
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import core_refine

                    _core_disk_cache = _open_disk_cache(self.cache_dir)
                    _base_honest_loss_for_auto = _honest_loss(
                        model_template, X_search, y_search, X_hold, y_hold, member_cols,
                        self.classification, resolve_metric(self.classification, self.metric),
                        cache=honest_cache, disk_cache=_core_disk_cache)

                if _refine_mode_effective == "auto":
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import auto_should_use_core_refine

                    def _confident_subset_loss(cols):
                        """Honest loss of a reduced (confident-only) unit subset, for the auto pre-gate's marginal comparison."""
                        return _honest_loss(
                            model_template, X_search, y_search, X_hold, y_hold, cols,
                            self.classification, resolve_metric(self.classification, self.metric),
                            cache=honest_cache, disk_cache=_core_disk_cache)

                    assert _base_honest_loss_for_auto is not None  # computed above whenever mode is "core" or "auto"
                    _use_core = auto_should_use_core_refine(
                        phi, tuple(int(u) for u in best_idx), unit_to_members, _base_honest_loss_for_auto, _confident_subset_loss
                    )
                    _refine_mode_effective = "core" if _use_core else "greedy"

                if _refine_mode_effective == "core":
                    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import _parallel_honest_losses

                    assert _base_honest_loss_for_auto is not None  # computed above whenever mode is "core" or "auto"
                    base_honest_loss = _base_honest_loss_for_auto
                    _tol_threshold = base_honest_loss + self.parsimony_tol * abs(base_honest_loss)

                    # core_refine's LP allocates credit across UNITS (proxy columns), never sub-members
                    # within a unit's own cluster - intra-cluster member redundancy (e.g. exact-duplicate
                    # columns the clustering step didn't merge into one unit) needs the SAME per-cluster
                    # collapse legacy stage 1 performs, so it stays "exactly as today" per gt_02 sec 3.
                    # Best-effort / independently-accepted (no cumulative re-verify): core_refine's own
                    # honest gate below re-checks the WHOLE final proposal and falls back to full legacy
                    # on any failure, so an over-eager collapse here can never surface as a silent
                    # regression - it is caught by the same safety net protecting the unit-level drop.
                    core_member_cols = list(member_cols)
                    core_unit_to_members = unit_to_members
                    multi_groups = [[int(c) for c in g if int(c) in set(member_cols)] for g in member_groups]
                    multi_groups = [g for g in multi_groups if len(g) > 1]
                    if multi_groups:
                        _protected_set = _residual_protected or set()
                        probe_tasks = [(sorted(c for c in core_member_cols if c not in set(g[1:]) - _protected_set), None) for g in multi_groups]
                        probe_losses = _parallel_honest_losses(
                            probe_tasks, model_template, X_search, y_search, X_hold, y_hold,
                            self.classification, resolve_metric(self.classification, self.metric), self.n_jobs,
                            cache=honest_cache, n_estimators_cap=self.refine_n_estimators,
                            template_id=("refine_cap", int(self.refine_n_estimators)) if self.refine_n_estimators is not None else None,
                            inner_n_jobs_cap=self.inner_n_jobs_cap, disk_cache=_core_disk_cache)
                        accepted_drops: set[int] = set()
                        for g, loss_val in zip(multi_groups, probe_losses):
                            drop_set = set(g[1:]) - _protected_set
                            if loss_val <= _tol_threshold:
                                accepted_drops.update(drop_set)
                        if accepted_drops:
                            core_member_cols = sorted(c for c in core_member_cols if c not in accepted_drops)
                            # unit_to_members is a sequence indexed by unit id (not a dict); rebuild the
                            # same indexable shape with dropped duplicate columns filtered out per unit.
                            core_unit_to_members = [[int(c) for c in cols_ if int(c) not in accepted_drops] for cols_ in unit_to_members]

                    core_evaluator = _Evaluator(phi, base, y_phi, resolve_metric(self.classification, self.metric))

                    def _honest_gate(cols):
                        """Accept core iff its honest loss is within parsimony_tol of pre-refine AND no worse than greedy's result (core's
                        documented contract). The band alone let core keep 14/41 cols on breast_cancer+decoys vs greedy's 23, -0.006 AUC."""
                        _metric = resolve_metric(self.classification, self.metric)
                        candidate_loss = _honest_loss(
                            model_template, X_search, y_search, X_hold, y_hold, cols,
                            self.classification, _metric, cache=honest_cache, disk_cache=_core_disk_cache)
                        if candidate_loss > _tol_threshold:
                            return False
                        greedy_cols = _run_legacy_refine()
                        if not greedy_cols or sorted(greedy_cols) == sorted(cols):
                            return True
                        greedy_loss = _honest_loss(
                            model_template, X_search, y_search, X_hold, y_hold, list(greedy_cols),
                            self.classification, _metric, cache=honest_cache, disk_cache=_core_disk_cache)
                        # Keeping MORE than greedy is core's purpose (weak-but-real recall): parsimony_tol of slack. Keeping FEWER must not cost loss.
                        return candidate_loss <= greedy_loss + (self.parsimony_tol * abs(greedy_loss) if len(cols) >= len(greedy_cols) else 0.0)

                    refined, core_info = core_refine(
                        core_member_cols, tuple(int(u) for u in best_idx), core_evaluator, _honest_gate,
                        drop_threshold=self.core_drop_threshold, n_coalitions=self.core_n_coalitions,
                        rng=self._rng, nucleolus_refine=self.core_nucleolus,
                        unit_to_members=core_unit_to_members, legacy_refine_fn=_run_legacy_refine,
                        legacy_refine_kwargs={})
                else:
                    refined = _run_legacy_refine()
                    core_info = None
                # Final full-template re-evaluation of the ONE chosen subset (uncapped n_estimators).
                # Refine's ranking trials use a cheaper capped booster (~100 trees) to decide WHICH
                # members to drop; the user-visible quality bar (and any downstream report consumer)
                # should see this subset's loss at the SAME booster size the other guards used, so the
                # values are apples-to-apples. The cache lookup is the full-template namespace (no
                # template_id), so this hits any prior pipeline retrain of the same subset (e.g. when
                # refine made no drops, this is a cache hit of the union retrain done elsewhere).
                refine_info: dict[str, Any] = dict(before=len(member_cols), after=len(refined), mode=_refine_mode_effective, requested_mode=self.refine_mode)
                if core_info is not None:
                    refine_info["allocation"] = {f"unit_{u}": v for u, v in core_info["allocation"].items()}
                    refine_info["eps_star"] = core_info["eps_star"]
                    refine_info["dropped_by_core"] = core_info["dropped_by_core"]
                    refine_info["fallback"] = core_info["fallback"]
                if refined:
                    # iter81: the full-template re-eval of the refined subset frequently hits the
                    # disk cache too - the same (cols, template, cap=None) tuple was retrained as
                    # the revalidation winner upstream, so a warm-cache lookup avoids an extra fit.
                    refine_info["honest_loss_full"] = float(_honest_loss(
                        model_template, X_search, y_search, X_hold, y_hold, list(refined),
                        self.classification, resolve_metric(self.classification, self.metric),
                        cache=honest_cache, disk_cache=_open_disk_cache(self.cache_dir)))
                report["within_cluster_refine"] = refine_info
                member_cols = refined
        return member_cols
