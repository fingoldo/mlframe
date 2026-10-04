"""``ShapProxiedFitMixin`` - the search + fit machinery for :class:`ShapProxiedFS`.

Carved out of ``shap_proxied_fs.py`` to keep both files under the 1k LOC ceiling. The mixin holds
the optimizer dispatch (``_run_search``) and the full ``fit`` pipeline; ``ShapProxiedFS`` inherits
it so ``self`` (all constructor state + the resolver methods on the concrete class) stays intact.
Heavy dependencies are lazy-imported in-body as in the original, so this module imports only the
few module-scope names the two method bodies reference directly.
"""

from __future__ import annotations

from mlframe.utils.budgets import active_budget

import logging
from typing import Optional

import numpy as np
import pandas as pd
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_holdout import split_search_and_holdout
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import (
    _apply_min_selected_ratio, _resolve_adaptive_prescreen_width, _resolve_knee_prescreen_cap,
    ShapProxiedNoCandidatesError, resolve_effective_min_features, unit_importance_to_feature_map)
from mlframe.feature_selection._selection_log import logs_fit
from mlframe.utils.misc import rng_hygienic_fit
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_report_slice import score_on_report_slice, split_report_slice

from ._shap_proxied_fit_steps import ShapProxiedFitStepsMixin

logger = logging.getLogger(__name__)


class ShapProxiedFitMixin(ShapProxiedFitStepsMixin):
    """Search-dispatch + fit pipeline for :class:`ShapProxiedFS` (see module docstring)."""

    def _run_search(self, optimizer, phi, base, y):
        """Dispatch to the chosen optimizer; returns list of (proxy_loss, feature_idx tuple)."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_search import run_search

        return run_search(self, optimizer, phi, base, y)

    # ------------------------------------------------------------------ fit
    @logs_fit("ShapProxiedFS", lambda self: (self.selected_features_, int(self.n_features_in_), None))
    @rng_hygienic_fit
    def fit(self, X, y):
        """Fit the SHAP-proxied selector: compute OOF SHAP values, run the configured proxy-search optimizer, then apply the budget-gated refinement stages (revalidation, ablation, cluster-refine) before finalising the selected subset."""
        import time
        from contextlib import contextmanager

        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix, make_default_estimator

        # Optional per-stage wall-clock instrumentation for the scaling benchmark / profiling. Set ``self._stage_timings`` to a dict before calling fit and each
        # stage's seconds land in it; a no-op otherwise (zero overhead beyond a dict lookup), so production fits are unaffected.
        _timings = getattr(self, "_stage_timings", None)

        @contextmanager
        def _stage(name):
            """Accumulate wall-clock seconds spent in the ``with`` block under ``_timings[name]``; a no-op context when no timing dict was requested."""
            if _timings is None:
                yield
                return
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _timings[name] = _timings.get(name, 0.0) + (time.perf_counter() - t0)

        # Control/safety budget (parity with MRMR / RFECV). The OOF-SHAP + proxy-search core always
        # runs (it produces the candidate subsets), but the OPTIONAL expensive refinement phases
        # below - honest revalidation, importance ablation, within-cluster refine - are each gated
        # on this budget. When the wall-clock budget is exceeded OR ``stop_file`` exists, they are
        # skipped and fit() finalises with the proxy-best subset (a valid selection). ``getattr``
        # keeps old pickled instances (without the new attrs) working.
        from os.path import exists as _stop_file_exists
        _budget_t0 = time.perf_counter()
        _budget_max_mins = getattr(self, "max_runtime_mins", None)
        _budget_stop_file = getattr(self, "stop_file", None)

        def _budget_exhausted() -> bool:
            """True once the optional refinement wall-clock budget has elapsed or the caller-provided ``stop_file`` has appeared."""
            if (minutes := active_budget(_budget_max_mins)) is not None and (time.perf_counter() - _budget_t0) > minutes * 60.0:
                return True
            return bool(_budget_stop_file) and _stop_file_exists(str(_budget_stop_file))

        X = self._to_pandas(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        # Duplicate column names make ``X[label]`` return a DataFrame (not a Series), whose ``.dtype`` access raises inside the parallel SHAP/booster path, and break the feature -> selected-subset mapping. Surface a clear error at fit entry.
        if X.columns.has_duplicates:
            dup_names = X.columns[X.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"ShapProxiedFS.fit: duplicate column names not supported: {dup_names[:10]}. "
                f"De-duplicate (e.g. ``X.loc[:, ~X.columns.duplicated()]`` or rename) before fitting."
            )
        self.feature_names_in_ = np.asarray(list(X.columns))
        self.n_features_in_ = int(X.shape[1])
        n_features = self.n_features_in_
        y = self._coerce_target(y)
        # Reset per-fit su_seeded_interactions scratch so a re-fit never reuses a prior fit's pairs.
        self._su_seeded_pairs_orig = None
        self._su_seeded_screen_info = {}

        if self.model is not None:
            model_template = self.model
        else:
            # Resolve + validate booster_kind FIRST so a typo (e.g. "bogus") raises ValueError before
            # the expensive prefilter / OOF-SHAP stages start. Auto-detect when None.
            _booster_kind = self._resolve_booster_kind()
            # The catboost ``cat_features`` template (iter78) is forwarded to the booster, but the
            # surrounding pipeline is NOT categorical-aware: the prefilter densifies ``X.values``
            # to float64 (crashes on string/categorical columns inside ``f_classif_chunked``), the
            # clustering treats columns as numeric, and column-slicing between stages does not
            # remap name-based ``cat_features``. Fail fast with an actionable message here instead
            # of the obscure downstream ValueError. Numeric-only fits (cat_features unset) are fine.
            if _booster_kind == "catboost" and self.cat_features:
                raise ValueError(
                    "ShapProxiedFS(booster_kind='catboost', cat_features=...) is not yet supported: "
                    "the prefilter / clustering / column-slicing stages are not categorical-aware and "
                    "would crash on densification or mis-route the cat-feature indices. Pre-encode the "
                    "categorical columns numerically and pass cat_features=None, or supply an explicit "
                    "fitted-pipeline ``model=`` template that handles categoricals upstream."
                )
            model_template = make_default_estimator(
                self.classification, random_state=int(self.random_state),
                booster_kind=_booster_kind, cat_features=self.cat_features,
            )

        # Disjoint holdout for honest re-validation + trust guard (avoids winner's curse).
        stratify = y if self.classification else None
        idx_all = np.arange(len(X))
        idx_search, idx_hold = split_search_and_holdout(self, idx_all, len(X), stratify)
        idx_hold, _report_slice = split_report_slice(idx_hold, X, y, getattr(self, "report_holdout_fraction", 0.0), self.classification, int(self.random_state))
        # Wide-frame split with deferred holdout materialisation. At C4 (width=20000, n_rows=10000) the original frame is 1.49 GiB, the search slice (75% rows) is 1.12 GiB, and the holdout
        # slice (25% rows) is 381 MiB; the legacy back-to-back
        # `X.iloc[idx_search].reset_index(drop=True)` + `X.iloc[idx_hold].reset_index(drop=True)`
        # held all three simultaneously plus reset_index transient buffers and OOM'd on a
        # 17 GB / 6.4 GB-free host (iter46). The prefilter only needs `X_search`; once it returns
        # `working_cols` (typically <=704 entries - effective_prefilter_top is bounded by the
        # SHAP-prefilter cap), the holdout can be built directly at that narrow column count
        # (~5 MiB instead of 381 MiB). Keep `X_vals_full` alive (a view into the original block,
        # zero extra alloc on a single-dtype input) so the deferred holdout materialisation has a
        # source to slice from. `reset_index(drop=True)` is dropped throughout: downstream consumers
        # all read via `.values` / `.iloc[:, cols]` / positional row access, none depend on the row
        # index being 0..n-1 RangeIndex (and `compute_shap_matrix` does its own
        # `reset_index(drop=True)` on the narrow post-prefilter frame anyway).
        X_cols = X.columns
        n_cols = X.shape[1]
        X_vals_full = X.values  # single-block float64 view on bench / homogeneous input
        # Build wide X_search via column-batched copy from the parent block. One batch's worth of
        # transient memory (~80 MiB at default 1024-col batch), not a full extra split copy.
        X_search_arr = np.empty((idx_search.shape[0], n_cols), dtype=X_vals_full.dtype)
        col_batch = self._split_col_batch
        for c0 in range(0, n_cols, col_batch):
            c1 = min(c0 + col_batch, n_cols)
            X_search_arr[:, c0:c1] = X_vals_full[idx_search, c0:c1]
        X_search = pd.DataFrame(X_search_arr, columns=X_cols, copy=False)
        del X, X_search_arr
        y_search = y[idx_search]
        # Defer X_hold materialisation: store the inputs and let the post-prefilter step build it
        # at the narrow working-column count.
        self._deferred_holdout = (X_vals_full, idx_hold, X_cols)
        X_hold = None  # built post-prefilter (or pre-clustering if prefilter is skipped)
        y_hold = y[idx_hold]

        report: dict = {}

        # iter66: validate + align the precomputed cross-selector artifacts
        # (canonically from ``MRMR(retain_artifacts=True).export_artifacts()``)
        # against X.columns. On any mismatch ``align_precomputed_to_X`` logs a
        # warning and returns None so the prefilter falls back to legacy
        # behaviour. The diagnostic block is always surfaced under
        # ``report['precomputed_used']`` so callers can confirm which path ran.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_precomputed import (
            align_precomputed_to_X,
        )
        _precomputed_aligned, _precomputed_report = align_precomputed_to_X(
            self.precomputed, X_search,
        )
        report["precomputed_used"] = _precomputed_report
        report["precomputed_bins_available"] = bool(isinstance(_precomputed_aligned, dict) and _precomputed_aligned.get("bins"))

        # Cheap native-importance pre-filter BEFORE the expensive OOF-SHAP. SHAP cost scales with the
        # column count, and clustering only compresses CORRELATED features (independent noise stays as
        # singletons), so on wide data SHAP would otherwise run on ~all columns. Rank all features and
        # keep the top-K; ``working_cols`` maps the surviving working columns back to original indices
        # for the final selector output. ``prefilter_method`` trades speed against interaction-awareness
        # (model / univariate / fast_model / gpu_model / two_stage); "auto" stays quality-safe for
        # moderate widths and routes very wide data (n_features >= 8000) to the cheap-funnel +
        # capped-booster two_stage path - see ``_shap_proxy_prefilter``.
        #
        # SHAP-pre-prefilter (iter31): when enabled (default), tighten the effective ``prefilter_top``
        # to ``shap_prefilter_top = max(brute_force_max_features * safety_factor,
        # shap_prefilter_min_features)`` (default 88) so the post-prefilter cohort that feeds OOF-SHAP
        # is sized to the downstream search budget plus a 4x cushion, instead of the default 2000.
        # The downstream search only consumes top-``brute_force_max_features`` by mean |phi| anyway,
        # so noise-tail columns between the search cap and the loose default were paying full TreeSHAP
        # cost for no contribution. Realized by REUSING the existing prefilter's booster ranking (a
        # separate post-clustering booster fit was bench-attempt-rejected 2026-05-28: the extra fit
        # cost ~1.2s while saving ~1.3s on OOF-SHAP for a +0.1s wash at width=1000/rows=5000/seed=1,
        # despite a +17% gain at the cold-start seed=0). Tightening at the prefilter step avoids the
        # double-booster work AND keeps the lever's win on warm runs.
        effective_prefilter_top = self.prefilter_top
        if self.shap_prefilter_enabled and self.prefilter_top is not None:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_shap_prefilter import resolve_shap_prefilter_top

            sp_top = (self.shap_prefilter_top if self.shap_prefilter_top is not None
                      else resolve_shap_prefilter_top(
                          brute_force_max_features=self.brute_force_max_features,
                          safety_factor=self.shap_prefilter_safety_factor,
                          min_features=self.shap_prefilter_min_features))
            # Tighten only - never expand the user's prefilter budget.
            effective_prefilter_top = min(int(self.prefilter_top), int(sp_top))
            report["shap_prefilter"] = dict(
                requested_top=int(sp_top), effective_prefilter_top=int(effective_prefilter_top), user_prefilter_top=int(self.prefilter_top)
            )
        working_cols = np.arange(n_features)
        X_search, working_cols = self._fit_step1_effective_prefilter_top(effective_prefilter_top, n_features, X_search, report, _stage, _precomputed_aligned, model_template, y_search, X_cols, working_cols)
        # Materialise the deferred holdout at the narrow post-prefilter column count, then run the
        # optional correlated-feature clustering. Carved verbatim into a sibling helper (Tier E LOC
        # carve); returns (X_hold, X_proxy, unit_to_members) and mutates report["clustering"] in
        # place + clears self._deferred_holdout exactly as the inline block did.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_prefilter import materialise_holdout_and_cluster

        X_hold, X_proxy, unit_to_members = materialise_holdout_and_cluster(
            self=self, working_cols=working_cols, n_features=n_features,
            _precomputed_aligned=_precomputed_aligned, X_search=X_search,
            report=report, _stage=_stage)

        # SHAP attribution on the proxy (unit or raw) columns. Request per-model attribution variance
        # only when the uncertainty lever is active AND we actually have multiple models to vary.
        want_var = self.uncertainty_penalty > 0 and self.n_models > 1
        want_per_fold_phi = bool(self.adaptive_prescreen_by_stability) and bool(self.out_of_fold)
        with _stage("oof_shap"):
            shap_out = compute_shap_matrix(
                model_template, X_proxy, y_search, classification=self.classification,
                out_of_fold=self.out_of_fold, n_splits=self.n_splits, n_models=self.n_models,
                config_jitter=self.config_jitter, return_variance=want_var,
                rng=self._rng, tqdm_desc=("shap-oof" if self.tqdm else None), n_jobs=self.n_jobs,
                n_estimators_cap=self.oof_shap_n_estimators,
                inner_n_jobs_cap=self.inner_n_jobs_cap,
                return_per_fold_phi_mean=want_per_fold_phi,
                cache_dir=self.cache_dir, cv_policy=getattr(self, "_search_cv_policy", None))
        base, per_fold_phi_mean, phi, phi_var, y_phi = self._fit_step2_want_var_want(want_var, want_per_fold_phi, shap_out)

        # Two-phase residual attribution (gt_09, OPT-IN via residual_passes=0 default): a second SHAP
        # pass on pass-1's residual re-credits weak features the additive coalition proxy under-weighs
        # when strong features absorb most of the shared credit. Runs on the PRE-prescreen phi/X_proxy
        # - rescue only helps if it can save a column the prescreen would otherwise cut, so it must
        # happen BEFORE the knee/prescreen block below, never after.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_residual import run_residual_pass

        residual_rescue_proxy_idx, residual_blend_importance, residual_protected_working_cols = run_residual_pass(
            self, phi, base, y_phi, X_proxy, model_template, unit_to_members, working_cols, X_cols, report, _stage)

        # Persist the per-feature mean |SHAP| the subset search ranks by. Computed HERE, on the
        # PRE-prescreen phi, so it covers every column the SHAP pass actually attributed - after the
        # prescreen narrows phi the tail columns are gone and the map would be a top-K slice. Values are
        # unit-level (a clustering unit is accepted/rejected whole, so its members share the number the
        # search saw); features the prefilter dropped BEFORE the SHAP pass are absent rather than
        # zero-filled - no attribution was ever computed for them. ``mean_abs_shap_coverage`` states
        # exactly how much of the input frame the map spans so a consumer never has to guess.
        _mean_abs_shap = unit_importance_to_feature_map(np.abs(phi).mean(axis=0), unit_to_members, working_cols, self.feature_names_in_)
        report["mean_abs_shap"] = _mean_abs_shap
        report["mean_abs_shap_coverage"] = dict(
            n_covered=len(_mean_abs_shap), n_input_features=int(n_features),
            complete=bool(len(_mean_abs_shap) == int(n_features)),
            unit_level=bool(unit_to_members is not None and len(unit_to_members) < len(_mean_abs_shap)),
        )

        # Adaptive prescreen narrowing (iter59): when SHAP per-fold ranks are unstable, NARROW the
        # cap so noisy mid-rank features don't get injected into beam's candidate pool. The lever is
        # measurement-driven (median pairwise Spearman of per-fold mean |phi| feature ranks) and only
        # ever narrows; high-stability regimes keep the current cap. Computed BEFORE the prescreen
        # block below so the resolved cap drives that block's keep count.
        effective_brute_force_cap = self.brute_force_max_features
        adaptive_info: Optional[dict] = None
        ladder_mode = str(getattr(self, "prescreen_ladder_mode", "knee") or "knee").lower()
        if ladder_mode == "knee":
            # Data-driven ladder: narrow the cap toward the knee of the sorted |phi| importance curve.
            # Dense-signal frames keep the full cap; sparse frames prune harder. Always runs.
            importance_full = np.abs(phi).mean(axis=0)
            effective_brute_force_cap, adaptive_info = _resolve_knee_prescreen_cap(importance_full, default_cap=self.brute_force_max_features)
            report["adaptive_prescreen"] = adaptive_info
        elif ladder_mode == "hardcoded" and want_per_fold_phi and per_fold_phi_mean is not None and per_fold_phi_mean.shape[0] >= 2:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_phi_rank_stability

            stability = compute_phi_rank_stability(per_fold_phi_mean, top_k=2 * max(self.brute_force_max_features, 40))
            effective_brute_force_cap = _resolve_adaptive_prescreen_width(stability, default_cap=self.brute_force_max_features)
            adaptive_info = dict(
                stability=float(stability),
                default_cap=int(self.brute_force_max_features),
                effective_cap=int(effective_brute_force_cap),
            )
            report["adaptive_prescreen"] = adaptive_info

        # su_seeded_interactions screen (#5b, OPT-IN) - resolve the synergistic operand pairs in
        # PROXY-column space BEFORE the importance pre-screen, so their operand proxy-columns can be
        # RESCUED past the prescreen (pure-interaction operands have ~0 mean|phi|, the regime the
        # additive proxy is blind to, so they sit in the tail the prescreen drops). The prefilter
        # stage already ran the cheap screen + rescued the operands past the PREFILTER when the
        # prefilter narrowed the frame (storing ``self._su_seeded_pairs_orig``); if it did NOT run
        # (narrow data, no prefilter) we run the screen on X_proxy here. Either way the cost is the
        # O(P)+O(K) screen, never the O(P^2) tensor, and it NO-OPs cleanly when the SNR gate clears
        # nothing. Operand columns are keyed by ORIGINAL feature name and mapped to proxy columns via
        # unit_to_members so clustering's unit-rename does not break the pairing.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_interactions import resolve_su_seeded_pairs

        _su_kept_pairs, _su_screen_info, _su_rescue_proxy_idx = resolve_su_seeded_pairs(
            self, phi, X_proxy, y_phi, unit_to_members, working_cols, X_cols, report, _stage)

        # Importance pre-screen: when the proxy still has more columns than the exact-search budget,
        # keep the top-K by SHAP importance (mean |phi|) so exhaustive-approx stays feasible.
        n_proxy = phi.shape[1]
        proxy_cols_kept = np.arange(n_proxy)  # proxy(unit) columns behind the current phi columns
        prescreen_top = self.prescreen_top
        if prescreen_top is None and n_proxy > effective_brute_force_cap and self.optimizer in ("auto", "bruteforce", "bruteforce_gpu"):
            prescreen_top = effective_brute_force_cap
        phi, phi_var, proxy_cols_kept, unit_to_members = self._fit_step3_prescreen_top_none(prescreen_top, n_proxy, _stage, residual_blend_importance, phi, base, y_phi, _su_rescue_proxy_idx, residual_rescue_proxy_idx, phi_var, unit_to_members, report, proxy_cols_kept)

        optimizer = self._resolve_optimizer(phi.shape[1], n_rows=phi.shape[0])
        # One clamp, reused by the search AND by every downstream stage that samples subsets of the
        # SAME phi (trust-guard anchors): min_features is an original-feature-space floor, phi is in
        # proxy space, and an unsatisfiable floor empties their candidate/anchor pools alike.
        _effective_min_card = resolve_effective_min_features(self.min_features, int(phi.shape[1]))
        with _stage("search"):
            candidates = self._run_search(optimizer, phi, base, y_phi)

        # Merge interaction_aware / proxy_mode="interaction" / su_seeded-sparse candidates into the
        # search's proxy-best subsets (carved to ``_shap_proxied_fit_interactions``; each augmentation
        # is independently opt-in and no-ops cleanly when its gate doesn't clear).
        from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_interactions import augment_candidates_with_interactions

        candidates = augment_candidates_with_interactions(
            self, candidates, phi, base, y_phi, X_proxy, proxy_cols_kept, y_search, model_template,
            unit_to_members, report, _stage, _su_kept_pairs, _su_screen_info)

        # min_selected_ratio guard: the proxy degrades for small subsets (the <50% wall). Ratio is in
        # proxy-column space (units/pre-screened columns).
        n_proxy_cols = phi.shape[1]
        candidates = _apply_min_selected_ratio(candidates, n_proxy_cols, self.min_selected_ratio)
        if _effective_min_card != int(self.min_features):
            report["min_features_clamped"] = dict(requested=int(self.min_features), effective=int(_effective_min_card), n_proxy_cols=int(n_proxy_cols))
        if not candidates:
            # Distinct type (still a RuntimeError subclass) so a caller can tell "this selector found
            # nothing it was allowed to return" apart from "this selector crashed".
            raise ShapProxiedNoCandidatesError(
                f"ShapProxiedFS: search produced no candidate subsets "
                f"(optimizer={optimizer}, proxy_cols={n_proxy_cols}, min_card={_effective_min_card}, "
                f"max_features={self.max_features}, min_selected_ratio={self.min_selected_ratio}).")

        report.update(optimizer=optimizer, n_candidates=len(candidates), proxy_best=dict(features=tuple(candidates[0][1]), proxy_loss=candidates[0][0]))

        # One honest-retrain memo shared across trust guard, re-validation, ablation, and within-cluster
        # refine: within this fit the train/holdout split + model + metric are fixed, so a retrain's
        # loss is determined by the (column subset, seed). seed=None fits (trust anchors, ablation,
        # refine) frequently repeat the SAME large subset (e.g. the chosen winner is retrained in BOTH
        # the ablation and as refine's starting base) - the cache returns those identical floats
        # without a duplicate fit. Random-seeded re-validation fits get distinct seeds, never wrongly
        # merged. Numerically identical to the uncached path (deterministic model on fixed data).
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import HonestLossCache

        honest_cache = HonestLossCache()
        # iter80: extend iter79's disk-cache wiring through the honest-retrain stages
        # (proxy_trust_guard + revalidate_top_n + active_learning_revalidate). The disk cache adds a
        # cross-process layer underneath the in-memory ``honest_cache``: within-fit reuse stays in
        # RAM (HonestLossCache), repeat-fit reuse hits disk (DiskCache keyed on (X_search, y_search,
        # X_holdout, y_holdout, cols, template, seed, cap)). ``cache_dir=None`` (default) keeps the
        # legacy in-memory-only contract bit-identical. See ``_shap_proxy_revalidate.disk_cache_dir``
        # docstrings for the cache-key composition + best-effort failure policy.
        rv = dict(classification=self.classification, metric=self.metric, n_jobs=self.n_jobs,
                  unit_to_members=unit_to_members, cache=honest_cache,
                  inner_n_jobs_cap=self.inner_n_jobs_cap,
                  disk_cache_dir=self.cache_dir)

        # Proxy-trust diagnostic (proxy ranks units; honest retrains on member columns).
        self._fit_step4_proxy_trust_diagnostic(report, working_cols, phi, unit_to_members, _stage, base, y_phi, model_template, X_search, X_hold, y_hold, _effective_min_card, rv)

        # Unified candidate re-ranking before the expensive top-N honest retrains: order by the
        # corrector's predicted honest loss (#3/#6, falls back to raw proxy) PLUS an uncertainty
        # penalty (#7). Focuses the retrain budget on subsets that are honestly-best AND stable.
        candidates, score = self._fit_step5_penalty_focuses_retrain(candidates, report, phi, phi_var, n_features)

        # Expose the ranked candidate subsets (expanded to feature names) so downstream patterns
        # (e.g. proposal-generator seeding RFECV/genetic honest search) can consume them.
        best_idx = self._fit_step6_proposal_generator_seeding(unit_to_members, candidates, report, _budget_exhausted, _budget_max_mins, _budget_stop_file, _stage, model_template, X_search, y_search, X_hold, y_hold, phi, rv, n_features, score, honest_cache)

        # Expand best proxy subset -> original member columns, then optionally prune redundant members.
        member_cols = self._fit_step7_expand_best_proxy(unit_to_members, best_idx)
        member_cols = self._fit_step8_expand_best_proxy(unit_to_members, member_cols, _budget_exhausted, _stage, best_idx, residual_protected_working_cols, model_template, X_search, y_search, X_hold, y_hold, honest_cache, phi, base, y_phi, report)

        # Expose sklearn contract: map working-space member columns back to ORIGINAL indices (the
        # pre-filter may have restricted the working set), names in INPUT column order.
        best_set = {int(working_cols[i]) for i in member_cols}
        self.selected_features_ = [c for i, c in enumerate(self.feature_names_in_) if i in best_set]
        self.support_ = np.array([i in best_set for i in range(n_features)], dtype=bool)
        score_on_report_slice(self, report, X_search, y_search, _report_slice, working_cols, member_cols, model_template)  # unbiased figure, if carved
        self.shap_proxy_report_ = report
        if self.verbose:
            logger.info("ShapProxiedFS: optimizer=%s selected %d/%d features: %s", optimizer, len(self.selected_features_), n_features, self.selected_features_)
        return self
