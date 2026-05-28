"""ShapProxiedFS -- feature selection by ranking subsets via summed SHAP values.

Idea (Mazzanti / TDS, extended): train one model on all features, compute SHAP values once, then
approximate the OOS prediction of a model trained on any feature subset ``S`` by the coalition value
``base + sum_{j in S} phi_j``. Rank subsets by a proper ML metric of ``(y, proxy_pred)`` instead of
honestly retraining 2^n models, then honestly re-validate the cheap top-N to pick the final subset.

Pipeline: fit big model -> OOF SHAP (per-fold base) -> proxy-rank subsets (exact numba/GPU brute
force for n<=~22, else beam/greedy/GA/annealing/gradient) -> proxy-trust guard -> honest re-validate
top-N on a disjoint holdout -> expose the sklearn selector contract.

Honest about limits: the proxy attributes the *full* model restricted to S, not a model retrained on
S, so it under-credits subsets that drop features whose signal correlated survivors could recover
(the empirical "<50% coverage breaks down" wall). Hence the trust guard, the disjoint-holdout
re-validation, the ``min_selected_ratio`` knob, and the importance-top-k ablation in the report.

sklearn contract mirrors BorutaShap: ``support_`` (bool mask in input-column order),
``selected_features_`` (names in input order), ``feature_names_in_``, ``n_features_in_``;
``transform`` uses name-based ``X.loc[:, selected]``; ``NotFittedError`` before fit.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_EXACT_OPTIMIZERS = {"bruteforce", "bruteforce_gpu"}
_HEURISTIC_OPTIMIZERS = {"beam", "greedy_forward", "greedy_backward", "multistart", "genetic", "annealing", "gradient"}


class ShapProxiedFS(BaseEstimator, TransformerMixin):
    """SHAP-coalition-proxy feature selector (sklearn transformer)."""

    def __init__(
        self,
        model=None,
        classification: bool = True,
        metric: Optional[str] = None,
        optimizer: str = "auto",
        *,
        out_of_fold: bool = True,
        n_splits: int = 5,
        n_models: int = 1,
        min_features: int = 1,
        max_features: Optional[int] = None,
        top_n: int = 30,
        holdout_size: float = 0.25,
        revalidate: bool = True,
        n_revalidation_models: int = 3,
        lambda_stab: float = 0.5,
        parsimony_tol: float = 0.02,
        min_selected_ratio: float = 0.0,
        trust_guard: bool = True,
        n_anchors: int = 30,
        spearman_floor: float = 0.6,
        run_importance_ablation: bool = True,
        use_bias_corrector: bool = True,
        active_learning: bool = False,
        active_learning_budget: int | None = None,
        config_jitter: bool = False,
        uncertainty_penalty: float = 0.0,
        interaction_aware: bool = False,
        max_interaction_features: int = 16,
        beam_width: int = 8,
        brute_force_max_features: int = 22,
        use_gpu: bool = False,
        prefilter_top: int | None = 2000,
        prefilter_method: str = "auto",
        prefilter_n_estimators: int | None = 100,
        cluster_features: bool | str = "auto",
        cluster_corr_threshold: float = 0.7,
        cluster_weighting: str = "pca_pc1",
        cluster_use_gpu: bool | str = "auto",
        cluster_auto_threshold: int = 40,
        prescreen_top: int | None = None,
        within_cluster_refine: bool = True,
        refine_n_estimators: int | None = 100,
        trust_guard_n_estimators: int | None = 100,
        n_jobs: int = -1,
        random_state: int = 0,
        verbose: bool = True,
        tqdm: bool = False,
    ):
        self.model = model
        self.classification = classification
        self.metric = metric
        self.optimizer = optimizer
        self.out_of_fold = out_of_fold
        self.n_splits = n_splits
        self.n_models = n_models
        self.min_features = min_features
        self.max_features = max_features
        self.top_n = top_n
        self.holdout_size = holdout_size
        self.revalidate = revalidate
        self.n_revalidation_models = n_revalidation_models
        self.lambda_stab = lambda_stab
        self.parsimony_tol = parsimony_tol
        self.min_selected_ratio = min_selected_ratio
        self.trust_guard = trust_guard
        self.n_anchors = n_anchors
        self.spearman_floor = spearman_floor
        self.run_importance_ablation = run_importance_ablation
        self.use_bias_corrector = use_bias_corrector
        self.active_learning = active_learning
        self.active_learning_budget = active_learning_budget
        self.config_jitter = config_jitter
        self.uncertainty_penalty = uncertainty_penalty
        self.interaction_aware = interaction_aware
        self.max_interaction_features = max_interaction_features
        self.beam_width = beam_width
        self.brute_force_max_features = brute_force_max_features
        self.use_gpu = use_gpu
        self.prefilter_top = prefilter_top
        self.prefilter_method = prefilter_method
        # ``prefilter_n_estimators`` caps the cloned ranking booster's tree count inside the
        # pre-filter ("model" / "fast_model" / "gpu_model"). The pre-filter consumes only the rank
        # order of ``feature_importances_``, not an absolute loss number a user sees, so reducing
        # the tree count is a pure-speed lever: importance attribution stabilises well below the
        # default 300 trees. Same "cap-the-ranker" pattern as iter9's refine / trust-guard caps.
        # ``fast_model`` already sets a reduced budget (template / 4); the cap clamps via
        # ``min(current, cap)`` so it can never INCREASE fast_model's tree count. ``univariate`` is
        # a no-op. ``None`` disables the cap (legacy uncapped behaviour).
        self.prefilter_n_estimators = prefilter_n_estimators
        self.cluster_features = cluster_features
        self.cluster_corr_threshold = cluster_corr_threshold
        self.cluster_weighting = cluster_weighting
        self.cluster_use_gpu = cluster_use_gpu
        self.cluster_auto_threshold = cluster_auto_threshold
        self.prescreen_top = prescreen_top
        self.within_cluster_refine = within_cluster_refine
        # ``refine_n_estimators`` caps the per-trial booster size inside ``within_cluster_refine``.
        # Refine compares relative honest losses to decide whether a member-drop respects
        # ``parsimony_tol``; the ranking stabilises well before the default 300 trees, so capping at
        # ~100 trees cuts each fit ~3x while keeping the drop decision intact. None disables the cap.
        self.refine_n_estimators = refine_n_estimators
        # ``trust_guard_n_estimators`` caps the per-anchor booster size inside ``proxy_trust_guard``.
        # The trust report only consumes RANKS of anchor losses (Spearman / Kendall / recall@k); a
        # capped booster gives a faithful fidelity signal at ~3x lower cost. None disables the cap.
        self.trust_guard_n_estimators = trust_guard_n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tqdm = tqdm
        self._rng = np.random.default_rng(random_state)

    @staticmethod
    def preflight(X, y, *, classification: bool = True, **kwargs):
        """Cheap "will-it-shine?" check BEFORE a full fit: returns a recommendation
        (run / caution / fallback) + dataset diagnostics + reasons. See ``_shap_proxy_preflight``."""
        from mlframe.feature_selection._shap_proxy_preflight import preflight as _preflight

        return _preflight(X, y, classification=classification, **kwargs)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_pandas(X):
        try:
            import polars as pl
        except ImportError:
            pl = None
        if pl is not None and isinstance(X, pl.DataFrame):
            from mlframe.training.utils import get_pandas_view_of_polars_df

            return get_pandas_view_of_polars_df(X)
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    def _coerce_target(self, y):
        try:
            import polars as pl
            if isinstance(y, pl.Series):
                y = y.to_numpy()
        except ImportError:
            pass
        y = np.asarray(y)
        if self.classification:
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(
                    f"ShapProxiedFS(classification=True) supports binary targets only; got {len(classes)} classes."
                )
            self.classes_ = classes
            y = (y == classes[1]).astype(np.float64)
        else:
            y = y.astype(np.float64)
        return y

    def _resolve_optimizer(self, n_features: int) -> str:
        opt = self.optimizer
        if opt != "auto":
            return opt
        from mlframe.feature_selection._shap_proxy_search import total_subsets

        if n_features <= self.brute_force_max_features:
            n_sub = total_subsets(n_features, self.min_features, self.max_features)
            if n_sub <= 2_000_000:
                return "bruteforce_gpu" if self.use_gpu else "bruteforce"
        return "beam"

    def _run_search(self, optimizer, phi, base, y):
        """Dispatch to the chosen optimizer; returns list of (proxy_loss, feature_idx tuple)."""
        kw = dict(classification=self.classification, metric=self.metric,
                  max_card=self.max_features, top_n=self.top_n)
        if optimizer == "bruteforce":
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features,
                                     parallel=(phi.shape[1] >= 14), **kw)
        if optimizer == "bruteforce_gpu":
            from mlframe.feature_selection._shap_proxy_gpu import brute_force_top_n_gpu, gpu_available

            if gpu_available():
                return brute_force_top_n_gpu(phi, base, y, min_card=self.min_features, **kw)
            logger.warning("ShapProxiedFS: use_gpu requested but no CUDA device; falling back to numba brute force.")
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features, parallel=True, **kw)
        from mlframe.feature_selection import _shap_proxy_heuristics as H

        if optimizer == "beam":
            return H.beam_search(phi, base, y, beam_width=self.beam_width, min_card=self.min_features, **kw)
        if optimizer == "greedy_forward":
            return H.greedy_forward(phi, base, y, classification=self.classification, metric=self.metric,
                                    max_card=self.max_features, top_n=self.top_n)
        if optimizer == "greedy_backward":
            return H.greedy_backward(phi, base, y, classification=self.classification, metric=self.metric,
                                     min_card=self.min_features, top_n=self.top_n)
        if optimizer == "multistart":
            return H.multistart_local(phi, base, y, rng=self._rng, **kw)
        if optimizer == "genetic":
            return H.genetic(phi, base, y, rng=self._rng, **kw)
        if optimizer == "annealing":
            return H.simulated_annealing(phi, base, y, rng=self._rng, **kw)
        if optimizer == "gradient":
            from mlframe.feature_selection._shap_proxy_gradient import gradient_top_n

            return gradient_top_n(phi, base, y, classification=self.classification, metric=self.metric,
                                  random_state=int(self.random_state), top_n=self.top_n)
        raise ValueError(f"Unknown optimizer={optimizer!r}")

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        import time
        from contextlib import contextmanager

        from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

        # Optional per-stage wall-clock instrumentation for the scaling benchmark / profiling. Set
        # ``self._stage_timings`` to a dict before calling fit and each stage's seconds land in it; a
        # no-op otherwise (zero overhead beyond a dict lookup), so production fits are unaffected.
        _timings = getattr(self, "_stage_timings", None)

        @contextmanager
        def _stage(name):
            if _timings is None:
                yield
                return
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _timings[name] = _timings.get(name, 0.0) + (time.perf_counter() - t0)

        X = self._to_pandas(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        self.feature_names_in_ = np.asarray(list(X.columns))
        self.n_features_in_ = int(X.shape[1])
        n_features = self.n_features_in_
        y = self._coerce_target(y)

        model_template = self.model if self.model is not None else make_default_estimator(
            self.classification, random_state=int(self.random_state))

        # Disjoint holdout for honest re-validation + trust guard (avoids winner's curse).
        stratify = y if self.classification else None
        idx_all = np.arange(len(X))
        idx_search, idx_hold = train_test_split(
            idx_all, test_size=self.holdout_size, random_state=int(self.random_state),
            shuffle=True, stratify=stratify)
        X_search, y_search = X.iloc[idx_search].reset_index(drop=True), y[idx_search]
        X_hold, y_hold = X.iloc[idx_hold].reset_index(drop=True), y[idx_hold]

        report: dict = {}

        # Cheap native-importance pre-filter BEFORE the expensive OOF-SHAP. SHAP cost scales with the
        # column count, and clustering only compresses CORRELATED features (independent noise stays as
        # singletons), so on wide data SHAP would otherwise run on ~all columns. Rank all features and
        # keep the top-K; ``working_cols`` maps the surviving working columns back to original indices
        # for the final selector output. ``prefilter_method`` trades speed against interaction-awareness
        # (model / univariate / fast_model / gpu_model); "auto" stays quality-safe for moderate widths
        # and switches to a fast method only for very wide data -- see ``_shap_proxy_prefilter``.
        working_cols = np.arange(n_features)
        if self.prefilter_top is not None and n_features > self.prefilter_top:
            from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

            with _stage("prefilter"):
                working_cols, pf_info = prefilter_columns(
                    model_template, X_search, y_search, method=self.prefilter_method,
                    prefilter_top=self.prefilter_top, classification=self.classification,
                    n_features=n_features, n_estimators_cap=self.prefilter_n_estimators)
                if len(working_cols) < n_features:
                    X_search = X_search.iloc[:, working_cols].reset_index(drop=True)
                    X_hold = X_hold.iloc[:, working_cols].reset_index(drop=True)
                report["prefilter"] = pf_info

        # Optional correlated-feature clustering: collapse to denoised UNITS so SHAP + search run on
        # hundreds of columns, not tens of thousands. unit_to_members maps proxy(unit) index ->
        # original feature columns; None means proxy index == feature column (identity).
        do_cluster = self.cluster_features is True or (
            self.cluster_features == "auto" and n_features > self.cluster_auto_threshold)
        if do_cluster:
            from mlframe.feature_selection._shap_proxy_cluster import (
                build_unit_matrix, cluster_correlated_features, cluster_summary)

            with _stage("clustering"):
                labels = cluster_correlated_features(
                    X_search.values, threshold=self.cluster_corr_threshold, use_gpu=self.cluster_use_gpu)
                units, unit_to_members, _kind = build_unit_matrix(
                    X_search.values, labels, weighting=self.cluster_weighting)
                X_proxy = pd.DataFrame(units, columns=[f"unit{i}" for i in range(units.shape[1])])
                report["clustering"] = cluster_summary(unit_to_members)
        else:
            X_proxy = X_search
            unit_to_members = None

        # SHAP attribution on the proxy (unit or raw) columns. Request per-model attribution variance
        # only when the uncertainty lever is active AND we actually have multiple models to vary.
        want_var = self.uncertainty_penalty > 0 and self.n_models > 1
        with _stage("oof_shap"):
            shap_out = compute_shap_matrix(
                model_template, X_proxy, y_search, classification=self.classification,
                out_of_fold=self.out_of_fold, n_splits=self.n_splits, n_models=self.n_models,
                config_jitter=self.config_jitter, return_variance=want_var,
                rng=self._rng, tqdm_desc=("shap-oof" if self.tqdm else None), n_jobs=self.n_jobs)
        if want_var:
            phi, base, y_phi, phi_var = shap_out
        else:
            phi, base, y_phi, phi_var = (*shap_out, None)

        # Importance pre-screen: when the proxy still has more columns than the exact-search budget,
        # keep the top-K by SHAP importance (mean |phi|) so exhaustive-approx stays feasible.
        n_proxy = phi.shape[1]
        proxy_cols_kept = np.arange(n_proxy)  # proxy(unit) columns behind the current phi columns
        prescreen_top = self.prescreen_top
        if prescreen_top is None and n_proxy > self.brute_force_max_features and self.optimizer in ("auto", "bruteforce", "bruteforce_gpu"):
            prescreen_top = self.brute_force_max_features
        if prescreen_top is not None and prescreen_top < n_proxy:
            with _stage("prescreen"):
                importance = np.abs(phi).mean(axis=0)
                keep = np.sort(np.argsort(-importance)[:prescreen_top])
                phi = np.ascontiguousarray(phi[:, keep])
                proxy_cols_kept = keep
                if phi_var is not None:
                    phi_var = np.ascontiguousarray(phi_var[:, keep])
                if unit_to_members is not None:
                    unit_to_members = [unit_to_members[i] for i in keep]
                else:
                    unit_to_members = [np.array([int(i)], dtype=np.int64) for i in keep]
                report["prescreen"] = dict(kept=int(len(keep)), of=int(n_proxy))

        optimizer = self._resolve_optimizer(phi.shape[1])
        with _stage("search"):
            candidates = self._run_search(optimizer, phi, base, y_phi)

        # Interaction-aware coalition (#5): for interaction-heavy targets the main-effect sum can't
        # see a pair's joint signal (XOR partners have ~0 main effect). Add candidates ranked by the
        # SHAP-interaction coalition value and let honest re-validation arbitrate. Bounded to a small
        # proxy width (post pre-screen); tensor is O(P^2).
        if self.interaction_aware and phi.shape[1] <= self.max_interaction_features:
            from mlframe.feature_selection._shap_proxy_interactions import (
                compute_interaction_tensor, interaction_top_n)

            X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
            Phi, ibase = compute_interaction_tensor(
                model_template, X_proxy_kept, y_search, classification=self.classification, rng=self._rng)
            icands = interaction_top_n(
                Phi, ibase, y_phi, classification=self.classification, metric=self.metric,
                min_card=self.min_features, max_card=self.max_features, top_n=self.top_n,
                exhaustive_max=self.max_interaction_features)
            merged = {tuple(sorted(c)): l for l, c in candidates}
            for l, c in icands:
                merged.setdefault(tuple(sorted(c)), l)
            candidates = sorted(((l, c) for c, l in merged.items()), key=lambda t: t[0])
            report["interaction_aware"] = dict(applied=True, n_proxy=int(phi.shape[1]), n_interaction_candidates=len(icands))

        # min_selected_ratio guard: the proxy degrades for small subsets (the <50% wall). Ratio is in
        # proxy-column space (units/pre-screened columns).
        n_proxy_cols = phi.shape[1]
        if self.min_selected_ratio > 0:
            filtered = [(l, c) for l, c in candidates if len(c) / n_proxy_cols >= self.min_selected_ratio]
            candidates = filtered or candidates  # never return empty
        if not candidates:
            raise RuntimeError("ShapProxiedFS: search produced no candidate subsets.")

        report.update(optimizer=optimizer, n_candidates=len(candidates),
                      proxy_best=dict(features=tuple(candidates[0][1]), proxy_loss=candidates[0][0]))

        # One honest-retrain memo shared across trust guard, re-validation, ablation, and within-cluster
        # refine: within this fit the train/holdout split + model + metric are fixed, so a retrain's
        # loss is determined by the (column subset, seed). seed=None fits (trust anchors, ablation,
        # refine) frequently repeat the SAME large subset (e.g. the chosen winner is retrained in BOTH
        # the ablation and as refine's starting base) -- the cache returns those identical floats
        # without a duplicate fit. Random-seeded re-validation fits get distinct seeds, never wrongly
        # merged. Numerically identical to the uncached path (deterministic model on fixed data).
        from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache

        honest_cache = HonestLossCache()
        rv = dict(classification=self.classification, metric=self.metric, n_jobs=self.n_jobs,
                  unit_to_members=unit_to_members, cache=honest_cache)

        # Proxy-trust diagnostic (proxy ranks units; honest retrains on member columns).
        if self.trust_guard:
            from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

            with _stage("trust_guard"):
                report["trust"] = proxy_trust_guard(
                    phi, base, y_phi, model_template, X_search, X_hold, y_hold,
                    n_anchors=self.n_anchors, rng=self._rng, min_card=self.min_features,
                    max_card=self.max_features, spearman_floor=self.spearman_floor,
                    n_estimators_cap=self.trust_guard_n_estimators, **rv)

        # Unified candidate re-ranking before the expensive top-N honest retrains: order by the
        # corrector's predicted honest loss (#3/#6, falls back to raw proxy) PLUS an uncertainty
        # penalty (#7). Focuses the retrain budget on subsets that are honestly-best AND stable.
        score = np.array([c[0] for c in candidates], dtype=np.float64)  # raw proxy loss
        if self.use_bias_corrector and self.trust_guard and report.get("trust", {}).get("_corrector_data"):
            from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy

            cd = report["trust"]["_corrector_data"]
            corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
            if not corrector.fallback:
                cards = np.array([len(c[1]) for c in candidates], dtype=np.float64)
                redund = np.array([subset_redundancy(phi, c[1]) for c in candidates], dtype=np.float64)
                score = corrector.predict(score, cards, redund)
                report["bias_corrector"] = dict(applied=True, n_anchors=len(cd["proxy"]))
        if self.uncertainty_penalty > 0 and phi_var is not None:
            from mlframe.feature_selection._shap_proxy_objective import subset_uncertainty

            unc = np.array([subset_uncertainty(phi_var, c[1]) for c in candidates], dtype=np.float64)
            score = score + self.uncertainty_penalty * unc
            report["uncertainty"] = dict(applied=True, penalty=float(self.uncertainty_penalty))
        order = np.argsort(score, kind="stable")
        candidates = [candidates[i] for i in order]

        # Expose the ranked candidate subsets (expanded to feature names) so downstream patterns
        # (e.g. proposal-generator seeding RFECV/genetic honest search) can consume them.
        def _cand_names(idx):
            if unit_to_members is not None:
                cols = sorted({int(c) for u in idx for c in unit_to_members[int(u)]})
            else:
                cols = sorted(int(i) for i in idx)
            return [str(self.feature_names_in_[i]) for i in cols]

        report["candidates"] = [dict(proxy_loss=float(l), features=_cand_names(c))
                                for l, c in candidates[: self.top_n]]

        # Honest re-validation of the top-N on the disjoint holdout (active-learning variant when the
        # corrector anchors are available, else the static top-N retrain).
        if self.revalidate:
            cdata = report.get("trust", {}).get("_corrector_data")
            with _stage("revalidation"):
                if self.active_learning and cdata:
                    from mlframe.feature_selection._shap_proxy_revalidate import active_learning_revalidate

                    budget = self.active_learning_budget or self.top_n
                    best_idx, ranked, n_eval = active_learning_revalidate(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        corrector_data=cdata, phi=phi, budget=budget, n_models=self.n_revalidation_models,
                        parsimony_tol=self.parsimony_tol, rng=self._rng, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n],
                                                  active_learning=dict(n_evaluated=n_eval, budget=budget))
                else:
                    from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

                    best_idx, ranked, baseline = revalidate_top_n(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        n_models=self.n_revalidation_models, lambda_stab=self.lambda_stab,
                        parsimony_tol=self.parsimony_tol, rng=self._rng, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n], random_baseline=baseline)
        else:
            best_idx = tuple(candidates[0][1])

        # Importance-top-k ablation (unique-value gate vs plain SHAP importance).
        if self.run_importance_ablation and best_idx:
            from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

            with _stage("importance_ablation"):
                report["importance_ablation"] = importance_topk_ablation(
                    phi, best_idx, model_template, X_search, y_search, X_hold, y_hold,
                    classification=self.classification, metric=self.metric, unit_to_members=unit_to_members,
                    cache=honest_cache)

        # Expand best proxy subset -> original member columns, then optionally prune redundant members.
        if unit_to_members is not None:
            member_cols = sorted({int(c) for u in best_idx for c in unit_to_members[int(u)]})
        else:
            member_cols = sorted(int(i) for i in best_idx)
        if self.within_cluster_refine and unit_to_members is not None and len(member_cols) > 1:
            from mlframe.feature_selection._shap_proxy_objective import resolve_metric
            from mlframe.feature_selection._shap_proxy_revalidate import (
                _honest_loss, within_cluster_refine,
            )

            with _stage("within_cluster_refine"):
                # Pass the per-unit member lists so refine can collapse each cluster to a single
                # representative in ONE parallel batch (O(sum k_c) trials) instead of legacy
                # O(k^2) greedy drops. unit_to_members is in proxy-unit space; each chosen unit
                # contributes one group of member columns.
                member_groups = [
                    [int(c) for c in unit_to_members[int(u)]] for u in best_idx
                ]
                refined = within_cluster_refine(
                    member_cols, model_template, X_search, y_search, X_hold, y_hold,
                    classification=self.classification, metric=self.metric,
                    parsimony_tol=self.parsimony_tol, n_jobs=self.n_jobs, cache=honest_cache,
                    member_groups=member_groups, refine_n_estimators=self.refine_n_estimators)
                # Final full-template re-evaluation of the ONE chosen subset (uncapped n_estimators).
                # Refine's ranking trials use a cheaper capped booster (~100 trees) to decide WHICH
                # members to drop; the user-visible quality bar (and any downstream report consumer)
                # should see this subset's loss at the SAME booster size the other guards used, so the
                # values are apples-to-apples. The cache lookup is the full-template namespace (no
                # template_id), so this hits any prior pipeline retrain of the same subset (e.g. when
                # refine made no drops, this is a cache hit of the union retrain done elsewhere).
                refine_info = dict(before=len(member_cols), after=len(refined))
                if refined:
                    refine_info["honest_loss_full"] = float(_honest_loss(
                        model_template, X_search, y_search, X_hold, y_hold, list(refined),
                        self.classification, resolve_metric(self.classification, self.metric),
                        cache=honest_cache))
                report["within_cluster_refine"] = refine_info
                member_cols = refined

        # Expose sklearn contract: map working-space member columns back to ORIGINAL indices (the
        # pre-filter may have restricted the working set), names in INPUT column order.
        best_set = {int(working_cols[i]) for i in member_cols}
        self.selected_features_ = [c for i, c in enumerate(self.feature_names_in_) if i in best_set]
        self.support_ = np.array([i in best_set for i in range(n_features)], dtype=bool)
        self.shap_proxy_report_ = report
        if self.verbose:
            logger.info("ShapProxiedFS: optimizer=%s selected %d/%d features: %s",
                        optimizer, len(self.selected_features_), n_features, self.selected_features_)
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, X):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ShapProxiedFS.transform called before fit.")
        X = self._to_pandas(X)
        X.columns = [str(c) for c in X.columns]
        selected = list(self.selected_features_)
        if hasattr(X, "loc"):
            return X.loc[:, selected]
        return X[selected]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError("ShapProxiedFS.get_support called before fit.")
        return np.where(self.support_)[0] if indices else self.support_
