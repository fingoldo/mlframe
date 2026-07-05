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

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import (
    _ADAPTIVE_PRESCREEN_FLOOR, _DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS,
    _DEFAULT_BRUTE_FORCE_MAX_FEATURES, _DEFAULT_BRUTE_FORCE_N_SUB_GATE,
    _DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES, _EXACT_OPTIMIZERS, _HEURISTIC_OPTIMIZERS,
    _resolve_adaptive_prescreen_thresholds, _resolve_adaptive_prescreen_width,
    _resolve_adaptive_n_anchors, _resolve_knee_prescreen_cap,
    _resolve_brute_force_max_features, _resolve_brute_force_n_sub_gate,
    _resolve_cluster_su_auto_max_features,
)
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit import ShapProxiedFitMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_methods import ShapProxiedMethodsMixin
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_precomputed import restrict_artifacts

logger = logging.getLogger(__name__)


class ShapProxiedFS(ShapProxiedFitMixin, ShapProxiedMethodsMixin, BaseEstimator, TransformerMixin):
    """SHAP-coalition-proxy feature selector (sklearn transformer)."""

    def __init__(
        self,
        model=None,
        classification: bool = True,
        metric: Optional[str] = None,
        optimizer: str = "auto",
        *,
        out_of_fold: bool = True,
        # iter86: OOF-SHAP CV fold count. Lowered 5 -> 3 after a quality sweep across narrow
        # (n=2000, p=200), C3-tier (n=2000, p=2000) and wide (n=5000, p=10000) regimes: recall stayed
        # within 1 informative of the prior 5-fold default at all three widths (and beat it at C3),
        # while OOF-SHAP stage wall dropped 41-51% on the regimes where it dominates. Each held-out
        # split is correspondingly larger (33% vs 20% of rows), which adds a touch of attribution
        # noise; the mean-|phi| ranking that downstream search consumes is robust to it at the
        # measured widths. Callers can opt back into the legacy 5-fold by passing ``n_splits=5``.
        n_splits: int = 3,
        n_models: int = 1,
        min_features: int = 1,
        max_features: Optional[int] = None,
        top_n: int = 30,
        holdout_size: float = 0.25,
        revalidate: bool = True,
        n_revalidation_models: int = 3,
        lambda_stab: float = 0.5,
        # parsimony_tol: within_cluster_refine drops a member while the honest-loss increase stays below this. It is a
        # RECALL-vs-PRECISION dial, NOT a one-way win, so the default is the precision-tuned 0.02 that matches this
        # selector's native contract (exclude noise + redundant columns): on the biz_val bed 0.02 keeps 0 noise columns,
        # whereas 0.005 admits 1-2 (it prunes less). The opposite regime holds when you optimise DOWNSTREAM AUC across
        # models: on the fs_hybrid synthetic (3 scenarios x 2 seeds) 0.005 keeps ~11 feats / recovers 5.0/7 informative /
        # mean AUC 0.795 vs 0.02's ~6 feats / 4.3/7 / 0.792 (0.02 over-prunes there). So callers that want max recovery
        # for a downstream model set parsimony_tol=0.005 explicitly (the fs_hybrid ShapSel wrapper and HybridSelector do);
        # the standalone default stays 0.02 for clean, parsimonious, low-false-positive subsets. within_cluster_refine=
        # False skips refinement entirely.
        parsimony_tol: float = 0.02,
        min_selected_ratio: float = 0.0,
        trust_guard: bool = True,
        n_anchors: int | str = "auto",
        # ``None`` is the "unset" sentinel that resolves to 0.5 at fit time. A real float (incl. an
        # explicit 0.5) means the user pinned it, so the both-floors-set conflict guard below can
        # detect ``spearman_floor`` + an explicit ``fidelity_floor=0.5`` instead of mistaking the
        # explicit value for the default.
        fidelity_floor: Optional[float] = None,
        spearman_floor: Optional[float] = None,
        run_importance_ablation: bool = True,
        use_bias_corrector: bool = True,
        active_learning: bool = False,
        active_learning_budget: int | None = None,
        config_jitter: bool = False,
        uncertainty_penalty: float = 0.0,
        interaction_aware: bool = False,
        max_interaction_features: int = 16,
        # ``proxy_mode`` ("additive" DEFAULT | "interaction"): how a feature SUBSET is scored by the
        # proxy. "additive" = ``base + sum_{j in S} phi_j`` (purely additive SHAP coalition; blind to
        # non-additive pairs). "interaction" re-scores the additive candidates under
        # ``base + sum phi_j + 2*sum_{i<j in S} Phi_ij`` (adds the off-diagonal TreeSHAP interaction
        # values) + a gated pair sweep, so an XOR / multiplicative pair earns the joint credit the
        # additive proxy denies it. The pairwise term is GATED to the top-``interaction_proxy_top_k``
        # features by mean |phi| (O(k^2) memory/cost, not O(P^2)). Default stays "additive": bench
        # (_benchmarks/bench_shap_interaction_proxy.py) shows interaction WINS the competing-XOR bed by
        # ~+0.24 honest-holdout AUC REPLICATED 3/3 seeds, but only 1/6 beds and slightly regresses one
        # additive-redundant seed -- not the majority+no-regression win a default flip requires. Tree
        # models only; non-tree falls back to additive cleanly. REJECTED-as-default != deleted.
        proxy_mode: str = "additive",
        interaction_proxy_top_k: int = 30,
        # su_seeded_interactions (lever A4-4, OPT-IN, default OFF -- mirrors ``interaction_aware``):
        # a CHEAP pairwise-SU SYNERGY screen ranks candidate interaction PAIRS at O(P)+O(K) cost, then
        # the interaction objective runs on ONLY the top-K synergistic pairs (a sparse product-column
        # augmentation), NEVER the O(P^2) TreeSHAP interaction tensor that gates ``interaction_aware``
        # to phi<=16 (a no-op on wide proxies). The screen scores
        # ``synergy(a,b;y) = SU(joint_bin(a,b);y) - max(SU(a;y), SU(b;y))`` -- HIGH exactly for pure
        # interactions whose operands have ~0 marginal SU (XOR / sign(a*b)). A permutation-null SNR
        # GATE skips pairs whose synergy sits below the spurious-pair floor, so noise-buried
        # interactions (hard_synth's ia*ib among 200 noise cols) are correctly NOT seeded and the path
        # NO-OPs cleanly (the additive default is never regressed). Measured win (clear-SNR beds):
        # +0.388 AUC on pure-interaction, +0.072 on synth; no-op on hard_synth. See
        # ``_shap_proxy_interactions.su_synergy_screen`` / ``sparse_interaction_candidates``.
        su_seeded_interactions: bool = False,
        su_seeded_top_k: int = 8,
        su_seeded_n_bins: int = 8,
        su_seeded_max_screen_cols: int = 120,
        su_seeded_snr_z: float = 3.0,
        su_seeded_snr_null_quantile: float = 0.99,
        su_seeded_snr_abs_floor: float = 1e-3,
        su_seeded_n_permutations: int = 3,
        beam_width: int = 8,
        brute_force_max_features: int | None = None,
        adaptive_prescreen_by_stability: bool = False,
        prescreen_ladder_mode: str = "hardcoded",
        use_gpu: bool = False,
        prefilter_top: int | None = 2000,
        prefilter_method: str = "auto",
        prefilter_n_estimators: int | None = 100,
        oof_shap_n_estimators: int | None = 100,
        prefilter_stage1_keep: int | None = None,
        prefilter_univariate_batch_size: int | None = None,
        shap_prefilter_enabled: bool = True,
        shap_prefilter_top: int | None = None,
        shap_prefilter_safety_factor: int = 4,
        shap_prefilter_min_features: int = 40,
        shap_aware_stage1_keep: bool = True,
        shap_aware_stage1_cushion: int = 2,
        shap_aware_stage1_floor: int = 200,
        cluster_features: bool | str = "auto",
        cluster_corr_threshold: float = 0.7,
        cluster_weighting: str = "pca_pc1",
        cluster_use_gpu: bool | str = "auto",
        cluster_auto_threshold: int = 40,
        cluster_use_precomputed_bins: bool = True,
        cluster_su_threshold: float = 0.5,
        cluster_backend: str = "auto",
        cluster_su_auto_max_features: int | None = None,
        cluster_su_n_bins: int = 10,
        prescreen_top: int | None = None,
        # within_cluster_refine drops members while the honest holdout loss stays within ``parsimony_tol`` of best,
        # measured with ShapProxiedFS's OWN booster on a 25% holdout. At the precision-tuned default parsimony_tol=0.02
        # refine yields a clean, parsimonious subset (the native contract: exclude noise + redundancy). When the DOWNSTREAM
        # model is stronger / different (e.g. a 300-tree LightGBM) it can exploit features this proxy finds within-tol-
        # redundant, so AUC-optimising callers loosen the dial to parsimony_tol=0.005 (keeps ~2x the features, beats
        # refine=False in 4/6 fs_hybrid cells, +~0.6pt downstream LightGBM AUC) -- see ``parsimony_tol`` above for that
        # recall-vs-precision tradeoff. Set within_cluster_refine=False to skip refinement entirely.
        within_cluster_refine: bool = True,
        refine_n_estimators: int | None = 100,
        refine_ucb_enabled: bool = True,
        refine_ucb_min_eval_size: int | None = None,
        refine_ucb_slack: float | None = None,
        refine_ucb_stdev_multiplier: float = 1.0,
        revalidation_n_estimators: int | None = 100,
        revalidation_ucb_enabled: bool = True,
        revalidation_ucb_min_eval_size: int | None = None,
        revalidation_ucb_slack: float | None = None,
        revalidation_ucb_stdev_multiplier: float | None = None,
        revalidation_adaptive_n_models: bool = True,
        revalidation_mmr_jaccard_threshold: float | None = None,
        trust_guard_n_estimators: int | None = 25,
        trust_guard_stratified_anchors: bool = False,
        trust_guard_uniform_tail_frac: float = 0.2,
        trust_guard_cardinality_dist: str = "zipf",
        trust_guard_zipf_alpha: float = 0.25,
        trust_guard_fidelity_weights: tuple[float, float] = (0.6, 0.4),
        trust_guard_metric: str = "proxy_fidelity_score",
        n_jobs: int = -1,
        inner_n_jobs_cap: bool = False,
        random_state: int = 0,
        verbose: bool = True,
        tqdm: bool = False,
        # iter66: accept precomputed cross-selector artifacts (canonically from
        # ``MRMR(retain_artifacts=True).export_artifacts()``) so the stage-A
        # univariate F-statistic pre-screen can be replaced with the SU ranking
        # the MRMR screen already computed. ``None`` (default) preserves
        # legacy behaviour byte-identical.
        precomputed: dict | None = None,
        # iter78: GBT family for the default booster (used when ``model is None``). ``None`` (default)
        # preserves the legacy xgboost path byte-identical. ``"catboost"`` swaps in the catboost
        # template and routes SHAP attribution to catboost's native ``get_feature_importance(
        # type='ShapValues')`` kernel (oblivious-tree SHAP that the numba TreeSHAP path does not
        # cover); ``cat_features`` is forwarded to the catboost constructor so categorical columns
        # are handled natively instead of one-hot pre-encoded. Ignored when ``model`` is provided.
        booster_kind: str | None = None,
        cat_features: list | None = None,
        # iter79: content-addressable disk cache for the OOF-SHAP stage. ``None`` (default) disables.
        # Set to a directory path to share phi/base across re-fits with the same (X, y, template,
        # fold/seed) tuple -- e.g. hyperparam sweeps that vary downstream stages but leave the
        # SHAP-attribution input unchanged. The cache is content-addressable, multi-process safe,
        # LRU-evicted; see ``mlframe.utils.disk_cache``.
        cache_dir: str | None = None,
        max_runtime_mins: float = None,
        stop_file: str = "stop",
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
        # ``top_n`` (default 30) bounds the number of candidate subsets the search heuristic
        # forwards for honest revalidation; each candidate gets ``n_revalidation_models`` fits.
        # Iter96 sweep at C3 (width=10000, n_rows=10000, snr=8.0) over {20, 16, 12, 8}:
        # 20=baseline (recall=0.95, chosen_loss=0.135456, reval_wall=5.14s),
        # 16=IDENTICAL chosen subset / recall=0.95 but chosen_loss point-estimate +9.6%
        # (outside the +-5% gate even on the same subset because revalidation ucb resamples a
        # smaller candidate set and the point estimate moves),
        # 12,8=DIFFER (drops inf19, recall regresses to 0.90).
        # Default kept at 30 (and bench-pinned 20 baseline) because shrinking the ceiling
        # crosses the honest_loss point-estimate gate even on the bit-identical winner; the
        # iter50 MMR + iter77 adaptive paths already cut the EVALUATED count below the ceiling
        # when redundancy exists, so the static top_n is rarely the binding cost driver.
        self.top_n = top_n
        self.holdout_size = holdout_size
        self.revalidate = revalidate
        self.n_revalidation_models = n_revalidation_models
        self.lambda_stab = lambda_stab
        self.parsimony_tol = parsimony_tol
        self.min_selected_ratio = min_selected_ratio
        self.trust_guard = trust_guard
        # ``n_anchors`` default (30) was held against an iter93 sweep at C3 (width=10000, n_rows=10000,
        # n_inf=20, n_red=20, snr=8.0, seed=0). The CHOSEN SUBSET is bit-identical from n_anchors=30
        # down to n_anchors=8 and ``trustworthy=True`` everywhere (all composite fidelities clear the
        # 0.5 floor). But the trust SCORE components degrade sharply below 30: recall@k 1.0 (30) -> 0.75
        # (24) -> 0.667 (16) -> 0.50 (12) -> 0.0 (8); composite proxy_fidelity_score 0.9848 -> 0.8875 ->
        # 0.8420 -> 0.7790 -> 0.5571. Spearman alone (0.9746 -> 0.9286) stays inside +/-5% across the
        # whole sweep, but the composite is the gated metric so its dynamic range matters; at small
        # anchors recall@k loses range (k = max(1, n_anchors//5) collapses to 1 by n=8, so a single
        # anchor disagreement zeros it). trust_guard wall cut is also marginal until n=8: 1.54s (30) ->
        # 1.30s (24) -> 1.22s (16) -> 1.52s (12) -> 0.97s (8); the wall is parallelism-overhead bound,
        # not anchor-count bound, until cardinality drops far enough to shrink the dispatch fan-out.
        # ADAPTIVE anchor budget (default ``"auto"``, bench-FLIPPED from fixed 30): the historical
        # fixed 30 anchors thinly cover a very wide raw feature space, weakening the trust guard exactly
        # where the proxy is least trustworthy. ``"auto"`` self-tunes n_anchors = clip(round(6*sqrt(p)),
        # 10, 100) from the RAW input width ``p = n_features_in_`` (resolved at fit time in
        # _shap_proxied_fit): ~30 at p=25, ceiling 100 for p>=278. bench_shapproxied_adaptive_guards
        # WIDE majority win (5/6 seeds x {p=2000, p=6000}): proxy_fidelity_score auto>=fixed at 100
        # anchors -- e.g. p=6000 fid 0.913/0.960/0.974 (auto) vs 0.830/0.907/0.907 (fixed); the one
        # loss is a near-tie (0.898 vs 0.914). A literal int pins the count (recovers legacy fixed-30
        # via ``n_anchors=30``). The iter93 note below explains why a fixed small count erodes the trust
        # SCORE margin -- the adaptive scale lifts the count exactly on the wide frames it was thinnest.
        self.n_anchors = n_anchors
        # ``prescreen_ladder_mode`` (default ``"hardcoded"``): how the post-OOF prescreen cap narrows.
        #   - ``"hardcoded"`` (DEFAULT): the legacy stability-table ladder (requires
        #     ``adaptive_prescreen_by_stability=True`` + OOF to fire; otherwise a no-op).
        #   - ``"knee"`` (OPT-IN, bench-rejected as default): data-driven -- read the sorted |phi|
        #     importance distribution and narrow the cap toward the kneedle knee of the cumulative-
        #     importance curve. Dense-signal keeps the full cap; sparse prunes to the knee. Always runs,
        #     only ever narrows. REJECTED as default (bench_shapproxied_adaptive_guards): on WIDE DENSE
        #     frames (p=2000, 30 inf) it over-prunes to the floor and LOSES held-out AUC ~0.04-0.06
        #     (knee 0.749/0.751/0.762 vs off 0.791/0.798/0.819); on sparse frames it ties. Net loss on
        #     dense, so not the majority win required to flip the default. Kept recoverable for callers
        #     who know their signal is sparse.
        #   - ``"off"``: no narrowing.
        # Store verbatim (sklearn clone contract: __init__ must not transform params); normalised at the use-site in _shap_proxied_fit.
        self.prescreen_ladder_mode = prescreen_ladder_mode
        # ``fidelity_floor`` (iter18, effective default 0.5): below this composite the trust-guard
        # fires LOW. ``None`` is the "unset" sentinel resolved to 0.5 at fit time; storing it raw
        # (no coercion here) keeps the both-floors-set conflict guard able to distinguish an explicit
        # ``fidelity_floor=0.5`` from "user did not pin it" and preserves sklearn ``clone()`` identity.
        # The legacy ``spearman_floor`` kwarg name is preserved as a deprecated alias since iter18 --
        # supplying it emits a ``DeprecationWarning`` at fit time and copies into ``fidelity_floor``.
        # The legacy 0.6 default was set against the raw-Spearman scale (pre-iter16); on the composite
        # scale (iter16+) it is too conservative and trips on the partial-recovery ``interaction_heavy``
        # regime (recovery 6/8). The 0.5 default cleanly separates regimes with
        # ``recovery_rate >= 0.7`` (PASS, min composite 0.5384) from the only failing regime
        # ``xor_interaction`` (recovery_rate 0.333, composite 0.4742). See
        # ``_benchmarks/calib_iter18_fidelity_floor.py``.
        self.fidelity_floor = fidelity_floor
        self.spearman_floor = spearman_floor
        self.run_importance_ablation = run_importance_ablation
        self.use_bias_corrector = use_bias_corrector
        self.active_learning = active_learning
        self.active_learning_budget = active_learning_budget
        self.config_jitter = config_jitter
        self.uncertainty_penalty = uncertainty_penalty
        self.interaction_aware = interaction_aware
        self.max_interaction_features = max_interaction_features
        # Store raw for sklearn clone() identity; validate at use-site (lower()) to avoid mutating params.
        if str(proxy_mode).lower() not in ("additive", "interaction"):
            raise ValueError(f"proxy_mode must be 'additive' or 'interaction'; got {proxy_mode!r}")
        self.proxy_mode = proxy_mode
        self.interaction_proxy_top_k = int(interaction_proxy_top_k)
        self.su_seeded_interactions = su_seeded_interactions
        self.su_seeded_top_k = int(su_seeded_top_k)
        self.su_seeded_n_bins = int(su_seeded_n_bins)
        self.su_seeded_max_screen_cols = int(su_seeded_max_screen_cols)
        self.su_seeded_snr_z = float(su_seeded_snr_z)
        self.su_seeded_snr_null_quantile = float(su_seeded_snr_null_quantile)
        self.su_seeded_snr_abs_floor = float(su_seeded_snr_abs_floor)
        self.su_seeded_n_permutations = int(su_seeded_n_permutations)
        self.beam_width = beam_width
        # ``brute_force_max_features`` (iter56): raised default 22 -> 28. This is the prescreen
        # cap on ``phi.shape[1]``, NOT a direct guarantee that brute force runs at the dispatched
        # n. At default ``max_features=None`` the dispatcher's n_sub gate routes n<=26 to brute
        # force and n in {27, 28} to beam (beam consumes the wider 28-column prescreen pool);
        # iter56's measured wall-clock gain at C3 came from beam over the wider pool, NOT brute
        # at n=28. To actually run brute force at n=28 the caller must pin
        # ``max_features<=12`` (sum C(28,1..12)=76.7M < 80M gate). See the module-level comment on
        # ``_DEFAULT_BRUTE_FORCE_MAX_FEATURES`` for the full dispatcher truth table. RAM is
        # constant (~1MB). ``None`` consults ``pyutilz.performance.kernel_tuning.cache`` (key
        # ``mlframe.shap_proxied_fs.brute_force_max_features``) for per-HW override, falling back
        # to the module default. Explicit int pins always win.
        self.brute_force_max_features = int(brute_force_max_features) if brute_force_max_features is not None else _resolve_brute_force_max_features()
        # ``adaptive_prescreen_by_stability`` (iter59, OFF by default): when True, measure cross-fold
        # SHAP rank stability (median pairwise Spearman of per-fold mean |phi| feature rankings) and
        # NARROW (never widen) the post-OOF prescreen cap when stability drops. Surfaces the measured
        # stability + resolved cap under ``shap_proxy_report_['adaptive_prescreen']`` regardless of
        # whether the cap changes, so callers can inspect the diagnostic before opting in.
        #
        # Iter59 A/B at compact-C3 (width=4000, n_rows=4000, snr=8) MEASURED -3 recall + 4.2x e2e
        # wall when the lever was ON: post-cluster SHAP rankings have stability ~0.63 even at high
        # SNR (the unit-level SHAP fluctuates across folds more than raw-feature SHAP would), so the
        # mild-narrow bucket fires unintentionally, dropping the cap from 28 to 24 which crosses the
        # dispatcher's brute-force gate (2^24 = 16M < 80M default) and pays a heavy brute-force +
        # revalidation tail. Default flipped to False; the helper + report block stay shipped as
        # instrumentation so future iters can recalibrate thresholds against post-cluster scales or
        # gate the narrowing on dispatcher feasibility before re-enabling.
        self.adaptive_prescreen_by_stability = bool(adaptive_prescreen_by_stability)
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
        # Default kept at 100 after the iter95 sweep: at C3 (width=10000, n_rows=10000, n_inf=20,
        # n_red=20, snr=8.0, seed=0) values {100, 50, 25} produced prefilter walls 2.965s / 2.254s
        # / 1.430s and chosen-subset honest_loss 0.135456 / 0.148270 / 0.148710. cap=50 keeps the
        # chosen subset bit-identical but perturbs chosen-subset honest_loss by +9.46% (iter90
        # lesson: same-subset can hide a real loss regression downstream); cap=25 drops one true
        # informative column (recall 0.95 -> 0.90). Both fail the +-5% honest_loss gate, so 100
        # stays the default. Same-pattern caps that DID ship: iter94 trust_guard 100 -> 25,
        # iter19 oof_shap 300 -> 100.
        self.prefilter_n_estimators = prefilter_n_estimators
        # ``oof_shap_n_estimators`` (iter19) caps the per-fold booster size inside ``compute_shap_matrix``
        # so the OOF-SHAP stage trains 100-tree models instead of the 300-tree template default. The
        # proxy consumes the attribution RANKING and the coalition value; both are determined by the
        # fitted model's structural credit-allocation, which stabilises long before the last refinement
        # trees. iter4-baseline cProfile @ width=10000 showed OOF-SHAP dominated by xgboost ``update``
        # (29.6s tottime / 40.4s cum out of 43.3s SERIAL OOF-SHAP -- 96% of stage). Mirror of the
        # iter9/iter10 ``prefilter_n_estimators`` / ``trust_guard_n_estimators`` / ``refine_n_estimators``
        # pattern. Same clamp semantics: ``min(current_template_n_estimators, cap)`` so a custom
        # model whose ``n_estimators`` is already below the cap is left untouched. ``None`` disables
        # the cap (legacy 300-tree fit). Honest re-validation + trust-guard still use the FULL
        # template (uncapped) on the final chosen subset, so the user-facing OOF loss reported in
        # ``report['holdout_loss']`` is unaffected.
        # bench-attempt-rejected (iter90, 2026-06-01): tried lowering default 100 -> 50 at C3
        # (n_samples=10000, width=10000, n_inf=20, n_red=20, snr=8.0, seed=0). Recall tied at 17/20
        # AND oof_shap stage 7.84s -> 2.44s (3.2x faster). But chosen-subset honest brier loss
        # regressed 0.1348 -> 0.1498 (loss_ratio 0.581 -> 0.6518, +12% worse vs random baseline).
        # The OOF-SHAP ranking is recall-stable below 100 trees but the coalition value used in
        # the chosen-subset pick degrades, surfacing as a worse holdout brier. Keeping default=100.
        self.oof_shap_n_estimators = oof_shap_n_estimators
        # ``prefilter_stage1_keep`` overrides the two_stage prefilter's stage-A survivor count.
        # None -> the prefilter module computes ``min(2000, 0.2*n_features)`` (default funnel).
        # Other methods ignore this knob.
        self.prefilter_stage1_keep = prefilter_stage1_keep
        # ``prefilter_univariate_batch_size`` (iter37): column-batch width for the univariate /
        # two_stage stage-A ANOVA F-score. None auto-sizes from available RAM; explicit int forces
        # the batch (clamped to [1, n_features]). Sklearn's ``f_classif`` / ``f_regression``
        # densify the full (n_samples, n_features) design as float64 before per-class slicing
        # (~1.6 GB at width=20000 / n_rows=10000) and then carve K per-class halves on top,
        # OOMing the wide regime at the cheapest stage. The chunked replacement (parity to
        # float64 rounding) caps per-batch allocation at ``8 * n_samples * batch`` bytes
        # independent of feature count. Tradeoff: smaller batches reduce peak RSS at the cost
        # of per-batch Python loop overhead -- the auto-sizer targets a 256 MB chunk budget,
        # which empirically dominates Python overhead for n_features above ~256.
        self.prefilter_univariate_batch_size = prefilter_univariate_batch_size
        # ``shap_prefilter_*`` (iter31): SHAP-aware tightening of the effective ``prefilter_top``. The
        # downstream search only consumes top-``brute_force_max_features`` by mean |phi|; columns
        # between the search cap and the loose default (2000) pay full TreeSHAP cost in OOF-SHAP for
        # no contribution to the final pick. With ``shap_prefilter_enabled=True`` (default), the
        # selector passes ``min(prefilter_top, shap_prefilter_top)`` to ``prefilter_columns`` where
        # ``shap_prefilter_top = max(brute_force_max_features * safety_factor, min_features)``
        # (default ``max(22*4, 40) = 88``) so the EXISTING prefilter booster's ranking already
        # produces the tighter output -- no second booster fit is paid. ``safety_factor=4`` keeps a
        # 4x cushion over the search cap so OOF-SHAP variance can still surface signal the prefilter
        # booster's ranking missed; ``min_features=40`` is a floor for small ``brute_force_max_features``.
        # Bench-attempt-rejected variant (separate post-clustering booster fit, 2026-05-28): width=
        # 1000/n_rows=5000/seed=1 measured ~1.2s extra booster + ~1.3s OOF-SHAP savings = +0.1s wash.
        # The current realisation amortises into the prefilter booster the pipeline already pays.
        # Disable via ``shap_prefilter_enabled=False`` to restore the legacy default-2000 cap for
        # parity / regression checks. ``shap_prefilter_top`` overrides the derived cap.
        self.shap_prefilter_enabled = bool(shap_prefilter_enabled)
        self.shap_prefilter_top = shap_prefilter_top
        self.shap_prefilter_safety_factor = int(shap_prefilter_safety_factor)
        self.shap_prefilter_min_features = int(shap_prefilter_min_features)
        # ``shap_aware_stage1_keep`` (iter33): when the SHAP-aware prefilter cap shrinks
        # ``effective_prefilter_top`` far below the legacy stage-A default (``min(2000,
        # 0.2*n_features)``), the two_stage prefilter's stage-B booster fit is the dominant
        # wall-clock cost (cProfile at C2 width=10000/n_rows=5000 attributed 14.8s of a 30s fit to
        # xgboost ``update`` on a 2000-column matrix). The eventual stage-B output is
        # ``effective_prefilter_top`` (e.g. 112) anyway -- keeping stage A at 2000 forces the booster
        # to score 1900+ columns it will then discard. The lever tightens stage A to
        # ``max(shap_aware_stage1_floor, effective_prefilter_top * shap_aware_stage1_cushion)``
        # (default ``max(200, 112*2) = 224``) so the booster fits on ~9x fewer columns at the same
        # tree budget. ``shap_aware_stage1_cushion=2`` is the empirically-calibrated minimum that
        # preserves recall across C1/C2/C3 (iter76 sweep at width 5000/10000): cushion 8 -> 2 cut
        # prefilter wall 3.0-4.0x and e2e wall 1.42-1.58x with recall preserved or improved (+1 at
        # C3). The 2x headroom is what survives stage A's univariate F-rank for marginal-signal
        # informatives that the stage-B interaction-aware booster then recovers (4x cushion gave
        # the same recall at 1.42x e2e; 2x is the empirical optimum -- under-cushion would force
        # the floor to dominate). ``shap_aware_stage1_floor=200`` is a hard floor that protects
        # pathological tight ``brute_force_max_features`` configs (e.g. 5 * 2 = 10 would be too
        # aggressive a stage-A funnel). The lever is a strict tightening: ``min(default_stage1_keep,
        # ...)`` never widens beyond legacy. Gated off via ``shap_aware_stage1_keep=False`` for
        # parity / regression checks against iter32; ignored when the user pins
        # ``prefilter_stage1_keep`` explicitly (pinned value always wins) OR when
        # ``shap_prefilter_enabled=False`` OR when the resolved prefilter method is not
        # ``two_stage`` (only ``two_stage`` reads ``stage1_keep``).
        self.shap_aware_stage1_keep = bool(shap_aware_stage1_keep)
        self.shap_aware_stage1_cushion = int(shap_aware_stage1_cushion)
        self.shap_aware_stage1_floor = int(shap_aware_stage1_floor)
        self.cluster_features = cluster_features
        self.cluster_corr_threshold = cluster_corr_threshold
        self.cluster_weighting = cluster_weighting
        self.cluster_use_gpu = cluster_use_gpu
        self.cluster_auto_threshold = cluster_auto_threshold
        # iter67: when MRMR precomputed bins are supplied, the clustering stage
        # can switch from Pearson |corr| to pairwise Symmetric Uncertainty on
        # the binned columns. SU catches non-linear redundancy (XOR / saddle /
        # sinusoidal) that Pearson misses. ``cluster_use_precomputed_bins``
        # gates the switch (default True so callers passing precomputed get
        # the upgrade automatically). ``cluster_su_threshold`` is the SU
        # equivalent of ``cluster_corr_threshold`` -- different scale (SU is
        # bounded by 1 but only reaches it on deterministic relationships),
        # default 0.5 calibrated to a similar linking density as |corr| 0.7.
        self.cluster_use_precomputed_bins = bool(cluster_use_precomputed_bins)
        self.cluster_su_threshold = float(cluster_su_threshold)
        # iter75: SU is the UNCONDITIONAL DEFAULT clustering backend ("auto" picks it whenever
        # precomputed bins exist OR n_features <= ``cluster_su_auto_max_features``; bins are
        # computed on-the-fly from X via MRMR's ``categorize_dataset`` when missing). Pearson
        # stays accessible via ``cluster_backend="pearson"`` for the legacy regime + the wide
        # widths above the auto cap. ``"su"`` forces SU regardless of width (may bin X even at
        # width >> auto_max). The auto_max gate amortises the iter67-74 SU pairwise speedups
        # (33x cumulative at width=2000) against Pearson's vectorised correlation matrix; above
        # the cap Pearson still wins so auto falls back.
        # sklearn clone() compares ``get_params()`` output with the value re-set
        # by ``__init__`` BY IDENTITY (not equality); mutating ``cluster_backend``
        # via ``.lower()`` would create a fresh string object that fails the
        # identity check ("Cannot clone object: constructor modifies parameter").
        # Validate WITHOUT mutating, store the caller's value verbatim, and do
        # the lowercase normalisation at every use-site instead.
        if str(cluster_backend).lower() not in ("auto", "su", "pearson"):
            raise ValueError(f"cluster_backend must be one of 'auto', 'su', 'pearson'; got {cluster_backend!r}")
        self.cluster_backend = cluster_backend
        self.cluster_su_auto_max_features = (
            int(cluster_su_auto_max_features) if cluster_su_auto_max_features is not None else _resolve_cluster_su_auto_max_features()
        )
        self.cluster_su_n_bins = int(cluster_su_n_bins)
        self.prescreen_top = prescreen_top
        self.within_cluster_refine = within_cluster_refine
        # ``refine_n_estimators`` caps the per-trial booster size inside ``within_cluster_refine``.
        # Refine compares relative honest losses to decide whether a member-drop respects
        # ``parsimony_tol``; the ranking stabilises well before the default 300 trees, so capping at
        # ~100 trees cuts each fit ~3x while keeping the drop decision intact. None disables the cap.
        self.refine_n_estimators = refine_n_estimators
        # ``refine_ucb_*`` (iter35): batched-dispatch early-stop on within_cluster_refine's stage-2b
        # single-drop greedy round. Mechanism: each round ranks the k surviving members by ascending
        # stage-2a permutation importance (lowest importance = safest drop, most likely to produce the
        # lowest honest loss); trials dispatch in workers-sized batches; after each batch the running
        # best-loss-this-round is compared against every un-evaluated trial's UCB lower bound
        # ``importance + slack``. Dispatch stops when no remaining trial can beat the round leader.
        # The slack auto-calibrates from the round's (importance, honest_loss) pairs:
        # ``mean(delta) - stdev_multiplier * std(delta)``. The lever pays at width >= 10000 where each
        # honest fit is ~500 ms and ~5 stage-2b rounds dispatch ~10 trials each (Phase-0 C3 cProfile:
        # within_cluster_refine 6.14s of which ~5s is stage-2b parallel batches; reval-iter34 attribution
        # showed the joblib pool batches 2-3 deep at this regime, not 1-deep). Defaulted ON because the
        # importance ordering is a strong prior (stage 2a measured each member's holdout contribution
        # under the same model) and the gate only stops when even the most-optimistic remaining lower
        # bound exceeds the round leader. ``min_eval_size=None`` -> ``max(n_workers, 3)``; ``slack=None``
        # auto-calibrates per-round. With UCB disabled OR k <= min_eval_size OR ``n_jobs in (1, 0,
        # None)`` (test fixtures), falls through to the legacy single-batch-per-round path with zero
        # behaviour change. Mirror of the iter34 reval UCB knob design (same names, same defaults).
        self.refine_ucb_enabled = bool(refine_ucb_enabled)
        self.refine_ucb_min_eval_size = refine_ucb_min_eval_size
        self.refine_ucb_slack = refine_ucb_slack
        self.refine_ucb_stdev_multiplier = float(refine_ucb_stdev_multiplier)
        # ``revalidation_n_estimators`` (iter28) caps the per-candidate booster size inside
        # ``revalidate_top_n`` / ``active_learning_revalidate``. The honest re-validation stage's
        # parsimony-rule selection is a RELATIVE ranking decision (within ``parsimony_tol`` of the
        # best stable_score), which stabilises long before 300 trees -- same rationale as iter9
        # refine / iter19 oof_shap / iter10 trust_guard. Iter27 profile at width=1000/n_rows=5000/
        # snr=8 measured revalidation at 47.4% of total fit (the post-iter27 dominant stage). At
        # ``top_n=20`` x ``n_revalidation_models=3`` that's 60 honest retrains, each at 300 trees;
        # capping at 100 cuts per-fit cost ~2.6x (single-fit microbench: 0.349s at 300 vs 0.134s at
        # 100). Ranking stability microbench across 20 candidate subsets: Spearman(300, 100)=0.94,
        # identical argmin, top-5 overlap 4/5. After the parsimony rule picks the winner, ONE
        # full-template re-evaluation refreshes the user-visible ``report['revalidation']['ranked']``
        # entry's ``honest_loss`` so it stays apples-to-apples with the trust-guard / ablation
        # outputs (both use the full template); the capped value is preserved as
        # ``honest_loss_capped`` for diagnostics. The capped fits are cache-namespaced via
        # ``template_id=('reval_cap', cap)`` so they never collide with full-template entries from
        # elsewhere in the pipeline. ``None`` disables the cap (legacy 300-tree behaviour).
        self.revalidation_n_estimators = revalidation_n_estimators
        # ``revalidation_ucb_*`` (iter34): batched-dispatch early-stop on the candidate scoring loop.
        # Mechanism: sort top_n candidates by proxy_loss, evaluate first ``min_eval_size`` in parallel
        # to saturate workers, then dispatch follow-on batches one at a time. After each batch, the
        # running best stable_score is compared against every un-evaluated candidate's UCB lower
        # bound ``proxy_loss + slack``; dispatch stops when no remaining candidate can enter the
        # parsimony band. Pays at width >= 10000 where per-fit cost is ~300 ms and 60 honest fits
        # batch ~8 deep on 8 workers (Phase-0: wall 4.86s / per_fit 337 ms = 14.4x ratio). Defaulted
        # ON because un-evaluated tail candidates are by construction worse-proxy and have low odds of
        # winning; the UCB slack is auto-calibrated from the running batch's (proxy, honest) pairs so
        # the gate adapts per-fit to the proxy's local fidelity. ``min_eval_size=None`` picks
        # ``max(n_workers, 3)``; ``slack=None`` uses ``mean(delta) - stdev_multiplier * std(delta)``
        # where ``delta_i = honest_i - score_i`` over evaluated candidates (pessimistic on the
        # un-evaluated side -> fewer wrong stops). The gate consumes the facade's per-candidate
        # ``score`` (bias-corrector predicted honest loss when the corrector fit cleanly, raw
        # proxy_loss otherwise), passed through as ``candidate_score`` so the UCB lower bound lives
        # on the honest scale (raw proxy_loss spread is too tight to discriminate at width >= 10000
        # where every top_n=20 candidate is a near-duplicate union of the same SHAP-aware cluster
        # picks). ``stdev_multiplier`` (iter41): None (default) routes to a width-dependent auto -
        # 0.6 at ``n_features >= 10000``, 1.0 below. Explicit float overrides. Rationale: at C3 scale
        # (width 10000, n_rows 10000), iter38/iter40 prefilter-side tightening compressed the proxy
        # spread, widening the residual delta std and slackening the UCB gate (smaller `mean(delta) -
        # k*std(delta)` is more negative => lower UCB lower bounds => fewer stops). The two stages
        # are coupled via UCB slack calibration. Smaller k => less std subtraction => higher (less
        # pessimistic) lower bounds => gate fires sooner. The 0.6 floor was picked to cut revalidation
        # ~30% without crossing the parsimony threshold on near-tie candidates. Smaller-width regimes
        # retain k=1.0 because iter34 calibration is honest there and the proxy delta std is narrower.
        self.revalidation_ucb_enabled = bool(revalidation_ucb_enabled)
        self.revalidation_ucb_min_eval_size = revalidation_ucb_min_eval_size
        self.revalidation_ucb_slack = revalidation_ucb_slack
        self.revalidation_ucb_stdev_multiplier = float(revalidation_ucb_stdev_multiplier) if revalidation_ucb_stdev_multiplier is not None else None
        # ``revalidation_adaptive_n_models`` (iter77, default True): split the n_revalidation_models
        # stability seeds into separate rounds and early-stop once the parsimony-rule winner has been
        # identical across two consecutive rounds. Floor is 2 rounds (need >=1 stability check);
        # ceiling is n_revalidation_models. Conservation: when the loop runs the full ceiling the
        # accumulated per-candidate losses are identical to legacy (same seeds, same accumulation).
        # When winners differ every round, no early stop fires and total fit count matches legacy.
        # With n_revalidation_models=1 (calibration paths) the knob is a no-op. Surfaced via
        # ``report['revalidation']['random_baseline']['ucb']['n_models_run']``.
        self.revalidation_adaptive_n_models = bool(revalidation_adaptive_n_models)
        # ``revalidation_mmr_jaccard_threshold`` (iter50): MMR-style greedy de-duplication of the
        # corrector-sorted top_n candidates BEFORE the honest re-validation stage. At width>=20000
        # post-prefilter top_n=20 candidates are near-duplicate unions of the same SHAP-aware
        # stage-B survivors (Jaccard >0.7 overlap pairwise observed in iter48/iter49). UCB already
        # short-circuits the proxy-loss tail, but still pays per-batch dispatch on redundant subsets.
        # MMR processes candidates in their (corrector-sorted) order, keeping candidate i if
        # ``min_j Jaccard_distance(i, kept_j) > tau``. Conservation-preserving: dropped candidates
        # are corrector-equivalent (>= 1-tau feature overlap) to a retained one and would not pass
        # the parsimony band as a meaningful improvement. ``None`` (default) routes to a
        # width-dependent auto: 0.3 at ``n_features >= 20000`` (high redundancy regime), disabled
        # below (smaller-width top_n is less redundant; no measurable win to make and risk of
        # dropping a winner in distinct-tail candidates). Explicit float overrides for any width.
        self.revalidation_mmr_jaccard_threshold = float(revalidation_mmr_jaccard_threshold) if revalidation_mmr_jaccard_threshold is not None else None
        # ``trust_guard_n_estimators`` caps the per-anchor booster size inside ``proxy_trust_guard``.
        # The trust report only consumes RANKS of anchor losses (Spearman / Kendall / recall@k); a
        # capped booster gives a faithful fidelity signal at a small fraction of the full-template cost.
        # Default 25 (iter94 sweep at C3 width=10000 n_rows=10000 n_inf=20 n_red=20 snr=8.0 seed=0:
        # values {100, 50, 25} all yielded IDENTICAL chosen subsets, trustworthy=True, and composite
        # proxy_fidelity_score within 0.14% of the 100-tree baseline. trust_guard wall 4.106s -> 1.677s
        # (2.45x); e2e wall 69.91s -> 26.37s. Rank-only consumer is robust to coarser boosters because
        # tree ordering of anchor losses stabilises well before the residual-fit tail.) None disables
        # the cap.
        self.trust_guard_n_estimators = trust_guard_n_estimators
        # ``trust_guard_stratified_anchors``: opt-in (default OFF) softmax-by-F-score anchor sampler
        # for the trust guard. Activates only when the prefilter cached an F-score vector
        # (``prefilter_method`` in {two_stage, univariate}). ``trust_guard_uniform_tail_frac``
        # controls the fraction of each anchor that is uniform-sampled for tail-of-distribution
        # coverage (default 20%; 0 = pure-weighted, 1 = legacy uniform).
        #
        # Iter14 + iter16 originally rejected this lever because ``_softmax_weights`` was NOT
        # scale-invariant (raw F-scores in 10^2..10^4 collapsed softmax to ~one-hot). Iter97
        # (2026-06-01) fixed the softmax. Iter99 (2026-06-01) re-evaluated the lever as a default
        # flip vs uniform, running a 2x2 A/B at two production regimes PLUS the smaller W2K
        # biz_value regime:
        #
        #   W6K  (width=6000,  n=3000,  n_inf=12, snr=8.0): stratified sp=0.9902 fid=0.9941
        #                                                    uniform    sp=0.9813 fid=0.9888  +sp +fid
        #   W10K (width=10000, n=10000, n_inf=20, snr=8.0): stratified sp=0.9840 fid=0.9904
        #                                                    uniform    sp=0.9724 fid=0.9834  +sp +fid
        #   W2K  (width=2000,  n=2000,  n_inf=20+20red, snr=8.0): stratified sp=0.9684 fid=0.8811
        #                                                          uniform    sp=0.9805 fid=0.9883  REGRESS
        #
        # Stratified wins at the two noise-dominated wide regimes (W6K / W10K) but REGRESSES at the
        # dense-redundant narrow W2K regime (40 signal cols on 2000 with rho=0.85). The F-score
        # cohort prior concentrates anchors on the head of a tight redundant cluster where proxy
        # and honest losses agree trivially, compressing the Spearman spread. Default stays OFF
        # because the lever does NOT universally pay -- shipping True as the default would silently
        # regress fidelity for callers running on dense-redundant cohorts (small post-prefilter
        # widths, high inter-feature correlation).
        #
        # Iter100 (2026-06-01) tested the width-aware ``'auto'`` dispatcher hypothesis with a
        # 4x2x2 contour sweep (n_rows=5000, n_inf=20, snr=8.0, seed=0; widths {2k,4k,6k,10k} x
        # {sparse n_red=0, dense n_red=20 rho=0.85} x {stratified, uniform}). Per-cell deltas
        # (d_sp / d_fid = stratified - uniform):
        #
        #      width   cond     d_sp     d_fid    winner
        #       2000  dense   +0.0044  +0.0693    strat
        #       2000 sparse   +0.0031  -0.1315    uni
        #       4000  dense   +0.0116  +0.0069    strat
        #       4000 sparse   -0.0125  -0.1408    uni
        #       6000  dense   -0.0102  -0.1395    uni       <-- contradicts iter99 W6K-wins
        #       6000 sparse   +0.0022  +0.0013    tie
        #      10000  dense   +0.0040  -0.0643    uni
        #      10000 sparse   +0.0231  -0.0528    uni       <-- spearman wins but recall@k loses
        #
        # The contour is NOT separable by a single-axis threshold: width=4000 splits dense-cells
        # cleanly (strat below, uni above) but flips for sparse-cells; a 2-axis (width, redundancy)
        # rule would still need to canonicalise W6K-dense flipping uniform-wins (a regression vs
        # iter99's W6K result, traceable to n_rows=5000 here vs n_rows=3000 in iter99). Per the
        # iter100 calibration gate ("if the contour cannot be cleanly fit by a simple width
        # threshold alone, do not ship 'auto'"), the lever remains opt-in. Future work: a wider
        # sweep across n_rows + rho axes (~50 cells) may reveal a separable contour, but the
        # measured signal is too noisy at the current grid to ship an auto-router that wouldn't
        # silently regress callers on the contested cells. See
        # ``_benchmarks/bench_iter100_stratified_anchors_contour.py`` to reproduce.
        #
        # Iter101 (2026-06-01) extended the contour to 2D (width x n_rows) on dense-only
        # (rho=0.85, n_red=20, n_inf=20, snr=8.0, seed=0); widths {2k,4k,6k,10k} x
        # n_rows {2k,5k,10k} = 12 cells x {strat, uni}. d_fid (strat - uni) by cell:
        #
        #      width   n_rows    d_sp     d_fid    winner
        #       2000    2000   -0.0036  -0.0688    uni
        #       2000    5000   +0.0044  +0.0693    strat
        #       2000   10000   -0.0142  -0.0752    uni
        #       4000    2000   +0.0147  +0.0088    tie
        #       4000    5000   +0.0116  +0.0069    tie
        #       4000   10000   -0.0062  -0.0704    uni
        #       6000    2000   +0.0080  +0.0048    tie
        #       6000    5000   -0.0102  -0.1395    uni
        #       6000   10000   +0.0062  +0.0037    tie
        #      10000    2000   -0.0107  -0.0731    uni
        #      10000    5000   +0.0040  -0.0643    uni
        #      10000   10000   +0.0013  -0.0659    uni
        #
        # Stratified wins exactly 1/12 cells (w=2000, n=5000) on d_fid; 4 ties, 7 uniform-wins
        # (down to d_fid=-0.140 at w=6000/n=5000). No single-axis r=n_rows/width or
        # r=n_rows/sqrt(width) separator cleanly partitions the win-cells. iter99 W2K
        # (w=2000, n=2000, strat) re-measured at sp=0.9626 fid=0.8442 vs iter99's sp=0.9684
        # fid=0.8811 -- ~0.04 swing reflects real measurement variance from concurrent CPU
        # contention during the sweep. The iter99 W6K/W10K wins do NOT reproduce at n_rows=5000
        # in this sweep, suggesting those were specific to a (width, n_rows, seed) interaction
        # rather than a generalisable property of stratified sampling. Lever stays opt-in. See
        # ``_benchmarks/bench_iter101_stratified_anchors_2d.py`` to reproduce.
        self.trust_guard_stratified_anchors = bool(trust_guard_stratified_anchors)
        # ``trust_guard_uniform_tail_frac`` re-audited iter98 (2026-06-01) after iter97 made the
        # softmax scale-invariant. Question: does the calibrated 20% uniform tail still pay now that
        # stratified is well-behaved (no longer collapsing to ~one-hot on raw F-scores)? Sweep
        # {0.0, 0.1, 0.2, 0.3} at width=6000, n=3000, n_inf=12, snr=8.0, seed=0,
        # trust_guard_stratified_anchors=True:
        #
        #   tail_frac=0.0  spearman=0.9764  recall@k=0.833  fidelity=0.9192  recovery=11/12
        #   tail_frac=0.1  spearman=0.9803  recall@k=0.833  fidelity=0.9215  recovery=11/12
        #   tail_frac=0.2  spearman=0.9849  recall@k=1.000  fidelity=0.9909  recovery=11/12  <-- DEFAULT
        #   tail_frac=0.3  spearman=0.9778  recall@k=1.000  fidelity=0.9867  recovery=11/12
        #
        # All four produce IDENTICAL chosen subsets (jaccard=1.0) and equal recovery; 0.2 wins on
        # both spearman AND composite fidelity. Reducing the tail to 0.1 / 0.0 costs recall@k
        # (0.833 vs 1.000) because the pure-stratified draw never probes any column outside the
        # F-score head, so the proxy's top-k overlap with honest top-k drops one anchor.
        # iter97's scale-invariant softmax did NOT shift the optimum -- 20% uniform tail remains
        # the calibrated value.
        self.trust_guard_uniform_tail_frac = float(trust_guard_uniform_tail_frac)
        # ``trust_guard_cardinality_dist`` (iter15+iter16): how anchor cardinality ``k`` is drawn
        # over ``[min_card, max_card]`` inside ``proxy_trust_guard``. ``'zipf'`` (iter16 default after
        # composite-fidelity re-evaluation) over-samples small-k anchors via ``P(k) ∝ k^(-zipf_alpha)``;
        # ``'uniform'`` is the pre-iter15 flat draw. Iter15 originally shipped Zipf as opt-in because
        # raw Spearman regressed monotonically with alpha; iter16 introduced ``proxy_fidelity_score =
        # 0.5*spearman + 0.5*recall_at_k`` as the trust-guard's headline metric and re-ran the bench:
        # Zipf alpha=0.25 lifts recall@k from 0.667 to 0.833 enough to beat uniform on composite even
        # though raw Spearman dips. The width=6000 regime (n=3000, 12 informatives, 400-col cohort,
        # n_anchors=30) reproduced as:
        #
        #   uniform                spearman=0.8914 recall@k=0.667 composite=0.7791
        #   zipf alpha=0.25        spearman=0.8336 recall@k=0.833 composite=0.8335 (+0.0544)
        #   zipf alpha=0.5         spearman=0.7766 recall@k=0.500 composite=0.6383
        #   zipf alpha=1.0         spearman=0.4692 recall@k=0.500 composite=0.4846 (TRIPS GATE)
        #
        # Recovery 12/12 preserved across all variants. Zipf alpha=0.25 is the composite sweet spot
        # and is now the default. ``trust_guard_zipf_alpha`` defaults to 0.25 (was 1.0 in iter15);
        # alpha=0 degenerates Zipf to uniform; higher alpha overcompresses to small-k extremes where
        # proxy and honest agree trivially. Set ``trust_guard_cardinality_dist='uniform'`` to recover
        # pre-iter16 behaviour exactly.
        # Store the raw constructor value -- sklearn's ``clone()`` compares
        # post-init param identity (``param1 is not param2``) and a mid-init
        # ``.lower()`` rewrite returns a fresh string object on uncached inputs
        # which trips the "constructor modifies parameter X" check. Normalise
        # at call sites (or via a cached property) instead.
        self.trust_guard_cardinality_dist = trust_guard_cardinality_dist
        self.trust_guard_zipf_alpha = float(trust_guard_zipf_alpha)
        # ``trust_guard_fidelity_weights`` (iter17): weights ``(w_spearman, w_recall)`` for the
        # composite ``proxy_fidelity_score = w_spearman * spearman + w_recall * recall_at_k`` that
        # gates the trust-guard's ``trustworthy`` boolean. Iter17 default (0.6, 0.4) replaces iter16's
        # (0.5, 0.5) symmetric default after a 5-regime calibration study (additive high-SNR /
        # redundancy / interaction order-2 / xor / noise-heavy) measured corr(spearman, recovery_rate)
        # = 0.93 vs corr(recall@k, recovery_rate) = 0.55. Spearman tracks the proxy's whole-ranking
        # quality which actually predicts downstream selector recovery; recall@k is bounded above
        # (small anchor top-k overlap stays high even on half-broken proxies) so it lacks the dynamic
        # range to drive the gate. Corr-proportional split (0.63, 0.37) rounds to (0.6, 0.4). See
        # ``_benchmarks/calib_iter17_fidelity_weights.py``. The composite is still the honest headline
        # because iter15's Zipf alpha=0.25 lever moves spearman + recall in opposite directions; under
        # (0.6, 0.4) the Zipf-vs-uniform composite stays 0.833 vs 0.802 (Zipf still wins).
        # ``trust_guard_metric`` (iter16, default ``'proxy_fidelity_score'``): which scalar gates the
        # ``trustworthy`` boolean. ``'spearman'`` preserves pre-iter16 backwards-compat semantics for
        # callers that pinned ``spearman_floor`` against the raw Spearman scale.
        # Store raw constructor values; sklearn's ``clone()`` compares post-init
        # param identity (``param1 is not param2``) and any constructor-time
        # rewrite (tuple-of-floats coercion, ``.lower()`` etc.) returns fresh
        # objects that trip the "modifies parameter X" guard. Normalise at use
        # sites instead (see proxy_trust_guard call below).
        self.trust_guard_fidelity_weights = trust_guard_fidelity_weights
        self.trust_guard_metric = trust_guard_metric
        self.n_jobs = n_jobs
        # ``inner_n_jobs_cap`` (iter54, default False): controls the per-fit booster ``n_jobs`` inside
        # the OOF-SHAP / reval / refine / trust-guard parallel pools. The legacy iter4 cap (booster
        # n_jobs = max(1, n_cores // outer)) was added to prevent xgboost-vs-joblib oversubscription
        # on the era's xgboost (1.x). iter53 A/B at width 4000+10000 measured the cap as 8-9% e2e
        # SLOWER on 8-core modern boxes -- xgboost's own thread pool handles outer*inner > n_cores
        # more efficiently than the joblib-side cap (reval +8%, refine +11%, trust +12% wall-clock
        # loss with cap on; prefilter +2% small win). The selector now defaults inner=-1 so xgboost
        # decides; the chosen subset and honest losses are bit-identical between the two paths
        # (verified in iter53 A/B). Set ``inner_n_jobs_cap=True`` to restore the legacy cap on HW
        # where measurement says it helps (older xgboost, NUMA boxes with hwloc surprises).
        self.inner_n_jobs_cap = bool(inner_n_jobs_cap)
        self.random_state = random_state
        self.verbose = verbose
        # Control/safety knobs (parity with MRMR / RFECV): wall-clock budget + filesystem
        # stop-flag, honoured at the top of the elimination loop in fit(). max_runtime_mins=None
        # disables the budget; touch stop_file to abort cleanly with the current best subset.
        self.max_runtime_mins = max_runtime_mins
        self.stop_file = stop_file
        self.tqdm = tqdm
        # iter66: precomputed cross-selector artifacts; validated + aligned to
        # X.columns at fit() time via ``align_precomputed_to_X``. Stored as-is
        # so sklearn ``clone(estimator)`` round-trips the value.
        self.precomputed = precomputed
        # iter78: store raw constructor values so sklearn ``clone()`` round-trips byte-identically;
        # validation + auto-resolution happens at fit time via ``_resolve_booster_kind`` so a stale
        # ``cat_features`` list isn't repeatedly re-validated on every ``set_params`` call.
        self.booster_kind = booster_kind
        self.cat_features = cat_features
        self.cache_dir = cache_dir
        self._rng = np.random.default_rng(random_state)
        # Column-batch width for the disjoint train/holdout row split. The split copies
        # `X.values[idx, :]` block-by-block to bound peak transient memory at one batch's worth
        # of float64 cells (default = 1024 cols x n_rows x 8 bytes ~= 80 MiB at n_rows=10k);
        # private (not in `__init__` kwargs) because it's an internal memory-shaping knob,
        # not a quality lever. Override via ``self._split_col_batch = ...`` post-construction
        # if needed; sklearn `get_params()` ignores underscore-prefixed attrs.
        self._split_col_batch = 1024
        self._deferred_holdout = None


__all__ = [
    "ShapProxiedFS",
    "restrict_artifacts",
    "_resolve_brute_force_max_features",
    "_resolve_brute_force_n_sub_gate",
    "_resolve_cluster_su_auto_max_features",
    "_resolve_adaptive_prescreen_thresholds",
    "_resolve_adaptive_prescreen_width",
    "_resolve_adaptive_n_anchors",
    "_resolve_knee_prescreen_cap",
    "_DEFAULT_BRUTE_FORCE_MAX_FEATURES",
    "_DEFAULT_BRUTE_FORCE_N_SUB_GATE",
    "_DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES",
    "_DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS",
    "_ADAPTIVE_PRESCREEN_FLOOR",
    "_EXACT_OPTIMIZERS",
    "_HEURISTIC_OPTIMIZERS",
]
