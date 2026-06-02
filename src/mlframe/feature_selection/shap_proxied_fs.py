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

# Brute-force dispatcher gates (iter56 + iter57 audit). Both are overridable per HW via
# ``pyutilz.system.kernel_tuning_cache`` so a wider/narrower box can shift the boundary without
# touching code.
#
# Two distinct knobs, two distinct effects:
#
# 1. ``brute_force_max_features`` (default 28): cap on ``phi.shape[1]`` AFTER prescreen. The
#    prescreen step (shap_proxied_fs.py near line 866) narrows the candidate pool to this many
#    columns whenever the user runs ``optimizer="auto"|"bruteforce"|"bruteforce_gpu"``. The pool
#    feeds whichever optimizer the dispatcher ends up picking - brute_force when feasible, beam
#    otherwise - so this knob widens or narrows the CANDIDATE SPACE all optimizers see, not just
#    brute force.
#
# 2. ``brute_force_n_sub_gate`` (default 80M): cap on the EXHAUSTIVE subset count
#    ``total_subsets(n, min_card, max_card)`` that brute force would enumerate. The dispatcher
#    uses this to decide whether brute_force is feasible AT the post-prescreen n. When it isn't,
#    the dispatcher falls back to beam.
#
# Default-config behaviour at ``max_features=None`` (the default):
#   total_subsets(n, 1, None) = 2^n - 1 (kernel treats None as n_features).
#   - n in {1..26}: 2^n - 1 <= 67M, under the 80M gate -> brute force dispatches.
#   - n in {27, 28}: 134M, 268M, over the 80M gate -> beam dispatches.
#
# So at ``max_features=None`` the EFFECTIVE brute-force ceiling is n=26, NOT n=28. The cap of 28
# only unlocks the brute-force path when the user ALSO pins ``max_features<=12`` (sum C(28, 1..12)
# = 76.7M, under the 80M gate). At n=27,28 with default ``max_features=None`` the cap acts as a
# prescreen-pool widener that beam consumes - iter56's measured recall/wall gain came from beam at
# the wider pool, NOT from brute force. The cap is named after the brute-force kernel because
# that is the optimizer the dispatcher PREFERS at small n; at n=27,28 with default max_features
# the dispatcher correctly falls back to beam over the 28-column prescreen pool.
#
# Sizing rationale for the gate: 80M at the iter30 parallel kernel (~5M subsets/s on the 8-core
# dev box) caps a single brute-force search at ~16s wall. The next power-of-2 (n=28 with
# max_features=None: 268M subsets, ~54s wall) is beyond what the dispatcher should pick without
# the user explicitly opting in via ``optimizer="bruteforce"``.
#
# Iter58 beam-width sweep (``_benchmarks/bench_iter58_beam_width_sweep.py``) measured caps
# {22, 28, 32, 40} at C3 (snr=8) and C3_hard (snr=2). cap28 recall-dominates or ties all wider
# caps in both regimes; cap32 LOST a recall hit at C3_hard (15/20 vs cap28's 16/20), and cap40
# was slower without improving recall. Hypothesis: widening the prescreen pool past 28 lets
# noise features into beam's input, occasionally pushing the chosen subset off the truly
# informative one when the SHAP ranking is noisy. Default stays at 28.
_DEFAULT_BRUTE_FORCE_MAX_FEATURES = 28
_DEFAULT_BRUTE_FORCE_N_SUB_GATE = 80_000_000

# iter75: when auto-mode picks the SU clustering backend without precomputed bins, ShapProxiedFS
# bins X on-the-fly via MRMR's ``categorize_dataset`` and then runs pairwise SU. The pairwise scan
# is O(f^2) and despite the iter67-74 speedups (33x cumulative at width=2000) it is still slower
# than Pearson's vectorised correlation matrix at very wide search widths; above the cap the
# auto-mode falls back to Pearson rather than pay the SU cost. 2000 is the calibration point at
# which the iter69 column-major + iter71 fused-setup + iter73 popcount kernels keep SU within ~3x
# of Pearson on the dev box; per-HW tunable via kernel_tuning_cache key
# ``mlframe.shap_proxied_fs.cluster_su_auto_max_features``.
_DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES = 2000

# Iter59 adaptive-prescreen-width thresholds. The lever is a recall-protection device for low-SNR
# regimes: at low SHAP rank-stability across OOF folds, the top features past the strongly-informative
# core are essentially noise -- pulling them into the prescreen pool injects noise into beam's input
# and can perturb the chosen subset off the truly informative one. So we NARROW (never widen) the
# prescreen cap when stability drops. High-stability regimes keep the existing cap untouched.
#
# Thresholds (stability = median pairwise Spearman of per-fold mean |phi| feature rankings):
#   stability >= 0.8  -> use default cap (current behaviour, no regression risk)
#   0.6 <= stability < 0.8 -> cap = max(20, default - 4)  (mild narrow)
#   stability < 0.6   -> cap = max(16, default - 8)       (aggressive narrow)
#
# Overridable per-HW via kernel_tuning_cache key ``mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds``
# which accepts a list of ``[stability_threshold, cap_delta]`` pairs sorted descending by threshold.
_DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS = (
    (0.8, 0),    # stability >= 0.8: no narrowing, keep default cap
    (0.6, -4),   # 0.6 <= stability < 0.8: narrow by 4
    (-1.0, -8),  # stability < 0.6: narrow by 8 (catches negative correlations too)
)
_ADAPTIVE_PRESCREEN_FLOOR = 16  # never narrow below this regardless of stability


def _resolve_brute_force_max_features(default: int = _DEFAULT_BRUTE_FORCE_MAX_FEATURES) -> int:
    """Per-HW brute-force cap from ``pyutilz.system.kernel_tuning_cache`` (key
    ``mlframe.shap_proxied_fs.brute_force_max_features``), falling back to the module default."""
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.brute_force_max_features", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_brute_force_n_sub_gate(default: int = _DEFAULT_BRUTE_FORCE_N_SUB_GATE) -> int:
    """Per-HW feasibility cap on enumerated subset count (key
    ``mlframe.shap_proxied_fs.brute_force_n_sub_gate``). Above this the dispatcher falls through
    to ``beam`` regardless of ``brute_force_max_features``."""
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.brute_force_n_sub_gate", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_cluster_su_auto_max_features(
    default: int = _DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES,
) -> int:
    """Per-HW upper width at which ``cluster_backend='auto'`` still picks SU when no precomputed
    bins are supplied. Above this the auto path falls back to Pearson (the pairwise SU O(f^2)
    scan no longer amortises the iter73-era 33x-over-naive cost into a wall-clock win vs the
    vectorised Pearson |corr|). Reads kernel_tuning_cache key
    ``mlframe.shap_proxied_fs.cluster_su_auto_max_features``."""
    try:
        from pyutilz.system import kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su_auto_max_features", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_adaptive_prescreen_thresholds():
    """Return the (stability, delta) threshold list, ordered descending by stability.

    Reads ``mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds`` from kernel_tuning_cache
    when present (expected as an iterable of (stability, delta) pairs). Falls back to the module
    default. Always coerced to a tuple of (float, int) pairs sorted by descending stability.
    """
    raw = _DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS
    try:
        from pyutilz.system import kernel_tuning_cache

        cached = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds", default=None)
        if cached:
            raw = cached
    except Exception:
        pass
    pairs = [(float(s), int(d)) for s, d in raw]
    pairs.sort(key=lambda p: -p[0])
    return tuple(pairs)


def _resolve_adaptive_prescreen_width(stability: float, default_cap: int,
                                      floor: int = _ADAPTIVE_PRESCREEN_FLOOR) -> int:
    """Resolve the prescreen pool width from the measured cross-fold SHAP rank stability.

    Returns ``max(floor, default_cap + delta)`` where ``delta`` is read from the first threshold
    matching ``stability >= threshold`` in the descending-sorted table. The default table never adds
    a positive delta, so this lever can only NARROW the pool, never widen it past ``default_cap``.
    Conservative by design: high-stability regimes (the existing working configurations) keep the
    current cap untouched, and only the low-SHAP-rank-stability case (where the rank tail past the
    strongly-informative core is noise) sees a narrower pool that excludes that noise tail.
    """
    table = _resolve_adaptive_prescreen_thresholds()
    delta = 0
    for thr, d in table:
        if stability >= thr:
            delta = d
            break
    return max(int(floor), int(default_cap) + int(delta))


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
        # parsimony_tol: within_cluster_refine drops a member while the honest-loss increase stays below this.
        # MEASURED over-pruning dial (s2/S4 bench, _benchmarks/fs_hybrid): 0.02 EXCEEDS the marginal brier
        # contribution of 6/7 informative features (drop-one delta median ~0.013 < 0.02), so the refine drops
        # genuine signal that sits above noise but below the loose tol. Lower it (e.g. 0.005) - or set
        # within_cluster_refine=False - when a strong downstream benefits from more features. Kept at 0.02 by
        # default for back-compat; the fs_hybrid benchmark uses within_cluster_refine=False as the over-pruning fix.
        parsimony_tol: float = 0.02,
        min_selected_ratio: float = 0.0,
        trust_guard: bool = True,
        n_anchors: int = 30,
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
        beam_width: int = 8,
        brute_force_max_features: int | None = None,
        adaptive_prescreen_by_stability: bool = False,
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
        # measured with ShapProxiedFS's OWN booster on a 25% holdout. When the DOWNSTREAM model is stronger / different
        # (e.g. a 300-tree LightGBM) it may exploit features this proxy finds within-tol-redundant, so the refine can
        # over-prune real signal -> downstream loss. If the downstream benefits from more features, disable this
        # (within_cluster_refine=False) or tighten ``parsimony_tol``. Measured on a 52-feature synthetic: default
        # over-pruned to 6 feats (downstream LGBM AUC 0.73); within_cluster_refine=False kept 18 (AUC 0.77).
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
        # Keep n_anchors=30: the chosen-subset parity says the gate would still pass at lower anchors,
        # but the trust SCORE is what downstream consumers (bias corrector, fidelity reporting) read,
        # and degrading its margin to the floor for a sub-second wall cut is the wrong trade.
        self.n_anchors = n_anchors
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
        self.beam_width = beam_width
        # ``brute_force_max_features`` (iter56): raised default 22 -> 28. This is the prescreen
        # cap on ``phi.shape[1]``, NOT a direct guarantee that brute force runs at the dispatched
        # n. At default ``max_features=None`` the dispatcher's n_sub gate routes n<=26 to brute
        # force and n in {27, 28} to beam (beam consumes the wider 28-column prescreen pool);
        # iter56's measured wall-clock gain at C3 came from beam over the wider pool, NOT brute
        # at n=28. To actually run brute force at n=28 the caller must pin
        # ``max_features<=12`` (sum C(28,1..12)=76.7M < 80M gate). See the module-level comment on
        # ``_DEFAULT_BRUTE_FORCE_MAX_FEATURES`` for the full dispatcher truth table. RAM is
        # constant (~1MB). ``None`` consults ``pyutilz.system.kernel_tuning_cache`` (key
        # ``mlframe.shap_proxied_fs.brute_force_max_features``) for per-HW override, falling back
        # to the module default. Explicit int pins always win.
        self.brute_force_max_features = (
            int(brute_force_max_features) if brute_force_max_features is not None
            else _resolve_brute_force_max_features()
        )
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
            raise ValueError(
                f"cluster_backend must be one of 'auto', 'su', 'pearson'; got {cluster_backend!r}"
            )
        self.cluster_backend = cluster_backend
        self.cluster_su_auto_max_features = (
            int(cluster_su_auto_max_features) if cluster_su_auto_max_features is not None
            else _resolve_cluster_su_auto_max_features()
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
        self.revalidation_ucb_stdev_multiplier = (
            float(revalidation_ucb_stdev_multiplier)
            if revalidation_ucb_stdev_multiplier is not None else None)
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
        self.revalidation_mmr_jaccard_threshold = (
            float(revalidation_mmr_jaccard_threshold)
            if revalidation_mmr_jaccard_threshold is not None else None)
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

    def _resolve_booster_kind(self) -> str:
        """Pick the default booster family. ``None`` (default) auto-detects by availability so
        environments with only catboost installed still work without the user pinning the kwarg;
        an explicit user value always wins. Raises ``ValueError`` on an unknown kind so the typo
        surfaces at fit time instead of silently falling back."""
        from mlframe.feature_selection._shap_proxy_explain import _VALID_BOOSTER_KINDS

        if self.booster_kind is not None:
            kind = str(self.booster_kind).lower()
            if kind not in _VALID_BOOSTER_KINDS:
                raise ValueError(
                    f"ShapProxiedFS: unknown booster_kind={self.booster_kind!r}; "
                    f"expected one of {_VALID_BOOSTER_KINDS}."
                )
            return kind
        # Auto-detect: prefer xgboost (the historical default; SHAP fast paths are most mature
        # there), fall back to catboost when only catboost is installed.
        try:
            import xgboost  # noqa: F401
            return "xgboost"
        except ImportError:
            from mlframe.feature_selection._shap_proxy_catboost import catboost_available

            if catboost_available():
                return "catboost"
            raise ImportError(
                "ShapProxiedFS: neither xgboost nor catboost is installed; install one or pass an "
                "explicit ``model=`` template."
            )

    def _resolve_revalidation_ucb_stdev_multiplier(self, n_features: int) -> float:
        """Width-dependent default for the revalidation UCB stdev multiplier.

        Explicit user value (set on the instance) always wins. ``None`` (the auto sentinel) routes
        to ``0.6`` at ``n_features >= 10000`` and ``1.0`` below. Tightening at wide regimes cuts
        revalidation wall-clock by firing the early-stop sooner; smaller-regime calibration
        (iter34 default 1.0) stays untouched because the proxy residual spread is narrower there.
        """
        # bench-attempt-rejected (iter48, 2026-05-29): tried a third step at width>=20000 -> k=0.4
        # on top of the iter41 0.6 step. C4 (width=20000) measured reval 8.96s -> 8.87s (1% on
        # ~9s, well under noise floor). The iter41 0.6 step already saturates the gate at C4:
        # post-prefilter the top_n=20 candidates' proxy_loss values cluster tightly (near-duplicate
        # SHAP-aware picks from the same 88-feature stage-B survivors), so the residual delta std
        # is already small and dropping k further does not push (proxy + slack) above best_so_far
        # for the un-evaluated tail. The reval wall remaining at ~9s is the floor: ucb_min_eval_size
        # default ``max(n_workers, 3)`` first batch (~8 candidates) + 1-2 post-batch dispatches at
        # the iter41 setting. Any further reval cut needs a different mechanism (smaller min_eval_size,
        # or a corrector-aware sort that breaks the proxy_loss tie), not k tightening.
        if self.revalidation_ucb_stdev_multiplier is not None:
            return float(self.revalidation_ucb_stdev_multiplier)
        return 0.6 if int(n_features) >= 10000 else 1.0

    def _resolve_revalidation_mmr_jaccard_threshold(self, n_features: int) -> float | None:
        """Width-dependent default for the iter50 MMR Jaccard de-duplication threshold.

        Explicit user value (set on the instance) always wins, including ``0.0`` (no dedup) and
        any value in ``(0, 1]``. ``None`` (auto sentinel) routes to ``0.3`` at
        ``n_features >= 20000`` (where iter49 measured >0.7 pairwise overlap among the top_n=20
        corrector-sorted candidates after the SHAP-aware cluster picks) and ``None`` (disabled)
        below, since smaller-width top_n is less redundant and the floor risk (dropping a winner
        in genuinely distinct candidates) outweighs the wall-clock gain.
        """
        if self.revalidation_mmr_jaccard_threshold is not None:
            return float(self.revalidation_mmr_jaccard_threshold)
        return 0.3 if int(n_features) >= 20000 else None

    @staticmethod
    def _mmr_filter_by_jaccard(candidates, tau: float) -> list[int]:
        """Greedy MMR dedup over an already-(corrector-)sorted candidate list.

        Returns the list of kept candidate indices (preserving the input order). A candidate is
        kept if its Jaccard distance to every previously-kept candidate exceeds ``tau``; otherwise
        it is dropped as a near-duplicate. Jaccard distance is ``1 - |A intersect B| / |A union B|``;
        ``tau=0`` keeps everything (only exact duplicates would drop), ``tau=1`` keeps only the
        first candidate. Defensive: if ``tau`` is misconfigured such that no second candidate
        clears the gate, the first candidate (corrector-sorted winner) is always kept; the caller
        will never receive an empty list. Candidates' feature indices (``c[1]``) are interpreted
        as sets of unit/feature ids.
        """
        kept: list[int] = []
        kept_sets: list[set] = []
        for i, cand in enumerate(candidates):
            s = set(int(x) for x in cand[1])
            if not kept_sets:
                kept.append(i)
                kept_sets.append(s)
                continue
            min_dist = 1.0
            for ks in kept_sets:
                union = len(s | ks)
                if union == 0:
                    dist = 0.0
                else:
                    dist = 1.0 - (len(s & ks) / union)
                if dist < min_dist:
                    min_dist = dist
                    if min_dist <= tau:
                        break
            if min_dist > tau:
                kept.append(i)
                kept_sets.append(s)
        # Defensive: keep at least the top-1 (sorted-by-corrector winner) if the gate dropped all.
        if not kept and candidates:
            kept.append(0)
        return kept

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
        """Pick the optimizer for the post-prescreen candidate pool.

        At ``optimizer="auto"``, brute force is the preferred path when (a) the post-prescreen
        ``n_features`` is at or below the user's ``brute_force_max_features`` ceiling AND (b) the
        exhaustive subset count fits the dispatcher's ``brute_force_n_sub_gate`` feasibility cap.
        Otherwise the dispatcher falls back to beam over the SAME candidate pool. At default
        ``max_features=None`` the n_sub gate effectively caps brute-force dispatch at n<=26
        (2^26 = 67M < 80M default gate); n in {27, 28} runs beam over the wider prescreen pool.
        """
        opt = self.optimizer
        if opt != "auto":
            return opt
        from mlframe.feature_selection._shap_proxy_search import total_subsets

        if n_features <= self.brute_force_max_features:
            n_sub = total_subsets(n_features, self.min_features, self.max_features)
            if n_sub <= _resolve_brute_force_n_sub_gate():
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
        idx_search, idx_hold = train_test_split(
            idx_all, test_size=self.holdout_size, random_state=int(self.random_state),
            shuffle=True, stratify=stratify)
        # Wide-frame split with deferred holdout materialisation. At C4 (width=20000, n_rows=10000)
        # the original frame is 1.49 GiB, the search slice (75% rows) is 1.12 GiB, and the holdout
        # slice (25% rows) is 381 MiB; the legacy back-to-back
        # `X.iloc[idx_search].reset_index(drop=True)` + `X.iloc[idx_hold].reset_index(drop=True)`
        # held all three simultaneously plus reset_index transient buffers and OOM'd on a
        # 17 GB / 6.4 GB-free host (iter46). The prefilter only needs `X_search`; once it returns
        # `working_cols` (typically <=704 entries -- effective_prefilter_top is bounded by the
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
        from mlframe.feature_selection._shap_proxy_precomputed import (
            align_precomputed_to_X, su_to_prefilter_keep,
        )
        _precomputed_aligned, _precomputed_report = align_precomputed_to_X(
            self.precomputed, X_search,
        )
        report["precomputed_used"] = _precomputed_report
        report["precomputed_bins_available"] = bool(
            isinstance(_precomputed_aligned, dict) and _precomputed_aligned.get("bins")
        )

        # Cheap native-importance pre-filter BEFORE the expensive OOF-SHAP. SHAP cost scales with the
        # column count, and clustering only compresses CORRELATED features (independent noise stays as
        # singletons), so on wide data SHAP would otherwise run on ~all columns. Rank all features and
        # keep the top-K; ``working_cols`` maps the surviving working columns back to original indices
        # for the final selector output. ``prefilter_method`` trades speed against interaction-awareness
        # (model / univariate / fast_model / gpu_model / two_stage); "auto" stays quality-safe for
        # moderate widths and routes very wide data (n_features >= 8000) to the cheap-funnel +
        # capped-booster two_stage path -- see ``_shap_proxy_prefilter``.
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
            from mlframe.feature_selection._shap_proxy_shap_prefilter import (
                resolve_shap_prefilter_top)

            sp_top = (self.shap_prefilter_top if self.shap_prefilter_top is not None
                      else resolve_shap_prefilter_top(
                          brute_force_max_features=self.brute_force_max_features,
                          safety_factor=self.shap_prefilter_safety_factor,
                          min_features=self.shap_prefilter_min_features))
            # Tighten only -- never expand the user's prefilter budget.
            effective_prefilter_top = min(int(self.prefilter_top), int(sp_top))
            report["shap_prefilter"] = dict(
                requested_top=int(sp_top),
                effective_prefilter_top=int(effective_prefilter_top),
                user_prefilter_top=int(self.prefilter_top))
        working_cols = np.arange(n_features)
        if effective_prefilter_top is not None and n_features > effective_prefilter_top:
            from mlframe.feature_selection._shap_proxy_prefilter import (
                _default_stage1_keep, prefilter_columns, resolve_prefilter_method)

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
            if (effective_stage1_keep is None and self.shap_aware_stage1_keep
                    and self.shap_prefilter_enabled):
                _resolved = resolve_prefilter_method(
                    self.prefilter_method, n_features=n_features,
                    n_rows=int(X_search.shape[0]))
                if _resolved == "two_stage":
                    from mlframe.feature_selection._shap_proxy_shap_prefilter import (
                        resolve_shap_aware_stage1_keep)

                    effective_stage1_keep = resolve_shap_aware_stage1_keep(
                        effective_prefilter_top=int(effective_prefilter_top),
                        stage1_cushion=self.shap_aware_stage1_cushion,
                        stage1_floor=self.shap_aware_stage1_floor,
                        default_stage1_keep=_default_stage1_keep(n_features))
                    if "shap_prefilter" in report and isinstance(report["shap_prefilter"], dict):
                        report["shap_prefilter"]["stage1_keep_tightened"] = int(effective_stage1_keep)
                        report["shap_prefilter"]["stage1_keep_default"] = int(
                            _default_stage1_keep(n_features))

            with _stage("prefilter"):
                if (_precomputed_aligned is not None
                        and "su_to_target" in _precomputed_aligned):
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
                if len(working_cols) < n_features:
                    X_search = X_search.iloc[:, working_cols]
                report["prefilter"] = pf_info
        # Materialise X_hold at the narrow post-prefilter column count (or the full width if no
        # prefilter ran). Deferred from the row-split above so the C4 peak holds X + X_search and
        # NOT also a wide X_hold; with prefilter on, working_cols is typically <=704 entries so
        # this slice is ~5 MiB instead of 381 MiB.
        X_vals_full, idx_hold_saved, X_cols_full = self._deferred_holdout
        if len(working_cols) < n_features:
            hold_cols = [X_cols_full[c] for c in working_cols]
            X_hold = pd.DataFrame(
                X_vals_full[np.ix_(idx_hold_saved, working_cols)],
                columns=hold_cols, copy=False)
        else:
            X_hold = pd.DataFrame(X_vals_full[idx_hold_saved], columns=X_cols_full, copy=False)
        self._deferred_holdout = None
        del X_vals_full, X_cols_full

        # Optional correlated-feature clustering: collapse to denoised UNITS so SHAP + search run on
        # hundreds of columns, not tens of thousands. unit_to_members maps proxy(unit) index ->
        # original feature columns; None means proxy index == feature column (identity).
        do_cluster = self.cluster_features is True or (
            self.cluster_features == "auto" and n_features > self.cluster_auto_threshold)
        if do_cluster:
            from mlframe.feature_selection._shap_proxy_cluster import (
                build_unit_matrix, cluster_correlated_features, cluster_summary)

            with _stage("clustering"):
                # iter75: SU is now the UNCONDITIONAL DEFAULT clustering backend whenever auto-mode
                # picks it (precomputed bins available OR width <= ``cluster_su_auto_max_features``).
                # Pearson is opt-in via ``cluster_backend="pearson"`` and the wide-width auto fallback.
                # Bins are reused from MRMR.export_artifacts when present, else computed on the fly via
                # ``categorize_dataset`` on ``X_search`` so every default user gets non-linear
                # redundancy detection (XOR / saddle / sinusoidal) that Pearson |corr| misses.
                _bins = (
                    _precomputed_aligned.get("bins")
                    if isinstance(_precomputed_aligned, dict) else None
                )
                _have_precomputed_bins = (
                    self.cluster_use_precomputed_bins
                    and isinstance(_bins, dict)
                    and len(_bins) > 0
                )
                # ``cluster_use_precomputed_bins=False`` is a per-instance opt-out from the
                # SU-with-precomputed-bins fast path; we honour it as a forced Pearson dispatch
                # (legacy semantics) regardless of cluster_backend, so existing callers keep their
                # opt-out switch.
                _backend_norm = str(self.cluster_backend).lower()
                if not bool(self.cluster_use_precomputed_bins):
                    effective_backend = "pearson"
                elif _backend_norm == "pearson":
                    effective_backend = "pearson"
                elif _backend_norm == "su":
                    effective_backend = "su"
                else:
                    # auto: prefer SU whenever bins exist (cheap reuse) or width is below the
                    # on-the-fly-binning cost gate; above the cap Pearson's vectorised correlation
                    # matrix still beats pairwise SU on wall-clock.
                    n_search_cols = int(X_search.shape[1])
                    if _have_precomputed_bins or n_search_cols <= int(self.cluster_su_auto_max_features):
                        effective_backend = "su"
                    else:
                        effective_backend = "pearson"

                _cluster_bins_source = "n/a"
                if effective_backend == "su":
                    from mlframe.feature_selection._shap_proxy_cluster_su import (
                        cluster_correlated_features_su,
                    )

                    # Reuse precomputed bins when keyed to every working column; else bin
                    # X_search on the fly via MRMR's discretiser. Keeping the precomputed path
                    # avoids the second binning pass when MRMR already produced an aligned view.
                    _cluster_names = (
                        [c for c in X_search.columns.tolist() if c in _bins]
                        if _have_precomputed_bins else []
                    )
                    if _have_precomputed_bins and len(_cluster_names) == X_search.shape[1]:
                        _nbins_pf = (
                            _precomputed_aligned.get("nbins_per_feature")
                            if isinstance(_precomputed_aligned, dict) else None
                        )
                        _bins_for_su = _bins
                        _cluster_bins_source = "precomputed"
                    else:
                        # On-the-fly binning: categorize_dataset returns (data_array, column_names,
                        # nbins_per_col). The dict layout matches what MRMR.export_artifacts produces
                        # so cluster_correlated_features_su consumes it unchanged.
                        from mlframe.feature_selection.filters.discretization import (
                            categorize_dataset,
                        )
                        _bin_data, _bin_cols, _bin_nbins = categorize_dataset(
                            X_search, n_bins=int(self.cluster_su_n_bins),
                        )
                        _bins_for_su = {
                            name: _bin_data[:, i] for i, name in enumerate(_bin_cols)
                        }
                        _nbins_pf = {
                            name: int(_bin_nbins[i]) for i, name in enumerate(_bin_cols)
                        }
                        _cluster_names = list(X_search.columns)
                        _cluster_bins_source = "on_the_fly"

                    if len(_cluster_names) == X_search.shape[1]:
                        labels = cluster_correlated_features_su(
                            _bins_for_su,
                            threshold=self.cluster_su_threshold,
                            feature_names=_cluster_names,
                            nbins_per_feature=_nbins_pf,
                        )
                        _cluster_backend = "su"
                        _cluster_threshold = float(self.cluster_su_threshold)
                    else:
                        # Defensive: categorize_dataset may drop unsupported dtypes (rare in numeric
                        # X_search); fall back to Pearson rather than partial coverage.
                        labels = cluster_correlated_features(
                            X_search.values, threshold=self.cluster_corr_threshold,
                            use_gpu=self.cluster_use_gpu)
                        _cluster_backend = "pearson_fallback_bins_incomplete"
                        _cluster_threshold = float(self.cluster_corr_threshold)
                        _cluster_bins_source = "n/a"
                else:
                    labels = cluster_correlated_features(
                        X_search.values, threshold=self.cluster_corr_threshold,
                        use_gpu=self.cluster_use_gpu)
                    _cluster_backend = "pearson"
                    _cluster_threshold = float(self.cluster_corr_threshold)
                units, unit_to_members, _kind = build_unit_matrix(
                    X_search.values, labels, weighting=self.cluster_weighting)
                X_proxy = pd.DataFrame(units, columns=[f"unit{i}" for i in range(units.shape[1])])
                _cluster_block = cluster_summary(unit_to_members)
                _cluster_block["backend"] = _cluster_backend
                _cluster_block["threshold"] = _cluster_threshold
                _cluster_block["bins_source"] = _cluster_bins_source
                report["clustering"] = _cluster_block
        else:
            X_proxy = X_search
            unit_to_members = None

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
                cache_dir=self.cache_dir)
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

        # Adaptive prescreen narrowing (iter59): when SHAP per-fold ranks are unstable, NARROW the
        # cap so noisy mid-rank features don't get injected into beam's candidate pool. The lever is
        # measurement-driven (median pairwise Spearman of per-fold mean |phi| feature ranks) and only
        # ever narrows; high-stability regimes keep the current cap. Computed BEFORE the prescreen
        # block below so the resolved cap drives that block's keep count.
        effective_brute_force_cap = self.brute_force_max_features
        adaptive_info: Optional[dict] = None
        if want_per_fold_phi and per_fold_phi_mean is not None and per_fold_phi_mean.shape[0] >= 2:
            from mlframe.feature_selection._shap_proxy_explain import compute_phi_rank_stability

            stability = compute_phi_rank_stability(
                per_fold_phi_mean, top_k=2 * max(self.brute_force_max_features, 40))
            effective_brute_force_cap = _resolve_adaptive_prescreen_width(
                stability, default_cap=self.brute_force_max_features)
            adaptive_info = dict(
                stability=float(stability),
                default_cap=int(self.brute_force_max_features),
                effective_cap=int(effective_brute_force_cap),
            )
            report["adaptive_prescreen"] = adaptive_info

        # Importance pre-screen: when the proxy still has more columns than the exact-search budget,
        # keep the top-K by SHAP importance (mean |phi|) so exhaustive-approx stays feasible.
        n_proxy = phi.shape[1]
        proxy_cols_kept = np.arange(n_proxy)  # proxy(unit) columns behind the current phi columns
        prescreen_top = self.prescreen_top
        if prescreen_top is None and n_proxy > effective_brute_force_cap and self.optimizer in ("auto", "bruteforce", "bruteforce_gpu"):
            prescreen_top = effective_brute_force_cap
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
        if self.trust_guard:
            from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

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
                from mlframe.feature_selection._shap_proxy_prefilter import get_cached_f_scores

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
                                unit_f_scores = np.array(
                                    [float(np.mean(f_working[np.asarray(u, dtype=np.int64)]))
                                     for u in unit_to_members], dtype=np.float64)
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
                    raise ValueError(
                        "ShapProxiedFS: set either `fidelity_floor` (new name) or `spearman_floor` "
                        "(deprecated alias), not both.")
                warnings.warn(
                    "`ShapProxiedFS(spearman_floor=...)` is deprecated since iter18; use "
                    "`fidelity_floor=...` (same semantics). The kwarg name was inherited from the "
                    "iter15 raw-Spearman gate but the gate has been the composite "
                    "`proxy_fidelity_score` since iter16.",
                    DeprecationWarning, stacklevel=2,
                )
                effective_floor = self.spearman_floor
            with _stage("trust_guard"):
                report["trust"] = proxy_trust_guard(
                    phi, base, y_phi, model_template, X_search, X_hold, y_hold,
                    n_anchors=self.n_anchors, rng=self._rng, min_card=self.min_features,
                    max_card=self.max_features, fidelity_floor=effective_floor,
                    n_estimators_cap=self.trust_guard_n_estimators,
                    unit_f_scores=unit_f_scores,
                    anchor_uniform_tail_frac=self.trust_guard_uniform_tail_frac,
                    cardinality_dist=str(self.trust_guard_cardinality_dist).lower(),
                    zipf_alpha=self.trust_guard_zipf_alpha,
                    fidelity_weights=(float(self.trust_guard_fidelity_weights[0]),
                                       float(self.trust_guard_fidelity_weights[1])),
                    trustworthy_metric=str(self.trust_guard_metric).lower(), **rv)

        # Unified candidate re-ranking before the expensive top-N honest retrains: order by the
        # corrector's predicted honest loss (#3/#6, falls back to raw proxy) PLUS an uncertainty
        # penalty (#7). Focuses the retrain budget on subsets that are honestly-best AND stable.
        score = np.array([c[0] for c in candidates], dtype=np.float64)  # raw proxy loss
        if self.use_bias_corrector and self.trust_guard and report.get("trust", {}).get("_corrector_data"):
            from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy_many

            cd = report["trust"]["_corrector_data"]
            corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
            if not corrector.fallback:
                cards = np.array([len(c[1]) for c in candidates], dtype=np.float64)
                redund = subset_redundancy_many(phi, [c[1] for c in candidates])
                score = corrector.predict(score, cards, redund)
                report["bias_corrector"] = dict(applied=True, n_anchors=len(cd["proxy"]))
        if self.uncertainty_penalty > 0 and phi_var is not None:
            from mlframe.feature_selection._shap_proxy_objective import subset_uncertainty_many

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
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau),
                                                  n_kept=len(kept_idx), n_total=int(n_total))
            else:
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau),
                                                  n_kept=int(n_total), n_total=int(n_total))

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
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n],
                                                  active_learning=dict(n_evaluated=n_eval, budget=budget))
                else:
                    from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

                    best_idx, ranked, baseline = revalidate_top_n(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        n_models=self.n_revalidation_models, lambda_stab=self.lambda_stab,
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators,
                        ucb_enabled=self.revalidation_ucb_enabled,
                        ucb_min_eval_size=self.revalidation_ucb_min_eval_size,
                        ucb_slack=self.revalidation_ucb_slack,
                        ucb_stdev_multiplier=self._resolve_revalidation_ucb_stdev_multiplier(
                            n_features),
                        adaptive_n_models=self.revalidation_adaptive_n_models,
                        candidate_score=score, **rv)
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
                    cache=honest_cache, disk_cache_dir=self.cache_dir)

        # Expand best proxy subset -> original member columns, then optionally prune redundant members.
        if unit_to_members is not None:
            member_cols = sorted({int(c) for u in best_idx for c in unit_to_members[int(u)]})
        else:
            member_cols = sorted(int(i) for i in best_idx)
        if self.within_cluster_refine and unit_to_members is not None and len(member_cols) > 1:
            from mlframe.feature_selection._shap_proxy_objective import resolve_metric
            from mlframe.feature_selection._shap_proxy_revalidate import (
                _honest_loss, _open_disk_cache, within_cluster_refine,
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
                    member_groups=member_groups, refine_n_estimators=self.refine_n_estimators,
                    ucb_enabled=self.refine_ucb_enabled,
                    ucb_min_eval_size=self.refine_ucb_min_eval_size,
                    ucb_slack=self.refine_ucb_slack,
                    ucb_stdev_multiplier=self.refine_ucb_stdev_multiplier,
                    inner_n_jobs_cap=self.inner_n_jobs_cap,
                    disk_cache_dir=self.cache_dir)
                # Final full-template re-evaluation of the ONE chosen subset (uncapped n_estimators).
                # Refine's ranking trials use a cheaper capped booster (~100 trees) to decide WHICH
                # members to drop; the user-visible quality bar (and any downstream report consumer)
                # should see this subset's loss at the SAME booster size the other guards used, so the
                # values are apples-to-apples. The cache lookup is the full-template namespace (no
                # template_id), so this hits any prior pipeline retrain of the same subset (e.g. when
                # refine made no drops, this is a cache hit of the union retrain done elsewhere).
                refine_info = dict(before=len(member_cols), after=len(refined))
                if refined:
                    # iter81: the full-template re-eval of the refined subset frequently hits the
                    # disk cache too -- the same (cols, template, cap=None) tuple was retrained as
                    # the revalidation winner upstream, so a warm-cache lookup avoids an extra fit.
                    refine_info["honest_loss_full"] = float(_honest_loss(
                        model_template, X_search, y_search, X_hold, y_hold, list(refined),
                        self.classification, resolve_metric(self.classification, self.metric),
                        cache=honest_cache, disk_cache=_open_disk_cache(self.cache_dir)))
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

    def get_feature_names_out(self, input_features=None):
        """TODO C (2026-05-28): sklearn TransformerMixin convention. Returns the
        selected feature names; pairs with the existing ``get_support`` so both
        sklearn API surfaces work uniformly with the other mlframe selectors
        (MRMR has get_feature_names_out, RFECV has both, ShapProxied previously
        had only get_support). Surfaced by the shared FS contract suite.
        """
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ShapProxiedFS.get_feature_names_out called before fit.")
        return np.asarray(self.selected_features_, dtype=object)
