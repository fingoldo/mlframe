"""Field declarations for :class:`CompositeTargetDiscoveryConfig` (carved base).

Holds the leading block of discovery-config fields so the public class in
``_composite_target_discovery_config`` stays under the 1k-LOC monolith ceiling.
Fields are inherited verbatim; pydantic merges them across the MRO and the
validators on the child class apply to these inherited fields unchanged.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

from pydantic import Field

from ._configs_base import BaseConfig


class CompositeTargetDiscoveryConfigBase(BaseConfig):
    """Carved field-declaration base for :class:`CompositeTargetDiscoveryConfig`."""

    enabled: bool = False

    # Optional chronological-order column (timestamp / monotone index). When set, discovery SORTS the MI-screening sample by it so the tiny-model CV is a forward-walk (TimeSeriesSplit) not a shuffled K-fold -- the canonical non-monotone ``lag(y)`` base defeated the legacy base-monotonicity heuristic, so the screen leaked future->past. None keeps the legacy auto-detection.
    time_column: Optional[str] = None

    # Opt-in: add the 3 chronological-order transforms (ewma_residual / rolling_quantile_ratio / frac_diff) to the candidate set. They need the screen in time order, so set ``time_column`` too. Default OFF -- on a shuffled frame they model a meaningless row sequence.
    time_series_transforms_enabled: bool = False

    # Opt-in discovery steps (wired via ``discovery._opt_in_steps``). ALL default
    # OFF -> a flag-gated no-op that leaves the discovered specs byte-identical to
    # the legacy flow. Each enables one standalone helper over the already-kept
    # single-base specs at the END of ``fit``:
    #   - region_adaptive: per-region best-transform selection routed by frozen
    #     quantile edges of each kept base (``_region_adaptive.fit_region_adaptive``);
    #     results surface on ``CompositeTargetDiscovery.region_adaptive_specs_``.
    #   - interaction_base_discovery: surface ``a OP b`` synthetic interaction bases
    #     whose MI beats both marginals (``_interaction_bases.discover_interaction_bases``);
    #     results surface on ``interaction_bases_`` / ``interaction_base_records_``.
    #   - auto_chain_discovery: compose every ``residual x tail-unary`` chain and keep
    #     those that beat both single stages on held-out y-scale RMSE
    #     (``_auto_chain.discover_chains``); winning chains are APPENDED to ``specs_``
    #     (their composed Transform is registered so ``iter_transform`` resolves it)
    #     and also surface on ``auto_chains_``.
    # Default ON: these opt-in discovery steps each have test-confirmed business
    # value (region-adaptive +47% OOS on region-dependent data, interaction-base
    # +90% on pure-interaction targets, auto-chaining beats both single stages
    # 8/8 seeds) and are no-harm by construction -- region-adaptive +
    # interaction surface as informational artefacts (region_adaptive_specs_ /
    # interaction_bases_) WITHOUT altering the selected specs_, and auto-chaining
    # only APPENDS chains that already beat both single stages on held-out RMSE
    # (empty when none win). All three are compute-bounded by their caps below.
    # Set False to skip the extra discovery passes for the fastest possible fit.
    region_adaptive_enabled: bool = True
    region_adaptive_k: int = 4
    interaction_base_discovery_enabled: bool = True
    interaction_base_top_k: int = 4
    interaction_base_max_pairs: int = 3
    auto_chain_discovery_enabled: bool = True
    auto_chain_top_k: int = 2

    # Base candidate selection.
    # - "auto": rank all numeric features by structural MI gain
    #   (MI(y - LinearFit(x), X \ {x}) on train) and take the top
    #   ``auto_base_top_k`` after applying forbidden-pattern + corr
    #   + ptp filters.
    # - list[str]: explicit list of column names. Still passes through
    #   the forbidden / corr / ptp guards; columns failing the guard
    #   are skipped with a warning.
    base_candidates: Union[List[str], str] = "auto"
    auto_base_top_k: int = 3

    # Priority-base hint: features treated as base candidates regardless of pairwise ``MI(y, x)`` ranking. ``_auto_base`` puts these first (given order) and fills the rest up to ``auto_base_top_k`` with the top MI-ranked features. Exists because pairwise MI is fooled by features with global trend but no structural residual (e.g. spatial coords on geo-trended targets); ``BaselineDiagnostics`` ablation (drop-feature RMSE delta) is far more reliable for "what drives prediction" and ``train_mlframe_models_suite`` populates the hint from it automatically. Hint features still pass the standard filters (forbidden_pattern / non_numeric / constant / corr_threshold); failures are logged + dropped.
    dominant_features_hint: Optional[List[str]] = None

    # Transform names from the registry (mlframe.training.composite).
    #
    # The default extends beyond the original 4 to include the SINGLE-BASE, DROP-IN transforms:
    #   - ``quantile_residual`` -- conditional-on-bin centering + scaling.
    #   - ``monotonic_residual`` -- monotone PCHIP spline residual.
    # These accept the standard ``(y, base)`` signature and need no special orchestration -- discovery evaluates them like ``linear_residual``.
    #
    # NOT in default list (need orchestration): ``linear_residual_multi`` (multi-column base selection via forward stepwise; single-base mode == linear_residual), ``linear_residual_grouped`` (group_column extraction + groups kwarg), and the three chronological-order transforms ``ewma_residual`` / ``rolling_quantile_ratio`` / ``frac_diff`` (now reachable via ``time_series_transforms_enabled`` + ``time_column``). All accessible via explicit ``CompositeTargetEstimator(...)`` and ship their own tests.
    transforms: List[str] = Field(
        default_factory=lambda: [
            "diff", "additive_residual", "median_residual",
            "ratio", "logratio", "linear_residual",
            "linear_residual_robust",  # trimmed-LS variant; safe on outlier-contaminated bases.
            "quantile_residual", "monotonic_residual",
            # y_quantile_clip unary (requires_base=False) -- limit-damage
            # transform for neural / linear downstream models.
            "y_quantile_clip",
            # Unary y-transforms (``requires_base=False`` -- tried once across all bases via the discovery loop's per-transform dedup).
            "cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y",
            # Chain transforms (bivariate residual + unary tail compression).
            "chain_linres_cbrt", "chain_linres_yj",
            "chain_monres_cbrt", "chain_monres_yj",
            # Bivariate extensions. Plug specific
            # failure modes: signed bases (asinh_residual / centered_ratio),
            # curvature beyond OLS line (polynomial_residual_deg2),
            # distribution-free monotone (rank_residual), arbitrary smooth
            # non-monotone dependence (smoothing_spline_residual),
            # multiplicative-jump dynamics (reciprocal_residual). The two
            # multi-base transforms (geometric_mean_residual,
            # pairwise_interaction_residual) stay out of the default list
            # because they need multi-base orchestration (same status as
            # linear_residual_multi).
            "asinh_residual", "centered_ratio", "polynomial_residual_deg2",
            "rank_residual", "smoothing_spline_residual",
            "reciprocal_residual",
        ]
    )

    # Multi-base forward-stepwise auto-promotion: after single-base discovery + raw-y gate + rerank, each kept linear_residual spec greedily ADDS bases from the auto-base pool, upgrading to ``linear_residual_multi`` when the marginal CV-RMSE gain clears ``multi_base_min_marginal_rmse_gain``. Default ON (benchmark: +83% geo-mean on additive DGPs, no-harm on single-dominant / collinear pools; ``benchmarks/composite_multi_base_benchmark.py``); set ``multi_base_enabled=False`` for highly-correlated pools / very small n_train.
    multi_base_enabled: bool = True
    multi_base_max_k: int = 3
    # Marginal relative CV-RMSE gain a candidate base must clear to be ADDED. Default 0.005 (= 0.5%). The old 2% gate stopped early and left genuinely-helpful weak orthogonal bases out: on a disjoint honest holdout (discovery never sees it), 0.005 beats 0.02 in 20/20 non-tied seeds across additive multi-base DGPs (16-19% holdout-RMSE win) with ZERO regression on the single-dominant + noise-decoy DGP (the paired-fold majority gate in forward_stepwise_multi_base independently rejects noise bases regardless of this relative threshold). Bench: ``discovery/_benchmarks/bench_multibase_min_marginal_gain.py``. Set to 0.02 for the legacy conservative gate.
    multi_base_min_marginal_rmse_gain: float = 0.005

    # Robust CV-selector: argmin(mean(fold_rmses)) silently rewards lucky candidates whose mean wins by less than the per-fold std. ``cv_selector_mode`` != "mean" augments the per-fold scores with a dispersion penalty before the argmin (stable mediocre beats unstable lucky); see ``_cv_aggregation.aggregate_fold_scores``. Default "mean" is bit-identical.
    cv_selector_mode: Literal["mean", "mean_minus_std", "median_minus_mad", "t_lcb", "quantile"] = "mean"
    cv_selector_alpha: float = 1.0          # used by mean_minus_std / median_minus_mad
    cv_selector_confidence: float = 0.9     # one-sided Student-t confidence for t_lcb
    cv_selector_quantile_level: float = 0.9 # for aggregate="quantile" (auto-flip by direction)
    cv_persist_fold_scores: bool = False    # surface per-fold scores in forward-stepwise diagnostics

    # Stacked 2-pass composite discovery. When True the suite calls
    # ``CompositeTargetDiscovery.fit_stacked`` instead of plain ``fit``. Pass 1
    # is the normal discovery; for the top ``stacked_max_pass1_specs`` specs
    # we compute OOF predictions on the train rows, append them as new feature
    # columns, and re-run discovery on the augmented feature set. Pass-2 specs
    # may absorb residual-of-residual structure that the first pass missed
    # (e.g. y = f(x_a) + g(x_b): pass 1 takes f(x_a), pass 2 finds g(x_b) on
    # the leftover). Default False so the path is opt-in until measured
    # on real data; switch to True after biz_val on your target.
    # Default-flip eval measured on
    # ``profiling/bench_stacked_discovery_default_flip.py`` on a TRUE
    # residual-of-residual synthetic ``y = 1.5*x_a + 3*sin(x_b) + noise``
    # (pass-1 linear_residual captures the x_a slope; pass-2 should
    # find the non-linear sin-on-x_b structure that pass-1 cannot
    # represent). Even on this structurally favourable problem the
    # feature-stack pass discovered the SAME specs as single-pass and
    # yielded NO holdout-RMSE improvement on the measured n=4000.
    # Verdict: keep default False. Re-run the benchmark on a problem
    # closer to your production target before opting in.
    use_stacked_discovery: bool = False
    stacked_n_oof_folds: int = 3
    stacked_max_pass1_specs: int = 3

    # Residual-target stacked discovery (alternative to
    # ``use_stacked_discovery``). When True the suite calls
    # ``CompositeTargetDiscovery.fit_stacked_on_residual`` instead of
    # ``fit_stacked``. Pass 1 specs collectively predict ``pass1_pred``;
    # ``y - pass1_pred`` becomes the new target for pass 2 discovery.
    # Mathematically the more direct path for residual-of-residual
    # structure when the feature-stack route blocks at the discovery
    # gate (pass 1 OOF prediction is too correlated with target so
    # pass 2 ``mi_gain`` looks marginal).
    #
    # Mutual exclusion: if BOTH ``use_stacked_discovery`` and
    # ``use_stacked_discovery_residual`` are True, residual wins (per
    # the docstring's "more direct" recommendation) and a warning is
    # logged. Default False; opt-in after biz_val on your target.
    use_stacked_discovery_residual: bool = False

    # ``"mean"`` averages OOF predictions across pass-1 specs (robust to a
    # single overfit spec); ``"first"`` uses the best pass-1 spec only.
    # Forwarded to ``fit_stacked_on_residual(residual_aggregation=...)``.
    stacked_residual_aggregation: str = "mean"

    # Cap on how many top-ranked pass-1 specs (best tiny
    # CV-RMSE first) contribute their OOF predictions to the residual-target
    # aggregate in ``fit_stacked_on_residual``. Previously that path aggregated
    # EVERY pass-1 spec (a no-op ``ranked[:max(1,len(ranked))]`` slice), so weak
    # tail specs polluted the ``mean`` aggregate and the leftover residual.
    # Default 3 matches the feature-stack sibling (``stacked_max_pass1_specs``);
    # ``<=0`` restores the historical aggregate-all behaviour. Immaterial when
    # ``stacked_residual_aggregation="first"`` (single best spec only).
    stacked_residual_max_pass1_specs_to_aggregate: int = 3

    # Parallel evaluation of (base, transform) candidates in
    # CompositeTargetDiscovery.fit via joblib(threading). 0 = auto
    # (min(len(work_items), cpu_count)); 1 = serial; >1 = explicit.
    # Phase A is ~20% of fit time (Phase B / _tiny_model_rerank dominates),
    # but the per-candidate MI work is numpy/numba (GIL-released) so the
    # threading parallelism is near-free and bit-equivalent to serial
    # (tests/training/test_composite_discovery_parallel.py). Default 0
    # (auto) alongside the auto tiny_rerank_n_jobs for full Phase A + B
    # parallelism out of the box.
    discovery_n_jobs: int = 0

    # Skip the entire composite-target training block when the raw
    # model already dominates the dummy-baseline ceiling. The discovery's
    # raw-y baseline RMSE / y_std ratio is a cheap proxy for "raw is
    # already near-perfect": when ratio <
    # ``composite_skip_when_raw_dominates_ratio``, composite training
    # is unlikely to add measurable lift on the BASELINE model. Set 0.0
    # to never skip.
    #
    # Default is 0.0 (always run discovery). Prior defaults 0.02 / 0.03
    # were tuned against the Ridge / CB / XGB / LGB zoo where "raw R^2
    # already > 0.99" reliably meant "composite has no headroom".
    # That assumption silently breaks for model-mix suites containing
    # nonlinear / mis-configured downstream models: e.g. an Identity-
    # activation MLP can extrapolate to ~-17 sigma on the random-group
    # test split while Ridge nails R^2=1.00 on the same data. A
    # composite like ``y - top_ar_feature`` would have given the MLP a
    # residual target near zero and saved it. Because the gate fires
    # off the raw-Ridge baseline only, it skipped composite discovery
    # and the MLP collapsed. The 15+ min compute cost is the price for
    # not gambling on this assumption.
    composite_skip_when_raw_dominates_ratio: float = 0.0

    # Complementary skip signal using BaselineDiagnostics' ablation
    # delta%. When the top-ranked feature's drop causes ablation RMSE
    # to balloon by more than this fraction, the raw model is
    # essentially auto-regressive on that one feature. Set 0.0 to
    # never skip; positive value re-enables the heuristic.
    #
    # Default is 0.0 for the same model-mix reason as the ratio gate
    # above: "one feature explains everything for Ridge" doesn't mean
    # "all downstream models will be fine". An MLP that extrapolates
    # badly on unseen wells benefits enormously from the composite
    # ``y - top_ar_feature`` even when Ridge doesn't need it.
    composite_skip_when_ablation_delta_pct: float = 0.0

    # Skip the wrap-pass y-scale predict() calls per composite entry per split
    # (train+val+test). Wide model zoos x multi-million-row frames see 5-15
    # min wall time on this block (MLP predict is the worst at 15-30s each).
    # When True the wrapper still installs CompositeTargetEstimator (so
    # downstream consumers get y-scale predict output) but the y-scale
    # metadata block stays empty -- recover it on demand via
    # `_phase_composite_post.recover_composite_y_scale_metrics`. Safe because
    # T-scale metrics from the per-target phase already cover the watchdog
    # invariants (T-MAE == y-MAE for additive-invertible transforms).
    skip_wrap_pass_predict: bool = True

    # Disable the runtime watchdog when its
    # extra ``wrapper.predict + inner.predict`` per (entry, split) costs
    # more than the rare-bug catch is worth. Measured overhead on
    # n=200k, 2 composites x 5 models = 30 (entry, split) pairs:
    # +72.6% wall time (median 0.353s ON vs 0.204s OFF, see
    # ``profiling/bench_pack_g_watchdog_overhead.py``). On 4M-row frames
    # with MLP inners (predict cost 1-5s) the overhead scales linearly,
    # adding 1-5 min per training pass. Production environments that
    # have verified the wrapper math in staging can disable here.
    # Default True (catch silent bugs).
    enable_wrap_pass_watchdog: bool = True

    # MI screening. Sample to keep the diagnostic under one minute on
    # 4M-row datasets; mi_sample_n=None uses full train.
    #
    # Default lowered 200_000 -> 100_000: prod log analysis
    # showed 5.3 min discovery dominated by 200k MI compute. Halving to
    # 100k gives ~2x speedup with adequate sample size for typical
    # regression / balanced-binary scenarios.
    #
    # HONEST CAVEAT: 100k may be insufficient for two regimes:
    # (a) Imbalanced classification with minority-class rate < 5% --
    #     100k * 5% = 5000 positives per 20-bin MI is borderline;
    #     consider mi_sample_n=200_000 if you see spec drift between runs.
    # (b) Heavy-tail regression where the tail carries the signal --
    #     100k may under-sample the extremes; mi_sample_n=None (use full
    #     train) eliminates the risk at the cost of 20x compute.
    #
    # Final ``raw_baseline_rmse`` gate AND ``tiny_model_rerank`` use
    # FULL train_idx so final spec precision is unaffected by mi_sample_n.
    mi_sample_n: Optional[int] = 100_000
    # Top-K trim after the MI gate. Generous default so pure-lag
    # composites (``y - alpha*lag_y``) -- which have structurally
    # NEGATIVE mi_gain because their residual is noise -- aren't
    # sorted to the bottom of the (mi_gain, name) ordering and
    # truncated out. With 4 unary + ~5 bivariate transforms across
    # 3 bases the max candidate count is ~19; 32 is "keep them all".
    top_k_after_mi: int = 32
    # Pre-filter threshold for ``mi_gain = MI(T, X_no_base) - MI(y, X_no_base)``.
    # Defaults walked +0.01 -> -0.5 -> -10.0 across a real
    # regression incident: pure-lag composite ``T = y - y_prev = noise``
    # has ``MI(T, X_no_base) ~ 0`` while ``MI(y, X_no_base)`` can be
    # large (0.5-1.5 for AR-1 datasets where lag explains nearly
    # everything), so ``mi_gain`` is structurally very negative for
    # the correct composite. -0.5 left the MLP-saving composite still
    # below the gate (mi_y > 0.5). -10.0 effectively disables
    # the MI pre-filter; broken composites (e.g. logratio on negative
    # y) are still caught by the transform's own ``domain_check`` and
    # ``is_degenerate`` flag earlier in the pipeline. The downstream
    # raw-y baseline gate (Phase B) is the real "is this composite
    # useful" decision -- but that gate is now off by default for
    # the same model-mix safety reason (see ``require_beats_raw_baseline``).
    eps_mi_gain: float = -10.0
    mi_n_neighbors: int = 3  # sklearn mutual_info_regression k.

    # MI estimator. "knn" uses the Kraskov estimator (sklearn default,
    # accurate but slow on n>10k); "bin" uses a quantile-binning
    # estimator (5-10x faster, biased low on heavy-tail).
    #
    # Default flipped from "knn" -> "bin" after a
    # statistical review noted that:
    # 1. kNN is biased high on heavy-tail / mixed-density distributions
    #    and the bias scales DIFFERENTLY for raw y (potentially fat-
    #    tailed) vs T = transform(y, base) (sub-Gaussian after
    #    linear_residual). That asymmetric bias inflates apparent
    #    mi_gain even when the true gain is zero -- which matches the
    #    production failure mode where MI passes but RMSE doesn't.
    # 2. bin (quantile) estimator is approximately bias-free under
    #    monotone transforms because the bin edges follow the
    #    transformed distribution -- exactly what the registry's
    #    transforms (diff/ratio/logratio/linear_residual) do.
    # 3. bin is 10x faster on the 200K-row screening sample we
    #    typically run.
    #
    # Set to "knn" explicitly for non-monotone transforms or when
    # n < 5*nbins (bin floor needs ~80 rows at default nbins=16).
    mi_estimator: str = "bin"
    mi_nbins: int = 16  # Bin count when ``mi_estimator == "bin"``.

    # Aggregation across feature columns when
    # comparing MI(T, X_no_base) against MI(y, X_no_base). Legacy
    # ``"sum"`` is biased (overcounts shared information when X is
    # correlated, and the over-count differs between numerator and
    # denominator). Mean is invariant to feature count and is the
    # cleaner default; users on existing benchmarks can pin
    # ``"sum"`` for reproducibility.
    mi_aggregation: str = "mean"

    # MI sampling strategy. "stratified_quantile" (default) bins y into
    # ``mi_n_strata`` quantile bins and samples equally from each, guaranteeing
    # per-bin coverage so the rare-tail rows that carry most of the signal on
    # heavy-tail targets (financial returns, fraud scores, queue lengths) are
    # never dropped by an unlucky uniform draw. "random" is the cheaper legacy
    # uniform sample -- set it explicitly only when you have verified the target
    # is light-tailed and the per-stratum draw buys nothing.
    #
    # Default flipped "random" -> "stratified_quantile": under "random" the
    # heavy-tail ``mi_n_strata`` auto-boost in discovery (skew>2/kurt>5 ->
    # mi_n_strata_heavy_tail) was a dead no-op because the uniform draw never
    # consulted the strata. Stratified sampling is what makes that boost (and
    # the mi_n_strata knob at all) actually steer which rows the MI screen sees;
    # the per-stratum quotas raise tail coverage materially on skewed y without
    # changing the screen on already-uniform y.
    mi_sample_strategy: str = "stratified_quantile"
    mi_n_strata: int = 10

    # Phase B: tiny-model rerank. After MI screening narrows to top-K,
    # train a tiny model (LightGBM or per-family) per surviving
    # candidate and re-rank by CV-RMSE measured on the y-scale (after
    # inverse). This is the "true objective" -- MI is a proxy. Skip
    # by setting ``screening`` = ``"mi"``.
    #
    # Default raised from "mi" -> "hybrid" after a
    # production case where MI-only screening kept composites whose
    # bases (spatial coordinates) had trivial pairwise MI(y, x) but
    # zero structural signal for residual learning. The MI-gain test
    # passed barely (mi_gain ~ 0.01) but the resulting models had
    # WORSE OOF RMSE than raw-y because subtracting the base added
    # noise to the target. Phase B's CV-RMSE-on-y-scale catches this
    # directly. Cost: ~0.5-2 min per target on a 4M-row dataset.
    # Information-criterion validation of transform choice. When True, the
    # discovery loop MAY consult a WAIC/LOO-style score (the expected pointwise
    # out-of-fold predictive density of the tiny-CV residuals, penalised for
    # effective across-fold complexity -- see ``discovery._eval_waic``) as an
    # ADDITIONAL ranking signal alongside MI-gain, never replacing it. It
    # separates two transforms that MI-gain ties when one genuinely generalises and the other overfits the screen.
    # Default ON: the tiny-rerank folds the per-transform WAIC into its ordering as a tie-break, re-ranking only specs
    # whose tiny-CV RMSE falls within a relative noise band (so it never overrides a real RMSE difference) and only when
    # every tied spec has a valid score -- a strict refinement that picks the generalising transform over an overfit
    # one RMSE alone cannot tell apart. Costs one extra cheap K-fold OOF pass per surviving candidate on the small
    # screen sample. Set False to restore the pre-tie-break (RMSE+name) ordering.
    transform_waic_validation_enabled: bool = True
    transform_waic_n_folds: int = 4

    screening: str = "hybrid"  # "mi" | "tiny_model" | "hybrid"
    tiny_model_n_estimators: int = 60
    tiny_model_num_leaves: int = 15
    tiny_model_learning_rate: float = 0.1
    tiny_model_cv_folds: int = 3
    tiny_model_sample_n: int = 20_000  # rows used per tiny-model fit
    top_m_after_tiny: int = 10  # final top-M after Phase B re-rank
    tiny_model_n_jobs: int = 1  # >1 = parallelise CV folds via joblib

    # Parallelise the per-spec rerank loop in
    # ``_tiny_model_rerank``. Each spec runs ``_tiny_cv_rmse_y_scale_multiseed``
    # per family — typically the dominant wall-time slice of Phase B on
    # subsample=200k+ configs. Threads share base/x_matrix arrays via
    # ``backend="threading"``; LightGBM and the inner CV release the GIL.
    # Set to 0 = auto (min(len(kept_specs)*len(families), cpu_count)).
    # Default 0 (auto): Phase B dominates discovery wall-time and the
    # rerank threads share base/x_matrix arrays (no copy) while LightGBM +
    # inner CV release the GIL, so threading parallelism is near-free and
    # bit-equivalent to serial (test_composite_discovery_parallel.py).
    # Set 1 to force serial.
    tiny_rerank_n_jobs: int = 0

    # Force deterministic mode on the tiny models built INSIDE Phase B
    # (``_build_tiny_model``). When True, injects the well-known
    # determinism flags per family:
    # - LightGBM: ``deterministic=True``, ``force_row_wise=True``
    # - XGBoost: explicit ``tree_method="hist"``, ``predictor="auto"``
    # - CatBoost: ``boosting_type="Plain"`` (Plain is deterministic;
    #   Ordered is the non-deterministic default)
    # Bit-exact run-to-run reproducibility on the rerank stage at a
    # 5-10% per-fit cost. Default OFF.
    # Scope: this controls the tiny models we BUILD ourselves for
    # rerank. The actual composite-target inner training (the K
    # LightGBM/XGB models that train the per-spec composite targets)
    # is configured via ``hyperparams_config``, not this flag.
    deterministic_screening_models: bool = False

    # Per-family screening: instead of one tiny LightGBM, train a
    # tiny model of each family in the user's ``mlframe_models``
    # list (cb / lgb / xgb / linear). Different families pick
    # different top features on the same data, so a candidate that
    # wins for one family may lose for another. Aggregation via
    # ``tiny_consensus``:
    # - "single_lgbm" (legacy): one LightGBM, fastest but model-agnostic
    #   proxy gates lie for downstream linear / neural models.
    # - "per_family" (default): train one tiny model per entry in
    #   ``tiny_screening_families`` and aggregate by ``tiny_consensus``
    #   ("union": top-M from each family; "borda": Borda-count rank).
    #   Default families ``("lightgbm", "linear")`` cover the two
    #   distinct downstream regimes: tree boosters (LGBM proxy) and
    #   linear / neural models (Ridge proxy). A composite useful only
    #   for one of the two still survives via the union aggregation.
    #   The 2x screening compute is the price for model-mix safety
    #   (observed in prod: single-LGBM proxy rejected the only
    #   composite that would have saved the downstream Identity-MLP).
    tiny_screening_models: str = "per_family"  # "single_lgbm" | "per_family"
    tiny_screening_families: Tuple[str, ...] = ("lightgbm", "linear")
    tiny_consensus: str = "union"  # "union" | "borda"

    # Raw-y baseline gate. During tiny-model rerank, also train a tiny
    # model on the RAW target (no composite transform) on the same
    # screening sample / folds and use its CV-RMSE as a hard floor:
    # any composite whose CV-RMSE >= raw_baseline * tolerance is
    # rejected as a regression. Catches the "wrong base" case where
    # MI-gain passes but the resulting target is actually harder to
    # predict (e.g. subtracting a spatial coordinate that has global
    # trend with y but no structural residual signal).
    #
    # Tolerance > 1.0 allows composites that are *slightly* worse on
    # the screening sample but might still help in the cross-target
    # ensemble. 1.0 = strict (composite MUST beat raw); 1.02 = within
    # 2% of raw on tiny LGBM.
    #
    # Default flipped True -> False because the gate uses a tiny
    # LGBM as a model-agnostic proxy, and that proxy LIES for
    # downstream linear models. The pure-lag incident:
    # composite ``y - alpha*lag_y`` has residual T = noise, so any
    # downstream model trained on T predicts T_hat ~ 0 and the
    # composite estimator returns y_hat = lag_y -- essentially
    # ``predict y by its lag``. Tiny LGBM on this composite gets
    # CV-RMSE = std(noise) which can be 5-10x worse than tiny LGBM
    # on raw y (which directly fits y from features including
    # lag_y). The gate therefore rejects the composite, even
    # though for an Identity-MLP downstream model the composite
    # is the ONLY thing preventing OOD extrapolation collapse on
    # unseen-groups test splits.
    #
    # Set True to re-enable when running tree-only zoos and you
    # want to cut the per-target training compute on composites
    # the boosters won't benefit from.
    require_beats_raw_baseline: bool = False
    raw_baseline_tolerance: float = 1.02

    # Regime-aware gate. In addition to the
    # global mean RMSE comparison, also check per-quintile-of-base
    # RMSE: a spec is rejected if its tiny CV-RMSE in any quintile
    # exceeds raw_baseline-in-that-quintile by ``raw_baseline_per_bin_tolerance``.
    # This catches "two-regime" failure modes where logratio is
    # correct on multiplicative-regime rows but actively wrong on
    # additive-regime rows; mean RMSE hides this and the spec
    # ships even though it's miscalibrated half the time.
    #
    # Tolerance defaults looser than the global gate (1.10 vs 1.02)
    # because per-bin estimates have higher variance on small
    # screening samples. ``per_bin_n_bins=0`` disables the check.
    #
    # Default flipped 5 -> 0 (off) alongside ``require_beats_raw_baseline``
    # since the per-bin gate is a refinement of the same tiny-LGBM-
    # proxy logic and inherits the same model-mix safety problem.
    # Set > 0 to re-enable when running tree-only zoos.
    raw_baseline_per_bin_tolerance: float = 1.10
    per_bin_n_bins: int = 0

    # Force-inject the ``(top_ablation_feature,
    # diff)`` and ``(top_ablation_feature, additive_residual)`` specs into
    # the discovery output when the top-feature ablation delta% exceeds
    # this threshold, regardless of gate / MI / top-K filtering. This is
    # the "AR-target safety net" -- ensures the simplest possible
    # residualisation composite (``y - top_AR_feature``) is always tried
    # for AR-dominated targets, since that's the composite that bounds
    # linear-stack MLP extrapolation damage on group-aware splits.
    #
    # Default 0.0 (DISABLED). The current gate flips (eps_mi_gain=-10.0,
    # require_beats_raw_baseline=False, top_k_after_mi=32) + the
    # ``additive_residual`` transform already produce the AR-diff spec
    # organically on AR-style data; this flag is an explicit insurance
    # for paranoid configurations where a user re-enables the gates.
    # Enable by setting > 0.0 (typical threshold 50.0 to match
    # ``hint_strength_threshold_pct``). Full implementation pending
    # plumbing of per-feature ablation pct into discovery internals.
    force_inject_diff_on_top_ablation_pct: float = 0.0

    # Median-of-seeds gate. Tiny CV-RMSE with
    # 3 folds is variance-prone (one unlucky split can drag the mean).
    # Optionally repeat the K-fold split with multiple seeds and take
    # the MEDIAN across (folds × seeds) for both raw-y and per-spec
    # CV-RMSE. The gate then compares median composite vs median raw,
    # which is more stable than the mean. Compute scales linearly.
    #
    # Default 3 (raised from 1): a prod run showed
    # the previously-winning ``linres-lag1`` spec getting displaced
    # by ``monres-Y`` chain variants because the single-seed rerank
    # had high variance; with n_seed_repeats=3 the rerank picks the
    # spec that wins on the MEDIAN of 3 splits instead of one unlucky
    # split. 3x compute on screening sample is cheap (sub-minute) vs
    # losing the actual winning spec.
    tiny_model_n_seed_repeats: int = 3

    # Paired one-sided Wilcoxon signed-rank
    # test on per-fold-pair RMSE differences (composite minus raw).
    # Replaces the static ``raw_baseline * tolerance`` threshold with
    # a non-parametric significance test: spec is rejected unless
    # the median of per-fold differences is significantly negative
    # (composite < raw) at level ``gate_alpha``. Scipy must be
    # available; falls back to threshold-only gate if not.
    #
    # Cost: requires per-fold RMSE pairs from BOTH composite and
    # raw runs, which we already collect when
    # ``tiny_model_n_seed_repeats > 1``. With n_seed_repeats=1 the
    # test has 3 fold pairs total -- the test will be too low-power
    # to reject anything except egregious cases. Recommended:
    # n_seed_repeats=5 for the test to have meaningful power.
    use_wilcoxon_gate: bool = False
    gate_alpha: float = 0.05

    # Detect alpha-drift in linear_residual.
    # Fit alpha on first half of train and on second half; compare
    # via Chow-style |Δα| / pooled SE. If the absolute z-score
    # exceeds ``alpha_drift_z_threshold`` (default 3.0), the
    # linear_residual spec for that base is flagged in metadata
    # with reason ``alpha_drift_detected`` and (optionally) rejected.
    # Catches concept-drift / non-stationary y/base relationships
    # that LR's point-estimate alpha silently degrades on at test.
    detect_linear_residual_alpha_drift: bool = True
    alpha_drift_z_threshold: float = 3.0
    # When True, drop linear_residual specs that fail the drift
    # check; when False, keep them but log a warning + record in
    # metadata. Default False -- drift is informational only by
    # default; flag to True on series with known non-stationarity.
    reject_on_alpha_drift: bool = False

    # Bootstrap CI on mi_gain. The point-estimate
    # mi_gain has noise floor that scales with the screening sample
    # size and the heaviness of the y-tail; the eps_mi_gain absolute
    # threshold misses this. Optional bootstrap (resample the
    # screening sample, recompute MI, take 2.5/97.5 percentiles)
    # produces an honest CI; the gate then compares ``eps_mi_gain``
    # against the lower CI bound, not the point estimate.
    #
    # Cost: ``mi_gain_bootstrap_n`` extra MI evaluations per spec
    # (default 0 = disabled; recommended 50 for confidence band).
    mi_gain_bootstrap_n: int = 0
    mi_gain_bootstrap_random_state: int = 12345

    # Family-wise multiplicity control across the many (base, transform) gain
    # tests evaluated in one sweep. Each spec's per-comparison bootstrap CI / MI
    # prefilter controls only its OWN error rate; testing dozens of specs
    # inflates the chance that at least one noise spec spuriously "beats
    # baseline". When enabled AND a per-spec bootstrap p-value exists
    # (``mi_gain_bootstrap_n > 0``), discovery applies a Benjamini-Hochberg FDR
    # correction over the whole family of per-spec p-values AFTER all candidates
    # are scored, and drops specs whose BH-adjusted p-value exceeds
    # ``mi_gain_fdr_alpha`` (one-sided H0: ``mi_gain <= 0``). Default ON: it is a
    # no-op under the shipped defaults (bootstrap disabled -> no p-values -> no
    # specs filtered), so it never regresses recovery on default configs, and it
    # tightens the false-discovery rate the moment a user re-enables the
    # bootstrap gate. ``mi_gain_fdr_alpha`` is the target family-wise FDR level.
    mi_gain_fdr_control: bool = True
    mi_gain_fdr_alpha: float = 0.10

    # Boost n_strata on heavy-tail targets
    # when stratified MI sampling is enabled. Default 10 strata is
    # too few for tail-driven signal -- tail rows get one bin each
    # and MI estimates become unstable. Auto-detection: when y skew
    # > 2.0 OR kurtosis > 5.0, boost ``mi_n_strata`` to
    # ``mi_n_strata_heavy_tail``. Manual override via setting
    # ``mi_n_strata`` explicitly.
    mi_n_strata_heavy_tail: int = 30

    # Post-selection-inference holdout (winner's curse de-bias). The winner spec is
    # selected on the SAME mi_gain statistic that is then reported, so its reported
    # in-screen gain is the MAX over many candidates evaluated on the screening sample
    # -- optimistically biased upward (the curse grows with candidate count). Before
    # screening runs, carve ``honest_holdout_frac`` of the train rows into a holdout the
    # discovery NEVER touches (screening, FDR gate, tiny-rerank, multi-base promotion,
    # opt-in steps all consume only the screening pool); after the winner(s) are picked,
    # RE-SCORE the final spec(s) on this fresh holdout for an honest generalisation gain.
    # The honest gain is reported ALONGSIDE the in-screen gain (both labelled), so
    # downstream generalisation claims use the de-biased number, not the selection score.
    # Default ON per "enable corrective mechanisms by default"; set 0.0 / None to disable
    # (callers who need every row for screening, e.g. tiny train_idx).
    honest_holdout_frac: Optional[float] = 0.2


