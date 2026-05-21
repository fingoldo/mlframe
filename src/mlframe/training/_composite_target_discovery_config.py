"""Composite-target discovery configuration for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; the class is re-exported from
``configs`` so existing ``from mlframe.training.configs import
CompositeTargetDiscoveryConfig`` imports continue to resolve.
"""
from __future__ import annotations

from typing import Optional, List, Set, Tuple, Union

from pydantic import Field, field_validator

from ._configs_base import BaseConfig, DEFAULT_RANDOM_SEED


class CompositeTargetDiscoveryConfig(BaseConfig):
    """Configuration for composite-target discovery.

    Discovery looks for transformations of the target ``y`` of the form
    ``T = f(y, base)`` such that the model trained on ``T`` (and a
    feature set excluding ``base``) generalises better than the model
    trained on raw ``y``. Typical case: ``y = TVT`` with ``base = TVT_prev``
    where the autoregressive lag dominates feature importance.

    All fitted parameters (alpha/beta for linear_residual, MAD bounds
    for logratio, etc.) are computed from rows passed via ``train_idx``
    only. Validation and test rows are NEVER touched at fit time.

    Default OFF: opt in by setting ``enabled=True`` and configuring
    base candidates explicitly OR leaving ``base_candidates="auto"``
    for automatic discovery via MI-gain ranking.
    """

    enabled: bool = False

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

    # Priority-base hint -- features that should be treated as base
    # candidates regardless of pairwise ``MI(y, x)`` ranking. When
    # populated, ``_auto_base`` puts these first (in given order) and
    # fills remaining slots up to ``auto_base_top_k`` with the top
    # MI-ranked features.
    #
    # The hint exists because pairwise MI is fooled by features that
    # have global trend with y but no structural residual signal
    # (e.g. spatial coordinates on geographically-trended targets).
    # ``BaselineDiagnostics``'s ablation (drop feature -> measure RMSE
    # delta) is a much more reliable signal for "which feature
    # actually drives prediction": a feature whose removal hurts RMSE
    # by 500% is unambiguously dominant, regardless of MI estimation
    # noise. ``train_mlframe_models_suite`` populates the hint from
    # the ablation output automatically; users can also pass it
    # explicitly.
    #
    # Hint features still go through the standard filters
    # (forbidden_pattern / non_numeric / constant / corr_threshold);
    # any that fail are logged and dropped.
    dominant_features_hint: Optional[List[str]] = None

    # Transform names from the registry (mlframe.training.composite).
    #
    # 2026-05-11 (R10c brainstorm rollout): default extended from the original 4 to include the SINGLE-BASE, DROP-IN transforms shipped in commits 9e05955 + 0894369:
    #   - ``quantile_residual`` -- conditional-on-bin centering + scaling.
    #   - ``monotonic_residual`` -- monotone PCHIP spline residual.
    # These accept the standard ``(y, base)`` signature and need no special orchestration -- discovery evaluates them like ``linear_residual``.
    #
    # NOT in default list (require orchestration the discovery loop does not yet provide):
    #   - ``linear_residual_multi`` -- needs multi-column base selection (forward stepwise); single-base mode is identical to ``linear_residual``.
    #   - ``linear_residual_grouped`` -- needs ``group_column`` extraction + groups kwarg through fit/forward/inverse.
    #   - ``ewma_residual`` / ``rolling_quantile_ratio`` / ``frac_diff`` -- require chronological row order which most datasets lack at the discovery stage.
    # All four are accessible via explicit user configuration (``CompositeTargetEstimator(...)`` directly) and ship with their own tests; auto-discovery integration is the open item beyond this PR.
    transforms: List[str] = Field(
        default_factory=lambda: [
            "diff", "ratio", "logratio", "linear_residual",
            "linear_residual_robust",  # trimmed-LS variant; safe on outlier-contaminated bases.
            "quantile_residual", "monotonic_residual",
            # Pack J unary y-transforms (``requires_base=False`` -- tried once across all bases via the discovery loop's per-transform dedup).
            "cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y",
            # Pack K chain transforms (bivariate residual + unary tail compression).
            "chain_linres_cbrt", "chain_linres_yj",
            "chain_monres_cbrt", "chain_monres_yj",
        ]
    )

    # OPEN-1 integration (2026-05-12): multi-base forward-stepwise auto-promotion of kept ``linear_residual`` specs. After single-base discovery + raw-y baseline gate + tiny-model rerank, Discovery picks each kept linear_residual spec and tries greedily ADDING bases from the auto-base candidate pool. When the marginal CV-RMSE reduction clears ``multi_base_min_marginal_rmse_gain`` (default 2%), the spec is upgraded to ``linear_residual_multi`` with the expanded base list.
    #
    # Default ON: measure-first benchmark (``benchmarks/composite_multi_base_benchmark.py``) confirms geo-mean gain 83% on positive scenarios (S2: y = b1+b2+eps, S3: y = b1+b2+b3+eps) AND no-harm on negative scenarios (S1: single dominant b1 + noise candidates; S4: collinear b1+b1_dup pool). Decision rule met. Opt-out by setting ``multi_base_enabled=False`` if production data violates the benchmark's assumptions (highly correlated candidate pool, very small n_train, etc.).
    multi_base_enabled: bool = True
    multi_base_max_k: int = 3
    multi_base_min_marginal_rmse_gain: float = 0.02

    # Pack #3: stacked 2-pass composite discovery. When True the suite calls
    # ``CompositeTargetDiscovery.fit_stacked`` instead of plain ``fit``. Pass 1
    # is the normal discovery; for the top ``stacked_max_pass1_specs`` specs
    # we compute OOF predictions on the train rows, append them as new feature
    # columns, and re-run discovery on the augmented feature set. Pass-2 specs
    # may absorb residual-of-residual structure that the first pass missed
    # (e.g. y = f(x_a) + g(x_b): pass 1 takes f(x_a), pass 2 finds g(x_b) on
    # the leftover). Default False so the path is opt-in until measured
    # on real data; switch to True after biz_val on your target.
    # T3#20 2026-05-18 default-flip eval: measured on
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

    # T1#6 2026-05-18 #4: residual-target stacked discovery (alternative to
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

    # Parallel evaluation of (base, transform) candidates in
    # CompositeTargetDiscovery.fit via joblib(threading) when > 1. Default 1
    # because the parallel block covers ~20% of fit time -- _tiny_model_rerank
    # (Phase B) dominates the tail. Use tiny_rerank_n_jobs > 1 alongside this
    # for the full Phase A + B speedup. Parallel path is bit-equivalent to
    # serial (covered by tests/training/test_composite_discovery_parallel.py).
    discovery_n_jobs: int = 1

    # 2026-05-18 #10: skip the entire composite-target training block
    # when the raw model already dominates the dummy-baseline ceiling.
    # The discovery's raw-y baseline RMSE / y_std ratio is a cheap
    # proxy for "raw is already near-perfect": when ratio <
    # ``composite_skip_when_raw_dominates_ratio``, composite training
    # is unlikely to add measurable lift. Production TVT log: Ridge on
    # raw TVT achieved MAE=7.89 (better than CB / XGB / LGB); the two
    # discovered composites (monres-Y, monresYj-Y) produced IDENTICAL
    # metrics to raw -- pure compute loss.
    #
    # Default flipped 0.0 -> 0.02 2026-05-18 (Accuracy/perf over legacy):
    # 0.02 means "raw explains >= 98% of y's variance" -- a regime where
    # composites genuinely have no headroom. The conservative case
    # (composite captures the last 2%) is rare in practice on residual-
    # structure datasets. Set 0.0 to restore historical "never skip".
    #
    # 2026-05-21 raised 0.02 -> 0.03: TVT regression with ratio=0.0228
    # (R^2 already 0.9995) ran 15.6 min of discovery and produced 1
    # spec that scored identically to raw -- pure compute loss.
    # 0.03 = R^2 >= 0.9991, still extremely conservative; composites
    # capturing the last 0.1% of variance are vanishingly rare and
    # don't justify 15+ minutes per target.
    composite_skip_when_raw_dominates_ratio: float = 0.03

    # 2026-05-21: complementary skip signal using BaselineDiagnostics'
    # ablation delta%. When the top-ranked feature's drop causes
    # ablation RMSE to balloon by more than this fraction, the raw
    # model is essentially auto-regressive on that one feature
    # (production TVT log: top_ablation_delta%=3209% means dropping
    # TVT_prev makes RMSE 33x worse -- the model literally IS
    # ``y ~ TVT_prev``). Composite discovery in this regime spends
    # tens of minutes finding transforms that capture the same trivial
    # mapping; better to skip entirely. Set 0.0 to disable.
    composite_skip_when_ablation_delta_pct: float = 500.0

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

    # HIGH#4 2026-05-18: disable the Pack G runtime watchdog when its
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
    # 2026-05-18 default lowered 200_000 -> 100_000: TVT log analysis
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
    top_k_after_mi: int = 8
    # Pre-filter threshold for ``mi_gain = MI(T, X_no_base) - MI(y, X_no_base)``.
    # Default lowered from +0.01 -> -0.5 on 2026-05-11 (R10c bug #3)
    # after a production TVT regression run discovered 0 specs despite
    # BaselineDiagnostics ablation correctly identifying ``TVT_prev``
    # as the dominant feature. Root cause: pure-lag composite
    # ``T = y - y_prev = noise`` has ``MI(T, X_no_base) ~ 0`` while
    # ``MI(y, X_no_base) > 0``, so ``mi_gain`` is structurally
    # NEGATIVE for the correct composite -- a sign of a clean lag fit,
    # not a sign the composite is useless. The MI-gain pre-filter
    # was rejecting LEGITIMATE compositions.
    #
    # The actual "is this composite predictively useful" decision is
    # made downstream by the raw-y baseline gate (Phase B; compares
    # tiny CV-RMSE of composite vs raw-y on the same screening folds).
    # With ``eps_mi_gain=-0.5`` the pre-filter only drops composites
    # whose mi_gain is MUCH worse than raw -- typical "transform broke
    # the target" cases (logratio on negative y, ratio on near-zero
    # base). Pure-lag composites pass through to the raw-y gate where
    # they are correctly evaluated.
    eps_mi_gain: float = -0.5
    mi_n_neighbors: int = 3  # sklearn mutual_info_regression k.

    # MI estimator. "knn" uses the Kraskov estimator (sklearn default,
    # accurate but slow on n>10k); "bin" uses a quantile-binning
    # estimator (5-10x faster, biased low on heavy-tail).
    #
    # Default flipped from "knn" -> "bin" 2026-05-10 after a
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

    # R10b statistician #1: aggregation across feature columns when
    # comparing MI(T, X_no_base) against MI(y, X_no_base). Legacy
    # ``"sum"`` is biased (overcounts shared information when X is
    # correlated, and the over-count differs between numerator and
    # denominator). Mean is invariant to feature count and is the
    # cleaner default; users on existing benchmarks can pin
    # ``"sum"`` for reproducibility.
    mi_aggregation: str = "mean"

    # MI sampling strategy. "random" is the cheap default; switch to
    # "stratified_quantile" on heavy-tail targets (financial returns,
    # fraud scores, queue lengths) where random sampling can miss the
    # tail rows that carry most of the signal. Stratified sampling
    # bins y into ``mi_n_strata`` quantile bins and samples equally
    # from each, guaranteeing per-bin coverage.
    mi_sample_strategy: str = "random"
    mi_n_strata: int = 10

    # Phase B: tiny-model rerank. After MI screening narrows to top-K,
    # train a tiny model (LightGBM or per-family) per surviving
    # candidate and re-rank by CV-RMSE measured on the y-scale (after
    # inverse). This is the "true objective" -- MI is a proxy. Skip
    # by setting ``screening`` = ``"mi"``.
    #
    # Default raised from "mi" -> "hybrid" in 2026-05-10 after a
    # production case where MI-only screening kept composites whose
    # bases (spatial coordinates) had trivial pairwise MI(y, x) but
    # zero structural signal for residual learning. The MI-gain test
    # passed barely (mi_gain ~ 0.01) but the resulting models had
    # WORSE OOF RMSE than raw-y because subtracting the base added
    # noise to the target. Phase B's CV-RMSE-on-y-scale catches this
    # directly. Cost: ~0.5-2 min per target on a 4M-row dataset.
    screening: str = "hybrid"  # "mi" | "tiny_model" | "hybrid"
    tiny_model_n_estimators: int = 60
    tiny_model_num_leaves: int = 15
    tiny_model_learning_rate: float = 0.1
    tiny_model_cv_folds: int = 3
    tiny_model_sample_n: int = 20_000  # rows used per tiny-model fit
    top_m_after_tiny: int = 3  # final top-M after Phase B re-rank
    tiny_model_n_jobs: int = 1  # >1 = parallelise CV folds via joblib

    # 2026-05-20 #10: parallelise the per-spec rerank loop in
    # ``_tiny_model_rerank``. Each spec runs ``_tiny_cv_rmse_y_scale_multiseed``
    # per family — typically the dominant wall-time slice of Phase B on
    # subsample=200k+ configs. Threads share base/x_matrix arrays via
    # ``backend="threading"``; LightGBM and the inner CV release the GIL.
    # Set to 0 = auto (min(len(kept_specs)*len(families), cpu_count)).
    # Default 1 preserves serial behaviour for back-compat.
    tiny_rerank_n_jobs: int = 1

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
    # - "single_lgbm" (default): one LightGBM, fastest.
    # - "per_family": train ``tiny_screening_models`` per family;
    #   aggregate by ``tiny_consensus`` ("union": top-M from each
    #   family; "borda": Borda-count rank aggregation).
    tiny_screening_models: str = "single_lgbm"  # "single_lgbm" | "per_family"
    tiny_screening_families: Tuple[str, ...] = ("lightgbm",)
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
    # ensemble. 1.0 = strict (composite MUST beat raw). Default 1.02
    # = composite kept if within 2% of raw, rejected if worse.
    require_beats_raw_baseline: bool = True
    raw_baseline_tolerance: float = 1.02

    # R10b improvement #1: regime-aware gate. In addition to the
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
    # screening samples. Set ``per_bin_n_bins=0`` to disable the
    # per-bin check.
    raw_baseline_per_bin_tolerance: float = 1.10
    per_bin_n_bins: int = 5

    # R10b improvement #10: median-of-seeds gate. Tiny CV-RMSE with
    # 3 folds is variance-prone (one unlucky split can drag the mean).
    # Optionally repeat the K-fold split with multiple seeds and take
    # the MEDIAN across (folds × seeds) for both raw-y and per-spec
    # CV-RMSE. The gate then compares median composite vs median raw,
    # which is more stable than the mean. Compute scales linearly.
    #
    # Default flipped 1 -> 3 (2026-05-18): production TVT run showed
    # the previously-winning ``linres-TVT_prev`` spec getting displaced
    # by ``monres-Y`` chain variants because the single-seed rerank
    # had high variance; with n_seed_repeats=3 the rerank picks the
    # spec that wins on the MEDIAN of 3 splits instead of one unlucky
    # split. 3x compute on screening sample is cheap (sub-minute) vs
    # losing the actual winning spec.
    tiny_model_n_seed_repeats: int = 3

    # R10b statistician #4: paired one-sided Wilcoxon signed-rank
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

    # R10b statistician #6: detect alpha-drift in linear_residual.
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

    # R10b stat #8: bootstrap CI on mi_gain. The point-estimate
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

    # R10b stat #8 (continued): boost n_strata on heavy-tail targets
    # when stratified MI sampling is enabled. Default 10 strata is
    # too few for tail-driven signal -- tail rows get one bin each
    # and MI estimates become unstable. Auto-detection: when y skew
    # > 2.0 OR kurtosis > 5.0, boost ``mi_n_strata`` to
    # ``mi_n_strata_heavy_tail``. Manual override via setting
    # ``mi_n_strata`` explicitly.
    mi_n_strata_heavy_tail: int = 30

    @field_validator("screening", mode="before")
    @classmethod
    def _normalise_screening(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"mi", "tiny_model", "hybrid"}
        if v_lower not in valid:
            raise ValueError(f"screening must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_estimator", mode="before")
    @classmethod
    def _normalise_mi_estimator(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"knn", "bin"}
        if v_lower not in valid:
            raise ValueError(f"mi_estimator must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_sample_strategy", mode="before")
    @classmethod
    def _normalise_mi_sample_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"random", "stratified_quantile"}
        if v_lower not in valid:
            raise ValueError(
                f"mi_sample_strategy must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("tiny_screening_models", mode="before")
    @classmethod
    def _normalise_tiny_screening_models(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"single_lgbm", "per_family"}
        if v_lower not in valid:
            raise ValueError(
                f"tiny_screening_models must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("tiny_consensus", mode="before")
    @classmethod
    def _normalise_tiny_consensus(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"union", "borda"}
        if v_lower not in valid:
            raise ValueError(f"tiny_consensus must be one of {valid}, got '{v}'")
        return v_lower

    # Forbidden base filters. Block columns whose names match any of
    # these regex patterns (target leakage via target encoding /
    # rolling target stats / etc.).
    forbidden_base_patterns: List[str] = Field(
        default_factory=lambda: [
            r"^target_enc_",
            r"^mean_target_",
            r"_te$",
            r"^lagged_target_",
            r"^y_smooth_",
        ]
    )

    # Block columns whose Pearson |corr(base, y)| exceeds this threshold.
    # Intent: catch literal copies / trivial linear transforms of y
    # (e.g. ``y_renamed = y``, ``y_scaled = y / 1000``). NOT intended
    # to catch autoregressive lag features such as ``TVT_prev`` --
    # those legitimately reach corr ~ 0.999 on slow-moving series due
    # to autocorrelation, and they are exactly the kind of dominant
    # feature composite-target discovery exists to handle.
    #
    # The primary defence against target-encoding leakage is the regex
    # patterns above (``forbidden_base_patterns``); the corr threshold
    # is just a backstop. Default raised from 0.999 to 0.99999 in
    # 2026-05-10 after observing it filtered out legitimate
    # ``TVT_prev`` (lag-1) on a real production run.
    forbidden_base_corr_threshold: float = 0.99999

    # Block constant or near-constant base columns (zero variance ->
    # OLS in linear_residual is degenerate; ratio / logratio are
    # uninformative).
    constant_base_eps: float = 1e-12

    # Domain validity. Drop a (base, transform) candidate entirely if
    # fewer than this fraction of train rows pass the transform's
    # domain_check (e.g. logratio requires y, base > 0).
    min_valid_domain_frac: float = 0.7

    # Behaviour when no candidate clears eps_mi_gain.
    # - "fallback_raw": warn and emit no composite targets (caller
    #   trains on raw target only).
    # - "raise": raise RuntimeError -- useful in CI / scripted modes
    #   to flag degenerate inputs.
    # - "warn": warn but emit the best-of-bad candidates anyway.
    fail_on_no_gain: str = "fallback_raw"

    random_state: int = DEFAULT_RANDOM_SEED

    # Cross-target ensemble strategy. Run after each composite-target
    # model is wrapped to y-scale, builds one combined predictor over
    # all (raw + K composite) wrappers.
    # - "off": no ensemble; ``models[type][f"_CT_ENSEMBLE__{target}"]`` not created.
    # - "mean": equal-weight average over all components.
    # - "oof_weighted": gain-over-baseline weighting using per-component
    #   RMSE (train-RMSE proxy by default; honest holdout RMSE when
    #   ``oof_holdout_frac > 0``); auto-falls-back to best-single
    #   component if no component clears the baseline.
    # - "linear_stack": Ridge regression on per-component predictions.
    # - "nnls_stack": non-negative least squares on per-component preds.
    #
    # nnls_stack chosen as default after `composite_ensemble_shootout.py`
    # (6 scenarios x 3 seeds, 11 strategies): NNLS was the only strategy
    # with positive mean improvement vs best-single-by-train (+1.24%, 13/18
    # wins). Single-spec case falls back to best-single inside the ensemble
    # class. Set to "off" to skip ensemble construction entirely.
    cross_target_ensemble_strategy: str = "nnls_stack"

    # When True AND the per-target ``baseline_diagnostics`` reports
    # ``composite_recommendation == "unlikely_to_help"``, discovery
    # short-circuits with a warning and produces no specs. Saves
    # the MI / tiny-model / re-fit cost on targets where composite
    # mode is unlikely to add value (init_score baseline already
    # captures the dominance, or no feature dominates strongly).
    # Default False so explicit opt-ins don't get silently overridden.
    auto_skip_on_baseline_optimal: bool = False

    # Use BaselineDiagnostics ablation top-K as priority base
    # candidates (``dominant_features_hint``) instead of relying on
    # pairwise MI(y, x) ranking alone. Pairwise MI gets fooled by
    # features with global trend but no structural residual signal
    # (spatial coords on geographically-trended y); ablation directly
    # measures predictive contribution and is much more reliable.
    #
    # When True, ``train_mlframe_models_suite`` runs BaselineDiagnostics
    # inline (cached) before discovery and injects the top-K
    # ablation-ranked features as the hint. When the inline
    # diagnostic fails or returns no dominant features, falls back
    # silently to MI-only ranking.
    #
    # Default True since it strictly improves auto-base on the
    # production failure mode and the inline BD cost is amortised
    # (the same diagnostic runs in the per-target loop later;
    # caching reuses it).
    use_baseline_diagnostics_hint: bool = True
    baseline_diagnostics_hint_top_k: int = 3

    # R10c bug #5: hint-strength threshold for the adaptive hint cap.
    # When the top hint feature has BaselineDiagnostics ablation
    # ``delta_pct >= hint_strength_threshold_pct``, ``_auto_base``
    # uses the FULL hint list (no cap) instead of capping at
    # ``max(1, top_k // 2)``. Set to a high value (e.g. 1000) to
    # effectively disable the strong-hint shortcut.
    hint_strength_threshold_pct: float = 50.0

    # Cross-base correlation dedup (R10b improvement #9). After
    # auto-base ranking, drop a candidate base if its absolute Pearson
    # correlation against any already-kept candidate exceeds this
    # threshold on the screening sample. Stops near-duplicate lag
    # variants (``TVT_prev``, ``TVT_prev_lag2``, ``TVT_smooth_3``) from
    # all surviving into Phase B and inflating ensemble correlation.
    # Set to 1.0 to disable.
    auto_base_dedup_corr_threshold: float = 0.95

    # R10b improvement #2: permutation-MI null distribution test in
    # ``_auto_base``. For each candidate feature compute MI(y, x) AND
    # MI(y, shuffle(x)) on ``auto_base_null_perms`` shuffles, then
    # require the candidate's MI to exceed ``mean_null + n_sigma *
    # std_null``. Catches features whose MI(y, x) is non-trivial only
    # because of a shared monotonic component (time/spatial trend),
    # not structural information about y.
    #
    # Cost: ``auto_base_null_perms`` extra MI evaluations per
    # candidate (default 20 × ~1ms each on bin-MI estimator = ~20ms
    # per feature on the screening sample). Set
    # ``auto_base_null_perms=0`` to disable.
    auto_base_null_perms: int = 20
    auto_base_null_z_threshold: float = 3.0
    # Block-shuffle length for temporal datasets so the null
    # preserves marginal autocorrelation. ``"auto"`` uses
    # ``int(sqrt(n))``; explicit int for fixed length; ``1`` for
    # row-level shuffle (i.i.d. assumption).
    auto_base_null_block_length: Union[str, int] = "auto"

    # R10b improvement #7: structural detectors for time-index and
    # spatial-coordinate features. Demote (push to bottom of MI
    # ranking) features that look like:
    # - **Time index**: |Spearman(rank(x), arange(n))| > 0.95.
    #   Catches a row-counter or timestamp masquerading as a base
    #   candidate; on temporal data the row index correlates with
    #   y purely from drift, no structural information.
    # - **Spatial coordinate block**: pairwise correlations among
    #   3+ numeric features form a block where each pair has
    #   |corr| > 0.5. Catches X/Y/Z lat-lon-altitude triplets where
    #   pairwise MI(y, coord) is high purely from spatial drift,
    #   not structural information.
    # Demoted features are ALSO available as bases when their MI
    # genuinely exceeds non-demoted candidates (defensive, not
    # blocking). Set to False to disable.
    auto_base_demote_time_index: bool = True
    auto_base_demote_spatial_coords: bool = True

    # Collapse ``linear_residual`` -> ``diff`` when the fitted alpha
    # is approximately 1.0 (R10b improvement #6). ``linear_residual``
    # is a strict generalisation of ``diff`` (diff = linear_residual
    # with alpha=1, beta=0). When OLS lands at alpha~1 on stationary
    # lag features, the two transforms produce numerically identical
    # T columns -- but ``linear_residual`` carries TWO learned
    # parameters with train-time variance. ``diff`` is the lower-
    # variance answer. The threshold compares the scale-invariant
    # ratio ``|alpha - 1| * std(base) / std(y)``; below this value,
    # the linear_residual spec is considered redundant with diff and
    # dropped if a diff spec for the same base also kept. Set to 0.0
    # to disable (always keep both).
    collapse_linear_residual_alpha_eps: float = 0.05

    # R3.18: handling multilabel (multi-output) regression targets,
    # i.e. ``target_by_type[regression][name]`` is a 2-D array of
    # shape ``(n_rows, n_outputs)``.
    # - ``"per_target"`` (default): expand into ``n_outputs`` separate
    #   1-D regression targets named ``{name}_out{j}``; discovery
    #   runs independently per output, naming composites
    #   ``{name}_out{j}__{transform}__{base}``. Per-target training
    #   loop downstream sees them as ordinary 1-D targets.
    # - ``"skip"``: legacy behaviour -- mark with metadata note,
    #   produce no composites for that target. Useful when the
    #   caller knows they don't want the per-output expansion (e.g.
    #   the training loop downstream expects the 2-D shape intact).
    multilabel_strategy: str = "per_target"

    # Cap the number of components combined at predict time. Useful
    # for online single-row latency-sensitive serving where running
    # K=8 wrappers per row blows the SLA. When > 0, the ensemble
    # keeps only the top-N components by weight (after the standard
    # weight computation), drops the rest, and re-normalises. None
    # / 0 means keep all components (default).
    max_inference_components: Optional[int] = None

    # Honest OOF for the ensemble gate / stacking.
    #
    # When > 0, the suite carves an extra holdout slice (this fraction
    # of filtered_train_idx) and at ensemble-build time re-fits a
    # clone of every component on the (1-frac) stack_train slice,
    # then predicts on the held-out slice. The honest holdout
    # predictions feed the stacking solvers (linear_stack /
    # nnls_stack) and the gain-over-baseline weights, replacing the
    # train-RMSE proxy that overstates accuracy. Cost: re-fits every
    # component once on (1-frac) of train rows.
    #
    # Default flipped from 0.0 -> 0.2 on 2026-05-15 because the default
    # cross_target_ensemble_strategy is ``nnls_stack``. Fitting NNLS on
    # in-sample component predictions is a stacking leak: every
    # component has effectively memorised its training rows, so NNLS
    # picks weights that overweight whichever component fits noise
    # best. A 20% honest holdout is the standard "stacking on OOF" cure
    # documented in Sill et al. 2009 (Feature-Weighted Linear Stacking)
    # and removes the leak at the cost of one extra fit per component
    # on 80% of train rows. Set to 0.0 explicitly to opt out (e.g. when
    # using a non-stacking strategy like ``mean`` where the train-RMSE
    # proxy is harmless).
    oof_holdout_frac: float = 0.2
    oof_random_state: int = DEFAULT_RANDOM_SEED

    # Stacking-aware gate (measure-first NNLS gate). When True AND
    # ``cross_target_ensemble_strategy`` is ``linear_stack`` or
    # ``nnls_stack``, the ensemble-build path first runs
    # :func:`stacking_aware_gate` over the component predictions to drop
    # components whose NNLS weight falls below
    # ``stacking_aware_gate_min_weight``. The surviving subset feeds the
    # actual stacker. Skipped when fewer than 2 components survive (the
    # stacker handles single-component falls back on its own).
    stacking_aware_gate_enabled: bool = False
    stacking_aware_gate_min_weight: float = 0.05

    # Composite-feature stacking. When True, ``run_composite_target_discovery``
    # produces an opt-in stub call to ``composite_oof_predictions`` /
    # ``composite_predictions_as_feature`` on the discovered specs so
    # downstream code can attach the predictions as engineered features.
    # Default False; full wiring requires the downstream FE pipeline to
    # consume the new column, which is caller-specific.
    composite_feature_stacking_enabled: bool = False

    @field_validator("cross_target_ensemble_strategy", mode="before")
    @classmethod
    def _normalise_ensemble_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"off", "mean", "oof_weighted", "linear_stack", "nnls_stack"}
        if v_lower not in valid:
            raise ValueError(
                f"cross_target_ensemble_strategy must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("fail_on_no_gain", mode="before")
    @classmethod
    def _normalise_fail_mode(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"fallback_raw", "raise", "warn"}
        if v_lower not in valid:
            raise ValueError(f"fail_on_no_gain must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("multilabel_strategy", mode="before")
    @classmethod
    def _normalise_multilabel_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"per_target", "skip"}
        if v_lower not in valid:
            raise ValueError(
                f"multilabel_strategy must be one of {valid}, got '{v}'")
        return v_lower

