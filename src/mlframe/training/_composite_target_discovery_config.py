"""Composite-target discovery configuration for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; the class is re-exported from
``configs`` so existing ``from mlframe.training.configs import
CompositeTargetDiscoveryConfig`` imports continue to resolve.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Union

from pydantic import Field, field_validator, model_validator

from ._composite_target_discovery_config_base import CompositeTargetDiscoveryConfigBase
from ._configs_base import DEFAULT_RANDOM_SEED


class CompositeTargetDiscoveryConfig(CompositeTargetDiscoveryConfigBase):
    """Configuration for composite-target discovery.

    Discovery looks for transformations of the target ``y`` of the form
    ``T = f(y, base)`` such that the model trained on ``T`` (and a
    feature set excluding ``base``) generalises better than the model
    trained on raw ``y``. Typical case: ``y = target`` with ``base = lag1``
    where the autoregressive lag dominates feature importance.

    All fitted parameters (alpha/beta for linear_residual, MAD bounds
    for logratio, etc.) are computed from rows passed via ``train_idx``
    only. Validation and test rows are NEVER touched at fit time.

    Default OFF: opt in by setting ``enabled=True`` and configuring
    base candidates explicitly OR leaving ``base_candidates="auto"``
    for automatic discovery via MI-gain ranking.
    """

    @field_validator("screening", mode="before")
    @classmethod
    def _normalise_screening(cls, v: str) -> str:
        """Case-fold and validate the ``screening`` mode: ``"mi"`` (pairwise MI-gain ranking only), ``"tiny_model"``
        (tiny-booster consensus only), or ``"hybrid"`` (MI screening followed by tiny-model rerank of the survivors).
        Raises ``ValueError`` on any other value; no aliasing besides case-folding.
        """
        v_lower = str(v).lower()
        valid = {"mi", "tiny_model", "hybrid"}
        if v_lower not in valid:
            raise ValueError(f"screening must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_estimator", mode="before")
    @classmethod
    def _normalise_mi_estimator(cls, v: str) -> str:
        """Case-fold and validate the ``mi_estimator``: ``"knn"`` (Kraskov k-NN estimator) or ``"bin"`` (equi-frequency
        binned MI, bias-free under monotone transforms and the project default). Raises ``ValueError`` otherwise.
        """
        v_lower = str(v).lower()
        valid = {"knn", "bin"}
        if v_lower not in valid:
            raise ValueError(f"mi_estimator must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_sample_strategy", mode="before")
    @classmethod
    def _normalise_mi_sample_strategy(cls, v: str) -> str:
        """Case-fold and validate ``mi_sample_strategy``: how the MI-screening sample is drawn from train rows --
        ``"random"`` (uniform subsample) or ``"stratified_quantile"`` (sampled to preserve the target's quantile
        distribution, avoiding a subsample that misses a whole regime of y). Raises ``ValueError`` otherwise.
        """
        v_lower = str(v).lower()
        valid = {"random", "stratified_quantile"}
        if v_lower not in valid:
            raise ValueError(f"mi_sample_strategy must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("tiny_screening_models", mode="before")
    @classmethod
    def _normalise_tiny_screening_models(cls, v: str) -> str:
        """Case-fold and validate ``tiny_screening_models``: ``"single_lgbm"`` (one small LightGBM per candidate,
        cheapest) or ``"per_family"`` (a tiny model per transform-family, more expensive but catches transform-
        specific effects a single booster can miss). Raises ``ValueError`` otherwise.
        """
        v_lower = str(v).lower()
        valid = {"single_lgbm", "per_family"}
        if v_lower not in valid:
            raise ValueError(f"tiny_screening_models must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("tiny_consensus", mode="before")
    @classmethod
    def _normalise_tiny_consensus(cls, v: str) -> str:
        """Case-fold and validate ``tiny_consensus``: how tiny-model verdicts across candidates are combined --
        ``"union"`` (keep a candidate if ANY tiny model favours it) or ``"borda"`` (rank-aggregate across tiny
        models via Borda count, penalising candidates only some models like). Raises ``ValueError`` otherwise.
        """
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

    # Structural pre-discovery leakage guard (detect_base_target_leakage). Default ON, but it only ACTS when a
    # time_ordering is supplied to fit() -- the lag-probe then distinguishes a genuine time-shifted lag(y) base
    # (NOT leaky, the canonical composite case) from a same-time near-identity re-encoding of y (leaky), so it
    # never drops a legitimate lag. Catches target-encoding / rolling-target bases the forbidden_base regex misses.
    # On non-temporal data (no time_ordering) it is a no-op so it can never mistake autocorrelation for leakage.
    detect_base_leakage: bool = True

    # Block columns whose Pearson |corr(base, y)| exceeds this threshold.
    # Intent: catch literal copies / trivial linear transforms of y
    # (e.g. ``y_renamed = y``, ``y_scaled = y / 1000``). NOT intended
    # to catch autoregressive lag features such as ``y_prev`` --
    # those legitimately reach corr ~ 0.999 on slow-moving series due
    # to autocorrelation, and they are exactly the kind of dominant
    # feature composite-target discovery exists to handle.
    #
    # The primary defence against target-encoding leakage is the regex
    # patterns above (``forbidden_base_patterns``); the corr threshold
    # is just a backstop. Default raised from 0.999 to 0.99999
    # after observing it filtered out a legitimate
    # ``y_prev`` (lag-1) on a real production run.
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

    # Hint-strength threshold for the adaptive hint cap.
    # When the top hint feature has BaselineDiagnostics ablation
    # ``delta_pct >= hint_strength_threshold_pct``, ``_auto_base``
    # uses the FULL hint list (no cap) instead of capping at
    # ``max(1, top_k // 2)``. Set to a high value (e.g. 1000) to
    # effectively disable the strong-hint shortcut.
    hint_strength_threshold_pct: float = 50.0

    # Cross-base correlation dedup. After
    # auto-base ranking, drop a candidate base if its absolute Pearson
    # correlation against any already-kept candidate exceeds this
    # threshold on the screening sample. Stops near-duplicate lag
    # variants (``y_prev``, ``y_prev_lag2``, ``y_smooth_3``) from
    # all surviving into Phase B and inflating ensemble correlation.
    # Set to 1.0 to disable.
    auto_base_dedup_corr_threshold: float = 0.95

    # De-duplicate near-collinear feature columns from ``x_remaining`` BEFORE the
    # per-base ``mi_y_compare`` MI baseline. ``x_remaining`` excludes only the
    # base column itself; a near-duplicate sibling of the base (a second lag, a
    # smoothed copy) left in the remaining set carries almost the same info as
    # the removed base, so it inflates ``MI(y, x_remaining)`` while contributing
    # little to ``MI(T, x_remaining)`` -- biasing ``mi_gain`` DOWN. The bias is
    # conservative (it never over-keeps a spec, only wrongly sinks one), but it
    # most hurts exactly the lag-family bases discovery is built to find. When
    # enabled, columns whose absolute Pearson correlation with an
    # already-kept-column exceeds ``dedup_x_remaining_corr_threshold`` are
    # dropped from ``x_remaining`` (the FIRST of each collinear group is kept) so
    # both halves of ``mi_gain`` score the de-duplicated feature set. Default ON
    # per the "enable corrective mechanisms by default" convention; the dedup is
    # a strict no-op when no two surviving columns are that correlated, and the
    # threshold defaults high (0.99) so only genuine near-duplicates are removed.
    # Set ``False`` to reproduce the pre-fix full-``x_remaining`` baseline.
    dedup_x_remaining_for_mi_baseline: bool = True
    dedup_x_remaining_corr_threshold: float = 0.99

    # Rank auto-base candidates by MI computed
    # with PER-PAIR (per-column) NaN masking instead of the global
    # all-column finite intersection. For mid-range-NaN columns the global
    # intersection keeps only the rows where EVERY feature is observed -- a
    # non-random (MNAR) subset -- so MI(y, x_j) on it is biased by the
    # joint-observability pattern and silently shifts which base wins.
    # Per-pair masking estimates each column's MI on its own observed rows
    # (matching ``_mi_to_target`` and the prebinned ``-1``-sentinel path)
    # and is bit-identical when the screening sample has no NaN. Default ON
    # per the "enable corrective mechanisms by default" convention; set to
    # False only to reproduce the pre-fix global-mask ranking.
    auto_base_mi_per_pair_mask: bool = True
    # Fraction of the per-pair-available row mass below which the global
    # intersection is judged to be MNAR-shrinking the ranking sample; used
    # purely to LOG that the per-pair ranking diverged from what the global
    # mask would have produced (auditability). Does not change behaviour
    # when ``auto_base_mi_per_pair_mask`` is True (per-pair is always used).
    auto_base_mnar_per_pair_threshold: float = 0.5

    # Permutation-MI null distribution test in
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

    # Structural detectors for time-index and
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

    # Structural-affinity boost for ``_auto_base``. Surfaces OBVIOUS base
    # columns from data shape / correlation that the MI ranking alone can miss
    # when a noisier competitor's pairwise MI(y, x) lands a hair higher:
    # - **Near-affine predictor** (``|corr(y, x)|`` very high AND the OLS
    #   residual variance collapses): prime ``linear_residual`` base.
    # - **Low-cardinality integer column** (a small set of distinct integer
    #   levels, not one-per-row): prime ``grouped`` base.
    # - **Monotone / timestamp column** (forward diffs share one sign): prime
    #   ``time`` base.
    # The boost is a BOUNDED additive nudge scaled to the candidate MI spread
    # (``auto_base_structural_boost_fraction`` of the MI range), so it can lift
    # a near-tie but never override a clearly larger MI gap -- it AUGMENTS the
    # MI ranking, it does not replace it. Bit-identical to "no boost" on data
    # with no detectable structure. Default ON per the "enable corrective
    # mechanisms by default" convention; set False to reproduce the pre-boost
    # MI-only ranking.
    auto_base_structural_boost: bool = True
    auto_base_structural_boost_fraction: float = 0.25

    # Collapse ``linear_residual`` -> ``diff`` when the fitted alpha
    # is approximately 1.0. ``linear_residual``
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

    # Handling multilabel (multi-output) regression targets,
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

    # Post-hoc recalibration of the cross-target ensemble's blended output.
    #
    # When True, after the OOF gate the suite fits a monotone
    # ``OutputCalibrator`` on the SAME OOF holdout surface the NNLS /
    # gain weights were derived from -- it blends the OOF component
    # matrix with the frozen ensemble weights, then fits an isotonic
    # (or sigmoid / linear) map of that OOF blend onto the OOF truth and
    # applies it as a final monotone post-map at predict time. This
    # removes the systematic (often S-shaped) miscalibration a
    # least-squares blend of biased components leaves behind, without
    # changing the ensemble's ranking. Leakage-free: the fit consumes
    # only out-of-fold predictions, never a re-prediction of train.
    #
    # Default False: with no calibrator attached predict returns the raw
    # blend bit-for-bit, so the feature is bit-identical when off.
    calibrate_cross_target_output: bool = False
    # Calibration map family: "isotonic" (free-form monotone, default),
    # "sigmoid" (Platt-style 2-param S), or "linear" (affine scale+offset).
    cross_target_calibration_method: str = "isotonic"

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
    # Default flipped from 0.0 -> 0.2 because the default
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

    # OOF holdout source for the cross-target ensemble stacker / weights / gate. Three modes:
    #
    # - ``"kfold"`` (default): true train-K-fold OOF. Each component is re-fit on K-1 folds and predicts the
    #   held-out fold; the concatenated (n_train, K) OOF matrix drives stack weights, gain-over-naive weighting,
    #   and the honest-OOF gate. This is the only source that never reuses the early-stopping (val) surface for
    #   weighting: the booster components were early-stopped against val, so weighting on val double-dips a biased
    #   surface and systematically over-weights whichever component fit the val noise best. K-fold OOF is the
    #   standard "stacking on OOF" cure (Sill et al. 2009). Cost: K-1 extra fits per component on (K-1)/K of train.
    #
    # - ``"external_val"``: fit each component clone on the FULL train slice, predict on the suite's val frame.
    #   Cheaper (one fit per component) but the val frame was the early-stopping surface for the booster
    #   components, so the resulting weights/gate are optimistically biased toward components that overfit val.
    #   A one-time WARN is emitted. Keep this for a representativeness cross-check against the kfold source, or
    #   when K-fold is too expensive; do not use it as the production weighting surface for early-stopped models.
    #
    # - ``"train_tail"``: legacy single-slice carve from the trailing ``oof_holdout_frac`` of train (time-aware
    #   when ``time_ordering`` is monotone, random shuffle otherwise). Use when val is unavailable / single fit
    #   is required and the train-tail distribution matches test.
    oof_holdout_source: str = "kfold"

    # Number of folds for ``oof_holdout_source="kfold"``. 5 is the standard stacking default; higher K gives a
    # larger per-component training fraction (less pessimistic OOF) at linear extra fit cost.
    oof_kfold: int = 5

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

    # Residual-correlation dedup. Before weighting / stacking, compute the pairwise Pearson correlation of the
    # honest-OOF residuals (``residual_correlation_matrix``) and, for any pair whose |corr| exceeds
    # ``ct_ensemble_dedup_corr_threshold``, drop the WEAKER member (higher OOF RMSE). Near-duplicate components
    # otherwise split the NNLS weight between themselves and let a redundant pair dominate the stack. Always keeps
    # at least 2 members. Default OFF pending the committed bench
    # (training/_benchmarks/bench_ct_ensemble_residual_dedup.py); flip ON only if it wins on the majority of seeds.
    ct_ensemble_dedup_enabled: bool = False
    ct_ensemble_dedup_corr_threshold: float = 0.95

    # AR(1) failsafe: when ``lag_predict`` is injected into the
    # CompositeCrossTargetEnsemble component pool and its OOF holdout
    # RMSE is within ``(1 + lag_predict_failsafe_tolerance)`` of the
    # best single trained component, prefer the zero-parameter
    # ``lag_predict`` over the multi-component stack. Defends against
    # the train-tail-vs-test distribution mismatch on strong-AR targets.
    # Default 0.10 (10%). The earlier 0.50 was calibrated for
    # group-blind train-tail carves where the trained-model OOF was
    # artificially inflated by ~25% due to within-group leakage in
    # the inner early-stopping eval; now that ``_carve_inner_eval_split``
    # is group-aware, the trained-model honest OOF is no
    # longer biased high vs lag, so the tolerance no longer needs to
    # absorb the gap. Set to 0 to disable.
    lag_predict_failsafe_tolerance: float = 0.10

    # Dummy-floor gate: when honest OOF predictions are available, drop
    # any component from the CompositeCrossTargetEnsemble pool whose
    # OOF RMSE > raw target's strongest-dummy RMSE x (1 + tolerance).
    # A trained model that loses to a parameter-free dummy on the
    # honest holdout cannot improve the ensemble; keeping it dilutes
    # NNLS weight and harms test performance. Prod observed
    # this as the load-bearing failure mode: composite-target
    # models on residual T overfit on group-aware splits (pred_std up
    # to 5x target_std, R2 down to -22 on test), but NNLS still gave
    # them positive weight. With the gate the pool reduces to
    # {lag_predict} + components that genuinely beat the dummy, and
    # the ensemble cannot ship worse than the dummy. Set
    # ``ct_ensemble_dummy_floor_enabled = False`` to disable; bump
    # ``ct_ensemble_dummy_floor_tolerance`` to keep components within
    # a small slack above the dummy (default 0 = strict).
    ct_ensemble_dummy_floor_enabled: bool = True
    ct_ensemble_dummy_floor_tolerance: float = 0.0

    # Extreme-AR + group-aware skip. When the target is dominated by
    # an AR lag (``lag1_autocorr_per_group >= extreme_ar_threshold``)
    # AND the split is group-aware (test wells/groups unseen at
    # training), composite-target discovery is short-circuited because
    # the residual T = y - alpha * lag has near-zero signal on unseen
    # groups; any trained model on T overfits per-group patterns and
    # produces predictions worse than the trivial median(T) dummy on
    # the test split (observed in prod: 3 composite specs shipped,
    # all 9 trained models on residuals failed dummy gate with R2 <0,
    # ~10 min of wall-time wasted per target). Set
    # ``extreme_ar_group_aware_skip = False`` to force discovery to
    # run anyway.
    extreme_ar_group_aware_skip: bool = True
    extreme_ar_threshold: float = 0.99

    # Build CT_ENSEMBLE for raw targets even when ZERO composite specs
    # were discovered. Without this knob, the entry guard in
    # ``_phase_composite_post`` requires composite_specs_by_target_type
    # to be non-empty, so the dummy-floor gate + lag_predict injection
    # + AR(1) failsafe never run on raw-only targets. The suite then
    # ships a simple-arithmetic ensemble of raw models that is worse
    # than the best single component when 3-of-4 boosters are above the
    # lag floor (observed in prod: EnsARITHM TEST=12.45 vs Ridge
    # alone 11.63, lag_predict 11.58). With this knob the OOF gate sees
    # Ridge alone and prefers it, or falls back to lag_predict if the
    # AR-failsafe tolerance is met. Disable to revert to the legacy
    # raw-models-only ensemble path (mean/median/arith flavours only).
    always_build_ct_ensemble_for_raw: bool = True
    # produces an opt-in stub call to ``composite_oof_predictions`` /
    # ``composite_predictions_as_feature`` on the discovered specs so
    # downstream code can attach the predictions as engineered features.
    # Default False; full wiring requires the downstream FE pipeline to
    # consume the new column, which is caller-specific.
    composite_feature_stacking_enabled: bool = False

    @field_validator("cross_target_ensemble_strategy", mode="before")
    @classmethod
    def _normalise_ensemble_strategy(cls, v: str) -> str:
        """Case-fold and validate ``cross_target_ensemble_strategy``: ``"off"`` (no ensemble built), ``"mean"``
        (equal-weight average of all raw+composite components), ``"oof_weighted"`` (gain-over-baseline weighting
        from per-component RMSE), ``"linear_stack"`` (Ridge regression on component predictions), or
        ``"nnls_stack"`` (non-negative least squares stacking, the project default). Raises ``ValueError`` otherwise.
        """
        v_lower = str(v).lower()
        valid = {"off", "mean", "oof_weighted", "linear_stack", "nnls_stack"}
        if v_lower not in valid:
            raise ValueError(f"cross_target_ensemble_strategy must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("oof_holdout_source", mode="before")
    @classmethod
    def _normalise_oof_holdout_source(cls, v: str) -> str:
        """Case-fold and validate ``oof_holdout_source``: ``"kfold"`` (true train-K-fold OOF, the default -- never
        reuses the early-stopping val surface for stack weighting), ``"external_val"`` (cheaper single fit that
        predicts on the suite's val frame, but the weights end up optimistically biased toward val-overfit
        components), or ``"train_tail"`` (legacy single trailing-slice carve). Raises ``ValueError`` otherwise.
        """
        v_lower = str(v).lower()
        valid = {"kfold", "external_val", "train_tail"}
        if v_lower not in valid:
            raise ValueError(f"oof_holdout_source must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("fail_on_no_gain", mode="before")
    @classmethod
    def _normalise_fail_mode(cls, v: str) -> str:
        """Case-fold and validate ``fail_on_no_gain``: behaviour when no candidate clears ``eps_mi_gain`` --
        ``"fallback_raw"`` (warn and emit no composite targets, caller trains on raw target only, the default),
        ``"raise"`` (raise ``RuntimeError``, useful for CI/scripted modes to flag degenerate inputs), or ``"warn"``
        (warn but still emit the best-of-bad candidates). Raises ``ValueError`` on any other value.
        """
        v_lower = str(v).lower()
        valid = {"fallback_raw", "raise", "warn"}
        if v_lower not in valid:
            raise ValueError(f"fail_on_no_gain must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("multilabel_strategy", mode="before")
    @classmethod
    def _normalise_multilabel_strategy(cls, v: str) -> str:
        """Case-fold and validate ``multilabel_strategy`` for 2-D (multi-output) regression targets: ``"per_target"``
        (default -- expand into ``n_outputs`` independent 1-D targets named ``{name}_out{j}``, each discovered
        separately), ``"skip"`` (legacy -- mark with metadata, produce no composites), or
        ``"multi_target_regression"`` (keep the (N, K) target joint for a shared-trunk multi-output model instead
        of expanding). Raises ``ValueError`` on any other value.
        """
        v_lower = str(v).lower()
        # "multi_target_regression" keeps
        # (N, K) regression targets joint under
        # TargetTypes.MULTI_TARGET_REGRESSION instead of expanding to
        # K independent 1-D targets. Best for correlated targets that
        # benefit from a shared trunk / boosting ensemble (MultiRMSE,
        # multi_output_tree, K-head MLP).
        valid = {"per_target", "skip", "multi_target_regression"}
        if v_lower not in valid:
            raise ValueError(f"multilabel_strategy must be one of {valid}, got '{v}'")
        return v_lower

    @model_validator(mode="after")
    def _append_time_series_transforms(self) -> "CompositeTargetDiscoveryConfig":
        """When ``time_series_transforms_enabled`` is on, add the three
        chronological-order transforms to the candidate set (deduped, appended
        so the existing order is preserved). They are valid only once the
        screening sample is in time order, which ``time_column`` guarantees;
        enabling without a time signal is allowed but warned by discovery.
        """
        if getattr(self, "time_series_transforms_enabled", False):
            _ts = ["ewma_residual", "rolling_quantile_ratio", "frac_diff"]
            _present = set(self.transforms)
            for _t in _ts:
                if _t not in _present:
                    self.transforms.append(_t)
        return self

    @model_validator(mode="after")
    def _warn_on_no_win_stacked_discovery(self) -> "CompositeTargetDiscoveryConfig":
        """Warn when a stacked-discovery flag is enabled: the committed bench
        ``profiling/bench_stacked_discovery_default_flip.py`` measured NO holdout
        improvement vs single-pass discovery. The flags stay functional (just warned).
        """
        if getattr(self, "use_stacked_discovery", False) or getattr(self, "use_stacked_discovery_residual", False):
            warnings.warn(
                "use_stacked_discovery / use_stacked_discovery_residual is enabled, but on benchmarks "
                "this provides no measurable improvement over single-pass discovery "
                "(profiling/bench_stacked_discovery_default_flip.py). Re-benchmark on your target before relying on it.",
                stacklevel=2,
            )
        return self
