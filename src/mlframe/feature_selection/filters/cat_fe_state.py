"""Configuration and persistence dataclasses for cat-FE.

Why two dataclasses, not many parallel ``MRMR.__init__`` kwargs:

- ``CatFEConfig`` packages every cat-FE knob into a single attr on the estimator. ``MRMR(cat_fe_config=CatFEConfig(enable=True, ...))`` replaces fourteen
  parallel ``cat_fe_*`` kwargs that would otherwise bloat the init surface; once shipped flat, those names are forever pinned by sklearn's ``_get_param_names``
  introspection, so folding them into a config object is the only reversible move.
- ``CatFEState`` packages every cat-FE persistence attribute (recipes, diagnostics, search-phase MI cache) under one ``self._cat_fe_state_`` field.
  ``__setstate__`` injects a single default; ``clone()`` survives a single round-trip; future additions land inside the dataclass without bloating ``MRMR.__dict__``.

Default-selection rationale:

- ``include_numeric=False``        -- discretized noisy floats produce spurious aliasing; opt-in when domain knowledge supports mixing.
- ``full_npermutations=50``        -- zero is an anti-statistical trap (no FWER guarantee). 50 is a cheap default; bump to 500-1000 with bh_fdr / westfall_young for strict FWER control.
- ``permutation_null="joint_independence"`` -- matches what shuffle-Y actually tests; the misnamed "synergy null" stays a documented limitation, addressable later via conditional permutation.
- ``select_on="synergy"``          -- canonical use case; redundancy / absolute are explicit alternatives.
- ``min_interaction_information=None`` -- resolved at fit time to ``-3 / sqrt(n_samples)`` (small-negative absorbs finite-sample noise around the synergy boundary).
- ``backend="auto"``               -- GPU when N>=200 AND n>=500k AND CuPy available, else CPU prange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CatFEConfig:
    """All cat-FE configuration in one place.

    Attached to an MRMR instance via the ``cat_fe_config`` constructor
    kwarg: ``MRMR(cat_fe_config=CatFEConfig(enable=True, top_k_pairs=128))``.
    """

    # ----- core -----
    enable: bool = True
    """Master switch. ``True`` (default): cat-FE step runs after categorical detection, before screening. ``False``: legacy MRMR behaviour, no cat-FE.
    Cat-FE shows measurable wins (XOR biz_value test, no regressions) per the project's accuracy-over-compat policy. Users relying on legacy behaviour can
    pin ``cat_fe_config=None`` or ``CatFEConfig(enable=False)``."""

    marginal_floor: float = 0.0
    """Drop categorical columns with ``MI(X_i; Y) < marginal_floor`` from the pair-search pool. ``0.0`` keeps everything; e.g. ``0.05`` prunes low-signal cols before the N^2 pair search."""

    max_combined_nbins: int | None = None
    """Hard cap on ``nbins[i] * nbins[j]`` for any pair candidate. ``None`` resolves at fit time to the data-aware Paninski-derived ceiling
    (``max(4, n * 0.05 / 3 + 1)``). Absolute cap 10**7 to prevent OOM via ``cat_fe_max_combined_nbins=10**9``-style misconfig."""

    top_k_pairs: int = 32
    """How many top pairs (by II / synergy score) to keep after argpartition over all candidate pairs. Memory linear in this value. Default 32 is conservative
    for the on-by-default cat-FE path; bump to 64+ for richer FE."""

    max_kway_order: int = 2
    """Max k for k-way greedy expansion. ``2`` = pairs only; ``3..interactions_max_order`` = greedy triplet / quartet expansion from the top pairs. NOT
    capped by ``MRMR.interactions_max_order`` -- engineered cols are 1-way features to screening."""

    min_interaction_information: float | None = None
    """Floor for accepting a pair / k-way tuple. ``None`` resolves at fit time to ``-3 / sqrt(n_samples)`` (finite-sample noise margin). The value is
    interaction information (synergy minus redundancy), not pure synergy."""

    include_numeric: bool = False
    """Mix numeric columns into the cat pool. Default ``False`` because noisy floats produce spurious aliasing interactions; opt-in only when domain knowledge supports it.
    When ``True``, eligible NaN-free numeric columns are quantile-binned (``numeric_nbins`` bins, edges fitted on train and STORED in the recipe for leak-safe transform
    replay) and become cat-FE candidates -- their pair/k-way crosses capture axis-aligned AND non-product (e.g. diagonal / rotated) interactions that the numeric unary/binary
    FE cannot express (measured: +0.42 OOS AUC on a rotated quadrant target where ``mul(a,b)`` gives +0.00; +0.013 on an axis-aligned one, LogisticRegression downstream).
    NaN-bearing numeric columns are skipped in v1 (the quantile-edge replay path does not yet encode a NaN bin); pre-impute or one-hot the missingness to include them."""

    numeric_nbins: int = 10
    """Quantile bins for numeric columns when ``include_numeric=True``. The edges are fitted on the fit-time data and stored per source column in the engineered recipe so
    ``transform`` reproduces identical bin codes (no train/serve skew). 10 mirrors the MRMR default ``quantization_nbins``; raise for finer interaction grids at the cost of
    sparser cells (cardinality is still capped by ``max_combined_nbins`` and the ``sqrt(n)*2`` per-column ceiling)."""

    shortlist_npermutations: int = 0
    """Permutations for the search-phase point estimate. ``0`` skips permutation entirely during search; surviving pairs get a separate confirmation pass with ``full_npermutations``."""

    # ----- statistical rigor -----
    full_npermutations: int = 50
    """Permutations for the confirmation phase on top-K survivors. ``0`` is an anti-statistical trap. Default 50: cheap sanity check vs joint-independence
    null. For strict FWER control bump to 500-1000 AND switch ``fwer_correction`` to bh_fdr / westfall_young."""

    fwer_correction: Literal["none", "bonferroni", "bh_fdr", "westfall_young"] = "none"
    """Multiple-testing correction across the search-phase ``N(N-1)/2`` candidate pairs. WY amortises shuffles across pairs (the shuffle-Y cycle that does
    confirmation computes per-shuffle max-II over all pairs). ``none`` for legacy behaviour (FWER approx 1.0)."""

    permutation_null: Literal["joint_independence", "conditional"] = "joint_independence"
    """Null hypothesis for the permutation test. Default ``joint_independence`` (shuffle Y) tests ``I(X1,X2;Y) = 0``, NOT synergy specifically; the surfaced
    ``confidence`` field is named ``joint_dependence_confidence`` to make this honest. ``conditional`` (shuffle X2 within strata of Y) approaches the synergy
    null at higher cost."""

    use_miller_madow: bool | None = None
    """Miller-Madow bias correction to ALL six entropies in the II expansion as a unit. ``None`` auto-gates on ``(a*b*c) / n > 0.05``. Force ``True`` for
    high-cardinality joints, ``False`` for legacy plug-in behaviour."""

    # ----- modeling choice -----
    select_on: Literal["synergy", "redundancy", "absolute"] = "synergy"
    """Sign of interaction information to select on. ``synergy`` keeps positive II (canonical); ``redundancy`` keeps negative II (useful for noise-robust
    models that benefit from correlated parents); ``absolute`` keeps |II| > floor (any strong joint structure)."""

    # ----- backend / parallelism -----
    backend: Literal["auto", "cpu", "gpu"] = "auto"
    """Compute backend. ``auto`` picks CPU prange below ``N=200, n=500_000``, GPU above (when CuPy is installed). ``cpu`` forces numba prange; ``gpu`` requires CuPy and raises if unavailable."""

    # ----- stability / replication -----
    n_folds_stability: int = 0
    """K-fold stability filter. ``0`` disables. ``>0`` runs the pair search on each of K folds and keeps only pairs prevalent in ``min_fold_prevalence * K`` folds."""

    min_fold_prevalence: float = 0.6
    """When ``n_folds_stability > 0``, fraction of folds in which a pair must clear ``min_interaction_information`` to be kept."""

    # ----- ranking coupling -----
    anti_redundancy_beta: float = 0.0
    """mRMR-coupled scoring. ``0`` = pure II ranking. ``> 0`` = ``score = II - beta * max_z I(merged; Z)`` over already-selected features. Heuristic, NOT a
    derived mRMR criterion -- use the two-stage decoupled path (II gate then mRMR rank) for formally principled coupling."""

    # ----- diagnostics -----
    emit_diagnostics: bool = True
    """Populate ``self._cat_fe_state_.diagnostics`` with per-pair ``(II, marginal_X1_MI, marginal_X2_MI, joint_MI, n_uniq, n_obs_per_cell_p25)`` so users can debug "why was this pair selected?"."""

    # ----- gate combination -----
    gate_logic: Literal["and", "or", "permutation_primary"] = "and"
    """When both K-fold stability and permutation confidence are enabled, how to combine the gates. ``and`` (default, strictest) requires both; ``or``
    requires either; ``permutation_primary`` uses permutation as primary with K-fold as a tiebreaker."""

    # ----- transform contract -----
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip"
    """How ``transform()`` handles test-time category values not seen during fit. ``clip`` (default): cap at the highest trained bin. ``sentinel``: dedicated unknown bin (inflates cardinality). ``raise``: hard error."""

    # ----- pathological-input gates -----
    min_n_samples: int = 200
    """Hard floor on n -- below this II is dominated by sample-size bias and permutation tests have insufficient unique reorderings (n=2 has only 2 distinct shuffles)."""

    min_class_count: int = 50
    """Minimum samples per class of Y. Below this, marginal MI estimates on cat columns are dominated by which rows happen to be the minority (5 positives in
    100k means each (X1, X2, Y=1) cell has 0 or 1 obs). Cat-FE warns and disables when violated."""

    on_high_cardinality: Literal["skip", "raise"] = "skip"
    """How a categorical column whose nbins exceeds the ``sqrt(n)*2`` ceiling is handled. ``skip`` (default): drop it from the cat-FE candidate pool, warn once, and
    let it flow through the rest of MRMR as an ordinary (high-cardinality) column the relevance screen can still drop -- raw frames with id/hash/free-text columns no
    longer crash MRMR.fit. ``raise``: legacy hard ValueError (kept for callers who want a strict "this column shouldn't be categorical" signal)."""

    discretization_audit: bool = False
    """Re-run pair search with ``nbins +- 2`` for numeric columns and record per-pair II range. Surfaces pairs whose ranking is sensitive to the discretization scheme."""

    # ----- bootstrap / target-encoding / weights -----
    bootstrap_ci_n_replicates: int = 0
    """Bootstrap confidence intervals on top-K II values. ``0`` (default) disables. ``>0`` runs ``n_replicates`` subsamples of size 0.632*n, recomputes II per
    replicate, surfaces (lower, median, upper) CI in diagnostics. Dropping pairs whose lower-CI < ``min_interaction_information`` complements the permutation
    test: perm checks 'is there ANY signal'; bootstrap checks 'is II stably > floor under sample variation'."""

    bootstrap_ci_alpha: float = 0.10
    """Two-sided alpha for bootstrap CI (default 0.10 -> 90% CI). Set to 0.05 for 95% CI."""

    permutation_subsample: int | None = None
    """Subsample size for permutation null distribution computation. ``None`` (default) uses the full N rows in every shuffle iteration. ``int`` subsamples to
    this many rows BEFORE shuffling Y, for the permutation test ONLY. ``ii_obs`` (the observed test statistic) is always computed on the full N.

    Statistical / runtime trade-off:

    * The permutation null distribution shifts toward higher variance when computed on fewer samples (each shuffled MI is noisier). For the confidence-against-
      null gate this is mostly conservative -- noisier null -> wider tail -> fewer pairs pass. False-negative rate goes up slightly; false-positive rate stays bounded.
    * Cost cuts roughly linearly with subsample / N. On a 1M-row regression + MRMR profile, full-N permutation took ~25 s; subsampled to 100k it drops to ~2.5 s.

    Recommended: set to ~100_000 when ``n_rows >= 5e5`` and runtime matters more than the last decimal of confidence precision. Leave at ``None`` for batch-
    inference / model-card flows where the permutation test is a publication-grade statistic.
    """

    bootstrap_sample_frac: float = 0.632
    """Bootstrap subsample fraction. 0.632 is the canonical out-of-bag fraction (expected fraction of unique rows in a bootstrap-with-replacement sample of size n)."""

    sample_weight_col: str = ""
    """Name of a column in the input X holding per-row sample weights. Empty string (default) disables -- equivalent to uniform weights. When set, the column
    is consumed by cat-FE and NOT included in the candidate pool."""

    emit_target_encoding: bool = False
    """In addition to the factorize-recipe engineered cols, emit a target-encoded version: ``E[Y | merged_class]`` per cell. With ``target_encoding_oof_folds > 0``,
    uses out-of-fold encoding to prevent leakage. Useful when downstream models prefer numeric inputs (regression trees with continuous splits, linear / NN) over categorical codes."""

    target_encoding_oof_folds: int = 5
    """Number of K-fold splits for out-of-fold target encoding. ``0`` disables OOF (uses naive per-cell mean; leaks if used directly as training feature).
    Default 5: standard CV practice. Reference: Micci-Barreca 2001 ('A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems')."""

    target_encoding_smoothing: float = 10.0
    """Smoothing for target encoding: ``te = (n_c * te_raw + alpha * te_global) / (n_c + alpha)``. Reduces overfitting for rare cells. Alpha=10 shrinks rare cells toward the global mean."""

    # ----- advanced features -----
    use_kt_smoothing: bool = False
    """Krichevsky-Trofimov smoothing for entropy estimation. Adds ``0.5 / (n + K/2)`` pseudocounts to each cell before entropy. Less biased than plug-in for
    high-cardinality joints; alternative to Miller-Madow. Reference: Krichevsky & Trofimov 1981."""

    perm_budget_strategy: Literal["fixed", "bandit_ucb1"] = "bandit_ucb1"
    """Permutation budget allocation. ``"bandit_ucb1"`` (default): UCB1 allocates more shuffles to ambiguous pairs (II close to the floor); saves 2-5x total
    perms on typical workloads where some pairs are clearly significant or clearly noise, while maintaining the same FWER-guaranteed statistical power on
    borderlines. Reference: Auer 2002. ``"fixed"``: each pair gets ``full_npermutations`` shuffles (legacy behaviour, still available for reproducibility of old runs)."""

    refine_passes: int = 0
    """Coordinate-ascent refinement passes on k-way results. ``0`` (default) disables. ``>0`` runs N passes where each pass tries swapping each member of each
    k-way set with each non-member and keeps the swap if II improves. Catches cases where the greedy seed missed a better neighbor."""

    groups_col: str = ""
    """Name of a column in the input X holding group IDs (e.g. session IDs, user IDs). When set, permutation tests respect group boundaries -- shuffle Y
    values only across groups, not within. Prevents inflated significance from within-group autocorrelation. Reference: Anderson & ter Braak 2003."""

    enable_streaming_cache: bool = False
    """When True, cache marginal MIs and pair II values on the fitted MRMR instance. On subsequent fit() calls with the same cfg, only re-screen columns
    whose data distribution changed beyond a KL threshold. Saves ~70%% on production daily-refresh re-fits. Requires explicit opt-in because the cache
    persistence semantics surface in transform output."""

    streaming_cache_kl_threshold: float = 0.01
    """KL divergence threshold for invalidating cached marginal MIs. Lower = more aggressive recomputation; higher = more reuse."""

    enable_full_conditional_perm: bool = False
    """When True AND ``permutation_null='conditional'``, use full iterative-proportional-fitting (IPF / Deming-Stephan) to generate permutations that preserve
    BOTH marginals (P(X1, Y) and P(X2, Y)) -- the strictest synergy null. Materially more expensive than within-strata shuffle (~5-10x); enable only for high-stakes significance claims."""

    def __post_init__(self):
        """Validate ranges and types at construction time so misconfig fails fast (not deep in ``fit()``)."""
        if self.top_k_pairs <= 0:
            raise ValueError(f"top_k_pairs must be > 0; got {self.top_k_pairs}")
        if self.max_kway_order < 2:
            raise ValueError(f"max_kway_order must be >= 2 (pairs); got {self.max_kway_order}")
        if self.full_npermutations < 0:
            raise ValueError(f"full_npermutations must be >= 0; got {self.full_npermutations}")
        if self.shortlist_npermutations < 0:
            raise ValueError(f"shortlist_npermutations must be >= 0; got {self.shortlist_npermutations}")
        if self.marginal_floor < 0:
            raise ValueError(f"marginal_floor must be >= 0; got {self.marginal_floor}")
        if self.max_combined_nbins is not None and self.max_combined_nbins < 4:
            raise ValueError(f"max_combined_nbins must be >= 4 if set; got {self.max_combined_nbins}")
        if not 0 <= self.min_fold_prevalence <= 1:
            raise ValueError(f"min_fold_prevalence must be in [0, 1]; got {self.min_fold_prevalence}")
        if self.n_folds_stability < 0:
            raise ValueError(f"n_folds_stability must be >= 0 (0 disables); got {self.n_folds_stability}")
        if self.anti_redundancy_beta < 0:
            raise ValueError(f"anti_redundancy_beta must be >= 0; got {self.anti_redundancy_beta}")
        if self.min_n_samples < 2:
            raise ValueError(f"min_n_samples must be >= 2 (MI estimation degenerates below); got {self.min_n_samples}")
        if self.min_class_count < 1:
            raise ValueError(f"min_class_count must be >= 1; got {self.min_class_count}")
        # Cross-field sanity: shortlist_npermutations should not exceed full_npermutations -- the shortlist is supposed to be cheaper.
        if self.shortlist_npermutations > 0 and self.full_npermutations > 0 and self.shortlist_npermutations > self.full_npermutations:
            raise ValueError(
                f"shortlist_npermutations ({self.shortlist_npermutations}) must be <= full_npermutations ({self.full_npermutations}). "
                "Search-phase should be CHEAPER than confirmation."
            )


@dataclass
class CatFEState:
    """Persistence container for all cat-FE artefacts produced during fit.

    Attached as ``self._cat_fe_state_`` on the fitted MRMR. ``None`` until ``cat_fe_config.enable=True`` and the cat-FE step has run. ``__setstate__`` injects
    ``None`` as the BC default for legacy pickles. All fields are picklable / clonable -- no closures, no fitted estimators, no captured numpy views into ``data``.
    """

    recipes: list = field(default_factory=list)
    """List of ``EngineeredRecipe`` objects (kind=``"factorize"`` for cat-FE) that ``MRMR.transform`` replays on test data. Same container as the existing
    ``self._engineered_recipes_`` for numeric FE -- cat-FE recipes simply land alongside them at fit time."""

    diagnostics: dict = field(default_factory=dict)
    """``{engineered_name: dict}`` per-engineered-feature audit trail. Keys: ``II, II_holdout_unbiased, II_oof_std, marginal_X1_MI, marginal_X2_MI, joint_MI,
    joint_nclasses, n_obs_per_cell_p25, fold_stability, joint_dependence_confidence, fwer_corrected_p, parent_pair_if_kway``. Empty when ``emit_diagnostics=False``."""

    search_mis: dict = field(default_factory=dict)
    """``{frozenset(indices): float}`` private cache of search-phase MI values, KEPT SEPARATE from MRMR's shared ``cached_confident_MIs``. Search-phase values
    have ``confidence=0`` (they used ``shortlist_npermutations=0``); promoting them to the shared cache would silently zero downstream ``gain * confidence``
    products. Promoted only after the confirmation phase passes."""

    ii_stability: dict = field(default_factory=dict)
    """``{engineered_name: list[float]}`` per-fold II values when ``n_folds_stability > 0`` (length K). Used by the gate-logic step and mirrored into
    ``diagnostics[name]['ii_per_fold']``."""

    dropped_singleton_nbins: list = field(default_factory=list)
    """Names of categorical columns dropped at entry because ``nbins[i] == 1`` (constant or all-NaN). Surfaced so silently dropped cols don't cause confusion."""

    high_cardinality_warnings: list = field(default_factory=list)
    """Names of columns where ``nbins[i] > sqrt(n) * 2`` are refused. The list is informational; the actual refusal raises ``ValueError`` at fit time."""

    lineage: dict = field(default_factory=dict)
    """``{engineered_col_idx_in_augmented_data: frozenset(parent_idx_in_data)}``. Used by ``screen_predictors`` to skip redundant k-way candidates that combine
    an engineered column with one of its own parents."""

    streaming_cache_out: dict = field(default_factory=dict)
    """Snapshot of per-column signatures + marginal MIs after the current fit. ``MRMR.fit`` reads this and stores it on ``self._cat_fe_cache_`` for use by the
    NEXT ``fit()`` call. Empty when ``enable_streaming_cache=False``."""
