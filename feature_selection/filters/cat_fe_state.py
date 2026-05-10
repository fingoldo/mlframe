"""Configuration and persistence dataclasses for cat-FE.

Why two dataclasses, not 14 ``MRMR.__init__`` kwargs:

- ``CatFEConfig`` packages every cat-FE knob into a single attr on the
  estimator. ``MRMR(cat_fe_config=CatFEConfig(enable=True, ...))``
  replaces fourteen parallel ``cat_fe_*`` kwargs that would otherwise
  bloat the init surface from 55 to 69. Once shipped flat, those
  fourteen names are forever pinned by sklearn's ``_get_param_names``
  introspection; folding them into a config object now is the only
  reversible move.
- ``CatFEState`` packages every cat-FE persistence attribute (recipes,
  diagnostics, search-phase MI cache) under one ``self._cat_fe_state_``
  field. ``__setstate__`` injects a single default; ``clone()`` survives
  a single round-trip; future additions land inside the dataclass
  without bloating ``MRMR.__dict__``.

Defaults are chosen per the cat-FE design plan v3:

- ``enable=False``                -- opt-in; legacy behaviour bit-exact.
- ``include_numeric=False``        -- SM9: F12 trap fix, discretized
   noisy floats produce spurious aliasing. User opts in when domain
   knowledge supports mixing.
- ``full_npermutations=100``       -- SB4: zero is an anti-statistical
   trap (no FWER guarantee at all). 100 is the cheapest non-zero default
   that gives a usable α=0.01 floor with WY correction.
- ``fwer_correction="westfall_young"`` -- SB2: top-K from a 4950-pair
   search family without correction has FWER ≈ 1.0. WY is bundled
   because it amortises shuffles across all pairs.
- ``permutation_null="joint_independence"`` -- SB1: matches what shuffle-Y
   actually tests; the misnamed "synergy null" stays a documented
   limitation, addressable later via conditional permutation.
- ``select_on="synergy"``          -- SM1: matches the user's stated
   goal; redundancy / absolute are explicit alternatives.
- ``min_interaction_information=None`` -- resolved at fit time to
   ``-3 / sqrt(n_samples)`` (B4: small-negative absorbs finite-sample
   noise around the synergy boundary).
- ``backend="auto"``               -- P9: GPU when N≥200 AND n≥500k AND
   CuPy available, else CPU prange.
- Everything else off / zero / sane minimum.

Reference: ``C:\\Users\\TheLocalCommander\\.claude\\plans\\linear-shimmying-thimble.md``
SB1-SB10, SM1-SM14, SD1-SD33.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CatFEConfig:
    """All cat-FE configuration in one place.

    Attached to an MRMR instance via the ``cat_fe_config`` constructor
    kwarg: ``MRMR(cat_fe_config=CatFEConfig(enable=True, top_k_pairs=128))``.
    """

    # ----- core (8) -----
    enable: bool = False
    """Master switch. ``False`` -- legacy MRMR behaviour, no cat-FE step
    runs. ``True`` -- the step runs after categorical detection, before
    screening (see plan §"Точка интеграции")."""

    marginal_floor: float = 0.0
    """Drop categorical columns with ``MI(X_i; Y) < marginal_floor`` from
    the pair-search pool. ``0.0`` keeps everything (default). Set to
    e.g. ``0.05`` to prune low-signal columns before the N² pair search."""

    max_combined_nbins: Optional[int] = None
    """Hard cap on ``nbins[i] * nbins[j]`` for any pair candidate. ``None``
    resolves at fit time to the data-aware Paninski-derived ceiling
    (SM5: ``max(4, n * 0.05 / 3 + 1)``). Capped at 10**7 absolute (SB10
    F18) to prevent OOM via ``cat_fe_max_combined_nbins=10**9``-style
    misconfig."""

    top_k_pairs: int = 64
    """How many top pairs (by II / synergy score) to keep after
    argpartition over all candidate pairs. Memory linear in this value;
    the recommended default (64) balances bench time with finding rare
    synergies. See plan §"Memory model"."""

    max_kway_order: int = 2
    """Max k for k-way greedy expansion. ``2`` = pairs only;
    ``3..interactions_max_order`` = greedy triplet / quartet expansion
    from the top pairs. NOT capped by ``MRMR.interactions_max_order`` --
    engineered cols are 1-way features to screening (B7)."""

    min_interaction_information: Optional[float] = None
    """Floor for accepting a pair / k-way tuple. ``None`` resolves at fit
    time to ``-3 / sqrt(n_samples)`` (B4 finite-sample noise margin).
    Renamed from ``min_jakulin_ii`` per SB5 (the value is interaction
    information = synergy minus redundancy, not pure synergy)."""

    include_numeric: bool = False
    """Mix discretized numeric columns into the cat pool. Default ``False``
    per SM9 / F12: noisy floats produce spurious aliasing interactions.
    Opt-in only when domain knowledge supports it."""

    shortlist_npermutations: int = 0
    """Permutations for the search-phase point estimate. ``0`` skips
    permutation entirely during search; surviving pairs get a separate
    confirmation pass with ``full_npermutations``."""

    # ----- statistical rigor (4) -----
    full_npermutations: int = 100
    """Permutations for the confirmation phase on top-K survivors. ``0``
    is an anti-statistical trap (SB4) -- ``MRMR(cat_fe_config=
    CatFEConfig(enable=True))`` with full_npermutations=0 would surface
    pairs ranked purely by point estimate with full selection bias.
    Default 100 is the cheapest non-zero floor; bump to 1000 for
    α=0.001-grade FWER under WY correction."""

    fwer_correction: Literal["none", "bonferroni", "bh_fdr", "westfall_young"] = "westfall_young"
    """Multiple-testing correction across the search-phase ``N(N-1)/2``
    candidate pairs (SB2). WY is bundled because the same shuffle-Y
    cycle that does confirmation can compute the per-shuffle max-II
    over all pairs -- correction is amortised. ``none`` for legacy
    behaviour (FWER ≈ 1.0)."""

    permutation_null: Literal["joint_independence", "conditional"] = "joint_independence"
    """Null hypothesis for the permutation test (SB1). Default
    ``joint_independence`` (shuffle Y) tests ``I(X1,X2;Y) = 0``, NOT
    synergy specifically -- the surfaced ``confidence`` field is named
    ``joint_dependence_confidence`` to make this honest. ``conditional``
    (shuffle X2 within strata of Y, future) approaches the synergy null
    but at higher computational cost."""

    use_miller_madow: Optional[bool] = None
    """Apply Miller-Madow bias correction to ALL six entropies in the
    II expansion as a unit (SB6). ``None`` auto-gates on
    ``(a*b*c) / n > 0.05``. Force ``True`` for high-cardinality joints,
    ``False`` for legacy plug-in behaviour."""

    # ----- modeling choice (1) -----
    select_on: Literal["synergy", "redundancy", "absolute"] = "synergy"
    """Sign of interaction information to select on (SM1). ``synergy``
    keeps positive II (the canonical use case); ``redundancy`` keeps
    negative II (useful for noise-robust models that benefit from
    correlated parents); ``absolute`` keeps |II| > floor (any strong
    joint structure)."""

    # ----- backend / parallelism (1) -----
    backend: Literal["auto", "cpu", "gpu"] = "auto"
    """Compute backend (P9). ``auto`` picks CPU prange below
    ``N=200, n=500_000``, GPU above (when CuPy is installed). ``cpu``
    forces numba prange. ``gpu`` requires CuPy and raises if unavailable."""

    # ----- stability / replication (3) -----
    n_folds_stability: int = 0
    """K-fold stability filter (E6). ``0`` disables. ``>0`` runs the pair
    search on each of K folds and keeps only pairs prevalent in
    ``min_fold_prevalence × K`` folds."""

    min_fold_prevalence: float = 0.6
    """When ``n_folds_stability > 0``, fraction of folds in which a pair
    must clear ``min_interaction_information`` to be kept."""

    # ----- ranking coupling (1) -----
    anti_redundancy_beta: float = 0.0
    """mRMR-coupled scoring (E3). ``0`` = pure II ranking. ``> 0`` =
    ``score = II - β * max_z I(merged; Z)`` over already-selected
    features. Heuristic per SM6, NOT a derived mRMR criterion -- use
    the two-stage decoupled path (II gate then mRMR rank) for
    formally principled coupling."""

    # ----- diagnostics (1) -----
    emit_diagnostics: bool = True
    """Populate ``self._cat_fe_state_.diagnostics`` with per-pair
    ``(II, marginal_X1_MI, marginal_X2_MI, joint_MI, n_uniq, n_obs_per_cell_p25)``
    so users can debug "why was this pair selected?" (E4)."""

    # ----- gate combination (1) -----
    gate_logic: Literal["and", "or", "permutation_primary"] = "and"
    """When both K-fold stability and permutation confidence are
    enabled, how to combine the gates (SM3 / S9). ``and`` (default,
    strictest) requires both; ``or`` requires either; ``permutation_primary``
    uses permutation as primary with K-fold as a tiebreaker."""

    # ----- transform contract (1) -----
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip"
    """How ``transform()`` handles test-time category values that were
    not seen during fit (B2). ``clip`` (default): cap at the highest
    trained bin. ``sentinel``: dedicated unknown bin (inflates
    cardinality). ``raise``: hard error."""

    # ----- pathological-input gates (3) -----
    min_n_samples: int = 200
    """Hard floor on n -- below this, II is dominated by sample-size
    bias and permutation tests have insufficient unique reorderings
    (F4: ``n=2`` has only 2 distinct shuffles)."""

    min_class_count: int = 50
    """Minimum samples per class of Y. Below this, marginal MI estimates
    on cat columns are dominated by which rows happen to be the
    minority (F5: 5 positives in 100k means each (X1,X2,Y=1) cell has
    0 or 1 obs). Cat-FE warns and disables when violated."""

    discretization_audit: bool = False
    """Re-run pair search with ``nbins ± 2`` for numeric columns and
    record per-pair II range (SD2). Surfaces pairs whose ranking is
    sensitive to the discretization scheme."""


@dataclass
class CatFEState:
    """Persistence container for all cat-FE artefacts produced during fit.

    Attached as ``self._cat_fe_state_`` on the fitted MRMR. ``None``
    until ``cat_fe_config.enable=True`` and the cat-FE step has run.
    ``__setstate__`` injects ``None`` as the BC default for legacy
    pickles.

    All fields are picklable / clonable -- no closures, no fitted
    estimators, no captured numpy views into ``data``.
    """

    recipes: list = field(default_factory=list)
    """List of ``EngineeredRecipe`` objects (kind=``"factorize"`` for
    cat-FE) that ``MRMR.transform`` replays on test data. Same
    container as the existing ``self._engineered_recipes_`` for
    numeric FE -- cat-FE recipes simply land in this list alongside
    them at fit time."""

    diagnostics: dict = field(default_factory=dict)
    """``{engineered_name: dict}`` -- per-engineered-feature audit trail.
    Keys per SD7: ``II, II_holdout_unbiased, II_oof_std, marginal_X1_MI,
    marginal_X2_MI, joint_MI, joint_nclasses, n_obs_per_cell_p25,
    fold_stability, joint_dependence_confidence, fwer_corrected_p,
    parent_pair_if_kway``. Empty when ``emit_diagnostics=False``."""

    search_mis: dict = field(default_factory=dict)
    """``{frozenset(indices): float}`` -- private cache of search-phase
    MI values, KEPT SEPARATE from MRMR's shared ``cached_confident_MIs``
    (B1). Search-phase values have ``confidence=0`` because they used
    ``shortlist_npermutations=0``; promoting them to the shared cache
    would silently zero downstream gain * confidence products.
    Promoted to the shared cache only after confirmation phase passes."""

    ii_stability: dict = field(default_factory=dict)
    """``{engineered_name: list[float]}`` -- per-fold II values when
    ``n_folds_stability > 0``. Length K. Used by the gate-logic step
    and mirrored into ``diagnostics[name]['ii_per_fold']``."""

    dropped_singleton_nbins: list = field(default_factory=list)
    """Names of categorical columns dropped at entry because
    ``nbins[i] == 1`` (constant or all-NaN, F1/F2). Surfaced for
    debugging -- silently dropped cols cause confusion later when a
    user asks "why didn't pair (X, anything) appear in diagnostics?"."""

    high_cardinality_warnings: list = field(default_factory=list)
    """Names of columns where ``nbins[i] > sqrt(n) * 2`` -- refused per
    SM8 / F8. The list is informational; the actual refusal raises
    ``ValueError`` at fit time."""
