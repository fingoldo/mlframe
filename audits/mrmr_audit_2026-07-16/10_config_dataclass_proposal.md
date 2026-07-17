# MRMR constructor config-dataclass proposal (finding #1)

**Design doc only -- no code in this pass.** Per the user's explicit instruction, the actual
dataclass migration will be implemented separately, by the user, at the very end of the whole
audit. This document captures the proposed design so it isn't lost or re-derived later.

## Problem

`MRMR.__init__` is ~2600 lines with 250+ parameters. Findings #1 (this doc) and #2 (fit()
decomposition, done separately in this pass) are the two structural fixes the audit flagged for
this file. The parameter list has organically grown by feature area (quantization, FE hybrid-orth,
DCD, stability selection, group-aware MI, synergy/redundancy knobs, fast-search profile, ...) --
each addition was locally reasonable, but the aggregate is now a 250+-argument flat namespace with
no structural grouping beyond `#`-comments in the signature.

## Precedent already in the codebase

`CatFEConfig` (`src/mlframe/feature_selection/filters/cat_fe_state.py`) already does exactly this
pattern for the categorical-FE knob cluster: a dataclass passed as `cat_fe_config: Optional[CatFEConfig] = None`,
with `MRMR.__init__` accepting the nested object directly (not absorbing its fields as flat kwargs).
That precedent is the shape to generalize -- not a novel design.

## Proposed dataclass groupings

Derived from the actual `__init__` signature's own `#`-comment groupings (as of this audit pass,
commit `e9a95c662`). Each becomes a `@dataclass` (pydantic-validated, see below), one per cohesive
knob-cluster:

### `FastSearchConfig`
The `fe_fast_search`-gated sub-knobs already centralized in `MRMR._FAST_SEARCH_OVERRIDES` (the
class-level override table `_apply_fast_search_profile` reads). Fields: `fe_fast_search: bool`,
plus whatever profile-specific overrides `_FAST_SEARCH_OVERRIDES` currently lists (grep that table
at migration time -- it's already the single source of truth for which knobs this cluster touches,
so the dataclass fields should mirror it exactly rather than being re-derived from the signature).

### `StabilitySelectionConfig`
`stability_selection_method: str = "classic"`, `stability_selection_corr_threshold: float = 0.8`,
`stability_n_bootstrap: int = 50`, `stability_pi_threshold: float = 0.6`.

### `SynergyRedundancyConfig`
The cross-cutting redundancy/synergy research knobs (findings A1-F14 in the signature comments):
`redundancy_aggregator: str | None = None`, `bur_lambda: float = 0.0`,
`relaxmrmr_alpha: float = 0.0`, `cmi_perm_stop: bool = False`, `cmi_perm_n_permutations: int = 100`,
`cmi_perm_alpha: float = 0.05`, `cpt_test: bool = False`, `cpt_n_permutations: int = 200`,
`pid_synergy_bonus: float = 0.0`, `uaed_auto_size: bool = False`, `mi_correction: str = "none"`,
`mi_normalization: str = "none"`.

### `GroupAwareConfig`
`group_aware_mi: bool = False`, `strict_groups: bool = True` (post finding #20),
`group_mi_min_rows: int`, `group_mi_aggregate: str`.

### `DCDConfig` (Denoised Cluster-Aggregate)
`dcd_enable: bool`, `dcd_distance`, `dcd_min_cluster_size`, `dcd_max_cluster_size`,
`dcd_cluster_size_threshold`, `dcd_pairwise_cache_max`, `dcd_swap_method`, `dcd_swap_alpha`,
`dcd_swap_gain_threshold`, `dcd_swap_npermutations`, `dcd_tau_calibration_n_pairs`,
`dcd_tau_calibration_seed`, `dcd_postoc_compose`.

### `HybridOrthConfig` (single largest cluster, ~60 knobs)
Everything prefixed `fe_hybrid_orth_*`. Grep-confirmed fields as of this pass include (non-
exhaustive naming, but a complete grep-based field list at migration time is required since this
cluster is still growing): `fe_hybrid_orth_enable`, `_basis`, `_degrees`, `_top_k`,
`_default_scorer`, `_extra_bases`, `_spline_knots`, `_fourier_freqs`, `_fourier_powers`,
`_pair_enable` / `_pair_max_degree`, `_triplet_enable` / `_triplet_max_degree` / `_triplet_seed_k` /
`_triplet_top_count`, `_quadruplet_*` (mirrors triplet), `_adaptive_arity_*` (5 fields),
`_adaptive_degree_*` (3 fields), `_diff_basis_*` (4 fields), `_conditional_routing_*` (4 fields),
`_cluster_basis_*` (4 fields), `_three_gate_*` (3 fields), per-scorer enable/param pairs
(`_ksg_*`, `_copula_*`, `_dcor_*`, `_hsic_*`, `_cmim_*`, `_jmim_*`, `_tc_*`), `_ensemble_enable` /
`_ensemble_scorers` / `_ensemble_aggregator`, `_auto_scorer_enable` / `_auto_scorer_n_boot`,
`_meta_enable` / `_meta_force_scorer`, `_bootstrap_enable` / `_bootstrap_n_boot` /
`_bootstrap_sample_fraction`, `_elasticnet_*` / `_lasso_*`.

This cluster is large enough that it may warrant its OWN sub-grouping (e.g. a nested
`HybridOrthScorersConfig` for the per-scorer enable/param pairs) rather than one flat 60-field
dataclass -- a call the user should make at migration time once the full field list is
re-confirmed against the then-current signature.

### Remaining top-level knobs (not obviously clustered)
`quantization_method`, `quantization_nbins`, `quantization_dtype`, `max_categorical_cardinality`,
`nbins_strategy`, `nbins_strategy_kwargs`, `adaptive_nbins_large_n_reg*` (3 fields),
`nan_strategy`, `factors_names_to_use`, `factors_to_use`, `mrmr_relevance_algo`,
`mrmr_redundancy_algo`, `reduce_gain_on_subelement_chosen`, `use_simple_mode`,
`run_additional_rfecv_minutes`, `additional_rfecv_selection_rule`, `additional_rfecv_kwargs`,
`extra_x_shuffling`, `dtype`, `random_seed` / `random_state`, `use_gpu`, `n_workers`, `n_jobs`,
`min_occupancy`, `min_nonzero_confidence`, `full_npermutations`, `baseline_npermutations`, `cv`,
`cv_shuffle`, `verbose`, `mrmr_identity_cache_ycorr_threshold`,
`skip_retraining_on_same_content`, `min_features_fallback`, `min_relevance_gain*`,
`fe_max_steps`, `fe_*_enable` flags not covered above (count/frequency encoding, cat-num
interaction, missingness, ratio, grouped delta/lagged-diff/agg/composite/quantile, cat pair/triple,
kfold TE, conditional dispersion, wavelet, univariate basis/fourier), `additional_rfecv_kwargs`,
`retain_artifacts`, `parallel_kwargs`, `ndigits`, `cluster_aggregate_enable` /
`cluster_aggregate_mode`, `build_friend_graph`. These stay as either flat kwargs (if genuinely
few/orthogonal) or get folded into a `QuantizationConfig` / `FEMasterConfig` grouping -- a
secondary pass once the primary 5 clusters above are migrated, since they're lower-value (fewer
knobs each, less internal coupling) than the big five.

## Pydantic validation approach

Each dataclass becomes (or wraps) a `pydantic.BaseModel` so construction-time validation replaces
today's ad hoc, late `_validate_string_params` pass (which only fires at `fit()` time, not at
`MRMR(...)` construction time -- a typo'd enum value currently isn't caught until the first
`.fit()` call, sometimes minutes into a long-running pipeline). Concretely:

- Literal-typed fields (`redundancy_aggregator: Literal[None, "jmim", "auto"]`,
  `stability_selection_method: Literal["classic", "cluster", "complementary_pairs"]`, etc.)
  replace today's runtime `if x not in (...): raise ValueError(...)` checks scattered through
  `fit()` -- pydantic raises `ValidationError` at `Config(...)` construction time instead.
- Numeric range constraints (`bur_lambda: float = Field(0.0, ge=0.0)`,
  `stability_pi_threshold: float = Field(0.6, gt=0.0, lt=1.0)`) replace the handful of inline
  range asserts currently living inside `fit()`.
- `model_config = ConfigDict(frozen=True)` per config dataclass, matching sklearn's expectation
  that a constructor param is immutable after construction (mutation should go through
  `sklearn.base.clone()` + a new config object, not in-place attribute writes) -- EXCEPT where
  MRMR's own fit()-time override/restore mechanism (`_apply_fast_search_profile`,
  `_apply_default_screen_subsample`) needs to temporarily mutate a field for the fit's duration;
  those two call sites will need either a documented `frozen=False` carve-out for
  `FastSearchConfig` specifically, or a copy-on-override pattern (`config.model_copy(update=...)`)
  -- a concrete decision the user should make once implementing, since it affects whether the
  existing override/restore save-dict pattern (finding #21, already consolidated into
  `_override_if_at_default` this pass) needs to change shape too.

## Flat-kwarg backward-compatibility shim

Every existing caller (production code, ~50+ test files, benchmarks) constructs `MRMR(fe_max_steps=3,
dcd_enable=False, ...)` with flat kwargs today. Breaking this in one shot is not acceptable per
project convention (a migration this wide needs a deprecation path, not a hard break). Proposed
shim:

```python
def __init__(self, *, fast_search_config: FastSearchConfig | None = None,
             stability_config: StabilitySelectionConfig | None = None,
             synergy_config: SynergyRedundancyConfig | None = None,
             group_aware_config: GroupAwareConfig | None = None,
             dcd_config: DCDConfig | None = None,
             hybrid_orth_config: HybridOrthConfig | None = None,
             **deprecated_flat_kwargs):
    # 1. Build each *_config from its dataclass default when None.
    # 2. For every key in deprecated_flat_kwargs that belongs to one of the above clusters,
    #    override the corresponding config field, emit ONE DeprecationWarning per call (not
    #    per-field) naming every flat kwarg used and its replacement nested-config path, e.g.:
    #    "MRMR(dcd_enable=False) is deprecated; pass dcd_config=DCDConfig(enable=False) instead."
    # 3. Any deprecated_flat_kwargs key that does NOT belong to a clustered dataclass is a
    #    genuine unclustered top-level param (see "Remaining top-level knobs" above) and is
    #    accepted directly, unchanged, no warning.
    # 4. store_params_in_object still receives the FULL flat parameter set (both the nested
    #    configs' expanded fields AND the unclustered params) so sklearn's get_params()/clone()
    #    contract (every ctor param independently gettable/settable) keeps working without
    #    requiring callers to know about the new nested shape.
```

This mirrors the reasoning already established for `skip_retraining_on_same_shape`'s removal
(finding #15, this pass): a deprecated flat name is accepted, mapped onto the new mechanism, and
warns -- but unlike that removal, THIS migration should keep the flat-kwarg path indefinitely
(or for a long deprecation window) given its enormous blast radius, rather than removing it
outright the way the single-parameter alias was.

## Migration risk / sequencing notes (restated from the plan)

- This is the single highest blast-radius change in the whole audit: every constructor call site
  (production code, ~50+ test files, benchmarks, `_ctor_defaults()`/`__setstate__` reflection,
  `get_params()`/`clone()` sklearn contract) is affected simultaneously.
- It should be done LAST, after every other finding in this audit has landed and is green, so it's
  the only thing touching this region by the time it happens (no concurrent moving-target risk
  from smaller fixes still in flight).
- Finding #22 (ctor-default-resolution unification, done this pass) is a prerequisite completed
  ahead of time: `_ctor_defaults()` is now the single cached source of truth for every ctor
  default, which is exactly the "one clean place to plug into" hook this migration needs instead
  of the three-places-duplicated logic that existed before this pass.
- `_SETSTATE_LEGACY_OVERRIDES` (the documented legacy-pickle exception list) will need each
  cluster's affected keys re-verified once the dataclasses land -- a legacy pickle predating the
  dataclass migration will still carry flat attributes in its `__dict__`, so `__setstate__`'s
  existing "inject default for missing/renamed attrs" mechanism should keep working unchanged,
  but this needs an explicit pickle-round-trip test once implemented (old flat pickle -> new
  nested-config-capable class -> re-pickle -> reload).
