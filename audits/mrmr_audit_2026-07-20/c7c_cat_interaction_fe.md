# Categorical / pairwise / interaction / temporal FE

9 findings, 5 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/_cat_interactions_step.py:348

**Sample weights only affect the cat-FE point-estimate pair-search kernel; every downstream confirmation/rerank step (permutation test, bandit-UCB1, Westfall-Young, MM re-rank, bootstrap CI, K-fold stability, anti-redundancy rerank, k-way greedy expansion) recomputes its statistic UNWEIGHTED, so a weighted II_obs gets tested against an unweighted null/refinement.**

MRMR(cat_fe_config=CatFEConfig(sample_weight_col='w', full_npermutations=50)).fit(X, y) on data where weighting changes which rows dominate: `_pair_search_kernel_weighted_njit` (via `_cat_target_encoding_and_weighted.py`) computes the correct weighted `ii_arr`, but `_confirm_pairs_via_permutation`/`_confirm_pairs_bandit_ucb1` (`_cat_confirm_permutation.py`, `_cat_confirm_bandit.py`) call `merge_vars`/`compute_mi_from_classes`/`_shuffle_and_compute_three_mis` with NO weights parameter at all -- the permutation null is built from the UNWEIGHTED data distribution. A pair whose weighted II is only significant because of the weighting (e.g. weights concentrate mass on a synergy-carrying subset, mirrored in `test_biz_weights_can_recover_synergy_in_subset`) gets compared against a null that never saw that weighting, so the confirmation p-value is statistically meaningless -- it can both spuriously reject a real weighted-synergy pair and spuriously confirm a pair whose II only looked significant under the (wrong) unweighted null. Same gap in `_maybe_rerank_with_mm` (MM re-rank silently un-weights the ranking that determines which pairs get materialized), `_bootstrap_ii_cis`, `_kfold_stability_filter`, `_anti_redundancy_rerank`, and `_greedy_expand_one_seed`/k-way expansion (`_cat_kway_materialize.py`) -- none of these six take a `weights` argument.

### [P1] test_gap -- tests/feature_selection/fe/categorical/test_cat_fe_weighted_and_bootstrap.py:1549

**No test exercises sample weights together with full_npermutations>0 (or bootstrap/k-fold-stability/anti-redundancy/k-way), so the weight-blindness bug in the confirmation pipeline (see the paired bug finding) is completely unexercised.**

`TestWeightedKernelUnit` in this file tests only the isolated `_pair_search_kernel_weighted_njit` (search phase). The end-to-end `run_cat_interaction_step` weighted tests (`test_with_uniform_weights`, `test_with_nonuniform_weights`, lines ~1549-1574 in `tests/feature_selection/fe/categorical/test_cat_interactions_coverage.py`) explicitly set `full_npermutations=0`, disabling the exact confirmation phase where the weight-blindness bug lives. A regression that further breaks weight propagation into confirmation would pass the full suite silently.

### [P2] bug -- src/mlframe/feature_selection/filters/_composite_group_agg_fe.py:519

**`_auto_detect_group_cols` imports its Layer-87 helper via `from .._grouped_agg_fe import ...` (two dots = parent package `mlframe.feature_selection`), but `_grouped_agg_fe.py` only exists as a SIBLING module in `mlframe.feature_selection.filters` (one dot) -- this import always raises ModuleNotFoundError and is silently swallowed by the bare `except Exception`, making the primary code path permanently dead and masked only by an immediately-following correct single-dot fallback import a few lines later.**

Every call to `_auto_detect_group_cols` (e.g. from `hybrid_composite_group_agg_fe`'s auto-detect path) pays a wasted failing-import attempt and silently falls through to the `except Exception: _l87_detect = None` branch, then re-imports correctly via the single-dot fallback at line 528-529. Functionally masked today, but if the fallback block is ever refactored away (or the two import attempts are de-duplicated by someone who assumes the first one is live), `_l87_detect` becomes permanently `None` and `_auto_detect_group_cols` silently drops to the crude positional local heuristic (lines 533-542) with no warning -- a P2 because the bare `except Exception` hides a real, always-firing ImportError rather than a genuine optional-dependency probe.

### [P2] design -- src/mlframe/feature_selection/filters/_temporal_agg_fe.py:727

**None of the temporal (expanding/rolling/lag), grouped-agg, composite-group-agg, grouped-quantile, group-distance, conditional-gate, count/freq, ratio/delta, missingness, integer-lattice, periodic, numeric-decompose, semi-supervised, synergy-detector, or target-encoding FE families in this cluster accept a sample_weight parameter at all -- weighting is entirely a cat_interactions-only concept in this cluster.**

A user fitting `MRMR(sample_weight=w, ...)` with any of these FE families enabled gets the exact same engineered columns/scores regardless of `w` -- the per-group means, expanding stats, MI gates, and target-encoded cells are computed uniformly over all rows. This is consistent (not a divergent code path) but is a silent capability gap: sample weighting that the estimator otherwise honours does not propagate into feature *engineering* for the bulk of this cluster's families, so a use case that legitimately needs weighted aggregation (e.g. downweighting a biased data slice) gets features computed on the wrong effective population with no warning.

### [P2] design -- src/mlframe/feature_selection/filters/_pairwise_modular_resident.py:1

**GPU residency in this cluster is a clean design (no issue found): every resident-path helper (`_pairwise_modular_resident.py`, the GPU-strict candidate paths in `_conditional_gate_fe.py`) is a strict opt-in shim that uploads each operand ONCE via a content-keyed `resident_operand` cache, threads the resident handle through marginal MI + the permutation null, and falls back to the exact byte-identical host path on any cupy fault or when residency is disabled -- there is no unconditional host<->device round-trip per call in the default (non-STRICT) path.**

_(no issues found in this cluster for this angle)_

### [P2] design -- src/mlframe/feature_selection/filters/_cat_confirm_permutation.py:910

**cat-FE's own CPU<->GPU dispatch for the permutation-confirmation kernel is correct and well-tested (no issue found): `_perm_kernel_dispatch_use_gpu` gates on a kernel_tuning_cache-measured crossover, the cupy variant (`_count_nfailed_joint_indep_cupy`) is documented as statistically-equivalent-but-not-bit-identical to the CPU LCG shuffle (different RNG sequence, same uniform distribution), and any GPU exception falls back to CPU with a logged warning -- this is the correct/expected divergence class (RNG-sequence, not formula/threshold), not a parity bug.**

_(no issues found in this cluster for this angle)_

### [P2] test_gap -- src/mlframe/feature_selection/filters/_cat_target_encoding_and_weighted.py:50

**`_compute_target_encoding`'s multi-class (non-binary, non-ordinal) label path is explicitly documented as producing a semantically-wrong 'expected class index' rather than a class probability, but no test exercises/pins that documented limitation with a 3+ class target.**

A caller passes a genuinely nominal 3-class `y` (not ordinal) through `cfg.emit_target_encoding=True`; the emitted `te(...)` column silently encodes `E[class_index | cell]`, a number with no principled meaning for nominal classes, and nothing in the test suite would catch a further regression (e.g. an off-by-one in class-index handling) in that path since it is undertested relative to the binary/regression cases.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_cat_kway_materialize.py:167

**`_greedy_expand_one_seed`'s k-way greedy expansion (the core synergy-seeded triplet/quartet growth) has no dedicated behavioral test in the cluster's test files searched (`test_cat_interactions*.py`, `test_cat_fe_*`) verifying it recovers a genuine 3-way XOR/parity structure the pairwise stage cannot see -- only the pair-level XOR biz_value tests were found.**

A regression in the incremental-II acceptance logic (`inc_ii > best_inc_ii` / `best_inc_ii < min_inc_ii` early-stop) or in `_merge_vars_sorted_insert`'s prefix-splicing could silently degrade k-way expansion to never accepting a genuine 3-way parity target, and the existing pair-focused test suite would not detect it (k-way is opt-in via `max_kway_order>2`, and the located tests default to pairs-only).

### [P2] design -- src/mlframe/feature_selection/filters/_conditional_gate_fe.py:63

**Multiple bare `except Exception: return False`/`return None` GPU-strict-flag probes (e.g. lines 276, 355 here; also `_pairwise_modular_fe.py` lines 79/139/222/284/338) swallow ANY exception -- including a genuine bug in the imported flag function -- with zero logging, unlike sibling probes in the same files that do `logger.debug(..., exc_info=True)` before falling back.**

If `fe_gpu_strict_resident_enabled` or `fe_gpu_strict_bytematch_enabled` (imported from `._gpu_strict_fe`) ever raises due to an unrelated bug (e.g. a typo introduced in a future edit, or an env-var parsing exception), these call sites silently and permanently disable the GPU-resident fast path with no trace in logs -- a developer chasing 'why does STRICT residency never engage on this host' has no log line to find, unlike the fully-instrumented siblings a few lines below that do log.

## Proposals

### (coverage_gap) Add a weighted + full_npermutations>0 end-to-end regression test for cat-FE confirmation

Add a test that runs `run_cat_interaction_step` with non-uniform `weights` AND `full_npermutations>0` (and/or `bootstrap_ci_n_replicates>0`, `n_folds_stability>0`) on the signal-in-subset XOR fixture already used in `test_biz_weights_can_recover_synergy_in_subset`, asserting the surfaced `joint_dependence_confidence` reflects the WEIGHTED signal (currently it can't, since confirmation is unweighted) -- this both documents the current gap and will pin any future fix.

### (edge_case) Thread sample weights through cat-FE confirmation, MM re-rank, bootstrap CI, K-fold stability, anti-redundancy, and k-way expansion

Add an optional `weights` param to `_confirm_pairs_via_permutation`, `_confirm_pairs_bandit_ucb1`, `_compute_westfall_young_corrected_p`, `_maybe_rerank_with_mm`, `_bootstrap_ii_cis`, `_kfold_stability_filter`, `_anti_redundancy_rerank`, and `_greedy_expand_one_seed`, using a weighted `merge_vars`/`compute_mi_from_classes` equivalent (the weighted joint-histogram formula already exists in `_pair_search_kernel_weighted_njit` and can be generalized) so a weighted fit is weighted consistently end-to-end, not just at the point-estimate search stage.

### (other) Fix the wrong-depth relative import in _composite_group_agg_fe._auto_detect_group_cols

Change `from .._grouped_agg_fe import _auto_detect_group_cols as _l87_detect` (line 519) to `from ._grouped_agg_fe import ...` (single dot, matching the correct fallback at line 529), and collapse the now-redundant duplicate try/except into one import -- removes dead code and a wasted failing import on every call.

### (coverage_gap) Add a genuine nominal-3-class target-encoding test documenting the known expected-class-index limitation

A small test with a 3-class nominal `y` through `_compute_target_encoding` (or `kfold_target_encode_fit` with `stats=('mean',)`) asserting the emitted value equals the documented `E[class_index | cell]` semantics, so a future refactor that silently 'fixes' this into per-class probabilities (a behavior change) is caught, or a future regression that further breaks it is caught.

### (coverage_gap) Add a k-way (3-way XOR/parity) biz_value test for _greedy_expand_one_seed

A biz_value test with `max_kway_order=3` and a `y = x1 XOR x2 XOR x3` fixture (all pairwise IIs ~0, only the triple carries signal) asserting the k-way materialized column recovers the parity (measured MI on the materialized column well above the pair-only baseline) -- currently only pair-level XOR is covered per the searched test files.
