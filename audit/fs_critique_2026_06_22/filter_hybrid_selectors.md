# Filter / Hybrid / Voting feature-SELECTION audit (2026-06-22)

Scope: `feature_selection/{mi,importance,hetero_vote,hybrid_selector,structure_discovery,optbinning,pre_screen,general,compare_selectors}.py`
plus core selection scorers `filters/{fleuret,friend_graph,_jmim_scorer,screen,pre_screen,permutation,stability}.py`.
MRMR / FE modules explicitly out of scope.

## 1. Findings table (correctness / edge / silent-error)

| # | file:line | sev | issue | disposition |
|---|---|---|---|---|
| S1 | filters/stability.py:123 | HIGH | `np.asarray(est.support_, dtype=np.int64)` coerces a **boolean mask** (sklearn `SelectorMixin.get_support()` default, which the docstring says is supported) to `[1,0,1,...]`, so `counts[sup]` increments only positions 0/1 -> selection collapses to columns {0,1}. Silent numeric-coercion bug. | RESOLVED (added `_support_to_indices`) |
| S3 | filters/stability.py:144 | MED | `selection_probabilities_ >= support_threshold` compares float `counts/n` against float threshold; `12/20 != 0.6` in float64 so a feature selected in exactly threshold*n runs can spuriously fail. | RESOLVED (integer `counts >= ceil(thr*n)` gate; sentinel semantics 0.0/>1.0 preserved) |
| S2 | filters/stability.py:118 | MED | `local_rng.choice(..., replace=False)` subsample is not stratified; on rare/imbalanced y a bootstrap can be single-class -> degenerate inner MI fit silently folded into counts (per `rare_imbalance_needs_large_n`). | FUTURE (add `stratify=` option; needs a y-class-aware sampler, larger change) |
| S5 | filters/stability.py:106 | LOW | `max(2,int(round(frac*n)))` silently overrides a tiny user `sample_fraction` with no log. | DOC (documented in comment; acceptable) |
| P1 | filters/permutation.py:135,200,265,373,426 | MED | exceedance uses `mi_perm >= original_mi` (`>=`, tie counts as failure); on discrete/low-card data ties are frequent -> `nfailed` biased up -> confidence biased down -> weak-but-real signal forced to 0. Standard p-value uses `>=` WITH a +1 add-one which is absent here. | FUTURE (changing the comparator is selection-altering across the whole MRMR confidence path; needs a multi-seed bench before flipping, out of scope for a safe pass) |
| P2 | filters/permutation.py:787 | MED | early-break (`nfailed>=max_failed`) leaves `nchecked<npermutations`; `1-nfailed/(i+1)` over-states failure rate vs the full-budget path -> path-dependent confidence. | FUTURE (intertwined with P1; same bench gate) |
| P4 | filters/permutation.py:45 | LOW | `distribute_permutations` divides by `n_workers` with no `<1` guard. Only called under `n_workers>1`; it is `@njit` so a Python guard is awkward. | REJECTED (not reachable; njit) |
| F3 | filters/friend_graph.py:428 | MED | `red` garbage-flag threshold `garbage_unique_ratio*rel[i]` uses raw plug-in `I(X;Y)` (no finite-sample debias) on one side vs 2-var-inflated `total_unique` on the other -> bias-mismatched, can flip which features `prune_by_friend_graph` removes on small n. | FUTURE (debias both sides consistently; changes pruning decisions, needs bench) |
| F4 | filters/friend_graph.py:234 | MED | `cmi = max(0,joint-rel_i)` clamp silently zeroes noisy-but-real CMI; a red node with all neighbors clamped is flagged-but-unprunable (inconsistent state). | DOC |
| F6 | filters/friend_graph.py:339,257 | MED | bare `except Exception` on GPU dispatch / layout logs at debug only -> a real shape bug is indistinguishable from "GPU absent". | FUTURE (narrow except / warn on unexpected type) |
| F1/F2/F5/F7 | filters/friend_graph.py | LOW | entropy-cache key is index-only (stale if reused across SU/MM modes); arrow-direction tie biased to higher index; `h_max` all-zero fallback silent; O(k^2) python edge loop (gated by max_nodes). | DOC / FUTURE |
| H1 | hetero_vote.py:139 | LOW | `np.percentile(imp[P:], percentile)` with default `percentile=100.0` -> threshold = max shadow; a feature must STRICTLY exceed the single max shadow (`>`), which is the documented high-precision behaviour, but the all-equal-importance edge (e.g. constant X) makes every feature fail silently. Documented positioning. | DOC |
| H2 | hetero_vote.py:42 | LOW | permutation-importance fallback subsamples to 1000 rows with `np.random.default_rng(random_state)` but ignores `sample_weight`. No sample_weight anywhere in the voting path. | DOC (no sample_weight contract) |
| MI1 | mi.py:69-108 (grok) / 156 (chatgpt) | LOW | `fastmath=True` MI kernels assume finite, non-negative bin codes; chatgpt path validates `[0,127]` before int8 cast, grok/deepseek paths do NOT (they require pre-binned int8 by contract). Constant column -> single-bin histogram -> MI=0 (correct). Empty data guarded (grok:94, deepseek:244). | DOC (contract: pre-binned int8) |
| G1 | general.py:222 | LOW | `current_permuted_mis >= original_mi_results` same `>=`-tie family as P1 for the EFS relevancy gate; `nanmax`/`nanquantile` already guard NaN. | DOC |
| C1 | compare_selectors.py:221 | LOW | bare `except Exception` on `selector.fit` records a skip note (intentional: "unavailable dep / GPU-only / fit failure") — visible in `skipped`, not silent. | REJECTED (by design, surfaced) |
| O1 | optbinning.py:65 | LOW | `features.head().select_dtypes("category")` infers category cols from only the first 5 rows; a category dtype is a column-level attribute so `.head()` is harmless, but `nocat_cols.remove(col)` raises if a name is duplicated. | DOC |

## 2. Optimizations

No measured >=0.5% speedup was applied this pass. Candidates evaluated:

- `corr_clusters` (hybrid_selector.py): already carries a vectorized adjacency path + an O(n*p) blocked wide-data path with committed cProfile/tracemalloc numbers (3.83x on the mostly-singleton frame). No further safe win.
- `mi.py` three kernels: intentionally retained (cross-validation of summation order); the public dispatcher is benched at import in `general.benchmark_mi_algos`. Not touched.
- `_jmim_scorer._joint_mi_3d_njit`: allocates a fresh `(K_x*K_z, K_y)` cube per (candidate, selected) pair. A scratch-reuse variant is possible but the cube size varies per pair (K_z differs), so a pooled buffer needs max-size pre-alloc + reset; deferred (no profile evidence it is hot in the JMIM opt-in path).
- stability.py changes are strict correctness (bool->indices, integer threshold); the integer compare is marginally cheaper than the float divide+compare but not measurably so.

## 3. Strengths / weaknesses verdict (when each selector wins / fails)

| selector | best regime | worst regime | redundancy handling | runtime |
|---|---|---|---|---|
| **MI filters** (mi.py grok/chatgpt/deepseek, general.efs) | wide frames, marginal signal, cheap pre-screen; permutation-null gate gives 0-FP on noise | interaction-only signal (XOR / synergy): marginal MI ~0 -> misses operands entirely | none (univariate); pairs with EFS permutation test only | fastest (njit, 1M x 200 in benchmark) |
| **importance.py** (perm / SHAP / tree-gain) | nonlinear main effects a tree captures; honest held-out perm-FI | correlated groups (splits importance among collinear copies); needs a model fit | implicit only (tree splits) | medium (one model fit + permute) |
| **hetero_vote** (cross-model shadow) | HIGH-PRECISION / parsimony, interpretable compact set (accepted-noise ~0) | recall: drops weak-relevant features (structural to cross-model AGREEMENT); mean AUC ~0.74 trails single Boruta ~0.76 | none explicit; majority vote denoises model-specific leaks | slow (panel x trials x fits) |
| **hybrid_selector** | the all-rounder: shares MI/perm-FI/clusters once, votes cluster-aware; recovers interaction signal via tree co-occurrence products + MRMR FE | classification-only (raises on regression / multilabel); heaviest single object | explicit corr-clustering + cluster-aware vote (best of the set) | slowest (MRMR + shap + boruta + tree members) |
| **JMIM / fleuret** (CMIM aggregator) | fleuret: redundancy-controlled greedy, fast default; JMIM (opt-in): multi-collinear noisy-reflection groups + synergy | fleuret: rejects synergistic features (gain<0); JMIM: over-selects on additive redundancy (precision 0.67->0.33) | best (conditional / joint MI aggregator) | fleuret fast; JMIM 3-D cube heavier |
| **friend_graph** | post-hoc diagnostic of selected-set structure (sink/aggregator detection, arrow direction) | small-n bias in garbage/red flag (F3); not a primary selector | pairwise MI graph + CMI sink criterion | medium (k^2 pairs, gated) |
| **structure_discovery** | EDA surface for DISCRETE structure (gcd / mod / argmax / gate) that correlation + tree-FI cannot express; 0-FP inherited from detectors' perm-null | smooth/continuous signal (returns empty by design); integer-eligible cols only | inherits detector gates | cheap (bounded scans, ~1-2s @ n=2k) |
| **optbinning** | IV-based supervised binning pre-screen with categorical encoding | linear/weak signal; depends on optbinning + category_encoders deps | none (univariate IV) | medium |
| **stability (StabilityMRMR)** | small-n instability of MRMR permutation seed; FDR-controlled via bootstrap frequency | rare/imbalanced y (single-class bootstraps, S2); cost = n_bootstraps x base fit | inherits base estimator's | n_bootstraps x base |

Concrete improvement ideas:
- Resolve P1/P2 together with a multi-seed MRMR bench: switch tie comparator to `>` with explicit add-one `(1+#{>})/(n+1)` and unify early-break vs full-budget confidence; both are selection-altering so they must be benched, not patched.
- F3: debias `rel[i]` and `total_unique` with the same finite-sample correction (or apply the permutation floor to both) before the garbage-ratio compare.
- stability S2: add `stratify=True` bootstrap option for rare-class targets.
- hetero_vote recall: the skill-weight knob is measured-inert; the real lever is relaxing `vote_threshold`/`per_model_hit_frac`, not weighting (already documented in-module).

## 4. Tests

`tests/feature_selection/stability/test_stability_support_normalization.py`:
- `test_boolean_mask_selector_counts_correct_columns` — fails on pre-fix (support `{0,1}`), passes post-fix (`{2,4}`); pre-fix coercion verified by hand-simulation.
- `test_threshold_exactly_at_fraction_is_robust_to_float_rounding` — pins the integer-count gate at the 12/20==0.6 boundary.
- plus `_support_to_indices` unit coverage for mask + index inputs.

All 18 tests (new + existing `test_stability_coverage.py`) pass: `CUDA_VISIBLE_DEVICES="" pytest ... --no-cov -x --timeout=120` -> 18 passed.
