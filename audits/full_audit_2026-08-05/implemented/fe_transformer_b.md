# Audit: fe_transformer_b

**Scope:** `src/mlframe/feature_engineering/transformer/*.py` (alphabetically-sorted, second half by
count: `focal_lgb.py` through `y_quintile_baseline_knn.py`, 62 files), plus all files under
`src/mlframe/feature_engineering/transformer/_benchmarks/` (1 file: `bench_anomaly_score_global_mean_subsample.py`).

**Files reviewed:** 63 (62 transformer modules + 1 benchmark script), full-file reads, no sampling.

**LOC reviewed:** ~10,259 (10,204 transformer + 55 benchmark).

**Out of scope (per instructions):** `src/mlframe/feature_selection/filters/**` and
`src/mlframe/feature_selection/shap_proxied_fs/**` (dedicated audit 10 days prior, all findings closed);
`_benchmarks` was included per the cluster spec despite the general "not `_benchmarks`" rule for the main
transformer listing.

**Context found during the audit:** this exact cluster name (`fe_transformer_b`) was the subject of a
prior audit dated 2026-07-21 (`audits/full_audit_2026-07-21/fe_transformer_b.md`), whose findings F1-F22
and F25 were fixed and pinned by `tests/feature_engineering/transformer/test_fe_transformer_b_fixes.py`;
F23/F24/F26-F28/PR2 were explicitly assessed and deferred (documented, not fixed) as larger
architectural/perf asks. This report's findings are independent of and do not duplicate that list — every
finding below was verified against the current file contents and cross-checked against
`test_fe_transformer_b_fixes.py` to confirm it is not already covered.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|---|---|---|---|---|---|
| FE_TRANSFORMER_B-1 | P1 | `performer_attention.py:48-59` | `_performer_features`'s per-row numerical-stability shift (`log_phi -= log_phi.max(axis=1, keepdims=True)`) is applied to the KEY-side (training-row) features before they are summed into the Performer aggregates `A`/`B`, silently reweighting every training row's contribution by an arbitrary row-specific factor unrelated to true kernel similarity. | Stabilize `phi_k` with a single GLOBAL scalar shift (e.g. subtract `log_phi.max()` over the whole `(n_train, M)` block, not per-row `axis=1`) so the shift cancels uniformly out of `A`/`B`; keep the existing per-row shift for `phi_q` (harmless there — cancels in each query's own numerator/denominator ratio). | Property test: compute Performer `y_estimate` via the current per-row-stabilized path and via a brute-force `softmax`-exact attention on the same small synthetic dataset (n<200) at a fixed temperature-equivalent bandwidth; assert the two agree within the RFF approximation's own sampling-noise tolerance (repeat over multiple `n_features` and confirm convergence as `n_features` grows). A generic scanner: grep for `.max(axis=` / `-= ... max(` stabilization patterns applied to an array that is later reduced via `.T @` or `.sum(axis=0)` across the SAME axis the max was taken over per-row — flag for manual review. |
| FE_TRANSFORMER_B-2 | P1 | `multi_aux_ensemble.py:45-72,93-104,138-140` | The module docstring (lines 17, 21) promises a third, structurally-diverse regression aux model ("L1-regularized LightGBM (quantile loss at median)") to decorrelate the 3-model disagreement signal, but `_fit_aux_lgb`'s `focal=True` branch only special-cases `task == "binary"`; for `task == "regression"` it falls straight through to the same `lgb.LGBMRegressor(**identical hyperparams)` as the first aux model (just `seed+1`), so `proba_focal`/`m_focal` for regression is a near-duplicate of `proba_lgb` differing only by RNG seed, silently weakening the "cross-model disagreement" feature the whole module exists to produce. | Either implement the documented quantile-loss LGBMRegressor branch (`objective="quantile", alpha=0.5`) for the regression `focal=True` path, or update the docstring/column semantics to state the regression `_focal` column is just a second same-family model with a different seed. | Regression test: for `task="regression"`, assert `m_focal`'s hyperparameters (or fitted structure / `objective`) differ from `m_lgb`'s beyond the `random_state`; a scanner rule: any `_fit_aux_*`/`_fit_baseline_*` helper with a boolean "flavor" flag (`focal`, `variant`, ...) whose `if flavor and task == X` guard has no `else` branch touching `task == Y` should be flagged as a suspicious flavor-not-applied-for-Y gap. |
| FE_TRANSFORMER_B-3 | P2 | `geodesic_kgraph.py:66-74` | The kNN graph used for multi-source Dijkstra is built via `nn_graph = NearestNeighbors(n_neighbors=k_graph, ...).fit(Xt_s); dists, idxs = nn_graph.kneighbors(Xt_s)` — querying a fitted index with its own training data returns each row as its own nearest neighbor (distance 0) at one of the `k_graph` slots, unlike the correctly-handled sibling patterns in `local_density_gradient.py` (`n_neighbors=k_eff + 1`, then drops column 0) and `spectral_attention.py`'s `_build_knn_graph` (`n_neighbors=k_graph + 1`, `dists[:, 1:]`). The self-loop doesn't corrupt Dijkstra distances (0-weight self-edges never appear on a shortest path) but consumes one of the `k_graph=10` neighbor slots on every row, silently reducing the graph's real branching factor to `k_graph-1` and weakening connectivity in sparse regions. | Request `n_neighbors=k_graph + 1` and drop the self-match column (index 0) before building `rows`/`cols`/`weights`, matching the pattern already used in `local_density_gradient.py` and `spectral_attention._build_knn_graph`. | Generic scanner: grep every `NearestNeighbors(...).fit(A)` followed by `.kneighbors(A)` (same array object/variable on both sides) in this package; each hit must either request `k+1` neighbors and drop column 0, or have an explicit comment justifying why self-inclusion is intentional. |
| FE_TRANSFORMER_B-4 | P2 | `mdl_binning_pairwise.py:211-219` | The pairwise bin-combo dedup encoding hardcodes `train_combo = train_bins[:, 0] * 100 + train_bins[:, 1]` (and the matching `query_combo`), assuming feature-1's bin index never reaches 100. `train_bins[:, j] = np.digitize(Xt[:, j], all_edges[j])` can return up to `max_bins_per_feat` distinct values, and `max_bins_per_feat` is an unvalidated public parameter of `compute_mdl_binning_pairwise_features`. A caller passing `max_bins_per_feat >= 100` (or even lower, given `np.digitize` can return `len(edges)` for the top bin) gets silent combo collisions — e.g. `(feature0=1, feature1=100)` and `(feature0=2, feature1=0)` both hash to `200` — producing wrong `combo_count`/`n_unique_combos` features with no error or warning. | Use a radix derived from the actual max bin count (`radix = max(train_bins.max(), query_bins.max()) + 1`) or `np.ravel_multi_index`/a tuple-based `np.unique` instead of a fixed base-100 multiply; alternatively validate `max_bins_per_feat < 100` at entry and raise/clamp. | Property test: parametrize `max_bins_per_feat` up to 150 with a feature column engineered to actually produce >=100 distinct MDL bins (or monkeypatch `_mdl_bin_edges` to force a wide edge list) and assert no two distinct `(bin0, bin1)` pairs in the test data collide to the same `combo` code. Generic scanner: grep for `col_a * <int literal> + col_b`-style manual radix encodings anywhere two independently-bounded integer columns are combined, and confirm the literal is derived from (or asserted against) the columns' actual max cardinality, not a fixed constant. |
| FE_TRANSFORMER_B-5 | P2 | `pairwise_kl_divergence.py:39-58`, `nn_oof_target_mean.py:50-77`, `ib_baseline_codes.py:23-40`, `gradient_direction_agreement.py:18-37`, `multi_baseline_hard_row.py:43-69` | Five files in this cluster each independently define a near-identical `_fit_3baselines*` helper (LGB depth-3 + LGB depth-5 + LogisticRegression/Ridge, with a try/except around the linear model). The duplication has already produced observable behavioral drift: `multi_baseline_hard_row.py`'s LogisticRegression-failure fallback logs at `logger.info(...)`, while the other four log the identical failure at `logger.debug(...)` (invisible by default); `gradient_direction_agreement.py`'s fallback sets `m3 = None` (caller substitutes a zero gradient), while the other four fall back to a constant class-prior prediction — two materially different degenerate-fold semantics for the "linear baseline failed to converge" case, purely because each copy evolved independently. | Extract a single shared `_fit_3_diverse_baselines(Xt, y_t, task, seed, ...)` helper (e.g. in a new `_baseline_trio_shared.py` sibling, following the existing `_hard_row_shared.py` / `_focal_loss_shared.py` precedent) returning predictions + a uniform failure-fallback contract; have all 5 (and any first-half-cluster siblings, e.g. `baseline_disagreement*.py`, `disagreement_band.py`) call it. | Generic scanner: an AST/text near-duplicate detector (e.g. normalized-token Jaccard or `difflib.SequenceMatcher` over function bodies) run across all `_fit_*baseline*`/`_fit_3*` helpers in a package; flag any cluster of >=3 functions with >85% body similarity as a dedup candidate, and additionally diff their `except`/fallback branches specifically for behavioral drift (log level, fallback value semantics). |
| FE_TRANSFORMER_B-6 | P3 | `y_quintile_baseline_knn.py:131` | For `task="regression"`, `strata_edges = np.quantile(y_t, np.linspace(0.0, 1.0, _N_STRATA + 1))` has no tie-breaking, unlike the sibling `target_quantile.py` (`_compute_centroids`), which explicitly bumps any non-increasing adjacent edge by `1e-9` to guarantee non-degenerate buckets. On a tied/discrete/heavily-rounded regression target, `y_quintile_baseline_knn.py` can silently produce one or more fully-empty strata (`mask.sum()==0`), which `_knn_pred_stats` degrades to all-zero mean/std for that stratum rather than the graceful non-degenerate split the sibling pattern achieves. | Apply the same `for i in range(1, len(edges)): if edges[i] <= edges[i-1]: edges[i] = edges[i-1] + 1e-9` tie-protection used in `target_quantile.py._compute_centroids`. | Regression test mirroring `target_quantile.py`'s implicit contract: feed a heavily-tied/discrete regression target (e.g. `y = rng.integers(0, 3, n)`) through `compute_y_quintile_baseline_knn_features` and assert no stratum's `(mean, std)` pair is silently `(0.0, 0.0)` for every query row (i.e. every stratum actually received training rows, or the fallback is at least non-trivial). Generic scanner: grep every `np.quantile(y, np.linspace(...))` call in this package and confirm each is followed by (or the caller otherwise guarantees) strictly-increasing edges before being used as bucket boundaries. |
| FE_TRANSFORMER_B-7 | P3 | `multi_threshold_ordinal.py:78-84` | The binary sub-population classifier loop only guards `if target.sum() == 0:` before fitting `LGBMClassifier` on `target = ((y_t > 0.5) & (Xt_s[:, j] > median_j))`, whereas the regression branch of the SAME function (lines 59-61) guards both `target.sum() == 0 or target.sum() == target.shape[0]`. In the binary branch this asymmetry happens to be unreachable in practice (`X[:, j] > median(X[:, j])` can select at most ~half the rows by the definition of median, so `target` can never be all-True), but the inconsistency is a latent trap if the threshold logic is ever changed (e.g. to a different quantile cut where "count > cut" CAN exceed 50%). | Add the matching `or target.sum() == target.shape[0]` guard for symmetry/defensiveness with the sibling regression branch, or add a comment explaining why it's provably unreachable here. | Generic scanner: for every single-class-guard pattern (`if target.sum() == 0` / `== 0 or == shape[0]`) preceding an `LGBMClassifier.fit`/`LGBMRegressor.fit` call in this package, verify siblings of the same function guard both extremes consistently; flag asymmetric guards within the same function for review. |
| FE_TRANSFORMER_B-8 | P3 | `local_density_gradient.py:93` | `k_eff = min(k_neighbors, Xt_s.shape[0] - 1)` evaluates to `0` when a fold's training set has exactly 1 row (an extreme but not impossible edge case under a pathological splitter or a tiny dataset); the subsequent `NearestNeighbors(n_neighbors=k_eff + 1=1, ...)` and `train_dists[:, k_eff]`/`q_dist_to_kth = q_dists[:, k_eff - 1]` (the latter using Python negative-index wraparound for `k_eff=0`) do not crash, but produce a degenerate, essentially meaningless density estimate (`log_density ~ -d*log(1e-9)`, a huge constant) with no validation or warning. | Add an explicit `if Xt_s.shape[0] < 2: raise ValueError(...)` (or a documented degenerate-fold fallback consistent with the rest of the cluster's `_FAR`/global-mean sentinel convention) rather than silently computing a numerically-extreme value via negative-index wraparound. | Edge-case test: call `compute_local_density_gradient_features` with a 1-row training fold (via a custom splitter or `X_train.shape[0]==1` in Mode B) and assert either a clear `ValueError` or a documented, non-extreme sentinel output — not a silent huge/NaN value. Generic scanner: grep for `[:, k_eff - 1]`-style negative-index-capable slicing derived from a `min(k, n-1)` computation and confirm a lower bound on `n` is enforced upstream. |

## Narrative detail

**FE_TRANSFORMER_B-1 (performer_attention.py).** `_performer_features` computes
`log_phi = X @ W - 0.5*||X||^2` then subtracts `log_phi.max(axis=1, keepdims=True)` (a PER-ROW constant)
before `exp()`, for numerical stability. This function is called on BOTH the query rows (`phi_q`) and the
training/key rows (`phi_k`). For `phi_q`, the per-row constant `s_q` cancels exactly out of the ratio
`y_estimate = (phi_q @ A) / (phi_q @ B)` because it scales the entire numerator and denominator dot-products
identically for that one query row. For `phi_k`, however, `A = phi_k.T @ y_t` and `B = phi_k.sum(axis=0)`
are SUMS over different training rows `i`, each carrying its OWN row-specific stabilization constant `s_i`
(a function of that row's own `||k_i||^2` and random-projection alignment, unrelated to the target). This
means `A[m] = sum_i s_i * phi_true(k_i)[m] * y_t[i]` instead of the intended
`sum_i phi_true(k_i)[m] * y_t[i]` — every training row's contribution to the Performer aggregate is silently
reweighted by an arbitrary per-row factor. I found this by hand-deriving the Performer linear-attention
identity (`out_q = phi(q)·(phi(K)^T Y) / phi(q)·sum(phi(K))`) and checking which stabilization shifts are
provably invariant to that ratio; a single shared/global shift is invariant, a per-row shift on the
side that gets summed across rows (the key side) is not. Every call to this feature (not just an edge case)
is affected; the magnitude of the distortion depends on the spread of `||k_i||^2` across the standardized
training fold.

**FE_TRANSFORMER_B-2 (multi_aux_ensemble.py).** The module's own docstring (lines 17-19) states the
regression variant swaps in "L1-regularized LightGBM (quantile loss at median)" as the third, structurally
distinct aux model — the entire point of the module is that CB/downstream boostings can't compute
cross-model-family disagreement themselves, so the three aux models must actually be different objective
functions. But `_fit_aux_lgb`'s `if focal and task == "binary":` guard has no `elif focal and task ==
"regression":` branch; it falls through to the identical `lgb.LGBMRegressor(**params)` construction used
for the non-focal model, differing only by `seed=fold_seed+1` vs `seed=fold_seed`. I confirmed this by
tracing `compute_multi_aux_features._process`'s call `_fit_aux_lgb(Xt, y_t, task=task, seed=fold_seed+1,
focal=True, ...)` through to `_fit_aux_lgb`'s body: the `if focal and task == "binary"` short-circuits to
`False` for regression, and the function proceeds straight to the shared `if task == "binary": ... else:
model = lgb.LGBMRegressor(**params)` block with the SAME `params` dict the first model used. The
`proba_focal`/`m_focal` column for every regression call is therefore a same-family duplicate model, not
the documented quantile-loss model, silently weakening the `proba_std`/`proba_range` disagreement features'
signal for regression tasks specifically (binary tasks are unaffected — the focal branch works correctly
there).

**FE_TRANSFORMER_B-3 (geodesic_kgraph.py).** `_process`'s graph-construction step fits
`NearestNeighbors(n_neighbors=k_graph)` on `Xt_s` and immediately queries it with the SAME array
(`nn_graph.kneighbors(Xt_s)`), which per sklearn's documented behavior ("If not provided [X], the query
point is not considered its own neighbor" — implying if X IS provided, self-matches ARE included) returns
each row as one of its own `k_graph` nearest neighbors at distance 0. I verified this is a real inconsistency
by comparing directly against two sibling files in the SAME cluster that solve the identical self-match
problem correctly: `local_density_gradient.py` requests `k_eff + 1` neighbors and reads `train_dists[:,
k_eff]` (skipping index 0, the self-match), and `spectral_attention.py`'s shared `_build_knn_graph` helper
(reused by `per_class_spectral.py` in this same cluster) requests `k_graph + 1` and explicitly does
`dists[:, 1:]` / `ids[:, 1:]` before building the sparse graph. `geodesic_kgraph.py` requests only
`k_graph` (not `+1`) with no self-exclusion, so every row's graph degree from that self-loop edge is
effectively `k_graph - 1` useful neighbors instead of `k_graph`. This doesn't produce a WRONG shortest-path
distance (a 0-weight self-loop is never on a nontrivial shortest path), but it does silently under-connect
the graph relative to the intended `k_graph=10` design parameter, which matters most on sparse
target-class subsets (the exact regime the module targets — "nearest opposite-class row" for imbalanced
binary tasks).

**FE_TRANSFORMER_B-4 (mdl_binning_pairwise.py).** The pairwise-co-occurrence feature encodes two
independently-computed bin-index columns into one scalar via `bin0 * 100 + bin1`. `train_bins`/`query_bins`
come from `np.digitize(X[:, j], all_edges[j])`, where `all_edges[j]` has up to `max_bins_per_feat` split
points (so up to `max_bins_per_feat` distinct bin indices, 0-indexed). `max_bins_per_feat` is a public,
unvalidated keyword of `compute_mdl_binning_pairwise_features` (default 8, but caller-settable). I confirmed
via `test_mdl_binning_combo_count_vectorized.py` (the closest existing test) that its fuzzing only exercises
combo radices up to `hi=12` (`rng.integers(2, 12)` for its bin-count stand-in) — nowhere near the 100 the
production code hardcodes — so the >=100-bin collision path has zero test coverage. This is a genuine,
if currently-unlikely-to-trigger-at-the-default, silent-wrong-result bug class: two structurally different
`(bin0, bin1)` pairs can hash to the identical `combo` integer, corrupting both the `combo_count` feature
(wrong train-frequency lookup) and `n_unique_combos` (undercounts distinct combos) with no error raised.

**FE_TRANSFORMER_B-5 (duplicated `_fit_3baselines*` family).** Grepping this package for
`_fit_3baselines`-style function names surfaced 9 total occurrences repo-wide (5 in this cluster's scope:
`pairwise_kl_divergence.py`, `nn_oof_target_mean.py`, `ib_baseline_codes.py`,
`gradient_direction_agreement.py`, `multi_baseline_hard_row.py`; 4 more in the sibling first-half cluster).
Each independently re-implements "fit LGB depth-3 + LGB depth-5 + LogisticRegression/Ridge, with a
try/except fallback for the linear model." Reading all 5 in-scope copies side by side surfaced real,
already-manifested behavioral drift from the duplication rather than a purely hypothetical DRY complaint:
`multi_baseline_hard_row.py`'s except-block logs at `logger.info(...)` ("multi_baseline_hard_row:
LogisticRegression fit failed (%s); falling back to constant class prior.") while the other four
(`pairwise_kl_divergence.py`, `nn_oof_target_mean.py`, `ib_baseline_codes.py`, and the binary path of
`gradient_direction_agreement.py`) log the identical condition at `logger.debug(...)` — invisible under
default logging configuration, meaning the SAME real production failure (LogisticRegression convergence
failure on a pathological fold) is silently invisible in 4 of 5 modules and visibly logged in the 5th, purely
by accident of which copy a past editor happened to touch. Separately, `gradient_direction_agreement.py`'s
own fallback semantics diverge further: on LogReg failure it sets `m3 = None` and its caller substitutes a
literal zero GRADIENT (`g3 = np.zeros_like(g1)`), whereas the other four substitute a constant
CLASS-PRIOR PREDICTION for the failed model's output — two different degenerate-fold contracts for the
"third baseline failed to fit" case that a single shared helper would have kept consistent by construction.

**FE_TRANSFORMER_B-6 (y_quintile_baseline_knn.py).** For the regression path, `strata_edges =
np.quantile(y_t, np.linspace(0.0, 1.0, _N_STRATA + 1))` is used directly as bucket boundaries with no
tie-protection. I found this by diffing the stratification logic against `target_quantile.py`'s
`_compute_centroids` in the SAME cluster, which explicitly walks the computed edges and bumps any
non-strictly-increasing adjacent pair by `1e-9` specifically to prevent ties in `y` from collapsing a
bucket to empty. `y_quintile_baseline_knn.py` has no equivalent step, so a heavily-tied or coarsely-rounded
regression target (common for count-like or binned targets) can silently produce one or more `mask.sum()==0`
strata, each of which `_knn_pred_stats` degrades to `(mean=0.0, std=0.0)` for every query row — a plausible,
reachable robustness gap on exactly the kind of target this cluster's docstrings repeatedly call out as a
concern (quantized/skewed targets), not merely a theoretical corner case.

**FE_TRANSFORMER_B-7 (multi_threshold_ordinal.py).** The function's own regression branch (a few lines
above) guards `if target.sum() == 0 or target.sum() == target.shape[0]:` before fitting a classifier on a
derived binary target, explicitly handling both single-class-degenerate directions. The binary branch's
sub-population loop guards only `if target.sum() == 0:` for `target = ((y_t > 0.5) & (Xt_s[:, j] >
median_j))`. I verified analytically that the missing "all-True" branch is not currently reachable — because
`Xt_s[:, j] > median_j` can, by the definition of a median, select at most roughly half of any row set, the
AND'd `target` can never legitimately be all-True except in a degenerate `n<=1` case already excluded
upstream — so this is reported as a P3 code-consistency/latent-trap finding (the asymmetry would silently
become a live crash risk if the median-based threshold logic were ever swapped for a different cut, e.g. a
lower quantile that COULD select more than half the rows) rather than a currently-exploitable P1/P2 bug.

**FE_TRANSFORMER_B-8 (local_density_gradient.py).** `k_eff = min(k_neighbors, Xt_s.shape[0] - 1)` is `0`
when the training fold has exactly one row. The immediately following `NearestNeighbors(n_neighbors=k_eff +
1)` still constructs a valid (single-neighbor) index, and `q_dist_to_kth = q_dists[:, k_eff - 1]` becomes
`q_dists[:, -1]`, which Python silently resolves via negative-index wraparound to the same (only) column
rather than raising an `IndexError` — so the function does not crash, but produces a numerically-extreme,
uninformative `log_density` (`-d * log(~1e-9)`, a large constant unrelated to genuine density) with no
validation, warning, or documented sentinel, unlike this cluster's now-consistent convention (post the prior
audit's F3-F8 fixes) of using an explicit, documented far/near sentinel for genuinely-empty subsets.

## Dimension coverage notes

- **ML correctness / leakage:** no NEW leakage findings. OOF discipline is consistently well-documented and
  implemented across this batch (inner-`KFold` OOF baselines used specifically to avoid in-sample-residual
  bias in `sign_residual_baseline.py`, `multi_temp_cbhr.py`, `residual_stratified_distance.py`,
  `y_quintile_baseline_knn.py`; standardizers/quantile-edges/centroids consistently refit per outer fold).
- **Computational efficiency:** no new efficiency findings beyond what the prior 2026-07-21 audit already
  flagged and explicitly deferred (F26-F28: per-row Python loops in `local_classifier.py`,
  `persistence_diagram.py`, and (out-of-scope-here) `cluster_smote.py`/`cutmix.py`).
- **Edge cases / robustness:** covered by FE_TRANSFORMER_B-6, -7, -8 above; no additional gaps found beyond
  those three (single-class, degenerate-fold, and empty-subset handling elsewhere in this batch is
  consistently good, matching the prior audit's F3-F8/F19-F22 fixes).
- **Test coverage gaps:** `performer_attention.py`, `geodesic_kgraph.py`, `mdl_binning_pairwise.py`'s
  combo-encoding radix, `multi_aux_ensemble.py`'s regression `_focal` branch, and
  `y_quintile_baseline_knn.py`'s tie-protection all have zero direct test coverage of the specific defect
  (confirmed via `find`/`grep` across `tests/feature_engineering/transformer/`); this is the same gap the
  meta-test ideas above are meant to close.
- **Code quality / architecture:** FE_TRANSFORMER_B-5 (the `_fit_3baselines*` duplication) is the material
  finding here. The `_K_SCALES`/`pos_loggap_columns`/`class_or_quantile_slice`/`kth_nearest_dists` SMOTE-family
  boilerplate (`mixup_boundary.py`, `multiscale_smote.py`, `pseudo_smote.py`, `pure_pos_smote.py`,
  `smote_distance.py`) is ALSO duplicated at the driver-loop level, but the actually-reused logic (softmax,
  top-k-within-subset, kth-nearest-dist, class/quantile slicing) is already correctly factored into shared
  `_utils.py`/`_hard_row_shared.py` helpers, and this exact residual duplication was already assessed and
  explicitly deferred as F23/F24 in the prior 2026-07-21 audit — not re-reported here to avoid duplicating a
  disposition that already exists.
- **OSS/hygiene:** no stale audit-wave markers, mojibake, or comment-style violations found in this batch;
  comments consistently follow the WHY-only convention already established elsewhere in the codebase.

## Severity counts

- P0: 0
- P1: 2 (FE_TRANSFORMER_B-1, FE_TRANSFORMER_B-2)
- P2: 3 (FE_TRANSFORMER_B-3, FE_TRANSFORMER_B-4, FE_TRANSFORMER_B-5)
- P3: 3 (FE_TRANSFORMER_B-6, FE_TRANSFORMER_B-7, FE_TRANSFORMER_B-8)
