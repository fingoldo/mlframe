# ShapProxiedFS audit + optimization - 2026-06-22

Scope: `src/mlframe/feature_selection/shap_proxied_fs/` (29 modules, ~10.3k LOC) + the `ShapProxiedFS`
registration in `registry.py`. MRMR / worktree copies / other selector families untouched.

Method: 4 parallel read-only critique agents (cluster/SU, treeshap, prefilter/preflight/revalidate,
interactions/objective/search/calibrate) + a self-driven read of the fit path, registry, calibrate, and a
cProfile pass at C3-tier (n=2000, p=200, 6 informative + a 4-col correlated cluster), CPU-only.

## 1. Findings table (every finding gets an explicit disposition)

| # | file:line | severity | finding | disposition |
|---|---|---|---|---|
| C1 | `_shap_proxy_cluster_su.py:822,827` | P1 | Serial pairwise-SU constant-column test keyed off `freqs.size` (allocated bins, padded to max_bin+1 / n_bins_hint), but kernel/packed path keys off nonzero-bin count. A column with one realized bin but a multi-bin hint is treated non-constant serially, then runs `compute_su_from_classes` -> path-dependent clustering near the serial/parallel boundary. | **RESOLVED** - now `np.count_nonzero(freqs) <= 1` on both sides. Regression: `test_serial_su_treats_one_realized_bin_as_constant`. |
| I1 | `_shap_proxy_interactions.py:177-178` | P1 | Binary shap interaction base-selection reads `len(Phi)` AFTER `Phi` was reassigned from the list to an array, so the guard tests the ROW count (n) not the class count: picks the NEGATIVE-class base/tensor for any n!=2. | **RESOLVED** - capture `was_binary_list` before reassigning. Regression: `test_interaction_shap_binary_base_uses_positive_class`. |
| L1 | `_shap_proxy_revalidate/_shap_proxy_loss.py:249-250` | P2 | Single-class holdout AUC returned a magic `1.0` that passes the finite-mask and pollutes the corrector / stable-score ranking on rare-class anchors. | **RESOLVED** - returns `float("nan")` (dropped by every downstream finite-mask). Regression: `test_single_class_holdout_auc_loss_is_nan_not_sentinel`. |
| L2 | `_shap_proxy_revalidate/_shap_proxy_loss.py:333,412,431` | P2 | `predict_proba(...)[:, 1]` raises IndexError on a single-class anchor fit (1-column proba), surfacing as a fit failure inside the threaded pool. | **RESOLVED** - new `_positive_class_proba` helper falls back to the lone column. Regression: `test_positive_class_proba_handles_single_class_fit`. |
| L3 | `_shap_proxy_revalidate/_shap_proxy_loss.py:248,251` | P1 | Honest-loss layer (logloss `labels=[0,1]`, brier, AUC, `[:, 1]`) is binary-only with nothing enforcing it; a >2-class target gets silently-wrong losses. | **DOC/FUTURE** - documented binary-only contract; multiclass routing (logloss `labels=classes_`, AUC `multi_class="ovr"`) is a larger change deferred. The selector's default constructs a binary proxy; multiclass entry was already undefined. Tracked for next iteration. |
| CA1 | `_shap_proxy_calibrate.py:88-92` | P2 | Fitted Ridge corrector is not constrained monotone in proxy_loss; the `proxy*redund` term + a learned negative `proxy` coeff can invert the proxy order, violating the docstring's "never worse than raw proxy" guarantee. | **FUTURE** - real gap vs stated guarantee; fix is positive-coeff/NNLS on the proxy column OR identity-fallback when `pred` is not positively rank-correlated with proxy on anchors. Not yet applied (needs a biz_value bed to confirm the constrained fit does not lose the calibration win). |
| IX1 | `_shap_proxy_interactions.py:225-239` | P2 | Duplicated greedy-forward: the `else`-branch greedy is fully redundant with the unconditional pass below it (shared `cache`, so the second is all hits). Dead work, not incorrect. | **FUTURE** - safe simplification (delete 225-239); deferred to avoid bundling a behavior-neutral refactor with the correctness fixes in the same pass. |
| IX2 | `_shap_proxy_interactions.py:551` | P2 | Engineered product `X[a]*X[b]` has no NaN/inf/overflow guard; one bad pair poisons the whole in-sample SHAP fit. | **FUTURE** - guard with `np.nan_to_num` / skip non-finite pairs; only reachable on the opt-in `su_seeded_interactions` / interaction-proxy path. |
| IX3 | `_shap_proxy_interactions.py:90,109` | P2 | `np.full(n, float(base))` collapses/raises if a kernel ever returns a per-row base (contract says `(n,)`). Currently scalar in practice. | **DOC** - in-sample single-model base is genuinely scalar today; flagged so a future per-row kernel doesn't silently collapse it. |
| T1 | `_shap_proxy_treeshap_gpu.py:41-56` / `_interactions_gpu.py:56-58` | P1 (latent) | CUDA stack/scratch array sizes (`st_node[64]`, `MAXW=26`, `MAXLVL=27`, `MAXP=256`) are hand-written constants decoupled from the Python depth/width caps (`_MAX_SUPPORTED_DEPTH=24`). Currently SAFE at the asserted caps; silently overflows (OOB writes, garbage SHAP, no error) if a cap is bumped without touching the literals. | **DOC/FUTURE** - GPU path excluded from CPU test run; fix is to inject the caps into kernel source via f-string `#define` so they cannot drift. No live bug today. |
| T2 | `_shap_proxy_treeshap.py:121` | P2 | xgboost split resolution `int(str(sp).lstrip("f"))` strips ALL leading `f` chars (not the `f` prefix) and silently falls back to positional parse when `fmap` is present but missing a key -> silent SHAP misattribution for unusual feature names. | **FUTURE** - use `int(sp[1:])` on `^f\d+$`, raise on missing `fmap` key. Edge (numeric default feature names are the common case and parse fine). |
| T3 | `_shap_proxy_treeshap.py:78` | P2 | Logistic `base_score` clamp to +/-~16.1 for degenerate `base_score in {0,1}` (single-class training) yields a finite-but-arbitrary base that won't satisfy additivity. | **DOC** - single-class training is a degenerate caller error; CPU/GPU agree (not a parity bug). |
| T4 | `_shap_proxy_treeshap_gpu.py:152-158`, `_interactions_gpu.py:243-249`, `_shap_proxy_gpu.py:68-74` | P2 | `gpu_*_available()` / `_block_size()` swallow ALL exceptions; a genuine cupy/driver misconfig is silently demoted to CPU. | **FUTURE** - narrow to `(ImportError, CUDARuntimeError)`; let unexpected errors log/propagate. |
| P1f | `_shap_proxy_cluster_su.py:339` | Low | `inv_n = 1.0/n_samples` unguarded in `_pairwise_su_edges` (siblings guard `n_samples>0`). Unreachable today (empty short-circuited upstream). | **DOC** - defensive guard worth adding; not reachable. |
| D1 | `_shap_proxy_cluster_su.py:714` | P2 | Docstring claims `bitmap_max_n_bins` default 16; code default is 12 (the correct one per the iter73 bench). | **FUTURE** - docstring-only correction to 12. |
| SR1 | `_shap_proxy_subsetrank.py:133` / `_shap_proxy_gpu.py:145` | Low | `argpartition` ordering is undefined if `score_margin` returns NaN (degenerate single-class) -> could select a NaN as "top". | **DOC** - guard non-finite before argpartition; inputs clipped, low likelihood. |
| RF1 | `_shap_proxy_revalidate/_shap_proxy_refine.py:411` | Low | Random baseline `k=min(k,f)` then sample-without-replacement: when `k==f` the "random" baseline is the entire feature set -> meaningless winner's-curse baseline. | **DOC** - add a `k<f` guard; minor. |
| CA2 | `_shap_proxy_interaction_proxy.py:8-11,108` | Low (by design) | `margin = base + sum phi_j + 2*sum Phi_ij` double-credits within-subset interactions relative to a clean interaction coalition; this is the documented additive+bonus proxy, not File-1's `interaction_margin`. | **DOC** - cross-note the asymmetry; intentional. |

**Verified CLEAN** (explicitly checked, no bug): union-find relabeling; all 3 SU kernels' MI math + zero-prob guards; `_pack_onehot_bitmap` OOB/padding; NaN/inf `nan_to_num` up front in cluster entry points; single-feature / `f==0` / empty-cluster edges; build_unit_matrix has NO double-counting (one unit per cluster, downstream uses real member names); CPU/GPU TreeSHAP PARITY (identical recurrence, sign, cover-ratio weights, float64 accumulation, base_offset - no numeric divergence in any of the 3 explainers); interaction symmetrization + row-sum-consistent diagonal on both backends; LEAKAGE axis clean (honest loss always fits on `X_tr`/`X_search`, scores the disjoint `X_ev`/`X_holdout`; prefilter/preflight never peek; revalidate random baseline trains search->holdout); sampling reproducibility (pre-sampled seeds, deterministic rng); objective sign/stability (sigmoid branched, logloss clipped, AUC=1-auc, single-class guarded); search incremental-sum correctness + float64 + parallel bit-identity; heuristics incremental add/drop/swap + SA Metropolis guards + GA non-empty individuals + reproducible RNG.

## 2. Strengths / weaknesses verdict

**Wins:**
- **Runtime/memory vs true SHAP**: the additive coalition proxy scores subsets as `base + sum phi_j` in O(k) from a single OOF-SHAP pass, avoiding a retrain-per-subset. On wide proxies this is the whole value proposition; numba/GPU TreeSHAP kernels (bit-parity with CPU, ~1e-4 vs the shap library) make the attribution pass cheap.
- **Correlated clusters**: explicit SU/correlation clustering + PCA-PC1 unit collapse + honest within-cluster refine handles redundant groups more directly than vanilla SHAP ranking; the honest re-validation on a disjoint holdout corrects the proxy's redundancy bias.
- **Honest holdout discipline**: leakage axis is genuinely clean - the proxy is biased but the trust-guard / revalidate layer measures real honest loss on data the proxy never fit.

**Losses:**
- **Pure interactions (XOR / sign(a*b))**: the additive proxy is blind by construction (each pair folded into marginals ~0); only the OPT-IN `interaction_aware` / `su_seeded_interactions` paths recover them, and those carry the IX1-IX3 issues. vs MRMR's synergy-aware MI this is a default-off gap.
- **Multiclass**: honest-loss layer is binary-only (L3); the selector targets binary by default and multiclass is effectively unsupported in the honest path.
- **Calibration guarantee gap (CA1)**: the bias corrector can invert the proxy order despite the docstring's "never worse" claim.
- **vs Boruta**: no all-relevant shadow-feature statistical stopping; selection is loss-greedy, so it leans precision over recall (the `parsimony_tol=0.02` default is precision-tuned).

**Concrete improvement ideas (ranked):**
1. Constrain the corrector monotone in proxy (NNLS / positive Ridge or identity-fallback on negative rank-corr) - closes CA1, the only stated-guarantee violation.
2. Multiclass honest-loss routing (CA1-independent) - removes the silent-wrong-loss footgun (L3).
3. Make interaction product columns NaN-safe (IX2) + delete the dead greedy pass (IX1) before promoting any interaction default.
4. Inject CUDA caps from Python constants (T1) to remove the latent GPU buffer-drift class.

## 3. Optimization

cProfile at C3-tier (n=2000, p=200), CPU-only, warm (numba/shap pre-imported), `cumulative` + `tottime`:

- Total fit+transform ~28s steady-state. **~28s of it is external xgboost booster work**: `core.py:2362(update)` = 14.06s tottime (actual tree training), `_assign_dmatrix_features` (173s cumulative, recursive) re-setting DMatrix feature info, `predict_proba`/`predict` paths. `time.sleep` 5.67s tottime is xgboost's own multiprocessing pool (`pool.py:_wait_for_updates`) under each `n_jobs=-1` booster fit, plus the joblib threading parent retrieve-loop.
- **No mlframe-side Python frame is above ~1% of wall.** `_loss_from_predictions` cumtime is ~33s but tottime ~0.007s (it is pure dispatch around xgboost predict). `_honest_loss` likewise.
- The revalidate joblib pool already uses `prefer="threads"` with a prior `# bench-attempt-rejected (iter103/iter104)` note - the threading backend is correct here (CPU path; one thread per honest retrain, shared frames, no pickling). No change.

**Conclusion: no actionable mlframe-side speedup >=0.5% at the profiled scale** - the cost IS booster fitting, which is the expected shape for a SHAP-proxy selector. The honest-loss numpy-slice optimization (`_slice_cols_to_numpy`, strips pandas names so xgboost skips `from_cstr_to_pystr`) is already in place and was visible paying off in the suite biz_value output (reval 2.16x). Documented in the new bench harness `_benchmarks/profile_shap_proxied_fit.py` so the analysis is reproducible and not re-flagged. No optimization shipped this pass (correctness-only); the perf-measure-first rule says skip when no measurable win.

## 4. Tests

`tests/feature_selection/shap_proxied/test_shap_proxied_audit_fixes_2026_06_22.py` - 4 regression sensors (C1, I1, L1, L2), all verified to fail on pre-fix logic and pass post-fix. 76 affected existing tests re-run green (cluster_su, su_kernel_path_parity, interactions, revalidate, cluster_su_parallel).

## What was NOT covered (no silent truncation)

- **GPU code paths not executed** (CPU-only run per instructions): `_shap_proxy_treeshap_gpu.py`, `_treeshap_interactions_gpu.py`, `_shap_proxy_gpu.py`, the cupy/CUDA cluster-SU kernels. Reviewed statically (parity + T1/T4 findings) but not run; T1 buffer-drift is latent, not live.
- **Deferred fixes (FUTURE/DOC, not applied this pass)**: CA1 (corrector monotonicity), L3 (multiclass routing), IX1/IX2/IX3, T2/T3/T4, D1, P1f, SR1, RF1. Each has a concrete fix in the table; deferred to keep this pass correctness-focused and avoid bundling behavior-neutral refactors / larger redesigns with the verified P1/P2 fixes.
- **Modules read only at the interface level** (not line-audited by a dedicated agent): `_shap_proxied_methods.py`, `_shap_proxied_resolvers.py`, `_shap_proxy_compose.py`, `_shap_proxy_catboost.py`, `_shap_proxy_gradient.py`, `_shap_proxy_precomputed.py`, `_shap_proxy_explain.py` (781 LOC), `__init__.py` (805 LOC, read config surface only). The fit orchestration in `_shap_proxied_fit.py` (1061 LOC) was profiled and its hot path traced, but not exhaustively line-audited for correctness.
- **No biz_value test added** - this pass was bug-fix-only (per the bug-fix skip clause: regression sensor suffices; biz_value+cProfile required for new features, none added).
