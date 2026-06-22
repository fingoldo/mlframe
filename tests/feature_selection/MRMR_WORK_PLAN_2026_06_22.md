# MRMR FS/FE — detailed work plan (2026-06-22)

Exhaustive plan for the outstanding MRMR work. Captures EVERY finding from the 3 most-recent optimization
agents (GPU-residency, CPU-noise-gate, residual-FE) AND the deferred findings from the earlier 6 critique
agents. Ordered per request: **validatable optimizations/hygiene first; distribution-robustness (residual-
aware FE) LAST**.

Source / cross-reference files (read these for full context):
- `tests/feature_selection/MRMR_AUDIT_2026_06_22.md` — the 6-agent critique audit + dispositions (RESOLVED /
  QUEUED / REJECTED). The "QUEUED" + "code quality / hygiene" sections there are the deferred findings.
- `tests/feature_selection/MRMR_FE_PERF_NOTES.md` — wall report, dual-profiler (nsys/cProfile) numbers,
  every landed perf change + bench-rejected notes, and the GPU#2-already-done verification.

Invariants for EVERY item: CPU/no-CUDA path byte-unchanged; selection bit-identical where claimed; NO
unvalidated selection change ships; one item at a time → implement → validate against its GATE → commit+push.
Mark `[x]` + commit hash when landed.

---

## PART 1 — GPU-residency agent findings (nsys F2 100k: cuLaunchKernel 381,216; cub DeviceReduce::Max 52%
## /71,812 calls, Sum 31%/48,078, RadixSort 15%/23,942; 100,757 tiny D2H; 2.35 GB H2D)

- [x] **G1 (was #1, the top cp.max source).** `_hermite_fe_mi.py:249` recomputed `cp.min(y)+cp.max(y)+D2H`
  per chunk per pair though y is a fit-constant. FIX: optional `y_min`/`n_classes` params, computed once by
  the callers. → **DONE ca85bf3f** (37 passed).
- [x] **G5 (was #5).** `_gpu_route_bases_batched` re-ran `_gpu_detect_heavy_tail_batched` per candidate basis
  (basis-independent). FIX: compute `heavy_host` once, pass via new param. → **DONE 1dc6af37** (43 passed).
- [x] **G2 (was #2) — 2.35 GB H2D / operand residency.** `_gpu_resident_fe.py:1230-1231,1239`
  per-pair `cp.asarray(a/b/y)`. **VERIFIED ALREADY-DONE in production**: `build_resident_operand_table`
  (+ `register_prebuilt_operand_table`, `_resident_operand_table` weakref-cache) at `_pairs_core.py:946-952,
  1282-1285` builds operand columns ON the device + reuses per chunk. The agent mis-attributed the 2.35 GB
  to `gpu_resident_pair_candidate_mi` (the PROTOTYPE, not the production path). Residual H2D = necessary
  cached raw-input/y uploads × the 2 profiled fits. No clean further win. → resolved (see perf notes a45120f6).
- [ ] **G3 (was #3) — chunk sizing.** `_hermite_fe_mi.py:261,267-269` `cp.percentile` per chunk per pair.
  The chunking is per-pair; raising `k_chunk` (fewer/larger chunks) cuts the launch/reduction count. FIX:
  route `k_chunk` width through `kernel_tuning_cache` to maximise chunk width within the VRAM budget. GATE:
  `test_percentile_binning_chunk_invariant` (chunk-invariant binning) + FE pins; selection-equivalent. RISK:
  low (chunk-invariant already pinned). NOTE: partly addressed by the existing `_gpu_k_chunk` VRAM governor;
  the remaining lever is KTC-tuning the width.
- [x] **G4 (was #4) — sum(axis) per chunk per pair.** `_hermite_fe_mi.py:287-288,311` (≈ the 48k cub Sum).
  Same outer-multiplier lever as G1/G3; already optimally batched WITHIN a call. No separate change beyond
  G3's fewer-chunks. → fold into G3 / mark no-op after G3.
- [x] **G6 (was #6) — dead per-column scalar-D2H path.** `_gpu_resident_fe.py:589-655`
  (`_gpu_robust_scale`/`_gpu_detect_heavy_tail`/`_gpu_basis_preprocess` per-column scalar variants, ~6-10
  `float(cp...)` D2H each) are SUPERSEDED by the `_batched` twins. CONFIRM no live caller uses the per-column
  `_gpu_evaluate_basis_column` (`__init__.py` imports it but the live path uses `_gpu_evaluate_basis_matrix`)
  → if dead, remove/guard so it can't regress. GATE: grep callers + import smoke + FE pins. RISK: dead-path
  removal — verify zero live callers first.

## PART 2 — CPU-noise-gate agent findings (cProfile F2 100k: batch_mi_with_noise_gate = 190s / 93%)

- [x] **F1 + F2.** `batch_mi_with_noise_gate` perm loop: serial-outer with inner prange (fork/join barriers)
  + strided `classes_dense[:,k]` re-read + per-(perm,col) `np.zeros`. FIX: precompute all shuffled y once,
  prange-over-columns + serial-perm-inner via new `_perm_failcount_col` (contiguous col buffer + reused
  histogram + hoisted SU denom). → **DONE 900dd660** (106 passed).
- [x] **F3 — prange over live (om>0) columns.** Already present (`if original_mi[k] <= 0.0: continue` inside
  the prange) + the new `_perm_failcount_col` skips dead columns. → covered by 900dd660.
- [x] **F4 — fastmath.** REJECTED (would reassociate the log-sum → flip a boundary gate verdict). Left a note
  in the commit. → no-action by design.

## PART 3 — Residual-FE agent findings (the distribution-robustness fix) — TIER C, LAST

Goal test (self-policing): `test_f2_single_compound_across_distributions` (53a12960) — uniform = hard guard;
scaled_1_5 / heavy_tailed / mixed / with_outliers = `xfail(strict=True)`. Root cause: signal-scale imbalance
(a²/b dominates Var(y) → MI(log c·sin d ; y) ≈ 10% of MI(a/b ; y) → the weak half falls below the prevalence
gate → fragments). KEY in-code note `_mrmr_class.py:1728-1731`: a prior admission relaxation
(`fe_pair_perm_null_admission_enable`) admitted (c,d) but CONSTRUCTION still failed → "needs a separable
warp-product proposer, not admission relaxation."

- [x] **C0 — foundation.** `SufficientSummaryVerdict.residual` surfaces `r = y − E_hat[y|selected]`
  (`_fe_sufficient_summary.py:285`). → **DONE 53a12960**.
- [ ] **C1 — retarget wiring.** `_fit_impl_core.py:6731`: when `_ss_verdict.reached` is False AND
  `blocking_raw >= 0` AND `residual is not None` → `self._fe_residual_target_continuous_ = verdict.residual`
  for the next FE step (clear after). `_step_core.py:1108-1118`: when set, use the discretised residual as
  classes_y / `_fe_prewarp_y_continuous_` / `usability_y_continuous` for that step and recompute `cached_MIs`
  against it (the 1.05 prevalence constant is UNCHANGED — the residual makes the weak half clear it). Keep all
  noise gates ON applied to r (order-1 perm null `general.py:212-236`, maxT joint `_step_core.py:737,782`,
  CMI redundancy). Default-on, gated by residual-signal-present (no-op on balanced targets). GATE: the goal
  test flips xfail→pass (then REMOVE the xfail marks) + `test_feature_engineering_example_single_compound`
  no-regress + a NEW no-noise-admission test (y = a²/b + f/5 → no new c/d/e) + biz-value suite. RISK: the
  documented construction failure — verify r-step builds the clean `mul(log(c),sin(d))`; if not → C2.
- [ ] **C2 — separable warp-product proposer** (only if C1's construction fails the goal test). Terminal
  (no-fusion) proposer that builds the clean `log(c)·sin(d)` product on the residual directly. GATE: as C1.
- Escalation residualiser (`_fe_auto_escalation.py:678-697`) is NOT the path (its held-out-corr floors
  `fe_escalation_min_val_corr=0.15` / `pairness_margin=1.15` block a 0.13-MI signal; "admits ~nothing").

## PART 4 — Deferred findings from the earlier 6 critique agents
## (full list + dispositions in `tests/feature_selection/MRMR_AUDIT_2026_06_22.md`)

All P0/P1/P2 equivalency divergences from those agents are RESOLVED (see the audit doc's RESOLVED section +
the 2026-06-22 follow-through update). The REMAINING (invasive / low-value / parallel-session) ones:

- [ ] **D1. source-name `__` split** (hidden-flaw #5). 9 sites in `_orthogonal_univariate_fe/` (`__init__.py`
  :477,524,754-755; `_orth_pair_cross_fe.py`:202,442-444,507-508). `name.split("__",1)[0]` mis-stems a
  one-hot source `col__value` → `uplift = emi/1e-12` → always clears the gate. FIX: carry the source col in
  metadata instead of re-parsing. GATE: new `col__value`-named-source test + recipe round-trip tests. RISK:
  invasive naming-convention change; preserve recipe-replay byte-exactness.
- [x] **D2. `y.to_numpy()` 53× hoist** in `_fit_impl_core.py` (code-quality). Hoist once (y not reassigned);
  replace the 53 inline `y.to_numpy() if hasattr else np.asarray(y)`. GATE: grep-confirm no reassign +
  test_mrmr_feature_engineering + biz subset. RISK: mechanical (53 sites in 9.8k file); behavior-preserving.
- [x] **D3. lstsq→normal-eq** in `_orth_extra_basis_fe.py` deflation (perf #8). `_deflate_sincos:336`
  [IN PROGRESS] + the vander-poly sites `:449,589`. normal-eq solve + lstsq fallback (rank-robustness). GATE:
  test_fe_auto_escalation + test_biz_value_mrmr_adaptive_fourier + test_biz_val_extra_basis_fe + F2 pin.
  RISK: selection-bearing (not bit-identical), low live impact (escalation-gated).
- [x] **D4. `_env_truthy` env-flag DRY.** TWO helpers (`_env_opt_in` / `_env_opt_out_cuda`) — the gates have
  divergent opt-in vs opt-out+CUDA_VISIBLE_DEVICES+MLFRAME_DISABLE_GPU logic. GATE: import smoke + gate-
  default tests. RISK: a wrong default flips a gate; behavior-preserving.
- [ ] **D5. ctor-defaults single-source** in `mrmr/_mrmr_class.py` — derive `__setstate__` defaults from
  `_ctor_defaults()` (already drifted: cluster_aggregate_mode). GATE: pickle round-trip + edges-coverage.
  RISK: legacy-pickle back-compat — keep extra legacy keys.
- [ ] **D6. evaluation.py carve** (1144 LOC > 1000-LOC meta-test ceiling). Carve a sibling (prefill/JMIM
  block) + re-export. GATE: test_no_file_over_1k_loc green + greedy/CMI tests. RISK: parallel-session-owned;
  rebase carefully; verbatim move.
- [ ] **D7. M-tier edge cases** (audit doc LOW tier, opt-in/bench-rejected paths): imbalance-MI `n_bins`
  mismatch, MM-debias cardinality, `_class_balanced_mi` y-offset, etc. Revisit only if those paths are
  promoted to default.

## EXECUTION ORDER (agreed)
Tier A/D perf+hygiene first (D3 in progress → D2 → G3/G6 → D4/D5 → D1/D6), then **Tier C (residual-FE) LAST**.
One item at a time: implement → validate against its GATE → commit+push → tick the box here.

## PART 5 — TIER E: monolith carves (drain the >1k-LOC meta-test EXEMPT list)

WHY these exist (answer to "how did we breed mastodons when we have a <1k meta-test"): the meta-test
(`tests/test_meta/test_no_file_over_1k_loc.py`) DOES enforce <1000 LOC, but via an EXEMPT list with
FIXME(carve-wave-next) tags. The exempt files are **irreducible single giant FUNCTION bodies** -- e.g.
`_fit_impl` is ONE ~9.8k-line function bound onto MRMR. Prior carve waves split the SURROUNDING helpers into
siblings (the sibling-re-export pattern), but that pattern splits MULTIPLE functions, not ONE giant one. To
shrink these you must first EXTRACT BLOCKS of the giant function into helper functions (threading the shared
local state), THEN move them to sub-modules. New code kept being appended to the giant body instead of being
carved as it grew (now forbidden -- see CLAUDE.md "Build new code in focused submodules from the START").

Currently-exempt (drain to {}):
- `_mrmr_fit_impl/_fit_impl_core.py` (~9.8k) — extract the ~40 inline `if bool(getattr(self,"fe_X_enable")):`
  FE-stage blocks into uniform `_fe_stage_X(self, data, cols, nbins, y_np, ...) -> appended_cols` helpers
  driven by one ordered loop; then carve `_fit_setup.py` / `_fe_stages_*.py` / `_fit_screen.py` /
  `_fit_finalize.py`. (Biggest; do via stage-extraction first.)
- `mrmr/_mrmr_class.py` (~4.8k) — carve `__setstate__` legacy defaults (~357 lines) → `_mrmr_setstate_defaults.py`;
  `_VALID_*`/`_DEMOTED_*` validation constants → `_mrmr_param_constants.py`; the fit god-method's thread-local
  set/reset blocks → a contextmanager.
- `_mrmr_fe_step/_step_core.py` (~1.5k), `_feature_engineering_pairs/_pairs_core.py` (~1.6k),
  `_screen_predictors.py`, `shap_proxied_fs/_shap_proxied_fit.py` — each one giant orchestration function;
  lift the per-candidate scoring / external-validation / DCD-swap blocks into `_*_score.py` siblings (with a
  bit-identity selection pin -- these share mutable scratch).
- `_gpu_resident_fe.py` (~2.7k) → `_gpu_resident_select.py` (residency-buffer + radix-select) + `_gpu_fused.py`;
  `batch_mi_noise_gate_gpu.py` (~1.3k) → `_batch_mi_noise_gate_kernels.py`; `evaluation.py` (~1.16k, D6) → carve
  the prefill/JMIM block; training/* monoliths similarly.
- [x] `_orthogonal_univariate_fe/__init__.py` — already carved (→ ~860) via `_orth_dedup.py`; **de-exempted**.

RISK for all: these are single functions threading dozens of interdependent locals -> every carve MUST land
with a bit-identity / selection-equivalence pin. Do one at a time; the meta-test green is the structural gate.
