# MRMR FS/FE — active work plan (2026-06-22)

Living plan for the outstanding MRMR work. Ordered by the agreed priority: **validatable optimizations /
hygiene first; the distribution-robustness (residual-aware FE) goal LAST** (per request). Each item lists:
files, the exact change, the VALIDATION GATE (what must stay/go green), and the risk. Mark `[x]` when landed
(with the commit). Every change keeps the standing invariants: CPU/no-CUDA path byte-unchanged; selection
bit-identical where claimed; no unvalidated selection change ships; commit+push per item.

## DONE earlier this session (context)
- [x] ~21 critique findings (all P0/P1/P2 equivalency divergences + safe perf/quality) — see MRMR_AUDIT_2026_06_22.md
- [x] n_iters 5→2 for KTC sweeps (785c212 / fa767dbc)
- [x] CPU noise-gate restructure — batch_mi_with_noise_gate prange-over-cols + contiguous + reused histogram (900dd660)
- [x] GPU#1 — hoist fit-constant y min/max out of per-chunk resident MI (ca85bf3f)
- [x] GPU#5 — dedup heavy-tail detection across routing's bases (1dc6af37)
- [x] GPU#2 — VERIFIED already-done (operand residency wired in production: build_resident_operand_table); agent mis-attributed to the prototype path
- [x] F2 100k wall report + dual-profiler (perf notes 29cb613a); distribution-robustness GOAL encoded as self-policing xfail (53a12960)

## TIER A — perf / hygiene (validatable; do first)

- [ ] **A1. lstsq→normal-eq in `_deflate_sincos`** (perf #8). File: `_orth_extra_basis_fe.py:336`. Change:
  normal-equations solve on the 3-col [1,sin,cos] design + lstsq fallback on LinAlgError (keeps rank-
  robustness on degenerate freq). GATE: test_fe_auto_escalation + test_biz_value_mrmr_adaptive_fourier +
  test_biz_val_extra_basis_fe green; F2 single_compound pin no-regress. RISK: selection-bearing (not bit-
  identical, SVD vs normal-eq) but low live impact (Fourier path escalation-gated). [IN PROGRESS]
  - Also the sibling vander-poly lstsq sites `_orth_extra_basis_fe.py:449,589` (held-out poly deflation) —
    same normal-eq+fallback substitution, same gate.
- [ ] **A2. `y.to_numpy()` 53× hoist** in `_fit_impl_core.py`. Hoist `_y_np = y.to_numpy() if hasattr(y,
  "to_numpy") else np.asarray(y)` ONCE (y is never reassigned in `_fit_impl`); replace the 53 inline copies.
  GATE: confirm via grep that `y` is not reassigned between the hoist point and the last use; run
  test_mrmr_feature_engineering + a biz-value layer subset. RISK: mechanical (53 sites in a 9.8k file) —
  verify each replacement; behavior-preserving (same array).
- [ ] **A3. `_env_truthy` / env-flag DRY** across the `fe_gpu_*_enabled` gates + pairs env flags. NOTE: the
  gates have DIVERGENT logic (opt-in `in (1,true,...)`; opt-out default-on with `=0` + CUDA_VISIBLE_DEVICES
  + MLFRAME_DISABLE_GPU). So TWO helpers: `_env_opt_in(name)` and `_env_opt_out_cuda(name)` (the default-on-
  when-CUDA pattern). GATE: import smoke + the gate-default tests (test_gpu_routing_parity gate-off,
  test_plugin_mi_classif_dispatch). RISK: a wrong default flips a gate — replicate each gate's exact logic;
  behavior-preserving.
- [ ] **A4. ctor-defaults single-source** in `_mrmr_class.py` — derive `__setstate__` defaults from
  `_ctor_defaults()` (already drifted once: cluster_aggregate_mode). GATE: pickle round-trip tests +
  test_mrmr_edges_coverage. RISK: back-compat (legacy pickles) — keep the legacy dict's extra keys.

## TIER B — module hygiene / carves (validatable)

- [ ] **B1. evaluation.py carve** (1144 LOC, over the meta-test ceiling). Carve a sibling (e.g. the
  prefill/JMIM block) + re-export, per the monolith-split pattern. GATE: test_no_file_over_1k_loc green for
  evaluation.py + the MRMR greedy/CMI tests. RISK: parallel-session-owned file (last touched 2026-06-19) —
  rebase carefully; behavior-preserving (verbatim move + re-export).
- [ ] **B2. source-name `__` split** (hidden #5). 9 sites across `_orthogonal_univariate_fe/`. Carry the
  source col in metadata instead of `name.split("__",1)[0]` (mis-stems one-hot `col__value` → infinite
  uplift). GATE: a new test with a `col__value`-named source + the recipe round-trip tests. RISK: invasive
  naming-convention change — must preserve recipe replay byte-exactness.

## TIER C — distribution-robustness (residual-aware FE) — LAST, per request

Goal test (self-policing): `test_f2_single_compound_across_distributions` — uniform passes (hard guard);
scaled_1_5 / heavy_tailed / mixed / with_outliers are xfail(strict=True). Root cause: signal-scale imbalance
(a²/b dominates Var(y) → the weak log(c)·sin(d) half falls below the prevalence gate → fragments).

- [x] **C0. Foundation** — `SufficientSummaryVerdict.residual` surfaces r = y − E_hat[y|selected] (53a12960).
- [ ] **C1. Retarget wiring**. `_fit_impl_core.py:6731`: when `_ss_verdict.reached` is False AND
  `blocking_raw >= 0` AND `residual is not None`, set `self._fe_residual_target_continuous_ =
  verdict.residual` for the next FE step (clear after). `_step_core.py:1108-1118`: when set, use the
  discretised residual as classes_y / _prewarp_y_cont / _usab_y_cont for that step and recompute cached_MIs
  against it (the 1.05 prevalence constant unchanged — the residual makes the weak half clear it). Default-on
  gated by residual-signal-present (no-op on balanced/complete targets). GATE: the goal test (flip the
  xfails → pass, then REMOVE the xfail marks) + single_compound pin no-regress + a no-noise-admission test
  (y = a²/b + f/5, assert no new c/d/e). RISK (documented `_mrmr_class.py:1728-1731`): a prior admission
  relaxation admitted (c,d) but CONSTRUCTION still failed — verify the residual step actually builds the
  clean mul(log(c),sin(d)); if not → C2.
- [ ] **C2. Separable warp-product proposer** (only if C1's construction fails the goal test). A terminal
  (no-fusion) proposer that, on the residual, builds the clean log(c)·sin(d) product directly. GATE: same as
  C1.

## Execution rule
One item at a time: implement → validate against its GATE → commit+push → next. Never end a turn with an
item half-done-and-unvalidated; if context runs out mid-edit, commit what is safe + note the exact remaining
step here. Distribution-robustness (Tier C) is LAST.
