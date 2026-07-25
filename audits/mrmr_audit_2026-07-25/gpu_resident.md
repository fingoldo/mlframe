# GPU_RESIDENT — audit (2026-07-25)

Scope: the GPU-resident FE / MI / discretize / select / permutation-null kernels where data stays on device.
Files audited (all under `src/mlframe/feature_selection/filters/`): `_gpu_resident_basis.py`,
`_gpu_resident_discretize.py`, `_gpu_resident_extval.py`, `_gpu_resident_fe.py`, `_gpu_resident_histgate_ktc.py`,
`_gpu_resident_k_chunk_ktc.py`, `_gpu_resident_materialise.py`, `_gpu_resident_radix_ktc.py`,
`_gpu_resident_rank_bin.py`, `_gpu_resident_select.py`, `_gpu_resident_select_kernels.py`, `_resident_bincount.py`,
`_resident_candidate_mi.py`, `_resident_candidate_mi_ktc.py`, `_resident_raw_mi.py`, `_fe_resident_operands.py`,
`_permutation_null_resident.py`, `_permutation_null_pair_resident.py`, `_permutation_null_resident_ktc.py`,
`_permutation_null_shufflegen_ktc.py`. These build FE candidates / bin codes / permutation-null floors entirely on
the device (H2D-once operand caches, no per-candidate re-upload) with a host-CPU twin as the documented fallback.

## Prior-audit GPU findings — verified status against current source (git `d8091a138`)

- **GPU_INFRA_B-1** (blanket `clear_resident_codes_handoff()`): **FIXED** — `_gpu_resident_materialise.py:685,861`
  removed the unconditional blanket clear (relies on bounded-FIFO eviction).
- **GPU_INFRA_B-3** (`_RADIX_STATIC_SHARED_BYTES` v3 20-byte gap): **FIXED** — `_gpu_resident_select_kernels.py:281`.
- **GPU_INFRA_B-4** (`nbins<=255` int8-overflow comment + no guard): **FIXED** — comments corrected to `nbins<=128`
  and a real `ValueError` guard added at `_gpu_resident_materialise.py:692` (and `:855`) and
  `_gpu_resident_discretize.py:75`. (But NOT propagated to the extval sibling — see GPU_RESIDENT-2.)
- **GPU_INFRA_B-9** (`_env_gpu_default_on` inline opt-out reimplementation): **FIXED** — `_gpu_resident_fe.py:191`
  now delegates to `_gpu_policy.gpu_globally_disabled()`.
- **GPU_INFRA_C-3** (`best_existing_op_mi_resident` zero parity test): **FIXED** — a dedicated parity test now exists
  (`tests/feature_selection/gpu/test_best_existing_op_mi_resident_parity.py`).
- **GPU_INFRA_C-4** (`resident_bincount` UB-vs-docstring): **FIXED** — docstring corrected and an opt-in
  `debug_check_bounds` bounds-assert added (`_resident_bincount.py:30,48-51`).
- **SCREEN_CONFIRM_B-5** (order-1 maxT floor no logging/no breaker): **FIXED** for the order-1 path
  (`_permutation_null_resident.py:33-57` breaker + caller logging in `_permutation_null.py:387-438`). The order-2
  twin re-introduces the same failure mode — see GPU_RESIDENT-1.

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| GPU_RESIDENT-1 | P2 | reliability / broad-except-masks-fault | `_permutation_null_pair_resident.py:84,194-195` | The order-2 resident pair-maxT floor `pooled_pair_permutation_null_joint_mi_floor_cupy` wraps its ENTIRE body in `try: … except Exception: return None`, swallowing every cupy/CUDA fault internally and returning `None`. Its caller (`_mrmr_fe_step_helpers.py:439-452`) trips the process circuit breaker `trip_pair_maxt_gpu_circuit_breaker()` **only inside its own `except`**, which can never fire because the callee never re-raises. So `_PAIR_MAXT_GPU_FAILED` is never set via the fault path, and there is zero logging in the callee's except — the exact "silently re-faults every call with no trace" mode the order-1 twin's SCREEN_CONFIRM_B-5 fix eliminated. The order-1 twin `pooled_gain_floor_perms_cupy` deliberately has NO body-level try/except (lets the fault propagate so the caller trips the order-1 breaker); the two twins are asymmetric and the order-2 one defeats its own breaker. Also masks any logic bug (shape/contract error) as a benign CPU-fallback. | On a real WDDM-TDR `cudaErrorLaunchFailure` (documented on the project's GTX 1050 Ti) during the order-2 pair floor: callee catches it → returns None → CPU njit floor runs (correctness OK this call) BUT breaker stays un-tripped → `pair_maxt_perm_null_gpu_enabled` returns True next FE step → GPU re-attempted on the poisoned context → re-faults → swallowed again. Every subsequent FE step of the fit pays a futile multi-second GPU launch fault with no log line. |
| GPU_RESIDENT-2 | P2 | dtype-overflow (latent, B-4 class) | `_gpu_resident_extval.py:50-53,86-87` | `gpu_materialise_extval_codes_host` defaults `dtype=np.int8` and does an unconditional `codes_dev.astype(np.int8)` on the binned 0..nbins-1 codes with NO narrowing guard — unlike the B-4-fixed sibling `gpu_materialise_discretize_codes_host` (`_gpu_resident_materialise.py:692`) which raises `ValueError` when the dtype cannot hold `nbins-1`. `int8` is C `signed char` (−128..127), so a code of e.g. 150 wraps to −106. | Direct caller (public `__all__` export) invoking `gpu_materialise_extval_codes_host(pa, ext, ops, nbins=200)` with the default int8 gets silently negative-wrapped bin codes → corrupted extval MI. Unreachable via the sole production caller today (`_pairs_emit.py:315,337` passes `_narrow_code_dtype(quantization_nbins, quantization_dtype)`, which caps at the safe `nb<=127` int8 threshold), so latent — but the guard the B-4 fix added was never propagated to this sibling. |
| GPU_RESIDENT-3 | P3 | house-convention (leftover audit metadata) | 8 files (see list) | Leftover finding-ID / `.md #`-ref / date-stamped audit metadata in code comments, violating CLAUDE.md "No process/audit metadata in code comments … finding IDs, date stamps". The prior repo-wide cleanup missed these. Sites: `_resident_bincount.py:16` (`GPU_INFRA_C-4 fix:`); `_permutation_null_resident.py:33` (`SCREEN_CONFIRM_B-5 fix:`); `_gpu_resident_discretize.py:75` (`GPU_INFRA_B-4 fix:`); `_gpu_resident_fe.py:188` (`GPU_INFRA_B-9 fix:`), `:219` (`X_EFFICIENCY_ARCHITECTURE-6 fix:`); `_gpu_resident_materialise.py:337,366` (`FE_PAIRS_CORE-1 fix`), `:685,861` (`GPU_INFRA_B-1 fix:`), `:693,802,855` (`GPU_INFRA_B-4 fix`); `_gpu_resident_select_kernels.py:281` (`GPU_INFRA_B-3 fix,`); `_gpu_resident_radix_ktc.py:139` (`found 2026-07-18:`); `_fe_resident_operands.py:45,206` (`X_EDGE_CASES_BEST_PRACTICES-1 fix:`), `:108,178` (`.md #6, 2026-07-21`). | N/A (comment hygiene). Each should be reworded to state only the WHY (invariant/rationale) without the finding ID / date / audit-file reference. |
| GPU_RESIDENT-4 | P3 | correctness-gate laxity | `_permutation_null_resident_ktc.py:115` | The resident-vs-njit KTC sweep uses `equiv_rtol=5e-2, equiv_atol=5e-2` while the module's own docstring states the two paths differ only in FP reduction order (~1e-15). That is 13 orders of magnitude looser than the real divergence; on MI values as small as 0.001-0.5 a 5e-2 absolute tolerance can exceed the value itself, so the sweep alone could crown a genuinely-divergent resident kernel as "fastest". Same class as the prior GPU_INFRA_C-3 (candidate-MI sibling, since given a tight parity test). Mitigated here: a dedicated tight equivalence test exists (`tests/feature_selection/info_theory/test_perm_null_resident_equiv.py`), so the loose sweep tol is not the only correctness gate — hence P3, not P1. | Not a live wrong-selection today (tight parity test covers it); the sweep-gate tolerance itself remains far too loose to be a correctness backstop, worth tightening to `~1e-9`/`1e-12` to match the sibling KTC sweeps and the documented divergence. |

## Non-findings / confirmed-clean angles

- **Device-error → CPU fallback** on every resident entry point: verified. `best_existing_op_mi_resident` /
  `gate_grid_mi_resident` / `resident_raw_baseline_mi` / `rank_bin_codes_*` / `plugin_mi_classif_batch_rank_cuda_resident`
  / `gpu_materialise_extval_codes_host` all return `None` on any cupy fault and the caller takes the exact host path.
  The order-1 permutation-null floor (`pooled_gain_floor_perms_cupy`) correctly propagates its fault to the caller,
  which logs + trips the order-1 breaker + falls back. (The order-2 twin does NOT — GPU_RESIDENT-1.)
- **GPU opt-out honored**: `_env_gpu_default_on` (`_gpu_resident_fe.py:191`) now delegates to
  `gpu_globally_disabled()`; `permnull_use_resident`, `pair_maxt_perm_null_gpu_enabled`, `shufflegen_use_gpu`,
  `rescand_use_resident` all route through the STRICT/opt-out gates. The `_permutation_null.py` order-1 caller
  additionally checks `gpu_globally_disabled()` before engaging.
- **H2D-once residency / no per-candidate re-upload**: `_fe_resident_operands.py` content-keyed cache (with the
  X_EDGE_CASES_BEST_PRACTICES-1 device-id key fix confirmed present at `:217,222`) is lock-guarded
  (`_FE_RESIDENT_OPERANDS_LOCK`); candidate matrices are device-born + transient (never cached). Confirmed correct.
- **Unlocked module-level caches**: `_fe_resident_operands.py` caches are all `threading.Lock`-guarded. The small
  geometry caches in `_gpu_resident_rank_bin.py:70` (`_BIN_BOUNDARIES_CACHE`) and the `_gpu_resident_*` KTC
  `_*_SPEC` singletons are pure-function-of-key (worst case redundant recompute, never a wrong value) — benign.
- **`_permnull_resident_ktc` / `_shufflegen_ktc` / `_resident_candidate_mi_ktc`**: no hardcoded engage thresholds;
  all decisions route through `kernel_tuning_cache` sweeps with a documented fallback. Compliant with
  `feedback_use_kernel_tuning_cache_for_gpu`.
- **Dynamic-shared / CUDA-budget routing**: `_searchsorted_codes` / radix-select edge path fall back to
  `cp.percentile` when the radix path returns `None` (R over cap / shared-mem over device limit); the `K==1`
  `cp.percentile` cupy bug is guarded (`_gpu_resident_discretize.py:207-212`).
- **mypy**: no new implicit-Optional / return-type issues spotted in the audited files (annotations use `Optional`/
  union `| None` consistently). No SQL/HTTP/eval/exec/subprocess/pickle-of-untrusted surface anywhere in the cluster.

## Proposals (perf / refactor / test — not bugs)

1. **GPU_RESIDENT-1 fix**: either remove the body-level `try/except` in
   `pooled_pair_permutation_null_joint_mi_floor_cupy` and let the fault propagate (matching the order-1 twin so the
   caller trips the breaker), or trip `_PAIR_MAXT_GPU_FAILED` + log inside the callee's except before returning None.
   Add a regression test that raises a simulated `cudaErrorLaunchFailure` inside the callee and asserts
   `_PAIR_MAXT_GPU_FAILED is True` afterward (fails today).
2. **GPU_RESIDENT-2 fix**: mirror the `_gpu_resident_materialise.py:692` `ValueError` guard into
   `gpu_materialise_extval_codes_host` before the `.astype(dtype)` narrowing.
3. **GPU_RESIDENT-4**: tighten `_run_permnull_sweep`'s `equiv_rtol/atol` to `~1e-9`/`1e-12` to match the documented
   ~1e-15 divergence and the sibling KTC sweeps, so the sweep gate is itself a correctness backstop, not only the
   separate parity test.
