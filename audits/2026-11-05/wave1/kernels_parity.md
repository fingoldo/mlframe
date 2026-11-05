# Wave 1 (2026-11-05), attack class A7: numba and CUDA kernel safety, CPU/GPU parity

Read-only audit by one Opus 5.5 low-effort agent. Two confirmed defects and one informational item. The scope is large, so it was sampled rather than read kernel by kernel; "Not covered" says what was skipped. Nothing was edited or run. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| K7-01 | Medium | `src/mlframe/feature_selection/filters/_gpu_resident_shared.py:469-470`, `:501-502`, `:593-594` | The GPU `max`/`min` candidate ops do not propagate NaN, but the CPU reference does (`np.maximum`/`np.minimum` at `:427-428`, then `nan_to_num` at `:716`). |
| K7-02 | Low | `src/mlframe/feature_selection/filters/_fe_batched_mi.py:115-117`, `_null_max_perm_gpu.py:39-42`, `:91-93` | The in-kernel bin search puts NaN in bin 0. The `cp.searchsorted(side='right')` fallback it claims to match bit for bit (`_gpu_resident_discretize.py:219`) puts NaN in the last bin. |
| K7-03 | Info | `src/mlframe/feature_selection/filters/_null_max_perm_gpu.py:160`, `:186` | `per_launch` is floored at 1 with no check that `nbins*n_classes*4` fits in shared memory, so a large table fails to launch and the fallback is logged only at `debug`. |

### K7-01

Three kernels compute the operand grid on the GPU: `fused_gen`, `fused_gen_cm` (and its f32 twin) and `fused_gen_bin_hist`. They compute `v = (x > y) ? x : y` and `(x < y) ? x : y`, then zero any NaN or inf. Their docstrings, and `cpu_pair_candidate_mi`, the reference path the GPU must match (`:721`), say they mirror numpy. Numpy returns NaN whenever either side is NaN, and the scrub then turns that into 0.

Failure scenario: a raw operand column `a` has a NaN at row i and `b[i] = 3.0`. `max(identity(a), identity(b))` gives 3.0 on the GPU and 0.0 on the CPU; `min(b, a)` gives 3.0 on the GPU and 0.0 on the CPU. Every unary of `a` keeps the NaN (`log` uses `nanmin` for its shift), so all 8 x 8 max/min combinations differ on every NaN row, and the CPU and GPU paths then rank different candidate columns. Inf does not cause this, because both paths agree on it.

Precedent: the sibling `fe_materialise_cm` (`_gpu_resident_materialise.py:162-165`) and `extval_candidates` (`_gpu_resident_extval_fused.py:29-30`) already guard NaN explicitly. Not verified: whether every caller imputes operands before upload; the defect is live whenever a NaN-bearing column reaches these kernels.

### K7-02

The search loop `if (edge <= v) lo = mid + 1; else hi = mid;` never takes the first branch when v is NaN, so NaN gets code 0. `cp.searchsorted(..., side='right')` sorts NaN last and returns code `ne`. The kernel docstrings (`_fe_batched_mi.py:91-94`, `:492`, and `_gpu_resident_discretize.py:180-182`) promise these agree bit for bit. If the fused kernel raises, the warning fallback at `:214-219` silently moves NaN rows from the lowest bin to the highest.

Failure scenario: a candidate block with 5% NaN cells bins them into bin 0 on the fast path and bin `nbins-1` on the fallback; the MI changes, and so can the null floor. `extval_candidates` deliberately keeps NaN for ops 4, 5 and 7; whether that matrix reaches the MI kernels unscrubbed was not confirmed, which is why this is Low.

### K7-03

With `nbins=64` and `n_classes=200` the table is about 51 KB, over the 48 KB default shared-memory limit, and this kernel never sets `max_dynamic_shared_size_bytes`. The launch raises, both callers catch it (`_binned_numeric_agg_resident.py:428`, `_orth_mi_null_max.py:147`), and the run silently falls back to one launch per permutation, which is slower. The results stay correct. Fix: cap or return `None` up front and log the fallback with `log_throttle`, as the "say so when a gate makes a run slower" rule asks.

## Checked and clean

- Wide index arithmetic: `fe_materialise_cm`, `extval_candidates`, the `fused_gen*` kernels, `pair_mi_shared_fused`, `null_mi_perms(_bits)`, `mi_from_values` and `marginal_mi_entropies` all use `long long` for row and element indices; the int32 values are histogram cells bounded by shared memory.
- Shared memory: above 48 KB, `pair_mi_shared_fused` and `_gpu_pairs` both set the opt-in attribute; `marginal_mi_entropies` keeps within `MaxSharedMemoryPerBlock` with a 4 KB margin for its static arrays.
- Sorting: every device sort goes through `_device_sort_safe`; no bare `cp.argsort` or `cp.sort` in scope.
- `metrics/` prange kernels: `_max_abs_pct_error_kernel_par` uses a per-thread max array, the Spearman kernels use recognised `+=` reductions, the ranking kernels write per-group slots only; all use `fastmath=False`.
- `feature_engineering/hurst.py`: `_mfdfa_hurst_slope` runs `isnan` under `fastmath=True`, which can fold away, but the affected q~0 row is NaN on either branch, so the output does not change. Not reported.

## Not covered

- `hermite_fe/` CUDA paths, the `info_theory/_batch_kernels*` prange kernels, and `_family_maxt_null_v2_njit` were not read line by line.
- Of the 27 files with histogram-style `arr[idx[i]] += ` writes, `_parity_fe` and `_honest_oj_fused` were sampled; both are race-free because each iteration writes only its own local arrays.
- `cache=True` signature staleness, the GPU monitors and neural helpers under `training/`, and `shap_proxied_fs` TreeSHAP CUDA were not audited.
- The shared-memory size gate the callers of `fused_gen_bin_hist` use for `P1*nbins*K_y` was not reached.
