# Strict-fit device-to-host traffic: why `test_strict_fit_copies_back_only_result_columns` fails (2026-10-29)

The test allows `10 * n * 8` bytes (2.4 MB at n = 30000) of device-to-host copies in the strict F2 audit fit; the measured total is 3,275,410 B (3,295,159 B in the pytest run). It fails identically on a clean HEAD worktree (verified), so it is not caused by the kernel changes of 2026-10-29. Diagnosis with call-site capture (`D:/Temp/traffic_diag2.py`, 70 bulk copies):

| bytes | copies | call site | what it is |
|---:|---:|---|---|
| 793,152 | 51 x 15,552 | `_batch_mi_noise_gate_bits.bits_gate_mi` via `batch_mi_with_noise_gate_cuda_resident` | per-pair MI vector (3888 float32), a score table: legitimate |
| 480,000 | 2 x 240,000 | `_lazy_host_codes.host` from `_step_inert_operands.filter_inert_leaf_block:95` | int64 codes of a pair block read on the host for the inert-leaf CMI test |
| 480,000 | 4 x 120,000 | `_recipe_unary_binary_gpu.apply_unary_binary_gpu` via `_recipe_dispatch` | replayed columns copied back |
| 240,000 | 2 x 120,000 | `_step_score_parts2._device_codes_to_host` from `_discretise_survivor_block` | survivor block codes |
| 240,000 | 1 x 240,000 | `_lazy_host_codes.host` from `_heldout_gate.selected_design_columns` | held-out gate design columns |
| 240,000 | 1 x 240,000 | `_lazy_host_codes.host` from `_mi_greedy_cmi_fe._cmi_from_binned_cupy` (now moved to `_fe_cmi_redundancy_null.__init__`) | lazy codes read by the host |
| 138,720 | 1 | `_usability_pool_resident.score_pair_combos_table_resident` | usability table |
| 111,888 + 111,888 + 55,944 x 2 | 4 | `_honest_null_accept._continuous_accept_block`, `_family_maxt_cuda_bits._run` | honest-null accept blocks |

An attempted fix of one site (`_cmi_from_binned_cupy` taking `LazyHostCodes.dev` instead of reading the host copy) gave no gain: the same lazy array is then read by `_fe_cmi_redundancy_null.__init__`, so the 240 KB copy only moves to the next reader. The total comes down only when every reader of the same lazy codes stays on the device, or when the budget is re-based on the legitimate traffic (score tables 0.79 MB plus the result columns). Both are decisions for the owner of the residency contract; no change made. (The attempted patch was reverted.)
