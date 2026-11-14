# _mi_greedy_cmi_fe.py (CMI-greedy FE construction) and its carved modules: profile record (2026-11-14)

Family: `_mi_greedy_cmi_fe` (the construction loop, the host scorer `cmi_from_binned_fixed_yz` with its numba kernels), `_mi_greedy_cmi_fe_steps` (the round steps, `_batched_cmi_argmax`, the candidate-batch gate), `_mi_greedy_cmi_fe_gpu` (the cupy scorer `_cmi_from_binned_cupy`, resident cardinalities), `_mi_greedy_cmi_fe_binning` (candidate binning) and `_mi_greedy_cmi_fe_ktc` (the host-or-device tuner entry). The device work runs on the kernels of `_fe_batched_mi_cmi` (joint histograms and entropies, profiled in `_fe_batched_mi_cmi.md`).

## cProfile of the whole fit (100000 x 100, strict-resident default fit, one warm pass)

| function | calls | own s | cumulative s |
|---|---|---|---|
| `greedy_cmi_fe_construct` | 3 | 0.00 | 1.58 |
| `_batched_cmi_argmax` (batched scoring of a round) | 15 | 0.00 | 0.38 |
| `_block_argmax` (one column block of the round) | 25 | 0.00 | 0.38 |
| `cmi_device_argmax` | 25 | 0.00 | 0.25 |
| `_cmi_from_binned_cupy` (per-candidate fallback / gate scorer) | 1047 | 0.05 | 1.43 |
| `batched_cmi_gpu` (all callers, the gate included) | 1625 | 0.17 | 4.11 |
| `joint_entropy_gpu` | 5603 | 0.22 | 1.45 |

The construction itself is 1.6 s of the fit; the scoring rounds are batched (one `batched_cmi_gpu` workload per column block, 25 blocks for 15 rounds). The remaining host time sits in the callers (the redundancy gate's conditional nulls, measured in `_fe_cmi_redundancy_gate`), not in this family.

## Kernels (nvprof, same fit)

Kernel launches per call: `batched_joint_entropy2` 1432 launches over 1625 `batched_cmi_gpu` calls (about 0.9 launches per call), `batched_joint_hist2_i32` 621 and `batched_joint_hist2_i32_u8` 334 (about 0.6 launches per call together), `rows_ent_nnz_i32_lut_split` 497, `rows_ent_nnz_i32_lut4` 1352, the four-entropy kernel `cmi_joint_entropies` 126 launches for the 1047 per-candidate scorer calls (the fused path serves the shapes whose joint fits shared memory). GPU time of these kernels: 2.08 s + 1.09 s + 0.55 s + 0.35 s + 0.18 s, about 4.3 s of the 98 s of GPU time of the fit (4.4 %); the candidate scoring is therefore not the lever of the fit, the radix select (36 %) and the materialisation (12 %) are.

Occupancy and throughput of the family's main histogram kernel were measured with the nvprof counters of `_fe_batched_mi_cmi.md` (shared joint-histogram kernel: achieved occupancy 0.89, device-memory read throughput 81 GB/s of the card's ~112 GB/s, about 73 % of its bandwidth; the entropy kernels reach 38 GB/s at the production shapes). The utilisation of the card during the family's stages is bounded by the stages around it: the GPU is idle while the host builds the next round.

## Host-or-device choice

The batched device scorer replaces the per-candidate host loop when the call is large enough. The former fixed floors (64 candidates, 1e6 candidate-rows) are now the pre-sweep fallback of the tuner entry `cmi_candidate_batch_backend` (`_mi_greedy_cmi_fe_ktc`). Sweep on the development card (GTX 1050 Ti), 30000 rows, 10 bins, 30-class support: host for at most 120000 candidate-rows (4 candidates), device from 480000 (16 candidates) up to 30.7 million; both bodies give the same selection view (max abs difference 0.0). The fixed floors sent the batches of 16 to 63 candidates to the host.

## Streaming

The scoring rounds run in column blocks (`run_column_blocks` with `first_block`, sized from the free VRAM): a block that fits is scored in one workload, a larger candidate set goes through in several, and the first maximum over the blocks in order is the first maximum over all candidates.
