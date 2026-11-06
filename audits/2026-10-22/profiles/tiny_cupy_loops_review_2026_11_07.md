# Review of the 27 launch-bound cupy loops added to `_tiny_cupy_loops_baseline.json` (2026-11-07)

The meta-gate flags a `for` loop with several cupy calls per iteration. Each of the 27 entries added after the last merge was read, and the ones that could cost wall time were measured (`module_walls.py`, one warm 100000 x 100 every-family GPU fit, wall 93.8 s).

## Column-block / tile loops (13): the streaming the kernels are built around
`tied_replicate_mi_stream`, `nan_aware_quantile_codes_gpu`, `hsic_biased_columns_gpu`, `modular_family_codes_gpu`, `null_max_mi_perms_gpu`, `parity_mi_grids_gpu`, `resident_qbin_codes_block`, `sync_free_qbin_codes_block`, `quantile_codes_block_gpu`, `quantile_codes_resident_f32`, `triple_mi_gpu`, `_jmim_scores_device`, `radix_quantiles_grouped`. The block width comes from the free VRAM (`pool_column_block`: hundreds of columns), so a 100-column fit runs one or two iterations, each a full-size kernel. The loop is the VRAM-limited column streaming; it is launch-bound only when a card has too little memory, and then a smaller block is the memory-bound answer.

## Loops over a handful of folds or reference columns (not launch-bound, or too cheap to matter)
- `_honest_oj_fused_gpu.DeviceWinners.picks`: F = 3 to 5 folds, 2 cupy calls each over a `(B, P, M)` array. The fold order of the sum is kept on purpose so the totals equal the CPU's bit for bit.
- `_binagg_null_max_vector`: the loop runs only when `_batched_null_max` returned None (the batched path took 0.58 s in the fit).
- `_cmim_scores_device`: one `batched_cmi_gpu` call per reference column (about 40), each over all candidates. The per-reference host copy of the result is gone: the running minimum stays on the device and crosses once (`return_device=True`). Same-process A/B, 100000 rows x 150 candidates x 40 references, 5 warm runs each: 0.45 s against 0.47 s, i.e. within noise. The time is the batched CMI kernel itself (2.41 s for the two calls of a fit), not the copy. A kernel that takes all references in one launch would be the next step; it is not made here.
- `_fe_stability_vote._vote_recipe_over_folds`: a short chain of small operations per recipe and fold. Measured: `confirm_recipes_cross_fold` 0.53 s in the whole 93.8 s fit. Batching it would gain at most that.
- `_pair_admission_batch._score_groups`, `_block_verdicts`, `batched_asymmetric_relax_verdicts`: one batched CMI per block of candidates per anchor group, the batching is the design.
- `_fe_stage_concurrency.merge_stage_state`: no cupy in the loop; the detector reaches cupy through `_same`. It merges attributes of stage copies on the host.

## What stays on the worklist
Only the CMIM single-launch idea above. The rest is recorded in the baseline because the loop is the intended structure (streaming blocks) or costs under a second of a 94 s fit.
