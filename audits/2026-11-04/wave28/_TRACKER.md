# Wave 28 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: the neural training package (about 6 of 60 modules read), the dynamic cluster discovery / recipe / composite-transform modules, and the transformer modules from `l` to `z` (report pending). The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W28NN-1 | loss | Medium | the flat module's weighted multilabel loss summed B x K entries and divided by the sum of the B weights, so it was K times the unweighted mean; it now averages over the K labels |
| **RESOLVED** | W28NN-2 | precision | Medium | the recurrent module's zero-weight guard `1e-12` rounds to 0 in float16, so an all-zero-weight batch gave 0 / 0 = nan; the weighted mean is now taken in float32 (`_weighted_mean`) |
| **FUTURE** | W28RC-1 | replay | High | `_orth_basis_recipes.py:141-154,607-643`: NaN source values are filled with the mean of the batch being transformed instead of a frozen fit-time value |
| **FUTURE** | W28RC-2 | replay | High | `_orth_basis_recipes.py:82-84`: the `tanh` pre-transform std is recomputed from the replay batch and never stored in the recipe |
| **FUTURE** | W28GP-1 | precision | Medium | `_pairs_abs_corr_gpu.py:28,34-39`: the target is reduced in float32 on the device, so a target with a large offset collapses to constant and the GPU returns 0.0 where the CPU twin returns the true |r| |
| **FUTURE** | W28GP-2 | precision | Medium | `_pairs_abs_corr_gpu.py:84-94`: both operands are cast to float32 before centring, with the same collapse at large offsets |
| **FUTURE** | W28DC-1 | cache | Medium | `_dcd_metrics.py:37-39`: `_is_foreign` treats any same-shape array sharing memory with the state matrix as the state's own, so another column view reuses the cached SU/VI/entropy |
| **FUTURE** | W28CT-1 | scale | Low | `composite/transforms/simple.py:321,331,452,468`: the eps floor `sign(base + 1e-300) * eps` is 0 when `base == -1e-300` |
| **FUTURE** | W28CT-2 | ties | Low | `composite/transforms/unary.py:531-535`: tied knots map every tied y to the last duplicate position instead of the middle of the tie block |

## Not covered by this wave

- training/neural: `data.py`, `base/_base_fit*.py`, `base/_cuda_fallback.py`, `base/_base_losses.py`, the training/validation step bodies of `_flat_torch_module/__init__.py`, `flat.py`, `ranker.py`, `_ranker_losses.py`, the recurrent data/wrapper/arch/embedding modules, the optimizers (`_sam`, `_lookahead`, `_muon`), `_mixup.py`, `_numerical_embeddings.py`, `feature_prep.py`, `field_grouped_mlp.py`, `trunk_residual_mlp.py`, `tabular_1dcnn.py`, `fixed_sparse_linear.py`.
- `_dynamic_cluster_discovery/__init__.py`, `_pairs_dispatch.py`, `_pairs_chunks.py`, `_pairs_emit_extval.py`, `_pairs_score_step4.py` lines 20-289.

## Implemented

| ID | Regression test |
|---|---|
| W28NN-1, 2 | `tests/training/test_neural_weighted_loss_reductions.py` |
