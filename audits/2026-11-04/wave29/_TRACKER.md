# Wave 29 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: the neural base/flat modules, the neural recurrent/ranker/optimizer modules, and the pair-search dispatch/chunk/external-validation modules. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED, FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W29NB-1 | encoding | High | `base/_base_fit_helpers.py:104-116`: a full `fit()` kept the label encoder of an earlier fit, so a second fit with other classes raised, and a subset of classes kept the old class count and the wrong head; a full fit now re-learns the classes (only `partial_fit` continues) |
| **FUTURE** | W29NB-2 | crash | Medium | `base/_base_fit_helpers.py:120-124`: validation labels are encoded with the encoder fitted on the training labels only, so a class present only in validation raises |
| **FUTURE** | W29NB-3 | sampling | Medium | `data.py:236-243,615`: in internal-batch mode each batch is a fixed contiguous slice; `shuffle=True` reorders batches but never changes which rows share a batch; a last batch of one row with BatchNorm is unconfirmed |
| **FUTURE** | W29NR-1 | optimizer | High | `_sam_optimizer.py:206-213`: `SAM.step(closure)` assumes Lightning ran the closure first; under 32-true it perturbs with the previous batch's gradient, under 16-mixed it raises (depends on Lightning behaviour, not verified) |
| **RESOLVED** | W29NR-2 | restore | High | `ranker.py:839-870`: `MLPRanker.fit` adds `EarlyStopping` but never reloads the best weights, so `predict` uses the last epoch's; `_BestWeightsCallback` snapshots the weights of the lowest `val_loss` epoch and reloads them at train end (added whenever early stopping is on) |
| **FUTURE** | W29NR-3 | restore | Medium | `_lookahead_optimizer.py:221-245` with `_recurrent_torch_model.py:708-723`: the best-checkpoint reload overwrites the slow weights `on_train_end` installed |
| **FUTURE** | W29NR-4 | crash | Medium | `_recurrent_arch.py:141`: positional encoding `max_len=5000` raises for padded sequence length >= 5000 (the CLS token adds one) |
| **FUTURE** | W29NR-5 | nan | Medium | `tabular_1dcnn.py:158-250`: no NaN check or imputation, one NaN cell makes every weight and prediction NaN without an error |
| **RESOLVED** | W29NR-6 | crash | Low | `_recurrent_data.py:302-311`: the stratified sampler branch does not exclude 2-D (multilabel) labels; the stratified-sampler branch requires 1-D labels (one-line guard mirroring the wrapper's own copy; no separate regression test: no existing fixture builds the data module with 2-D labels) |
| **RESOLVED** | W29NR-7 | ties | Low | `fixed_sparse_linear.py:44-45`: the 1e-9 jitter is below float32 resolution for importances near 1, so `random_state` does not break ties; ties are broken by a seeded random permutation and a stable sort |
| **RESOLVED** | W29PX-1 | crash | Medium | `_pairs_chunks.py:318-322` with `_pairs_score_steps.py:371`: a chunk where every transform fails stores a 3-tuple but the consumer unpacks 4 values; the early return stores the four-field entry (candidates, None, times, None) the consumer unpacks |
| **RESOLVED** | W29PX-2 | ties | Medium | `_pairs_emit_extval.py:304-305`: the external factor subset is drawn per config, so tied leaders are compared on different factor sets and the winner depends on iteration order; the subset is drawn once per pair before the config loop and passed already capped |
| **FUTURE** | W29PX-3 | gate | Low | `_pairs_emit_extval.py:233-257`: a config with a failed chunk is left out of the family but keeps its partial validation MI |

## Not covered by this wave

- training/neural: `base/_base_fit.py`, `base/_base_fit_steps.py`, `base/_cuda_fallback.py`, `base/_base_losses.py`, `_flat_torch_predict_accel.py`, `_history_recorder.py`, `feature_prep.py`, `_base_tensor_helpers.py`, `_recurrent_config.py`, `_flat_layers.py`, `field_grouped_mlp.py`, `trunk_residual_mlp.py`, `keras_compat.py`, most of `ranker.py`, `_recurrent_wrapper_base.py` (partly), `_muon_optimizer.py` (partly).
- pairs: `_pairs_dispatch.py`, `_pairs_emit.py` (family gate bodies), `_dynamic_cluster_discovery/__init__.py`.

## Implemented

| ID | Regression test |
|---|---|
| W29NB-1 | `tests/training/test_neural_refit_relearns_the_class_labels.py` |
| W29NR-2 | `tests/training/neural/test_mlp_ranker_restores_the_best_epoch_weights.py` |
| W29PX-2 | `tests/feature_selection/fe/test_fe_extval_factor_subset_is_drawn_once_per_pair.py` |
| W29PX-1 | `tests/feature_selection/fe/test_fe_pair_chunk_where_every_transform_fails_keeps_the_entry_shape.py` |
| W29NR-7 | `tests/training/neural/test_fixed_sparse_linear_tie_break_follows_the_seed.py` |
