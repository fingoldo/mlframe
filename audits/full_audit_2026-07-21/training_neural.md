# training/neural (torch-based neural training) -- mlframe audit

## Scope

All 44 non-benchmark `.py` files under `src/mlframe/training/neural/**` were opened and read in full (12,501 LOC):

- `__init__.py`
- `_base_callbacks.py`
- `_base_logging.py`
- `_base_sklearn_params.py`
- `_base_tensor_helpers.py`
- `_categorical_embeddings.py`
- `_history_recorder.py`
- `_lookahead_optimizer.py`
- `_mixup.py`
- `_muon_optimizer.py`
- `_muon_triton_kernel.py`
- `_neural_numba_kernels.py`
- `_numerical_embeddings.py`
- `_ranker_losses.py`
- `_recurrent_arch.py`
- `_recurrent_cat_embeddings.py`
- `_recurrent_config.py`
- `_recurrent_data.py`
- `_recurrent_perf.py`
- `_recurrent_sequences.py`
- `_recurrent_torch_model.py`
- `_sam_optimizer.py`
- `_triton_bootstrap.py`
- `data.py`
- `feature_prep.py`
- `field_grouped_mlp.py`
- `fixed_sparse_linear.py`
- `flat.py`
- `group_causal_attention_mask.py`
- `keras_compat.py`
- `ranker.py`
- `recurrent.py`
- `recurrent_dataset_helpers.py`
- `tabular_1dcnn.py`
- `trunk_residual_mlp.py`
- `base/__init__.py`
- `base/_base_fit.py`
- `base/_base_fit_prep.py`
- `base/_base_losses.py`
- `base/_base_predict.py`
- `base/_cuda_fallback.py`
- `_flat_torch_module/__init__.py`
- `_flat_torch_module/_flat_torch_loss.py`
- `_flat_torch_module/_flat_torch_predict_accel.py`

**Not reviewed**: the `_benchmarks/` subdirectory (8 files, 580 LOC: `bench_extract_sequences_stack.py`, `bench_field_grouped_mlp.py`, `bench_fixed_sparse_linear.py`, `bench_group_causal_attention_mask.py`, `bench_tabular_1dcnn.py`, `bench_torch_from_numpy.py`, `bench_trunk_residual_mlp.py`, `profile_categorical_embeddings.py`). These are standalone dev/benchmark scripts not on any production import path and are excluded from the cluster's own "~44 files, ~12.5k LOC" scope statement (44 = 52 total minus these 8; 12,501 LOC matches exactly).

**Totals**: 44 files reviewed, 12,501 LOC reviewed.

I cross-checked several findings against `tests/training/neural/` (grep + targeted reads of `test_estimators.py`, `test_numerical_embeddings.py`) to confirm whether an existing test would have caught the issue; these checks are cited inline below.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness / reproducibility / sklearn-contract | src/mlframe/training/neural/base/_base_sklearn_params.py:32-46 | `get_params()` omits 10 of `PytorchLightningEstimator.__init__`'s 23 real parameters, so `sklearn.base.clone()` silently drops `random_state`, `class_weight`, `use_ema`, `ema_params`, `label_smoothing`, `focal_loss_gamma`, `focal_loss_alpha`, `capture_iteration_metrics`, `use_learnable_cat_embeddings`, `categorical_embed_dim`. |
| F2 | P1 | correctness | src/mlframe/training/neural/base/_base_fit.py:326-327 | `partial_fit(classes=...)` overwrites the correctly-sorted `self.classes_` (set at line 208-209 from `LabelEncoder.classes_`) with the caller's raw, possibly-unsorted `classes` array, desyncing `classes_[i]` from `predict_proba` column `i`. |
| F3 | P1 | GPU/CPU parity / robustness | src/mlframe/training/neural/recurrent_dataset_helpers.py:778-785, 967-975 | `RecurrentClassifierWrapper`/`RecurrentRegressorWrapper` predict-time `L.Trainer(accelerator=self._cfg.accelerator, ...)` bypasses the `safe_accelerator()` CUDA-broken-host probe that `_create_trainer` (fit path, line 478) uses, and has no `run_with_cuda_cpu_fallback`-style retry (unlike the flat MLP's `_predict_raw`). |
| F4 | P1 | sample-weight propagation | src/mlframe/training/neural/_recurrent_torch_model.py:516-520 | `RecurrentTorchModel.validation_step` hardcodes `sample_weights=None` when computing `val_loss`, silently ignoring `val_sample_weight` even when `RecurrentDataModule` threads it correctly into the batch -- early-stopping/checkpoint-selection run on an unweighted metric. |
| F5 | P1 | correctness / architecture | src/mlframe/training/neural/_numerical_embeddings.py:88-119 | `PeriodicLinearEmbedding.proj` is one dense `nn.Linear(in_features*per_feat_in, in_features*embed_dim)`; the docstring/comment claims "block-diagonal"/"per-feature independent Linear" but nothing enforces that -- every feature's periodic encoding is densely mixed into every other feature's output, and parameter/compute cost is O(in_features^2) instead of the intended O(in_features). |
| F6 | P1 | reproducibility / global-state leak | src/mlframe/training/neural/_recurrent_perf.py:50-76 | `maybe_enable_cudnn_rnn_autotune` sets the process-global `torch.backends.cudnn.benchmark = True` on Ampere+ and never restores it, silently changing the behaviour/determinism of every OTHER model trained later in the same process -- the same "global RNG/state mutation" class the codebase already fixed once in `ranker.py` (Wave 49). |
| F7 | P1 | memory discipline | src/mlframe/training/neural/data.py:76-91 | `TorchDataset`'s documented "2GB eager-vs-lazy" byte-size safety gate never fires for a `pandas.DataFrame` (which has neither `.nbytes` nor `.estimated_size`); `_bytes_estimate` silently resolves to 0, is treated as "unknown size, prefer eager", and a 100+GB pandas frame is always eagerly copied to a tensor regardless of size. |
| F8 | P2 | dead code / robustness | src/mlframe/training/neural/_base_logging.py:51-61 | `suppress_lightning_workers_warning` is defined, re-exported, and documented as "wrap the trainer.fit()/predict() invocations in this neural base" but is never called anywhere; the DataLoader `num_workers` `UserWarning` it exists to silence still fires on every fit/predict. |
| F9 | P2 | checkpoint/resume correctness | src/mlframe/training/neural/_muon_optimizer.py:144-225 | `MuonAdamWHybrid` has no `state_dict()`/`load_state_dict()` override, unlike its siblings `Lookahead`/`SAM` in the same optimizer family (both correctly forward to their wrapped optimizer(s)); a Lightning checkpoint resume of a Muon-hybrid-optimized model silently loses Muon momentum + AdamW moment-estimate state. |
| F10 | P2 | reproducibility / API contract | src/mlframe/training/neural/ranker.py:605, 784-792 | `MLPRanker(seed=...)` only seeds the `GroupBatchSampler` query-shuffle order, not the network weight-init RNG (torch's global RNG is intentionally left untouched) -- unlike the sibling `PytorchLightningEstimator.random_state`, which fully seeds torch via `L.seed_everything`; two `fit()` calls with the same `seed` are not reproducible. |
| F11 | P2 | memory discipline | src/mlframe/training/neural/recurrent_dataset_helpers.py:126, 765-790, 955-980 | `_RecurrentWrapperBase._prediction_cache` is an unbounded dict keyed by full-content hash, grown on every distinct `predict()`/`predict_proba()` call with no LRU cap (contrast the flat MLP's `_cuda_graph_predict_cache`, which has an explicit `_CUDA_GRAPH_PREDICT_CACHE_MAX` eviction) -- a permutation-importance-style loop issuing many distinct predicts per fit accumulates unbounded RAM. |
| F12 | P2 | memory discipline | src/mlframe/training/neural/data.py:401-418 | `TorchDataModule._convert_features_dtype` unconditionally `.astype("float32")`s the WHOLE train/val/test features frame at `setup()`, ahead of and regardless of `TorchDataset`'s byte-size-gated eager/lazy split -- a full extra copy of a large frame happens before that downstream gate (see F7) ever gets a say. |
| F13 | P2 | style / simplification | src/mlframe/training/neural/data.py:424-431 | `except (AttributeError, Exception):` -- `AttributeError` is already a subclass of `Exception`; the tuple is dead/pointless. |
| F14 | P2 | silent data drop | src/mlframe/training/neural/recurrent_dataset_helpers.py:671-682 | `MLPRanker._x_to_array` silently drops every non-numeric column via `select_dtypes(include=[np.number])` with no logged warning; a caller who forgot to encode a categorical column gets a silently smaller feature matrix instead of an error. |
| F15 | P2 | repo-convention / perf | src/mlframe/training/neural/_muon_triton_kernel.py:64-67, 300-311 | The per-device Triton-vs-eager calibration verdict is cached only in an in-process dict, "cleared between processes; cheap to repopulate" per its own comment -- doesn't use the repo's own `pyutilz.system.kernel_tuning_cache` convention for persisting per-hardware measured thresholds across runs. |
| F16 | P2 | test-gap tied to F1 | tests/training/neural/test_estimators.py:938-961 | `test_get_params_includes_all_init_params` hardcodes a static `required_params` list instead of introspecting `PytorchLightningEstimator.__init__`'s real signature, so it silently stopped guarding against the exact regression it exists to catch (`use_ema`, `label_smoothing`, `focal_loss_gamma`/`alpha`, `random_state`, `class_weight`, `use_learnable_cat_embeddings`, `categorical_embed_dim`, `capture_iteration_metrics` all postdate the test and are absent from its list -- see F1). |

### F1 -- `get_params()` drops 10 constructor params; `clone()` silently reverts them to defaults

`PytorchLightningEstimator.__init__` (`base/__init__.py:180-230`) accepts 23 parameters. `get_params()` (`_base_sklearn_params.py:32-46`) returns only 13 of them: `model_class`, `model_params`, `network_params`, `datamodule_class`, `datamodule_params`, `trainer_params`, `use_swa`, `swa_params`, `tune_params`, `tune_batch_size`, `float32_matmul_precision`, `early_stopping_rounds`, `monotonic_decline_patience`. Missing: `use_ema`, `ema_params`, `label_smoothing`, `focal_loss_gamma`, `focal_loss_alpha`, `capture_iteration_metrics`, `random_state`, `class_weight`, `use_learnable_cat_embeddings`, `categorical_embed_dim`.

sklearn's `clone()` calls `estimator.get_params(deep=False)` and constructs `klass(**those_params)` -- any parameter absent from that dict is simply never passed to the clone's constructor, so the clone silently falls back to the constructor's default (`random_state=None`, `class_weight=None`, `use_ema=False`, ...). Concretely: `cross_val_score`, `GridSearchCV`, `StackingClassifier`/`StackingRegressor`, any `Pipeline` step that clones, or any manual `sklearn.base.clone(estimator)` call on a `PytorchLightningRegressor`/`PytorchLightningClassifier` configured with e.g. `random_state=42` (for reproducibility) or `class_weight="balanced"` (for imbalance handling) will silently produce clones that are NOT seeded and NOT class-weighted, with no error or warning anywhere -- a correctness-critical behaviour change that is invisible to the caller. The `get_params()` docstring itself states "All __init__ parameters must be included for sklearn.base.clone() to work correctly," directly contradicting its own implementation.

There IS a dedicated regression test for this exact contract (`tests/training/neural/test_estimators.py::test_get_params_includes_all_init_params`), but it hardcodes a static list of params that predates several of the missing ones (see F16), so it currently passes despite the bug.

Suggested fix: rewrite `get_params()` to enumerate `inspect.signature(PytorchLightningEstimator.__init__).parameters` (skipping `self`) and read each via `getattr(self, name)`, deep-copying only the dict-typed ones -- eliminates the drift entirely instead of requiring a manual key list to be kept in sync forever.

### F2 -- `partial_fit(classes=...)` desyncs `classes_` from the label encoder's index space

In `_fit_common` (`base/_base_fit.py`), for `is_partial_fit and classes is not None`, `self.classes_` is set TWICE: first correctly at line 208-209 to `self._label_encoder.classes_` (guaranteed sorted, matching the index space `_label_encoder.transform`/`inverse_transform` use), then again at line 326-327 to `np.asarray(classes)` verbatim (the caller's raw, potentially unsorted array). If a caller passes `classes=[2, 0, 1]` (a legitimate "full label universe" argument per the sklearn `partial_fit` convention, and not necessarily sorted), `self.classes_` ends up `[2, 0, 1]` while the label encoder's index-to-label mapping is `[0, 1, 2]`. Any code that reads `self.classes_[i]` to interpret `predict_proba` column `i` (the sklearn public-attribute contract, and explicitly the intended fallback path documented at `_base_predict.py:373-376` for "estimators loaded from an older pickle that has classes_ but no encoder") gets the wrong label for that column.

Suggested fix: drop the redundant second assignment at line 326-327 (the first one, from the label encoder, is already correct and already ran a few lines earlier in the very same `if _classifier_single_label:` block), or explicitly re-derive it as `self._label_encoder.classes_` there too.

### F3 -- recurrent predict path has no CUDA-broken-host protection that fit has

`_create_trainer` (`recurrent_dataset_helpers.py:418-487`) resolves `accelerator=safe_accelerator(self._cfg.accelerator)` before building the fit-time `L.Trainer`, specifically to probe and downgrade to CPU on hosts where CUDA libraries are present but the runtime/driver is broken (documented at length in `_base_tensor_helpers.py`'s `_probe_cuda_is_usable`). `RecurrentClassifierWrapper.predict_proba`/`predict` and `RecurrentRegressorWrapper.predict` (lines 778-785 and 967-975) build their own `L.Trainer(accelerator=self._cfg.accelerator, ...)` directly, with no `safe_accelerator()` call and no `run_with_cuda_cpu_fallback`-style retry (the mechanism the flat MLP's `_predict_raw` uses for exactly this failure mode). On a host where `fit()` gracefully downgraded to CPU because the CUDA probe failed, `predict()`/`predict_proba()` on the SAME estimator can still crash with "CUDA error: an illegal memory access" because `self._cfg.accelerator` was never mutated to the resolved value.

Suggested fix: persist the resolved accelerator from `_create_trainer` (e.g. on `self._resolved_accelerator`) and use it (through `safe_accelerator()` again, or the cached value) at both predict call sites; ideally route recurrent predict through the same `run_with_cuda_cpu_fallback` helper `base/_cuda_fallback.py` already provides.

### F4 -- recurrent validation loss silently ignores sample weights

`RecurrentTorchModel.validation_step` (`_recurrent_torch_model.py:516-520`) calls `self._compute_weighted_loss(logits, batch["labels"], None)` -- the third argument is a hardcoded `None`, never `batch.get("sample_weights")`. `RecurrentDataModule.val_dataloader()` (`_recurrent_data.py:321-345`) DOES build its `RecurrentDataset` with `sample_weights=self.val_sample_weight` and `recurrent_collate_fn` DOES populate `batch["sample_weights"]` when present, so a caller who supplies `val_sample_weight` gets it silently discarded at the one place it's read. `EarlyStopping`/`ModelCheckpoint` both monitor `val_loss` (or `val_auprc`), so best-epoch selection and early-stopping decisions are computed on an unweighted metric even when the caller explicitly asked for weighted validation. Contrast the flat MLP sibling (`_flat_torch_module/__init__.py::validation_step`), which correctly threads `sample_weight` from `_unpack_batch` into `_compute_weighted_loss`.

Suggested fix: change line 519 to `self._compute_weighted_loss(logits, batch["labels"], batch.get("sample_weights"))`, mirroring `training_step`'s own handling on line 481-492 of the same file.

### F5 -- `PeriodicLinearEmbedding` is a dense cross-feature mixer, not the documented block-diagonal per-feature projection

The module docstring (`_numerical_embeddings.py:1-26`) and the constructor comment (lines 86-96) both describe `self.proj` as functioning like "in_features independent Linear(per_feat_in -> embed_dim)" with "the block-diagonal structure ... enforced implicitly because we keep per-feature slices separate." In the actual code, `self.proj = nn.Linear(in_features * per_feat_in, in_features * embed_dim)` is one ordinary dense `nn.Linear` applied to the FLATTENED `(N, D*(2K[+1]))` tensor (`forward`, lines 103-119: `per_feat.flatten(1)` then `self.proj(flat)`). Nothing about a plain `nn.Linear` enforces a block-diagonal weight matrix; gradient descent is free to (and generically will) learn nonzero weights connecting feature `i`'s periodic encoding to feature `j`'s output embedding slot, contradicting both the docstring's own claim and the cited RealMLP-TD/PLR paper design (each numerical feature is meant to get an independent embedding). Practically this also means the parameter count and matmul FLOPs of this layer are `O(in_features^2)` instead of the intended `O(in_features)` -- for a dataset with e.g. 200 numeric columns and the defaults (`embed_dim=8`, `n_frequencies=24` -> `per_feat_in=49`), `self.proj` alone is a 9800x1600 matrix (~15.7M params) versus the ~78K params a true block-diagonal/per-feature implementation would use, a 200x blow-up that risks overfitting and burns memory/compute on datasets this repo explicitly targets (100+ column tabular frames). `tests/training/neural/test_numerical_embeddings.py` covers shape/gradient-flow/basic-benefit but has no test asserting per-feature independence (e.g. a Jacobian-sparsity check), so this went uncaught.

Suggested fix: implement the projection as a batched per-feature matmul (`torch.einsum("bdk,dko->bdo", per_feat, weight)` with `weight` shaped `(in_features, per_feat_in, embed_dim)`, or `nn.Conv1d(..., groups=in_features)`) so the block-diagonal structure is actually enforced, and add a regression test that perturbing feature `i`'s input does not change feature `j`'s output slice.

### F6 -- recurrent cuDNN autotune leaks global process state across models

`maybe_enable_cudnn_rnn_autotune` (`_recurrent_perf.py:50-76`) sets `torch.backends.cudnn.benchmark = True` unconditionally when an LSTM/GRU/RNN recurrent model fits on an Ampere+ GPU, and this is never reset back to its prior value afterward (no try/finally, no `on_train_end` restore). This is a process-wide PyTorch flag: once one `RecurrentClassifierWrapper`/`RecurrentRegressorWrapper` fit flips it on, EVERY subsequent model trained in the same process (another recurrent fit, a flat MLP fit, a completely unrelated CNN) inherits `cudnn.benchmark=True`, which can pick non-deterministic cuDNN algorithms and select different (sometimes faster, sometimes not) kernels depending on prior process history -- i.e. results become dependent on fit ORDER within a process, not just on the data/seed. This is the identical bug class the codebase already diagnosed and fixed once, explicitly, in `ranker.py` ("Wave 49 (2026-05-20): drop global RNG mutations -- they silently overwrote caller's torch/numpy stream and broke reproducibility for any sibling code in the same process").

Suggested fix: snapshot `torch.backends.cudnn.benchmark` before the mutation and restore it in the wrapper's `fit()` after `trainer.fit()` returns (success or failure), or scope the flag change to a context manager around the `trainer.fit()` call only.

### F7 -- pandas DataFrames always take the "eager" (unbounded) path in `TorchDataset`, defeating the documented 2GB safety gate

`TorchDataset.__init__` (`data.py:74-104`) computes `_bytes_estimate` via `features.nbytes if hasattr(features, "nbytes") else features.estimated_size() if hasattr(...) else 0`. `numpy.ndarray` has `.nbytes`; `polars.DataFrame` has `.estimated_size()`; `pandas.DataFrame` has NEITHER (only `pandas.Series`/`Index` expose `.nbytes`). For any `pandas.DataFrame` input, `_bytes_estimate` is therefore always `0`, and the very next line treats `_bytes_estimate == 0` as `"unknown size, prefer eager (small frame)"`, setting `self._eager_features = True` unconditionally. The module's own docstring (line 74-77) explains this gate exists precisely to avoid doubling peak RAM "for HUGE frames (100GB+)" by falling back to a per-batch conversion path -- but that fallback path is unreachable for the pandas carrier the MLP/estimator's own fit path (`base/_base_fit.py`) routinely hands it (`X` arrives as a `pd.DataFrame` after the categorical-factorization / embedding-text steps).

Suggested fix: add a pandas branch, e.g. `features.memory_usage(deep=True).sum()`, to the byte-estimate chain.

### F8 -- `suppress_lightning_workers_warning` is dead code

`_base_logging.py:51-61` defines `suppress_lightning_workers_warning()`, whose docstring explicitly instructs "Wrap the trainer.fit() / trainer.predict() invocations in this neural base" -- it is re-exported from `base/__init__.py` but grep across the whole `training/neural` package finds no call site at all; `base/_base_fit.py`'s `trainer.fit(...)` call and `base/_base_predict.py`'s `t.predict(...)` call are both unwrapped. The Lightning DataLoader `num_workers` warning it exists to suppress therefore still fires on every fit/predict, contrary to the module's stated intent.

Suggested fix: wrap the `trainer.fit()`/`predict()` calls in `base/_base_fit.py` and `base/_base_predict.py` (and the `RecurrentDataModule`-based equivalents) with this context manager, or remove the helper if it's genuinely obsolete.

### F9 -- `MuonAdamWHybrid` doesn't checkpoint its wrapped optimizers' state

`Muon`'s AdamW-hybrid facade (`_muon_optimizer.py:144-225`) delegates `step()`/`zero_grad()` to two internally-constructed optimizer instances (`self._muon`, `self._adamw`) but never overrides `state_dict()`/`load_state_dict()`. It inherits `torch.optim.Optimizer`'s defaults, which serialize/restore `self.state` and `self.param_groups` -- but `self.state` on `MuonAdamWHybrid` itself is never written to (all real per-parameter state lives in `self._muon.state`/`self._adamw.state`, which the base `Optimizer.state_dict()` has no way to see). Contrast the sibling wrappers in the same optimizer family: `Lookahead.state_dict()`/`load_state_dict()` (`_lookahead_optimizer.py:139-199`) and `SAM.state_dict()`/`load_state_dict()` (`_sam_optimizer.py:128-141`) both explicitly forward to their wrapped optimizer(s), with `Lookahead`'s docstring even citing a prior audit finding ("F-A fix... Pre-fix the slow weights were dropped on save") for exactly this bug class. Because the estimator's own checkpoint callback uses `save_weights_only=True` and `save_last=False` (`base/_base_fit.py:489-510`) the default flow never exercises this, but any caller enabling normal Lightning resume (`ckpt_path=...`, or a custom `ModelCheckpoint(save_weights_only=False)`) with `optimizer=MuonAdamWHybrid` silently loses all Muon momentum buffers and AdamW moment estimates on reload.

Suggested fix: add `state_dict()`/`load_state_dict()` to `MuonAdamWHybrid` that compose `{"muon": self._muon.state_dict() if self._muon else None, "adamw": self._adamw.state_dict() if self._adamw else None}`, mirroring the pattern already used by `Lookahead`/`SAM`.

### F10 -- `MLPRanker(seed=...)` doesn't fully seed training, unlike the sibling estimator's `random_state`

`ranker.py:784-792` explicitly documents dropping "global RNG mutations" so `MLPRanker.fit` never calls `torch.manual_seed`/`L.seed_everything`; `self.seed` only feeds `GroupBatchSampler`'s numpy-based query shuffle order. Network weight initialization (`generate_mlp(...)` inside `fit`, line 806-817) and dropout draw from whatever the process's global torch RNG state happens to be at call time -- unseeded by `self.seed`. `PytorchLightningEstimator.random_state` (the sibling flat-MLP/recurrent estimators in the very same package) DOES fully seed torch/numpy/python via `L.seed_everything(int(self.random_state), workers=True)` (`base/_base_fit.py:110-122`). A user reasonably expects `MLPRanker(seed=42).fit(X, y, group_ids)` run twice to be bit-identical (that's the constructor param's name and docstring: "Seed for model init and training"); it is not, since weight init differs across the two runs unless the caller ALSO separately calls `torch.manual_seed` themselves.

Suggested fix: either seed `torch.manual_seed`/`torch.cuda.manual_seed_all` locally (save/restore the prior global RNG state around the fit, so sibling code in the same process isn't polluted) or update the docstring/parameter name to make clear only the query order is seeded.

### F11 -- `_RecurrentWrapperBase._prediction_cache` grows without bound

`recurrent_dataset_helpers.py:126` declares `self._prediction_cache: dict[bytes, np.ndarray] = {}`; every `predict()`/`predict_proba()` call (lines 765-790, 955-980) computes a full-content hash key (`_compute_cache_key`, hashing the ENTIRE input array's bytes) and stores the full output array under it, with no size cap, no LRU eviction, and no TTL -- the dict is only ever cleared at the START of the next `fit()` call (`_clear_cache()`). The flat MLP's analogous CUDA-graph predict cache (`_flat_torch_module/_flat_torch_predict_accel.py`) explicitly caps itself via `_CUDA_GRAPH_PREDICT_CACHE_MAX` with LRU eviction precisely because "an inference run over many distinct batch shapes ... would otherwise grow VRAM without bound" -- the same reasoning applies here to host RAM. A permutation-importance loop, or an OOF-prediction loop calling `predict()` many times per fold with different feature subsets/rows, accumulates one full-size cached array per distinct call for the whole lifetime of the (possibly long-lived) fitted estimator object.

Suggested fix: cap `_prediction_cache` with an LRU eviction policy (mirroring the flat MLP's CUDA-graph cache), or drop content-hash caching in favour of a simple "last call only" memo if the intent is just to dedupe an immediately-repeated call.

### F12 -- `_convert_features_dtype` copies the whole frame ahead of the size-gated dataset conversion

`TorchDataModule._convert_features_dtype` (`data.py:401-418`) runs at `setup()` (called automatically by Lightning before every fit/test/predict stage) and unconditionally does `features.astype("float32")` on the full `train_features`/`val_features`/`test_features`/`predict_features` frame whenever it isn't already float32 -- this always allocates a full copy of the frame BEFORE `TorchDataset.__init__`'s byte-size-gated eager/lazy conversion (see F7) has any chance to matter, since the frame handed to `TorchDataset` is already the post-`.astype()` copy.

Suggested fix: skip the eager `.astype()` for frames above the same byte-size threshold `TorchDataset` uses, letting the per-batch conversion path (already dtype-aware via `features_dtype`) do the cast lazily instead.

### F13 -- redundant exception tuple

`data.py:424-431`: `except (AttributeError, Exception):` -- `AttributeError` is already covered by `Exception`; harmless but pointless (a lint/readability nit).

### F14 -- `MLPRanker._x_to_array` silently drops non-numeric columns

`recurrent_dataset_helpers.py:671-682`: for a `pd.DataFrame` input, `X.select_dtypes(include=[np.number])` silently discards every non-numeric column with no logged warning. The docstring frames this as "defence-in-depth" for a caller who forgot to strip qid/target columns, but it equally masks a caller who forgot to *encode* a genuine categorical feature -- the model silently trains on fewer features than intended, with `n_features_in_` reflecting the reduced (not original) width and no signal that anything was dropped.

Suggested fix: log a warning naming the dropped columns when any are discarded.

### F15 -- Muon-Triton per-device calibration isn't persisted via the shared kernel-tuning cache

`_muon_triton_kernel.py:64-67` caches the Triton-vs-eager calibration verdict in a plain module-level dict, explicitly documented as "Cleared between processes; cheap to repopulate" -- every new process re-pays the one-shot calibration cost instead of reusing a persisted, per-hardware measurement the way this repo's own convention (`pyutilz.system.kernel_tuning_cache`, referenced by CLAUDE.md for "New GPU dispatchers") is meant to provide.

Suggested fix: route `_TRITON_VERDICT` through `pyutilz.system.kernel_tuning_cache` so the calibration survives across process restarts on the same host.

### F16 -- the clone-safety regression test itself has drifted stale

`tests/training/neural/test_estimators.py::test_get_params_includes_all_init_params` (lines 938-961) hardcodes a `required_params` list that predates `use_ema`, `ema_params`, `label_smoothing`, `focal_loss_gamma`, `focal_loss_alpha`, `capture_iteration_metrics`, `random_state`, `class_weight`, `use_learnable_cat_embeddings`, and `categorical_embed_dim` -- exactly the params missing per F1. The test currently passes, giving false confidence that `get_params()`/`clone()` are safe.

Suggested fix: replace the hardcoded list with `inspect.signature(PytorchLightningEstimator.__init__).parameters.keys() - {"self"}` so the test can never silently fall behind the real constructor again.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | test-coverage | tests/training/neural/ | No `test_sklearn_api_compliance_*` exists for `PytorchLightningEstimator`/`Regressor`/`Classifier`, unlike the parallel `test_sklearn_api_compliance_keras.py` and `test_sklearn_api_compliance_recurrent.py` -- would have caught F1 directly. |
| PR2 | test-coverage | tests/training/neural/test_numerical_embeddings.py | Add a per-feature-independence (Jacobian sparsity) test for `PeriodicLinearEmbedding` -- would have caught F5. |
| PR3 | test-coverage | tests/training/neural/ | Add a regression test that `RecurrentClassifierWrapper`/`RegressorWrapper.predict()` succeeds after a `fit()` that fell back to CPU via the CUDA-broken-host probe (would catch F3 class of bug going forward). |
| PR4 | test-coverage | tests/training/neural/ | Add a regression test that `fit(..., eval_set=..., sample_weight=...)` (or the recurrent equivalent, `val_sample_weight`) actually changes `evals_result_["val"]`/early-stopping behaviour vs an unweighted eval_set, for both the flat MLP and the recurrent estimator (would catch F4). |
| PR5 | refactor | src/mlframe/training/neural/_recurrent_perf.py | Scope `torch.backends.cudnn.benchmark` mutation with a context manager / explicit save-restore instead of a permanent global flip (fixes F6). |
| PR6 | perf | src/mlframe/training/neural/recurrent_dataset_helpers.py | Cap `_prediction_cache` with LRU eviction (fixes F11), mirroring the flat MLP's `_cuda_graph_predict_cache` pattern already in this same package. |
| PR7 | ML best practice | src/mlframe/training/neural/ranker.py | Either fully seed `MLPRanker` (network init too) or rename/redocument `seed` to make the partial-determinism contract explicit (fixes F10). |
| PR8 | docs | src/mlframe/training/neural/field_grouped_mlp.py | Module docstring already carries an "honest-negative note" that the hypothesized generalization win did not reproduce -- consider linking the actual bench/ablation that established this, for future readers deciding whether to invest further here. |

## Coverage notes

- The `_benchmarks/` subdirectory (8 files, 580 LOC) was not reviewed in depth; these are standalone profiling/benchmark entry points (`if __name__ == "__main__"` scripts), not imported by any production code path, and fall outside the cluster's own stated "~44 files, ~12.5k LOC" scope.
- I did not execute any test, benchmark, or training run (read-only audit per instructions); all findings are from static reading plus targeted `grep`/test-file reads to check whether an existing test would catch each issue. Where I state "no test covers X" this is based on `grep`-based search of `tests/training/neural/` and `tests/training/ranking/`, not a full pytest collection run.
- I did not trace every call site of `PytorchLightningEstimator`/`RecurrentClassifierWrapper`/`MLPRanker` outside `training/neural/` (e.g. the composite-ensemble / suite orchestration layer in `training/composite/` and `training/`) to confirm whether F1's `clone()` gap or F3/F4's recurrent gaps are actually exercised in a real multi-model training suite run; the findings are established from the estimator code itself and sklearn's documented `clone()` contract, not from an observed production failure.
- The two excluded packages (`feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`) were not touched, per the task's exclusion list; no file under `training/neural/` imports from either.

```json
{"cluster": "training_neural", "report_file": "training_neural.md", "files_reviewed": 44, "loc_reviewed": 12501, "p0": 1, "p1": 6, "p2": 9, "proposals": 8, "headline": "PytorchLightningEstimator.get_params() omits random_state, class_weight, use_ema and 7 other constructor params, so sklearn.base.clone() (used by cross_val_score/GridSearchCV/StackingClassifier/etc.) silently drops seeding and class-rebalancing on every clone with no error or warning."}
```
