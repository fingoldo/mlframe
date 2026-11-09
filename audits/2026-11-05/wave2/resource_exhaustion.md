# Wave 2: resource exhaustion audit (mlframe, src/mlframe)

Scope: n x n / n x k allocations sized by input, unbounded caches, decompression limits, GPU budgets. Read-only; every finding below was read at the cited lines.

| id | sev | file:line | disposition |
|---|---|---|---|
| RES-W2-1 | High | src/mlframe/training/composite/qrf.py:294-319 | RESOLVED-needed |
| RES-W2-2 | Medium | src/mlframe/training/_bounded_zstd.py:14, :84 | RESOLVED-needed |
| RES-W2-3 | Low | src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:107-121 | DOC |
| RES-W2-4 | Info | rf_proximity.py:35/100, graph_spectral_features.py:95, qrf.py:208, io.py:667/678 | REJECTED (already bounded) |

## RES-W2-1 (High): QRF fit keeps about 9 GB of per-tree leaf tables at 2M rows
- File: `src/mlframe/training/composite/qrf.py:294-319` (`_LeafResidualForest.fit`)
- Statement: fit stores `train_leaves_` (n_train x n_trees, int64), `_sorted_train_idx_` (n_trees x n_train, int64), `_leaf_inv_` (n_trees x max_leaf, f64) and `_leaf_offsets_` (n_trees x (max_leaf+1), int64). None of these has a size check, and all of them live on the fitted estimator.
- Scenario: defaults are `n_estimators=200` and `min_samples_leaf=5`. At n=2M:
  - `train_leaves_` = 2M x 200 x 8 B = 3.2 GB.
  - `_sorted_train_idx_` = 3.2 GB.
  - Node ids go up to about 2n/5 = 800k, so `_leaf_inv_` and `_leaf_offsets_` take about 1.3 GB each.
  - Total is about 9 GB on top of the sklearn forest. The same 9 GB is pickled into the model bundle.

  On a 16-32 GB host the fit OOMs or thrashes. Predict-time weights are already batched (`_PREDICT_BATCH=512`); only fit is unbounded.
- Proposed fix:
  - Store indices as int32 when n_train < 2^31.
  - Index `_leaf_inv_` and `_leaf_offsets_` by compacted leaf rank (number of leaves, about n/5) instead of raw node id, which halves those tables.
  - Exclude `train_leaves_` from pickling (it can be rebuilt via `apply`) or drop it, since the kernel uses only the CSR tables.
  - Add a byte-budget check that logs through `log_throttle` and raises above a configurable ceiling.

## RES-W2-2 (Medium): bounded-zstd ceiling defaults to 64 GiB, and `max_output_size` can preallocate it
- File: `src/mlframe/training/_bounded_zstd.py:14` (`_DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 1024**3`) and `:84` (`dctx.decompress(data, max_output_size=limit)`); caller `src/mlframe/training/core/_metadata_loader.py:34`.
- Statement: the decompression-bomb ceiling is larger than the RAM of most hosts, so in practice it limits nothing. In `decompress_bounded` the one-shot path also passes the same 64 GiB as `max_output_size`.
- Scenario 1 (one-shot path): python-zstandard sizes its output buffer from `max_output_size` when the frame header has no content size. A `.pkl.zst` metadata file written by a streaming compressor (or crafted to omit the size) therefore triggers a 64 GiB allocation attempt. The result is an immediate MemoryError, or commit-charge thrash on Windows, before any data is read.
- Scenario 2 (streaming path, `io.py:667`): a 1 MB bomb that expands to 40 GB stays under the cap and exhausts a 32 GB host.
- Proposed fix:
  - Lower the default to a fraction of physical RAM (for example `min(64 GiB, 0.5 * psutil.virtual_memory().total)`).
  - For frames with unknown content size, decompress through `stream_reader` + `BoundedReader` instead of `max_output_size`.

## RES-W2-3 (Low): F-order CMI cache bounds bytes per entry but not the total
- File: `src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:107-121`
- Statement: each entry is capped at `MLFRAME_CMI_FORDER_MAX_MB` (4 GB by default), but there is no cap on entry count or total bytes. Entries are purged only after their source array is garbage-collected.
- Scenario: two or three concurrent fits in one process (threaded suite, or several retained `factors_data` arrays of 2M x 500 int32, about 4 GB each) keep a full F-order copy alive per source. That doubles the resident footprint by up to 4 GB per live source.
- Proposed disposition: DOC the total-memory behaviour. Optionally add a total-bytes cap with FIFO eviction, since a miss only rebuilds a copy.

## RES-W2-4 (Info): candidates checked and found bounded
- `models/rf_proximity.py:55`: the n x n proximity matrix is guarded by `max_n=20000` (:100).
- `feature_engineering/graph_spectral_features.py:56`: the dense adjacency is guarded by `max_nodes=2000` (:95).
- `feature_selection/filters/friend_graph.py:284`: n is the size of the selected feature set (small).
- `training/composite/qrf.py:92/342`: the weight matrix is built in query batches of 512 (:208, :379).
- `training/io.py:667`: model bundle reads go through `BoundedReader` (subject to RES-W2-2).
- `_LOAD_MODEL_CACHE` (`io.py:678`) is capped at 32.
- These caches have FIFO caps: `_PREP_CACHE`, `_ADVERSARIAL_CACHE`, `_PSI_CACHE`, `_MATRIX_CACHE`, `_CB_POOL_CACHE`, `_CB_VAL_POOL_CACHE`, `_DRIFT_INVARIANT_CACHE`, `_INFONET_Y_PREP_CACHE`.
- `metrics/classification/_weighted_kappa.py:27`: n is the number of classes.

Not covered in this wave: recursion depth and the GPU VRAM budget checks beyond the sites above. Only the module-level `*CACHE*` dict pattern was swept; instance-level memo dicts were not.
