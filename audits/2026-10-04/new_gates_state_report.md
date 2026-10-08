# New py-ci-shared gates wired into mlframe (security / state)

Gates run over src/mlframe, min_files=500, no baselines.

| Item | Disposition |
|---|---|
| unsafe_deserialization: training/io.py dill.load | RESOLVED: explicit safe=False opt-in carries `# deserialize-ok:` |
| unsafe_deserialization: infonet/infer.py torch.load | RESOLVED: always weights_only=True, TypeError fallback removed; test_infonet_weights_only_load.py now requires zero fallbacks |
| unsafe_deserialization: 10 hits under _benchmarks | REJECTED as scope: benchmark scripts, gate excludes "/_benchmarks/" |
| id_keyed_cache: _hermite_fe_mi shifted-y cache | RESOLVED: weakref + identity check in `_shifted_y_cached()`; test_hermite_shifted_y_cache_identity.py |
| persisted_negative_probe | 0 findings, gate wired |
| cuda_kernel_integer_width: gpu.py shared kernels (2) | RESOLVED: gid_offset/my_row/row are long long; test_joint_hist_shared_kernels_grid_beyond_int32_offset.py |
| cuda_kernel_integer_width: _gpu_resident_fe.py, _fe_batched_mi_cmi.py (2) | DOC: `// width-ok:` (bounded by shared memory) |
| Meta tests | test_no_unsafe_deserialization, test_id_keyed_cache_validates_identity, test_no_persisted_negative_hardware_probe added; test_cuda_kernel_sources_use_explicit_64bit_types replaced by the shared gate (include_advisory=True) |

Verification: 11 new/touched tests pass, 14 existing hermite/kernel-equivalence tests pass; gates flag 2 CUDA + 1 id-cache findings on the HEAD sources; ruff, mypy (_hermite_fe_mi), filtered black clean.
