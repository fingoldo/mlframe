# new_gates_state checkpoint (paused)

## Files changed so far (all parse; none tested yet)
- src/mlframe/training/io.py: dill.load line now carries `# deserialize-ok: ...` (safe=False opt-in).
- src/mlframe/feature_selection/filters/_vendored/infonet/infer.py: torch.load always weights_only=True, TypeError fallback removed.
- src/mlframe/feature_selection/filters/_hermite_fe_mi.py (CRLF): `import weakref`, new `_shifted_y_cached()` helper (weakref + identity check), call site uses it.
- src/mlframe/feature_selection/filters/gpu.py: `gid_offset`, `my_row`, `row` now `long long` in the two shared kernels.
- src/mlframe/feature_selection/filters/_gpu_resident_fe.py:865 and _fe_batched_mi_cmi.py:769: `// width-ok:` suppressions (bounded by shared memory).
- New tests: tests/feature_selection/gpu/test_hermite_shifted_y_cache_identity.py, tests/feature_selection/gpu/test_joint_hist_shared_kernels_grid_beyond_int32_offset.py (GPU, grid 2**23+2048 blocks, not yet run).

## Gate findings (find_* over src/mlframe, min_files=500)
- unsafe_deserialization: 12 (10 in _benchmarks incl. test_hybrid_tree_member.py: exclude "/_benchmarks/"; io.py suppressed; infonet fixed) -> now 10 benchmark-only.
- id_keyed_cache: 1 (_hermite_fe_mi) fixed -> 0. persisted_negative_probe: 0. cuda_kernel_integer_width: 4 advisory -> 0 (2 fixed, 2 suppressed).

## DONE (see new_gates_state_report.md); old remaining list:
1. Write 4 meta tests in tests/test_meta/: unsafe deserialization (exclude=("__pycache__","/_benchmarks/")), id-keyed cache, persisted negative probe, and REPLACE tests/test_meta/test_cuda_kernel_sources_use_explicit_64bit_types.py with the shared gate (include_advisory=True). Every test def needs a docstring.
2. Update tests/feature_selection/test_infonet_weights_only_load.py: second test still expects one `except TypeError` fallback; now require zero fallbacks and all loads weights_only=True.
3. Run new + touched tests (gpu tests tiny; verify the 64-bit test would fail on old source via a scratch RawKernel probe), existing tests in test_hermite_mi_y_resident_upload.py and test_gpu_kernel_equivalence.py, the meta -k subset, mypy and ruff on edited src, black_filtered_apply on edited files.
4. Write audits/2026-10-04/new_gates_state.md (hits, fixes, suppressions, baselines: none).

## Next exact step
Create the four meta test files, then run:
`PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 python -m pytest tests/test_meta/<new files> tests/feature_selection/gpu/test_hermite_shifted_y_cache_identity.py -n 1 -s --durations=200 --no-cov --timeout=0 > log 2>&1`
