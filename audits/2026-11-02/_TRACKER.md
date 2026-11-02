# Wave 4 audit 2026-11-02 -- tracker

Four read-only reports on `src/mlframe` and `.github`, by Opus 5.5 low-effort agents, one per attack class: log forging and leakage, weak crypto and hashing, GitHub Actions injection, native index safety. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (safe, or fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | LOG5-1 | log_leakage | Low | `utils/_param_oracle.py:117-120`: hostname hashed in the fallback host key |
| **DOC** | LOG5-2 | log_leakage | Low | label lists in exception messages (`evaluation/bootstrap.py:625,754`, `reports.py:810`, `venn_abers.py:231`, `conformal_classification.py:133`) |
| **RESOLVED** | LOG5-3 | log_leakage | Low | `reporting/_async_render.py:233` env snapshot filtered to `MLFRAME_PLOT_*` |
| **RESOLVED** | LOG5-4 | log_leakage | Info | `__init__.py:119` CUDA path probe |
| **REJECTED** | LOG5-5 | log_leakage | Info | benchmark script copies the environment |
| **DOC** | CRY5-1 | crypto | Low | `utils/disk_cache.py:101-120` non-adversarial cache key |
| **RESOLVED** | CRY5-2 | crypto | Low | three in-process memo keys moved to a 128-bit digest |
| **REJECTED** | CRY5-3 | crypto | Low | `hash()` seeds are over integer tuples; fingerprints are in-process |
| **DOC** | CRY5-4 | crypto | Info | truncated digests (`provenance.py:103`, `_fe_family_budget.py:229`) |
| **REJECTED** | CRY5-5 | crypto | Info | `replay.py:96` compares non-secret hashes |
| **RESOLVED** | CI5-1 | ci_injection | Low-Medium | fork PR coverage could feed `codecov-full`: lookup filtered to push events, PR upload skipped |
| **RESOLVED** | CI5-2 | ci_injection | Info | numba-coverage lookup relies on the producer's gate (safe today) |
| **RESOLVED** | CI5-3 | ci_injection | Info | `ci.yml:551-559` test-durations commit job is push-only |
| **RESOLVED** | CI5-4 | ci_injection | Info | `macos-abort-probe.yml:146` ref input is writer-only |
| **RESOLVED** | MEM5-1 | native_index | Low | `_group_aware_shuffle` raises on an out-of-range group id |
| **RESOLVED** | MEM5-2 | native_index | Low/Info | `_scatter_factorize_lookup` and `_dense_renumber_codes` raise on an out-of-range code |
| **DOC** | MEM5-3 | native_index | Info | TreeSHAP kernel trusts node ids from the fitted model |

## Counts

17 findings: 0 High, 0 Medium, 1 Low-Medium, 7 Low, 1 Low/Info, 8 Info. Dispositions: 10 RESOLVED, 4 DOC, 3 REJECTED, 0 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| LOG5-1 | Hash the host name in the Param-Oracle fallback host key | `tests/utils/test_param_oracle_host_key_hashed.py` |
| MEM5-1, MEM5-2 | Range-check the codes three categorical njit kernels index with | `tests/feature_selection/filters/test_native_kernel_index_checks.py` |
| CI5-1 | Keep fork pull_request coverage out of the merged codecov report | workflow change, checked by YAML parse |
| CRY5-2 | Key three in-process memos on a 128-bit content digest | `tests/feature_selection/filters/test_memo_keys_use_content_digest.py` |

## Not covered by this wave

- The large RawKernel families (`_gpu_resident_select_kernels.py`, `_gpu_resident_basis.py`, `_fe_batched_mi.py`, `_batch_mi_noise_gate_kernels.py`, `_batch_pair_mi_cuda_*`, `_cmi_cuda.py`, `_hermite_fe_mi.py`, `transformer/_kernels_cupy.py`): int32 products such as `i * K` were not checked. Next wave.
- f-string log calls that put `{col}` without `!r`: a follow-up grep settles it.
- Not run in this wave: public API validation, symlinks in output paths, `pip-audit`, determinism, GPU context and shared memory.
