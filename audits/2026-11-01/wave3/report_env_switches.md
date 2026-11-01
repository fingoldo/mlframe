# Wave 3 audit 2026-11-01: environment-variable switches

Scope: `src/mlframe` (read-only). Class: every environment variable read in `src` that turns off, relaxes or redirects a check, how its value
is parsed, whether children inherit it, whether the default is safe. Builds on the three earlier trackers; DESER-01, 02, 07, 08, SUB2-03,
FS2-1, FS2-2 and NET2-01 are not repeated.

The auditor could not write this file itself; the content below is its returned text. Of roughly 300 grep hits, about 150 performance and
GPU toggles were classified as out of scope because they relax no check (listed as a family in the inventory).

## Summary

11 findings: 0 High, 1 Medium, 6 Low, 4 Info.

| ID | Severity | Location | Variable (default) | Disposition |
|---|---|---|---|---|
| ENV4-1 | Medium | `training/io.py:538-541`, `utils/safe_pickle.py:35`, `training/core/_metadata_loader.py:17-23` | `MLFRAME_ALLOW_UNVERIFIED_PICKLE` (unset, closed) | DOC |
| ENV4-2 | Low | `training/_bounded_zstd.py:19-27` | `MLFRAME_MAX_DECOMPRESSED_BYTES` (64 GiB) | DOC |
| ENV4-3 | Low | `training/suite_artefact_cache.py:71-74` | `MLFRAME_SUITE_CACHE_DIR` (`~/.mlframe/cache/suite`) | DOC |
| ENV4-4 | Low | `training/core/_setup_helpers_pipeline_cache.py:125-127, 136-137` | `MLFRAME_PIPELINE_DISK_CACHE_PATH` (a temp-dir file) | DOC |
| ENV4-5 | Info | `_fe_family_budget.py:41`, `system/kernel_tuning_cache/_ensure.py:98-99`, `utils/_param_oracle.py:131` | `MLFRAME_FE_BUDGET_CACHE_DIR`, `PYUTILZ_KERNEL_CACHE_DIR` | REJECTED |
| ENV4-6 | Low | `_lgb_shim_helpers.py:222`, `xgb_shim.py:202`, `_ttr_eval_set_scaling.py:108`, `_prediction_envelope_clip.py:174`, `__init__.py:67, 97, 203, 299` | cache-disable and autoconfig switches | FUTURE |
| ENV4-7 | Info | `reporting/renderers/plotly.py:134-145` | `MLFRAME_PLOTLY_JS` (`cdn`) | DOC |
| ENV4-8 | Low | `training/neural/base/_base_fit_helpers.py:263-273` | `MLFRAME_TORCH_PROFILE` (`0`), `MLFRAME_TORCH_PROFILE_DIR` | DOC |
| ENV4-9 | Info | `_eval_helpers.py:124, 196`, `_flat_torch_predict_accel.py:108`, `_core_numba_warmup.py:393` | `MLFRAME_CAT_DIAG`, `MLFRAME_TORCH_COMPILE_DEBUG`, `MLFRAME_WARMUP_MAPE_DIAG` | DOC |
| ENV4-10 | Low | `utils/native_import_probe.py:73-77, 86-87` | per-module verdict variable (unset) | FUTURE |
| ENV4-11 | Low | `utils/safe_pickle.py:39-45` (callers `composite/cache_store.py:375-385`) | `MLFRAME_ALLOW_UNVERIFIED_PICKLE` | FUTURE |

## Findings

**ENV4-1 (Medium, DOC).** `MLFRAME_ALLOW_UNVERIFIED_PICKLE` skips the sha256 sidecar check before unpickling. `io.py` accepts
1/true/yes/on, while other callers use the pyutilz parser, and no mlframe test pins that set. Children inherit the variable, so one global
setting disables verification for the whole process tree. The default is safe and the error messages name the variable, but the README does
not. Follow-up: route `io.py:538` through the shared parser.

**ENV4-2 (Low, DOC).** A value of 0 or a negative number disables the decompression-bomb cap; a non-numeric value falls back to the
default with a warning. The 64 GiB default is intentional because frames can exceed 100 GB. Document that 0 means no cap.

**ENV4-3 (Low, DOC).** Redirects the directory pickles are loaded from; the path is not checked for ownership or permissions. Loads go
through `safe_load` with a sidecar, so a planted file fails closed unless ENV4-1 is also set. Document that the override must be private.

**ENV4-4 (Low, DOC).** Redirects the pipeline JSON cache and takes precedence over the monkeypatch route. The default directory is in
the shared temp dir and is created with no mode. The file is JSON, not pickle, so the worst case is a poisoned pipeline definition. Later
tightening: `mode=0o700` on the temp default, as with FS2-2.

**ENV4-5 (Info, REJECTED).** The FE budget and kernel tuning JSON caches are read with orjson: no code-execution path, worst case a
skewed tuning or budget choice. `_ensure.py:99` creates its directory with no mode, which falls under the FS2 class.

**ENV4-6 (Low, FUTURE).** Parsing is inconsistent. "Any non-empty string is on" means `=0`, `=false` and `=no` turn a safeguard off, the
opposite of intent; the four `__init__.py` switches treat only "", "0", "false", "False" as off, so "FALSE", "no" and "off" enable them.
Same bug class as INT-16, which led to `utils/env_flags.py`. All defaults are safe and none guards a trust boundary. Migrate each site to
`env_flag` with a regression test.

**ENV4-7 (Info, DOC).** By default saved plotly HTML loads plotly.js from a third-party CDN with no integrity hash, so a report opened
from a shared drive runs remote script. Recommend `MLFRAME_PLOTLY_JS=inline` for sensitive reports.

**ENV4-8 (Low, DOC).** Writes profiler traces to any directory, creating it. Default off; traces hold op names and shapes, not data.

**ENV4-9 (Info, DOC).** Diagnostic logging switches; the individual log lines were not read. Diagnostics may include column names; verify
they log shapes and counts rather than row values.

**ENV4-10 (Low, FUTURE).** A parent-exported "1" is trusted by children and skips the isolation probe, so a native import that would
crash runs in-process. A reliability issue, not code execution. Possible hardening: ignore an inherited "1" when the parent PID differs.

**ENV4-11 (Low, FUTURE).** Only `io.py` warns when the opt-in is used; the other load paths go through pyutilz, which does not, so a
globally set value is invisible in logs. Add one throttled per-process warning in `safe_pickle.py` when the opt-in is observed.

## Inventory of classified variables

| Variable | File:line | Default | Parsing |
|---|---|---|---|
| `MLFRAME_ALLOW_UNVERIFIED_PICKLE` | `utils/safe_pickle.py:35`, `training/io.py:538` | unset (closed) | `io.py`: 1/true/yes/on; pyutilz parser elsewhere |
| `MLFRAME_FEATURE_CACHE_STRICT` | `training/feature_handling/cache.py:575` | off | 1/true/yes/on |
| `MLFRAME_MAX_DECOMPRESSED_BYTES` | `training/_bounded_zstd.py:19` | 64 GiB | int; 0 or negative = no cap |
| `MLFRAME_SUITE_CACHE_DIR` | `training/suite_artefact_cache.py:71` | `~/.mlframe/cache/suite` | stripped path |
| `MLFRAME_SUITE_CACHE_MAX_BYTES` | `suite_artefact_cache.py:79` | `DEFAULT_BYTES_LIMIT` | int, `max(0, ...)` |
| `MLFRAME_PIPELINE_DISK_CACHE_PATH` | `training/core/_setup_helpers_pipeline_cache.py:125` | temp-dir file | any non-empty path |
| `MLFRAME_FE_BUDGET_CACHE_DIR` | `feature_selection/filters/_fe_family_budget.py:41` | `~/.cache/mlframe/fe_family_budget` | any path |
| `PYUTILZ_KERNEL_CACHE_DIR` | `system/kernel_tuning_cache/_ensure.py:98`, `utils/_param_oracle.py:131` | `~/.pyutilz/kernel_tuning` | stripped path |
| `MLFRAME_LGB_CACHE_DISABLE`, `MLFRAME_XGB_CACHE_DISABLE` | `training/_lgb_shim_helpers.py:222`, `training/xgb_shim.py:202` | off | any non-empty string is on |
| `MLFRAME_TTR_DISABLE_PREDICT_CLIP`, envelope-clip disable | `training/targets/_ttr_eval_set_scaling.py:108`, `training/_prediction_envelope_clip.py:174` | off | any non-empty string is on |
| `MLFRAME_NO_MACOS_OMP_AUTOCONFIG`, `MLFRAME_NO_CUDA_AUTOCONFIG`, `MLFRAME_KEEP_BROKEN_CUPY`, `MLFRAME_KEEP_COLORAMA_REINIT` | `__init__.py:67, 97, 203, 299` | off | off values: "", "0", "false", "False" |
| `MLFRAME_PLOTLY_JS` | `reporting/renderers/plotly.py:140` | `cdn` | enum: inline/embed/true/1/yes/on, directory, else cdn |
| `MLFRAME_PLOT_*` (forwarded to workers) | `reporting/_async_render.py:233, 212-213` | n/a | prefix `MLFRAME_PLOT_` copied |
| `MLFRAME_TORCH_PROFILE`, `MLFRAME_TORCH_PROFILE_DIR` | `training/neural/base/_base_fit_helpers.py:263, 269` | `0` / `<ckpt>/torch_traces` | `== "1"` / path |
| `MLFRAME_CAT_DIAG`, `MLFRAME_TORCH_COMPILE_DEBUG`, `MLFRAME_WARMUP_MAPE_DIAG` | see ENV4-9 | off | any non-empty / `== "1"` |
| `MLFRAME_TRUST_LGB_CUDA` | `training/_gpu_probe.py:66` | off | `== "1"` |
| per-module native-probe verdict | `utils/native_import_probe.py:73, 87` | unset | "0" or "1" |
| `MLFRAME_LGB_SERIALISE`, `MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD`, `MLFRAME_NUMBA_PARALLEL_GUARD` | `_lightgbm_thread_safety.py:38`, `training/lgb_shim.py:96`, `_numba_parallel_guard.py:53` | `1` / `0` / empty | off values 0/false/no/off; on values 1/true/yes/on; `== "always"` |
| `MLFRAME_LOAD_MODEL_CACHE_MAX_MB`, `MLFRAME_LOAD_MODEL_CACHE_MAX` | `training/io.py:573, 627` | 2048 / 32 | int |
| `MLFRAME_BOOTSTRAP_BACKEND`, `MLFRAME_DISABLE_BATCHED_PREPROCESS_SCAN`, `MLFRAME_FORCE_HELPER_PD_VIEW` | `evaluation/bootstrap.py:413`, `training/preprocessing.py:437`, `training/utils.py:488` | `threading` / unset / unset | string / `== "1"` |
| `HF_HOME`, `TRANSFORMERS_CACHE`, `env:VAR` references in `EmbeddingProvider.params` | `training/feature_handling/hf_provider.py:109, 111`, `providers.py:259` | unset | path / `KeyError` if unset |
| thread and platform variables (`OMP_NUM_THREADS`, `KMP_DUPLICATE_LIB_OK`, `LOKY_MAX_CPU_COUNT`, `TQDM_DISABLE`, `MPLBACKEND`, `JULIA_NUM_THREADS`, `CUDA_HOME`, `CUDA_PATH`, `PYTHONHASHSEED`) | `__init__.py`, `training/lgb_shim.py:106-107`, `training/__init__.py:95`, `training/pipeline/__init__.py:41-45`, `utils/misc.py:126` | set only if unset (`CUDA_*` overwritten) | `setdefault` or conditional |
| about 100 `MLFRAME_FE_*`, `MLFRAME_CMI_*`, `MLFRAME_MI_*`, `MLFRAME_MRMR_*`, `MLFRAME_SHAP_*`, `MLFRAME_DTW_*` and similar backend, device and size selectors | `feature_selection/filters/`, `shap_proxied_fs/`, `models/ensembling/`, `signal/dtw.py` and others | defaults in code | on/off flags or numeric caps; none gates a trust boundary (classified from name and grep line, not read one by one) |
| benchmark-only variables (`_benchmarks/`) | `_benchmarks/` directories | n/a | not production paths |

Limits stated by the auditor: the pyutilz parser's truthy set was not run or confirmed (the ENV4-1 inconsistency is inferred from `io.py`
against the delegated call in `utils/safe_pickle.py`).
