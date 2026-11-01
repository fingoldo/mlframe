# Second-wave audit 2026-11-01: tracker

Six read-only reports on `src/mlframe`, one per class: YAML and config parsing, numba and CUDA kernel memory safety, crypto and hash
primitive choice, logging and exception disclosure, downloads and provenance, public API input validation. Vocabulary as in the first
wave: RESOLVED, FUTURE, DOC, REJECTED. Earlier rounds are in `../` and are not re-listed.

## Counts (from the six reports)

| Class | Report | High | Medium | Low | Info |
|---|---|---|---|---|---|
| YAML and config | [yaml_config.md](yaml_config.md) | 0 | 0 | 1 | 5 |
| Kernel memory safety | [kernel_memory.md](kernel_memory.md) | 0 | 1 | 2 | 5 |
| Crypto and hash | [crypto_hash.md](crypto_hash.md) | 0 | 0 | 1 | 4 |
| Logging and disclosure | [logging_disclosure.md](logging_disclosure.md) | 0 | 0 | 2 | 4 |
| Provenance | [provenance.md](provenance.md) | 0 | 0 | 2 | 3 |
| API validation | [api_validation.md](api_validation.md) | 0 | 0 | 2 | 3 |
| **Total** | | **0** | **1** | **10** | **24** |

## Implemented in this wave

| ID | Commit | Change | Test |
|---|---|---|---|
| MEM3-1 | `010ee6f09` | full-matrix cupy DTW refuses a buffer over the 32-bit kernel index before any device work | `tests/signal_tests/test_dtw_kernel_index_guard.py` (fails on the previous code) |
| MEM3-2 | `010ee6f09` | banded cupy DTW refuses a band buffer over the same limit | same file (fails on the previous code) |
| PRV3-2 | `de23249f7` | HF load lock defaults to the per-user cache, mode 0o700, not the shared temp dir | `tests/training/feature_handling/test_hf_lock_path_per_user.py` (fails on the previous code) |
| YAML3-2 | `de23249f7` | Keras `from_config` runs inside `SafeModeScope` when the installed Keras has it | existing Keras tests (Keras absent on this host: skipped) |

## Dispositions without code change

- **MEM3-3** (low, REJECTED): the `K*M` int32 counts index is unreachable under the existing shared-memory gate (agent verified).
- **MEM3-4, MEM3-5, MEM3-6, MEM3-7, MEM3-8** (info): bounded by existing caps and gates, RESOLVED by those guards.
- **PRV3-1** (low, DOC): an explicit `revision` is not required to be a commit SHA. Documented: the pinned-revision rule is the caller's choice.
- **PRV3-3, PRV3-4, PRV3-5** (info): RESOLVED where the revision pin applies (`hf_provider.py`, `_neural_mi.py`), cited in the report.
- **API3-1** (low, DOC): inference errors name the featureset path and column names by design; the operator sees them, the caller does not get server internals.
- **API3-3** (low, DOC): `load_mlframe_model` returns a shared cached object. Documented in the function's own docstring. Copying a multi-GB bundle on every cache hit costs more than the mutation risk, so callers that mutate must call `_load_model_cache_clear()` first.
- **API3-2, API3-4, API3-5** (info): as stated in the report (logged path is local; normalisation has no bypass; Windows naming is a non-issue for the current callers).
- **LOG3-1, LOG3-2, LOG3-3, LOG3-4, LOG3-5, LOG3-6** (low and info, DOC): local-operator diagnostics by design; the full traceback and reprs are for the operator. Documented here.
- **CRY3-1** (info, DOC): the docstring wording is corrected in the tracker: the digest is unkeyed.
- **CRY3-2** (low, REJECTED): the salted blake2b orders benchmark labels; not a secrecy control.
- **CRY3-3** (info, REJECTED): unkeyed content addressing of pickled cache entries is intended; the sidecar digest is the integrity check.
- **CRY3-4, CRY3-5, CRY3-6** (info): non-security uses are correct; token-like identifiers use a CSPRNG (per report).
- **YAML3-3, YAML3-4, YAML3-5, YAML3-6, YAML3-7** (info, RESOLVED): safe_load, fixed tables, fixed maps, no config-value expansion (per report).
- **YAML3-8** (info, REJECTED): sizes in the shipped InfoNet config are the package's own, not user input.
