# Full attack-surface audit 2026-10-31 -- tracker

Eight read-only reports on `src/mlframe`, one per attack class. No code was changed by the audit. Each finding carries a
proposed disposition: **RESOLVED** (already safe, citation given), **FUTURE** (real, deferred, reason given),
**DOC** (intended behaviour, to be documented), **REJECTED** (not a risk, reason given).

## Counts by class

| Class | Report | High | Medium | Low | Info | Total |
|---|---|---|---|---|---|---|
| Unsafe deserialization | [deserialization.md](deserialization.md) | 1 | 7 | 4 | 10 | 22 |
| Dynamic code execution | [dynamic_exec.md](dynamic_exec.md) | 0 | 0 | 10 | 19 | 29 |
| Subprocess / shell | [subprocess.md](subprocess.md) | 0 | 0 | 3 | 9 | 12 |
| Network / servers | [network.md](network.md) | 0 | 0 | 4 | 5 | 9 |
| Temp files / filesystem | [filesystem.md](filesystem.md) | 0 | 1 | 5 | 10 | 16 |
| Secrets / credentials | [secrets.md](secrets.md) | 0 | 1 | 3 | 1 | 5 |
| Dependencies / supply chain | [supply_chain.md](supply_chain.md) | 0 | 3 | 6 | 6 | 15 |
| Untrusted input / resources | [untrusted_input.md](untrusted_input.md) | 0 | 2 | 3 | 4 | 9 (sampled) |

## Priority items (proposed)

- **DESER-10 (High), RESOLVED** in `303e54be4`: the cache loader deferred to `MLFRAME_ALLOW_UNVERIFIED_PICKLE` (default closed) instead
  of forcing `allow_unverified=True`. Pinned by `tests/training/composite/cache/test_cache_unsigned_pickle_rejected.py`, which fails
  on the previous code.
- **DESER-02 (Medium)**, `training/io.py` `load_mlframe_model`: the default model-bundle loader has no sha256 check.
  **Proposed: FUTURE** (needs a manifest format change).
- **DESER-06 (Medium)**, `estimators/pipelines.py:111`: unrestricted `joblib.load` after an integrity-only sidecar check.
  **Proposed: FUTURE.**
- **FS-02 (Medium)**, `estimators/pipelines.py:163-166`: predictable file in the shared temp dir, read back as a pickle.
  **Proposed: FUTURE** (per-user 0o700 directory and `mkstemp`).
- **SEC-01 (Medium), RESOLVED** in `620c0d2d2`: the give-up log carries the scrubbed message only, and the re-raised error is no longer
  chained to the original. Pinned by `tests/integrations/test_mlflow_scrubbed_giveup_log.py`, which fails on the previous code.
- **SUP-02 (Medium)**, `pyproject.toml:154`: `pyutilz>=1.0.0` is not on PyPI, so pip users are exposed to dependency confusion.
  **Proposed: FUTURE** (pin or document the pre-install).
- **SUP-03 (Medium)**, `.github/workflows/dependabot-auto-merge.yml:36-43`: every dependabot PR auto-merges without review.
  **Proposed: FUTURE** (review gate or narrower allow-list).
- **INP-01 / INP-02 (Medium)**, `calibration/policy.py:461`, `votenrank/shapley_blend.py:183`: unbounded allocations
  driven by caller sizes. **Proposed: FUTURE** (caps and chunking).

## Status

Two items fixed (DESER-10, SEC-01), each with a regression test that fails on the previous code. The rest are proposals. Per the project's audit rule, the fixes are triaged with the
owner before any code changes, because several change public behaviour (DESER-10 most of all). The remaining Low and Info
rows are recorded in their class reports with the same vocabulary.

Scope limits, stated by the reports themselves: the secrets pass did not finish the whole tree and the tests and YAML/JSON were
not fully enumerated; the resource-exhaustion pass was sampled, not exhaustive; the supply-chain pass could not check live
PyPI or branch protection from this environment.
