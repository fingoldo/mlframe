# Wave 3 audit 2026-11-01: fail-open error handling in security checks

Scope: `src/mlframe` (read-only). Class: guards wrapped in try/except that proceeds as if the check passed (integrity and sidecar checks,
path containment, allowlists, VRAM and resource guards that should refuse, probes whose exception means "assume safe"). Builds on the three
earlier trackers.

The auditor could not write this file itself; the content below is its returned text. The sidecar verification, path containment and
unpickle allowlist checks it read all fail closed.

## Summary

9 findings: 0 High, 0 Medium, 4 Low, 5 Info.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| FOP4-1 | Low | `training/io.py:570-590` | FUTURE |
| FOP4-2 | Low | `calibration/_post_train_calibrators.py:243-300` | FUTURE |
| FOP4-3 | Low | `feature_selection/filters/_mrmr_validate_transform.py:157-170` | DOC |
| FOP4-4 | Low | `_orth_gpu_resident.py:36-42`, `_gpu_resident_shared.py:77-80`, `_fe_cmi_perm_null_gpu.py:366-368` | FUTURE |
| FOP4-5 | Info | FE shared-buffer RAM budget | DOC |
| FOP4-6 | Info | feature cache and discovery filter RAM probe | DOC |
| FOP4-7 | Info | InfoNet checkpoint hash is trust-on-first-use | DOC |
| FOP4-8 | Info | broad excepts around sidecar checks | RESOLVED |
| FOP4-9 | Info | tuning-lock liveness probe | DOC |

## Findings

**FOP4-1 (Low, FUTURE).** A `_LOAD_MODEL_CACHE` hit returns before the digest and `strict_version` checks run. The cache key omits
`strict_version`, so a strict caller receives an object that an earlier lenient caller loaded. The cached object was verified at load time
and the key includes the file's mtime, so the bytes cannot have changed; the gap is only the strictness level.

**FOP4-2 (Low, FUTURE).** The three calibration/test leakage guards each skip silently (debug log or pass) when a probe raises. If all three
skip, nothing records that no leakage check ran.

**FOP4-3 (Low, DOC).** The MRMR RAM-headroom guard turns itself off when `psutil` fails; it logs a WARNING.

**FOP4-4 (Low, FUTURE).** These VRAM guards assume "OK" or 1 GiB when `memGetInfo` fails. The sibling guard in `_fe_gpu_vram.py`
deliberately fails closed.

**FOP4-5 (Info, DOC).** The FE shared-buffer RAM budget hoists unconditionally when the RAM probe fails.

**FOP4-6 (Info, DOC).** The feature cache and discovery filter treat a failed RAM probe as unbounded.

**FOP4-7 (Info, DOC).** The InfoNet checkpoint hash is trust-on-first-use by design (the NET2-03 fix).

**FOP4-8 (Info, RESOLVED).** The broad excepts around the sidecar checks cover only version-drift validation. The integrity check runs first
and is not wrapped.

**FOP4-9 (Info, DOC).** The tuning-lock liveness probe fails toward "stale, take over". This overlaps CON2-1.
