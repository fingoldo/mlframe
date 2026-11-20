# MRMR audit wave 2026-11-20 -- master tracker

Four read-only agents (A core correctness, B FE families and recipes, C GPU / resources / concurrency, D API / tests / performance), 22 findings, reports in this folder. The main session implements every fix itself;
each fix carries a sensor that fails on the pre-fix code. Coverage caveats: agent A did not trace the honest-null device kernels or the CMI / FE gates line by line; agents A and C found fewer than the 8-14 requested.

Statuses: FIXED (code changed, sensor named), WONT-FIX (reason given), TODO (open).

| Finding | Severity | Status | Where / sensor |
|---|---|---|---|
| A-F01 id-keyed content-hash memo | High | FIXED | `_mrmr_fingerprints.py` weakref slot; `test_mrmr_xhash_recycled_id.py` |
| B-F01 float32 subtractions on offset columns | High | FIXED | `_ratio_delta_fe.py`, `_grouped_agg_fe.py`, `_grouped_coerce_shared.py`; `test_float64_deviation_for_offset_columns.py`; benchmark selection identical |
| D-F01 identity fast path vs permuted support_ | High | FIXED | `_mrmr_validate_transform.py`; `test_mrmr_transform_identity_fastpath_identity.py` |
| D-F02 fast-path test stub sorted only | Medium | FIXED | same test file (permuted + boolean mask cases) |
| A-F02, A-F03, A-F04, A-F05 | Medium / Low | TODO | |
| B-F02, B-F03, B-F04, B-F05, B-F06 | Medium / Low | TODO | |
| C-F01 .. C-F05 | Medium / Low | TODO | |
| D-F03 .. D-F06 | Low | TODO | |
