# Wave 7 audit 2026-11-02 -- tracker

Three read-only reports on `src/mlframe` by Opus 5.5 low-effort agents (text encoding and locale, pickle hygiene of estimators, integer overflow in index and class-id arithmetic). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (safe, or fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | ENC7-1 | text_encoding | Medium | `nvidia-smi` output decoded with the machine code page in `training/_gpu_state_probe.py` |
| **RESOLVED** | ENC7-2 | text_encoding | Medium | import-probe child output decoded with the machine code page in `utils/native_import_probe.py` |
| **REJECTED** | ENC7-3 | text_encoding | Info | file reads already pass an explicit encoding |
| **RESOLVED** | ENC7-4 | text_encoding | Low | child process not told to write UTF-8 |
| **REJECTED** | ENC7-5 | text_encoding | Info | locale-dependent formatting not found in data paths |
| **RESOLVED** | INT7-1 | int_overflow | Medium | `merge_vars` and `_merge_vars_freqs` wrapped class ids in a narrow `dtype` |
| **REJECTED** | INT7-2 | int_overflow | Info | indexes already int64 at the flagged sites |
| **DOC** | INT7-3 | int_overflow | Low | `compute_ncrossings` int32 caps n at 2^31 rows |
| **REJECTED** | INT7-4 | int_overflow | Info | flagged counter bounded by the row count held in int64 |
| **REJECTED** | INT7-5 | int_overflow | Info | flagged product already widened before the multiply |
| **RESOLVED** | INT7-6 | int_overflow | Low | covered by the INT7-1 change |
| **RESOLVED** | PKL7-1 | pickle | High | `UniversalCallback` on fitted boosters held a lambda and a live widget |
| **RESOLVED** | PKL7-2 | memory | Medium | MRMR kept the permutation-null closure (a float64 copy of the frame) after fit |

## Counts

13 findings: 1 High, 4 Medium, 2 Low, 6 Info and below. Dispositions: 6 RESOLVED, 1 DOC, 6 REJECTED, 0 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| ENC7-1, ENC7-2, ENC7-4 | Decode child-process output as UTF-8 with replacement in the GPU and import probes | `tests/utils/test_subprocess_output_decoding.py` |
| INT7-1, INT7-6 | Refuse a too-narrow working dtype in merge_vars instead of wrapping class ids | `tests/feature_selection/filters/test_merge_vars_narrow_dtype.py` |
| PKL7-1 | Make the early-stopping callbacks on fitted boosters picklable | `tests/training/callbacks/test_callback_pickles.py` |
| PKL7-2 | Release MRMR's permutation-null closure once fit ends | `tests/feature_selection/mrmr/core/test_mrmr_releases_null_closure_after_fit.py` |

## Not covered by this wave

- `shap_proxied_fs` int32 index arrays, `discretization/_quantile_radix.py`, `_batch_kernels.py` encoding and the `_ktc` launchers were only grepped.
- The XGBoost `callbacks` param still references the callback after fit; with the callback picklable this no longer blocks saving.
- Rows ENC7-3, ENC7-5, INT7-2, INT7-4, INT7-5 and the `compute_ncrossings` DOC row are condensed from the agents' text, not re-derived line by line.
