# Wave 2 tracker: full attack-class audit of mlframe (2026-11-05)

Eleven read-only Opus 5.5 (low effort) audits, one per attack class. Each finding was checked against HEAD before a fix. Dispositions: RESOLVED (fixed, regression test in the commit), PARTIAL (fixed part, next step written), DOC (documented, reason given), REJECTED (not a defect, reason given), FUTURE (real, deferred, with next step).

| ID | Sev | Disposition | Commit / note |
|---|---|---|---|
| DESER-W2-1 | Medium | RESOLVED | `602e5d400` feature cache reads and verifies one payload; sidecar required with pickle mode on |
| DESER-W2-2 | Low | RESOLVED | `602e5d400` same change: legacy sidecar-less entries refused unless `MLFRAME_FEATURE_CACHE_STRICT` is set to a false value |
| DESER-W2-3 | Info | DOC | Model sidecar detects corruption, not replacement of model and sidecar together. Signing is future work. |
| DESER-W2-4 | Info | REJECTED | `cloudpickle.loads` reads only data the parent created for its own worker |
| DESER-W2-5 | Info | REJECTED | Benchmark scripts read caches they wrote; not on production paths |
| EXEC-W2-1 | Info | DOC | `private_temp` skips the owner check on Windows; safe while `%TEMP%` is per-user |
| EXEC-W2-2 | Info | DOC | Non-private temp dir leads to a new `mkdtemp` per call; disk growth, warning shown |
| EXEC-W2-3 | Info | REJECTED | Output location is the caller's choice |
| SEC-W2-1 | Info | DOC | `embed_website_to_mlflow` takes any URL scheme; docstring says callers pass pipeline-generated URLs |
| SEC-W2-2 | Info | DOC | `model_dump(scrub_secrets=False)` returns secrets by design for provider load |
| NET-W2-1 | Low | DOC | Unlisted HF model loads `main` with a warning; the owner chooses revisions (same as N3-03) |
| NET-W2-2 | Low | RESOLVED | `027d1f344` `packaging` pinned in both matrix workflows; meta test `test_workflow_packaging_is_pinned.py` |
| RES-W2-1 | High | PARTIAL | `b212aa330` int32 tables and 2**31 row guard; `_n_train_` stored instead of the leaf matrix (see QRF commit). Next: logged size limit on the forest. |
| RES-W2-2 | Medium | RESOLVED | `8adef3764` decompression cap is half of RAM, at most 64 GiB |
| RES-W2-3 | Low | DOC | Column-order cache caps each entry at 4 GB but not the total; the owner can choose a total cap |
| RES-W2-4 | Info | REJECTED | Already bounded (checked) |
| STAT-W2-1 | Low | RESOLVED | `5a0627f0e` Nadeau-Bengio corrected test in forward selection patience rule |
| STAT-W2-2 | Low | DOC | Compare bootstrap p-value is not a null-distribution test; docstring should say so |
| STAT-W2-3 | Low | RESOLVED | `db5539e82` `fit_outlier_bounds` + `apply_outlier_policy(bounds=...)` |
| STAT-W2-4 | Info | REJECTED | Unsupervised utility; no held-out contract |
| NUM-W2-1 | Medium | REJECTED | Measured on HEAD: NaN PIT gives NaN at n=1000 and n=300000, so the claimed finite result does not reproduce. Guard reverted. |
| NUM-W2-2 | Low | RESOLVED | `5a0627f0e` Hurst zero checks relative to window scale |
| PAR-W2-1 | High | RESOLVED | `ae1cac81c` CUDA discretization reads Fortran input as C-contiguous; GPU test |
| SIL-W2-1 | Medium | RESOLVED | `5a0627f0e` warnings on ragged member lengths and stability-chart cut |
| SIL-W2-2 | Medium | RESOLVED | `de6d95942` pre-pipeline and cat-code replay failures logged at WARNING |
| SIL-W2-3 | Medium | RESOLVED | `64346550c` no clipping from scored labels; throttled warning instead |
| SIL-W2-4 | Medium | RESOLVED | `71e5ec2e1` clustered bootstrap counts dropped resamples; warns above 25% |
| SIL-W2-5 | Medium | RESOLVED | `5959652da` `read_trained_models` warns with served / total counts |
| SIL-W2-6 | Low | RESOLVED | `682c36b7c` restore failure logged at WARNING. No deterministic test: a logging-level change on a hard-to-trigger error path. |
| SIL-W2-7 | Low | RESOLVED | `682c36b7c` jackknife drop count warning; test `test_jackknife_reports_dropped_evaluations.py` |
| SIL-W2-8 | Low | DOC | Calibrator inner CV averages over the folds that succeeded; the owner decides on a strict mode |
| SIL-W2-9 | Low | RESOLVED | `de6d95942` GPU correlation fallback logs at WARNING |
| CONC-W2-1 | Medium | RESOLVED | `f9ef97628` sweep restores the previous device override |
| CONC-W2-2 | Low | RESOLVED | `80f292d25` CUDA fallback save under a lock; race not deterministically reproducible, behaviour test added |
| CONC-W2-3 | Low | RESOLVED | `80f292d25` depth-counted TQDM toggle; overlap test |
| CONC-W2-4 | Info | DOC | Fused-backend sweep briefly takes the fused path while it runs; equivalent results |
| API-W2-1 | Medium | RESOLVED | `31819e06d` identity estimators raise NotFittedError before fit |
| API-W2-2 | Low | RESOLVED | `4bfc3c616` decorrelator leaves `feature_names_in_` absent for arrays |
| API-W2-3 | Low | RESOLVED | `4bfc3c616` random classifier width check |
| API-W2-4 | Info | DOC | `monotonic_decline_patience=None` annotated as `int`; the owner decides the annotation |

## Not covered by this wave
- Kernel parity: 77 of 79 GPU-kernel files were not read (PAR audit). Next round should cover MI/CMI kernels, `_gpu_metrics.py`, DTW, the transformer kernels, honest-null and resident-select.
- API audit read two files of about 80 estimator classes. Next round should cover the rest.
- Concurrency audit did not check the CHANGELOG for earlier fixes.
