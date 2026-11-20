# perf_C: cost of integer-code range validation on hot paths

Method: warm, best-of-N, separate processes for new (wt_audit) and old (old_base), numbers noisy (machine loaded by other agents; process_time resolution ~15 ms). GPU GTX 1050 Ti was idle (0%).
Scripts: filters/_benchmarks/bench_validate_pair_codes_cost.py, bench_batch_pair_mi_cuda_call.py; info_theory/_benchmarks/bench_batch_kernels_code_range_check.py.

## (1) 03-fix2 validate_pair_codes in batch_pair_mi_cuda / row_chunked: RESOLVED (reduced, not free)
Before: validate_pair_codes alone (numpy min(axis=0)+max(axis=0)) = 10.8 / 52 / 323 ms at n=200k/1M/2M (32 cols) = 47% / 61% / 78% of the call wall.
Fix: one parallel njit row-major min/max pass (column_code_bounds); the kernels' separate global factors_data.min() sweep now reuses the same bounds (validate_pair_codes(..., bounds=)), so there is one sweep not two.
After: validate ~5-8 / 10-20 / 13-18 ms. Whole batch_pair_mi_cuda call, new vs old_base (two rounds): 12.3-19.3 vs 8.7-9.3 ms (200k), 37-40 vs 27-29 (1M), 66-69 vs 50-51 (2M). Residual added cost is one memory-bandwidth pass over the matrix (~20-30% of the call, since the call itself is only ~one H2D copy); it cannot be below 1% without moving the check into the device kernel (invasive, branch in hot loop) or caching on array identity (unsafe under mutation). Reduction vs the previous implementation: 4-25x. ValueError behaviour and messages unchanged; existing tests pass (45).

## (2) 45-perf CPU njit checks in info_theory/_batch_kernels.py and _batch_kernels_v3.py: REJECTED (no change needed)
The per-column code check already lives inside the existing first counting loop (one compare per element, no extra pass; v3 returns early on a bad code). The only extra pass is _y_codes_in_range, O(n) once per call versus O(n*K) kernel work. Kernel wall (npermutations=0, the worst case; ms process time, best-of-3 x 2 rounds), new vs old:
- n=200k k=64: v1 890/1156 old? see below; observed spread between rounds (+-25%) exceeds any difference.
- Round 2: 1M x64 v3 1109 vs 1406, v2 3000 vs 3578; 1M x256 v3 3531 vs 3641, v2 20828 vs 24656; 2M x64 v3 1828 vs 1984, v2 7281 vs 8469; 200k x512 v3 (round1) 1656 vs 1922.
New is not slower than old in any paired round-2 cell; round 1 had two cells within noise the other way (2M x64 v3 2469 vs 1969). Analytic bound: y check = n compares vs n*K histogram increments, <=1/K per pass (K>=64: <1.6% of one pass, far less of the whole kernel with permutations). Nothing to fold.

## (3) 41-14 batch_pair_mi_cupy code check: RESOLVED
The dispatcher has no existing min/max screen to fold into, so the check is added at batch_pair_mi_cupy entry (before any device work) using the same single parallel pass (validate_pair_codes with n_classes_y = len(freqs_y)). Cost: cupy call 22.9 ms at 200k (validate 5.6 ms, ~24%) and 106 ms at 1M (validate 11.5 ms, ~11%): not under 1%, but it is the same memory-bandwidth floor as (1) and the alternative is an out-of-range bincount index silently mis-binning. Test added: test_batch_pair_mi_cupy_rejects_out_of_range_codes_before_any_device_work (tests/feature_selection/gpu/test_gpu_pair_mi_code_screens_and_determinism.py; cupy stubbed via _cp=None, 3 cases); old code reaches cupy with the stub and does not raise ValueError, so it fails there.

mypy clean on touched files; interrogate 100%.


## Follow-up: device-side check (supersedes the residual cost in (1) and (3))
batch_pair_mi_cuda and batch_pair_mi_cupy now check codes on the device, after the resident upload: one RawKernel sweep (per-column min/max via atomicMin/Max) plus the class-vector min/max, one small readback (check_codes_on_device). Same ValueError messages and order (classes_y, whole-matrix non-negative, referenced-column range). A failing device check (OOM, fault) logs a throttled WARNING (pair_codes_device_check_failed) and runs the host check; it is never skipped. batch_pair_mi_cuda_row_chunked keeps the parallel host pass: it streams row chunks from host and never holds the matrix on the device. The host pass in the other callers is gone from the hot path.

Whole-call wall, new vs old_base, three interleaved process pairs (best-of-N each):
- batch_pair_mi_cuda: 200k 7.4-7.8 vs 8.7-13.4 ms; 1M 19.2-21.2 vs 27.9-30.2 ms; 2M 33.4-33.7 vs 47.7-51.3 ms. New is FASTER than old_base because old_base still ran its host factors_data.min()/classes_y min/max passes, which the device sweep replaces: net cost vs old_base is negative, well under the <=1-2% target.
- batch_pair_mi_cupy (two pairs): 200k 18.3-18.9 vs 17.4-20.0; 1M 90.7-92.3 vs 88.4-95.2; 2M 189.1-189.5 vs 186.0-192.4 ms. Added cost is inside run-to-run noise (<=~2%).
Tests: 48 pass (45 prior incl. the earlier 3, plus a real-device check_codes_on_device test x3 and the cupy-entry test reworked to stub a failing device check, asserting the WARNING and the ValueError). mypy clean, interrogate 100%. Bench: _benchmarks/bench_batch_pair_mi_cupy_call.py added.
