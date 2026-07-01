# /loop fuzz-profile-optimize log (300k input shape)

Random fuzz combos profiled under cProfile at n_rows=300k via
`profiling/profile_fuzz_chains.py`. Each iteration: profile fresh-seed combos,
pick the top mlframe-side hotspot, optimize (measured + bit-identical + regression
test), commit. Termination = 100 consecutive REJECTs (see CLAUDE.md policy).

Consecutive-reject streak: 0

| iter | seed | mlframe hotspot | verdict | notes |
|------|------|-----------------|---------|-------|
| 1 | 20260702 | `reporting/charts/slice_finder.find_weak_slices` — built bounds/features labels for every valid cell of every combo, discarded all but top-k | RESOLVED+4.18x | 58.6ms→14.0ms @ n2000/p12; bit-identical 60 seeds × arity{1,2,3}; also fixed profiler-driver KeyError on multi_target_regression combos. commit 21e45c3f |
| 2 | 20260703 | `reporting/charts/calibration.bootstrap_reliability_band` — refit IsotonicRegression per bootstrap resample, O(n log n) sort ×150 inside the loop | RESOLVED+3.9x | 2217ms→564ms @ n50k; per-fit 5x; distinct-score fast path (sort once + bincount sample_weight), tied-score estimator fallback; bit-identical 40 seeds × {distinct,tied}. commit f5043a50 |
| 3 | 424242 | `training/targets/…analyzer.analyze_target_distribution` — skew+kurtosis helpers each recomputed mean/std/z already available | RESOLVED+2.3x (moment pair) | 14.1ms→6.2ms @ n300k; fuse onto one z; bit-identical 80 seeds. Also REJECTED `renderers/_trend.robust_fit_endpoints` (0.645s tottime = cProfile mis-attribution of sklearn TheilSen fit; numpy parts microbench ~4ms). commit 3cc9e0ba |
