# bias_A: 22-13 (hinge tau scan) and 10-13 (fastmi bandwidth)

## 22-13 hinge tau scan on held-out rows - RESOLVED
Finding: the tau argmin scanned all rows, including idx%3==0 rows the held-out uplift gate validates on.
Fix: `_detect_hinge_breakpoints(..., scan_on_train_rows=True)` scans (QR, FWL, candidates scored) on idx%3!=0 only; `False` = legacy. Same in the GPU twins (`_hinge_detect_gpu_resident._hinge_full_scan`, `detect_hinge_breakpoints_gpu`, `_batch`), GPU path NOT executed here (no device run).
Measured (uplift floor 0, 60 seeds, max_breakpoints=1), chosen tau changed on: n=300 18/31, n=600 12/23, n=1000 3/3, n=4000 33/60. Strong-kink fixture (slope 5, noise .1, n=4000, default floor): 0/40 changed.
Chance hinge on linear+noise (300 seeds): n=200 floor default old 45 -> new 36; floor .01 81 -> 62; n=400 default 10 -> 7; floor .01 33 -> 19.
Cost: tau accuracy on a true kink slightly worse (mean |tau-0.7|: n=400 0.132 -> 0.159; n=4000 0.0185 -> 0.0218; scan sees 2/3 of rows). Moves are material for borderline cases, so legacy kept behind the flag; default = corrected.
Test: `test_tau_scan_excludes_heldout_rows_cuts_false_hinges` (fails on old code: no parameter / 33 vs 19 false hinges). `test_detect_hinge_fwl_rank1_taus_bit_identical_to_lstsq_per_cut` now passes scan_on_train_rows=False (it pins FWL vs all-row lstsq). Bug caught on the way: the relu leg of found taus must use the train slice.
Benchmark: src/mlframe/feature_selection/_benchmarks/bench_hinge_tau_scan_rows.py. Not threaded through generate_hinge_features / _detect_hinge_breakpoints_for_columns (default only).

## 10-13 fastmi MISE bandwidth rescale - RESOLVED
Fix: `fastmi(..., mise_rescale=True)` multiplies the 1000-row-subsample bandwidth by (1000/n)^(1/6); False = legacy.
Measured mean relative error vs -0.5 log(1-rho^2), rho in {.3,.6,.9}, 3 seeds: n=1e3 0.176/0.176 (legacy/rescaled, identical, no subsample), 1e4 0.181/0.080, 1e5 0.178/0.037, 1e6 0.200/0.024.
Test: tests/feature_selection/filters/test_fastmi_mise_bandwidth_rescale.py (n=20000: rescaled 0.062 vs legacy 0.158; asserts <0.09 and <0.7x legacy; fails on old code, kwarg missing). Benchmark: _benchmarks/bench_fastmi_bandwidth_rescale.py.

## Checks
mypy on 6 touched source/bench files: no issues. interrogate on touched tests/benchmarks: 100%. Existing fastmi tests 34 passed. test_hinge_basis_fe.py: 18 passed + 1 fixed (rerun in h3) ; other hinge files (subsample/gpu/precheck/ols) not re-run after the final edit.
