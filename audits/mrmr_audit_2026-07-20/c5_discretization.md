# Discretization / adaptive binning

10 findings, 6 proposals.

## Findings

### [P0] bug -- src/mlframe/feature_selection/filters/discretization/_discretization_edges.py:332

**get_binning_edges's njit quantile branch calls np.percentile on an empty array for an all-NaN column, and the resulting ValueError is silently swallowed by numba's parallel (prange) execution instead of propagating -- the caller gets garbage int codes, not an exception.**

Confirmed by direct reproduction: get_binning_edges(all_nan_col, method='quantile') alone correctly raises ValueError('zero-size array to reduction operation minimum...'), and _discretize_array_impl (single-column njit path) also raises correctly. But discretize_2d_array(arr, method='quantile', prefer_gpu=False) -- the batch entry point that routes through the @njit(parallel=True) _discretize_2d_array_njit prange wrapper -- does NOT raise for the same all-NaN column; it silently returns WILD out-of-range int8 codes (observed range -120..122 for a supposed [0,9] n_bins=10 output) with zero warning or exception. Any downstream consumer that uses these codes to index a (K_x, K_y) joint-count array (e.g. _plug_in_mi_njit, or MRMR's own MI kernels) can receive negative or oversized indices, silently corrupting or crashing further downstream. discretize_2d_array's default method is 'quantile', so this is the DEFAULT code path, not an opt-in one. No existing test constructs an all-NaN column and calls method='quantile' (only the 'uniform' method has an all-NaN regression test, in test_discretization_nan_uniform.py).

### [P1] cpu_gpu_parity -- src/mlframe/feature_selection/filters/discretization/__init__.py:1038

**discretize_2d_array_cuda's 'uniform' and 'quantile' branches compute column min/max/percentile without filtering NaN (unlike the CPU path's explicit NaN-aware arrayMinMax / mask-before-percentile), so a single NaN anywhere in a column corrupts the derived edges and silently collapses the WHOLE column's real values, not just the NaN row.**

Reproduced directly: with one NaN injected at row 7 of an otherwise-clean 500-row column, CPU discretize_2d_array(method='uniform') correctly bins the other 499 rows across bins 0-9 and routes only row 7 to the dedicated NaN bin (10); GPU discretize_2d_array_cuda(method='uniform') collapses ALL 500 rows of that column to bin 0 (np.unique(gpu_col) == [0]), destroying the real signal for every row, not just the NaN one. Same divergence for an all-NaN column: CPU uniform -> [10] (dedicated bin, correct by design); GPU uniform -> [0] (whole column mis-binned). The existing GPU/CPU parity test suite (tests/feature_selection/gpu/test_discretize_cuda_cpu_parity.py) only injects ONE scattered NaN per 2000-row column for the quantile method and concludes 'NaN routes to the TOP bin on both backends... no fix needed' -- that conclusion is incomplete: it never tests the uniform method's NaN handling at all, and never tests an all-NaN or NaN-dense column where the corruption is severe rather than a single-row edge effect. Currently NOT reachable via MRMR.fit's categorize_dataset (which always scrubs NaN via _handle_missing before calling discretize_2d_array), but IS reachable by any direct caller of the public discretize_2d_array/discretize_2d_array_cuda API, which the module's own top-of-file docstring documents as public entry points with no stated NaN precondition.

### [P1] bug -- src/mlframe/feature_selection/filters/supervised_binning.py:175

**mdlp_bin_edges (and its two near-duplicated siblings mdlp_bin_edges_validated / mdlp_bin_edges_oos_validated in _mdlp_validated_split.py) drop NaN rows from x explicitly, but never filter NaN out of y before casting it to int64 -- a NaN-y row silently becomes a garbage class label instead of being dropped or raising.**

Reproduced: mdlp_bin_edges(x, y, fast_mode=True) with y[5]=np.nan (x all-finite) emits a RuntimeWarning 'invalid value encountered in cast' from `y = _y_arr.astype(np.int64)` (line 175) -- the NaN row is folded into the recursion as a spurious class label instead of being dropped like the symmetric NaN-x handling three lines below (which IS explicitly implemented and tested). The same unguarded `.astype(np.int64)` on a NaN-bearing y appears at _mdlp_validated_split.py:386 (mdlp_bin_edges_validated) and :595 (mdlp_bin_edges_oos_validated). Existing regression coverage (tests/feature_selection/discretization/test_mdlp_nan_handling.py, test_mdlp_validated_split_fast.py::test_nan_handling_matches_documented_contract) only injects NaN into x, never into y, so this gap has never been exercised.

### [P2] bug -- src/mlframe/feature_selection/filters/discretization/__init__.py:440

**quantize_dig (and the unused njit `digitize` helper) use side='left'-equivalent tie-breaking while quantize_search -- the only one of the three actually called on the production hot path -- and every other edge-application site in the cluster use side='right'; quantize_search's own docstring falsely claims it is 'equivalent to quantize_dig'.**

Reproduced: for bins=[-inf,2,4,6,inf] and a value exactly on an edge (2.0), quantize_dig(arr,bins) assigns bin 0 while quantize_search(arr,bins) assigns bin 1 -- for arr=[2.0,4.0,6.0,3.0,5.0] the two functions return [0,1,2,1,2] vs [1,2,3,1,2], a systematic off-by-one for every on-edge value. quantize_dig has zero non-warmup callers (only _legacy.py's re-export and _prewarm.py's JIT-compile warmup reference it) and zero test coverage, so the divergence is currently latent, but the module docstring lists it as public API ('Lower-level numba helpers digitize, quantize_dig, quantize_search') with no warning that it disagrees with its documented 'equivalent' sibling. Every strategy that IS live (uniform/quantile/knuth/blocks/mdlp/fayyad_irani/optimal_joint/cv/mah) funnels through side='right' consistently (verified via a full grep of every `side=` / `searchsorted` call site in the cluster), so the inconsistency is confined to this one dead pair of helpers -- but it is a real trap for any future caller of the documented public API.

### [P2] bug -- src/mlframe/feature_selection/filters/_adaptive_nbins.py:344

**edges_optimal_joint filters NaN out of x and applies that same x-derived mask to y (keeping the two aligned), but never independently checks y for NaN -- a NaN-y row with a finite x is NOT dropped and flows into _bin_y_for_mi's non-nan np.quantile call, propagating NaN into that fold's y-quantization edges.**

`mask = np.isfinite(x); x = x[mask]; y = y[mask]` (lines 344-347) only ever removes rows where X is non-finite. If y independently contains NaN (plausible for a partially-labeled or messy regression target passed as y_for_strategy to categorize_dataset with nbins_strategy='optimal_joint'/'cv'), those rows survive into `_bin_y_for_mi` (line 470-475), whose `np.quantile(y.astype(np.float64), ...)` (non-nan variant) can return NaN quantile edges for that fold, corrupting `y_b`/`K_y` for every row in that fold, not just the NaN one. Same bug class as the mdlp_bin_edges finding above, in a sibling supervised strategy; MRMR's normal contract likely guarantees a clean y upstream, which is why this is P2 rather than P1, but the code has no explicit guard or documented assumption either way.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/discretization/__init__.py:1156

**discretize_2d_array_cuda_row_chunked's 'uniform' branch uploads every row-chunk to the GPU TWICE per fit (once in the min/max reduction pass, once again in the affine-map pass) -- a real double round-trip, but load-bearing: exact global min/max requires two data passes and the whole point of this fallback path is that not all row-chunks fit in VRAM simultaneously, so they cannot be kept resident between passes.**

No failure -- flagging per the audit's explicit residency-angle requirement. `d_chunk = cp.asarray(arr[row_start:row_end])` appears once in the min/max loop (line 1158) and again, independently, in the affine-map loop (line 1169) for the SAME chunk, each followed by `del d_chunk`. This only engages on the rare VRAM-constrained fallback path (discretize_2d_array_cuda_row_chunked, reached only when the VRAM guard in discretize_2d_array rejects the single-shot upload), and the 'quantile' sibling on the same function already documents an accepted approximation (subsample-based edges) to avoid exactly this kind of double pass; 'uniform' deliberately stays exact (per its own docstring: 'EXACT, no approximation'), so the double upload is the documented cost of that accuracy choice, not an oversight.

### [P2] design -- src/mlframe/feature_selection/filters/discretization/_discretization_dataset.py:281

**None of the supervised nbins_strategy values (mdlp/fayyad_irani/fayyad_irani_validated/optimal_joint/cv/mah/sci/marx) have any GPU implementation, so CPU/GPU parity is structurally moot for more than half the strategies this cluster was asked to check -- this is a real, already-tracked gap, not a clean bill of health.**

No incorrect output -- explicit design-coverage note. categorize_dataset's own comment (line ~281) documents that the supervised path 'has NO GPU kernel' and defers a CUDA MDLP behind other work; per_feature_edges (_adaptive_nbins.py) has no cuda/cupy code anywhere. Only the unsupervised uniform/quantile strategies have a GPU twin (discretize_2d_array_cuda) to compare against CPU at all. Listed explicitly per the audit instruction to state when an angle is not applicable/clean rather than silently omit it.

### [P2] test_gap -- tests/feature_selection/gpu/test_discretize_cuda_cpu_parity.py:58

**The existing GPU/CPU parity test suite's own documented conclusion ('P1-2 NOT a divergence -- agent mis-trace: NaN routes to the TOP bin on both backends... no fix needed') is based on incomplete testing (a single scattered NaN in one row per 2000-row column, quantile method only) and does not hold for an all-NaN column, a NaN-dense column, or the uniform method at all -- exactly the cases the P0/P1 findings above expose.**

test_cuda_quantile_nan_routes_to_top_like_cpu only ever sets a[7,:]=np.nan (1 NaN row out of 2000) for method='quantile'; it never tests method='uniform' with NaN, never tests an all-NaN column, and never tests a NaN-dense column (e.g. >10% NaN) where percentile-based inner edges themselves become NaN. Because the suite's docstring asserts the NaN question is already closed ('P1-2 was a mis-trace'), a reviewer skimming it would reasonably conclude GPU/CPU NaN parity is verified, when it is not for the uniform method or for denser/whole-column NaN.

### [P2] test_gap -- src/mlframe/feature_selection/filters/supervised_binning.py:468

**optimal_bin_edges (the optbinning.OptimalBinning wrapper, one of the cluster's public non-trivial functions) has zero test references anywhere in the repository.**

grep across tests/ for 'optimal_bin_edges' returns no matches. No behavioral test exercises the monotonic-constraint wrapper, its -inf/+inf sentinel contract, or its optbinning>=0.21 compatibility note (the docstring documents a specific historical breakage with optbinning<0.21 + sklearn>=1.6 that has no regression test pinning the fix).

### [P2] design -- src/mlframe/feature_selection/filters/discretization/_discretization_edges.py:310

**get_binning_edges has no return type annotation (returns np.ndarray on every path but the signature ends at the parameter list with no `-> ...`); noted per the audit's mypy-hygiene check, not fixed.**

Not a runtime bug -- mypy-hygiene note only, per instructions to note but not fix. All three branches (uniform/quantile/raise) are consistent in returning np.ndarray or raising, so the omission is purely a missing annotation, not a type mismatch.

## Proposals

### (edge_case) Guard the njit quantile edge-builder against an all-NaN column instead of relying on numba to propagate the ValueError

get_binning_edges's quantile branch (_discretization_edges.py:326-332) should check `arr_finite.size == 0` explicitly (mirroring discretize_uniform's existing `if not (_rng > 0):` degenerate guard) and return a sentinel/degenerate edge set instead of calling np.percentile on an empty array -- closes the root cause of the P0 finding rather than relying on numba's prange exception behavior, which was empirically shown to swallow the exception and return garbage instead of raising.

### (edge_case) NaN-scrub (or explicitly document as undefined + raise) inside discretize_2d_array_cuda before computing min/max/percentile

Add an `assume_finite`-style guard to discretize_2d_array_cuda's uniform and quantile branches (mirroring discretize_2d_quantile_batch's existing `np.isnan(arr2d).any()` check), or at minimum raise a clear error when NaN is present rather than silently producing a corrupted whole-column result -- the current 'caller's contract to scrub' stance is undocumented in the function's own docstring and is violated silently rather than loudly.

### (coverage_gap) Add a NaN-in-y regression test alongside every existing NaN-in-x test for the MDLP family

mdlp_bin_edges (supervised_binning.py), mdlp_bin_edges_validated and mdlp_bin_edges_oos_validated (_mdlp_validated_split.py), and edges_optimal_joint (_adaptive_nbins.py) all need a test that injects NaN into y (not just x) and asserts either a clean drop of those rows or an explicit raise -- mirroring test_mdlp_nan_handling.py's existing x-NaN coverage exactly, just on the other array.

### (coverage_gap) Extend test_discretize_cuda_cpu_parity.py with all-NaN-column and NaN-dense-column cases for BOTH methods

Current coverage is a single scattered NaN, quantile-only. Add: (a) an entirely-NaN column for both uniform and quantile, (b) a column with NaN density high enough to reach an inner quantile position (e.g. >10% NaN) so the test actually exercises edge corruption rather than only the extreme-percentile case. These would have caught the P0/P1 findings above directly.

### (coverage_gap) Add a direct unit test for optimal_bin_edges

A short pytest.importorskip('optbinning')-guarded test exercising the -inf/+inf edge contract and monotonic_trend='auto' would close the zero-coverage gap on this public function.

### (other) Factor the near-identical x/y NaN + dtype-normalization prologue shared by mdlp_bin_edges, mdlp_bin_edges_validated, and mdlp_bin_edges_oos_validated into one helper

All three functions currently duplicate the same ~15-line y-dtype-normalization + x-NaN-drop block verbatim (by the module's own stated design, 'kept... rather than imported... per file-ownership convention'). That convention means the y-NaN fix above has to be applied identically in three places and will silently drift out of sync on the next edit to any one of them. A shared `_prepare_mdlp_xy(x, y, max_y_classes)` helper (even if only supervised_binning.py owns it and _mdlp_validated_split.py imports it, which the module already does for other symbols) would apply any future fix once.
