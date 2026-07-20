# Orthogonal univariate/basis FE (Hermite, wavelet, hinge, polynomial)

7 findings, 4 proposals.

## Findings

### [P1] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py:189

**GPU-resident hinge breakpoint detector silently strided-subsamples to <=250k rows (MLFRAME_HINGE_MAX_ROWS, default 250000) before proposing tau candidates, while the CPU detector (_hinge_basis_fe._detect_hinge_breakpoints) always scans the full n -- a genuine algorithmic divergence, not FP-reorder noise.**

On a fit with n>250k rows and MLFRAME_FE_GPU_STRICT_RESIDENT on, detect_hinge_breakpoints_gpu strides x/y down to ~250k rows (`x = x[::_st]`) before running the FWL scan and held-out validation; a genuine kink whose signal is concentrated in a subsample-thinned region (e.g. a narrow but sharp tier boundary with few rows in that band) can be found at a different tau, or missed/spuriously found, versus the full-n CPU path on the exact same data. The module's own docstring calls this 'selection-equivalent' by appeal to a downstream re-scoring gate, but nothing in this cluster tests that claim end-to-end for n>250k.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_orthogonal_univariate_fe/_fourier_detect_gpu_resident.py

**No test exercises the GPU-resident multi-frequency Fourier detector (detect_fourier_freqs_for_col_gpu) at all, let alone compares it against the CPU twin _detect_fourier_freqs_for_col for selection-equivalence.**

A future change to this module (e.g. a dtype/relaxed-precision tweak, a refactor of _refine_peak_freq_gpu's argmax tie-break, or a bug in the seeded-subsample cache reuse) can silently return a different frequency list than the CPU path under MLFRAME_FE_GPU_STRICT_RESIDENT and no test would catch it: grepping the whole tests/ tree finds zero references to detect_fourier_freqs_for_col_gpu or _fourier_detect_gpu_resident. test_coarse_basis_njit_parity.py only checks CPU exact-vs-fast-njit; test_extra_basis_device_born_parity.py only rebuilds columns from ALREADY-KNOWN frequencies stored in meta, never exercises the detection algorithm itself.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py

**No test directly compares the CPU hinge detector (_hinge_basis_fe._detect_hinge_breakpoints) against its GPU twin (detect_hinge_breakpoints_gpu) for the same input -- existing GPU hinge tests only check GPU-internal properties (resident-cache byte-identity, subsample-still-finds-a-breakpoint, batched-vs-per-column GPU agreement), never CPU-vs-GPU.**

Combined with the P1 subsample-divergence finding above, the codebase's own selection-equivalence claim for the hinge GPU twin is untested end-to-end: a regression in the on-device FWL/QR math, or a change to the CPU detector, could silently desync CPU and GPU tau selection at any n and nothing would fail.

### [P2] bug -- src/mlframe/feature_selection/filters/polynom_pair_fe.py:562

**run_polynom_pair_fe's per-pair injection loop appends the new column to _new_data_cols/_new_col_names/_new_col_nbins/_existing_col_names BEFORE assigning it into X and registering it in engineered_features/hermite_features_list/engineered_recipes; if the X assignment (line 567/569) raises, the caught exception (line 593) leaves data/cols/nbins committed to the new column at function end (the concatenate at the bottom is unconditional) while X, engineered_recipes, and engineered_features never got it.**

If `X[_new_col_name] = _t_vals` (pandas) or `X.with_columns(...)` (polars) raises for any reason after `_new_data_cols.append(_new_binned)` has already run (e.g. an OOM, a duplicate-column edge case, or a future refactor that makes the assignment fallible), the function still concatenates that column into `data`/`cols`/`nbins` at the end, producing a `cols` entry with no matching `X` column and no `engineered_recipes` entry -- MRMR.transform() would then fail to replay that feature (KeyError on recipe lookup) or silently misalign column indices against `data`.

### [P2] design -- src/mlframe/feature_selection/filters/_orthogonal_univariate_fe (whole cluster)

**Bare `except Exception:` swallow sites are pervasive (~70 across the cluster) but every one inspected follows the codebase's documented GPU-residency contract (device/optional-dep fault -> exact host fallback, default path byte-identical) rather than masking a real logic bug; no new silent-real-error-swallow instance found beyond the parity/test gaps already reported above.**

No issues found in this cluster for this angle; explicitly reported per audit instructions rather than omitted. One prior narrowing fix is already on record at _orth_mi_backends.py:314-330 (FIX1) that deliberately excludes ValueError/IndexError from the swallowed set so a genuine OOB bug still surfaces.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_orthogonal_univariate_fe (whole cluster)

**No wasteful/avoidable host<->device round trips found in the GPU-resident twins inspected (_orth_gpu_resident.py, _gpu_resident_cross_basis.py, _uplift_univariate_resident.py, _extra_basis_resident.py, _hinge_detect_gpu_resident.py, _hinge_detect_gpu_resident_batch.py, _fourier_detect_gpu_resident.py, hermite_fe/_hermite_prewarp_gpu_resident.py, _extra_fe_families_dispersion_resident.py) -- every cp.asnumpy/.get() site is a documented, bounded SCALAR pull (O(iterations)/O(candidates)), never a re-upload of an n-scaled bulk array per fit.**

No issues found in this cluster for this angle; explicitly reported per audit instructions rather than omitted.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_hinge_detect_gpu_resident.py:194

**test_hinge_detect_subsample.py's parametrized test only asserts a clear single-breakpoint synthetic signal is still found near its true tau under the 250k-row cap; it never asserts the found tau matches the full-n (uncapped) result to within a numeric tolerance, so the cap's claimed 'selection-equivalent' property has no quantitative regression pin.**

A future change that widens the stride or weakens the held-out uplift floor could silently degrade tau precision on capped large-n fits without any test catching the drift, since the existing test only checks abs(tau - 0.5) < 0.4 (a very loose tolerance) rather than comparing against the uncapped CPU/GPU result.

## Proposals

### (fe_idea) Tent/triangular (linear, C0-continuous) multiresolution basis alongside Haar

The wavelet family (_wavelet_basis_fe.py) only ships the Haar step wavelet (+1/-1 discontinuous jump). A localized SMOOTH bump (not a sharp step) forces Haar to approximate a continuous rise/fall with a staircase of +/-1 legs, which either needs many legs (defeating the candidate-count self-limit) or leaves visible discretization error at the bump's shoulders. A dyadic 'tent'/triangular basis (psi_{j,k} linearly ramps 0->1->0 across the same dyadic interval Haar uses, i.e. the hat function of a linear-spline multiresolution) is a genuinely different, non-duplicate operator: it is continuous (no Gibbs-like ringing at leg boundaries the way a truncated Fourier series has, and no jump discontinuity the way Haar has), while still being LOCALIZED and MULTISCALE like Haar. It would reuse the exact same held-out scale-selection + incremental-MI admission machinery already built for Haar (_select_wavelet_legs, _heldout_incremental_mi), just swapping the leg-generating closed form. Not a duplicate of the cubic B-spline (fixed, UNSUPERVISED quantile knots, global support per basis function) or of Haar (discontinuous).

### (fe_idea) Oblique (2-column) hinge: relu(w_a*x_i + w_b*x_j - tau) as a joint-threshold basis

The hinge family (_hinge_basis_fe.py) only detects axis-aligned breakpoints in a SINGLE column (max(x-tau,0)). Many real thresholds are joint: e.g. risk jumps when a linear combination of two columns (income - k*age, or pressure - k*depth) crosses a cutoff -- a case an axis-aligned split cannot express in one feature (a decision tree needs several splits to approximate it; this is exactly the 2-D generalization backlog #11's own docstring flags pricing tiers / dose-response as motivating, extended to 2 correlated drivers). Detect (w_a, w_b, tau) via a small grid/CMA search over the direction (analogous to detect_pair_symmetry's existing marginal-MI framing already in _hermite_fe_optimise.py) combined with the existing 1-D breakpoint scan on the projected axis w_a*x_i + w_b*x_j, then emit relu(proj - tau) as a leak-safe closed-form recipe (store w_a, w_b, tau; replay is a pure linear-combination-then-relu of the two source columns, no y needed at transform time -- mirrors the existing hinge_basis recipe contract). Complementary to, not redundant with, the existing pair-cross orthogonal-polynomial product basis (which captures SMOOTH bilinear interactions, not a sharp joint threshold) and the univariate hinge (axis-aligned only).

### (fe_idea) Random Fourier Features (Bochner/RBF-kernel proxy) over an arbitrary k-column subset

The cross-basis families in this cluster (pair/triplet/quadruplet product basis in _orth_pair_cross_fe.py and siblings) only scale to 2-4 columns before combinatorial explosion forces a small seed pool. Random Fourier Features -- cos(w.x + b) for w ~ N(0, gamma*I) sampled once at fit time over a chosen k-column subset -- give a fixed-size (e.g. 8-16 columns), CLOSED-FORM, leak-safe proxy for an arbitrary-dimension smooth RBF-kernel interaction (Bochner's theorem), without needing a combinatorial degree x degree x ... sweep. Recipe stores only the sampled (w, b) matrix (no y) so replay is a pure cos(X[cols] @ w + b) of the source columns -- structurally identical in leak-safety shape to every other closed-form recipe in this cluster. This is NOT a duplicate of the existing adaptive/chirp Fourier detector (_orth_extra_basis_fe.py), which is strictly UNIVARIATE (one column, one detected frequency) -- RFF instead targets smooth MULTI-column interactions the polynomial-product cross-basis families currently cannot reach past arity 4.

### (coverage_gap) Add a direct CPU-vs-GPU parity test for the hinge and Fourier detectors

Given findings #2/#3 above: add a test that runs `_detect_hinge_breakpoints` (CPU) and `detect_hinge_breakpoints_gpu` (GPU, with MLFRAME_HINGE_MAX_ROWS=0 to disable the subsample so it is an apples-to-apples comparison) on the SAME synthetic slope-change fixture and asserts the returned tau lists agree to a documented tolerance; do the same for `_detect_fourier_freqs_for_col` vs `detect_fourier_freqs_for_col_gpu` under MLFRAME_FE_GPU_STRICT_RESIDENT=1. Separately, add a test that compares the GPU hinge detector's capped (n>250k, default cap) output against the CPU full-n output on a fixture where the true breakpoint sits in a region that would be thinned by the stride, to give the 'selection-equivalent' claim in the module docstring an actual regression pin.
