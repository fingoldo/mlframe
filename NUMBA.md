# numba in mlframe

Companion of `CUDA.md`: the same rules for the CPU twins (`@njit` kernels). The project has about 1400 `@njit` kernels, of which 245 set `fastmath` (about 10 percent in `feature_selection/filters`); no systematic A/B of the flag was ever recorded.

Evidence convention. **Measured** = run on this host in the session of 2026-11-17 (Windows 10, Python 3.11, numba 0.65.1, 8 logical threads, GTX 1050 Ti idle). **Source** = read through a page extractor (WebFetch summarises the page, so a quote is only as good as the extraction) and tagged with the URL in section 9. **UNVERIFIED** = no primary source read and no measurement here; it is a hypothesis to test on this host before it becomes a rule.

## 1. The CPU twin contract

Every GPU kernel has a numba CPU twin (CUDA.md section 2): the twin is the reference of the equivalence tests and the fallback when no device answers, so it has to be optimised with the same rigour as the GPU kernel, profiled, and routed through the kernel-tuning cache (the host/device choice is a measured crossover, never a constant; CUDA.md section 4).

Rule of the pair: **optimise the twin first, then sweep the crossover.** A crossover swept against an unoptimised CPU kernel is stale the moment the CPU kernel improves (the SHAP-proxy score dispatch is the case in point: its row mark moved from 10000 to 30000 rows on this host once measured honestly, and the CPU kernel was still unoptimised).

## 2. Look before you tune

1. **cProfile sees an njit call as one opaque frame.** Measured: a 20000 x 100 x 100-tree TreeSHAP call showed 99.9 percent of 3.6 s in the single `_treeshap_batch` frame and 0.000 s everywhere else. Use cProfile and pyinstrument only to rank kernels against each other and against the Python glue around them. `line_profiler` cannot see inside nopython code (source; not read first-hand: UNVERIFIED as a primary claim); `NUMBA_DISABLE_JIT=1` profiles interpreter-speed code whose time shares are not the compiled ones: use it for call counts and structure, never for "where the time goes" (measured example in section 3).
2. **Write the cost model before the experiments.** Count what the kernel must do (operations per element times elements) and divide by the machine rate. TreeSHAP path kernel: about 35 thousand cycles per (sample, tree) from an operation count, which matched the measured 13.8 microseconds; that said "compute-bound in the leaf unwinding, O(depth^2) per leaf" before any flag was tried. Every micro-optimisation that followed (a reciprocal table, fast-math) moved the time by less than the run-to-run spread; the algorithmic change (section 6) moved it 5.6 times.
3. **Look inside a kernel without a profiler.** `fn.inspect_asm(sig)` / `inspect_llvm` / `inspect_types` (source: numba jit reference). Measured: `inspect_asm()` returns a dict, about 930 lines for a 6-line kernel (mostly exception and NRT boilerplate), so count inside the inner loop only: `vdiv*` / `vsqrt*` (divisions, square roots), `call` (to `exp`, `log`, `NRT_MemInfo_alloc`), packed `p`-suffix operations against scalar `sd`/`ss`. Measured: a loop `1.0 / x` shows `vdivsd` and stays scalar without fast-math. `inspect_types()` finds unwanted `float64` / `int64` widening and object-mode fallback.
4. **Compilation must never be inside a timed region.** Measured: first call of a 6-line kernel 1.22 s, warm 2.1 ms. Assert it: `with numba.core.event.install_recorder("numba:compile") as rec:` around the timed part and check `rec.buffer` is empty (measured: the recorder saw start and end events for a cold call); or check `fn.signatures` did not grow. Warm up with the exact dtype and layout (C/F/A) of the timed call.
5. **Time like CUDA.md section 6:** warm best-of-N and the median next to it; record CPU clock and load and the GPU state before and after (`profiling/hardware_state.py` does it); interleave A and B in one process; compare the outputs before the timings. If best and median differ by more than a few percent the machine was not quiet: say so, do not average it away. Measured on this host: repeated runs of the same kernel differed by up to 5 percent (2639 to 2781 ms), so any gain below that is noise.

## 3. Tools that look inside compiled code, by host

| tool | on this Windows host | note |
|---|---|---|
| `NUMBA_DISABLE_JIT=1` + cProfile on a **small** input | yes (measured) | the kernel runs as plain Python, so every helper (including `inline='always'` ones) is its own frame with call counts. Gives the **structure**: how often each helper runs and which are called per leaf, per node, per sample. Does **not** give time shares: interpreter overhead per call and per loop iteration replaces the CPU cost. Measured on the TreeSHAP path kernel (40 samples, 20 trees): `_extend_path` 38 percent of `tottime` (21200 calls), the tree walk itself 31 percent, `_unwound_sum` 26 percent (28640 calls), `_unwind_path` 4 percent; the compiled kernel's limiter is the per-leaf unwinding (the table form that removes it ran 5.6 times faster), so the Python ranking of the helpers is not the compiled one. Use it for call counts and for choosing what to ablate, then confirm every claim with a timing of the compiled kernel |
| ablation timing of the compiled kernel | yes (measured) | run copies of the kernel with one part removed or replaced (the divisions, the leaf loop, the allocations) and compare against the full kernel on the same data: the drop is that part's share. Slower to set up than a profiler and the only method here that measures the compiled code per part |
| operation counters in a debug copy | yes | a copy of the kernel that returns counts (leaves visited, inner iterations, branches taken) next to the result; counts times the cost model give the expected shares. Counts are exact, the weights are a model |
| `inspect_asm` instruction count | yes (measured) | the cheapest way; see section 2 |
| `numba.core.event` recorder / timer | yes (measured) | compile side only: tells what compiled and for how long, not where run time goes |
| `parallel_diagnostics(level=1..4)`, `NUMBA_DEBUG_ARRAY_OPT_STATS` | yes (source) | loop fusion, serialised inner loops, hoisted allocations; read before timing a `parallel=True` kernel |
| `NUMBA_ENABLE_PROFILING=1` + Intel VTune | plumbing present, end result UNVERIFIED | numba turns profiling on by itself when `VS_PROFILER` is set (VTune detection, source); measured: `enable_jit_events()` returned 1, so a JIT listener is linked into the llvmlite wheel. VTune is not installed; the one candidate that would give instruction-level samples on `prange` kernels |
| `py-spy --native` | UNVERIFIED for numba | native stacks work on Windows x86-64, but nothing says numba JIT frames are symbolised: expect unknown frames |
| WPR / xperf / WPA | yes (general), JIT symbols no | system-wide CPU sampling: per-thread timelines, frequency, context switches. Use for parallel efficiency, not for lines |
| profila (pythonspeed) | **no** | Linux only, single-threaded numba only (WSL2 with `parallel=False` copies is the workaround); the line attribution is a hint because the optimiser merges lines |
| `perf` | **no** | Linux; stock numba does not emit a perf map (UNVERIFIED) |

No GitHub repository that packages numba optimisation or profiling as an agent skill was found in two searches. The one numba skill (`tondevrel/scientific-agent-skills`, 25 stars) holds generic advice (warm up, `cache=True`, `inspect_types`) and never mentions fast-math, races or threading layers: not used as a rule source. "Not found" does not mean "does not exist".

## 4. fastmath

1. `fastmath=True` is the flag set `{'fast'}` and LLVM `fast` implies **all** fast-math flags, `nnan` and `ninf` included (source: numba cpu_options and LLVM LangRef). The set form accepts `fast, nnan, ninf, nsz, arcp, contract, afn, reassoc`.
2. **Measured trap.** With `fastmath=True` a kernel that tests `np.isnan(x)` stopped routing missing values: the TreeSHAP numba kernel disagreed with the exact GPU kernel by **5.96 absolute** with 15 percent NaN in X, silently (no exception, no NaN in the output). LLVM assumes no NaN and folds the test (numba issue 2919, source). The 15 percent speed-up I measured first (3092 to 2639 ms) was obtained on NaN-free data and came from exactly these assumptions.
3. **Measured: the safe set gives nothing on that kernel.** `fastmath={'nsz','arcp','contract','afn','reassoc'}` (no `nnan`, `ninf`) restored the NaN behaviour (max difference 1.8e-15) and gave 3287 ms against the 3092 ms baseline: no gain beyond noise. The speed-up of the unsafe flag came from the removed checks, not from reciprocals or fusion. A kernel with a vectorisable float reduction is where `reassoc` pays (numba performance-tips example: a sum halves in time, source), the TreeSHAP leaf loop is not that shape.
4. **Rule.** Never `fastmath=True` where a NaN or an Inf can arrive (NaN-coded missing data, `-inf` as a max identity). For reductions that must vectorise use the set form without `nnan`/`ninf`, and **pin it with a NaN/Inf test** next to the kernel (feed NaN and Inf, assert the detection and the result against the non-fast variant). Record the flag set beside every timing. Reassociation changes the last bits of a sum: the project bar is selection equivalence (same selected features), pinned per kernel.
5. FMA is fused only where contraction is allowed and then by LLVM heuristics, not guaranteed (`contract`, source: numba discourse). SVML (`intel-cmplr-lib-rt`) vectorises math calls and fast-math selects the 4 ULP variants (source: performance tips).
6. `error_model='numpy'` removes the per-division zero check and returns inf/nan instead of raising: a legitimate speed knob for division-heavy loops that is independent of fast-math (source: jit options reference; effect not measured here).
7. **Project audit (open).** 1150 of the 1400 kernels do not set the flag; none has a recorded A/B. Any kernel adopted must come with the NaN pin and the measured gain; a blanket switch is wrong (section 2 point 2: find the limiter first).

## 5. Loops, memory and containers

Source for the whole section: numba performance tips, FAQ, pysemantics, pysupported.

1. An unjitted loop loses to a NumPy expression; skip functions under about 10 microseconds unless they are called from another jitted function.
2. **Contiguity** decides whether LLVM vectorises (the FAQ lists a non-contiguous layout among the rejection causes). Pass C-contiguous arrays (`np.ascontiguousarray` at the Python boundary): a kernel fed both layouts compiles two specialisations.
3. **No temporaries in the hot loop.** NumPy-style expressions inside a loop allocate every iteration; `astype` copies. Inside `parallel=True` numba hoists `np.empty` out of a `prange` body but keeps the zero-fill of `np.zeros` (check with `parallel_diagnostics`). Measured pattern in this project: per-row scratch of 12 arrays inside the `prange` body of the TreeSHAP kernel (20000 allocations) was not the limiter (the cost model said compute-bound).
4. **Integer width and bounds.** Integers have the width chosen by inference and overflow wraps; bounds checks are **off** by default (`boundscheck=True` or `NUMBA_BOUNDSCHECK=1` is a debugging switch, keep it off in production); keep index and count types consistent (intp) so no casts are inserted; use a wide accumulator where a threshold compares the sum (the CPU side of `{ACC}`). An unsigned-index view measured faster in one forum case (3.39 to 2.59 ms) with no confirmed cause and no numba version named: a hypothesis, benchmark it per kernel.
5. **Typed containers.** A `typed.Dict`/`typed.List` passed into nopython code is cheap, from the interpreter it is slower than a `dict` (boxing); a typed Dict is not thread-safe for modification. Reflected lists are pending deprecation: arrays or typed lists, and not in hot kernels called from Python.
6. AoS against SoA: no numba-specific source and no measurement found. Measured here only on the GPU side (CUDA.md): packing node records into one 32-byte struct gave **no gain** on a compute-bound kernel (1820 against 1819 ms), so measure before restructuring.
7. `inline='always'` inlines at numba IR level; LLVM already inlines small callees; no measurement found: use only where a call boundary blocks a hoist, and watch the compile time (UNVERIFIED).

## 6. Algorithmic change beats micro-tuning

When the cost model says compute-bound in a per-element inner loop, look for work that does not depend on the sample. **Measured case (SHAP-proxy TreeSHAP, 20000 samples, 100 trees of depth 6):**

| variant | time | note |
|---|---|---|
| path kernel, `fastmath=True` | 2639 to 2781 ms | wrong with NaN (section 4) |
| path kernel, safe fast-math set | 3287 ms | no gain |
| path kernel, reciprocal table instead of divisions in the leaf loop | 2806 ms | no gain, reverted |
| path kernel (reference) | 3092 ms | O(depth^2) work per (sample, leaf) |
| **table form** | **553 to 600 ms** | 5.6 times, same values to 2.7e-15 |
| public `shap` library on the same model | 3508 ms | reference point |

The table form: a leaf's path (its unique split features and their zero fractions) is fixed by the tree, and a sample only contributes which of the path's features it satisfies, a bit mask. The contribution table indexed by (leaf, mask) is built once per ensemble with the same arithmetic (warm build 6.6 ms for 424528 entries); a sample then walks the nodes keeping the mask and adds `ud` table entries per leaf. The dispatcher builds the table only when the build work is below a share of the path kernel's work over the rows and the table fits a size budget, and falls back to the reference otherwise. Pins: table equals path kernel on shallow and deep trees with repeated features, with 20 percent NaN, and on every fallback branch.

General pattern: precompute outside the kernel what depends only on the model, pass tables as arguments (a global array is frozen into the compiled code and the cache as a constant, so never rely on a global that changes), keep the old kernel as the reference and the fallback, and gate the new path on a work estimate.

## 7. parallel=True, prange, threads, cache

1. **Reductions** are inferred for `+= -= *= /=` and `max/min` (initial values are not inferred for min/max: set the identity); `//=` is unsupported. **Races are not detected** when threads write the same element; lists, sets and dicts are not thread-safe inside `prange`; the induction variable is not a thread id (indexing a buffer by `i % num_threads` is invalid). Equivalence-check every parallel kernel against its serial result.
2. **Only the outermost `prange` runs in parallel.** Static scheduling by default (one equal chunk per thread); `numba.set_parallel_chunksize(n)` switches to dynamic scheduling, effective only with the `tbb` layer.
3. **Threads.** numba always starts `NUMBA_NUM_THREADS` threads; `set_num_threads(n)` only masks them. Sweep 1, 2, 4, 8 in one process: efficiency = T1 / (n * Tn). The 8 logical threads of this host are probably 4 cores with SMT: UNVERIFIED (`wmic cpu get NumberOfCores,NumberOfLogicalProcessors`). Oversubscription: an outer joblib pool times an inner `prange` multiplies threads (the docs' example: 4 processes x 8 threads on 8 cores = 32); size `NUMBA_NUM_THREADS` per worker.
4. **Parallel kernels reached from Python threads** (our joblib threading fan-outs): the developer notes say the `workqueue` layer is not threadsafe and nested launches can abort; `safe` layers need `tbb`. Whether `omp` is threadsafe on Windows is documented nowhere I read: UNVERIFIED. Rule: if `parallel=True` kernels are reached from several threads, set `NUMBA_THREADING_LAYER=tbb` (or `threadsafe`) **before** the first parallel compile, check `numba.threading_layer()` after a parallel run, and keep the project's `parallel_kernel_entry` guard at the shared entry point (the TreeSHAP entry has it; CI aborted on macOS without it).
5. **nogil.** `nogil=True` releases the GIL on entry to a function that uses only native types; a joblib threading backend gets no speed-up from a kernel without it. nogil and parallel are independent options.
6. **False sharing:** the docs do not mention it. For hand-written adjacent per-thread accumulators pad the slots to 64 bytes and compare the time (a gain of more than a few percent is false sharing); scalar reductions are said to use per-thread slots (forum snippet, UNVERIFIED).
7. **Cache.** `cache=True` works with `parallel=True` (source: caching developer notes; a search summary of slides said otherwise and was not read). Invalidation watches only the source file of the main function: edits to a helper in another file, changed globals and compiler upgrades are **not** detected, the key includes CPU model and features. After merges or a numba/llvmlite upgrade delete `__pycache__/*.nbi *.nbc` (project memory: a stale cache is a silent crash source). `NUMBA_CACHE_DIR` private per benchmark; `NUMBA_DEBUG_CACHE=1` shows hits and misses. `numba.pycc` AOT is pending deprecation and does not suit 1400 kernels.

## 8. Diagnosis playbook for CPU kernels (signal, cause, first fix)

Each signal is a cheap experiment. Rows marked (M) were exercised in miniature on this host; the rest are standard method and UNVERIFIED here.

| | pattern | signal | first fix |
|---|---|---|---|
| A | compute-bound, scalar | time per element flat across sizes; asm inner loop shows `sd` scalar, no packed ops (M) | `reassoc` for reductions, restructure to vectorise, float32; if the loop is irregular, an algorithmic change (section 6) |
| B | division / transcendental heavy | replace `/ x` by `* c` in a copy: time drops by the op's share; `vdiv`, `vsqrt`, `call exp` in the loop (M) | hoisted reciprocal, `error_model='numpy'`, SVML, table. **Measured: no gain on the TreeSHAP leaf loop**, check the share first |
| C | memory-bound | time per element rises when the array leaves L2/L3; removing arithmetic changes nothing | fuse passes, float32, row-block tiles, contiguous C layout |
| D | allocation-bound | `NRT_MemInfo_alloc` / `np.empty` inside a hot loop (asm grep, `NUMBA_DEBUG_ARRAY_OPT_STATS`); time scales with call count | hoist buffers, per-thread scratch, `out=` |
| E | branch-bound | sorted or constant input much faster than random | branchless select, sort or partition inputs |
| F | bounds / error checks | exception-path calls in the inner loop asm | pre-check ranges outside the loop, `error_model='numpy'` |
| G | parallel, low speed-up | under about 4x at 8 threads; `parallel_diagnostics` shows the loop not parallel or wrongly fused | parallelise the outer loop, bigger tasks |
| H | parallel, load imbalance | per-thread times unequal (WPA) or a dynamic chunk size beats static | dynamic chunks (tbb), sort by cost |
| I | parallel, false sharing | efficiency collapses only for adjacent per-thread slots; padding fixes it | pad, accumulate in locals |
| J | oversubscription | more threads than cores (outer joblib x inner prange); worse than serial | `NUMBA_NUM_THREADS`, one level of parallelism |
| K | compile in the timed region | `signatures` grows or the `numba:compile` event fires (M) | warm up with identical types and layout |
| L | wrong dtype | `inspect_types()` shows float64/int64 where float32 was meant | cast once at the boundary |

Order of discrimination: K, then A against C by size scaling, then B by removing the division, then D by the asm grep, then E by changing the data order, then the parallel rows.

## 9. Sources

Official: numba performance tips, parallel, threading layer, threading implementation, caching, jit reference, env vars, FAQ, pysemantics, pysupported, pycc and deprecation pages, event API (all under numba.readthedocs.io/en/stable); numba source `core/cpu_options.py`, `core/fastmathpass.py`, `core/config.py`, `core/codegen.py`; LLVM LangRef fast-math flags; numba issue 2919 (NaN under fastmath); numba discourse threads on unsigned indices, missing optimisations, FMA. Tools: `pythonspeed/profila` (90 stars) and its article, `benfred/py-spy`, Microsoft WPR. All fetched through a page extractor; the Intel VTune pages, the numba-examples SIMD notebook (HTTP 429), the Stack Overflow and discourse threads on line profilers, and one blog on a custom llvmlite perf map were not read and are marked UNVERIFIED where used.

## 10. Checklist for a new or changed numba kernel

1. A GPU kernel has its twin; the twin is the reference of the equivalence test (selection equivalence, tolerance matched to the dtype).
2. Contiguous inputs; no allocation inside the hot loop; types consistent (`inspect_types`).
3. Cost model written down (operations per element, expected time) before any flag is tried.
4. fast-math only as a set without `nnan`/`ninf`, with a NaN/Inf pin test and a measured gain; the flag set recorded beside the timing.
5. Parallel: outer `prange` only, reduction identities set, serial equivalence test, `parallel_diagnostics` read, threading layer asserted when reached from threads, `parallel_kernel_entry` at the shared entry, `NUMBA_NUM_THREADS` sized per worker.
6. Timing: warm-up with identical types, no compile in the timed region (event recorder), best-of-N and median, interleaved A/B, CPU clock/load and GPU state recorded (`hardware_state.py`), private cache dir.
7. A profile record under `audits/2026-10-22/profiles/<module>.md` with the cProfile ranking, the instruction counts, the thread sweep and the numbers of the before/after table.
8. After a merge or a numba/llvmlite upgrade: wipe `__pycache__/*.nbi *.nbc` before measuring or running.
9. The CPU/GPU crossover is swept **after** both kernels are optimised, persisted as one monotone threshold, with a size rule as the fallback (CUDA.md section 4).
