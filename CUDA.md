# CUDA in mlframe

How GPU code is written, dispatched, tested and profiled in this repository. It merges what we learned writing the feature-engineering (FE) kernels with what is publicly available for AI-assisted CUDA work. Where a statement comes from outside, the source is named. External claims were read from the primary files through an extraction tool (section 8 says how) and were not reproduced on this host, so a number from outside is a pointer to check, not a measurement of ours.

Scope: the FE modules the MRMR layer imports (`src/mlframe/feature_selection/filters/`). The per-module status is kept in `audits/2026-10-22/fe_gpu_coverage.md`, and the contract below is pinned by `tests/test_meta/test_fe_gpu_streaming_contract.py`.

## 1. Development host

| item | value |
|---|---|
| GPU | GeForce GTX 1050 Ti, Pascal, compute capability 6.1 (sm_61), 4 GB, 6 SMs |
| CPU | Core i7-7700HQ, 4 cores, 8 threads |
| stack | cupy 13.6, numba, CUDA toolkits 11.8, 12.8, 12.9 |

There is a second machine with a newer GPU architecture. On it only `ncu` and `nsys` work, so the profiling recipes for that machine are in section 9; its hardware fingerprint and kernel-tuning cache are separate from this host's.

Everything measured in this file was measured on this host unless it says otherwise. A kernel that wins here can lose on another GPU class (atomic latency, shared-memory budget and register pressure differ between compute capabilities), so every kernel version is kept and the dispatcher chooses per host (section 4).

## 2. The contract for an FE module

A feature-engineering module used by MRMR meets all of these:

1. **CPU kernel and fused GPU kernel**, both kept as versions. Fused means one launch per unit of work and no intermediate table written to global memory. The generic project dispatchers are not a substitute when they are slower on the module's own shape (section 7).
2. **Column-block streaming with a RAM and a VRAM bound.** No `n x p` temporary. A device that runs out of memory hands the block to the CPU kernel instead of failing the fit.
3. **Type-generic kernel source.** No bare `double` or `float`. `{REAL}` is the type of values and per-cell terms, taken from the crit dtype; `{ACC}` is the accumulator type of sums and of statistics compared with a threshold. Counts and indices stay integer types.
4. **Dispatch through the kernel-tuning cache (KTC)**, with a measured fallback heuristic.
5. **Profiled twice and optimized**: cProfile for the host path and a GPU profiler for the device path, with the numbers kept in `audits/2026-10-22/profiles/<module>.md`. The GPU profile covers the whole-GPU picture, not only the hotspot kernel: achieved occupancy and its limiter, SM throughput, DRAM bandwidth, launch statistics (grid, block, registers, shared memory), the kernel launches per call, and the share of wall time in kernels against copies and host work.

`tests/test_meta/_fe_gpu_contract.json` is a ratchet: modules that meet the contract are checked in full, the pending list only shrinks, and a new FE module that is neither compliant nor exempt fails the test.

## 3. Kernel design rules

From our kernels:

- **One block per work item, histogram in shared memory, reduce in place.** The pair residual kernel runs a CUDA block per column pair, fills the joint count table with shared-memory atomics, and reduces the MI inside the block, so nothing but the winner leaves the device.
- **Power-of-two block size** when the block reduces with a tree; 128 threads for the pair kernel.
- **Dynamic shared memory sized from the table** (`k*k*kr` ints), so the same kernel serves any bin count that fits.
- **Integer counts, real arithmetic only where needed.** Counts are `int`; the log terms are `{REAL}`; the sum is `{ACC}` (float64). A float32 sum over a few hundred terms is accurate to about 1e-7 relative, but a statistic that is compared with a ceiling or whose argmax picks a pair keeps a float64 accumulator.
- **Placeholders are substituted before compilation**, one compiled module per type, cached in a dict. The reference pattern is `hermite_fe/_als_kernels_gpu.py` (`_SRC.replace("{REAL}", "double" | "float")`, `_MODULES_BY_REAL`); `_fourier_fused_gpu.py`, `_basis_fused_gpu.py` and `_wavelet_legs_fused_gpu.py` follow it. The type comes from `crit_float_dtype()` (float32 by default, float64 when `MLFRAME_CRIT_DTYPE_RELAXED=0`).
- **Test both REAL types against the CPU kernel.** float32 terms agree with the float64 CPU result to about 4e-9 relative on the pair kernel; the test allows 1e-6 for the float path and 1e-9 for double.
- **Keep every kernel version under a distinct name** (`_v1`, `_v2`, `_fused`) and write a short "win axis" note at the top: which shapes it wins and which it does not.

From the surveyed public guidance (section 8; the tensormux skills were read in full, the Hugging Face rules target H100, A100 and T4 and are unverified):

- block size a multiple of 32, start at 128 or 256; occupancy is the minimum of the register limit, the shared-memory limit and the hardware block limit; 50% or more is a fair target, a compute-bound kernel is fine near 25%, and a spill is not worth a higher occupancy; take the register count from `ptxas -v` or the profiler, never from the source;
- `gridDim.y` and `gridDim.z` stay below 65,535; a launch costs about 5 to 20 us, so a grid under 100 blocks is a candidate for fusion;
- coalesced global access first, vectorized loads, and a shared tile padded `[32][33]` against bank conflicts;
- reductions: warp shuffle with an explicit mask, one value per warp into shared memory with a barrier between the write and the read, a two-pass form for reproducibility; a floating-point atomic sum is non-deterministic, while our integer counts in shared memory are exact and the final tree reduction has a fixed order;
- guard every global access in the tail block, and initialize outputs to the identity or to a sentinel, because a zeroed buffer can hide a bug;
- choose the architecture target explicitly (sm_61 here).

More rules from the other tensormux skills, all read in full (generic guidance, not measured by us):

- **Global memory** (`optimize-global-memory-access`): classify every access by the warp footprint. Stride-1 touches 1 to 4 lines of 128 bytes; stride S touches up to S lines and wastes about (S-1)/S of the traffic, and above S=32 the efficiency is sizeof(dtype)/128 (fp32 about 3%). A `float4`/`int4` load (16 bytes, 16 `int8` values) is only for an already coalesced stride-1 access and needs `(base + thread offset) % 16 == 0` for every thread, with a scalar tail; vectorizing a strided access is a failure mode. A transposed or strided pattern is staged through shared memory with a barrier between the coalesced global write and the shared read. Array of structures is strided by the struct size; structure of arrays is stride-1; decide at the data-structure level, across all kernels that share the data. `const T* __restrict__` and `__ldg()` use the read-only path, only for data not written in the same launch. Coalesce writes too (a partial line is a read-modify-write). Memory-bound target: above 80% of peak bandwidth; vectorizing a coalesced access typically gives 10 to 30%. Small inputs hide a stride behind L1/L2, so profile at production size.
- **Shared-memory tiling** (`optimize-shared-memory-tiling`): bank of a 4-byte word is `(byte address / 4) % 32`; a conflict is k distinct addresses in one bank, the same address is a broadcast. Compute the conflict multiplier analytically before writing code. Pad the conflicting dimension only (one 4-byte column for fp32, two fp16 columns for fp16), and use the padded stride in every index. Double buffering on sm_80 and newer: `cp.async.cg.shared.global` 16-byte copies, `commit_group` after each tile, `wait_group(1)` before computing on the filled buffer, and a barrier after every wait (the wait is DMA completion, the barrier is block-wide visibility); trace ping/pong over two iterations by hand. Shared memory per SM: 96 KB (sm_70), 164 KB (sm_80), 100 KB (sm_86), 228 KB (sm_90); more than 48 KB per block needs `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. Aim for at least two resident blocks per SM. Counter: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`.
- **Warp divergence** (`avoid-warp-divergence`): classify a branch as warp-uniform (free), geometry-based (peel the loop or use a boundary kernel), data-dependent (reorganize or sort the data) or unavoidable boundary (accept). Cost model: divergent cost is C_A + C_B against max(C_A, C_B), so an overhead ratio of 1.1 caps the gain at 10%. Ignore boundary divergence that touches under 0.1% of warps. Compaction pays only for long inner loops, and its extra traffic must be compared with the saving. `__ballot_sync` and `__any_sync` need a mask that matches the truly active lanes (`__activemask()` inside a divergent region only sees the current path). Branchless `y = (x > 0) * f(x)` evaluates `f` regardless, which is wrong for `sqrtf(-1)`. Divergence removal does not help a memory-bound kernel. Metrics: `branch_efficiency`, `smsp__inst_executed_pipe_alu_pred_off_pct`.
- **Numerical stability** (`write-numerically-stable-kernel`): accumulate any reduction over more than 32 to 64 elements in fp32 or wider when the inputs are fp16 or bf16; variance by Welford or two passes, never E[x^2]-E[x]^2; subtract the row maximum before `exp`; log-sum-exp directly, never `log(softmax)`. fp16 epsilon is about 1e-3 and bf16 about 8e-3, so bf16 buys range and not mantissa. Tolerances against an fp64 reference: fp32 rtol 1e-5 and atol 1e-8, fp16 rtol 1e-2 and atol 1e-3, bf16 rtol 5e-2 and atol 1e-3. Test adversarial values (absolute value above 1000 and below 1e-4, mixed signs, identical values, the maximum at the last position), not only `randn`. State the cost of each stabilization (extra pass, registers). This is the reason for our `{ACC}` placeholder: sums and thresholded statistics are wider than the per-cell terms.
- **Boundaries** (`handle-boundary-conditions`): compute the validity predicate on global indices before any load; fill out-of-range shared slots with the identity of the operation (0 for a sum, -inf for a max or softmax, never uninitialized values and never zero for a softmax); a zero-padded softmax is a bug because `exp(0) = 1` joins the denominator. Mask every tile, not only the last one, in persistent kernels; test one tile, tile-1, tile+1, one element and a large prime. Off-by-one `idx <= N`, masking only the column in a 2D tile, and a wrong identity for max versus sum are the usual failures.
- **Tile size and partitioning** (`choose-tile-size-and-work-partitioning`): shared memory per block, registers per thread (`65536 / (warps x 32 x blocks per SM)` as the budget) and occupancy (the minimum of the shared-memory, register and hardware block limits; 32 resident blocks on sm_86) decide the tile; the innermost tile width is a multiple of 32; the grid has at least as many blocks as SMs; a tile whose last-tile useful fraction is below 0.1 is padded or shrunk. Persistent kernels (one block per SM looping over tiles) cut launch cost and balance irregular shapes, and should map tiles in index order so neighbours share L2 lines. Occupancy: 50% or more is a starting point, under 25% limits latency hiding, above 75% usually gains little.
- **Fusing elementwise ops** (`fuse-elementwise-ops`): fuse only ops whose output at index i depends on input at index i, with no other consumer of the intermediate; traffic drops from 2NS to 2S bytes (N ops, S bytes); keep the chain under about 20 to 30 live values to avoid spills; one coalesced load, all ops, one store; target 70 to 90% of peak bandwidth for large tensors; a reduction in an epilogue needs two passes. This matches our fused pair kernel, where fusing the histogram, the MI reduction and the argmax removed the intermediate table.

## 4. Dispatch through the kernel-tuning cache

Do not hard-code thresholds, block sizes or backend choices. Register the kernel with `pyutilz.performance.kernel_tuning.registry.kernel_tuner(...)`: a sweep over the module's own axes (`sweep_backend_grid`), an equivalence tolerance for the variants, a fallback callable with a measured heuristic, and `gpu_capable=True`. The per-host cache lives in `~/.pyutilz/kernel_tuning/<hw_fingerprint>.json`. Keep an environment override as an escape hatch. The reference registration is `batch_pair_mi_gpu.py` (`_BPMI_SPEC`); the pair residual test registers `pair_residual_test` the same way.

`MLFRAME_DISABLE_GPU=1` or an empty `CUDA_VISIBLE_DEVICES` switches the GPU off for the whole run and wins over any forced backend.

## 5. Memory and streaming

The incident that produced the rule: the pair residual wrapper built several `n x p` int64 temporaries (`astype(int64)`, `* k`, `// levels`, `np.minimum`), which is about 1.6 GB per matrix at n=100k, p=2000 and about 6 GB at the peak, on top of the caller's own `n x p` matrix.

The rule:

- **Stride rows to the work budget before anything is built**, not after.
- **Convert and score by column blocks.** Hold two blocks of int8 codes at once (`best_pair_streamed`): the diagonal blocks score the pairs inside a block, the off-diagonal blocks score the rectangle between block I and a later block J, and together they cover every pair once.
- **Size the block from free memory:** a fraction of free RAM for the host, a third of free VRAM for the device (`cp.cuda.runtime.memGetInfo()`).
- **Make a memory bound a test**, not a claim: the pair suite asserts that `tracemalloc` peak stays under a quarter of one `n x p` int64 matrix.
- **Fall back instead of failing.** An out-of-memory error or a driver fault on the device finishes the block pair on the CPU kernel and logs a warning.
- **Withhold, do not guess, when the work cannot be done:** if the pair count leaves too few rows inside the budget, the check returns `None` and the caller keeps the FE search running.

## 6. Timing and profiling honestly

Rules we learned the hard way on this host:

- **Synchronize inside the timed region.** cupy and numba.cuda launches are asynchronous. An unsynchronized cupy timing once showed a 16x win that was launch overhead; with a per-call `cp.cuda.Stream.null.synchronize()` it was a loss.
- **Measure the resident case.** Time three points: host CPU, GPU on already-resident device arrays, GPU with the per-call upload. A kernel that wins resident and loses end to end is a residency problem, not a bad kernel.
- **Warm up, then time.** Exclude numba and NVRTC compilation. A cold first call can be several times slower and made one run look like a path difference that was only compilation.
- **Beware memoized wrappers.** `dispatch_batch_pair_mi_chunked` remembers large requests by content, so a repeat returns in about a millisecond. Time `_dispatch_batch_pair_mi_chunked_uncached`, and use novel seeds for fits that are cached by data hash.
- **cProfile is blind to async GPU work.** Kernel-queue time is attributed to the launching Python frame, so a large `tottime` on a GPU call is often wait time. Confirm a hotspot by direct synchronized timing before optimizing.
- **nvprof per-kernel percentages are serialization-distorted.** nvprof forces syncs; a kernel at 35% of GPU time was close to 0% of the wall. The wall truth is cProfile cumulative time plus a full-fit A/B on a quiet machine.
- **A kernel microbenchmark can lie.** A coalescing fix made a kernel 7.2x faster in isolation and gave 0.99x on the function wall. Confirm the full-fit wall moves before shipping.
- **Benchmark real shapes.** A synthetic shape three times the real one gave conclusions that did not transfer.
- **Contention dominates variance.** The same fit measured 60 s and 147 s under concurrent GPU load. Check `nvidia-smi` and your own running processes, run one heavy job at a time, and pair the A/B in one process.

### Profilers on this host

| tool | status on the GTX 1050 Ti |
|---|---|
| `nvprof` | works for kernel timing without admin; counter metrics (`achieved_occupancy`, `sm_efficiency`, `dram_utilization`, ...) need an elevated shell (`ERR_NVGPUCTRPERM` otherwise) and the CUPTI directory on `PATH` (`/d/CUDA/v12.8/extras/CUPTI/lib64`) |
| `nsys` | works, no admin: `C:/Program Files/NVIDIA Corporation/Nsight Systems 2022.4.2/target-windows-x64/nsys.exe` (not on `PATH`); profile the warm region through the `cudaProfilerApi` window |
| `ncu` | installed (2022.3.0 and 2025.2.1) but refuses Pascal ("Profiling is not supported on device 0"); use it on Turing or newer |

On a GPU that `ncu` supports, record the sections described below. On this host take the same quantities from `nvprof --metrics`.

### Nsight Compute reference (from the primary files of section 8)

Workflow, in order:

1. **Locate the target launch first:** `ncu --print-summary per-kernel python script.py`. Framework setup launches kernels before the target, so a blind `--launch-skip 10` can profile the wrong kernel.
2. Measure the kernel time and the utilization; below 90% profile with `ncu --set full --launch-skip N --launch-count 1 --kill yes [--kernel-name NAME] -o report python script.py`. For the per-line pass add `--set source --section SourceCounters` (needs `-lineinfo`). Open a report with `ncu --import`; export with `--csv` or `--page raw|details`.
3. Read the sections: SpeedOfLight (SM% against Memory%: high SM and low memory is compute-bound, low SM and high memory is memory-bound, both low is latency-bound), Occupancy (theoretical against achieved, and the limiter: registers, shared memory, block size), LaunchStatistics (grid, block, registers per thread, shared memory per block), MemoryWorkloadAnalysis, SchedulerStatistics and WarpStateStatistics (the stall breakdown), SourceCounters.
4. Change one thing and profile again.

Metric names (Nsight Compute, Turing and newer):

| quantity | metric |
|---|---|
| SM utilization | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bandwidth | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| occupancy limiters | `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem` |
| launch statistics | `launch__grid_size`, `launch__waves_per_multiprocessor`, `launch__registers_per_thread`, `launch__shared_mem_per_block_dynamic` |
| coalescing | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` divided by `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` (1 to 4 good, 8 to 16 poor, 32 and more very poor) |
| shared-memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `..._st.sum`, divided by the wavefronts `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` (above 1 per operation is significant) |
| warp stalls | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` (global memory), `..._short_scoreboard_...` (shared memory and L1), `..._wait_...` (synchronization), `..._math_pipe_throttle_...`, `..._mio_throttle_...`, `..._not_selected_...` (ready but not scheduled: too little occupancy) |
| spills | `smsp__sass_inst_executed_op_local_ld.sum`, `..._local_st.sum` |
| unintended double precision | `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active` |

The `{REAL}` and `{ACC}` design makes the last row a useful check: a float32 `{REAL}` kernel should show a near-zero FP64 pipe except for the `{ACC}` sums.

Bottleneck rules used by the atrex classifier (`ncu_classification.py`; heuristics, applied in this order, each appends a finding):

| finding | condition |
|---|---|
| small grid, idle SMs | waves per SM below 0.5 (high confidence), or grid size below the SM count (medium) |
| tail effect | waves above 1 and the idle fraction of the last wave, (ceil(W) - W) / ceil(W), above 0.15 |
| uncoalesced loads | sectors per request above 5 |
| sparse stores | bytes per sector of global stores below 16 |
| latency-bound | long-scoreboard stall ratio above 3 (high confidence if DRAM read is below 10% of peak) |
| compute without tensor cores | FMA pipe above 50% and tensor pipe below 1% |
| atomic contention | atomic and reduction sectors above 0 and the lg_throttle stall ratio above 1.0 (low confidence) |
| bank conflicts | short-scoreboard stall ratio above 2 (low confidence, heuristic) |
| synchronization overhead | barrier stall ratio above 2 |
| low achieved occupancy | achieved warps below half of the theoretical value, when that value is above 50% |
| register pressure | any local load or store (high), or more than 128 registers per thread (medium) |
| unintended FP64 | FP64 pipe above 1% |
| warp divergence | threads per executed instruction below 24 |

Caveats from the same sources:

- **Treat ncu's recommendations and its "% est. speedup" as hypotheses.** In one atrex wiki record all three independent fixes of ncu-cited levers regressed wall time; three such strikes are a stop signal, and cheap changes that reduce contention inside the existing structure succeed more often than restructures.
- **ncu's duration under replay is not production latency** (565 us under ncu against 383 us in an nsys split of the same kernel in that record). Use nsys or a strict benchmark for latency and ncu for counters.
- Missing counters are absent, not zero; a metrics-only run shows N/A in sections; use `--set full` for complete data.
- Counter access can need an elevated shell (`ERR_NVGPUCTRPERM`).
- An optimization that helps at a large size can hurt at a small one. Record the failed ones with the context, the result, the reason and a retry condition.

### The same quantities with nvprof on this host (Pascal)

From NVIDIA's `nvprof` metric reference, not from the surveyed repositories; confirm the exact names on the machine with `nvprof --query-metrics`.

| Nsight Compute quantity | nvprof metric or flag |
|---|---|
| achieved occupancy | `achieved_occupancy` |
| SM utilization | `sm_efficiency`, `ipc`, `issue_slot_utilization` |
| DRAM bandwidth | `dram_utilization`, `dram_read_throughput`, `dram_write_throughput` |
| coalescing | `gld_transactions_per_request`, `gld_efficiency`, `gst_efficiency` |
| shared-memory conflicts | `shared_efficiency`, `shared_load_transactions_per_request` |
| warp stalls | `stall_memory_dependency`, `stall_sync`, `stall_exec_dependency` |
| warp divergence | `warp_execution_efficiency` |
| spills | `local_load_transactions`, `local_store_transactions` |
| launch statistics | `--print-gpu-trace` (grid, block, registers per thread, static and dynamic shared memory) or `nvcc --ptxas-options=-v` |

## 7. Case study: the pair residual test

`_fe_sufficient_summary_pairs.py` scores the joint MI of the MRMR residual with every pair of raw columns (the early stop of the FE search used to miss interactions such as a product or an XOR, because the per-column test is blind to them).

Measured at n=5000 rows, p=400 columns (79,800 pairs), warm, same input, same result (best pair (3, 7), MI 0.638576):

| path | time |
|---|---|
| CPU kernel v1 (row-parallel) | 0.37 to 0.42 s |
| CPU kernel v2 (one pass per pair, marginals from the table, rows handed out in mirrored pairs) | 0.19 s |
| generic dispatcher, njit | 3.66 s |
| generic dispatcher, numba.cuda | 2.44 s |
| generic dispatcher, cupy | 149.6 s |
| generic dispatcher, auto (chose cuda) | 2.80 s |

Findings:

- The generic GPU dispatcher is about twelve times slower than the CPU kernel here. It is built for few pairs over very many rows (its tuning grid is 16 to 256 pairs and 50 thousand to 2 million rows), and this workload is many pairs over a few thousand rows. That is why the module has its own fused kernel, and why the generic path is kept only as a measured baseline.
- v2 beat v1 by 1.9x from three changes: no marginal counter in the hot loop (derived from the count table afterwards), the per-pair cell index built from a precomputed row base, and mirrored row assignment so that the first threads do not get most of the pairs.
- The fused cupy kernel (block per pair, shared-memory histogram, in-place MI reduction, device argmax) agrees with the CPU kernel on the best pair for both REAL types, and covers every pair exactly once wherever the winner sits relative to the block boundaries (tests in `tests/feature_selection/fe/test_sufficient_summary_pairs.py`).
- Fused kernel, same input (p=400, n=5000): 16 ms against 245 ms for the CPU kernel (about 15x warm and synchronized; 13x in the earlier A/B), and about 149x faster than the generic dispatcher on cupy. The v2 kernel (a thread per cell, warp-shuffle reduction) is 13 to 29% faster than v1 (tree reduction with 7 barriers). `{REAL}`=float is 2.2x faster than double on this Pascal part, whose double rate is 1/32.
- The GPU has a fixed cost of about 0.5 ms per launch, so the CPU wins below roughly one million pair-rows (pairs times rows). The KTC sweep (20 cells, 119 s) sends 100 pairs at 1500 and 5000 rows to the CPU and every other cell to the GPU, and the equivalence tolerance of the sweep follows the crit dtype: `max(1e-9, 8 * eps(REAL))`. A first version used a fixed 1e-9, dropped the GPU variant as divergent (its difference to the CPU is about 1e-8 for float terms) and chose the CPU in every cell; that is the case `tests/test_meta/test_ktc_dispatch_selects_more_than_one_path.py` pins.
- nvprof counters of the v1 kernel: achieved occupancy 0.98, SM efficiency 99.9%, IPC 2.84, issue-slot utilization 67%, DRAM utilization low (4.6 GB/s read), warp execution efficiency 96.5%, shared-memory efficiency 4.5%, stalls: memory dependency 31%, synchronization 29%, execution dependency 21%. The kernel is bound by shared-memory atomics and barriers, not by DRAM. The v2 kernel (default): occupancy 0.96, IPC 3.08, issue-slot utilization 72%, warp execution efficiency 98.3%, synchronization stall 5.9% (from 29%), memory-dependency stall 31.7%, shared efficiency 2.1%; the barrier stall is gone and the shared-memory atomics are the remaining limiter. The full record is `audits/2026-10-22/profiles/_fe_sufficient_summary_pairs.md`.
- Candidate optimizations not done: 4-byte vector loads with a padded row stride, an upper-triangle-only launch for the diagonal blocks (5 to 10% of the blocks are redundant), warp-private histograms to lift the shared-memory efficiency.

## 8. AI-assisted CUDA: what exists

Surveyed in October 2026. All of it is third-party and none of it is an Anthropic product.

How it was read. The primary files (the `SKILL.md` bodies, the reference guides, the profiling scripts) were fetched from the repositories and read through a fetch tool that returns an extraction made by a small model over the page text. The commands, metric names and thresholds below come from those extractions, and every URL is given so that a number can be spot-checked. Nothing was installed or run. A `SKILL.md` is instructions that execute with your permissions, and the technical content is available by reading the text, so installing added risk and nothing else.

### technillogue `cuda` ([repo](https://github.com/technillogue/ptx-isa-markdown))

230 stars, no repository license shown (the converted NVIDIA documentation is copyright NVIDIA), 5 commits. `cuda_skill/SKILL.md` is about 1,900 words: Core Philosophy, Debugging Workflow, Performance Optimization Workflow, Compilation Reference, Local API Documentation, Checklist. It defers the detail to `references/ncu-guide.md` (about 250 lines) and `references/performance-traps.md` (about 700 words), both read.

- Debugging: reproduce on the smallest input, `printf` in device code guarded by `if (idx == 0)`, then `compute-sanitizer --tool memcheck | racecheck | initcheck | synccheck`, then `cuda-gdb -batch -ex "run" -ex "bt"`, then minimize the diff between working and broken code. Build with `nvcc -g -G -lineinfo`; inspect with `cuobjdump -ptx | -sass`. An out-of-bounds shared-memory report often means the dynamic shared-memory size in `<<<grid, block, smem>>>` is too small, not the indexing.
- Performance: baseline, `nsys profile -o report` and `nsys stats report.nsys-rep --report cuda_gpu_kern_sum` to find hotspots, `ncu` to explain one kernel, one hypothesis, change one thing, re-profile. Registers: `nvcc --ptxas-options=-v`. Instrumentation: NVTX ranges with `--trace=cuda,nvtx`.
- Time budget advice: spend about 80% of the optimization time on the problems that were not predicted.
- `ncu-guide.md`: sets `basic`, `full`, `memory`, `compute`, `launch`, `occupancy`, `roofline`, `source`; sections LaunchStatistics, Occupancy, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, SchedulerStatistics, WarpStateStatistics, SpeedOfLight, SourceCounters; the metric names listed in section 6; filters `--kernel-name`, `--kernel-name-base "regex:..."`, `--launch-skip N`, `--launch-count M`. Caveat in the file itself: expert-system recommendations are heuristic, "treat recommendations as hypotheses to test, not prescriptions". The guide is slightly inconsistent on bank conflicts (it routes them through SchedulerStatistics while naming `l1tex__data_bank_conflicts_*` counters) and on `-lineinfo` versus `-G`.
- `performance-traps.md` covers only bank conflicts (pad `[32][32]` to `[32][33]`, stated gain 2 to 3x when memory-bound), coalescing (sectors per request 1 to 4 good, 8 to 16 poor, 32 and more essentially random; vectorized loads, SoA instead of AoS, 128-byte alignment; stated gain 1.5 to 2x), scale-dependent optimizations ("always profile at your target scale") and recording failed optimizations with context, result, reason and a retry condition. It has nothing on occupancy, atomics, divergence or precision.

### AKO4ALL ([repo](https://github.com/TongmingLAIC/ako4all))

MIT, 379 stars, 7 commits, young. The entry point is `SKILL.md`; the benchmark lives in `bench/kernelbench/` (`GUIDE.md`); `knowledge/` is empty apart from `.gitkeep` and is meant for user-supplied notes.

- **No `ncu` command, section or metric appears in its `SKILL.md` or README.** The skill treats `ncu` as best effort: probe once after the baseline, profile again after three stagnant iterations, and "don't gate iteration progress on ncu availability". Its value is the loop and the benchmark protocol, not the profiling.
- Loop: a `bash scripts/bench.sh iter-N` step, then a log entry in `ITERATIONS.md`, then `git commit -m "[iter N] <direction>"`; three consecutive iterations under 3% improvement mean stop and re-assess; stop on an iteration cap, a physical floor, or exhausted directions with at least three documented.
- Benchmark protocol (`GUIDE.md`): tolerances atol = rtol = 1e-4 for float32 and 1e-2 for float16 and bfloat16; 10 correctness trials; 50 performance trials with the first discarded, 10 untimed warmups, the median reported; device-side timing with CUDA events by default, with the L2 cache thrashed before each trial (cold-cache numbers), or wall-clock `host_time`; fresh input values before every timed trial; after timing the inputs are re-randomized in place and the result must still match the reference (a mutation sentinel); outputs `COMPILED`, `CORRECT`, `RUNTIME`, `REF_RUNTIME`, `SPEEDUP`, `DEVIATION`, `MUTATION_SENTINEL`; a speedup above 10x is flagged for review. The guide does not say whether the event timing synchronizes.
- Rules against cheating: no stream injection to evade timing, no monkey-patching the benchmark, no returning uninitialized results, no `get_inputs` in the solution file.
- Red flag: it generates and runs a benchmark script, so read `bench-wrapper.sh` before using it.

### Alibaba `atrex-kernel-agent` ([repo](https://github.com/alibaba/atrex-kernel-agent))

Apache-2.0, 190 stars, 98 commits, tags v0.1.0 to v0.3.0. An agent framework, not a single skill. All correctness, benchmark and profiling work goes through a sandbox command line; a supervisor, not the agent, owns acceptance; full candidates must pass an ABBA schedule on the same allocation. The profiling material is the part worth reading:

- `src/aka/legacy/execution/profiling/nvidia/profile_nvidia.sh` (about 390 lines): `ncu --set full --force-overwrite --launch-skip N --launch-count M --kill yes [--kernel-name NAME] -o report python kernel.py`, and with a source pass `ncu --set source --section SourceCounters ...`. Reports open with `ncu --import`; helper scripts export key metrics, stall hotspots per source line and disassembly.
- `profiling/profile_guide.md`: Step 0 is mandatory, locate the launch with `ncu --print-summary per-kernel`, because framework setup launches kernels before the target and a blind `--launch-skip 10` can profile the wrong one; then kernel time, then compute utilization, and only below 90% a full profile; then warp stalls. Starting section: SpeedOfLight. "Memory SOL is not DRAM utilization." Missing counters stay absent, not zero; a metrics-only run shows N/A in sections, so use `--set full`; `ERR_NVGPUCTRPERM` needs elevated permissions.
- `profiling/nvidia/ncu_classification.py`: fifteen rules with numeric thresholds (listed in section 6).
- Its `gpu-wiki` carries pitfall records, two of which were read: ncu's replay-inflated duration is not production latency (565 us under ncu against 383 us in an nsys split for the same kernel), and ncu's per-rule "% est. speedup" does not predict wall time (three independent fixes of ncu-cited levers all regressed wall time; the advice is to treat three such strikes as a stop signal, and to prefer cheap changes that reduce contention inside the existing structure).
- The helper scripts that parse the reports come from the separate `ncu-report-skill` project, now read (section 9). The wiki has several hundred records under `gpu-wiki/kernel_wiki/records/` (anti-strategy, doc, symptom-card, technique-card; NVIDIA, AMD, Triton, CuTe DSL, Gluon, FlyDSL). Ten were read, chosen for relevance to atomics, histograms, shared memory, reductions, occupancy and measurement; the rest were not:
  - `ncu-measurement-discipline`: ncu injects a library, serializes launches and replays the kernel over several passes, so its `Duration` is not the latency of a timing harness; short kernels measured across separate processes drift with cold cache, JIT warm-up, lazy library init, thermal state and neighbour load, so establish your own noise floor (the record gives no number); under CUDA-graph capture a kernel goes through `cuGraphLaunch` and a profile can differ from the eager path; a kernel that runs at capture and is skipped at replay leaves stale but possibly correct-looking output; one harness saw a fixed 80 us between entering the launch function and the first GPU instruction, so anything near that is overhead.
  - `ncu-rule-est-speedup-meta-rules`: the estimate is reliable when one stall exceeds 60% of the cycles per instruction and the fix is structural and additive, and misleading when three or more stalls each take 30% or more, because the parallelism the rule calls wasteful was hiding the other stall (the page is a summary; its case-study rows are missing).
  - `smem-swizzling-bank-conflicts`: 32 banks of 4 bytes, tree reductions `sdata[tid] += sdata[tid + s]` give 2-way to 32-way conflicts as the stride grows, `[32][33]` padding costs about 3% extra shared memory, an XOR swizzle `Swizzle<3,4,3>` serves a 128-byte row; the source says its own bit-field labels are inconsistent, so verify the positions in code.
  - `hierarchical-reduction-memory-bound`: reduce in levels (registers, warp shuffle with `__shfl_xor_sync` in 5 steps, shared memory across warps, then a cluster on Hopper); 128 threads for a row up to 4K elements, 256 for up to 16K; multi-pass kernels pay for every extra pass (one pass reaches the full bandwidth, two passes half); its own SOL arithmetic is inconsistent (16 GB against 17.2 GB), so do not reuse the 89.7% figure.
  - `nsight-profiling-practice`: `nsys profile --trace=cuda,nvtx,osrt --delay --duration --cudabacktrace`, NVTX ranges to label phases, `ncu --set detailed --kernel-name "regex:..."`, build with `-O3 -lineinfo` and the right `-arch`, never profile `-O0`; "GPU utilization at 100% means the kernel is optimal" is listed as a misconception.
  - `sm120-moe-data-prep-pitfalls` item 1 (the closest record to our histogram work): warp-aggregated shared-memory `atomicAdd` through `__match_any_sync` regressed wall time by 24.5% with 256 buckets, because the expected number of same-bucket lanes in a warp was about 1.94 and the break-even is roughly more than 4 duplicates; the bank-conflict ratio did not move (68.52% to 68.57%), which points at inter-warp contention that intra-warp aggregation cannot fix. Our pair kernel has k*k*kr = 250 cells and 128 threads, so the same arithmetic applies: do not add warp aggregation before measuring the duplicate rate.
  - `sm120-moe-data-prep-pitfalls` item 3: removing a barrier-bound histogram's barriers by warp specialization (31 of 32 warps exit early) cut warp cycles per instruction by 43% but dropped achieved occupancy from 64.6% to 20.8% and made the kernel 25% slower; the barriers were natural convergence points where all warps hid each other's load latency.
  - `register-pressure-warp-occupancy`: resident warps are the minimum of the register, architecture and shared-memory limits; check spills with `nvcc --ptxas-options=-v` (a local load or store is a spill; a spill in the inner loop is the priority; local memory costs about 30 to 100 cycles on an L1 hit and 200 to 800 from L2 or DRAM); `max_regs_per_thread = 65536 / (maxThreadsPerBlock x minBlocksPerMultiprocessor)`; Hopper and Blackwell specifics (`setmaxnreg`, TMEM accumulators) do not apply to our GPUs.
  - `sm120-ncu-l1-hit-rate-shared-pollution`: `l1tex__t_sector_hit_rate.pct` also counts shared-memory hits, because L1 and shared memory are one SRAM, so it is not evidence of global-load L1 hits on a kernel that uses shared memory heavily; the record states the same sharing for Ampere and Ada, and its own lesson field is empty.

### tensormux `kernel-skills` ([repo](https://github.com/tensormux/kernel-skills))

MIT, 86 stars, 47 commits, default branch `master`. 35 skills in six categories (CUDA 9, Triton 5, inference 10, quantization 3, patterns 5, portability 3). Eleven were read in full: the four below, and seven whose rules are folded into section 3 (global-memory access, shared-memory tiling, warp divergence, numerical stability, boundary conditions, tile size and partitioning, elementwise fusion):

- `cuda/choose-launch-configuration`: block size a multiple of 32; start with 128 or 256, 512 when registers are very low, avoid 1024 (two blocks per SM on a 2048-thread SM); occupancy is the minimum of the register limit (65,536 registers per SM divided by registers per block), the shared-memory limit and the hardware block limit (32 resident blocks on SM70 and later); aim at 50% or more, but a compute-bound kernel is fine near 25% and a spill is not worth a higher occupancy; get the real register count from `ptxas -v` (hand estimates can be off by 2x), and validate with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; `gridDim.y` and `gridDim.z` are below 65,535; a launch costs about 5 to 20 us, so under 100 blocks consider fusion or a persistent kernel; guard every global access with `if (idx < N)`; a non-multiple-of-32 block can break warp primitives unless the mask matches the active threads.
- `cuda/write-cuda-reduction-kernel`: warp-shuffle tree with `__shfl_down_sync` and an explicit mask (offsets from 16); one value per warp into shared memory with a `__syncthreads()` between write and read; the second stage uses a mask that matches the number of partials; the two-pass form (per-block partials, then a second kernel) is preferred for reproducibility; floating-point `atomicAdd` is non-deterministic, so initialize the output to the identity from the host and document it; accumulate fp16 and bf16 sums in fp32; carry `(value, index)` through argmax and take the smaller index on ties; test N = 1, 32, 33, 1024, 1025 and a large N against a CPU reference.
- `cuda/debug-cuda-kernel-correctness`: test sizes such as 31, 63, 65, 127, 129 ("powers of two are not representative of all failure modes"); classify the error pattern (uniform scale, wrong rows or columns, last rows only, random few elements means a race, NaN or Inf); initialize outputs to a sentinel because a zeroed buffer can mask a bug; `compute-sanitizer --tool memcheck` is mandatory for random or intermittent failures; isolate one block; disable fast math when hunting a race; correctness must not depend on the block count.
- `patterns/write-kernel-test-plan`: the reference must not share code or algorithm with the kernel; tolerances fp64 reference against fp32 kernel rtol 1e-5 and atol 1e-8, fp16 rtol 1e-2 and atol 1e-3, bf16 rtol 5e-2 and atol 1e-3; shape sweep of 1, 7, 13, 37, 64, 128, 512, 4096, 100, 1000, 4097, primes 1009 and 8191, and tile-size plus or minus one; contiguous, transposed and sliced layouts; adversarial inputs (zeros, ones, large, near underflow, alternating signs, a single NaN); a fixed seed in every test; time with CUDA events or a benchmark timer and not wall-clock alone, exclude the first call, report GB/s or TFLOPS, include one case larger than L2; the kernel must not be more than 10% slower than the baseline; run `compute-sanitizer` memcheck and racecheck in CI, not in the main test loop.
- Not read: the other 24 skills (GEMM, layernorm and softmax kernels, the inference and Triton skills, quantization, portability). None of the eleven covers shared-memory histograms or integer atomics for count tables.

### Hugging Face `cuda-kernels` and `upskill` ([docs](https://huggingface.co/docs/kernels/en/cli-skills), [skill](https://www.skills.sh/huggingface/kernels/cuda-kernels), [blog](https://huggingface.co/blog/upskill))

The primary file was read this time: `kernel-builder/skills/cuda-kernels/SKILL.md` in `huggingface/kernels` (663 lines, fetched with `gh api` as data, not installed; the skill-site page truncates the body). It is a kernel-builder and Hub skill for diffusers and transformers on H100, A100 and T4, and most of it does not apply to a numba and cupy codebase:

- About a third is hard build constraints: ABI3 (Python limited API), so no pybind11, no hand-written `setup.py`, `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` for bindings, relative imports only, `torch` as the only runtime dependency, a git repository for Nix builds, and a pre-flight checklist of seven greps and commands. None of this concerns us.
- Architecture table: H100 132 SMs, 192 KB shared memory per SM, 50 MB L2, 3.35 TB/s; A100 108 SMs, 164 KB, 2.0 TB/s; T4 40 SMs, 64 KB, 320 GB/s, no bf16. Guidance "aim for grid sizes that are multiples of the SM count" and "all reductions use warp shuffles".
- Patterns: vectorized access with `__nv_bfloat162`, `__half2` and `float4` (`#pragma unroll 4`); `warp_reduce_sum` as a template over `__shfl_xor_sync(0xffffffff, val, offset)` for offsets 16 down to 1; `BLOCK_SIZE = 256` for element-wise kernels; for reductions `threads = min(hidden/2, MAX_THREADS)` rounded up to a multiple of 32; attention tiles 128 by 64 by 64 with 8 warps.
- Profiling is two lines: `nsys profile -o profile python script.py` and `ncu --set full -o metrics python script.py`. It describes no timing method; the benchmark scripts take `--warmup-iterations` (default 2) and were not read.
- A useful lesson in its own numbers: a vectorized RMSNorm kernel was 2.67x faster than PyTorch in isolation (38% of the H100 bandwidth), yet the end-to-end speedup was 6% (RMSNorm is about 5% of the total compute), against 34% for plain `torch.compile`; the custom kernels and `torch.compile` could not be combined. This is the same finding as our "a kernel microbenchmark can lie" rule.

The `upskill` blog reports that a Claude Opus 4.5 generated CUDA skill gave Opus itself no clear accuracy gain and raised the token use, helped some smaller models (a local GLM-4.7-Flash from 40% to 85% in the sample evaluation), and was tested with simple input and expected-output cases, not kernel speedups.

### What to take

1. **Profile before editing, change one thing, keep an iteration log, keep a reference as the golden.** All the sources agree, and it matches our rules.
2. **The Nsight Compute rule set, metric names and thresholds** of section 6, as the reading guide for any profile.
3. **The three honesty rules about `ncu`:** its duration is not latency, its estimated speedup does not predict the wall, and three strikes are a stop signal. They agree with our `nvprof` rule.
4. **The benchmark protocol of AKO4ALL** (discard the first trial, warm up, median, fresh inputs, mutation sentinel, cold-cache option) and **the test plan of tensormux** (size sweeps around tile boundaries, sentinel-initialized outputs, adversarial inputs, tolerances per dtype, sanitizers in CI).
5. **Do not copy** the Hugging Face build and architecture rules; they solve a different problem. Take its one transferable lesson: a kernel that is 2.67x faster in isolation moved the end-to-end time by 6%.
5a. **The atrex histogram records** (warp-aggregated atomics lose below about four duplicates per warp; removing barriers by warp specialization collapses occupancy) as a reason to measure before restructuring our histogram kernel, and the ncu-report-skill playbook (section 9) as the reading guide on the newer-GPU machine.
6. **No surveyed source covers shared-memory histograms or integer atomics for count tables.** Our own measurements in sections 3 and 7 are the guidance for that.
7. **A project-specific skill would be worth more than a generic one.** It would encode the contract of section 2, the `{REAL}` and `{ACC}` placeholders, the KTC registration, the streaming rule and the honest-timing rules of section 6, with `hermite_fe/_als_kernels_gpu.py` as the reference module.

## 9. Profiling on the newer-GPU machine with Nsight Compute (ncu-report-skill)

For the second machine, where `ncu` and `nsys` work and `nvprof` does not. Source: `mit-han-lab/ncu-report-skill` ([repo](https://github.com/mit-han-lab/ncu-report-skill)), MIT, 259 stars, 13 commits. The atrex project names this skill as the origin of its report parsers under a different owner (`DongyunZou/ncu-report-skill`); the repository that resolves today is the MIT one. It was read as text (`SKILL.md`, and reference files 03, 05, 06, 08 and 09 through the extraction tool; `README.md`, `blackwell-cuda-programming.md`, the helper scripts and reference files 00, 01, 02, 04 and 07 were not). Nothing was installed or run, and the helper scripts were not audited, so read them before running.

Scope and limits. It targets Nsight Compute 2026.3.1 and Blackwell (B200 sm_100 and B300 sm_103, 148 SMs), with "general advice for other NVIDIA GPUs". Metric names and defaults depend on the ncu version and the architecture, so every name below is a starting point to check with `ncu --query-metrics` or `action.metric_names()` on the machine. The advice about boost clocks and removed sections is specific to 2026.1 to 2026.3.

### What it provides

A workflow (decide what to profile, get a target with line information, collect one report, parse it with helpers, work through six dimensions, match the signals to a playbook, write an evidence-ranked report) and helpers: `analyze_reports.py` (key metrics, the ncu rule results sorted by estimated speedup, comparisons), `extract_stall_hotspots.py` (per-line stalls; `--binary` attributes inlined CUTLASS, CuTe and Triton code to the kernel's own lines), `plot_timeline.py` (ASCII PM-sampling timelines for tail effects), `ncu_utils.py`, a standalone harness template. Only `ncu` and the profiled program need the GPU; the helpers read reports and can run on any machine, so a report can be collected on the GPU machine and analysed on this one (ncu can read older reports).

### Collection recipes (ncu 2026.3)

- Survey the kernels first: `ncu --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second -o survey.ncu-rep ./harness`, then list the names with `--import survey.ncu-rep --print-summary per-kernel --print-kernel-base function`, and use those names for `-k`.
- Full profile: `ncu --set full --import-source yes --clock-control boost -k "regex:K" -c 1 -o full.ncu-rep ./harness`. Runtime grows with the number of replay passes, so a millisecond-scale kernel is slow; profile a smaller representative workload, and keep expensive setup outside the profiled window because it repeats on every replay.
- `--section SourceCounters` re-checks the hot lines in a few passes (`--set source` adds no sections, per this source; the atrex script uses it with `--section SourceCounters`, which agrees). `--page details` on an existing report shows the built-in rule suggestions; `--page raw --csv` exports a metric table; `--page source --print-source cuda,sass` shows per-instruction counters; `--metrics LIST` takes one or two passes and also collects metrics no section holds.
- Sets: `basic` (default smoke test), `detailed` (adds compute and memory analysis, SourceCounters, roofline), `full` (everything, including PM sampling, scheduler and warp-state statistics; start here), `pmsampling`, `roofline`, `nvlink`.
- Skip warm-up launches with `-s N`, cap with `-c N`; for JIT or autotuning code use `--profile-from-start off` and bracket the target call with `cudaProfilerStart/Stop` or `torch.cuda.profiler.start()/stop()`. For us the equivalent is the cupy profiler window used with `nsys` in section 6.
- Clock control: ncu locks the clock while profiling; the default is `boost` in 2026.1 to 2026.3 and `base` in older versions, so name the setting in the command. To match an externally locked clock, lock it with `nvidia-smi -lgc`, pass `--clock-control none`, and reset with `nvidia-smi -rgc`. Read the actual clock from `sm__cycles_elapsed.avg.per_second`, and compare it across reports before comparing durations.
- Template kernels: `-k` matches the function name; add `--kernel-name-base demangled` and the full name for instances. `-f` overwrites, and give `-o` an explicit `.ncu-rep` extension (2026.3 appends `.ncu-repz` otherwise). `--import-source yes` embeds the source at collection time only, so add `--source-folders` when the build paths are absent, and build with `-lineinfo`. Never use `-G` (about 100x slower and useless for profiling).
- ncu must launch the program itself: `ncu ... python app.py` or the interpreter path; wrapping ncu around a launcher such as `uv run` hangs, so put ncu inside it.

### The six analysis dimensions

1. **Occupancy and launch geometry**: `launch__grid_size`, `launch__waves_per_multiprocessor`, `launch__occupancy_limit_{blocks,registers,shared_mem,warps}`, theoretical `sm__maximum_warps_per_active_cycle_pct`, achieved `sm__warps_active.avg.pct_of_peak_sustained_active`. Waves per SM under 1 means idle SMs; between 1 and 2 a tail wave; above 4 the scheduling averages out; theoretical 100% with low achieved occupancy is stall-bound.
2. **Block balance (tail effect)**: `sm__cycles_active.{avg,max,min,sum}`; per-block work max over average above 5x signals tail risk; an ncu message "max X% above, min Y% below" with large values is imbalance.
3. **Stalls and per-line hotspots**: `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` and the per-PC `smsp__pcsamp_*` counters. Share of samples: `long_scoreboard` above 40% is memory-latency bound; `short_scoreboard` above 30% is dependency chains or heavy shared-memory use; `barrier` above 20% is too much synchronization or divergence before barriers; `selected` under 10% is stall-bound. Barrier stalls are charged to the following instruction, and in warp-specialized kernels idle producer warps inflate the samples.
4. **Tensor cores**: pipe utilization; irrelevant to our integer histogram and MI kernels.
5. **SM utilization timeline** (PM sampling, about 1 us resolution, so kernels under 20 us give few samples and vGPU or MIG give none): flat-high with a clean drop is ideal, a long tail is dimension 2, flat-low is dimension 1 or 3, a compute and memory sawtooth is missing overlap.
6. **Memory pattern and cache**: `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` (80 to 100% is bandwidth-bound, under 10% on a slow kernel is latency- or compute-bound), `l1tex__t_sector_hit_rate.pct` (above 90% good; see the shared-memory caveat in section 8), `lts__t_sector_hit_rate.pct` (under 50% poor), sectors per request against the ideal for the access size, store bytes per sector under 16, any local load or store as a spill, and shared memory at zero as possibly missed reuse.

End a diagnosis with one sentence naming the top three or four signals, each tied to its dimension.

### Diagnosis playbook (signal, cause, first fix)

| | pattern | signal | first fix |
|---|---|---|---|
| A | small grid | waves per SM under 0.5 | split the work finer (split-K, per head or channel, grid-stride loops) |
| B | tail effect | max over average length above 3, wide per-SM cycle spread, waves above 1.05 with a partial last wave | sort or pack by length, split long items across more blocks |
| C | uncoalesced loads | sectors per request well above 4, 8 or 16 for 4-, 8- or 16-byte loads | remap thread to data, structure of arrays, stage through shared memory, vector loads |
| D | sparse stores | bytes per store sector under 16 | gather with shuffles or shared memory so contiguous lanes write contiguous addresses |
| E | latency-bound | `long_scoreboard` above 40%, above 3 stalls per issue, DRAM under 10% | issue several loads before use, add warps, `cp.async`, software pipelining |
| F | no tensor cores | FMA pipe above 50%, tensor pipe 0% on a matmul-shaped kernel | tensor-core instructions or a library |
| G | atomic contention | heavy `long_scoreboard` on ATOM or RED, high L2 atomic traffic | reduce hierarchically: warp shuffles, then shared memory, then one atomic per block; bucket the updates |
| H | bank conflicts | high shared wavefront utilization, `short_scoreboard` on shared loads | pad `[32][33]`, swizzle, verify per-instruction wavefronts |
| I | synchronization | barrier stalls above 20%, hotspot on `BAR.SYNC` | warp-level primitives, fewer barriers, warp specialization with mbarrier |
| J | occupancy gap | theoretical above 50%, achieved well below | fix the stall or imbalance behind it (usually B, E, H or I) |
| K | register spill | local load or store above 0, over 128 registers per thread | `__launch_bounds__`, fewer live values, per-thread arrays to shared memory |
| L | unintended FP64 | FP64 pipe active in an FP32 kernel | `f` suffix on literals, `__expf`, `__logf`, `__sinf` |
| M | pipeline bubbles | compute and DRAM alternate in a sawtooth | double-buffer shared tiles, multi-stage `cp.async` |
| N | divergence | threads per instruction well below 32, low branch efficiency | warp-uniform branches by sorting or partitioning, branchless select when both sides are cheap |
| O | serial chain | `wait` and `short_scoreboard` dominate, issue-active under 40%, runtime scales with chain length | split state across more warps or blocks, fewer larger steps |

For our kernels the live candidates are G (the count histogram in shared memory; the atrex record in section 8 says to measure the duplicate rate before adding warp aggregation), I (the 29% synchronization stall of v1), H (shared efficiency 4.5%), A (few pairs in a block) and L (check the FP64 pipe: with a float `{REAL}` it should be near zero except for the `{ACC}` sums). Rank matches by the ncu estimated speedup only after checking each estimate against the counters it rests on; the source says the estimates for small grids, bank conflicts, compression and broadcast loads are sometimes wrong, and that they overlap and sum to more than 100%.

### Metric names that changed between versions

| old | new |
|---|---|
| `smsp__inst_executed_op_{global,local,shared}_{ld,st}.sum` | `smsp__sass_inst_executed_op_{global,local,shared}_{ld,st}.sum` |
| `smsp__warps_issue_stalled_<reason>_per_issue_active.pct` | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |
| `pmsampling:sm__throughput.*`, `pmsampling:dram__throughput.*` | `<UNIT>.<group>.<metric>` series such as `TPC.TriageCompute.sm__...` and `FBSP.TriageCompute.dram__throughput.*` |

Not collected by `--set full`, so request them with `--metrics` or compute them: `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` (use sectors divided by requests), `dram__bytes.sum` (use read plus write), `sm__inst_executed_pipe_fmaheavy.*` (use `..._pipe_fma.*`). The `PmSampling_WarpStates` section exists in 2026.1 only and is an error in 2026.3. A metric that `--query-metrics` lists but returns `None` was not collected; wrap reads in a helper with a default. Our section 6 table uses the older `smsp__warp_issue_stalled_*` spelling and `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, which is still the recommended replacement; check the stall names on the machine before scripting them.

### Permissions, hangs and other issues

- `ERR_NVGPUCTRPERM`: on Linux run `sudo ncu`, or set the driver option `NVreg_RestrictProfilingToAdminUsers=0` and reboot, or start the container with `--cap-add SYS_ADMIN`. On Windows the driver setting is in the NVIDIA control panel developer options, or run elevated, as we did for `nvprof`.
- `Unable to access the following 6 metrics: ctc__...` is harmless. `Could not deploy stock section files`: set `HOME` to a writable directory.
- Crashes or NaN only under ncu: ncu saves and restores memory between replays and exposes reads of uninitialized global memory; initialize all inputs.
- Empty source view: no `-lineinfo`. For cupy `RawKernel` and numba.cuda this means passing the option to the compiler (`options=("-lineinfo",)` for cupy; check the numba equivalent) and not using `-G`.
- Results jitter: clocks. Compare `sm__cycles_elapsed.avg.per_second`, enable persistent mode with `nvidia-smi -pm 1`, pin a stream, and compare `ncu --version`, driver version and compile flags with any other machine's report.
- How to judge `sm__throughput`: a GEMM should be above 50% on Blackwell; element-wise and reduction kernels often sit at 10 to 30% because they are DRAM-bound, so read it next to the DRAM percentage. Both SM and DRAM idle means latency-bound.
- Our inference, not from the source: a cupy `RawKernel` is an `extern "C"` function, so `-k "regex:pair_mi_block"` should match it by name; confirm with the survey recipe above. The Windows driver setting for `ERR_NVGPUCTRPERM` and the cupy `-lineinfo` option are also our additions, to be checked on the machine.

### How this machine's profile should be recorded

Cross-machine numbers are not comparable: record the GPU model, the architecture, the ncu version, the clock-control setting and the achieved clock next to every figure, and keep one profile record per kernel version under `audits/2026-10-22/profiles/`. A kernel-tuning-cache entry is per hardware fingerprint, so the sweep must be rerun on the newer machine, and the single-path allowances in `test_ktc_dispatch_selects_more_than_one_path.py` (CPU-only choices measured on the GTX 1050 Ti) are the first thing to re-check there.

## 10. Checklist for a new GPU module

1. Look for an existing batched GPU entry to reuse and measure it on your own shape before writing a kernel.
2. Write the CPU kernel first (numba `@njit`, parallel where it pays) and keep it as the reference.
3. Write the fused GPU kernel with `{REAL}` and `{ACC}`; compile one module per type.
4. Stream by column blocks with a RAM and a VRAM bound, and an out-of-memory fallback to the CPU kernel.
5. Register with the KTC, with a fallback heuristic and a sweep over the module's own axes.
6. Tests: kernel versions equal; streamed equals unblocked for several block widths; GPU equals CPU for both REAL types; every pair covered once at block boundaries; memory-bound test; a regression test for the behaviour the module exists for.
7. Profile: cProfile on the host path; a GPU profiler with occupancy, throughput, launch statistics and launches per call; record both under `audits/2026-10-22/profiles/<module>.md`.
8. Move the module from `pending` to `compliant` in `tests/test_meta/_fe_gpu_contract.json`.

## 11. Open items

- Try warp-private histograms in the fused pair kernel (the v2 counters show the memory-dependency stall and 2.1% shared efficiency as the limiter). Do a profile on the newer-GPU machine with `ncu --set full` and compare the stall mix and the FP64 pipe (section 9).
- Walk the 90 pending FE modules against points 2 to 5 of the contract, starting with those that have the most expensive kernels.
- Decide whether to write the project-specific CUDA skill described above.
- Re-measure the kernels that the cache sends only to the CPU on this host (`cmi_batched_cpu_cuda_crossover`, `discretize_2d_array`) on the newer GPU.
- Not read from the surveyed sources: 24 of the 35 tensormux skills (none concerns histograms or atomics), the helper scripts and the remaining reference files of `ncu-report-skill`, and most of the atrex `gpu-wiki` records (ten were read, section 8). The Hugging Face benchmark scripts were not read, so its timing method is unknown.
- Spot-check the quoted thresholds against the URLs in section 8 before relying on one in a decision; the extractions were made by a small model.
