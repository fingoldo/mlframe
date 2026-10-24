# CUDA in mlframe

How GPU code is written, dispatched, tested and profiled in this repository. It merges what we learned writing the feature-engineering (FE) kernels with what is publicly available for AI-assisted CUDA work. Where a statement comes from outside, the source is named. External claims were read from the primary files through an extraction tool (section 8 says how) and were not reproduced on this host, so a number from outside is a pointer to check, not a measurement of ours.

Scope: the FE modules the MRMR layer imports (`src/mlframe/feature_selection/filters/`). The per-module status is kept in `audits/2026-10-22/fe_gpu_coverage.md`, and the contract below is pinned by `tests/test_meta/test_fe_gpu_streaming_contract.py`.

## 1. Development host

| item | value |
|---|---|
| GPU | GeForce GTX 1050 Ti, Pascal, compute capability 6.1 (sm_61), 4 GB, 6 SMs |
| CPU | Core i7-7700HQ, 4 cores, 8 threads |
| stack | cupy 13.6, numba, CUDA toolkits 11.8, 12.8, 12.9 |

Everything measured in this file was measured on this host. A kernel that wins here can lose on another GPU class (atomic latency, shared-memory budget and register pressure differ between compute capabilities), so every kernel version is kept and the dispatcher chooses per host (section 4).

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
- Pending, to be filled by the profile record: fused-kernel time per backend and `{REAL}`, the KTC sweep result, achieved occupancy, SM and DRAM utilization, and launches per call.

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
- Not read: the helper scripts that parse the reports come from the separate `ncu-report-skill` project (`DongyunZou/ncu-report-skill`), and most of the wiki.

### tensormux `kernel-skills` ([repo](https://github.com/tensormux/kernel-skills))

MIT, 86 stars, 47 commits, default branch `master`. 33 skills in five categories (CUDA, Triton, inference, quantization, patterns). Four were read in full:

- `cuda/choose-launch-configuration`: block size a multiple of 32; start with 128 or 256, 512 when registers are very low, avoid 1024 (two blocks per SM on a 2048-thread SM); occupancy is the minimum of the register limit (65,536 registers per SM divided by registers per block), the shared-memory limit and the hardware block limit (32 resident blocks on SM70 and later); aim at 50% or more, but a compute-bound kernel is fine near 25% and a spill is not worth a higher occupancy; get the real register count from `ptxas -v` (hand estimates can be off by 2x), and validate with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; `gridDim.y` and `gridDim.z` are below 65,535; a launch costs about 5 to 20 us, so under 100 blocks consider fusion or a persistent kernel; guard every global access with `if (idx < N)`; a non-multiple-of-32 block can break warp primitives unless the mask matches the active threads.
- `cuda/write-cuda-reduction-kernel`: warp-shuffle tree with `__shfl_down_sync` and an explicit mask (offsets from 16); one value per warp into shared memory with a `__syncthreads()` between write and read; the second stage uses a mask that matches the number of partials; the two-pass form (per-block partials, then a second kernel) is preferred for reproducibility; floating-point `atomicAdd` is non-deterministic, so initialize the output to the identity from the host and document it; accumulate fp16 and bf16 sums in fp32; carry `(value, index)` through argmax and take the smaller index on ties; test N = 1, 32, 33, 1024, 1025 and a large N against a CPU reference.
- `cuda/debug-cuda-kernel-correctness`: test sizes such as 31, 63, 65, 127, 129 ("powers of two are not representative of all failure modes"); classify the error pattern (uniform scale, wrong rows or columns, last rows only, random few elements means a race, NaN or Inf); initialize outputs to a sentinel because a zeroed buffer can mask a bug; `compute-sanitizer --tool memcheck` is mandatory for random or intermittent failures; isolate one block; disable fast math when hunting a race; correctness must not depend on the block count.
- `patterns/write-kernel-test-plan`: the reference must not share code or algorithm with the kernel; tolerances fp64 reference against fp32 kernel rtol 1e-5 and atol 1e-8, fp16 rtol 1e-2 and atol 1e-3, bf16 rtol 5e-2 and atol 1e-3; shape sweep of 1, 7, 13, 37, 64, 128, 512, 4096, 100, 1000, 4097, primes 1009 and 8191, and tile-size plus or minus one; contiguous, transposed and sliced layouts; adversarial inputs (zeros, ones, large, near underflow, alternating signs, a single NaN); a fixed seed in every test; time with CUDA events or a benchmark timer and not wall-clock alone, exclude the first call, report GB/s or TFLOPS, include one case larger than L2; the kernel must not be more than 10% slower than the baseline; run `compute-sanitizer` memcheck and racecheck in CI, not in the main test loop.
- Not read: the other 29 skills. None of the four covers shared-memory histograms or integer atomics for count tables.

### Hugging Face `cuda-kernels` and `upskill` ([docs](https://huggingface.co/docs/kernels/en/cli-skills), [skill](https://www.skills.sh/huggingface/kernels/cuda-kernels), [blog](https://huggingface.co/blog/upskill))

Read by a low-effort agent in an earlier pass and not re-read here, so treat the details as unverified. `SKILL.md` about 650 lines in 19 sections, installed with `kernel-builder skills add --claude`; it targets H100, A100 and T4 and the diffusers and transformers stack; its build rules (kernel-builder through Nix, `build.toml`) do not apply to a numba and cupy codebase. The `upskill` blog reports that a Claude Opus 4.5 generated CUDA skill gave Opus itself no clear accuracy gain and raised the token use, helped some smaller models (a local GLM-4.7-Flash from 40% to 85% in the sample evaluation), and was tested with simple input and expected-output cases, not kernel speedups.

### What to take

1. **Profile before editing, change one thing, keep an iteration log, keep a reference as the golden.** All the sources agree, and it matches our rules.
2. **The Nsight Compute rule set, metric names and thresholds** of section 6, as the reading guide for any profile.
3. **The three honesty rules about `ncu`:** its duration is not latency, its estimated speedup does not predict the wall, and three strikes are a stop signal. They agree with our `nvprof` rule.
4. **The benchmark protocol of AKO4ALL** (discard the first trial, warm up, median, fresh inputs, mutation sentinel, cold-cache option) and **the test plan of tensormux** (size sweeps around tile boundaries, sentinel-initialized outputs, adversarial inputs, tolerances per dtype, sanitizers in CI).
5. **Do not copy** the Hugging Face build and architecture rules; they solve a different problem.
6. **No surveyed source covers shared-memory histograms or integer atomics for count tables.** Our own measurements in sections 3 and 7 are the guidance for that.
7. **A project-specific skill would be worth more than a generic one.** It would encode the contract of section 2, the `{REAL}` and `{ACC}` placeholders, the KTC registration, the streaming rule and the honest-timing rules of section 6, with `hermite_fe/_als_kernels_gpu.py` as the reference module.

## 9. Checklist for a new GPU module

1. Look for an existing batched GPU entry to reuse and measure it on your own shape before writing a kernel.
2. Write the CPU kernel first (numba `@njit`, parallel where it pays) and keep it as the reference.
3. Write the fused GPU kernel with `{REAL}` and `{ACC}`; compile one module per type.
4. Stream by column blocks with a RAM and a VRAM bound, and an out-of-memory fallback to the CPU kernel.
5. Register with the KTC, with a fallback heuristic and a sweep over the module's own axes.
6. Tests: kernel versions equal; streamed equals unblocked for several block widths; GPU equals CPU for both REAL types; every pair covered once at block boundaries; memory-bound test; a regression test for the behaviour the module exists for.
7. Profile: cProfile on the host path; a GPU profiler with occupancy, throughput, launch statistics and launches per call; record both under `audits/2026-10-22/profiles/<module>.md`.
8. Move the module from `pending` to `compliant` in `tests/test_meta/_fe_gpu_contract.json`.

## 10. Open items

- Record the fused pair kernel's profile (nvprof metrics with an elevated shell, or nsys) and the KTC sweep, then promote `_fe_sufficient_summary_pairs` to compliant.
- Walk the 90 pending FE modules against points 2 to 5 of the contract, starting with those that have the most expensive kernels.
- Decide whether to write the project-specific CUDA skill described above.
- Not yet read from the surveyed sources: the other 29 tensormux skills (the 4 that apply were read), the `ncu-report-skill` helper project behind the atrex report parsers, most of the atrex `gpu-wiki` records, and the Hugging Face `cuda-kernels` `SKILL.md` itself (only a low-effort agent summary exists).
- Spot-check the quoted thresholds against the URLs in section 8 before relying on one in a decision; the extractions were made by a small model.
