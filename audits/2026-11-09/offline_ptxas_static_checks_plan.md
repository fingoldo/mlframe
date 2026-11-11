# Offline static checks for CUDA kernels (NVRTC → PTX → ptxas), plan

Status: DRAFT for critique (GPU11-32). Owner: GPU kernel audit.

## Goal

Find defects in every CUDA kernel source in `src/mlframe` without a GPU, for every supported compute capability,
and gate the results the same way as the existing kernel contract (ratchet tests + reasoned exemptions, no silent passes).

## Current state

- `tests/test_meta/test_cuda_kernel_register_pressure.py` compiles only the registered kernels (2 today) and only on the
  local device. Offline per-arch coverage exists only for `lof_scores` (sm_52, 61, 70, 75, 86, 89, 90, 120; four shapes; two dtypes;
  64 cases; spill-free in every case).
- Kernel sources are Python string constants. Extraction is AST-based (`extract_cuda_sources.py`, session scratchpad, not in repo yet).

## Checks, in priority order

1. **Compile failures per architecture.** NVRTC errors on a compute capability the project claims to support
   (e.g. `atomicAdd(double*)` needs sm_60+). Blocks: any kernel that fails on a claimed arch.
2. **Spills and local memory.** `ptxas -v` stack frame and spill bytes; `ld.local`/`st.local` count in PTX (catches
   dynamically indexed thread-private arrays even when they do not spill).
3. **Static shared memory.** Static `__shared__` bytes from the ptxas report, checked against 48 KB. Dynamic shared memory
   is passed at launch and is checked at the call site, not here.
4. **Register count versus occupancy.** Registers per thread per arch, with resident-thread estimate from the
   per-arch register file. Flags kernels limited by registers (CUDA.md section 3).
5. **FP64 instruction share in PTX.** Static count of `.f64` instructions versus `.f32`. Flags double-precision kernels
   where float would do, on consumer parts (1/32 to 1/64 rate, CUDA.md section 13).
6. **ptxas warnings.** Ignored launch bounds, unsupported options, and similar messages.

## Scope and cost

- Kernel count: about 85 extracted sources; each has placeholders `{REAL}`, `{ACC}`, `%THREADS%`, `%BLOCK%`
  substituted as in the extraction script.
- Shapes: each kernel with its module-level defaults; kernels that take `k`/`p`-style compile-time parameters need
  their own shape list (lof_scores already has one).
- Architectures: 8 (52, 61, 70, 75, 86, 89, 90, 120). Full run is roughly 85 x 8 x dtypes ptxas invocations, so it runs in
  CI. Local default: sm_61 and sm_120.
- Toolchain caveat: the NVRTC shipped with cupy can emit a PTX ISA newer than the toolkit's ptxas
  (8.8 against 8.7 here). The PTX version line is rewritten for the check only; the instructions used are unchanged.

## Dispositions (to be filled per finding)

- FIXED: the kernel is changed, the offline test covers it, and the tracker records the commit.
- EXEMPT: a reasoned exemption in the registry, same shape as `_SPILL_EXEMPT`, with the number that justifies it.
- TOOLCHAIN: the check cannot run on this arch or ptxas version; recorded, not silent.

## Open questions for critique

- Is ptxas register count a good proxy for occupancy on newer architectures, or does the hardware register file size need its own table?
- Should check 5 use the NVRTC PTX or the cubin, and does it survive the version rewrite?
- Is 48 KB the right static-shared cap for every claimed arch?

## Critique (independent review, 2026-11-11), and what changes

1. Static shared memory: ptxas already fails above the static limit (checked: a 56000-byte static array gives
   `uses too much shared data`). Fold it into the compile-failure check; do not parse it from the report.
2. `re.search` reads only the first function in the report. Use `findall` and require one resource block per `.entry`.
3. ptxas warnings are not gated; the test checks the return code only. Count `warning` lines per kernel and gate them
   against an explicit allowlist.
4. The `.version 8.8 -> 8.7` rewrite is a PTX-validity check under ptxas 12.8, not a statement about what production
   runs (production uses cupy's NVRTC and the driver JIT). State this, or pin the NVRTC used for the check.
5. Occupancy: the register file is 65536 per SM on all listed archs, but threads per SM differ (1024 on sm_75, 1536 on
   sm_86/89/120, 2048 on sm_52/61/70/90) and so do blocks per SM. Needs a per-arch table, not a single number. The
   block limits in CUDA.md section 3 are unverified and must be checked against the CUDA programming guide before use.
6. FP64 share: count arithmetic opcodes only, weight by loop depth, and report rather than gate. Loads, stores and moves
   of f64 are not compute.
7. Placeholders beyond `{REAL}/{ACC}/%THREADS%/%BLOCK%`: also `{K}`, `{P}`, `{N}`, `{EDGE}`, `__BOUNDS__`, `__REDUCE__`
   (`_pairwise_modular_rank_gpu.py`). `__launch_bounds__(%THREADS%)` sets the register cap, so a made-up literal makes the
   register number meaningless. Take production values from call sites, and grep the substituted source for leftover
   `{UPPER}`, `%UPPER%`, `__UPPER__`.
8. nvcc 12.8 warns that offline compilation below sm_75 will be removed in a future release. Add a TOOLCHAIN disposition
   for sm_52 and sm_61 as well as for the ptxas version.
9. Scope: 109 files contain `__global__`, the plan says 85 kernels. DECIDED (user, 2026-11-11): `_benchmarks/` and
   `fe_operator_factory/kernel_prototypes/` are OUT of scope; production modules are IN. Recorded here so it is not re-litigated.

## Additions

- Production compile options: mirror the `RawKernel`/`RawModule` options used at the call sites, not the test's `-std=c++14`.
- Leftover-placeholder scan after substitution (item 7).
- Launch bounds honoured: for each `__launch_bounds__(T, B)`, the ptxas register count must fit 65536/(T*B) after
  allocation granularity, on each arch. Verifies the register cap is not silently ignored (e.g. the 32-register claim in
  `_pairwise_modular_rank_gpu.py`'s v3 comment at (128, 16)).
- Arch-gated intrinsics: grep for double `atomicAdd`, `__dp4a`, wmma, `cp.async`, `__reduce_*_sync`; each kernel using one
  must be absent from the lower arches or exempt.
- Per-arch failure table in the report, so a kernel that fails only on sm_52 is an exemption candidate, not a silent skip.
- Coverage diff between files containing `__global__` and kernels extracted and checked, to catch silent drops.
