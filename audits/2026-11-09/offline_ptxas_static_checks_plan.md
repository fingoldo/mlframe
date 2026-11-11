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
