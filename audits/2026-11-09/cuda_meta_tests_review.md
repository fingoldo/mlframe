# Review of the static CUDA meta-tests (2026-11-13)

Two read-only reviewer agents were given the offline NVRTC -> ptxas suite and the static launch / dtype / coalescing / register gates, with the recent kernels (`strata_xy_entropy`, `strata_xy_entropy_rows`,
`family_null_bits_v3`, the u8 joint histogram) as ground truth. What follows is their findings, each with an ID, the evidence they cite (verified by reading or by a non-GPU probe unless marked inferred) and the
disposition. Dispositions: **DONE** (implemented and tested, commit named in the tracker), **OPEN** (accepted, with the design and the reason it is not done yet), **REJECTED** (with the reason). The tracker rows are MT11-*.

## A. Offline NVRTC -> ptxas suite (`test_cuda_kernels_offline_per_arch.py`, `_offline_kernel_sources.py`, `_offline_kernel_report.py`)

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| OP-1 | High | `_family_maxt_cuda_bits.py::_SOURCE_V3` had no recipe; NVRTC failed on `PB`, `NY1`, `CAP` (verified: run on sm_61) | **DONE**: recipe over `(NY1, PB, CAP)` in {(1,4,8), (2,2,8), (1,2,2), (3,2,5)} x float/double, commit `79b466656` |
| OP-2 | High | Kernels passed inline as `cp.RawKernel("...", name)` are never extracted (`info_theory/_cmi_cuda.py` `cmi_joint_hist_cuda`, `_hermite_fe_mi.py` `legval_kernel`, `chebval_kernel`, `lagval_kernel`); the file-level coverage test passes because the same files have another extracted kernel | see MT11-03 |
| OP-3 | Med | Only the first kernel-bearing literal and the first assignment of a name are taken (`literals[0]`, `found.setdefault`); a source built as `A + B` or assigned twice loses a kernel; 6 kernel names in 3 files are never compiled (verified by a name-level diff) | see MT11-03 |
| OP-4 | Med | Only `.entry` names are matched against ptxas "Function properties" blocks; a `__noinline__` helper that spills is not checked | see MT11-04 |
| OP-5 | Med | A macro with a default (`#ifndef X #define X big`) is never swept; one `-D` value per recipe | OPEN: static scan for defaults in kernels that production always passes by `-D` (design in MT11-07) |
| OP-6 | Med | Dynamic shared memory (`extern __shared__`, 61 sites) is never checked against the per-arch limit; `ARCH_LIMITS` shared figures feed only the informational report | OPEN: needs a table of the dynamic request per kernel at the production shapes (design in MT11-07) |
| OP-7 | Med | No register budget: only `<= 255`; a kernel going from 32 to 96 registers passes if it does not spill | REJECTED as a ratchet (user rule: no seeded baselines); the occupancy report stays informational, `__launch_bounds__` already gates kernels that declare a cap |
| OP-8 | High | NVRTC warnings of a SUCCESSFUL build are never inspected (`getProgramLog` only on failure) | **DONE**: log joined to the report, same allowlist; found and fixed three dead variables (see MT11-02), proof test in the suite |
| OP-9 | Low | `_PLACEHOLDER_RE` matches `{UPPER}`, `%UPPER%`, `__UPPER__` only; `@VAR@`, `OUTTYPE`, lower-case `{real}` slip through unless NVRTC errors | OPEN, low: NVRTC does error on every such leftover in the current sources (a leftover identifier is undefined) |
| OP-10 | Low | The `.version 8.7` rewrite accepts an instruction too new for the toolkit's ptxas (stated in the module docstring) | REJECTED: documented trade-off of checking the PTX under the installed ptxas |
| OP-11 | Low | `launch_bounds_violations` does not see host launch sites (a block above `maxntid`) | covered by the launch-geometry gate for the sizes it can resolve; see GT-1 |
| OP-12 | Med | A toolchain without an arch makes `compileProgram` raise and the test FAILS, contradicting the docstring that says it skips; a missing ptxas or `cupy_backends` silently skips the whole suite in a CI image without the toolkit; the report test does not assert that the per-arch test ran | see MT11-05 |
| OP-13 | Med | `ElementwiseKernel` / `ReductionKernel` bodies (about 30 files) and numba `@cuda.jit` kernels have no NVRTC source (inferred from grep counts) | OPEN: ElementwiseKernel bodies compile through a template NVRTC wraps; needs a wrapper recipe (design in MT11-07) |
| OP-14 | Low | Every recipe fixes `{ACC}` to double; float32 is the project default for `{REAL}` | REJECTED: `{ACC}` is float64 by contract (CUDA.md section 3); the `{REAL}` float variants are compiled |
| OP-15 | Low | `_benchmarks/` and `kernel_prototypes/` (12 files with `__global__`) are excluded | REJECTED: benchmark kernels are not production; excluded by decision |

## B. Launch geometry, pointer dtype, coalescing, register pressure, FE contract, tiny-cupy gates

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| GT-1 | High | `py_ci_shared.cuda_launch_geometry._is_launch` needs a tuple grid and a tuple/list args: two literal-256 launches in `_fe_batched_mi_cmi.py` (grid a variable) are invisible; also `args=(...)` bound to a name, `block=` keyword, `B=(256,)` bound name, `(2*128,)`, `int(256)`, numba `kern[blocks, 256](x)`; the `# launch-ok:` marker accepts any text (verified on synthetic launches) | **DONE** in py-ci-shared (commit `c3a6894`): the detector reports the loose shapes; 22 live launches in this package found and fixed, MT11-06 |
| GT-2 | High | The uncoalesced detector matches only `loopvar * name` inside one bracket of a `for` loop: hoisted offsets, kernels without a `for`, data-dependent strides (`for (r = st[s]; r < st[s+1]; ++r) ... X[r]`, i.e. `strata_xy_entropy`, measured gld efficiency 6.7%), shifts, helper strides and `while` loops pass (verified by running `uncoalesced_kernels` on snippets and on the real file) | see MT11-09 |
| GT-3 | High | No grid-fill gate: the 14-block histogram grid (occupancy 0.25) and the low-parallelism launches were invisible | see MT11-01 |
| GT-4 | Med | Pointer-dtype gate checks only `double*`; a `{REAL}` anywhere in the parameters exempts the whole kernel; `__global__ __launch_bounds__(..) void` is not matched by `_KERNEL_RE` | OPEN: design in MT11-07 |
| GT-5 | Med | The register-pressure registry covers 6 of about 205 `__global__` kernels and compiles one shape each (`kernel_v3` builds only `ny1=1, pb=4, cap=8`); the test skips without a device | see MT11-10 |
| GT-6 | Med | `fe_modules()` filters by name pattern: `_cmi_null_strata_gpu` and `_family_maxt_cuda_bits` are in no list, even as pending; `contract_violations` accepts keywords ("occupancy", "cProfile") anywhere in the record with no numbers and no tie to the kernel names | OPEN: design in MT11-07 |
| GT-7 | Low | Tiny-cupy gate: single calls in a loop, operator expressions and attribute receivers are not counted (its own known-limitations list) | OPEN, low |
| MG-1 | High | Grid-fill model | = GT-3 |
| MG-2 | High | A measured coalescing / occupancy smoke test under `-m gpu` | OPEN: needs the nvprof or CUPTI path in a test; MT11-09 gives the static part |
| MG-3 | Med | Kernel registry completeness: every `(cp, real)` factory registered or exempt, every profile record names its kernels | = GT-5 / MT11-10 |
| MG-4 | Med | Dynamic shared memory against the device limit at the shape grid | = OP-6 |
| MG-5 | Med | `double` math (`log`, `exp`, `pow`) in a hot loop without a marker (the per-cell double `log` cost 10x the count in `strata_xy_entropy`) | OPEN: design in MT11-07 |
| MG-6 | Low-Med | Global `atomicAdd` inside a loop on a few cells without a marker | OPEN: design in MT11-07 |
| MG-7 | Low | Profile record with a numeric occupancy row, naming every kernel of its module | OPEN: design in MT11-07 |

## Order of work

MT11-02 (NVRTC warnings and the dead variables they found), MT11-01 (grid-fill gate), MT11-03 (name-level coverage), MT11-04 (helper spills), MT11-05 (hard-fail switch and the ran-count),
MT11-10 (registry completeness and shape grids), MT11-06 and MT11-09 (launch and stride shapes the gates miss), then the OPEN designs in MT11-07 / MT11-08.


## Where these tests live (instruction of 2026-11-13)

The generic part of every CUDA meta-test belongs in py-ci-shared (the central store the other gates use), with the package-specific data (recipes, grid models, exemptions) passed in by the consumer. The launch-geometry
detector (GT-1) is the first one moved; MT11-11 in the tracker lists the rest in order.
