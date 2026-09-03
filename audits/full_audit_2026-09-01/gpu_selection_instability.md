# GPU/CPU selection equivalence: a pre-existing instability, measured but not closed

Not one of the 202 audit findings. Surfaced while validating the P1 batch, investigated to the point where the
cause is understood and the next step is concrete, and left open rather than closed with a change that did not
replicate.

## What fails

`tests/feature_selection/gpu/test_gpu_cpu_mi_selection_equivalence.py::test_mrmr_gpu_cpu_selection_identical`
asserts that `MRMR(use_gpu=True)` and `MRMR(use_gpu=False)` select the identical feature set. It fails on a
borderline candidate, and WHICH parametrisation fails depends on run conditions.

## What was established

**It is pre-existing.** Run on the pre-audit baseline `bc2de5068` (before any fix in this campaign), the
`clf_binary` case fails with byte-identical output:

```
CPU-only (2): ['a', 'div(sin(a),exp(b))']
GPU-only (3): ['a', 'add(qubed(c),rint(e))', 'div(sin(a),exp(b))']
```

**The 63 failures in the parallel GPU run are contention, not logic.** `pytest tests/feature_selection/gpu -n 4`
produced 63 failures, 39 of them `CUDARuntimeError: cudaErrorIllegalAddress` -- a corrupted CUDA context, after
which everything downstream in that worker fails. Re-running those same 63 serially: **62 passed**, leaving only
`test_mrmr_gpu_cpu_selection_identical[clf_binary]`. Four xdist workers against one 4GB card is the documented
environment limitation, not a defect in the code under test.

**A seeded fit IS reproducible across processes.** Three separate processes running the same
`MRMR(random_seed=42, use_gpu=True)` fit returned identical selections, matching the CPU fit. So the GPU
permutation stream is not the unseeded free-running entropy its `base_seed=None` default suggests -- the confirm
gate derives a per-candidate seed (`_confirm_predictor.py:438`) and threads it into both backends.

**The two backends draw different permutation streams by design.** `mi_direct_gpu`'s own docstring states the
contract: "each path is internally reproducible under a seed", NOT cross-backend bit-parity -- CuPy XORWOW
against the CPU LCG scheme. For a candidate whose MI sits at the gate, the two streams can land on opposite
sides of it.

**The disagreeing candidate is genuinely borderline.** `add(qubed(c),rint(e))` scores MI 0.0021 on the
`clf_binary` fixture. The combo scoring itself is not the divergence: `_pair_combo_mi_njit`,
`_pair_combo_mi_njit_parallel` and `_pair_combo_mi_cupy` agree on it to 4.4e-16.

**Raising the permutation budget does not fix it, and the first measurement that suggested it does not
replicate.** Sequentially in one process at budgets 64 / 256 / 512, `clf_binary` went CPU-drops / CPU-drops /
both-keep, which looked like "a precise enough null settles both paths". In a FRESH process at 512, `reg_ratio`
mismatched (GPU keeps `e`) and `clf_binary` mismatched the OTHER way (CPU keeps the combo). A budget change made
on the first measurement was written and then reverted: it fixed one parametrisation and broke another.

## What is left to do

The remaining variation is order-dependent WITHIN a process -- several fits in sequence change the outcome for a
borderline candidate while separate processes are stable. The two mechanisms worth checking first, in order:

1. **The kernel-tuning cache warming mid-run.** `_MI_PRANGE_PARALLELISM_SPEC.choose(...)` and the other
   KTC-gated dispatchers pick a variant from a measured sweep; a sweep that runs during the first fit changes
   which variant later fits use. The mi_prange variants are documented bit-identical, but the other GPU-capable
   dispatchers in the FE path have not been checked for the same guarantee.
2. **CuPy memory-pool pressure across sequential fits**, which can change which candidates take the resident
   path versus the host fallback.

The test itself may also be asserting more than the system promises. If the two backends cannot agree on a
borderline candidate by construction, the honest assertion is that the GPU path does not SYSTEMATICALLY
over-reject -- which is what the historical CUDA-reducer stride bug did, and what this test was written to
catch -- rather than exact set equality. Deciding that needs the two mechanisms above ruled out first, so that
the reframing is justified by evidence rather than by the test being inconvenient.
