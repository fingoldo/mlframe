# _rff_gram_energy_gpu (mixed-partial energies of the RFF surrogate of the gradient seeder)

Module: `src/mlframe/feature_selection/filters/_rff_gram_energy_gpu.py`; caller: `_gradient_interaction_seeder.rank_gradient_interaction_pairs` (the permutation null of the max mixed-partial energy). Measured 2026-11-25 on the development
host (GTX 1050 Ti, 6 SMs; i7-7700HQ), GPU 62 C at the start, nothing else running. Tools: wall timers, nvprof 12.8 (`nvprof_shared_design_null.py`, cupy profiler window). The kernels of this module are cuBLAS GEMMs and cupy
elementwise / reduction launches (no hand-written kernel), so there is no launch geometry or register budget of its own to tune; what the record fixes is the launch count, the device time and the host share.

## What changed

The null of the seeder fitted 12 permutation surrogates, each on a random-feature design of its own (`rff_gram_null_maxima_device` stacks 12 designs: 12 cosine matrices, 12 `D x D` Grams). `rff_gram_shared_design_maxima_device` is the
version for ONE design shared by the observed surrogate and its 12 permutation surrogates: the cosine features and the Gram are formed once, the ridge systems are factorised once on the host
(`fit_ridge_closed_form_multi`, one Cholesky for the 12 targets), and the energies of the 12 solutions for a chunk of pairs are one batched product. The stacked version is kept (`shared_null_design=False`).

## Launches and time of the null (12 surrogates, 2000 rows, 30 columns = 435 pairs, 400 features; one warm call inside the profiler window)

| route | wall | GPU time | kernels | per kernel |
|---|---:|---:|---:|---|
| shared design (new default) | 62 ms | 1.7 ms | 4 (+ 1 reduction) | batched `gram @ coef` 0.99 ms, Gram `mcos' mcos` 0.55 ms, `cos` 0.07 ms, coefficient product 0.06 ms |
| 12 designs stacked | 481 ms | 8.5 ms | 4 (+ 1 reduction) | batched GEMMs 5.9 ms and 1.2 ms, `cos` 1.2 ms |

The wall is mostly host: 12 per-surrogate ridge fits of a 400-column design (the cProfile of one seeder call: 13 `fit_ridge_closed_form` 0.24 s of 0.42 s) against one fit with 12 right-hand sides. The device part is 5 launches either way
(the surrogates are batched in both), so the launch count is not what the change buys; the host factorisations and the 12 Gram builds are.

## Whole seeder call (`grad_ab.py`, 2000 rows, 30 columns, 60 repeats per fixture)

| fixture | per-surrogate designs | shared design |
|---|---:|---:|
| additive target | 467 ms | 186 ms |
| + interaction 0.3 / 0.6 / 1.0 | 395 / 390 / 391 ms | 189 / 188 / 188 ms |

## Quality (the null changes: permutation distribution for FIXED features instead of a mixture over designs)

- detection of the planted `x2 * x3` interaction: 60 of 60 at strength 0.3, 0.6 and 1.0, with both nulls; floor / observed maximum 0.127 / 0.079 / 0.069 (per-surrogate) against 0.132 / 0.080 / 0.070 (shared);
- false proposals on an additive target (`x0 + 0.7 sin x1 + 0.5 x2^2`, 300 repeats, nominal level 1 percent): 3 of 300 (1.0 percent) per-surrogate, 4 of 300 (1.3 percent) shared;
- the shared-design maxima equal the largest energy of each surrogate scored alone on the same design (host to 1e-9, device to 1e-3 relative): `tests/feature_selection/gpu/test_gradient_seeder_shared_null_design.py`.

## Not done

- The observed surrogate's Gram is formed again by `_rff_analytic_mixed_partial_energy_gram` (once more, 0.5 ms on the device): passing it into the null would save one GEMM.
- At 1000 candidate columns the seeder is routed off (`max_p = 200`), so the 100000 x 1000 benchmark does not run it.
