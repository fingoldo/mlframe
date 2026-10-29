# fe_hybrid_orth_triplet_enable: profile inside the all-flags fit (n=20000)

Measured 2026-10-28 with `D:/Temp/trip_allflags.py` (all FE flags on, warm fit on other data, default device mode). Family cumulative 1.60 s profiled; CPU family wall 2.01 s in `fe_family_walls_gpu_vs_cpu_n20000.md`.

| part | cum s | share of family | note |
|---|---:|---:|---|
| `score_triplet_cross_basis_by_cmi` | 1.29 | 81% | conditional MI of every triplet with y given its strongest leg, against a permutation null |
| `_triple_mi_kernel` | 0.10 | 6% | njit |
| `_quantile_codes` (57 calls) | 0.12 | 8% | binning of raw and engineered columns |
| `_enumerate_triplet_candidates` | 0.18 | 11% | candidate generation |

Reading: the scorer's permutation null is sized by the Bonferroni correction over the candidate count m: `n_perm = max(199, ceil(m / (1 - quantile)))`, so n_perm is about 100 m. The null is `stratified_cmi_null` (njit, prange over permutations, O(n_perm x n) work), already parallel. Its cost therefore grows with m, and the level is a statistical requirement (the best of m noise triplets clears a 0.99 null by chance without it), so a smaller n_perm is not a free change. The per-candidate permutation sources are also keyed by candidate name (`crc32`), so sharing one permutation set across candidates would change every null and the selection. No cheap lever found; the remaining lever is a smaller candidate count m (earlier pruning), which is a selection change and needs its own equivalence study. No code change in this entry.
