# _null_max_perm_gpu (family-wise null of the binned-numeric-aggregate gate)

100000 x 100 fit: `_binagg_null_calibrated_floor` scores 1016 candidates x 100000 rows under 39 target permutations, one fused bin + histogram + MI launch
(`binned_mi_from_values_gpu`) per permutation, each re-reading and re-binning every value: 1.04 s in the fit (0.91 s standalone).

Attempts (standalone, same block, floor bit-equal in every case):
- one launch for a block of permutations, each value binned once and added to every permutation's shared table (39 atomics per value): 1.90 s, slower: shared
  atomics on a 20-cell table per permutation serialise.
- binary target, bitset: a warp bins 32 consecutive rows and keeps a ballot mask per bin; lane l counts permutation l's label-1 rows per bin as popc(mask & label word);
  bin totals once; register counts flushed once per lane: 0.354 s (kernel 0.224 s, edges 0.037 s). 2 permutations per lane instead of 4 (local 768 -> 512 bytes): 0.342 s.

Kept: bitset kernel for two classes (2.6x), the atomic kernel for more classes, the per-permutation loop as the fallback (`MLFRAME_BINAGG_NULL_BATCHED=0` forces it).
Block width from `fixed_shared_block_size` (64 on the GTX 1050 Ti: 40 registers, 512 bytes local).
