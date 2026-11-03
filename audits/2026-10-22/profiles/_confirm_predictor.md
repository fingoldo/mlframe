
## 2026-11-03: Fleuret confidence on compacted columns

Whole-fit py-spy (100000 x 100, all threads, 50 Hz): 12.6 s of the main thread in `get_fleuret_criteria_confidence` (serial path of
`fleuret_conditional_recheck`, screen confirmation): 28 calls of 99 permutations on a (30000, 194) int32 matrix, 0.3 s each. Every permutation computes H(XZ),
H(YZ), H(Z), H(XYZ) for each selected Z from `data_copy[:, col]`, a read strided over a whole 194-column row per value. `_compact_confidence_inputs` gathers the
few columns the call reads (x, y, selected variables) into a contiguous matrix with renumbered indices and private caches (the confidence kernel never touches
the conditional-MI cache and reads the entropy cache only for H(Z) of an unshuffled Z, seeded under the new numbers). Captured call: 0.25 -> 0.058 s (4.3x);
failure counts equal permutation by permutation (thresholds through the null: 95/99, 98/99, 44/99, 29/99 both ways).
