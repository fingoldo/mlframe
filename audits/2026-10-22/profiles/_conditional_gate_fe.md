
## 2026-11-03: resident operands keyed on the caller's array

Stage sample (100000 x 100): 6 of 17 samples in `_fe_resident_operands.resident_operand` (the content-hash line), from `gate_grid_mi_resident` and
`best_existing_op_mi_resident`. They pass float32 host columns with `dtype=float64`: the conversion made a new array per call, so the identity memo of
`_content_hash_memoized` never hit and every call copied and hashed 100000 values. The key is now the content of the caller's array plus its dtype and the
target dtype (the conversion is a pure function of those); the copy and upload happen only on a miss. After: 0 samples in `resident_operand`, about 0.6 s of
the stage. Side effect: `test_pair_search_residency_codes_resident_into_mi_gate` (an xfail for a host-code re-upload) now passes.
