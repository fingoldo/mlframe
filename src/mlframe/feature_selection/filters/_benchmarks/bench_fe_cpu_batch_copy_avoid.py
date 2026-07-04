"""P-2 bench: does cpu_fe_batch_mi force a float64 copy of an already-f64/C-contiguous candidate matrix?

Verdict: NO. ``np.ascontiguousarray(arr, dtype=np.float64)`` already returns the INPUT UNCHANGED (same
object, shared memory) when the array is already float64 and C-contiguous -- the proposed
``arr if arr.dtype==float64 and arr.flags.c_contiguous else np.ascontiguousarray(...)`` branch is exactly
what numpy does internally. So the copy the item claims to avoid does not exist; the site is already optimal.

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_fe_cpu_batch_copy_avoid
"""
from __future__ import annotations

import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    a_c64 = np.ascontiguousarray(rng.standard_normal((100000, 200)), dtype=np.float64)
    b = np.ascontiguousarray(a_c64, dtype=np.float64)
    print("f64 C-contig -> same object:", a_c64 is b, "| shares memory:", np.shares_memory(a_c64, b))

    a_f = np.asfortranarray(a_c64)
    c = np.ascontiguousarray(a_f, dtype=np.float64)
    print("f64 F-contig -> copies (expected, non-contiguous):", not np.shares_memory(a_f, c))

    a32 = a_c64.astype(np.float32)
    d = np.ascontiguousarray(a32, dtype=np.float64)
    print("f32 -> copies (expected, dtype cast):", not np.shares_memory(a32, d))

    print("\nCONCLUSION: ascontiguousarray(X, float64) is already a no-op on the common already-f64/C-contig")
    print("candidate matrix -> P-2's dtype/flags branch is redundant. REJECT (premise false).")


if __name__ == "__main__":
    main()
