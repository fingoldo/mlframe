"""Bench/proof: the shared resample-index matrix RAM ceiling, and whether a chunked-block stream is a cheap cap.

``_build_resample_indices`` materializes a single ``(n_bootstrap, n)`` int64 matrix ( ``8 * n_bootstrap * n`` bytes:
~0.8 GB @ n=200k, ~2 GB @ n=500k for the 1000-resample default). It is built ONCE in ``pick_best_calibrator`` and
SHARED (read-only) across every candidate -- it does NOT compound per candidate.

This bench shows two things:
  1. The ceiling is the SINGLE shared matrix; the per-candidate path (``_bootstrap_ece_with_indices``) already streams
     it row-by-row (``idx = idx_matrix[b]``) with no per-candidate copy.
  2. A chunked-block alternative (build ``block`` resample rows at a time, accumulate per-candidate ECE samples,
     discard the block) caps peak index RAM to ``8 * block * n`` bytes and is BIT-IDENTICAL when the RNG draw order
     is preserved -- but it requires holding ALL candidates' calibrated-prob arrays simultaneously and re-walking the
     candidate set per block. We verify bit-identity and measure the wall/RAM trade-off.

Verdict driver: the calibration OOF set is the calibration sample (typically << the 100 GB feature frame), the
matrix is a one-time shared alloc, and chunking trades a higher candidate-loop complexity + per-block re-walk for a
RAM cap that only matters at pathological n. Document the ceiling; chunk only if a caller actually bootstraps an OOF
set that large.

Run:
    python src/mlframe/calibration/_benchmarks/bench_resample_index_ram_chunked.py
"""
from __future__ import annotations

import numpy as np


def _build_full(n: int, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    for b in range(n_bootstrap):
        out[b] = rng.integers(0, n, size=n, dtype=np.int64)
    return out


def _ece_like(y: np.ndarray, p: np.ndarray, idx: np.ndarray, n_bins: int = 10) -> float:
    # tiny ECE stand-in: |mean(p)-mean(y)| over the resample -- order-independent, enough to prove bit-identity.
    yy = y[idx]
    pp = p[idx]
    return float(abs(pp.mean() - yy.mean()))


def _samples_full(y, p, n: int, n_bootstrap: int, seed: int) -> np.ndarray:
    mat = _build_full(n, n_bootstrap, seed)  # peak: 8*n_bootstrap*n bytes
    return np.array([_ece_like(y, p, mat[b]) for b in range(n_bootstrap)])


def _samples_chunked(y, p, n: int, n_bootstrap: int, seed: int, block: int) -> np.ndarray:
    """Bit-identical to _samples_full: SAME single-rng draw order, but only ``block`` rows resident at once."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        bsz = min(block, n_bootstrap - done)
        chunk = np.empty((bsz, n), dtype=np.int64)  # peak: 8*block*n bytes
        for i in range(bsz):
            chunk[i] = rng.integers(0, n, size=n, dtype=np.int64)
        for i in range(bsz):
            out[done + i] = _ece_like(y, p, chunk[i])
        done += bsz
        del chunk
    return out


def main() -> None:
    import time
    rng = np.random.default_rng(0)
    n = 50_000
    n_bootstrap = 1000
    y = (rng.random(n) < 0.3).astype(np.int64)
    p = np.clip(rng.random(n), 0.0, 1.0)

    full_bytes = 8 * n_bootstrap * n
    block = 64
    chunk_bytes = 8 * block * n
    print(f"n={n}  n_bootstrap={n_bootstrap}")
    print(f"  full matrix peak index RAM  : {full_bytes/1e6:8.1f} MB  (8*n_bootstrap*n)")
    print(f"  chunked (block={block}) peak : {chunk_bytes/1e6:8.1f} MB  ({full_bytes/chunk_bytes:.1f}x smaller)")

    t0 = time.perf_counter(); s_full = _samples_full(y, p, n, n_bootstrap, 42); t_full = time.perf_counter() - t0
    t0 = time.perf_counter(); s_chunk = _samples_chunked(y, p, n, n_bootstrap, 42, block); t_chunk = time.perf_counter() - t0

    identical = bool(np.array_equal(s_full, s_chunk))
    print(f"  wall full={t_full*1e3:.1f} ms   chunked={t_chunk*1e3:.1f} ms")
    print(f"  per-resample ECE samples BIT-IDENTICAL (full vs chunked, same rng order): {identical}")
    assert identical, "chunked stream must preserve the single-rng draw order -> bit-identical samples"  # nosec B101 - internal invariant check in src/mlframe/calibration/_benchmarks, not reachable with untrusted input
    print(
        "\nVerdict (DOC -- document the ceiling, do NOT chunk by default): the matrix is a ONE-TIME shared (read-only) "
        "alloc, not a per-candidate copy, and the per-candidate path already streams it row-by-row. Chunking caps "
        "peak index RAM (e.g. ~15x at block=64) and is bit-identical, BUT in pick_best_calibrator it would force "
        "holding ALL candidates' calibrated-prob arrays + a per-block re-walk of the candidate set (restructuring the "
        "candidate loop) -- complexity unjustified when the calibration OOF set is small (<< the 100 GB feature frame) "
        "and the alloc does not compound. The docstring ceiling note (8*n_bootstrap*n bytes; lower n_bootstrap if it "
        "dominates) is the right disposition; revisit chunking only if a caller bootstraps an OOF set large enough "
        "that this single matrix dominates RAM."
    )


if __name__ == "__main__":
    main()
