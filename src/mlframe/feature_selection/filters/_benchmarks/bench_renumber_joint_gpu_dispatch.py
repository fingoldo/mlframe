"""A/B: per-call GPU dispatch for ``_renumber_joint`` at the wellbore-100k CMI-greedy call shapes.

Context: an isolated wellbore-100k MRMR.fit cProfile flagged ``_renumber_joint`` as 51.9s tottime / 6766
calls. The obvious question is whether those calls should route through the already-built device twin
(``_renumber_joint_gpu``) instead of the host njit path. This bench measures the REAL cost of doing so:
H2D upload of the conditioning columns (they are host arrays at these call sites -- see
``_fe_raw_redundancy_helpers.py``'s ``raw_retains_signal_given_children``, which only takes the device
join when the caller ALREADY has resident twins; the plain ``_renumber_joint`` calls counted by cProfile
are precisely the cases where no resident twin exists) + the device join + a single scalar D2H (the
occupied cardinality), against the host njit combine-factorize walk, at n=100k rows and 2/3/4 conditioning
columns (the shapes seen at ``_fe_raw_redundancy_drop.py``'s sibling-conditioning loop and
``_orthogonal_three_gate_mi_fe.py``'s support build -- the two heaviest non-GPU-wrapped call sites).

Run: python bench_renumber_joint_gpu_dispatch.py
"""
import time

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _renumber_joint, _renumber_joint_gpu

N = 100_000
REPS = 30
RNG = np.random.default_rng(0)


def _make_cols(n, k, nbins=12):
    return [RNG.integers(0, nbins, size=n).astype(np.int64) for _ in range(k)]


def _bench_host(cols):
    # warm njit
    _renumber_joint(*cols)
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        _renumber_joint(*cols)
        best = min(best, time.perf_counter() - t0)
    return best


def _bench_gpu_fresh_upload(cols):
    """Per-call cost INCLUDING H2D of every column (the realistic case: host arrays, no residency)."""
    import cupy as cp

    def _call():
        dev_cols = [cp.asarray(c) for c in cols]
        joint_dev, mult = _renumber_joint_gpu(*dev_cols)
        k = int(mult)  # forces the single D2H sync
        return k

    _call()  # warm cupy kernel cache
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(REPS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _call()
        best = min(best, time.perf_counter() - t0)
    return best


def _bench_gpu_already_resident(cols):
    """Isolated device-only cost (columns already resident, the residency-hit case)."""
    import cupy as cp

    dev_cols = [cp.asarray(c) for c in cols]
    cp.cuda.Stream.null.synchronize()

    def _call():
        joint_dev, mult = _renumber_joint_gpu(*dev_cols)
        return int(mult)

    _call()
    cp.cuda.Stream.null.synchronize()
    best = float("inf")
    for _ in range(REPS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _call()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    try:
        import cupy as cp  # noqa: F401
        has_gpu = True
    except Exception as e:
        has_gpu = False
        print(f"cupy/CUDA unavailable ({e}); host-only numbers below.")

    print(f"{'k_cols':>7} {'host_ms':>10} {'gpu_fresh_ms':>13} {'gpu_resident_ms':>16} {'fresh_speedup':>14} {'resident_speedup':>17}")
    for k in (2, 3, 4, 8):
        cols = _make_cols(N, k)
        host_t = _bench_host(cols)
        if has_gpu:
            try:
                fresh_t = _bench_gpu_fresh_upload(cols)
                res_t = _bench_gpu_already_resident(cols)
            except Exception as e:
                print(f"k={k}: GPU path raised {e}")
                continue
            print(f"{k:>7} {host_t*1e3:>10.4f} {fresh_t*1e3:>13.4f} {res_t*1e3:>16.4f} {host_t/fresh_t:>14.2f} {host_t/res_t:>17.2f}")
        else:
            print(f"{k:>7} {host_t*1e3:>10.4f} {'n/a':>13} {'n/a':>16} {'n/a':>14} {'n/a':>17}")


if __name__ == "__main__":
    main()
