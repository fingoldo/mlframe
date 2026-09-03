"""Re-validates the 2026-07-21 bench-attempt-rejected verdict in ``_fe_cmi_redundancy_null.py``'s
``_conditional_perm_null`` (the per-permutation within-stratum-shuffle argsort loop): does batching all
``n_perm`` per-perm ``np.argsort(z_rank + keys, kind="stable")`` calls into one ``axis=0`` argsort over a
``(n, n_perm)`` key matrix beat the streaming per-perm loop CatBoost-cProfile flagged (1279s tottime / 19%
of a wellbore-50k STRICT-CPU MRMR.fit)?

The original verdict (0.63-0.75x, i.e. SLOWER) carried a caveat: the microbench ran on a machine under
heavy concurrent load, so a genuine allocator/cache-contention artifact from that load could have inflated
the batched variant's cost. This script re-measures with PAIRED, INTERLEAVED trials (alternating which
variant runs first each trial, cancelling systematic drift) specifically to resolve that caveat -- it does
NOT require an idle machine, by construction.

2026-08-16 re-run: median ratio (batched/per_perm) 1.43x SLOWER, 14/15 interleaved trials slower, ratio of
per-trial minimums 1.50x slower. Confirms the original rejection; not a load artifact. See the source
comment for the full writeup.
"""

from __future__ import annotations

import time

import numpy as np


def _bench(n: int = 8000, n_perm: int = 25, n_strata: int = 400, reps_per_trial: int = 25, n_trials: int = 15) -> None:
    """Run the paired/interleaved A/B and print the per-trial + aggregate verdict."""
    rng_master = np.random.default_rng(0)
    z = rng_master.integers(0, n_strata, size=n)
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    z_rank = np.zeros(n, dtype=np.float64)
    z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])

    def per_perm_loop(rng, reps):
        """Current production path: one argsort per permutation, streamed."""
        out = np.empty(reps, dtype=np.int64)
        for i in range(reps):
            keys = rng.random(n)
            within = np.argsort(z_rank + keys, kind="stable")
            out[i] = within[0]  # touch the result so it isn't optimized away
        return out

    def batched_axis0(rng, reps):
        """Previously-rejected batched variant: build the (n, reps) key matrix once, one axis=0 argsort."""
        keys_all = np.empty((n, reps), dtype=np.float64)
        for i in range(reps):
            keys_all[:, i] = rng.random(n)
        within_all = np.argsort(z_rank[:, None] + keys_all, axis=0, kind="stable")
        return within_all[0, :]

    per_perm_times = []
    batched_times = []
    for trial in range(n_trials):
        seed = 1000 + trial
        if trial % 2 == 0:
            t0 = time.perf_counter()
            per_perm_loop(np.random.default_rng(seed), reps_per_trial)
            t_pp = time.perf_counter() - t0
            t0 = time.perf_counter()
            batched_axis0(np.random.default_rng(seed), reps_per_trial)
            t_b = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            batched_axis0(np.random.default_rng(seed), reps_per_trial)
            t_b = time.perf_counter() - t0
            t0 = time.perf_counter()
            per_perm_loop(np.random.default_rng(seed), reps_per_trial)
            t_pp = time.perf_counter() - t0
        per_perm_times.append(t_pp)
        batched_times.append(t_b)
        print(f"trial {trial}: per_perm={t_pp * 1000:.2f}ms batched={t_b * 1000:.2f}ms ratio(batched/per_perm)={t_b / t_pp:.3f}", flush=True)

    pp = np.array(per_perm_times)
    bt = np.array(batched_times)
    print()
    print(f"median per_perm: {np.median(pp) * 1000:.2f}ms")
    print(f"median batched:  {np.median(bt) * 1000:.2f}ms")
    print(f"median ratio (batched/per_perm): {np.median(bt / pp):.3f}  (<1.0 = batched faster)")
    print(f"ratio of per-trial minimums: {bt.min() / pp.min():.3f}")

    rng_a = np.random.default_rng(42)
    r1 = per_perm_loop(rng_a, 10)
    r2 = batched_axis0(np.random.default_rng(42), 10)
    print("bit-identical first-elements:", np.array_equal(r1, r2))


if __name__ == "__main__":
    _bench()
