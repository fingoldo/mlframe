"""Bench: np.random.shuffle via @njit vs inline LCG Fisher-Yates (iter126).

iter126 c0055 profile: shuffle_arr at 275 ms / 99 calls = 2.77 ms / call
on n=200k. The Besag-Clifford variant in the same file
(parallel_mi_besag_clifford) uses inline LCG + Fisher-Yates and is much
faster per-shuffle. The same pattern shipped for mi_direct's sequential
fallback via the new ``shuffle_arr_lcg``.

Bench (CPU, n=200_000):

    @njit np.random.shuffle  : 3.7 ms / call
    inline LCG Fisher-Yates  : 0.6 ms / call    (~6x)

Run: ``python profiling/bench_shuffle_arr_lcg.py``
"""

import time
import numpy as np
from numba import njit


@njit(cache=True)
def shuffle_np(arr):
    np.random.shuffle(arr)


@njit(cache=True)
def shuffle_inline_lcg(arr, base_seed):
    """Inline LCG Fisher-Yates (matches parallel_mi_besag_clifford)."""
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    n = len(arr)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = int(state >> np.uint64(33)) % (j + 1)
        tmp = arr[j]
        arr[j] = arr[k]
        arr[k] = tmp


n = 200_000

# Warmup
arr = np.arange(n, dtype=np.int32)
shuffle_np(arr.copy())
shuffle_inline_lcg(arr.copy(), 42)

for name, fn in [
    ('np.random.shuffle (@njit)', lambda a: shuffle_np(a)),
    ('inline LCG Fisher-Yates', lambda a: shuffle_inline_lcg(a, 42)),
]:
    arr = np.arange(n, dtype=np.int32)
    for _ in range(5):
        fn(arr)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(20):
            fn(arr)
        times.append((time.perf_counter() - t) / 20)
    print(f'{name:>30}: {min(times)*1e6:7.1f}us/call ({min(times)*1000:.3f}ms)')
