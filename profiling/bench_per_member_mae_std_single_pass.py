"""Bench: _per_member_mae_std two-pass vs single-pass E[X^2]-E[X]^2 (iter123, REJECTED).

Bench result at the c0123 shape (K=3, N=200_000):

    old (two-pass)   : 7.5 ms / call
    new (single-pass): 6.4 ms / call    (~15 %)

3-D (K=4, N=100k, C=3):

    old (two-pass)   : 25.6 ms / call
    new (single-pass): 20.4 ms / call   (~26 %)

MAE matches bit-identical; STD differs by at most 8.99e-15 (machine
epsilon). The optimisation was REJECTED to keep the numerical-stability
guarantee the numba ``_per_member_mae_std_njit`` branch explicitly notes:
``E[X^2] - E[X]^2`` loses precision when std/mean << 1 (members nearly
identical), and we never want the perf-tier numpy path to silently shift
the ensemble-member-gate kept/excluded decision under that regime.

See the ``bench-attempt-rejected`` comment at ensembling.py around the
numpy fallback in ``_per_member_mae_std``. Per
feedback_no_tradeoff_optimizations + feedback_document_failed_optimization_attempts.

Run: ``python profiling/bench_per_member_mae_std_single_pass.py``
"""

import time
import numpy as np


def old_two_pass(arr, median_preds):
    diffs = np.abs(arr - median_preds)
    if arr.ndim == 2:
        per_member_mae = diffs.mean(axis=1)
        per_member_std = np.sqrt(((diffs - per_member_mae[:, None]) ** 2).mean(axis=1))
    else:
        mae_per_col = diffs.mean(axis=1)
        std_per_col = np.sqrt(((diffs - mae_per_col[:, None, :]) ** 2).mean(axis=1))
        per_member_mae = mae_per_col.mean(axis=1)
        per_member_std = std_per_col.mean(axis=1)
    return per_member_mae, per_member_std


def new_single_pass(arr, median_preds):
    diffs = np.abs(arr - median_preds)
    if arr.ndim == 2:
        per_member_mae = diffs.mean(axis=1)
        # Var(X) = E[X^2] - E[X]^2 -- one allocation less than the explicit
        # (diffs - mean)**2 path.
        mean_of_sq = (diffs * diffs).mean(axis=1)
        per_member_std = np.sqrt(np.maximum(mean_of_sq - per_member_mae ** 2, 0.0))
    else:
        mae_per_col = diffs.mean(axis=1)  # (K, C)
        mean_of_sq_per_col = (diffs * diffs).mean(axis=1)  # (K, C)
        std_per_col = np.sqrt(np.maximum(mean_of_sq_per_col - mae_per_col ** 2, 0.0))
        per_member_mae = mae_per_col.mean(axis=1)
        per_member_std = std_per_col.mean(axis=1)
    return per_member_mae, per_member_std


rng = np.random.default_rng(0)

print("== 2-D (K=3, N=200k) -- c0123 shape ==")
arr = rng.standard_normal((3, 200_000))
median_preds = rng.standard_normal(200_000)

for name, fn in [('old (two-pass)', old_two_pass), ('new (single-pass)', new_single_pass)]:
    # Warmup
    for _ in range(3):
        fn(arr, median_preds)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(20):
            mae, std = fn(arr, median_preds)
        times.append((time.perf_counter() - t) / 20)
    print(f'{name:>20}: {min(times)*1000:7.3f}ms/call')

mae_old, std_old = old_two_pass(arr, median_preds)
mae_new, std_new = new_single_pass(arr, median_preds)
print(f'mae match: max abs diff = {np.abs(mae_old - mae_new).max():.2e}')
print(f'std match: max abs diff = {np.abs(std_old - std_new).max():.2e}')

print()
print("== 3-D (K=4, N=100k, C=3) ==")
arr3 = rng.standard_normal((4, 100_000, 3))
median_preds3 = rng.standard_normal((100_000, 3))

for name, fn in [('old (two-pass)', old_two_pass), ('new (single-pass)', new_single_pass)]:
    for _ in range(3):
        fn(arr3, median_preds3)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(20):
            mae, std = fn(arr3, median_preds3)
        times.append((time.perf_counter() - t) / 20)
    print(f'{name:>20}: {min(times)*1000:7.3f}ms/call')

mae_old3, std_old3 = old_two_pass(arr3, median_preds3)
mae_new3, std_new3 = new_single_pass(arr3, median_preds3)
print(f'3D mae max diff: {np.abs(mae_old3 - mae_new3).max():.2e}')
print(f'3D std max diff: {np.abs(std_old3 - std_new3).max():.2e}')
