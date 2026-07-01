"""Bench PERF#5: fuse 2 np.nanmean passes in bin_predictions into one nan-aware loop."""
import time
import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def bin_predictions_old(y_true, y_pred, indices, nbins=20):
    pockets_predicted, pockets_true = np.zeros(nbins, dtype=np.float64), np.zeros(nbins, dtype=np.float64)
    data = np.zeros((nbins, 4), dtype=np.float64)
    s = len(y_pred)
    l = 0
    bin_size = s // nbins
    for i in range(nbins):
        if i == nbins - 1:
            r = s
        else:
            r = l + bin_size
        avg_x = np.nanmean(y_pred[indices[l:r]])
        avg_y = np.nanmean(y_true[indices[l:r]])
        pockets_predicted[i] = avg_x
        pockets_true[i] = avg_y
        data[i, :] = np.array([avg_x, avg_y * (r - l), r - l, avg_y], dtype=np.float64)
        l = r
    return pockets_predicted, pockets_true, data


@njit(cache=True, nogil=True, fastmath=False)
def bin_predictions_fused(y_true, y_pred, indices, nbins=20):
    pockets_predicted, pockets_true = np.zeros(nbins, dtype=np.float64), np.zeros(nbins, dtype=np.float64)
    data = np.zeros((nbins, 4), dtype=np.float64)
    s = len(y_pred)
    l = 0
    bin_size = s // nbins
    for i in range(nbins):
        if i == nbins - 1:
            r = s
        else:
            r = l + bin_size
        sum_x = 0.0
        sum_y = 0.0
        cnt_x = 0
        cnt_y = 0
        for j in range(l, r):
            idx = indices[j]
            vx = y_pred[idx]
            if not np.isnan(vx):
                sum_x += vx
                cnt_x += 1
            vy = y_true[idx]
            if not np.isnan(vy):
                sum_y += vy
                cnt_y += 1
        # match np.nanmean of all-NaN / empty slice -> nan
        avg_x = sum_x / cnt_x if cnt_x > 0 else np.nan
        avg_y = sum_y / cnt_y if cnt_y > 0 else np.nan
        pockets_predicted[i] = avg_x
        pockets_true[i] = avg_y
        data[i, :] = np.array([avg_x, avg_y * (r - l), r - l, avg_y], dtype=np.float64)
        l = r
    return pockets_predicted, pockets_true, data


def run(n, nbins, seed=0, reps=7, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    y_pred = rng.random(n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)
    if nan_frac:
        m = rng.random(n) < nan_frac
        y_pred[m] = np.nan
        y_true[rng.random(n) < nan_frac] = np.nan
    indices = np.argsort(y_pred)

    a = bin_predictions_old(y_true, y_pred, indices, nbins)
    b = bin_predictions_fused(y_true, y_pred, indices, nbins)
    identical = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))

    def timeit(fn):
        best = np.inf
        for _ in range(reps):
            t = time.perf_counter()
            fn(y_true, y_pred, indices, nbins)
            best = min(best, time.perf_counter() - t)
        return best

    told = timeit(bin_predictions_old)
    tnew = timeit(bin_predictions_fused)
    print(f"n={n:>8} nbins={nbins:>4} nan={nan_frac:.2f}  old={told*1e6:9.1f}us  fused={tnew*1e6:9.1f}us  speedup={told/tnew:5.2f}x  identical={identical}")
    return identical


if __name__ == "__main__":
    ok = True
    for n in (10_000, 100_000, 1_000_000):
        for nb in (20, 100):
            ok &= run(n, nb)
    # nan path correctness
    ok &= run(100_000, 20, nan_frac=0.1)
    print("ALL IDENTICAL:", ok)
