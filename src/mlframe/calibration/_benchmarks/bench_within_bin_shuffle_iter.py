"""Bench PERF#2: within-bin shuffle in generate_similar_probs_by_ranking.

Old path: per bin does np.unique + full-array np.where scan -> O(n_bins*N).
New path: single argsort-by-bin + contiguous-slice walk (bins ascending, same
draw order per bin as np.unique -> bit-identical shuffled output).
"""
import time
import numpy as np
from sklearn.utils import check_random_state


def old_shuffle(predicted_probs, bins, similar_probs, rng):
    for bin_value in np.unique(bins):
        bin_indices = np.where(bins == bin_value)[0]
        bin_probs = similar_probs[bin_indices]
        rng.shuffle(bin_probs)
        similar_probs[bin_indices] = bin_probs
    return similar_probs


def new_shuffle(predicted_probs, bins, similar_probs, rng):
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    boundaries = np.flatnonzero(np.diff(sorted_bins)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(bins)]))
    for st, en in zip(starts, ends):
        seg = order[st:en]
        bin_probs = similar_probs[seg]
        rng.shuffle(bin_probs)
        similar_probs[seg] = bin_probs
    return similar_probs


def new_shuffle_contig(predicted_probs, bins, similar_probs, rng):
    # gather to contiguous once, shuffle contiguous slices, scatter once.
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    gathered = similar_probs[order]
    boundaries = np.flatnonzero(np.diff(sorted_bins)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(bins)]))
    for st, en in zip(starts, ends):
        seg = gathered[st:en]
        rng.shuffle(seg)
        gathered[st:en] = seg
    similar_probs[order] = gathered
    return similar_probs


def run(n, n_bins, seed=0, reps=5):
    rng0 = np.random.default_rng(seed)
    probs = rng0.random(n)
    from scipy.stats import rankdata
    ranks = rankdata(probs, method="ordinal")
    bins = np.floor_divide(ranks * n_bins, n)

    # correctness: bit-identical
    a = old_shuffle(probs, bins, probs.copy(), check_random_state(123))
    b = new_shuffle(probs, bins, probs.copy(), check_random_state(123))
    c = new_shuffle_contig(probs, bins, probs.copy(), check_random_state(123))
    identical = np.array_equal(a, b) and np.array_equal(a, c)

    def timeit(fn):
        best = np.inf
        for _ in range(reps):
            sp = probs.copy()
            r = check_random_state(123)
            t = time.perf_counter()
            fn(probs, bins, sp, r)
            best = min(best, time.perf_counter() - t)
        return best

    told = timeit(old_shuffle)
    tnew = timeit(new_shuffle)
    tcontig = timeit(new_shuffle_contig)
    print(f"n={n:>8} n_bins={n_bins:>4}  old={told*1e3:8.2f}ms  argsort={tnew*1e3:8.2f}ms  contig={tcontig*1e3:8.2f}ms  contig_speedup={told/tcontig:5.2f}x  identical={identical}")
    return identical


if __name__ == "__main__":
    ok = True
    for n in (100_000, 1_000_000):
        for nb in (10, 100):
            ok &= run(n, nb)
    print("ALL IDENTICAL:", ok)
