"""Ground-truth false/missed-merge bench for chance-corrected SU clustering (old = chance_correct=False).

Also reports candidate (A): columns with K > n/divisor treated as singletons (emulated by dropping them from merging).
Run: python bench_su_chance_correction.py [--quick]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import cluster_correlated_features_su

NS = [5_000, 50_000, 555_921]
CARDS = [50, 1_000, 10_000, 50_000]
SEEDS = range(5)


def build(n, K, seed):
    """Returns bins dict, set of true-merge pairs, set of must-not-merge pairs (all other pairs)."""
    rng = np.random.default_rng(seed)
    cols, true_pairs = {}, set()
    Ki = min(K, n)
    a = rng.integers(0, Ki, n)
    cols["ind0"] = a
    cols["ind1"] = rng.integers(0, Ki, n)
    cols["ind2"] = rng.integers(0, Ki, n)
    # deterministic recoding of ind0 (bijective permutation of levels)
    perm = rng.permutation(Ki)
    cols["dup0"] = perm[a]
    true_pairs.add(("ind0", "dup0"))
    # noisy copy: 10% of rows resampled
    noisy = a.copy()
    m = rng.random(n) < 0.10
    noisy[m] = rng.integers(0, Ki, int(m.sum()))
    cols["noisy0"] = noisy
    true_pairs.add(("ind0", "noisy0"))
    true_pairs.add(("dup0", "noisy0"))
    # low-card correlated pair
    lo = rng.integers(0, 8, n)
    lo2 = np.where(rng.random(n) < 0.05, rng.integers(0, 8, n), lo)
    cols["lo0"], cols["lo1"] = lo, lo2
    true_pairs.add(("lo0", "lo1"))
    return cols, true_pairs


def evaluate(labels, names, true_pairs):
    fm = mm = nf = nt = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = labels[i] == labels[j]
            tp = (names[i], names[j]) in true_pairs or (names[j], names[i]) in true_pairs
            if tp:
                nt += 1
                mm += (not same)
            else:
                nf += 1
                fm += same
    return fm, nf, mm, nt


def main(quick=False):
    ns = [5_000, 50_000] if quick else NS
    print(f"{'n':>8} {'K':>7} | {'old FM':>7} {'old MM':>7} | {'new FM':>7} {'new MM':>7} | {'A FM':>6} {'A MM':>6} | t_old  t_new")
    for n in ns:
        for K in CARDS:
            if K > n:
                continue
            acc = np.zeros(6)
            tot = np.zeros(2)
            t_o = t_n = 0.0
            for seed in (SEEDS if n < 500_000 else range(3)):
                cols, tp = build(n, K, seed)
                names = list(cols)
                t0 = time.process_time()
                lo = cluster_correlated_features_su(cols, feature_names=names, chance_correct=False, use_gpu=False)
                t_o += time.process_time() - t0
                t0 = time.process_time()
                ln = cluster_correlated_features_su(cols, feature_names=names, chance_correct=True, use_gpu=False)
                t_n += time.process_time() - t0
                # candidate A: singleton for K > n/20
                big = {c for c, v in cols.items() if len(np.unique(v)) > n / 20}
                keep = [c for c in names if c not in big]
                la = np.arange(len(names))
                if keep:
                    sub = cluster_correlated_features_su({c: cols[c] for c in keep}, feature_names=keep, chance_correct=False, use_gpu=False)
                    for k, c in enumerate(keep):
                        la[names.index(c)] = 1000 + sub[k]
                for m, lab in enumerate([lo, ln, la]):
                    fm, nf, mm, nt = evaluate(lab, names, tp)
                    acc[2 * m] += fm
                    acc[2 * m + 1] += mm
                    tot = np.array([nf, nt])
            s = len(SEEDS) if n < 500_000 else 3
            print(f"{n:>8} {K:>7} | {acc[0]/s:>7.2f} {acc[1]/s:>7.2f} | {acc[2]/s:>7.2f} {acc[3]/s:>7.2f} | {acc[4]/s:>6.2f} {acc[5]/s:>6.2f} | {t_o/s:.2f} {t_n/s:.2f}   (nonTrue={tot[0]}, true={tot[1]})")


if __name__ == "__main__":
    main("--quick" in sys.argv)
