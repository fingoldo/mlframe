"""Cost of the deviance-gain criterion: full Newton LRT versus a one / two step Newton approximation (research agent B1, wave 2).

For every fixture and seed the candidate set is scored with the full likelihood-ratio gain (Newton to tol 1e-3) and with k = 1, 2, 3 Newton steps from the offset-only fit. Reported: how often the argmax over the
non-identity candidates is the same candidate (or one with the same judge-relevant gain: within 2 % of the full-LRT top value), the Spearman correlation of the candidate scores, and the time per candidate.
Usage: python tt_b1_cheap_lrt.py [--seeds 4] [--rows 20000]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import collections
import time
import warnings

import numpy as np
from scipy import stats

import tt_b1_cls_feature_transform as T

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)


def lrt_steps(F, Y, L0, ll0, iters):
    out = np.empty(F.shape[1])
    ones = np.ones(F.shape[0])
    for j in range(F.shape[1]):
        _, ll1 = T.fit_mnl(np.c_[ones, F[:, j]], Y, L0, iters=iters if iters else 40, tol=0.0 if iters else 1e-3)
        out[j] = 2.0 * (ll1 - ll0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    agree = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    tm = collections.defaultdict(float)
    ncand = 0
    for K, kinds in ((2, ["log", "log_weak", "recip", "sqrt_g", "square", "abs", "beta_logit", "count_logn", "heavy_signal", "ctl_linear", "ctl_gamma", "ctl_pois", "ctl_noise"]),
                     (3, ["log", "square", "beta_logit", "ctl_linear", "ctl_noise"])):
        for kind in kinds:
            for seed in range(a.seeds):
                x, Z, y = T.make(kind, K, a.rows, seed)
                h = a.rows // 2
                C = T.cands(x)
                names = list(C)
                Fs, _, _ = T.stdz(np.c_[[C[k][:h] for k in names]].T)
                Zs, _, _ = T.stdz(Z[:h])
                Y = T.onehot(y[:h], K)
                L0, ll0, _ = T.base_offset(Zs, Y)
                t0 = time.perf_counter()
                full = lrt_steps(Fs, Y, L0, ll0, 0)
                tm["full"] += time.perf_counter() - t0
                ncand += len(names)
                iid = names.index("id")
                oth = [j for j in range(len(names)) if j != iid]
                top = max(oth, key=lambda j: full[j])
                for it in (1, 2, 3):
                    t0 = time.perf_counter()
                    ap_ = lrt_steps(Fs, Y, L0, ll0, it)
                    tm[it] += time.perf_counter() - t0
                    pick = max(oth, key=lambda j: ap_[j])
                    rec = agree[it]
                    rec[0] += pick == top
                    rec[1] += full[pick] >= 0.98 * full[top]
                    rec[2] += 1
                    rec[3] += stats.spearmanr(ap_[oth], full[oth])[0] if np.ptp(full[oth]) > 0 else 1.0
    print(f"cases {agree[1][2]}, candidates scored {ncand}")
    print(f"time per candidate: full {1e3 * tm['full'] / ncand:.2f} ms, " + ", ".join(f"{k} step(s) {1e3 * tm[k] / ncand:.2f} ms" for k in (1, 2, 3)))
    for it in (1, 2, 3):
        r = agree[it]
        print(f"{it} Newton step(s): same argmax as full LRT {r[0]}/{r[2]}, argmax within 2% of the full top value {r[1]}/{r[2]}, mean Spearman over non-identity candidates {r[3] / r[2]:.3f}")


if __name__ == "__main__":
    main()
