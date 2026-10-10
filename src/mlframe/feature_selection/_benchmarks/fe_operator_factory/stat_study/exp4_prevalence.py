"""Bias of the 0.9 x joint-MI prevalence ceiling: ratio of the true 1-D form MI to the joint 2-D MI versus n and bins, raw, Miller-Madow and null-debiased.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp4_prevalence``."""

# ratio MI(true 1-D form)/joint 2-D MI vs n and bins; raw, Miller-Madow, and permutation-null-subtracted
import numpy as np

from ..common.binning import mi, mi_b, qbin

rng = np.random.default_rng(3)


def jointmi(c, d, yb, k, mm=False):
    """MI of the k x k joint cell of two columns against the target codes."""
    return mi_b(qbin(c, k) * k + qbin(d, k), yb, k * k, yb.max() + 1, mm)


print(
    "case2 y exactly (a,b,f noise included). ratio=MI(true form)/joint(c,d).  cols: raw | Miller-Madow both | null-debiased (subtract mean perm-null of each)"
)
for n in [2000, 5000, 10000, 30000, 100000]:
    for k in [5, 10, 20]:
        if k * k * 10 * 20 > n and n < 5000:
            pass
        R = []
        RM = []
        RD = []
        J = []
        F = []
        for _r in range(6 if n < 100000 else 3):
            a, b, c, d, e, f = (rng.random(n) for _ in range(6))
            y = 0.2 * a**2 / b + f / 5 + np.log(2 * c) * np.sin(d / 3)
            yb = qbin(y, 10)
            tf = np.log(2 * c) * np.sin(d / 3)
            m1 = mi(tf, yb, k)
            j = jointmi(c, d, yb, k)
            m1m = mi(tf, yb, k, True)
            jm = jointmi(c, d, yb, k, True)
            nul1 = np.mean([mi(tf, rng.permutation(yb), k) for _ in range(5)])
            nulj = np.mean([jointmi(c, d, rng.permutation(yb), k) for _ in range(5)])
            R.append(m1 / j)
            RM.append(m1m / jm)
            RD.append((m1 - nul1) / (j - nulj))
            J.append(j)
            F.append(m1)
        print(
            f"n={n:6d} bins={k:2d}: joint {np.mean(J):.3f} true1D {np.mean(F):.3f} | ratio raw {np.mean(R):.3f} (sd {np.std(R):.3f}) | MM {np.mean(RM):.3f} | null-debiased {np.mean(RD):.3f}",
            flush=True,
        )
