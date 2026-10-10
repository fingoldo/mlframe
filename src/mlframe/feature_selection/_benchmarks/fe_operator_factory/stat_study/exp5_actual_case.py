"""The actual case-2 data (n=30000): preset, shift grid, input shift, median-centering, zero-crossing and ALS rank-1 against the joint MI and the 0.9 ceiling, plus 5-fold out-of-sample ALS.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp5_actual_case``."""

import warnings

import numpy as np

from ..common.binning import mi, mi_b, qbin
from .exp2_general import als_rank1, best, centered, preset, shift_in, shift_out, zerocross

warnings.simplefilter("ignore")
n = 30000
rng = np.random.default_rng(0)
a, b, c, d, e, f = (rng.random(n) for _ in range(6))
y = 0.2 * a**2 / b + f / 5 + np.log(2 * c) * np.sin(d / 3)
yb = qbin(y, 10)
allr = np.ones(n, bool)
joint = mi_b(qbin(c, 10) * 10 + qbin(d, 10), yb, 100, 10)
print("joint", round(joint, 4), "ceiling", round(0.9 * joint, 4), "raw c", round(mi(c, yb), 4))
for nm, fn in (
    ("preset", lambda: preset(c, d, allr)),
    ("shiftOut", lambda: shift_out(c, d, allr)),
    ("inputShift", lambda: shift_in(c, d, allr)),
    ("median-centered", lambda: centered(c, d, allr)),
    ("zerocross", lambda: zerocross(c, d, allr, y)),
):
    m, k = best(fn(), allr, yb)
    print(f"{nm:16s} {m:.4f} ratio {m / joint:.3f} pass={m >= 0.9 * joint} {k}")
fr, _ = als_rank1(c, d, y, allr)
m = mi(fr, yb)
print("ALS rank1", round(m, 4), round(m / joint, 3))
# CV of ALS: 5-fold OOS MI
idx = rng.permutation(n)
oos = []
for i in range(5):
    B = np.zeros(n, bool)
    B[idx[i::5]] = True
    fr, _ = als_rank1(c, d, y, ~B)
    oos.append(mi(fr[B], yb[B]))
print("ALS 5-fold OOS MI mean", round(np.mean(oos), 4))
yr = qbin(y, 1000) / 1000.0
fr, _ = als_rank1(c, d, yr, allr)
print("ALS on rank(y) in-sample", round(mi(fr, yb), 4))
oos = []
for i in range(5):
    B = np.zeros(n, bool)
    B[idx[i::5]] = True
    fr, _ = als_rank1(c, d, yr, ~B)
    oos.append(mi(fr[B], yb[B]))
print("ALS on rank(y) 5-fold OOS", round(np.mean(oos), 4))
