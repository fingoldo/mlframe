"""Wall time of one pass of each shift family over the 578 (unary, unary, role) pairs of the medium preset at n=30000; backs the cost rows of the report.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.time_cost``."""

import time

import numpy as np

from .core import family_eval, get_presets, qbin, rank01, unary_mat
from .core2 import family_eval2

UN, BI = get_presets("medium")
n = 30000
rng = np.random.default_rng(0)
x, z = rng.random(n), rng.random(n)
tr = (x - 0.3) * (z - 0.7)
y = tr + rng.standard_normal(n) * tr.std()
yb = qbin(y, 10)
yr = rank01(y)
UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
family_eval(UX, UZ, yb, yr, 3, 10, 10, False, 8, 0.0, True, False, False, False, 9)
family_eval2(UX, UZ, yb, yr, 6, 10, 10, False, False, False, 9)


def T(f, *a):
    """Wall time of one call ``f(*a)``."""
    t = time.time()
    f(*a)
    return time.time() - t


print("pairs", 2 * 17 * 17)
print("plain (1 MI/pair)  ", T(family_eval, UX, UZ, yb, yr, 0, 10, 10, False, 8, 0.0, True, False, False, False, 9))
print("ols1 closed form   ", T(family_eval, UX, UZ, yb, yr, 3, 10, 10, False, 8, 0.0, True, False, False, False, 9))
print("zc                 ", T(family_eval, UX, UZ, yb, yr, 2, 10, 10, False, 8, 0.0, True, False, False, False, 9))
print("grid1 G=9          ", T(family_eval, UX, UZ, yb, yr, 4, 10, 10, False, 8, 0.0, True, False, False, False, 9))
print("ols2 closed form   ", T(family_eval2, UX, UZ, yb, yr, 6, 10, 10, False, False, False, 9))
print("ols2 huber         ", T(family_eval2, UX, UZ, yb, yr, 6, 10, 10, False, True, True, 9))
print("grid2 G=9 (81/pair)", T(family_eval2, UX, UZ, yb, yr, 7, 10, 10, False, False, False, 9))
