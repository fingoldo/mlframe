"""Smoke run of ``core``: preset best and every shift estimator on one case-2 draw (n=30000).
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.smoke``."""

import time

import numpy as np

from .core import family_eval, get_presets, preset_matrix_best, qbin, rank01, unary_mat

UN, BI = get_presets("medium")
print(len(UN), len(BI), list(BI))
n = 30000
rng = np.random.default_rng(0)
x, z = rng.random(n), rng.random(n)
tr = np.log(2 * x) * np.sin(z / 3)
y = tr + rng.standard_normal(n) * tr.std()
yb = qbin(y, 10)
yr = rank01(y)
UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
t0 = time.time()
print(preset_matrix_best(UX, UZ, BI, yb), time.time() - t0)
for mode, zthr in [(0, 0.0), (1, 0.0), (2, 0.0), (2, 2.0), (3, 0.0), (4, 0.0), (5, 0.0)]:
    t0 = time.time()
    mis, ts = family_eval(UX, UZ, yb, yr, mode, 10, 10, False, 8, zthr, True, False, False, False, 9)
    j = mis.argmax()
    print(mode, zthr, round(mis[j], 4), ts[j], j, round(time.time() - t0, 2))
mis, ts = family_eval(UX, UZ, yb, yr, 3, 10, 10, False, 8, 0.0, True, False, True, True, 9)
j = mis.argmax()
print("ols robust", round(mis[j], 4), ts[j])
