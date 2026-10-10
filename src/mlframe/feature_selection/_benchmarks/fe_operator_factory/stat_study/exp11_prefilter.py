"""usage: exp11_prefilter.py seed n -> per dataset (5 numeric cols, 10 pairs): joint MM MI, preset-best MM MI, ratio, shift-family best (ols1, ols2) MM MI, ratio after."""

import itertools
import json
import sys

import numpy as np

from ..common._paths import scratch_dir
from .core import bins_of, family_eval, get_presets, mi_cnt, preset_matrix_best, qbin, rank01, unary_mat
from .core2 import family_eval2

seed = int(sys.argv[1])
n = int(sys.argv[2])
UN, BI = get_presets("minimal")
rng = np.random.default_rng(300 + seed)
X = rng.random((n, 5))
c = [np.ascontiguousarray(X[:, i]) for i in range(5)]
D = {
    "D1 case2-like: .2a^2/b + log(2c)sin(d/3)": (0.2 * c[0] ** 2 / c[1] + np.log(2 * c[2]) * np.sin(c[3] / 3), {(2, 3)}),
    "D2 (x0-.3)(x1-.7)+x2^2": ((c[0] - 0.3) * (c[1] - 0.7) + c[2] ** 2, {(0, 1)}),
    "D3 log(2x0)sin(x1)+x2*x3": (np.log(2 * c[0]) * np.sin(c[1]) + c[2] * c[3], {(0, 1)}),
    "D4 noise": (rng.standard_normal(n), set()),
    "D5 XOR(x0,x1)+x2": (np.sign(c[0] - 0.5) * np.sign(c[1] - 0.5) + c[2], {(0, 1)}),
    "D6 (x0-.5)x1 + 0.7*sqrt(x2)x3": ((c[0] - 0.5) * c[1] + 0.7 * np.sqrt(c[2]) * c[3], {(0, 1)}),
}
out = []
for dn, (tr, planted) in D.items():
    y = tr + (rng.standard_normal(n) * (np.subtract(*np.percentile(tr, [75, 25])) / 1.35) * 0.5 if dn != "D4 noise" else 0)
    yb = qbin(y, 10)
    yr = rank01(y)
    for i, j in itertools.combinations(range(5), 2):
        UX, UZ = unary_mat(UN, c[i]), unary_mat(UN, c[j])
        jm = mi_cnt((qbin(c[i], 10) * 10 + qbin(c[j], 10)).astype(np.int8), yb, 100, 10, True)
        pm = preset_matrix_best(UX, UZ, BI, yb, 10, True)[0]
        o1 = family_eval(UX, UZ, yb, yr, 3, 10, 10, True, 8, 0.0, True, False, False, False, 9)[0].max()
        o2 = family_eval2(UX, UZ, yb, yr, 6, 10, 10, True, False, False, 9)[0].max()
        mi_i = mi_cnt(bins_of(c[i], 10), yb, 10, 10, True)
        mi_j = mi_cnt(bins_of(c[j], 10), yb, 10, 10, True)
        out.append(
            dict(ds=dn, pair=[i, j], planted=(i, j) in planted, joint=float(jm), preset=float(pm), ols1=float(o1), ols2=float(o2), marg=float(max(mi_i, mi_j)))
        )
(scratch_dir("stat_study") / f"pf_{n}_{seed}.json").write_text(json.dumps(out))
