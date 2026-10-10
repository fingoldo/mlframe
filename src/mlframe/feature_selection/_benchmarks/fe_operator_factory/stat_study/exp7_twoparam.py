"""Two-sided targets: 1-parameter versus 2-parameter (closed-form and 2-D grid) shift families, raw and Miller-Madow MI; writes ``p2_<n>_<seed>.json``.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp7_twoparam <seed> [n]``."""

import json
import sys

import numpy as np

from ..common._paths import scratch_dir
from .core import bins_of, family_eval, get_presets, mi_cnt, preset_matrix_best, qbin, rank01, unary_mat
from .core2 import family_eval2

seed = int(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
UN, BI = get_presets("minimal")
rng = np.random.default_rng(2000 + seed)
x, z = rng.random(n), rng.random(n)
T = {
    "(x-.3)(z-.7)": (x - 0.3) * (z - 0.7),
    "(x-.5)(z-.5) quadrant": (x - 0.5) * (z - 0.5),
    "log(2x)(z-.4)": np.log(2 * x) * (z - 0.4),
    "(e^x-1.8)(sqrt(z+.1)-.5)": (np.exp(x) - 1.8) * (np.sqrt(z + 0.1) - 0.5),
    "mix 3xz+.5z+1.5x": 3 * x * z + 0.5 * z + 1.5 * x,
    "mix xz+2z-x": x * z + 2 * z - x,
    "CTRL (x-.5)z": (x - 0.5) * z,
    "CTRL log(2x)sin(z/3)": np.log(2 * x) * np.sin(z / 3),
    "CTRL log(x)sin(z) noshift": np.log(x) * np.sin(z),
    "CTRL x^2+z add": x**2 + z,
}
UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
out = []
for tn, tr in T.items():
    y = tr + rng.standard_normal(n) * tr.std()
    yb = qbin(y, 10)
    yr = rank01(y)
    rec = dict(target=tn, seed=seed, n=n)
    for mm in (False, True):
        sfx = "_mm" if mm else ""
        rec["truth" + sfx] = float(mi_cnt(bins_of(tr.astype(float), 10), yb, 10, 10, mm))
        rec["joint" + sfx] = float(mi_cnt((qbin(x, 10) * 10 + qbin(z, 10)).astype(np.int8), yb, 100, 10, mm))
        rec["preset" + sfx] = preset_matrix_best(UX, UZ, BI, yb, 10, mm)[0]
        for name, mode in (("ols1", 3), ("zc1", 2), ("grid1", 4)):
            mis, ts = family_eval(UX, UZ, yb, yr, mode, 10, 10, mm, 8, 0.0, True, False, False, False, 9)
            rec[name + sfx] = float(mis.max())
        for name, mode, wi, hu in (("ols2", 6, False, False), ("ols2rob", 6, True, True), ("grid2", 7, False, False)):
            if mm and mode == 7:
                continue
            mis, ss, tt = family_eval2(UX, UZ, yb, yr, mode, 10, 10, mm, wi, hu, 9)
            rec[name + sfx] = float(mis.max())
    out.append(rec)
(scratch_dir("stat_study") / f"p2_{n}_{seed}.json").write_text(json.dumps(out))
