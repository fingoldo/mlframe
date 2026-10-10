"""Main table of the study: 13 targets x shift estimators (median, zero-crossing, OLS, robust OLS, grid, ...) at n=30000, one preset and seed per call; writes ``t_<preset>_<seed>.json``.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp6_tables <minimal|medium> <seed>``."""

import json
import sys

import numpy as np

from ..common._paths import scratch_dir
from .core import als_rank1, bins_of, family_eval, get_presets, mi_cnt, preset_matrix_best, qbin, rank01, unary_mat

preset = sys.argv[1]
seed = int(sys.argv[2])
n = 30000
UN, BI = get_presets(preset)
rng = np.random.default_rng(1000 + seed)
x, z = rng.random(n), rng.random(n)
T = {
    "case2 log(2x)sin(z/3)": np.log(2 * x) * np.sin(z / 3),
    "(x-.5)z": (x - 0.5) * z,
    "log(x+.3)z": np.log(x + 0.3) * z,
    "(exp(x)-1.8)z": (np.exp(x) - 1.8) * z,
    "sqrt(x+.1)sin(3z)": np.sqrt(x + 0.1) * np.sin(3 * z),
    "z/(x+.1)": z / (x + 0.1),
    "log(x)sin(z) NOSHIFT": np.log(x) * np.sin(z),
    "x^2+z ADD": x**2 + z,
    "XOR": np.sign(x - 0.5) * np.sign(z - 0.5),
    "OFF (x-.25)z": (x - 0.25) * z,
    "OFF (x-.75)z": (x - 0.75) * z,
    "OFF log(3x)sin(z/3)": np.log(3 * x) * np.sin(z / 3),
    "OFF log(1.3x)sin(z)": np.log(1.3 * x) * np.sin(z),
}
UX, UZ = unary_mat(UN, x), unary_mat(UN, z)
allr = np.ones(n, bool)
out = []
for tn, tr in T.items():
    y = tr + rng.standard_normal(n) * tr.std()
    yb = qbin(y, 10)
    yr = rank01(y)
    rec = dict(target=tn, preset=preset, seed=seed)
    rec["truth"] = float(mi_cnt(bins_of(tr.astype(float), 10), yb, 10, 10, False))
    rec["joint"] = float(mi_cnt((qbin(x, 10) * 10 + qbin(z, 10)).astype(np.int8), yb, 100, 10, False))
    rec["preset_best"] = preset_matrix_best(UX, UZ, BI, yb)[0]
    for name, mode, zthr, fb, wi, hu in [
        ("median", 1, 0.0, False, False, False),
        ("zc", 2, 0.0, False, False, False),
        ("zc_sig_fb", 2, 2.0, True, False, False),
        ("ols", 3, 0.0, False, False, False),
        ("ols_rob", 3, 0.0, False, True, True),
        ("grid", 4, 0.0, False, False, False),
        ("both_centred", 5, 0.0, False, False, False),
    ]:
        mis, ts = family_eval(UX, UZ, yb, yr, mode, 10, 10, False, 8, zthr, True, fb, wi, hu, 9)
        rec[name] = float(mis.max())
    rec["als"] = float(mi_cnt(bins_of(als_rank1(x, z, yr, allr), 10), yb, 10, 10, False))
    out.append(rec)
(scratch_dir("stat_study") / f"t_{preset}_{seed}.json").write_text(json.dumps(out))
