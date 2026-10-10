"""Wall time of every brainstorm operator at n=100000 versus the existing 1734-combo pair table; writes ``results_cost.json`` to the scratch folder.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.cost``."""

import itertools
import json
import time

import numpy as np

from ..common._paths import scratch_dir
from . import ops_M as MM
from . import ops_multi as M
from . import ops_O as O
from . import ops_pair as P
from .h import UN, _best, clean, ybins

n = 100000
rng = np.random.default_rng(0)
X2 = rng.random((n, 2))
y = X2[:, 0] * X2[:, 1] + 0.1 * rng.standard_normal(n)
Xte2 = rng.random((10000, 2))


def T(f, *a):
    """Wall time of one call ``f(*a)``."""
    t = time.time()
    f(*a)
    return time.time() - t


yb = ybins(y, y)
UA = np.array([clean(UN[k](X2[:, 0])) for k in UN])
UB = np.array([clean(UN[k](X2[:, 1])) for k in UN])
_best(UA[:, :100], UB[:, :100], yb[0][:100], yb[2])
res = {"existing_1734_njit_subsorted": T(_best, UA, UB, yb[0], yb[2])}
XC = np.column_stack([rng.random(n) * 24, rng.random(n)])
X5 = rng.random((n, 5))
X4 = rng.random((n, 4))
Xi = np.column_stack([rng.integers(0, 1024, n).astype(float), rng.random(n)])
G = rng.gamma(2, 1, (n, 4))
for nm, f, args in [
    ("A_rank", P.A_new, (X2, y, Xte2, rng)),
    ("B_cell2d", P.B_new, (X2, y, Xte2, rng)),
    ("B2_tree", P.B2_new, (X2, y, Xte2, rng)),
    ("E_knn", P.E_new, (X2, y, Xte2, rng)),
    ("C_warp1d", P.C_new, (X2, y, Xte2, rng)),
    ("D_proto", P.D_new, (X2, y, Xte2, rng)),
    ("H_angle", P.H_new, (X2, y, Xte2, rng)),
    ("I_circ", P.I_new, (XC, y, XC[:1000], rng)),
    ("F_count", M.F_new, (X5, y, X5[:1000], rng)),
    ("F2_exh", M.F2_new, (X5, y, X5[:1000], rng)),
    ("G_rowstats", M.G_new, (X4, y, X4[:1000], rng)),
    ("J_lse", M.J_new, (X4, y, X4[:1000], rng)),
    ("K_trop", M.K_new, (X4, y, X4[:1000], rng)),
    ("L_parity", M.L_new, (X5, y, X5[:1000], rng)),
    ("AA_comp", M.AA_new, (G, y, G[:1000], rng)),
    ("S_bits", M.S_new, (Xi, y, Xi[:1000], rng)),
    ("O_symb6000", O.O_new, (X2, y, Xte2, rng)),
]:
    res[nm] = T(f, *args)
    print(nm, round(res[nm], 3), flush=True)
# M screen: 28 pairs of p=8, restricted 216 combos
X8 = rng.random((n, 8))
y8 = rng.standard_normal(n)
pairs = list(itertools.combinations(range(8), 2))
res["M_resid_screen_p8"] = T(lambda: (MM.resid_oof(X8, y8, rng), MM.screen(X8, y8, pairs)))
print("M", res["M_resid_screen_p8"])
print("existing", res["existing_1734_njit_subsorted"])
(scratch_dir("brainstorm") / "results_cost.json").write_text(json.dumps(res, indent=1))
