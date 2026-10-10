"""Shift estimators on off-median crossings (``off``) and the family-level null / right-target gain of the shift grid over the preset (``null``).
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp3_offmedian_family_null off | null <n> <bins> <reps>``."""

import sys
import warnings

import numpy as np

from ..common.binning import mi, mi_b, qbin
from .exp2_general import als_rank1, best, centered, preset, shift_out, zerocross

warnings.simplefilter("ignore")
which = sys.argv[1]
if which == "off":
    n = 30000
    rng = np.random.default_rng(11)
    x, z = rng.random(n), rng.random(n)
    allr = np.ones(n, bool)
    T = {
        "(x-.25)*z": (x - 0.25) * z,
        "(x-.75)*z": (x - 0.75) * z,
        "log(3x)*sin(z/3) [cross@.33]": np.log(3 * x) * np.sin(z / 3),
        "log(1.3x)*sin(z) [cross@.77]": np.log(1.3 * x) * np.sin(z),
    }
    for tn, tr in T.items():
        y = tr + rng.standard_normal(n) * tr.std()
        yb = qbin(y, 10)
        joint = mi_b(qbin(x, 10) * 10 + qbin(z, 10), yb, 100, 10)
        P = best(preset(x, z, allr), allr, yb)
        S = best(shift_out(x, z, allr), allr, yb)
        C = best(centered(x, z, allr), allr, yb)
        Z = best(zerocross(x, z, allr, y), allr, yb)
        fr, _ = als_rank1(x, z, y, allr)
        print(
            f"[{tn}] truth {mi(tr, yb):.4f} joint {joint:.4f} | preset {P[0]:.4f} | shiftGrid {S[0]:.4f} ({S[1]}) | medianCentered {C[0]:.4f} | zerocross {Z[0]:.4f} ({Z[1]}) | ALS {mi(fr, yb):.4f}",
            flush=True,
        )
else:
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    nrep = int(sys.argv[4])
    rng = np.random.default_rng(5)
    for tgt in ("noise", "right"):
        gains = []
        tops = []
        for _r in range(nrep):
            x, z = rng.random(n), rng.random(n)
            allr = np.ones(n, bool)
            y = rng.standard_normal(n) if tgt == "noise" else np.log(x) * np.sin(z) + rng.standard_normal(n) * 1.0
            yb = qbin(y, k)

            def bst(c):
                """Best MI over a candidate dict."""
                return max(mi(v, yb, k) for v in c.values())

            P = preset(x, z, allr)
            bp = bst(P)
            bs = max(bp, bst(shift_out(x, z, allr)))
            gains.append(bs - bp)
            tops.append(bp)
        g = np.array(gains)
        print(
            f"{tgt} n={n} bins={k}: preset-best mean {np.mean(tops):.4f}; gain(union-preset) mean {g.mean():.5f} q50 {np.quantile(g, 0.5):.5f} q95 {np.quantile(g, 0.95):.5f} max {g.max():.5f}; frac gain>0 {np.mean(g > 1e-9):.2f}",
            flush=True,
        )
