"""MI of candidate forms of the case-2 ``(c, d)`` interaction ``log(2c) sin(d/3)`` versus the raw columns and the 2-D joint MI (n=30000).
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.case2_mi``."""

import warnings

warnings.simplefilter("ignore")
import numpy as np

n = 30000
rng = np.random.default_rng(0)
a, b, c, d, e, f = (rng.random(n) for _ in range(6))
y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)


def binned(x, k=10):
    """Decile codes of ``x`` (quantile edges)."""
    return np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)


def mi(xc, yc):
    """Plug-in MI of two integer code vectors."""
    jc = np.zeros((xc.max() + 1, yc.max() + 1))
    np.add.at(jc, (xc, yc), 1)
    p = jc / jc.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


yb = binned(y)
print("MI raw c:", round(mi(binned(c), yb), 4), " raw d:", round(mi(binned(d), yb), 4))
for name, v in (
    ("mul(log(c),sin(d))", np.log(c) * np.sin(d)),
    ("log(2c)*sin(d/3) true", np.log(2 * c) * np.sin(d / 3)),
    ("mul(log(c),d)", np.log(c) * d),
    ("c*d", c * d),
    ("sin(d)*c", np.sin(d) * c),
    ("log(c)", np.log(c)),
):
    print(f"MI {name:24s}: {mi(binned(v), yb):.4f}")
joint = binned(c, 10) * 10 + binned(d, 10)
print("joint 2D MI (c,d) 10x10 bins:", round(mi(joint, yb), 4))
