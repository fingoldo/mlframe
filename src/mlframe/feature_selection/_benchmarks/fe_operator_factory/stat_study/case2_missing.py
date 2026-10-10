"""What the unary / binary presets miss for the case-2 ``(c, d)`` product: best preset forms (minimal, medium), the gain of each missing ingredient and the rank-1 ALS polynomial route.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.case2_missing``."""

import itertools
import warnings

warnings.simplefilter("ignore")
import numpy as np

from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations, create_unary_transformations

n = 30000
rng = np.random.default_rng(0)
a, b, c, d, e, f = (rng.random(n) for _ in range(6))
y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)


def binned(x, k=10):
    """Decile codes of ``x`` with non-finite values zeroed."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)


def mi(x):
    """MI of the decile-binned value against the case-2 target codes."""
    xc, yc = binned(x), yb
    jc = np.zeros((xc.max() + 1, yc.max() + 1))
    np.add.at(jc, (xc, yc), 1)
    p = jc / jc.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


yb = binned(y)
print("y = 0.2*a^2/b + f/5 + log(2c)*sin(d/3); the (c,d) part is a PRODUCT  f(c)*g(d)  with f=log(2c)=ln2+ln(c),  g=sin(d/3)")
print("raw c:", round(mi(c), 4), "  raw d:", round(mi(d), 4), "  the true (c,d) term alone:", round(mi(np.log(2 * c) * np.sin(d / 3)), 4))
for preset in ("minimal", "medium"):
    un = create_unary_transformations(preset)
    bi = create_binary_transformations(preset)
    best = []
    for (un1, f1), (un2, f2) in itertools.product(un.items(), un.items()):
        u, v = np.asarray(f1(c), float), np.asarray(f2(d), float)
        for bn, fb in bi.items():
            try:
                val = np.asarray(fb(u, v), float)
            except Exception:
                continue
            if np.isfinite(val).all() and val.std() > 0:
                best.append((mi(val), f"{bn}({un1}(c),{un2}(d))"))
    best.sort(reverse=True)
    print(f"\nbest {preset}-preset (c,d) forms:", [(round(m, 4), nm) for m, nm in best[:5]], f"  [{len(best)} candidates]")
print("\nwhat each missing ingredient would add (product forms):")
ln2 = np.log(2.0)
for label, v in (
    ("ln(c) * sin(d)            [best the preset can do ~]", np.log(c) * np.sin(d)),
    ("ln(c) * sin(d/3)          [+ scaled argument]", np.log(c) * np.sin(d / 3)),
    ("(ln(c)+ln2) * sin(d)      [+ additive shift inside the log]", (np.log(c) + ln2) * np.sin(d)),
    ("(ln(c)+ln2) * sin(d/3)    [both = the truth]", (np.log(c) + ln2) * np.sin(d / 3)),
    ("(ln(c)+ln2) * d           [shift only, linear in d]", (np.log(c) + ln2) * d),
    ("(ln(c)+0.5) * d", (np.log(c) + 0.5) * d),
    ("(ln(c)+1) * d", (np.log(c) + 1.0) * d),
):
    print(f"  MI {mi(v):.4f}   {label}")

print("\nthe rank-1 polynomial route (ALS warm start, f(c)*g(d) fitted to y directly):")
from mlframe.feature_selection.filters.hermite_fe import build_basis_matrix
from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import warm_start_als_seed

for basis in ("legendre", "chebyshev", "hermite"):
    for deg in (3, 5, 8):
        za = (2 * c - 1) if basis != "hermite" else (c - c.mean()) / c.std()
        zb = (2 * d - 1) if basis != "hermite" else (d - d.mean()) / d.std()
        Ba, Bb = build_basis_matrix(basis, za, deg), build_basis_matrix(basis, zb, deg)
        ca, cb = warm_start_als_seed(Ba, Bb, y, iters=5)
        if ca is None:
            print(f"  {basis:9s} deg {deg}: ALS declined")
            continue
        print(f"  {basis:9s} deg {deg}: MI of (Ba@ca)*(Bb@cb) = {mi((Ba @ ca) * (Bb @ cb)):.4f}")
