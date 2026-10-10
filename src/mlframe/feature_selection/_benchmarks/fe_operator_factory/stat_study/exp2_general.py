"""Generality of the shift family on nine synthetic targets: preset best versus shift grid, input shift, median-centering, zero-crossing, affine and ALS rank-1 against the 0.9 x joint ceiling, with a held-out selection.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp2_general``."""

# Generality: synthetic targets, preset-best vs offset-grid vs alternatives, prevalence ceiling, held-out selection
import itertools
import warnings

import numpy as np

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations, create_unary_transformations

from ..common.binning import mi, mi_b, qbin
import logging

logger = logging.getLogger(__name__)

UN = create_unary_transformations("minimal")
BI = create_binary_transformations("minimal")


def clean(a):
    """NaN and inf to 0."""
    return np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)


QG = np.linspace(0.1, 0.9, 9)


def preset(x, z, rows):
    """All preset (unary x unary x binary) candidate columns of a pair, keyed by name."""
    out = {}
    U = {(n, "x"): clean(f(x)) for n, f in UN.items()}
    U.update({(n, "z"): clean(f(z)) for n, f in UN.items()})
    for (n1, _f1), (n2, _f2) in itertools.product(UN.items(), UN.items()):
        u, v = U[(n1, "x")], U[(n2, "z")]
        for bn, fb in BI.items():
            try:
                val = clean(fb(u, v))
            except Exception as exc:
                logger.debug("operator %s failed on the study inputs: %s", bn, exc)
                continue
            if val[rows].std() > 0:
                out[f"{bn}({n1}(x),{n2}(z))"] = val
    return out


def shift_out(x, z, rows):  # (u+t)*v with t=-quantile_q(u) ; both roles
    """Output-shift candidates ``(u + t) v`` over the quantile grid, both roles."""
    out = {}
    for a, b, tag in ((x, z, "xz"), (z, x, "zx")):
        for n1, f1 in UN.items():
            u = clean(f1(a))
            for n2, f2 in UN.items():
                v = clean(f2(b))
                for q in QG:
                    t = -np.quantile(u[rows], q)
                    val = (u + t) * v
                    if val[rows].std() > 0:
                        out[f"mul({n1}({tag[0]})+t{q:.1f},{n2}({tag[1]}))"] = val
    return out


def shift_in(x, z, rows):  # u(x+k)*v(z)
    """Input-shift candidates ``u(x + k) v(z)``."""
    out = {}
    for a, b, tag in ((x, z, "xz"), (z, x, "zx")):
        rng_ = np.ptp(a[rows])
        for n1 in ("identity", "sqrt", "log", "reciproc"):
            for kk in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
                u = clean(UN[n1](a + kk * rng_ - a[rows].min() * 0))  # a>=0 in all targets
                for n2, f2 in UN.items():
                    v = clean(f2(b))
                    val = u * v
                    if val[rows].std() > 0:
                        out[f"mul({n1}({tag[0]}+{kk}R),{n2}({tag[1]}))"] = val
    return out


def centered(x, z, rows):  # closed form: center u at its median (t=-median), v unchanged / centered
    """Median-centred product candidates."""
    out = {}
    for a, b, tag in ((x, z, "xz"), (z, x, "zx")):
        for n1, f1 in UN.items():
            u = clean(f1(a))
            u = u - np.median(u[rows])
            for n2, f2 in UN.items():
                v = clean(f2(b))
                val = u * v
                if val[rows].std() > 0:
                    out[f"mul(c({n1}({tag[0]})),{n2}({tag[1]}))"] = val
                v2 = v - np.median(v[rows])
                val = u * v2
                if val[rows].std() > 0:
                    out[f"mul(c({n1}({tag[0]})),c({n2}({tag[1]})))"] = val
    return out


def zerocross(x, z, rows, y):  # data-driven t: sign flip of conditional slope of v on y across u-quantile bins
    """Zero-crossing shift candidates from the conditional slope sign flip of v on y across u-bins."""
    out = {}
    for a, b, tag in ((x, z, "xz"), (z, x, "zx")):
        for n1, f1 in UN.items():
            u = clean(f1(a))
            ub = qbin(u[rows], 8)
            idx = np.where(rows)[0]
            for n2, f2 in UN.items():
                v = clean(f2(b))
                s = np.array([np.corrcoef(v[idx[ub == i]], y[idx[ub == i]])[0, 1] for i in range(8)])
                s = np.nan_to_num(s)
                sg = np.sign(s)
                flips = np.where(sg[:-1] * sg[1:] < 0)[0]
                if len(flips) == 0:
                    continue
                i = flips[np.argmax(np.abs(s[flips] - s[flips + 1]))]
                # boundary between bin i and i+1 in u
                t = -0.5 * (u[idx[ub == i]].max() + u[idx[ub == i + 1]].min())
                val = (u + t) * v
                if val[rows].std() > 0:
                    out[f"mul({n1}({tag[0]})+tZC,{n2}({tag[1]}))"] = val
    return out


def affine(x, z, rows, top):  # (u+t)(v+s) on given top (n1,n2,tag) pairs
    """Two-sided affine candidates ``(u - q)(v - r)`` on the given top pairs."""
    out = {}
    for n1, n2, tag in top:
        a, b = (x, z) if tag == "xz" else (z, x)
        u = clean(UN[n1](a))
        v = clean(UN[n2](b))
        for q in QG:
            for r in QG:
                val = (u - np.quantile(u[rows], q)) * (v - np.quantile(v[rows], r))
                if val[rows].std() > 0:
                    out[f"aff({n1},{n2},{tag},{q:.1f},{r:.1f})"] = val
    return out


def als_rank1(x, z, y, rows_fit, B=16, iters=8):
    # f(x)*g(z) piecewise-linear on B quantile bins, fit to y by ALS
    """Binned rank-1 ALS fit; returns the product and the sum feature."""

    def ed(a):
        """Interior quantile edges of ``a`` on the fit rows."""
        return np.quantile(a[rows_fit], np.linspace(0, 1, B + 1)[1:-1])

    ex, ez = ed(x), ed(z)
    bx = np.searchsorted(ex, x)
    bz = np.searchsorted(ez, z)
    f = np.ones(B)
    g = np.ones(B)
    yc = y - y[rows_fit].mean()
    r = rows_fit
    for _ in range(iters):
        gz = g[bz]
        num = np.bincount(bx[r], gz[r] * yc[r], B)
        den = np.bincount(bx[r], gz[r] ** 2, B) + 1e-9
        f = num / den
        fx = f[bx]
        num = np.bincount(bz[r], fx[r] * yc[r], B)
        den = np.bincount(bz[r], fx[r] ** 2, B) + 1e-9
        g = num / den
        s = np.abs(f).max() + 1e-12
        f /= s
        g *= s
    return f[bx] * g[bz], f[bx] + g[bz]


def targets(n, rng):
    """Draw x, z and the nine synthetic truth targets."""
    x, z = rng.random(n), rng.random(n)
    T = {
        "case2 log(2x)sin(z/3)": np.log(2 * x) * np.sin(z / 3),
        "(x-.5)*z": (x - 0.5) * z,
        "log(x+.3)*z": np.log(x + 0.3) * z,
        "(exp(x)-1.8)*z": (np.exp(x) - 1.8) * z,
        "sqrt(x+.1)*sin(3z)": np.sqrt(x + 0.1) * np.sin(3 * z),
        "z/(x+.1)": z / (x + 0.1),
        "log(x)*sin(z) NOSHIFT": np.log(x) * np.sin(z),
        "x^2+z ADDITIVE": x**2 + z,
        "XOR sign": np.sign(x - 0.5) * np.sign(z - 0.5),
    }
    return x, z, T


def best(c, rows, yb, k=10):
    """Best candidate by MI on the given rows; returns (MI, name)."""
    bst = (-1, None)
    for nm, v in c.items():
        m = mi(v[rows], yb[rows], k)
        if m > bst[0]:
            bst = (m, nm)
    return bst


if __name__ == "__main__":
    n = 30000
    rng = np.random.default_rng(7)
    x, z, T = targets(n, rng)
    allr = np.ones(n, bool)
    idx = rng.permutation(n)
    A = np.zeros(n, bool)
    A[idx[: n // 2]] = True
    Bm = ~A
    print(
        f"n={n}, noise sd = 1.0*sd(truth), 10 bins. cols: truthMI | preset | +shiftOut grid | +inputShift grid | centered(median) | zerocross(closed) | affine(top5) | ALS rank1 (in-sample, OOS) | joint2D | ratio best/joint (gate .9)"
    )
    for tn, tr in T.items():
        y = tr + rng.standard_normal(n) * tr.std()
        yb = qbin(y, 10)
        joint = mi_b(qbin(x, 10) * 10 + qbin(z, 10), yb, 100, 10)
        truth = mi(tr, yb)
        P = preset(x, z, allr)
        bp = best(P, allr, yb)
        S = shift_out(x, z, allr)
        bs = best(S, allr, yb)
        Si = shift_in(x, z, allr)
        bi = best(Si, allr, yb)
        C = centered(x, z, allr)
        bc = best(C, allr, yb)
        Zc = zerocross(x, z, allr, y)
        bz = best(Zc, allr, yb) if Zc else (float("nan"), "none")
        # top5 preset product pairs for affine
        pr = sorted(((mi(v, yb), k) for k, v in P.items() if k.startswith("mul(")), reverse=True)[:5]
        import re

        top = [((*re.match(r"mul\((\w+)\(x\),(\w+)\(z\)\)", k).groups(), "xz")) for _, k in pr]
        Af = affine(x, z, allr, top)
        ba = best(Af, allr, yb)
        fr, fa = als_rank1(x, z, y, A)
        als_a = mi(fr[A], yb[A])
        als_b = mi(fr[Bm], yb[Bm])
        als_in = mi(als_rank1(x, z, y, allr)[0], yb)

        # held-out: select on A, score on B
        def ho(builder_rows):
            """Select on half A, score on half B: returns (A MI, B MI) of the winner."""
            C_ = builder_rows(A)
            b = best(C_, A, yb)
            return b[0], mi(C_[b[1]][Bm], yb[Bm]) if b[1] else float("nan")

        h_p = ho(lambda r: preset(x, z, r))
        h_s = ho(lambda r: {**preset(x, z, r), **shift_out(x, z, r)})
        print(f"\n[{tn}] truth {truth:.4f} joint {joint:.4f} ceiling0.9={0.9 * joint:.4f}")
        for nm, b in (("preset", bp), ("shiftOut", bs), ("inputShift", bi), ("centered", bc), ("zerocross", bz), ("affine", ba)):
            print(f"   {nm:10s} MI {b[0]:.4f} ratio {b[0] / joint:.3f} pass={b[0] >= 0.9 * joint}  {b[1]}")
        print(f"   ALS rank1 in-sample(full) {als_in:.4f} ratio {als_in / joint:.3f}; fit A -> A {als_a:.4f}, B {als_b:.4f}")
        print(
            f"   held-out (select on A, score on B): preset-only A {h_p[0]:.4f} -> B {h_p[1]:.4f} ; preset+shiftOut A {h_s[0]:.4f} -> B {h_s[1]:.4f}",
            flush=True,
        )
