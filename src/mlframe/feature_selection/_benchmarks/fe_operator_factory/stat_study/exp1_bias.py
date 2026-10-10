"""Selection bias of a quantile-offset grid on pure noise: MI inflation of the grid maximum and the false-accept rate of the whole-grid null versus the selected-candidate-only null.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.exp1_bias``."""

# Selection bias of a t-grid (zero-crossing-quantile parametrisation) on noise and on no-shift-needed targets
import numpy as np

from ..common.binning import mi, mi_b, qbin

rng = np.random.default_rng(1)


def QS(G):
    """The G quantile levels in [0.1, 0.9]."""
    return np.linspace(0.1, 0.9, G)


def grid_mis(u, v, yb, G, k, mm):
    """MI of ``(u + t) v`` for the zero-crossing-quantile offsets ``t = -quantile_q(u)``."""
    out = []
    for q in QS(G):
        t = -np.quantile(u, q)
        out.append(mi((u + t) * v, yb, k, mm))
    return np.array(out)


def run(n, G, k, mm, target, nrep=60, nperm=40):
    """Averages over replicates: MI of the base form and grid max, mean null max / selected null, false-accept rates."""
    res = []
    for _ in range(nrep):
        x, z = rng.random(n), rng.random(n)
        u = np.log(x)
        v = np.sin(z)
        if target == "noise":
            y = rng.standard_normal(n)
        elif target == "trueprod":
            y = (u * v) + 0.0  # base form exactly right (t=0 irrelevant -> quantile grid contains none exactly)
        yb = qbin(y, k)
        base = mi(u * v, yb, k, mm)
        m = grid_mis(u, v, yb, G, k, mm)
        best = m.max()
        ibest = m.argmax()
        # whole-grid permutation null: permute y, rerun grid
        nullmax = []
        nullsel = []
        t_sel = -np.quantile(u, QS(G)[ibest])
        cand_sel = (u + t_sel) * v
        cb = qbin(cand_sel, k)
        for _p in range(nperm):
            yp = rng.permutation(yb)
            nullmax.append(grid_mis(u, v, yp, G, k, mm).max())
            nullsel.append(mi_b(cb, yp, k, k, mm))
        nullmax = np.array(nullmax)
        nullsel = np.array(nullsel)
        res.append(
            (
                base,
                best,
                np.mean(nullmax),
                np.mean(nullsel),
                best > np.quantile(nullmax, 0.95),
                best > np.quantile(nullsel, 0.95),
                base > np.quantile(nullsel, 0.95),
            )
        )
    r = np.array(res, float).mean(0)
    return r


print("n G k mm target | MI(base) MI(gridmax) E[null_max] E[null_selected] FA_wholegrid FA_selected-only")
for target in ["noise"]:
    for n in [2000, 10000]:
        for G in [1, 5, 9, 17]:
            for k, mm in [(10, False), (10, True), (20, False)]:
                r = run(n, G, k, mm, target, nrep=30, nperm=30)
                print(n, G, k, int(mm), target, "|", " ".join(f"{a:.4f}" for a in r[:4]), "|", f"{r[4]:.2f} {r[5]:.2f}", flush=True)
