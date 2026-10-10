"""Operator O: random symbolic expression-tree search scored by MI (a cost reference, 26x the preset); cases O_W1 O_W2 O_N O_0.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_O <case names> [--seeds S] [--n N]``."""

import numpy as np

from ..common._paths import scratch_dir
from .h import BI, UN, case_args, clean, mi_pair, run_case, ybins
from .ops_pair import U2, mk

KU = list(UN)


def make_trees(ntree, seed=7):
    """Random expression trees (two shapes over unary / binary preset operators and random affine arguments)."""
    r = np.random.default_rng(seed)
    T = []
    A = np.array([-5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5.0])
    for _ in range(ntree):
        if r.random() < 0.5:
            T.append((1, r.choice(KU), r.choice(KU), r.choice(list(BI)), r.choice(A), r.uniform(-1, 1), r.choice(A), r.uniform(-1, 1)))
        else:
            T.append((2, r.choice(KU), r.choice(KU), r.choice(list(BI)), r.choice(KU), r.choice(A), r.uniform(-1, 1)))
    return T


def evalt(t, a, b):
    """Evaluate one tree on columns ``a`` and ``b`` with non-finite values cleaned."""
    with np.errstate(all="ignore"):
        if t[0] == 1:
            return clean(BI[t[3]](UN[t[1]](t[4] * a + t[5]), UN[t[2]](t[6] * b + t[7])))
        return clean(UN[t[4]](t[5] * BI[t[3]](UN[t[1]](a), UN[t[2]](b)) + t[6]))


def O_new(Xtr, ytr, Xte, rng, nt=6000):
    """Operator O: best of ``nt`` random trees by train MI; returns train / test feature and the tree."""
    yb = ybins(ytr, ytr)
    best = None
    for t in make_trees(nt):
        ft = evalt(t, Xtr[:, 0], Xtr[:, 1])
        if ft.std() == 0:
            continue
        m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, t)
    t = best[1]
    return evalt(t, Xtr[:, 0], Xtr[:, 1]), evalt(t, Xte[:, 0], Xte[:, 1]), t


cases = {
    "O_W1": (mk(U2, lambda X: np.sin(6 * X[:, 0] * X[:, 1]), lambda X: np.sin(6 * X[:, 0] * X[:, 1]), 0.3)),
    "O_W2": (mk(U2, lambda X: np.log(2 * X[:, 0]) * np.sin(3 * X[:, 1]), lambda X: np.log(2 * X[:, 0]) * np.sin(3 * X[:, 1]), 0.3)),
    "O_N": (mk(U2, lambda X: X[:, 0] / (X[:, 1] + 0.05), lambda X: X[:, 0] / (X[:, 1] + 0.05), 0.3)),
    "O_0": (mk(U2, lambda X: X[:, 0], None, null=True)),
}
if __name__ == "__main__":
    args = case_args()
    with (scratch_dir("brainstorm") / "results_O.jsonl").open("a") as out:
        for nm in args.names:
            run_case(nm, cases[nm], O_new, [0, 1], seeds=args.seeds, n=args.n, out=out)
