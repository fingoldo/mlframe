"""Pair operators A (rank copula product), B / B2 / E (cross-fitted 2-D cell table, tree, kNN), C (1-D warp), D (distance to extreme centroid), H (oblique angle), I (circular phase) with W / N / 0 cases.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_pair <case names, e.g. B_W B_N B_0> [--seeds S] [--n N]``."""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from ..common._paths import scratch_dir
from .h import case_args, mi_pair, oof, run_case, ybins


def mk(Xf, sf, tf=None, nz=0.3, null=False):
    """Build a case generator ``(rng, n) -> (X, y, truth)``; ``null`` makes y pure noise and drops the truth."""

    def gen(rng, n):
        """Draw X, a noisy y from the signal ``sf`` (noise level ``nz`` of its std) and the truth feature."""
        X = Xf(rng, n)
        s = sf(X)
        y = rng.standard_normal(n) if null else s + nz * s.std() * rng.standard_normal(n)
        return X, y, (None if (tf is None) else tf(X)) if not null else None

    return gen


def U2(rng, n):
    """Generator of two uniform columns."""
    return rng.random((n, 2))


def trunc_truth(tf):
    """Identity helper kept from the original case table."""
    return tf


# ---- A rank copula product
def A_new(Xtr, ytr, Xte, rng):
    """Operator A: product of train-CDF distances to the corner chosen by train MI (rank copula product)."""

    def cdf(ref, v):
        """Empirical CDF of ``v`` against the sorted reference."""
        return np.searchsorted(np.sort(ref), v) / len(ref)

    Fa, Fb = cdf(Xtr[:, 0], Xtr[:, 0]), cdf(Xtr[:, 1], Xtr[:, 1])
    Ga, Gb = cdf(Xtr[:, 0], Xte[:, 0]), cdf(Xtr[:, 1], Xte[:, 1])
    yb = ybins(ytr, ytr)
    best = None
    for sa in (0, 1):
        for sb in (0, 1):
            ft = (abs(sa - Fa)) * (abs(sb - Fb))
            m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, sa, sb)
    _, sa, sb = best
    return abs(sa - Fa) * abs(sb - Fb), abs(sa - Ga) * abs(sb - Gb), best


def LN(rng, n):
    """Generator of two log-normal columns."""
    return np.exp(1.5 * rng.standard_normal((n, 2)))


from scipy.stats import norm


def A_sig(X):
    """A target signal: product of Gaussian CDFs of the log columns."""
    return norm.cdf(np.log(X[:, 0]) / 1.5) * norm.cdf(np.log(X[:, 1]) / 1.5)


# ---- B / B2 / E : 2-D nonparametric E[y|x,z] cross-fitted
def cell_fp(K, m=20):
    """Fit-predict function: smoothed mean of y per K x K quantile cell."""

    def fp(Xa, ya, Xb):
        """Smoothed per-cell mean fitted on ``Xa, ya`` and looked up for ``Xb``."""
        ea = [np.quantile(Xa[:, j], np.linspace(0, 1, K + 1)[1:-1]) for j in (0, 1)]
        ia = np.searchsorted(ea[0], Xa[:, 0]) * K + np.searchsorted(ea[1], Xa[:, 1])
        ib = np.searchsorted(ea[0], Xb[:, 0]) * K + np.searchsorted(ea[1], Xb[:, 1])
        s = np.bincount(ia, ya, K * K)
        c = np.bincount(ia, minlength=K * K)
        mu = ya.mean()
        return ((s + m * mu) / (c + m))[ib]

    return fp


def B_new(Xtr, ytr, Xte, rng):
    """Operator B: out-of-fold 2-D cell table E[y|x, z], best K by train MI."""
    yb = ybins(ytr, ytr)
    best = None
    for K in (6, 10):
        ftr, fte = oof(cell_fp(K), Xtr[:, :2], ytr, Xte[:, :2], rng=rng)
        m = mi_pair(ftr, ftr, yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, ftr, fte, K)
    return best[1], best[2], best[3]


def tree_fp(Xa, ya, Xb):
    """Fit-predict function: small regression tree."""
    return DecisionTreeRegressor(max_leaf_nodes=24, min_samples_leaf=80).fit(Xa, ya).predict(Xb)


def B2_new(Xtr, ytr, Xte, rng):
    """Operator B2: out-of-fold regression-tree prediction on the pair."""
    a, b = oof(tree_fp, Xtr[:, :2], ytr, Xte[:, :2], rng=rng)
    return a, b, 0


def knn_fp(Xa, ya, Xb):
    """Fit-predict function: 40-nearest-neighbour mean on standardised columns."""
    mu, sd = Xa.mean(0), Xa.std(0)
    return KNeighborsRegressor(40).fit((Xa - mu) / sd, ya).predict((Xb - mu) / sd)


def E_new(Xtr, ytr, Xte, rng):
    """Operator E: out-of-fold kNN prediction on the pair."""
    a, b = oof(knn_fp, Xtr[:, :2], ytr, Xte[:, :2], rng=rng)
    return a, b, 0


def bumps(X):
    """Two-Gaussian-bump signal on the unit square."""

    def f(c1, c2):
        """One Gaussian bump centred at ``(c1, c2)``."""
        return np.exp(-((X[:, 0] - c1) ** 2 + (X[:, 1] - c2) ** 2) / 0.02)

    return f(0.25, 0.25) + f(0.75, 0.7)


# ---- C : 1-D cross-fitted binned warp (X col0 signal, col1 noise)
def warp_fp(Xa, ya, Xb, nb_=20):
    """Fit-predict function: piecewise-linear interpolation of per-bin mean y over the first column."""
    x = Xa[:, 0]
    e = np.quantile(x, np.linspace(0, 1, nb_ + 1)[1:-1])
    i = np.searchsorted(e, x)
    cx = np.array([np.median(x[i == k]) for k in range(nb_)])
    cy = np.array([ya[i == k].mean() for k in range(nb_)])
    return np.interp(Xb[:, 0], cx, cy)


def C_new(Xtr, ytr, Xte, rng):
    """Operator C: out-of-fold 1-D warp of the first column."""
    a, b = oof(warp_fp, Xtr, ytr, Xte, rng=rng)
    return a, b, 0


# ---- D : distance to target-extreme centroid
def D_new(Xtr, ytr, Xte, rng):
    """Operator D: distance to the top-decile or bottom-decile target centroid, chosen by train MI."""

    def Z(X):
        """Standardise the first two columns with the train mean and std."""
        return (X[:, :2] - Xtr[:, :2].mean(0)) / Xtr[:, :2].std(0)

    Ztr, Zte = Z(Xtr), Z(Xte)
    yb = ybins(ytr, ytr)
    best = None
    for _name, msk in (("top", ytr >= np.quantile(ytr, 0.9)), ("bot", ytr <= np.quantile(ytr, 0.1))):
        c = Ztr[msk].mean(0)
        ft = np.hypot(*(Ztr - c).T)
        m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, c)
    c = best[1]
    return np.hypot(*(Ztr - c).T), np.hypot(*(Zte - c).T), c


# ---- H : oblique angle scan
def H_new(Xtr, ytr, Xte, rng):
    """Operator H: best oblique projection angle over 24 angles by train MI."""
    mu, sd = Xtr[:, :2].mean(0), Xtr[:, :2].std(0)
    Ztr, Zte = (Xtr[:, :2] - mu) / sd, (Xte[:, :2] - mu) / sd
    yb = ybins(ytr, ytr)
    best = None
    for th in np.linspace(0, np.pi, 25)[:-1]:
        w = np.array([np.cos(th), np.sin(th)])
        ft = Ztr @ w
        m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, w)
    w = best[1]
    return Ztr @ w, Zte @ w, w


# ---- I : circular phase distance, period 24
def I_new(Xtr, ytr, Xte, rng):
    """Operator I: circular distance (period 24) to the best phase on a half-hour grid."""

    def cd(x, p):
        """Circular distance of ``x`` to the phase ``p`` for period 24."""
        return np.minimum(np.abs(x - p), 24 - np.abs(x - p))

    yb = ybins(ytr, ytr)
    best = None
    for p in np.arange(0, 24, 0.5):
        ft = cd(Xtr[:, 0], p)
        m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, p)
    return cd(Xtr[:, 0], best[1]), cd(Xte[:, 0], best[1]), best[1]


def I_four(Xtr, ytr, Xte, rng):
    """Operator I (Fourier variant): best sin / cos harmonic among k = 1, 2, 3."""
    yb = ybins(ytr, ytr)
    best = None
    for k in (1, 2, 3):
        for fn in (np.sin, np.cos):
            ft = fn(2 * np.pi * k * Xtr[:, 0] / 24)
            m = mi_pair(ft, ft, yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, k, fn)
    _, k, fn = best
    return fn(2 * np.pi * k * Xtr[:, 0] / 24), fn(2 * np.pi * k * Xte[:, 0] / 24), k


def XC(rng, n):
    """Generator: an hour-of-day column in [0, 24) and a uniform noise column."""
    return np.column_stack([rng.random(n) * 24, rng.random(n)])


def cd_t(x, p):
    """Circular distance of ``x`` to the phase ``p`` for period 24 (truth feature)."""
    return np.minimum(np.abs(x - p), 24 - np.abs(x - p))


CASES = {
    "A_W": (mk(LN, A_sig, A_sig, 0.2), A_new),
    "A_N": (mk(U2, lambda X: X[:, 0] * X[:, 1], lambda X: X[:, 0] * X[:, 1], 0.2), A_new),
    "A_0": (mk(LN, A_sig, None, null=True), A_new),
    "B_W": (mk(U2, bumps, bumps, 0.3), B_new),
    "B_N": (mk(U2, lambda X: X[:, 0] + X[:, 1] ** 2, lambda X: X[:, 0] + X[:, 1] ** 2, 0.3), B_new),
    "B_0": (mk(U2, bumps, None, null=True), B_new),
    "B2_W": (mk(U2, bumps, bumps, 0.3), B2_new),
    "B2_N": (mk(U2, lambda X: X[:, 0] + X[:, 1] ** 2, lambda X: X[:, 0] + X[:, 1] ** 2, 0.3), B2_new),
    "B2_0": (mk(U2, bumps, None, null=True), B2_new),
    "E_W": (mk(U2, bumps, bumps, 0.3), E_new),
    "E_N": (mk(U2, lambda X: X[:, 0] + X[:, 1] ** 2, lambda X: X[:, 0] + X[:, 1] ** 2, 0.3), E_new),
    "E_0": (mk(U2, bumps, None, null=True), E_new),
    "C_W": (mk(U2, lambda X: np.sin(9.4 * X[:, 0]), lambda X: np.sin(9.4 * X[:, 0]), 0.3), C_new),
    "C_N": (mk(U2, lambda X: 2 * X[:, 0], lambda X: X[:, 0], 0.3), C_new),
    "C_0": (mk(U2, lambda X: X[:, 0], None, null=True), C_new),
    "D_W": (mk(U2, lambda X: np.exp(-((X[:, 0] - 0.3) ** 2 + (X[:, 1] - 0.7) ** 2) / 0.05), lambda X: np.hypot(X[:, 0] - 0.3, X[:, 1] - 0.7), 0.3), D_new),
    "D_N": (mk(U2, lambda X: X[:, 0] * X[:, 1], lambda X: X[:, 0] * X[:, 1], 0.3), D_new),
    "D_0": (mk(U2, lambda X: X[:, 0], None, null=True), D_new),
    "H_W": (mk(U2, lambda X: np.sin(9.4 * (0.8 * X[:, 0] + 0.6 * X[:, 1])), lambda X: 0.8 * X[:, 0] + 0.6 * X[:, 1], 0.3), H_new),
    "H_N": (mk(U2, lambda X: X[:, 0] * X[:, 1], lambda X: X[:, 0] * X[:, 1], 0.3), H_new),
    "H_0": (mk(U2, lambda X: X[:, 0], None, null=True), H_new),
    "I_W": (mk(XC, lambda X: np.exp(-(cd_t(X[:, 0], 23.0) ** 2) / 8), lambda X: cd_t(X[:, 0], 23.0), 0.3), I_new),
    "I_N": (mk(XC, lambda X: X[:, 0], lambda X: X[:, 0], 0.3), I_new),
    "I_0": (mk(XC, lambda X: X[:, 0], None, null=True), I_new),
    "If_W": (mk(XC, lambda X: np.exp(-(cd_t(X[:, 0], 23.0) ** 2) / 8), lambda X: cd_t(X[:, 0], 23.0), 0.3), I_four),
}
if __name__ == "__main__":
    args = case_args()
    with (scratch_dir("brainstorm") / "results_pair.jsonl").open("a") as out:
        for nm in args.names:
            g, f = CASES[nm]
            run_case(nm, g, f, [0, 1], seeds=args.seeds, n=args.n, out=out)
