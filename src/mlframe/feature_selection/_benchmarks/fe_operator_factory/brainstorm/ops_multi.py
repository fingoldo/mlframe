"""Multi-column operators F (count above median), G (row statistics), J (log-sum-exp), K (tropical), L (parity), AA (compositional shares / entropy), S (bit features) with their W / N / 0 cases.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_multi <case names, e.g. G_W G_N G_0> [--seeds S] [--n N]``."""

import itertools

import numpy as np

from ..common._paths import scratch_dir
from .h import case_args, clean, mi_pair, run_case, ybins
from .ops_pair import mk


def UP(p):
    """Generator of ``p`` uniform columns."""
    return lambda rng, n: rng.random((n, p))


def topk_cols(Xtr, yb):
    """Column indices sorted by univariate MI, best first."""
    sc = [mi_pair(Xtr[:, j], Xtr[:, j], yb[0], yb[0], yb[2])[0] for j in range(Xtr.shape[1])]
    return np.argsort(sc)[::-1]


def sel(cands_fn, Xtr, ytr, Xte, ks):
    """Pick the best candidate (by train MI) over column-subset sizes ``ks`` of the top columns and every candidate the generator yields."""
    yb = ybins(ytr, ytr)
    order = topk_cols(Xtr, yb)
    best = None
    for k in ks:
        S = order[:k]
        for nm, ftr, fte in cands_fn(Xtr[:, S], Xte[:, S], Xtr[:, S]):
            m = mi_pair(clean(ftr), clean(ftr), yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, ftr, fte, (nm, k))
    return best[1], best[2], best[3]


# F count above median
def F_c(A, B, ref):
    """Candidate: count of top columns above their train median."""
    med = np.median(ref, 0)
    yield "cnt", (A > med).sum(1), (B > med).sum(1)


def F_new(Xtr, ytr, Xte, rng):
    """Operator F: count above median over the top-k columns."""
    return sel(F_c, Xtr, ytr, Xte, range(2, Xtr.shape[1] + 1))


def F_sig(X):
    """F target signal: squared deviation of the count of the first four columns above 0.5."""
    c = (X[:, :4] > 0.5).sum(1)
    return (c - 2.0) ** 2


def F2_new(Xtr, ytr, Xte, rng):
    """Operator F2: exhaustive subset search for the count feature."""
    med = np.median(Xtr, 0)
    yb = ybins(ytr, ytr)
    best = None
    p = Xtr.shape[1]
    for k in range(2, p + 1):
        for S in itertools.combinations(range(p), k):
            S = list(S)
            f = (Xtr[:, S] > med[S]).sum(1).astype(float)
            m = mi_pair(f, f, yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, S)
    S = best[1]
    return (Xtr[:, S] > med[S]).sum(1).astype(float), (Xte[:, S] > med[S]).sum(1).astype(float), S


# G row stats
def G_c(A, B, ref):
    """Candidates: row min, max, median, range, std, mean over the selected columns."""
    for nm, f in (
        ("min", lambda Z: Z.min(1)),
        ("max", lambda Z: Z.max(1)),
        ("med", lambda Z: np.median(Z, 1)),
        ("rng", lambda Z: Z.max(1) - Z.min(1)),
        ("std", lambda Z: Z.std(1)),
        ("mean", lambda Z: Z.mean(1)),
    ):
        yield nm, f(A), f(B)


def G_new(Xtr, ytr, Xte, rng):
    """Operator G: row statistics over the top-k columns."""
    return sel(G_c, Xtr, ytr, Xte, range(2, Xtr.shape[1] + 1))


# J LSE
def J_c(A, B, ref):
    """Candidates: log-sum-exp with positive and negative sharpness."""
    from scipy.special import logsumexp

    for b in (2, 5, 10, 20, -2, -5, -10, -20):
        yield f"lse{b}", logsumexp(b * A, 1) / b, logsumexp(b * B, 1) / b


def J_new(Xtr, ytr, Xte, rng):
    """Operator J: log-sum-exp over the top-k columns."""
    return sel(J_c, Xtr, ytr, Xte, range(2, Xtr.shape[1] + 1))


from scipy.special import logsumexp


def J_sig(X):
    """J target signal: log-sum-exp with sharpness 7."""
    return logsumexp(7 * X, 1) / 7


# K tropical (p=4)
def K_new(Xtr, ytr, Xte, rng):
    """Operator K: min / max of two sums over the three pairings of four columns."""
    yb = ybins(ytr, ytr)
    best = None
    for (i, j), (k, l) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        for nm, op in (("min", np.minimum), ("max", np.maximum)):

            def f(X, i=i, j=j, k=k, l=l, op=op):
                """Tropical form for one pairing."""
                return op(X[:, i] + X[:, j], X[:, k] + X[:, l])

            m = mi_pair(f(Xtr), f(Xtr), yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, f, nm)
    return best[1](Xtr), best[1](Xte), best[2]


# L parity
def L_new(Xtr, ytr, Xte, rng):
    """Operator L: parity of 2 or 3 median-binarised columns."""
    med = np.median(Xtr, 0)
    Btr, Bte = (Xtr > med).astype(int), (Xte > med).astype(int)
    yb = ybins(ytr, ytr)
    best = None
    for r in (2, 3):
        for S in itertools.combinations(range(Xtr.shape[1]), r):
            f = Btr[:, S].sum(1) % 2
            m = mi_pair(f.astype(float), f.astype(float), yb[0], yb[0], yb[2])[0]
            if best is None or m > best[0]:
                best = (m, S)
    S = best[1]
    return (Btr[:, S].sum(1) % 2).astype(float), (Bte[:, S].sum(1) % 2).astype(float), S


def L_sig(X):
    """L target signal: parity of three binarised columns."""
    return ((X[:, :3] > 0.5).sum(1) % 2).astype(float)


# AA compositional
def GM(p):
    """Generator of ``p`` gamma(2, 1) columns."""
    return lambda rng, n: rng.gamma(2.0, 1.0, (n, p))


def AA_c(A, B, ref):
    """Candidates: entropy, Herfindahl index, max and min share of the row-normalised columns."""
    for nm, f in (
        ("ent", lambda Z: -(lambda s: (s * np.log(s)).sum(1))(Z / Z.sum(1, keepdims=True))),
        ("hhi", lambda Z: ((Z / Z.sum(1, keepdims=True)) ** 2).sum(1)),
        ("maxsh", lambda Z: (Z / Z.sum(1, keepdims=True)).max(1)),
        ("minsh", lambda Z: (Z / Z.sum(1, keepdims=True)).min(1)),
    ):
        yield nm, f(A), f(B)


def AA_new(Xtr, ytr, Xte, rng):
    """Operator AA: compositional shares / entropy over all columns."""
    return sel(AA_c, Xtr, ytr, Xte, [Xtr.shape[1]])


def AA_sig(X):
    """AA target signal: Shannon entropy of the row shares."""
    s = X / X.sum(1, keepdims=True)
    return -(s * np.log(s)).sum(1)


# S popcount
def pc(k):
    """Popcount of integer values."""
    k = k.astype(np.int64)
    c = np.zeros(len(k))
    for b in range(11):
        c += (k >> b) & 1
    return c


def tz(k):
    """Trailing zero bits of integer values."""
    k = k.astype(np.int64)
    c = np.zeros(len(k))
    m = np.ones(len(k), bool)
    for b in range(11):
        m &= ((k >> b) & 1) == 0
        c += m
    return c


def ds(k):
    """Decimal digit sum of integer values (up to four digits)."""
    k = k.astype(np.int64)
    return (k % 10) + (k // 10) % 10 + (k // 100) % 10 + (k // 1000)


def S_new(Xtr, ytr, Xte, rng):
    """Operator S: best of popcount, trailing zeros, digit sum and Gray-code popcount of an integer column."""
    yb = ybins(ytr, ytr)
    best = None
    for nm, f in (("pop", pc), ("tz", tz), ("digsum", ds), ("gray", lambda k: pc(k.astype(np.int64) ^ (k.astype(np.int64) >> 1)))):
        m = mi_pair(f(Xtr[:, 0]), f(Xtr[:, 0]), yb[0], yb[0], yb[2])[0]
        if best is None or m > best[0]:
            best = (m, f, nm)
    return best[1](Xtr[:, 0]), best[1](Xte[:, 0]), best[2]


def SX(rng, n):
    """Generator: an integer column in [0, 1024) and a uniform noise column."""
    return np.column_stack([rng.integers(0, 1024, n).astype(float), rng.random(n)])


def XM(rng, n):
    """Generator: same as ``SX`` (kept for the original case table)."""
    return np.column_stack([rng.integers(0, 1024, n).astype(float), rng.random(n)])


def P(f, p):
    """Pair constructor kept from the original case table."""
    return (f, p)


C = {}


def add(nm, X, sig, new, truth=None, nz=0.3, null=False, pairs=True, cols=None):
    """Register a W / N / 0 case in the module-level case table ``C``."""
    C[nm] = (mk(X, sig, truth, nz, null), new, cols, pairs)


add("F_W", UP(5), F_sig, F_new, F_sig, cols=[0, 1, 2, 3])
add("F_N", UP(5), lambda X: X[:, 0] * X[:, 1], F_new, lambda X: X[:, 0] * X[:, 1], cols=[0, 1, 2, 3])
add("F_0", UP(5), F_sig, F_new, null=True, pairs=False, cols=[0, 1, 2, 3, 4])


def G_s(X):
    """G target signal: row range over the columns."""
    return X.max(1) - X.min(1)


add("G_W", UP(4), G_s, G_new, G_s, cols=[0, 1, 2, 3])
add("G_N", UP(4), lambda X: X[:, 0] / (X[:, 1] + 0.1), G_new, lambda X: X[:, 0] / (X[:, 1] + 0.1), cols=[0, 1, 2, 3])
add("G_0", UP(4), G_s, G_new, null=True, pairs=False, cols=[0, 1, 2, 3])
add("J_W", UP(4), J_sig, J_new, J_sig, cols=[0, 1, 2, 3])
add("J_N", UP(4), lambda X: X[:, 0] * X[:, 1], J_new, lambda X: X[:, 0] * X[:, 1], cols=[0, 1, 2, 3])
add("J_0", UP(4), J_sig, J_new, null=True, pairs=False, cols=[0, 1, 2, 3])


def K_s(X):
    """K target signal: min of two pair sums."""
    return np.minimum(X[:, 0] + X[:, 1], X[:, 2] + X[:, 3])


add("K_W", UP(4), K_s, K_new, K_s, cols=[0, 1, 2, 3])
add("K_N", UP(4), lambda X: X[:, 0] * X[:, 1], K_new, lambda X: X[:, 0] * X[:, 1], cols=[0, 1, 2, 3])
add("K_0", UP(4), K_s, K_new, null=True, pairs=False, cols=[0, 1, 2, 3])
add("L_W", UP(5), L_sig, L_new, L_sig, cols=[0, 1, 2, 3, 4])
add("L_N", UP(5), lambda X: X[:, 0] * X[:, 1], L_new, lambda X: X[:, 0] * X[:, 1], cols=[0, 1, 2, 3, 4], pairs=False)
add("L_0", UP(5), L_sig, L_new, null=True, pairs=False, cols=[0, 1, 2, 3, 4])
add("AA_W", GM(4), AA_sig, AA_new, AA_sig, cols=[0, 1, 2, 3])
add("AA_N", GM(4), lambda X: X[:, 0] / X[:, 1], AA_new, lambda X: X[:, 0] / X[:, 1], cols=[0, 1, 2, 3])
add("AA_0", GM(4), AA_sig, AA_new, null=True, pairs=False, cols=[0, 1, 2, 3])
add("S_W", SX, lambda X: pc(X[:, 0]), S_new, lambda X: pc(X[:, 0]))
add("S_N", SX, lambda X: np.sqrt(X[:, 0]), S_new, lambda X: np.sqrt(X[:, 0]))
add("S_0", SX, lambda X: pc(X[:, 0]), S_new, null=True, pairs=False)

add("F2_W", UP(5), F_sig, F2_new, F_sig, cols=[0, 1, 2, 3])
add("F2_0", UP(5), F_sig, F2_new, null=True, pairs=False, cols=[0, 1, 2, 3, 4])
add("F2_N", UP(5), lambda X: X[:, 0] * X[:, 1], F2_new, lambda X: X[:, 0] * X[:, 1], cols=[0, 1, 2, 3], pairs=False)
add("JG_W", UP(4), J_sig, G_new, J_sig, cols=[0, 1, 2, 3])
if __name__ == "__main__":
    args = case_args()
    with (scratch_dir("brainstorm") / "results_multi.jsonl").open("a") as out:
        for nm in args.names:
            g, f, cols, pairs = C[nm]
            run_case(nm, g, f, cols or [0, 1], seeds=args.seeds, n=args.n, out=out, pairs=pairs)
