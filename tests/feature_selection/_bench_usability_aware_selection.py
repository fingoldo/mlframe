"""Reproducible prototype for the usability-aware (multi-list) MRMR selection design.

See MRMR_USABILITY_AWARE_SELECTION_DESIGN.md. Builds the full all-pairs-all-forms FE
candidate pool on the F2 target, then runs a greedy selection with relevance =
(1-w)*(MI - redundancy) + w*|held-out partial corr with the RESIDUAL after selected|.
Measured: w=0 (pure MI) -> linear MAE 0.134 (no (c,d) feature); w=0.8 -> 0.052 (the
irreducible floor), selecting genuine mul(log(c),...) interaction forms. NOT a pytest
(standalone bench: run with the repo PYTHONPATH). The residual-partial-corr usability
target is the key -- an R^2-based term is dominated by the a**2/b heavy tail and fails.
"""

import numpy as np, itertools, warnings

warnings.filterwarnings("ignore")
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
from mlframe.feature_selection.filters.feature_engineering import smart_log
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error as MAE

np.random.seed(0)
n = 60000
a = np.random.rand(n)
b = np.random.rand(n)
c = np.random.rand(n)
d = np.random.rand(n)
e = np.random.rand(n)
f = np.random.rand(n)
y = 0.2 * a**2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
rng = np.random.default_rng(0)
idx = rng.permutation(n)
tr = idx[: int(0.7 * n)]
va = idx[int(0.7 * n) : int(0.85 * n)]
te = idx[int(0.85 * n) :]
raw = {"a": a, "b": b, "c": c, "d": d, "e": e}
UN = {
    "id": lambda x: x,
    "sqr": lambda x: x**2,
    "log": smart_log,
    "sin": np.sin,
    "exp": np.exp,
    "cbrt": np.cbrt,
    "neg": np.negative,
    "abs": np.abs,
    "rec": lambda x: np.power(x, -1.0),
}
BI = {"add": lambda u, v: u + v, "sub": lambda u, v: u - v, "mul": lambda u, v: u * v, "div": lambda u, v: u / v}


def scrub(v):
    """Helper that scrub."""
    return np.nan_to_num(np.asarray(v, float), nan=0, posinf=0, neginf=0)


def mi(v):
    """Quantile-bin v and y then return their plug-in mutual information."""
    return float(_cmi_from_binned(_quantile_bin(scrub(v), 12), _quantile_bin(y, 12), None))


pool = {}
for k, v in raw.items():
    pool[k] = scrub(v)
for (n1, v1), (n2, v2) in itertools.combinations(raw.items(), 2):
    for ua, ub in itertools.product(UN, UN):
        for bn, bf in BI.items():
            val = scrub(bf(UN[ua](v1), UN[ub](v2)))
            if np.std(val) < 1e-9:
                continue
            if mi(val) < 0.05:
                continue
            pool[f"{bn}({ua}({n1}),{ub}({n2}))"] = val
names = list(pool)
V = np.column_stack([pool[k] for k in names])
MIs = np.array([mi(pool[k]) for k in names])
print(f"pool={len(names)}")


def lin_fit_resid(cols, split):
    # residual of y after a linear fit on cols (fit on tr), evaluated on split
    """Lin fit resid."""
    if not cols:
        return y[split] - y[tr].mean()
    m = make_pipeline(StandardScaler(), LinearRegression()).fit(V[:, cols][tr], y[tr])
    return y[split] - m.predict(V[:, cols][split])


def lin_mae(cols):
    """Lin mae."""
    m = make_pipeline(StandardScaler(), LinearRegression()).fit(V[:, cols][tr], y[tr])
    return MAE(y[te], m.predict(V[:, cols][te]))


def abscorr(u, v):
    """Helper that abscorr."""
    if np.std(u) < 1e-12 or np.std(v) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(r) if np.isfinite(r) else 0.0


def greedy(K=6, w=0.0):
    """Helper that greedy."""
    sel = []
    for _ in range(K):
        r_va = lin_fit_resid(sel, va)  # held-out residual after selected (REMOVES a2/b dominance)
        lin_fit_resid(sel, tr)
        best, bi = -1e18, -1
        for i in range(len(names)):
            if i in sel:
                continue
            red = max((abscorr(V[:, i][tr], V[:, j][tr]) for j in sel), default=0.0)
            use = abscorr(V[:, i][va], r_va)  # held-out |partial corr| with residual = LINEAR usability
            if w > 0:
                score = (1 - w) * (MIs[i] - 0.5 * red) + w * use
            else:
                score = MIs[i] - 0.7 * red
            if score > best:
                best, bi = score, i
        if bi < 0:
            break
        sel.append(bi)
    return sel


for w in (0.0, 0.5, 0.8):
    sel = greedy(6, w)
    print(f"w={w}: linMAE={lin_mae(sel):.4f} sel={[names[i] for i in sel]}")
