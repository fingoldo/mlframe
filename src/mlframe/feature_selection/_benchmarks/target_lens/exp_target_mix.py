"""Does mixing the TARGET with features inside MRMR find terms the identity-target search misses? (prototype of the owner's idea, built on the existing selector as a black box)

Arms, each a full MRMR fit whose engineered output feeds the same downstream models:

* ``A`` identity: ``MRMR(X, y)``, the production behaviour.
* ``B`` residual target (boosting): ``r = y - oof_hgb(features of A)``; ``MRMR(X, r)``; features = A union B. The base of the mix is the model's own prediction.
* ``D`` placebo for C: the same bases, each randomly permuted (same marginal, no link to y), ``T = y - beta * perm(u)``: whatever D adds is the price of a second pass with a noisier target, not of the mix.
* ``E`` compute-matched alternative: ``MRMR(X, y)`` with ``fe_max_steps=3`` (more search on the identity target), features = A union E.
* ``C`` feature-base mix: the base is a feature MRMR itself built or kept in A (raw or engineered), ``T_u = y - beta * u`` with beta by OLS, ``MRMR(X, T_u)`` for the top ``N_BASES`` bases by their
  correlation with y; features = A union all C. This varies the base through the transforms the selector already has, as proposed.

The downstream models are ridge and gradient boosting on the union, scored by held-out RMSE; the extra columns an arm adds are counted, and the columns built only from noise are counted as false
positives. Processes (``PROCESSES``) put a weak term under a dominant one (the rank-space masking the residual rescue targets), add a variance-only signal (a heteroscedastic noise scale, which no
point-prediction metric can reward), a regime switch, two dominant terms with a weak interaction, a pure-noise control and a monotone-y control.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_target_mix [seeds] [n_train] [processes comma separated]``.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")

N_TEST = 20000
ARMS_ON = {"B", "C", "D", "E"}  # which of the arms besides A to fit
N_BASES = 3
MRMR_KW = dict(verbose=0, n_jobs=1)


def _cols(n, rng, k=5):
    return pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(1, k + 1)})


def p_masked_interaction(n, rng):
    """A dominant sine hides a weak product of two columns that have no marginal effect."""
    X = _cols(n, rng)
    weak = 0.7 * X.x2 * X.x3
    return X, 10 * np.sin(3 * X.x1) + weak + 0.3 * rng.standard_normal(n), {"x2", "x3"}, weak


def p_two_dominant_weak_pair(n, rng):
    """Two dominant additive terms and a weak interaction of two other columns."""
    X = _cols(n, rng)
    weak = 0.6 * X.x3 * X.x4
    return X, 6 * np.sin(3 * X.x1) + 6 * X.x2**2 + weak + 0.3 * rng.standard_normal(n), {"x3", "x4"}, weak


def p_heteroscedastic(n, rng):
    """Mean from x1*x2; the noise scale from x3 (a signal for the variance only)."""
    X = _cols(n, rng)
    return X, X.x1 * X.x2 + (0.3 + np.abs(X.x3)) * rng.standard_normal(n), {"x1", "x2"}, None


def p_regime(n, rng):
    """One regime follows x1, the other the product x2*x3, selected by the sign of x4."""
    X = _cols(n, rng)
    return X, np.where(X.x4 > 0, 3 * X.x1, 3 * X.x2 * X.x3) + 0.3 * rng.standard_normal(n), {"x1", "x2", "x3", "x4"}, None


def p_noise_control(n, rng):
    """A single linear term and nothing else: nothing may be added."""
    X = _cols(n, rng)
    return X, 3 * X.x1 + 0.5 * rng.standard_normal(n), {"x1"}, None


def p_monotone_control(n, rng):
    """y = exp(0.8 x1 + 0.5 x2): a monotone transform of an additive signal; the rank bins of y do not change."""
    X = _cols(n, rng)
    return X, np.exp(0.8 * X.x1 + 0.5 * X.x2) + 0.01 * rng.standard_normal(n), {"x1", "x2"}, None


def p_masked_ratio(n, rng):
    """A dominant sine hides a weak ratio of two columns."""
    X = _cols(n, rng)
    weak = 0.8 * X.x2**2 / (1.5 + X.x3**2)
    return X, 10 * np.sin(3 * X.x1) + weak + 0.3 * rng.standard_normal(n), {"x2", "x3"}, weak


def p_derived_base(n, rng):
    """A dominant saturating transform of one column hides a weak product of two others (the base is a transform of a column, not a column)."""
    X = _cols(n, rng)
    weak = 0.8 * X.x2 * X.x3
    return X, 8 * np.tanh(2 * X.x1) + weak + 0.3 * rng.standard_normal(n), {"x2", "x3"}, weak


PROCESSES = {
    "masked_ratio": p_masked_ratio,
    "derived_base": p_derived_base,
    "masked_interaction": p_masked_interaction,
    "two_dominant_weak_pair": p_two_dominant_weak_pair,
    "heteroscedastic": p_heteroscedastic,
    "regime": p_regime,
    "noise_control": p_noise_control,
    "monotone_control": p_monotone_control,
}


def _models():
    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0),
    }


def _fit_mrmr(X, target, seed, **extra):
    from mlframe.feature_selection.filters.mrmr import MRMR

    fs = MRMR(random_seed=seed, **MRMR_KW, **extra)
    fs.fit(X, pd.Series(np.asarray(target), name="y"))
    return fs


def _frame(fs, X):
    out = fs.transform(X)
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out, columns=[str(c) for c in fs.get_feature_names_out()])


def _union(frames):
    cols = {}
    for f in frames:
        for c in f.columns:
            cols.setdefault(str(c), np.asarray(f[c], dtype=np.float64))
    return pd.DataFrame(cols)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _oof_pred(F, y, seed):
    pred = np.zeros(len(y))
    for tr, te in KFold(4, shuffle=True, random_state=seed).split(F):
        m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=0).fit(F.iloc[tr], y[tr])
        pred[te] = m.predict(F.iloc[te])
    return pred


def run(process: str, n_train: int, seed: int) -> dict:
    """One draw of one process: the three arms and the downstream scores."""
    rng = np.random.default_rng(1000 + seed)
    X_all, y_all, truth, weak_all = PROCESSES[process](n_train + N_TEST, rng)
    weak_te = None if weak_all is None else np.asarray(weak_all)[n_train:]
    Xtr, ytr = X_all.iloc[:n_train].reset_index(drop=True), np.asarray(y_all)[:n_train]
    Xte, yte = X_all.iloc[n_train:].reset_index(drop=True), np.asarray(y_all)[n_train:]
    t0 = time.perf_counter()
    fa = _fit_mrmr(Xtr, ytr, seed)
    A_tr, A_te = _frame(fa, Xtr), _frame(fa, Xte)
    t_a = time.perf_counter() - t0
    arms = {"A": (A_tr, A_te)}
    # B: residual of a model on A's features
    t0 = time.perf_counter()
    if "B" in ARMS_ON:
        r = ytr - _oof_pred(A_tr, ytr, seed)
        fb = _fit_mrmr(Xtr, r, seed)
        B_tr, B_te = _frame(fb, Xtr), _frame(fb, Xte)
        arms["A+B"] = (_union([A_tr, B_tr]), _union([A_te, B_te]))
    t_b = time.perf_counter() - t0
    # C: bases from the features of A
    t0 = time.perf_counter()
    corr = sorted(((abs(np.corrcoef(A_tr[c], ytr)[0, 1]) if A_tr[c].std() > 0 else 0.0, c) for c in A_tr.columns), reverse=True)[:N_BASES]
    c_tr, c_te = [A_tr], [A_te]
    d_tr, d_te = [A_tr], [A_te]
    for _, c in corr:
        u = np.asarray(A_tr[c], dtype=np.float64)
        beta = float(np.dot(u - u.mean(), ytr - ytr.mean()) / max(np.dot(u - u.mean(), u - u.mean()), 1e-12))
        if "C" in ARMS_ON:
            fc = _fit_mrmr(Xtr, ytr - beta * u, seed)
            c_tr.append(_frame(fc, Xtr))
            c_te.append(_frame(fc, Xte))
        if "D" in ARMS_ON:
            fd = _fit_mrmr(Xtr, ytr - beta * rng.permutation(u), seed)
            d_tr.append(_frame(fd, Xtr))
            d_te.append(_frame(fd, Xte))
    if "C" in ARMS_ON:
        arms["A+C"] = (_union(c_tr), _union(c_te))
    if "D" in ARMS_ON:
        arms["A+D"] = (_union(d_tr), _union(d_te))
    t_c = time.perf_counter() - t0
    if "E" in ARMS_ON:
        fe = _fit_mrmr(Xtr, ytr, seed, fe_max_steps=3)
        arms["A+E"] = (_union([A_tr, _frame(fe, Xtr)]), _union([A_te, _frame(fe, Xte)]))
    res = {"time": {"A": t_a, "B": t_b, "C": t_c}}
    for arm, (tr, te) in arms.items():
        row = {"n_feat": tr.shape[1], "n_extra": tr.shape[1] - A_tr.shape[1]}
        for name, make in _models().items():
            sc = StandardScaler().fit(tr)
            m = make().fit(sc.transform(tr), ytr)
            row[name] = _rmse(m.predict(sc.transform(te)), yte)
        row["recovery"] = float("nan") if weak_te is None else max([abs(np.corrcoef(te[c], weak_te)[0, 1]) for c in te.columns if te[c].std() > 0] or [0.0])
        row["covers"] = int(truth <= {t for c in tr.columns for t in str(c).replace("(", " ").replace(")", " ").replace(",", " ").split()})
        arms[arm] = row
    res["arms"] = arms
    return res


def main() -> None:
    """Run the processes and print, per process, the median downstream RMSE of each arm and the paired differences against A."""
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PROCESSES)
    global ARMS_ON
    ARMS_ON = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else {"B", "C", "D", "E"}
    for process in which:
        runs = []
        for s in range(seeds):
            runs.append(run(process, n_train, s))
            print(f"  {process} seed {s} done ({sum(runs[-1]['time'].values()):.0f}s)", flush=True)
        print(f"\n{process}  (n_train={n_train}, {seeds} seeds; {PROCESSES[process].__doc__})")
        for model in ("ridge", "hgb"):
            base = np.array([r["arms"]["A"][model] for r in runs])
            parts = [f"A {np.median(base):.3f}"]
            for arm in [a for a in ("A+B", "A+C", "A+D", "A+E") if a in runs[0]["arms"]]:
                v = np.array([r["arms"][arm][model] for r in runs])
                d = v - base
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                parts.append(f"{arm} {np.median(v):.3f} (paired {np.median(d) / np.median(base):+.1%}, {np.median(d) / se if se else float('nan'):+.1f} se)")
            print(f"  {model:5} " + "   ".join(parts))
        print("  extra columns per arm (median): " + ", ".join(f"{arm} {np.median([r['arms'][arm]['n_extra'] for r in runs]):.0f}" for arm in [a for a in ("A+B", "A+C", "A+D", "A+E") if a in runs[0]["arms"]]) + "; time (median s): " + ", ".join(f"{k} {np.median([r['time'][k] for r in runs]):.1f}" for k in ("A", "B", "C")))
        print("  truth operands covered: " + ", ".join(f"{arm} {sum(r['arms'][arm]['covers'] for r in runs)}/{len(runs)}" for arm in [a for a in ("A", "A+B", "A+C", "A+D", "A+E") if a in runs[0]["arms"]]), flush=True)
        if runs[0]["arms"]["A"]["recovery"] == runs[0]["arms"]["A"]["recovery"]:
            print("  recovery of the weak term, max |corr| of any column with it on the held-out rows (median): " + ", ".join(f"{arm} {np.median([r['arms'][arm]['recovery'] for r in runs]):.2f}" for arm in [a for a in ("A", "A+B", "A+C", "A+D", "A+E") if a in runs[0]["arms"]]), flush=True)


if __name__ == "__main__":
    main()
