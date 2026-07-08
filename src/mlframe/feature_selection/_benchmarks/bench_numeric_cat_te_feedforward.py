"""Quality benchmark: do binned-numeric cat crosses / target-encoding / TE->hermite feedforward
help a LINEAR downstream beyond the existing numeric-pair FE route?

Answers the two open questions:
  H1  include_numeric: bin numeric factors -> temp cats -> factorize cross (+ target encoding).
  H2  TE feedforward : apply an orth-poly (hermite) basis on top of the target-encoded column.

Everything is measured on an HONEST train/test split (test = OOS, never seen at fit / encode time):
target encoding is OOF on train, and the train-fitted cell-means are replayed on test (leakage-free),
exactly mirroring the production transform() contract. Downstream = LogisticRegression (the linear
case where pair interactions must be made explicit; trees recover them natively so they are not the
target audience -- see the project's poly-FE-for-trees null result).

Run:  python -m mlframe.feature_selection._benchmarks.bench_numeric_cat_te_feedforward
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermevander
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

N_BINS = 10
SEEDS = (0, 1, 2, 3, 4)


def _make_data(n: int, seed: int, kind: str = "axis") -> tuple[pd.DataFrame, np.ndarray]:
    """2-D interaction signal + weak linear main effect + 8 noise cols.

    ``kind="axis"``: axis-aligned threshold ``y ~ sign((x0-0.5)*(x1-0.5))`` -- the case a per-axis quantile
    cross captures EXACTLY (favourable to binning).
    ``kind="rotated"``: the SAME quadrant signal rotated 45 deg, so the decision boundary is diagonal in
    (x0, x1) and a per-axis bin grid must staircase-approximate it (favourable to the smooth product term).
    Both marginals are ~uninformative; only a cross / product / target-encoded joint recovers the signal.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    x_lin = rng.normal(0, 1, n)  # weak honest linear main effect
    noise = rng.normal(0, 1, (n, 8))
    if kind == "rotated":
        u = (x0 - 0.5 + x1 - 0.5) / np.sqrt(2.0)
        v = (x0 - 0.5 - (x1 - 0.5)) / np.sqrt(2.0)
        interaction = np.sign(u * v)
    else:
        interaction = np.sign((x0 - 0.5) * (x1 - 0.5))
    if kind == "modulated":
        # The (x0,x1) cross-cell MODULATES the sign of the x_lin main effect: the model needs the
        # cross-cell TARGET STAT interacted with x_lin (TE-as-operand feedforward), not the cross alone.
        logit = 2.6 * interaction * x_lin
    else:
        logit = 3.0 * interaction + 0.6 * x_lin
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(np.int64)
    cols = {"x0": x0, "x1": x1, "x_lin": x_lin}
    for j in range(8):
        cols[f"noise{j}"] = noise[:, j]
    return pd.DataFrame(cols), y


def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Inner quantile edges fitted on TRAIN only; reused to bin test (leakage-free)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs)
    return np.unique(edges)


def _apply_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(x, edges).astype(np.int64)


def _oof_target_encode(codes_tr: np.ndarray, y_tr: np.ndarray, n_folds: int, smoothing: float, seed: int):
    """Leak-safe OOF target encoding of a joint cell code on train; returns (te_train, cell_means_global).

    cell_means_global is the full-train shrunk per-cell mean used to encode the held-out test set.
    """
    global_mean = float(y_tr.mean())
    n = len(y_tr)
    te = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, n)
    for f in range(n_folds):
        oof = folds == f
        infold = ~oof
        df = pd.DataFrame({"c": codes_tr[infold], "y": y_tr[infold]})
        agg = df.groupby("c")["y"].agg(["mean", "count"])
        shrunk = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        lut = shrunk.to_dict()
        te[oof] = [lut.get(c, global_mean) for c in codes_tr[oof]]
    # Full-train table for test replay
    df = pd.DataFrame({"c": codes_tr, "y": y_tr})
    agg = df.groupby("c")["y"].agg(["mean", "count"])
    shrunk = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    cell_means_global = shrunk.to_dict()
    return te, cell_means_global, global_mean


def _logreg_auc(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def _rf_auc(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> float:
    """Tree downstream context: trees recover axis-aligned interactions natively (no FE needed)."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def _onehot(codes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(codes), n_classes), dtype=np.float64)
    out[np.arange(len(codes)), np.clip(codes, 0, n_classes - 1)] = 1.0
    return out


def run_one(n: int, seed: int, kind: str = "axis") -> dict:
    df, y = _make_data(n, seed, kind=kind)
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(n)
    cut = n // 2
    tr, te = idx[:cut], idx[cut:]
    Xtr_df, Xte_df = df.iloc[tr], df.iloc[te]
    ytr, yte = y[tr], y[te]

    raw_tr = Xtr_df.to_numpy()
    raw_te = Xte_df.to_numpy()

    # --- bin x0, x1 on TRAIN edges ---
    e0 = _quantile_edges(Xtr_df["x0"].to_numpy(), N_BINS)
    e1 = _quantile_edges(Xtr_df["x1"].to_numpy(), N_BINS)
    c0_tr, c1_tr = _apply_edges(Xtr_df["x0"].to_numpy(), e0), _apply_edges(Xtr_df["x1"].to_numpy(), e1)
    c0_te, c1_te = _apply_edges(Xte_df["x0"].to_numpy(), e0), _apply_edges(Xte_df["x1"].to_numpy(), e1)
    nb0, nb1 = c0_tr.max() + 1, c1_tr.max() + 1
    cross_tr = c0_tr + c1_tr * nb0
    cross_te = c0_te + c1_te * nb0
    n_cells = int(nb0 * nb1)

    # --- target encoding of the cross (leak-safe) ---
    te_tr, cell_means, gmean = _oof_target_encode(cross_tr, ytr, n_folds=5, smoothing=10.0, seed=seed)
    te_te = np.array([cell_means.get(int(c), gmean) for c in cross_te], dtype=np.float64)

    res = {}

    # 0) baseline: raw only
    res["raw"] = _logreg_auc(raw_tr, ytr, raw_te, yte)

    # 1) existing numeric route proxy: raw + explicit product mul(x0,x1) (what hermite-pair FE recovers)
    mul_tr = (Xtr_df["x0"].to_numpy() * Xtr_df["x1"].to_numpy())[:, None]
    mul_te = (Xte_df["x0"].to_numpy() * Xte_df["x1"].to_numpy())[:, None]
    res["raw+mul(x0,x1)"] = _logreg_auc(np.hstack([raw_tr, mul_tr]), ytr, np.hstack([raw_te, mul_te]), yte)

    # 2) include_numeric factorize cross, one-hot (linear can read the joint cell)
    oh_tr, oh_te = _onehot(cross_tr, n_cells), _onehot(cross_te, n_cells)
    res["raw+cross_onehot"] = _logreg_auc(np.hstack([raw_tr, oh_tr]), ytr, np.hstack([raw_te, oh_te]), yte)

    # 3) include_numeric + target encoding (single numeric col)
    res["raw+cross_TE"] = _logreg_auc(np.hstack([raw_tr, te_tr[:, None]]), ytr, np.hstack([raw_te, te_te[:, None]]), yte)

    # 4) TE feedforward: orth-poly (HermiteE) basis of degree 4 on the standardized TE column
    def _herm_basis(v_tr, v_te, deg=4):
        s = StandardScaler()
        z_tr = s.fit_transform(v_tr[:, None])[:, 0]
        z_te = s.transform(v_te[:, None])[:, 0]
        H_tr = hermevander(z_tr, deg)[:, 1:]  # drop constant
        H_te = hermevander(z_te, deg)[:, 1:]
        return H_tr, H_te

    h_tr, h_te = _herm_basis(te_tr, te_te, deg=4)
    res["raw+cross_TE+hermite(TE)"] = _logreg_auc(np.hstack([raw_tr, te_tr[:, None], h_tr]), ytr, np.hstack([raw_te, te_te[:, None], h_te]), yte)

    # 5) TE-as-operand feedforward: feed TE forward as an operand into a pair interaction mul(TE, x_lin).
    #    This is the strongest fair form of "pass target stats into further FE" -- it only pays off when the
    #    cross-cell modulates a third feature. Centre TE on its train mean so the product has the right sign.
    te_c_tr = te_tr - te_tr.mean()
    te_c_te = te_te - te_tr.mean()
    mte_tr = (te_c_tr * Xtr_df["x_lin"].to_numpy())[:, None]
    mte_te = (te_c_te * Xte_df["x_lin"].to_numpy())[:, None]
    res["raw+cross_TE+mul(TE,x_lin)"] = _logreg_auc(np.hstack([raw_tr, te_tr[:, None], mte_tr]), ytr, np.hstack([raw_te, te_te[:, None], mte_te]), yte)

    # Tree-downstream context (per project: trees recover axis-aligned interactions natively -> expect no lift)
    res["[tree] raw"] = _rf_auc(raw_tr, ytr, raw_te, yte)
    res["[tree] raw+cross_TE"] = _rf_auc(np.hstack([raw_tr, te_tr[:, None]]), ytr, np.hstack([raw_te, te_te[:, None]]), yte)
    return res


def _report(rows: list, n: int, kind: str, elapsed: float):
    keys = list(rows[0].keys())
    print(f"\n=== scenario={kind!r}  honest OOS test AUC, n={n}, {len(SEEDS)} seeds, {elapsed:.1f}s ===\n")
    print(f"{'variant':<30} {'mean AUC':>10} {'std':>8}   per-seed")
    base = np.array([r["raw"] for r in rows])
    for k in keys:
        vals = np.array([r[k] for r in rows])
        delta = vals.mean() - base.mean()
        dstr = "" if k == "raw" else f"  (delta vs raw {delta:+.4f})"
        print(f"{k:<30} {vals.mean():>10.4f} {vals.std():>8.4f}   {np.round(vals,3).tolist()}{dstr}")
    mul = np.array([r["raw+mul(x0,x1)"] for r in rows]).mean()
    te = np.array([r["raw+cross_TE"] for r in rows]).mean()
    the = np.array([r["raw+cross_TE+hermite(TE)"] for r in rows]).mean()
    tem = np.array([r["raw+cross_TE+mul(TE,x_lin)"] for r in rows]).mean()
    t_raw = np.array([r["[tree] raw"] for r in rows]).mean()
    t_te = np.array([r["[tree] raw+cross_TE"] for r in rows]).mean()
    print()
    print(f"H1  [linear] cross_TE vs existing mul(x0,x1) route: {te:.4f} vs {mul:.4f}  (d {te-mul:+.4f})")
    print(f"H2a [linear] hermite(TE) feedforward vs TE alone:   {the:.4f} vs {te:.4f}  (d {the-te:+.4f})")
    print(f"H2b [linear] mul(TE,x_lin) operand feedforward:     {tem:.4f} vs {te:.4f}  (d {tem-te:+.4f})")
    print(f"    [tree]   cross_TE vs raw:                       {t_te:.4f} vs {t_raw:.4f}  (d {t_te-t_raw:+.4f})")


def main():
    n = 8000
    for kind in ("axis", "rotated", "modulated"):
        t0 = time.perf_counter()
        rows = [run_one(n, s, kind=kind) for s in SEEDS]
        _report(rows, n, kind, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
