"""Task A bench: should rich binary ops (abs_diff/signed/ratio_abs/hypot) be promoted
to the DEFAULT `minimal` preset, or stay in `medium`?

We bench MRMR with fe_binary_preset="minimal" (6 ops) vs "medium" (10 ops; the
rich ops live there now). Measures BOTH downstream AUC (40% honest holdout, lgbm+logit)
AND MRMR.fit wall-time (the FE pair-search cost scales with operator count).

Beds:
  synth       -- mild bilinear interaction + linear signal (the "synth" baseline)
  hard_synth  -- a*b / a^2 interaction drowned in noise (the prior agent's hard bed)
  nonprod     -- y = -|a-b| (a NON-product proximity target the prior agent flagged as
                 the cleanest win for abs_diff: products cannot linearize it)

Frugal: n_jobs<=2, single bench run; stdout goes to the launching redirection.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False

from mlframe.feature_selection.filters.mrmr import MRMR

RNG = np.random.default_rng(20260604)
N_JOBS = 2


def _to_df(X):
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def make_synth(n=6000, p=30, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # linear signal on x0,x1 + a bilinear interaction x2*x3
    lin = 0.9 * X[:, 0] + 0.7 * X[:, 1]
    inter = 1.1 * X[:, 2] * X[:, 3]
    logit = lin + inter
    p1 = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p1).astype(np.int64)
    return X, y


def make_hard_synth(n=6000, p=60, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # true a*b plus a^2 interaction, weak marginals, lots of noise cols
    inter = 0.9 * X[:, 0] * X[:, 1] + 0.6 * (X[:, 2] ** 2)
    weaklin = 0.3 * X[:, 3]
    logit = inter + weaklin
    p1 = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p1).astype(np.int64)
    return X, y


def make_nonprod(n=6000, p=30, seed=3):
    """y = sigmoid(-|a-b|*k + noise): proximity target. abs_diff carrier;
    a product of a,b cannot linearize |a-b|."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    prox = -np.abs(X[:, 0] - X[:, 1])
    # add a second non-product term: -|c-d|
    prox2 = -np.abs(X[:, 2] - X[:, 3])
    logit = 1.6 * prox + 1.2 * prox2 + 1.0  # offset to balance classes
    p1 = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p1).astype(np.int64)
    return X, y


def downstream_auc(Xtr, ytr, Xte, yte):
    """Return dict of model -> AUC on the holdout. Uses lgbm + logit."""
    out = {}
    # logit (linear -- where rich ops should help most)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0))
    Xte_s = sc.transform(np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0))
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(Xtr_s, ytr)
    out["logit"] = roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1])
    if _HAVE_LGBM:
        gb = LGBMClassifier(n_estimators=150, num_leaves=31, n_jobs=N_JOBS,
                            verbose=-1, random_state=42)
        gb.fit(Xtr, ytr)
        out["lgbm"] = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
    out["mean"] = float(np.mean(list(out.values())))
    return out


def run_bed(name, X, y, preset):
    """Fit MRMR with given binary preset; transform train+holdout; measure AUC+time."""
    n = X.shape[0]
    idx = np.arange(n)
    RNG.shuffle(idx)
    cut = int(n * 0.6)
    tr, te = idx[:cut], idx[cut:]
    Xtr_df, Xte_df = _to_df(X[tr]), _to_df(X[te])
    ytr, yte = y[tr], y[te]

    sel = MRMR(
        fe_binary_preset=preset,
        fe_unary_preset="minimal",
        fe_max_steps=1,
        max_runtime_mins=None,
        verbose=0,
        random_seed=42,
        n_jobs=N_JOBS,
    )
    t0 = time.perf_counter()
    sel.fit(Xtr_df, pd.Series(ytr, name="y"))
    fit_t = time.perf_counter() - t0

    # transform both splits via the fitted selector (engineered cols replayed leak-safe)
    Xtr_t = sel.transform(Xtr_df)
    Xte_t = sel.transform(Xte_df)
    n_eng = len(getattr(sel, "_engineered_features_", []) or [])
    n_sel = Xtr_t.shape[1]

    Xtr_a = np.asarray(Xtr_t, dtype=np.float64)
    Xte_a = np.asarray(Xte_t, dtype=np.float64)
    aucs = downstream_auc(Xtr_a, ytr, Xte_a, yte)
    return {"preset": preset, "fit_t": fit_t, "n_sel": n_sel,
            "n_eng": n_eng, **{f"auc_{k}": v for k, v in aucs.items()}}


def main():
    beds = {
        "synth": make_synth(),
        "hard_synth": make_hard_synth(),
        "nonprod": make_nonprod(),
    }
    print(f"have_lgbm={_HAVE_LGBM} n_jobs={N_JOBS}")
    print(f"{'bed':<12} {'preset':<8} {'fit_s':>7} {'n_sel':>5} {'n_eng':>5} "
          f"{'auc_logit':>9} {'auc_lgbm':>9} {'auc_mean':>9}")
    rows = []
    for bed_name, (X, y) in beds.items():
        print(f"# bed={bed_name} n={X.shape[0]} p={X.shape[1]} prevalence={y.mean():.3f}")
        for preset in ("minimal", "medium"):
            r = run_bed(bed_name, X, y, preset)
            rows.append((bed_name, r))
            print(f"{bed_name:<12} {preset:<8} {r['fit_t']:>7.2f} {r['n_sel']:>5} "
                  f"{r['n_eng']:>5} {r.get('auc_logit',float('nan')):>9.4f} "
                  f"{r.get('auc_lgbm',float('nan')):>9.4f} {r['auc_mean']:>9.4f}")
    # summary deltas
    print("\n=== DELTAS (medium - minimal) ===")
    by_bed = {}
    for bn, r in rows:
        by_bed.setdefault(bn, {})[r["preset"]] = r
    for bn, d in by_bed.items():
        mi, me = d["minimal"], d["medium"]
        dt = me["fit_t"] - mi["fit_t"]
        dt_pct = 100.0 * dt / max(mi["fit_t"], 1e-9)
        d_logit = me.get("auc_logit", np.nan) - mi.get("auc_logit", np.nan)
        d_lgbm = me.get("auc_lgbm", np.nan) - mi.get("auc_lgbm", np.nan)
        d_mean = me["auc_mean"] - mi["auc_mean"]
        print(f"{bn:<12} d_fit={dt:+.2f}s ({dt_pct:+.0f}%)  "
              f"d_logit={d_logit:+.4f}  d_lgbm={d_lgbm:+.4f}  d_mean={d_mean:+.4f}  "
              f"n_eng {mi['n_eng']}->{me['n_eng']}")


if __name__ == "__main__":
    main()
