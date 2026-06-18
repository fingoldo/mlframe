"""Bench: MRMR ``bur_lambda`` (MRwMR-BUR unique-relevance bonus, Gao 2022) default 0.0 vs >0 on KNOWN-relevant-set synthetics.

The BUR term adds ``lambda * max(0, I(X;Y) - max_j I(X; X_j))`` to the post-Fleuret MRMR gain (``evaluation.py:713``), boosting a candidate whose
marginal-y relevance cannot be explained by any already-selected feature. The DGP where it should help: a uniquely-relevant MODERATE driver competing for
the next selection slot against a redundant CLUSTER of features that share a stronger common signal -- the cluster members keep out-ranking the lone unique
driver on raw relevance, so the unique driver's recall suffers. BUR rewards exactly the unexplained-relevance feature.

Ground-truth metrics: (a) recall of the KNOWN unique driver(s) in the selected set, (b) selected-set precision, (c) downstream honest-holdout metric.
This harness changes NO default by itself. Run:

    python -m mlframe.feature_selection._benchmarks.bench_bur_lambda_qual22            # 7 seeds
    python -m mlframe.feature_selection._benchmarks.bench_bur_lambda_qual22 --quick    # 5 seeds

REJECTED!=DELETED: re-run to re-validate before reconsidering a flip of the ``bur_lambda`` default.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def _scenarios(seed: int, n: int):
    """{name: (X, y, task, relevant_idx, unique_idx)} where a lone unique driver competes against a redundant cluster."""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple] = {}

    # S1 regression: a redundant cluster of 4 noisy copies of a strong latent z0 + 1 UNIQUE moderate driver u (orthogonal) + pure noise.
    # The cluster shares the strongest signal; u contributes independent variance the cluster cannot explain -> BUR should pull u in.
    z0 = rng.standard_normal(n)
    cluster = np.column_stack([z0 + 0.30 * rng.standard_normal(n) for _ in range(4)])
    u = rng.standard_normal(n)
    noise = rng.standard_normal((n, 4))
    X1 = np.column_stack([cluster, u, noise])  # cols 0..3 cluster, 4 unique, 5..8 noise
    y1 = (1.6 * z0 + 1.1 * u + 0.4 * rng.standard_normal(n)).astype(float)
    out["redundant_cluster_vs_unique_regr"] = (X1, y1, "regression", {0, 1, 2, 3, 4}, {4})

    # S2 classification: 2 redundant copies of latent a + 1 UNIQUE ternary driver b (independent) + high-card / gaussian noise; binary target.
    a = rng.standard_normal(n)
    cl2 = np.column_stack([a + 0.25 * rng.standard_normal(n), a + 0.25 * rng.standard_normal(n)])
    b = rng.integers(0, 3, n).astype(float)
    hc = np.column_stack([rng.integers(0, c, n) for c in (40, 60)]).astype(float)
    gn = rng.standard_normal((n, 3))
    X2 = np.column_stack([cl2, b, hc, gn])  # cols 0,1 cluster, 2 unique, 3.. noise
    logit = 1.5 * a + 1.2 * (b - 1)
    y2 = (logit + 0.6 * rng.standard_normal(n) > 0).astype(int)
    out["redundant_cluster_vs_unique_clf"] = (X2, y2, "classification", {0, 1, 2}, {2})

    # S3 regression HARD: large strong cluster (6 tight copies of z) + a WEAK unique driver u the baseline tends to drop;
    # cluster dominates raw relevance + the per-candidate gain so the lone weak unique driver loses the next-slot race -> BUR's targeted regime.
    z = rng.standard_normal(n)
    cluster3 = np.column_stack([z + 0.15 * rng.standard_normal(n) for _ in range(6)])
    u3 = rng.standard_normal(n)
    noise3 = rng.standard_normal((n, 3))
    X3 = np.column_stack([cluster3, u3, noise3])  # cols 0..5 cluster, 6 unique-weak, 7..9 noise
    y3 = (2.0 * z + 0.55 * u3 + 0.4 * rng.standard_normal(n)).astype(float)
    out["strong_cluster_weak_unique_regr"] = (X3, y3, "regression", set(range(7)), {6})

    return out


_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_no_bur": {"bur_lambda": 0.0},
    "bur_0_5": {"bur_lambda": 0.5},
    "bur_1_0": {"bur_lambda": 1.0},
}


def _eval(X, y, task, relevant, unique, kw, seed):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters import MRMR

    cols = [f"f{i}" for i in range(X.shape[1])]
    Xdf = pd.DataFrame(X, columns=cols)
    strat = y if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(Xdf, y, test_size=0.3, random_state=seed, stratify=strat)
    try:
        sel = MRMR(n_workers=1, verbose=0, fe_max_steps=0, max_runtime_mins=2, **kw)
        sel.fit(Xtr, ytr)
        Xtr_s = sel.transform(Xtr)
        Xte_s = sel.transform(Xte)
        chosen = list(Xtr_s.columns)
    except Exception as exc:
        return {"prec": float("nan"), "rec": float("nan"), "urec": float("nan"), "metric": float("nan"), "err": repr(exc)}
    chosen_raw = {int(c[1:]) for c in chosen if c.startswith("f") and c[1:].isdigit()}
    tp = len(chosen_raw & relevant)
    prec = tp / max(1, len(chosen_raw)) if chosen_raw else 0.0
    rec = tp / max(1, len(relevant))
    urec = len(chosen_raw & unique) / max(1, len(unique))  # recall of the UNIQUE driver -- the BUR-targeted metric
    metric = _downstream(Xtr_s, Xte_s, ytr, yte, task)
    return {"prec": prec, "rec": rec, "urec": urec, "metric": metric, "nsel": len(chosen)}


def _downstream(Xtr_s, Xte_s, ytr, yte, task):
    if Xtr_s.shape[1] == 0:
        return float("nan")
    Xtr_s = np.asarray(Xtr_s); Xte_s = np.asarray(Xte_s)
    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        m = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
        try:
            return float(roc_auc_score(yte, m.predict_proba(Xte_s)[:, 1]))
        except Exception:
            return float("nan")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    m = Ridge().fit(Xtr_s, ytr)
    return float(r2_score(yte, m.predict(Xte_s)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = list(range(5 if args.quick else args.seeds))
    res: dict[str, dict[str, dict[str, list]]] = {}
    for s in seeds:
        for name, (X, y, task, rel, uniq) in _scenarios(s, args.n).items():
            for cfg, kw in _CONFIGS.items():
                r = _eval(X, y, task, rel, uniq, kw, s)
                d = res.setdefault(name, {}).setdefault(cfg, {})
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        d.setdefault(k, []).append(v)

    print(f"\n=== BUR bur_lambda bench (n={args.n}, {len(seeds)} seeds) ===")
    grand = {}
    for scen, by_cfg in res.items():
        base = by_cfg["baseline_no_bur"]
        print(f"\n[{scen}]")
        bm = np.asarray(base["metric"]); bu = np.asarray(base["urec"])
        print(f"  baseline   urec={np.nanmean(bu):.3f}  metric={np.nanmean(bm):.4f}  nsel={np.nanmean(base['nsel']):.1f}")
        for cfg in ("bur_0_5", "bur_1_0"):
            c = by_cfg[cfg]
            cu = np.asarray(c["urec"]); cm = np.asarray(c["metric"])
            paired = [(bv, mv) for bv, mv in zip(bm, cm) if np.isfinite(bv) and np.isfinite(mv)]
            mwins = sum(1 for bv, mv in paired if mv > bv + 1e-6)
            uwins = sum(1 for bv, cv in zip(bu, cu) if cv > bv + 1e-9)
            print(f"  {cfg:8s}  urec={np.nanmean(cu):.3f} (d{np.nanmean(cu)-np.nanmean(bu):+.3f})  "
                  f"metric={np.nanmean(cm):.4f} (d{np.nanmean(cm)-np.nanmean(bm):+.4f})  "
                  f"metric_wins={mwins}/{len(paired)}  urec_wins={uwins}/{len(bu)}")
            g = grand.setdefault(cfg, [0, 0, 0, 0])
            g[0] += mwins; g[1] += len(paired); g[2] += uwins; g[3] += len(bu)
    print("\nAGGREGATE (vs baseline_no_bur):")
    for cfg, (mw, mt, uw, ut) in grand.items():
        print(f"  {cfg}: metric wins {mw}/{mt} | unique-driver recall wins {uw}/{ut}")
    print("FLIP only on a MAJORITY metric win across scenarios x seeds WITHOUT material regression. This harness changes no default.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
