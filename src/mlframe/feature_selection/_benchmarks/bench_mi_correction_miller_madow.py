"""Bench: ``MRMR(mi_correction='miller_madow')`` vs the plug-in default on small-n high-cardinality-noise synthetics with a KNOWN relevant set.

Ground truth: the plug-in MI ``I(X;Y)=H(X)+H(Y)-H(X,Y)`` carries a positive finite-sample bias that scales with the OCCUPIED-bin product, so a high-cardinality
NOISE feature gets an inflated relevance and can out-rank a low-cardinality TRUE-relevant feature at small n. Miller-Madow subtracts the closed-form bias
``(k_x-1)(k_y-1)/(2n)``. We measure (a) selected-set precision/recall against the KNOWN relevant set and (b) downstream honest-holdout metric.

This harness changes NO default by itself -- it prints/saves deltas so the flip decision is made on numbers. Run:

    python -m mlframe.feature_selection._benchmarks.bench_mi_correction_miller_madow            # 7 seeds
    python -m mlframe.feature_selection._benchmarks.bench_mi_correction_miller_madow --quick    # 3 seeds

VERDICT (qual-21, n=600, 7 seeds, 2 scenarios -> tests/perf/results/mi_correction_mm.json): NO DEFAULT FLIP.
  lowcard_signal_vs_hicard_noise : precision 1.000->1.000, recall 1.000->1.000, downstream R^2 0.9154->0.9155 (+0.0001), 3/7 metric cells.
  ternary_signal_classif         : precision 0.667->0.667, recall 1.000->1.000, downstream AUC 0.9447->0.9447 (+0.0000), 0/7 metric cells.
  AGGREGATE 3/14 seed-scenario metric cells -- far below majority. Selection BIT-IDENTICAL on every cell.
Root cause (confirmed, not hypothesised): the default ``nbins_strategy='mdlp'`` (supervised discretization) collapses pure-noise high-cardinality columns to 1-2 occupied
bins, so ``k_x<=1`` and the Miller-Madow bias term ``(k_x-1)(k_y-1)/(2n)`` is already ~0; AND the default permutation null-debias screen subtracts the EMPIRICAL plug-in
bias non-parametrically (the MM bias term is constant across y-permutations -> cancels in ``observed - null_mean``). Both production defaults pre-empt exactly the
small-sample plug-in inflation MM targets. The kernel-level correction is REAL (a 60-level noise feature vs binary y: plug-in MI 0.088 -> MM 0.015, 0.074 nats removed,
pinned in test_biz_val_filters_mrmr_mi_correction.py) but does not survive end-to-end on the production path. Kept as an OPT-IN (``mi_correction='miller_madow'``) for the
legacy fixed-bin / no-permutation-screen regime where it is the only bias defense.

REJECTED!=DELETED: re-run this to re-validate before reconsidering a flip of ``mi_correction``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def _scenarios(seed: int, n: int):
    """{name: (X, y, task, relevant_idx)} where MM should help: low-card true signal buried under high-card noise at small n."""
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, np.ndarray, str, set]] = {}

    # S1 regression: 3 binary true drivers + 5 high-cardinality (40..80 level) integer noise + 4 gaussian noise. Small n so the plug-in bias bites.
    p_noise_hc = [40, 55, 70, 80, 50]
    xt = rng.integers(0, 2, (n, 3)).astype(float)
    hc = np.column_stack([rng.integers(0, c, n) for c in p_noise_hc]).astype(float)
    gn = rng.standard_normal((n, 4))
    X1 = np.column_stack([xt, hc, gn])
    y1 = (xt @ np.array([2.0, -1.5, 1.0]) + 0.4 * rng.standard_normal(n)).astype(float)
    out["lowcard_signal_vs_hicard_noise"] = (X1, y1, "regression", {0, 1, 2})

    # S2 classification: 2 ternary true drivers + high-card noise; binary target. MM on relevance ranking should keep precision high.
    xa = rng.integers(0, 3, n)
    xb = rng.integers(0, 3, n)
    hc2 = np.column_stack([rng.integers(0, c, n) for c in (60, 75, 45, 90)]).astype(float)
    gn2 = rng.standard_normal((n, 3))
    X2 = np.column_stack([xa.astype(float), xb.astype(float), hc2, gn2])
    logit = 1.2 * (xa - 1) - 1.0 * (xb - 1)
    y2 = (logit + 0.6 * rng.standard_normal(n) > 0).astype(int)
    out["ternary_signal_classif"] = (X2, y2, "classification", {0, 1})
    return out


_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_plugin": {},
    "miller_madow": {"mi_correction": "miller_madow"},
}


def _eval(X, y, task, relevant, kw, seed):
    """Fit MRMR(kw) on train, score selected-set precision/recall vs known relevant set + downstream holdout metric."""
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
    except Exception as exc:  # harness must not die on one config
        return {"prec": float("nan"), "rec": float("nan"), "metric": float("nan"), "err": repr(exc)}
    # Precision/recall on the RAW features only (engineered names carry no ground-truth index); raw true drivers are f<relevant>.
    chosen_raw = {int(c[1:]) for c in chosen if c.startswith("f") and c[1:].isdigit()}
    tp = len(chosen_raw & relevant)
    prec = tp / max(1, len(chosen_raw)) if chosen_raw else 0.0
    rec = tp / max(1, len(relevant))
    metric = _downstream(Xtr_s, Xte_s, ytr, yte, task)
    return {"prec": prec, "rec": rec, "metric": metric, "nsel": len(chosen)}


def _downstream(Xtr_s, Xte_s, ytr, yte, task):
    import numpy as np

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
    ap.add_argument("--n", type=int, default=600, help="small n so the plug-in bias bites")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = list(range(3 if args.quick else args.seeds))
    res: dict[str, dict[str, dict[str, list]]] = {}
    for s in seeds:
        for name, (X, y, task, rel) in _scenarios(s, args.n).items():
            for cfg, kw in _CONFIGS.items():
                r = _eval(X, y, task, rel, kw, s)
                d = res.setdefault(name, {}).setdefault(cfg, {})
                for k, v in r.items():
                    if isinstance(v, (int, float)):
                        d.setdefault(k, []).append(v)

    print(f"\n=== Miller-Madow mi_correction bench (n={args.n}, {len(seeds)} seeds) ===")
    win_cells = total_cells = 0
    for scen, by_cfg in res.items():
        b = by_cfg["baseline_plugin"]; m = by_cfg["miller_madow"]
        bp = float(np.nanmean(b["prec"])); mp = float(np.nanmean(m["prec"]))
        br = float(np.nanmean(b["rec"])); mr = float(np.nanmean(m["rec"]))
        bm = float(np.nanmean(b["metric"])); mm = float(np.nanmean(m["metric"]))
        print(f"\n[{scen}]")
        print(f"  precision  plugin={bp:.3f}  mm={mp:.3f}  (delta {mp-bp:+.3f})")
        print(f"  recall     plugin={br:.3f}  mm={mr:.3f}  (delta {mr-br:+.3f})")
        print(f"  downstream plugin={bm:.4f}  mm={mm:.4f}  (delta {mm-bm:+.4f})")
        # Per-seed paired metric win count (the majority test).
        bmv = np.asarray(b["metric"]); mmv = np.asarray(m["metric"])
        paired = [(bv, mv) for bv, mv in zip(bmv, mmv) if np.isfinite(bv) and np.isfinite(mv)]
        w = sum(1 for bv, mv in paired if mv > bv + 1e-6)
        print(f"  metric per-seed wins: mm > plugin in {w}/{len(paired)}")
        win_cells += w; total_cells += len(paired)
    print(f"\nAGGREGATE metric wins (mm > plugin): {win_cells}/{total_cells} seed-scenario cells")
    print("FLIP only on a MAJORITY win across scenarios x seeds WITHOUT material regression. This harness changes no default.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
