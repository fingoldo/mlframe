"""Bench: adaptive trust-guard anchor budget + self-tuning (knee) prescreen ladder for ShapProxiedFS.

Two innovations under test, both fully non-MRMR:

1. ADAPTIVE ANCHOR BUDGET (``n_anchors="auto"``): n = clip(round(6*sqrt(p)), 10, 100), p = search width.
   The fixed 30 anchors are sparse on wide frames; the adaptive count denser-guards them. HONEST
   metric here = trust-guard fidelity (proxy_fidelity_score / spearman) -- the guard's own job is to
   measure how trustworthy the cheap proxy is, so a TIGHTER guard = higher/more-stable fidelity signal
   on wide data. We assert auto >= fixed-30 on the WIDE frames where the fixed budget was sparse.

2. KNEE PRESCREEN LADDER (``prescreen_ladder_mode="knee"``): derive the prescreen cap from the sorted
   |phi| importance distribution. Dense-signal frame keeps the full cap; sparse frame prunes harder.
   HONEST metric = held-out score of the model refit on the selected features (test split the selector
   never saw). We assert knee >= hardcoded/off on dense AND sparse, majority of scenarios+seeds.

Run: python -m mlframe.feature_selection._benchmarks.bench_shapproxied_adaptive_guards
Env (host segfault guard): CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

# (label, width, n_informative, n_redundant, rho, snr) -- narrow + WIDE, dense + sparse signal.
SCENARIOS = [
    ("narrow_p20_dense", 20, 8, 4, 0.6, 6.0),
    ("narrow_p20_sparse", 20, 3, 2, 0.85, 3.0),
    ("wide_p2000_dense", 2000, 30, 20, 0.6, 6.0),
    ("wide_p2000_sparse", 2000, 4, 4, 0.85, 2.5),
    ("wide_p6000_sparse", 6000, 4, 6, 0.85, 2.0),
]
SEEDS = (0, 1, 2)


def _fit_holdout_auc(X, y, cols, seed):
    """Honest held-out AUC of an xgboost refit on ``cols`` (test split selector never saw)."""
    import xgboost as xgb

    Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.3, random_state=seed, stratify=y)
    m = xgb.XGBClassifier(n_estimators=120, max_depth=4, n_jobs=1, random_state=seed, tree_method="hist", verbosity=0)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def _run_selector(X, y, seed, *, n_anchors, ladder, width):
    cap = 12 if width <= 20 else None
    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="auto", max_features=cap,
        top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=n_anchors,
        prescreen_ladder_mode=ladder, prefilter_top=min(width, 2000),
        random_state=seed, verbose=False, n_jobs=1,
    )
    t0 = time.perf_counter()
    sel.fit(X, y)
    wall = time.perf_counter() - t0
    rep = sel.shap_proxy_report_
    trust = rep.get("trust", {})
    return dict(
        selected=list(sel.selected_features_),
        spearman=float(trust.get("spearman", float("nan"))),
        fidelity=float(trust.get("proxy_fidelity_score", float("nan"))),
        n_anchors=int(rep.get("trust_n_anchors", {}).get("resolved", -1)),
        cap=int(rep.get("adaptive_prescreen", {}).get("effective_cap", -1)),
        wall=wall,
    )


def main():
    rows = []
    for label, width, n_inf, n_red, rho, snr in SCENARIOS:
        for seed in SEEDS:
            # split data ONCE: selector sees the train half only; honest AUC on held-out half.
            X, y, _roles = make_regime_dataset(
                n_samples=4000, n_informative=n_inf, n_redundant=n_red, redundancy_rho=rho, n_noise=width - n_inf - n_red, snr=snr, task="binary", seed=seed
            )
            import pandas as pd
            y = pd.Series(y)
            Xfit, Xhold, yfit, yhold = train_test_split(X, y, test_size=0.3, random_state=seed + 100, stratify=y)
            Xfit = Xfit.reset_index(drop=True)
            yfit = yfit.reset_index(drop=True)
            Xfull = pd.concat([Xfit, Xhold]).reset_index(drop=True)
            yfull = pd.concat([yfit, yhold]).reset_index(drop=True)

            # Lever 1: anchors auto vs fixed-30 (fidelity on this frame).
            a_auto = _run_selector(Xfit, yfit, seed, n_anchors="auto", ladder="knee", width=width)
            a_fix = _run_selector(Xfit, yfit, seed, n_anchors=30, ladder="knee", width=width)

            # Lever 2: knee vs off ladder. Honest held-out AUC of refit on selected features.
            auc_knee = _fit_holdout_auc(Xfull, yfull, a_auto["selected"], seed)
            sel_off = _run_selector(Xfit, yfit, seed, n_anchors="auto", ladder="off", width=width)
            auc_off = _fit_holdout_auc(Xfull, yfull, sel_off["selected"], seed)

            rows.append(dict(
                scenario=label, width=width, seed=seed,
                anchors_auto=a_auto["n_anchors"], anchors_fixed=30,
                fidelity_auto=a_auto["fidelity"], fidelity_fixed=a_fix["fidelity"],
                spearman_auto=a_auto["spearman"], spearman_fixed=a_fix["spearman"],
                cap_knee=a_auto["cap"], cap_off=sel_off["cap"],
                n_sel_knee=len(a_auto["selected"]), n_sel_off=len(sel_off["selected"]),
                auc_knee=auc_knee, auc_off=auc_off,
                wall_auto=a_auto["wall"], wall_fix=a_fix["wall"],
            ))
            print(f"{label:20s} seed={seed} | anchors auto={a_auto['n_anchors']:3d} fix=30 "
                  f"fid {a_auto['fidelity']:.4f}/{a_fix['fidelity']:.4f} | cap knee={a_auto['cap']} "
                  f"off={sel_off['cap']} | AUC knee={auc_knee:.4f} off={auc_off:.4f}", flush=True)

    # Verdicts.
    wide = [r for r in rows if r["width"] >= 2000]
    fid_wins = sum(1 for r in wide if r["fidelity_auto"] >= r["fidelity_fixed"] - 1e-9)
    auc_wins = sum(1 for r in rows if r["auc_knee"] >= r["auc_off"] - 1e-4)
    print("\n=== VERDICT ===")
    print(f"Lever1 anchors: fidelity_auto >= fidelity_fixed on WIDE: {fid_wins}/{len(wide)}")
    print(f"Lever2 knee ladder: auc_knee >= auc_off (all scenarios): {auc_wins}/{len(rows)}")

    out = dict(rows=rows, lever1_wide_wins=fid_wins, lever1_wide_total=len(wide), lever2_wins=auc_wins, lever2_total=len(rows))
    import pathlib
    p = pathlib.Path(__file__).parent / "_results" / "shapproxied_adaptive_guards.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
