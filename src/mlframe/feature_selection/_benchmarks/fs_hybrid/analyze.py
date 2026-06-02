"""Aggregate _results/results.jsonl into decision tables."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")


def main():
    rows = []
    with open(os.path.join(OUT, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    recs = []
    for r in rows:
        if r.get("error"):
            recs.append(dict(strategy=r["strategy"], seed=r["seed"], error=r["error"][:60]))
            continue
        a = r.get("auc", {})
        recs.append(dict(
            strategy=r["strategy"], seed=r["seed"],
            n_feat=r.get("n_features"), n_eng=r.get("n_engineered"),
            fit_s=r.get("fit_seconds"),
            base_recall=r.get("base_recall"), base_missed=r.get("base_missed"),
            noise_sel=r.get("n_noise_selected"), red_sel=r.get("n_redundant_selected"),
            lgbm=a.get("lgbm"), logit=a.get("logit"), knn=a.get("knn"),
            auc_mean=r.get("auc_mean"),
        ))
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)

    print("\n================ RAW (per seed) ================")
    print(df.to_string(index=False))

    ok = df[df.get("error").isna()] if "error" in df else df
    num = ["n_feat", "n_eng", "fit_s", "base_recall", "base_missed", "noise_sel", "red_sel", "lgbm", "logit", "knn", "auc_mean"]
    agg = ok.groupby("strategy")[num].mean().round(4)
    agg["n_seeds"] = ok.groupby("strategy").size()
    # preserve roster order
    order = ["all", "mrmr_filter", "mrmr_fe", "boruta", "boruta_stable", "rfecv_lgbm", "rfecv_lgbm_perm", "rfecv_logit",
             "H1_mrmrfilter__rfecv_lgbm", "H2_mrmrfe__rfecv_logit", "H3_boruta__rfecv_lgbm",
             "H_union_mrmr_boruta", "H_intersect_mrmr_boruta", "H5_mrmr_boruta__rfecv_lgbm",
             "H7_mrmr_borutastable__rfecv_lgbm",
             "shap_proxied", "H4_mrmrfilter__shap", "H6_mrmrfe__shap"]
    agg = agg.reindex([s for s in order if s in agg.index])

    print("\n================ MEAN over seeds ================")
    print(agg[["n_feat", "n_eng", "fit_s", "base_recall", "base_missed", "noise_sel", "red_sel", "lgbm", "logit", "knn", "auc_mean"]].to_string())

    print("\n================ BEST STRATEGY per downstream model (mean AUC) ================")
    for m in ["lgbm", "logit", "knn", "auc_mean"]:
        s = agg[m].dropna().sort_values(ascending=False)
        print(f"\n  [{m}] top 5:")
        for k, v in s.head(5).items():
            print(f"    {v:.4f}  {k}  (n_feat={agg.loc[k, 'n_feat']:.0f})")

    print("\n================ ONE-SIZE-FITS-ALL CHECK ================")
    best = {m: agg[m].idxmax() for m in ["lgbm", "logit", "knn"]}
    print("  argmax strategy per model:", best)
    # For the single best shared strategy by auc_mean, how far is each model from its own best?
    shared = agg["auc_mean"].idxmax()
    print(f"  best shared (by auc_mean): {shared}")
    for m in ["lgbm", "logit", "knn"]:
        gap = agg[m].max() - agg.loc[shared, m]
        print(f"    {m}: shared={agg.loc[shared, m]:.4f}  model_best={agg[m].max():.4f} ({best[m]})  gap={gap:+.4f}")


if __name__ == "__main__":
    main()
