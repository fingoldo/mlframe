"""Aggregate _results/results.jsonl into decision tables."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")


# strategies that are HybridSelector variants (the union/vote ensemble under test), vs everything else = individuals.
HYBRID_STRATEGIES = {"hybrid", "hybrid_nofe", "hybrid_strict", "hybrid_expand"}
# the named member each hard scenario expects to do the rescuing. NB HybridSelector's member_selections_ keys are
# exactly {mrmr, shap, boruta, tree} (hybrid_selector.py:494-515) -- it has NO rfecv member, so the rescuer map must
# name only members the hybrid actually owns: mrmr (prewarp-FE handles heavy tails / rare class via MI), tree
# (depth-3 GBM catches XOR/high-card/imbalance via splits), boruta (null-dist rejects noise), shap.
RESCUER_BY_SCENARIO = {
    "D1_pure_xor_zeromain": ["tree", "mrmr"],
    "D2_heavytail_linear": ["mrmr", "tree"],
    "D3_rare_class_imbalance": ["mrmr", "tree"],
    "D4_categorical_highcard": ["tree", "mrmr"],
    "D5_synth_pgg_n": ["mrmr", "boruta"],
    "D6_noise_dominated_weaksparse": ["boruta", "tree"],
}
HYBRID_BEAT_TOL = 0.002


def per_scenario_hybrid_check(rows):
    """Machine-checkable per (scenario): hybrid_beats_all_individuals (auc_mean(hybrid) >= max individual - tol),
    plus the rescuing-member check via member_selections_ where available."""
    ok = [r for r in rows if not r.get("error") and r.get("auc_mean") is not None]
    if not ok:
        return
    df = pd.DataFrame([dict(scenario=r.get("scenario", "default"), strategy=r["strategy"], seed=r["seed"], auc_mean=r["auc_mean"]) for r in ok])
    print("\n================ PER-SCENARIO HYBRID-BEATS-ALL CHECK ================")
    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen]
        agg = sub.groupby("strategy")["auc_mean"].mean()
        hyb = agg[[s for s in agg.index if s in HYBRID_STRATEGIES]]
        ind = agg[[s for s in agg.index if s not in HYBRID_STRATEGIES]]
        if hyb.empty or ind.empty:
            print(f"  [{scen}] insufficient strategies (hybrid={list(hyb.index)} individuals={len(ind)})")
            continue
        best_hyb = float(hyb.max()); best_ind = float(ind.max()); best_ind_name = ind.idxmax()
        beats = bool(best_hyb >= best_ind - HYBRID_BEAT_TOL)
        # rescuing-member check: did any expected rescuer member actually contribute selections for the hybrid?
        rescued = _rescuer_contributed(rows, scen)
        print(f"  [{scen}] hybrid_beats_all_individuals={beats}  "
              f"hybrid={best_hyb:.4f} (best={hyb.idxmax()})  best_individual={best_ind:.4f} ({best_ind_name})  "
              f"rescuer_contributed={rescued}")


def _rescuer_contributed(rows, scen):
    """True if, for the headline 'hybrid' strategy on this scenario, an expected rescuing member selected
    >=1 feature (member_selections_). Returns None if member_selections unavailable."""
    expected = RESCUER_BY_SCENARIO.get(scen)
    if not expected:
        return None
    found = None
    for r in rows:
        if r.get("scenario", "default") != scen or r.get("strategy") != "hybrid":
            continue
        ms = r.get("member_selections")
        if not isinstance(ms, dict):
            continue
        found = {m: len(ms.get(m, [])) for m in expected if m in ms}
        if any(v > 0 for v in found.values()):
            return found
    return found


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
            recs.append(dict(scenario=r.get("scenario", "default"), strategy=r["strategy"], seed=r["seed"], error=r["error"][:60]))
            continue
        a = r.get("auc", {})
        recs.append(dict(
            scenario=r.get("scenario", "default"),
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
             "shap_proxied", "H4_mrmrfilter__shap", "H6_mrmrfe__shap",
             "hybrid", "hybrid_strict", "hybrid_expand"]
    # never silently drop a strategy that ran but is missing from the manual order list (e.g. a new hybrid):
    agg = agg.reindex([s for s in order if s in agg.index] + [s for s in agg.index if s not in order])

    print("\n================ MEAN over seeds ================")
    print(agg[["n_feat", "n_eng", "fit_s", "base_recall", "base_missed", "noise_sel", "red_sel", "lgbm", "logit", "knn", "auc_mean"]].to_string())

    print("\n================ BEST STRATEGY per downstream model (mean AUC) ================")
    for m in ["lgbm", "logit", "knn", "auc_mean"]:
        s = agg[m].dropna().sort_values(ascending=False)
        print(f"\n  [{m}] top 5:")
        for k, v in s.head(5).items():
            print(f"    {v:.4f}  {k}  (n_feat={agg.loc[k, 'n_feat']:.0f})")

    per_scenario_hybrid_check(rows)

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
