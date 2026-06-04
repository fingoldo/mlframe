"""DIAGNOSTIC (round-4, READ-ONLY, no production edit): does the SUPERVISED MDLP binner
collapse PURE-INTERACTION operands to 1 bin (= constant) BEFORE feature engineering, destroying
their joint MI and thus blocking the madelon / synergy-bootstrap rescue?

Hypothesis A1-7 operand_mi_preserve_binning: MDLP (nbins_strategy='mdlp' = MRMR default) minimizes
description length vs y. A pure-interaction operand has ~0 MARGINAL MI with y, so no single split
improves the marginal MDL -> MDLP returns the trivial 1-bin partition -> the operand is a CONSTANT
-> its JOINT MI with any partner is destroyed before FE can pair them.

We test THREE binning layers per column and count bins:
  (R) RAW MDLP                -- supervised_binning.mdlp_bin_edges (the literal hypothesis target).
  (P) PRODUCTION per_feature_edges(method='mdlp') -- what MRMR ACTUALLY calls via categorize_dataset;
      this layer has a documented collapsed-column fallback to quantile (lines 634-641) + a
      sparse-aware fallback (642-688). Tells us if the hypothesis is already neutralized in prod.
  (Q) UNSUPERVISED quantile   -- per_feature_edges(method='fd') and a plain n=10 quantile, the contrast.

bins reported = len(inner_edges) + 1. 1 bin == collapsed-to-constant.

Beds:
  1. hard_synth (n=5000): KNOWN operands -- ia, ib (pure interaction: only ia*ib enters logit),
     sq (quadratic: only sq^2-1 enters); strong str_* (large marginal); weak_* (small marginal);
     noise_* (none). Count bins per known group.
  2. designed pure-interaction frame: y=sign(a*b), a,b ~ N(0,1) + a few marginal + noise columns.
  3. madelon (load_real, n=2600 x 500): count how many of 500 cols collapse to 1 bin under each layer;
     cross-tab against an lgbm-importance top-20 (madelon's informative set) -- are the informative
     features among the MDLP-1-bin-collapsed?

VERDICT: CONFIRMED if operands collapse to 1 bin under RAW MDLP but keep >1 bin under quantile
(and we report whether the PRODUCTION fallback already rescues them); REFUTED otherwise.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_synth import make_hard_dataset
from round3_realdata_bench import load_real

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

PROGRESS = r"D:\Temp\mdlp_diag_progress.txt"
RESULTS = r"D:\Temp\mdlp_diag_results.md"


def ckpt(msg: str):
    with open(PROGRESS, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg.rstrip() + "\n")
    print("CKPT " + msg, flush=True)


def raw_mdlp_bins(x, y):
    """bins under RAW supervised MDLP (the literal hypothesis target). 1 == collapsed."""
    edges = mdlp_bin_edges(np.asarray(x, dtype=np.float64), np.asarray(y))
    # edges include -inf/+inf sentinels; inner cuts = edges.size - 2.
    inner = edges.size - 2
    return int(inner + 1)


def prod_bins_matrix(X, y, method, **kw):
    """bins per column from PRODUCTION per_feature_edges (incl. fallback chain). list of bin-counts."""
    edges_list = per_feature_edges(np.asarray(X, dtype=np.float64), y=np.asarray(y),
                                   method=method, **kw)
    return [int(np.asarray(e).size + 1) for e in edges_list]


def quantile10_bins(x):
    """bins under plain unsupervised 10-quantile (np.unique-dedup'd). 1 == collapsed (constant)."""
    q = np.linspace(0.0, 100.0, 11)
    e = np.unique(np.nanpercentile(np.asarray(x, dtype=np.float64), q))
    inner = max(0, e.size - 2)
    return int(inner + 1)


def summarize_group(name, cols, raw_b, prodmdlp_b, prodfd_b, q10_b, lines):
    """Emit a per-group min/median/max/n_collapsed line for each layer."""
    def stat(d):
        v = np.array([d[c] for c in cols], dtype=float)
        n1 = int((v <= 1).sum())
        return f"min={int(v.min())} med={int(np.median(v))} max={int(v.max())} n_1bin={n1}/{len(cols)}"
    lines.append(f"\n### {name}  (n_cols={len(cols)})")
    lines.append(f"  RAW MDLP        : {stat(raw_b)}")
    lines.append(f"  PROD mdlp(+fb)  : {stat(prodmdlp_b)}")
    lines.append(f"  PROD fd (unsup) : {stat(prodfd_b)}")
    lines.append(f"  quantile10      : {stat(q10_b)}")
    for ln in lines[-5:]:
        print(ln, flush=True)


def run_named_bed(bedname, X, y, groups, lines):
    """X: DataFrame; groups: dict group_name -> list of column names."""
    ckpt(f"bed={bedname} shape={X.shape} start")
    cols = list(X.columns)
    yv = np.asarray(y)

    # RAW MDLP per column.
    raw_b = {}
    for c in cols:
        raw_b[c] = raw_mdlp_bins(X[c].values, yv)
    ckpt(f"bed={bedname} raw_mdlp done")

    # PRODUCTION per_feature_edges (mdlp w/ fallback chain) + (fd unsup).
    pm = prod_bins_matrix(X[cols].values, yv, method="mdlp")
    prodmdlp_b = {c: pm[i] for i, c in enumerate(cols)}
    ckpt(f"bed={bedname} prod_mdlp done")
    pf = prod_bins_matrix(X[cols].values, yv, method="fd")
    prodfd_b = {c: pf[i] for i, c in enumerate(cols)}
    ckpt(f"bed={bedname} prod_fd done")

    # plain quantile10 per column.
    q10_b = {c: quantile10_bins(X[c].values) for c in cols}

    lines.append(f"\n## {bedname}  shape={X.shape}")
    for gname, gcols in groups.items():
        gcols = [c for c in gcols if c in raw_b]
        if gcols:
            summarize_group(gname, gcols, raw_b, prodmdlp_b, prodfd_b, q10_b, lines)
    return raw_b, prodmdlp_b, prodfd_b, q10_b


def main():
    open(PROGRESS, "w").close()
    lines = ["# MDLP-1-bin-collapse diagnostic (round-4, READ-ONLY)\n"]
    lines.append("bins = inner_edges + 1. 1 bin == collapsed-to-constant (joint MI destroyed).")
    lines.append("RAW MDLP = supervised_binning.mdlp_bin_edges (literal hypothesis target).")
    lines.append("PROD mdlp(+fb) = per_feature_edges(method='mdlp') = what MRMR actually calls "
                 "(has collapsed->quantile fallback).")

    # ----- BED 1: hard_synth with KNOWN operands -----
    Xh, yh, truth = make_hard_dataset(n_samples=5000, seed=0)
    groups_h = {
        "interaction_operands ia,ib (pure-interaction, ~0 marginal)": truth["interaction_operands"],
        "quadratic_operand sq (~0 marginal)": truth["quadratic_operand"],
        "strong str_* (large marginal)": truth["strong"],
        "weak_sparse weak_* (small marginal)": truth["weak_sparse"],
        "redundant copies of str_0": [c for c in Xh.columns if c.startswith("red_0_")],
        "noise_* (none)": [c for c in Xh.columns if c.startswith("noise_")][:30],  # sample 30 of 200
    }
    run_named_bed("hard_synth", Xh, yh, groups_h, lines)

    # ----- BED 2: designed PURE-interaction frame y=sign(a*b) -----
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.standard_normal(n); b = rng.standard_normal(n)
    noise_ab = 0.10 * rng.standard_normal(n)
    yab = (np.sign(a * b + noise_ab) > 0).astype(int)
    marg = rng.standard_normal(n)  # a column WITH marginal signal (correlated to y) for contrast
    # give 'marg' real marginal MI: flip y-correlated
    marg = marg + 1.2 * (yab - 0.5)
    cols_ab = {"a": a, "b": b, "marg": marg}
    for i in range(10):
        cols_ab[f"noise_{i}"] = rng.standard_normal(n)
    Xab = pd.DataFrame(cols_ab)
    groups_ab = {
        "pure-interaction operands a,b (y=sign(a*b))": ["a", "b"],
        "marginal feature marg (real marginal MI)": ["marg"],
        "noise_*": [f"noise_{i}" for i in range(10)],
    }
    run_named_bed("designed_pure_interaction", Xab, pd.Series(yab), groups_ab, lines)

    # ----- BED 3: madelon -----
    ckpt("loading madelon")
    try:
        Xr, yr, rname = load_real()
    except Exception as e:
        ckpt(f"madelon load FAILED {type(e).__name__}: {e}")
        Xr = None
    if Xr is not None and rname == "madelon":
        ckpt(f"madelon loaded shape={Xr.shape}")
        cols = list(Xr.columns)
        yv = np.asarray(yr)
        # bin counts under each layer (all 500 cols).
        raw_b = np.array([raw_mdlp_bins(Xr[c].values, yv) for c in cols])
        ckpt("madelon raw_mdlp done")
        pm = np.array(prod_bins_matrix(Xr[cols].values, yv, method="mdlp"))
        ckpt("madelon prod_mdlp done")
        pf = np.array(prod_bins_matrix(Xr[cols].values, yv, method="fd"))
        ckpt("madelon prod_fd done")
        q10 = np.array([quantile10_bins(Xr[c].values) for c in cols])

        n_raw1 = int((raw_b <= 1).sum())
        n_pm1 = int((pm <= 1).sum())
        n_pf1 = int((pf <= 1).sum())
        n_q1 = int((q10 <= 1).sum())
        lines.append(f"\n## madelon  shape={Xr.shape}  (500 cols)")
        lines.append(f"  RAW MDLP        : n_1bin={n_raw1}/500  (min={int(raw_b.min())} "
                     f"med={int(np.median(raw_b))} max={int(raw_b.max())})")
        lines.append(f"  PROD mdlp(+fb)  : n_1bin={n_pm1}/500  (min={int(pm.min())} "
                     f"med={int(np.median(pm))} max={int(pm.max())})")
        lines.append(f"  PROD fd (unsup) : n_1bin={n_pf1}/500  (min={int(pf.min())} "
                     f"med={int(np.median(pf))} max={int(pf.max())})")
        lines.append(f"  quantile10      : n_1bin={n_q1}/500  (min={int(q10.min())} "
                     f"med={int(np.median(q10))} max={int(q10.max())})")
        for ln in lines[-5:]:
            print(ln, flush=True)

        # lgbm top-20 importance = madelon's informative set; are they among RAW-MDLP-collapsed?
        ckpt("madelon lgbm importance")
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=16,
                               learning_rate=0.1, n_jobs=2, verbose=-1, random_state=0)
        m.fit(Xr, yr)
        imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
        top20 = list(imp.index[:20])
        raw_map = {c: int(raw_b[cols.index(c)]) for c in cols}
        pm_map = {c: int(pm[cols.index(c)]) for c in cols}
        q10_map = {c: int(q10[cols.index(c)]) for c in cols}
        top_raw1 = [c for c in top20 if raw_map[c] <= 1]
        top_pm1 = [c for c in top20 if pm_map[c] <= 1]
        top_q1 = [c for c in top20 if q10_map[c] <= 1]
        lines.append(f"\n### madelon lgbm-top20 informative features -- bin counts")
        lines.append(f"  top20 RAW-MDLP collapsed-to-1bin : {len(top_raw1)}/20  {top_raw1}")
        lines.append(f"  top20 PROD-mdlp collapsed-to-1bin: {len(top_pm1)}/20  {top_pm1}")
        lines.append(f"  top20 quantile10 collapsed-to-1bin: {len(top_q1)}/20  {top_q1}")
        lines.append(f"  per-feature (col: raw_mdlp / prod_mdlp / q10 bins):")
        for c in top20:
            lines.append(f"    {c:8s}: raw={raw_map[c]:3d}  prod={pm_map[c]:3d}  q10={q10_map[c]:3d}")
        for ln in lines[-25:]:
            print(ln, flush=True)
    else:
        lines.append(f"\n## madelon UNAVAILABLE (got {rname if Xr is not None else 'load-fail'}); "
                     f"skipping real-data layer")

    with open(RESULTS, "w") as f:
        f.write("\n".join(lines) + "\n")
    ckpt("DONE wrote results")
    print("\n=== wrote " + RESULTS + " ===", flush=True)


if __name__ == "__main__":
    main()
