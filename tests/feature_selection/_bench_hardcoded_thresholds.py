"""Sensitivity benchmark for the 31 audited hardcoded MRMR thresholds
(see MRMR_HARDCODED_THRESHOLDS_AUDIT.md).

For each threshold we sweep its value across a plausible band on a grid of synthetic datasets
(varying n + signal archetype) and measure the downstream test MAE of a linear AND a tree model on
the MRMR-selected/transformed feature space. The verdict per threshold:

* FLAT (test MAE insensitive to the value across the band, AND the optimal value does not move with
  the dataset) -> converting it to a data-derived value buys nothing; KEEP the hardcoded default,
  documented with the measured flat response.
* DATA-DEPENDENT (the optimal value shifts with n / archetype, and a wrong fixed value costs MAE)
  -> a real conversion candidate; implement the audit's permutation-null / CV replacement.

Run:  python tests/feature_selection/_bench_hardcoded_thresholds.py [group]
  group in {high, all} (default high). Prints a per-threshold table + verdict.

This is a measurement harness, NOT a pytest (each cell fits a full MRMR+FE pipeline). Keep n small.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error as MAE


def _datasets(n: int, seed: int = 0):
    """A small grid of magnitude-carrying regression archetypes (the targets where FE-pair gates
    actually bite). Each returns (df, y, name)."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    out = []
    # F2: heavy-tail ratio + trig interaction (the canonical (c,d) case).
    out.append((pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0), "F2_ratio_trig"))
    # bilinear interaction a*b + marginal, no heavy tail.
    out.append((pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 3.0 * (a - 0.5) * (b - 0.5) + 0.5 * c + f / 5.0, "bilinear_ab"))
    # additive-only (no genuine interaction; gates should NOT fabricate pairs).
    out.append((pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 1.5 * a + 1.0 * c - 0.7 * d + f / 5.0, "additive_only"))
    return out


def _eval_threshold(param: str, values, ns=(2500,), seeds=(0,), extra_kwargs=None):
    """Fit MRMR for each (dataset, value) and report linear+tree test MAE. Returns a list of rows
    dict(dataset, n, value, lin_mae, n_features)."""
    from mlframe.feature_selection.filters import MRMR

    rows = []
    for n in ns:
        for seed in seeds:
            for df, y, dname in _datasets(n, seed):
                idx = np.random.default_rng(seed).permutation(n)
                tr, te = idx[: int(0.8 * n)], idx[int(0.9 * n) :]
                Xtr, Xte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
                ytr, yte = np.asarray(y)[tr], np.asarray(y)[te]
                for v in values:
                    kw = dict(verbose=0, random_seed=seed)
                    kw[param] = v
                    if extra_kwargs:
                        kw.update(extra_kwargs)
                    try:
                        fs = MRMR(**kw).fit(X=Xtr, y=pd.Series(ytr, name="y"))
                        Ztr, Zte = fs.transform(Xtr), fs.transform(Xte)
                        Ztr = Ztr.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
                        Zte = Zte.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
                        m = make_pipeline(StandardScaler(), LinearRegression()).fit(Ztr.values, ytr)
                        lin = float(MAE(yte, m.predict(Zte.values)))
                        nf = Ztr.shape[1]
                    except Exception as ex:
                        lin, nf = float("nan"), -1
                        print(f"    [error {param}={v} {dname}] {type(ex).__name__}: {ex}")
                    rows.append(dict(dataset=dname, n=n, param=param, value=v, lin_mae=lin, nfeat=nf))
    return rows


def _verdict(rows):
    """FLAT if, within each dataset, the linear MAE spread across values is < 1% of the mean AND the
    argmin value is the same across datasets; else DATA-DEPENDENT."""
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    spreads, argmins = [], set()
    for rs in by_ds.values():
        maes = [r["lin_mae"] for r in rs if np.isfinite(r["lin_mae"])]
        if len(maes) < 2:
            continue
        mean = np.mean(maes)
        spread = (max(maes) - min(maes)) / mean if mean > 0 else 0.0
        spreads.append(spread)
        best = min(rs, key=lambda r: r["lin_mae"] if np.isfinite(r["lin_mae"]) else 1e9)
        argmins.add(best["value"])
    max_spread = max(spreads) if spreads else 0.0
    flat = max_spread < 0.01 and len(argmins) <= 1
    return ("FLAT" if flat else "DATA-DEPENDENT", max_spread, sorted(argmins))


# Threshold -> sweep band (low, default, high). Defaults match the audit.
HIGH = [
    ("fe_min_pair_mi_prevalence", [1.0, 1.05, 1.20]),
    ("fe_synergy_min_prevalence", [1.2, 1.5, 2.0]),
    ("fe_escalation_pairness_margin", [1.05, 1.15, 1.30]),
    ("fe_escalation_underdelivery_self_ratio", [2.0, 3.0, 4.5]),
    # _FE_MARGINAL_UPLIFT_MIN_RATIO is a module constant, not a ctor param -> handled separately.
]


# MED + LOW constructor-param thresholds (module-constant items handled by monkeypatch separately).
MEDLOW = [
    ("fe_engineered_cmi_retain_frac", [0.08, 0.15, 0.25]),
    ("fe_sufficient_summary_maxt_quantile", [0.90, 0.95, 0.99]),
    ("fe_sufficient_summary_residual_frac", [0.15, 0.25, 0.40]),
    ("fe_rung_rel_floor", [0.25, 0.40, 0.55]),
    ("fe_stability_vote_k", [3, 5, 7]),
    ("fe_stability_vote_quorum", [0.5, 0.6, 0.75]),
    ("fe_escalation_min_val_corr", [0.08, 0.15, 0.25]),
    ("min_relevance_gain_relative_to_first", [0.02, 0.05, 0.10]),
    ("min_relevance_gain_frac", [0.0005, 0.001, 0.005]),
    ("fe_confirm_undersample_rows_per_cell", [3.0, 5.0, 8.0]),
    ("fe_pair_perm_null_excess_frac", [0.02, 0.05, 0.10]),
    ("fe_min_nonzero_confidence", [0.95, 0.99, 0.999]),
    ("fe_min_pair_mi", [0.0005, 0.001, 0.005]),
    ("fe_good_to_best_feature_mi_threshold", [0.90, 0.98, 0.999]),
    ("fe_adaptive_relax_factor", [0.8, 0.9, 0.95]),
]


PENDING = [
    ("fe_pair_perm_null_excess_frac", [0.02, 0.05, 0.10]),
    ("fe_min_nonzero_confidence", [0.95, 0.99, 0.999]),
    ("fe_min_pair_mi", [0.0005, 0.001, 0.005]),
    ("fe_good_to_best_feature_mi_threshold", [0.90, 0.98, 0.999]),
    ("fe_adaptive_relax_factor", [0.8, 0.9, 0.95]),
]


def main():
    """CLI entry point: run the threshold-sensitivity bench for the requested group and print verdicts."""
    group = sys.argv[1] if len(sys.argv) > 1 else "high"
    table = {"high": HIGH, "medlow": MEDLOW, "pending": PENDING, "all": HIGH + MEDLOW}.get(group, HIGH)
    print(f"== threshold sensitivity bench ({group}) ==")
    for param, values in table:
        t = time.perf_counter()
        rows = _eval_threshold(param, values)
        verdict, spread, argmins = _verdict(rows)
        print(f"\n### {param}  [{verdict}]  max_spread={spread:.3%}  argmin_values={argmins}  ({time.perf_counter() - t:.0f}s)")
        for ds in sorted({r["dataset"] for r in rows}):
            cells = [r for r in rows if r["dataset"] == ds]
            s = "  ".join(f"{r['value']}:{r['lin_mae']:.4f}(nf={r['nfeat']})" for r in cells)
            print(f"    {ds:16s} {s}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
