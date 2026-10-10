"""Aggregate the wave-2 result rows into ``wave2/results/summary.txt`` and ``summary.json``: per operator / n / case means and sd over seeds, win and acceptance counts, calibrated c.

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.aggregate``.
Acceptance: a seed is accepted when ``gain * n_fit > c`` (gain = held-out MI of the new feature minus the best existing candidate). ``c`` is calibrated per operator on the ``calib`` rows (other seeds,
cases 0 / N / HN / H-null) as the maximum observed ``gain * n_fit``; the reported acceptance on the eval rows uses both the project's c = 40 and the calibrated value.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

from ..common._paths import results_dir

PROJECT_C = 40.0
NULL_CASES = ("0", "N", "HN")


def load(pattern: str) -> list:
    """All JSON rows of the files matching ``pattern`` inside the wave2 results folder."""
    rows = []
    for f in sorted(glob.glob(str(results_dir("wave2") / pattern))):
        rows += [json.loads(line) for line in open(f)]
    return rows


def ms(v) -> str:
    """``mean+-sd`` text of a list (NaN ignored)."""
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return f"{a.mean():+.3f}+-{a.std():.3f}" if len(a) else "nan"


def label(r: dict) -> str:
    """Operator label of a row; operator B rows carry the shrinkage (rows written before the field existed used 20)."""
    if r["op"] != "B":
        return r["op"]
    m = r.get("cell_m") or 20.0
    return f"B(m={m:g})"


def calibrate(calib: list) -> dict:
    """Per operator: max, q95 and count of ``gain * n_fit`` over null-type calibration rows (cases 0, N, HN)."""
    out = {}
    for op in sorted({label(r) for r in calib}):
        g = [r["gain_n"] for r in calib if label(r) == op and r["case"] in NULL_CASES]
        if g:
            out[op] = {"max": float(np.max(g)), "q95": float(np.quantile(g, 0.95)), "n": len(g)}
    return out


SCREENS = ("preset_y", "preset_r", "cell_y", "cell_r")


def summarize_M(rows: list) -> dict:
    """Pair-screen summary per (n, ideal / hard, kind): true-pair top-1 and top-3 rates per screen, mean rank of the first true pair, residual-vs-y pick change rate,
    null-scale of the screen score (``score * n_fit``), downstream relative MAE / RMSE of engineering each screen's top pair, and the null max of the residual cell screen.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["n"], "hard" if r["hard"] else "ideal", r["kind"])].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        d = {"seeds": len(rs)}
        for s in SCREENS:
            d[f"{s}|top1_true"] = float(np.mean([r[f"{s}|hit_top1"] for r in rs]))
            if rs[0]["true_pairs"]:
                d[f"{s}|top3_all_true"] = float(np.mean([all(r[f"{s}|hit_top3"]) for r in rs]))
                d[f"{s}|mean_rank_first_true"] = float(np.mean([r[f"{s}|rank_true"][0] for r in rs]))
                d[f"{s}|mean_rank_last_true"] = float(np.mean([r[f"{s}|rank_true"][-1] for r in rs]))
            d[f"{s}|top_score_x_n"] = ms([r[f"{s}|top_score"] * r["n_fit"] for r in rs]) if s.startswith("cell") else ms([r[f"{s}|top_score"] for r in rs])
        d["pick_changed_cell_r_vs_cell_y"] = float(np.mean([r["cell_r|top"] != r["cell_y|top"] for r in rs]))
        d["pick_changed_preset_r_vs_preset_y"] = float(np.mean([r["preset_r|top"] != r["preset_y|top"] for r in rs]))
        d["null_cell_r_max_x_n"] = ms([r["null_cell_r_max"] * r["n_fit"] for r in rs])
        d["t_resid"] = float(np.mean([r["t_resid"] for r in rs]))
        d["t_screens"] = float(np.mean([r["t_screens"] for r in rs]))
        for k in rs[0]:
            if k.startswith(("mae|", "rmse|", "mi_te|", "ab|mae|", "ab|rmse|")) and not k.endswith("|raw"):
                d[k] = ms([r.get(k) for r in rs])
        out["_".join(map(str, key))] = d
    return out


def main() -> None:
    """Write the summary tables."""
    ev, cal = load("rows_[BCG]_*_eval*.jsonl"), load("rows_[BCG]_*_calib*.jsonl")
    calib = calibrate(cal)
    groups = defaultdict(list)
    for r in ev:
        groups[(label(r), r["n"], r["case"])].append(r)
    lines, js = [], {"calibration": calib, "cases": {}}
    lines.append("calibration of c (max / q95 of gain*n_fit on null-type calib rows): " + json.dumps(calib))
    for (op, n, case), rs in sorted(groups.items()):
        c_cal = calib.get(op, {}).get("max", float("nan"))
        gn = np.array([r["gain_n"] for r in rs])
        d = {
            "seeds": len(rs),
            "ex_te": ms([r["ex_te"] for r in rs]),
            "new_te": ms([r["new_te"] for r in rs]),
            "truth_te": ms([r.get("truth_te") for r in rs]),
            "newfa_te": ms([r.get("newfa_te") for r in rs]),
            "chosen": sorted({json.dumps(r["info"]) for r in rs})[:3],
            "wins": int(sum(r["gain"] > 0 for r in rs)),
            "gain_n_mean": float(gn.mean()),
            "gain_n_max": float(gn.max()),
            "accept_c40": int((gn > PROJECT_C).sum()),
            "accept_ccal": int((gn > c_cal).sum()) if np.isfinite(c_cal) else None,
            "replay_all_pass": int(sum(all(r["replay"].values()) for r in rs)),
            "t_new": float(np.mean([r["t_new"] for r in rs])),
            "t_existing": float(np.mean([r["t_existing"] for r in rs])),
        }
        for k in rs[0]:
            if k.startswith(("mae|", "rmse|", "leak|mae", "leak|rmse")) and not k.endswith("|raw"):
                d[k] = ms([r.get(k) for r in rs])
        js["cases"][f"{op}_{n}_{case}"] = d
        lines.append(f"[{op} n={n} case={case}] " + json.dumps(d))
    m = summarize_M(load("rows_M_*_eval.jsonl"))
    js["M"] = m
    lines += [f"[M {k}] " + json.dumps(v) for k, v in m.items()]
    (results_dir("wave2") / "summary.json").write_text(json.dumps(js, indent=1))
    (results_dir("wave2") / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
