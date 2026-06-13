"""Frontier-candidate FE probe: do CONDITIONAL-GATE and COMPARISON-CHAIN operators beat the best shipped operator on their natural targets?

Second-pass discovery bench (after gcd/lcm shipped). For each candidate family we build a synthetic where the family SHOULD win, then
measure binned-MI (the shipped ``_mi``, apples-to-apples with the modular/lattice detectors) of the candidate column vs the best existing
mlframe operator/basis on that same target. The delta is the whole point: if hinge / ratio / modular / a raw interaction already captures
it, it is not a gap. We also run controls (smooth / noise / ordinary interaction) so a candidate must be SPECIFIC (not fire everywhere).

Families probed:
* GATE  -- multiplicative threshold gate ``c>tau ? a : b`` and ``1[c>tau]*a`` (two raw features routed/masked by a third's threshold).
  Best existing baselines: raw a, raw b, raw c, a*b, a*c, a-b, ratio a/b, hinge(relu) of each, conditional-residual a-E[a|bin(c)].
* CHAIN -- row argmax (which of a,b,c is largest) and concordance-count (#{a>b, b>c, a>c}). Best existing baselines: raw cols, pairwise
  diffs a-b/b-c/a-c, products, max(a,b,c), and the binned MI of each.

Run: ``CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m
mlframe.feature_selection._benchmarks.bench_frontier_candidates``
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._pairwise_modular_fe import _mi

SEEDS = (1, 7, 13, 42, 101)
RESULTS_DIR = Path(__file__).parent / "_results"
NBINS = 12


def _cond_residual(a, c, nbins=8):
    """Best existing cross-feature operator that uses a third column's bins: a - E[a | bin(c)] (conditional_residual family)."""
    edges = np.quantile(c, np.linspace(0, 1, nbins + 1)[1:-1])
    binc = np.digitize(c, edges)
    out = a.astype(np.float64).copy()
    for b in np.unique(binc):
        m = binc == b
        out[m] = a[m] - a[m].mean()
    return out


def _hinge(x, tau=None):
    if tau is None:
        tau = np.median(x)
    return np.maximum(x - tau, 0.0)


def _best_existing_mi(cands: dict, y) -> tuple[str, float]:
    """Max binned-MI over a dict of existing-operator columns vs y."""
    best_name, best = "", -1.0
    for nm, col in cands.items():
        v = _mi(np.asarray(col, dtype=np.float64), y, nbins=NBINS)
        if v > best:
            best, best_name = v, nm
    return best_name, best


# ---------------- GATE family ----------------

def make_gate_select(seed, n=2000):
    """Target = (c>tau ? a : b) thresholded. The signal is a REGIME SWITCH: y follows a where c is high, b where c is low."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    tau = 0.0
    sel = np.where(c > tau, a, b)
    y = (sel > np.median(sel)).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    cand = np.where(c > tau, a, b)  # the gate operator (tau detectable via median/quantile scan)
    return X, y, cand, ("a", "b", "c")


def make_gate_mask(seed, n=2000):
    """Target = 1[c>tau] * a thresholded. a only matters in the sub-region c>tau; elsewhere y is noise-driven."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    tau = 0.0
    masked = (c > tau).astype(float) * a
    y = (masked > np.median(masked)).astype(int)
    X = pd.DataFrame({"a": a, "c": c})
    cand = (c > tau).astype(float) * a
    return X, y, cand, ("a", "c")


def gate_baselines(X, cols):
    arrs = {cn: np.asarray(X[cn], dtype=np.float64) for cn in cols}
    out = {f"raw_{cn}": arrs[cn] for cn in cols}
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = names[i], names[j]
            out[f"prod_{u}_{v}"] = arrs[u] * arrs[v]
            out[f"diff_{u}_{v}"] = arrs[u] - arrs[v]
            out[f"ratio_{u}_{v}"] = arrs[u] / (np.abs(arrs[v]) + 1e-6)
    for cn in cols:
        out[f"hinge_{cn}"] = _hinge(arrs[cn])
    if "a" in arrs and "c" in arrs:
        out["condresid_a_given_c"] = _cond_residual(arrs["a"], arrs["c"])
    if "b" in arrs and "c" in arrs:
        out["condresid_b_given_c"] = _cond_residual(arrs["b"], arrs["c"])
    return out


# ---------------- CHAIN family ----------------

def make_argmax_row(seed, n=2000):
    """Target = which of (a,b,c) is the row-max. argmax-of-row is a comparison pattern no smooth/arith op synthesises directly."""
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(0, 1, n) for _ in range(3))
    y = np.argmax(np.stack([a, b, c], axis=1), axis=1)  # 3-class
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    cand = y.astype(float)  # the argmax operator value (deterministic fn of X, no y)
    return X, y, cand, ("a", "b", "c")


def make_concordance(seed, n=2000):
    """Target keyed on concordance-count #{a>b, b>c, a>c} (an ordinal pattern / monotone-run length)."""
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(0, 1, n) for _ in range(3))
    cnt = (a > b).astype(int) + (b > c).astype(int) + (a > c).astype(int)
    y = (cnt >= 2).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    cand = cnt.astype(float)
    return X, y, cand, ("a", "b", "c")


def chain_baselines(X, cols):
    arrs = {cn: np.asarray(X[cn], dtype=np.float64) for cn in cols}
    out = {f"raw_{cn}": arrs[cn] for cn in cols}
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = names[i], names[j]
            out[f"diff_{u}_{v}"] = arrs[u] - arrs[v]
            out[f"prod_{u}_{v}"] = arrs[u] * arrs[v]
    stk = np.stack([arrs[cn] for cn in cols], axis=1)
    out["rowmax"] = stk.max(axis=1)
    out["rowmin"] = stk.min(axis=1)
    out["rowmean"] = stk.mean(axis=1)
    out["rowspread"] = stk.max(axis=1) - stk.min(axis=1)
    return out


# ---------------- controls (candidate must NOT win) ----------------

def make_smooth_control(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(0, 1, n) for _ in range(3))
    y = ((a + 0.5 * b) > 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y, ("a", "b", "c")


def make_noise_control(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(0, 1, n) for _ in range(3))
    y = rng.integers(0, 2, n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y, ("a", "b", "c")


def make_ordinary_interaction(seed, n=2000):
    rng = np.random.default_rng(seed)
    a, b, c = (rng.normal(0, 1, n) for _ in range(3))
    y = ((a * b) > 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c}), y, ("a", "b", "c")


def _run_family(name, gen, baseline_fn, cand_builder, is_tp):
    """For a candidate scenario: cand-MI vs best-existing-MI, averaged over seeds. cand_builder(X, cols)->candidate column (no y)."""
    rows = []
    for s in SEEDS:
        if is_tp:
            X, y, cand, cols = gen(s)
        else:
            X, y, cols = gen(s)
            cand = cand_builder(X, cols)
        yi = np.asarray(y).astype(np.int64)
        t0 = time.perf_counter()
        cand_mi = _mi(np.asarray(cand, dtype=np.float64), yi, nbins=NBINS)
        base = baseline_fn(X, cols)
        bname, base_mi = _best_existing_mi(base, yi)
        dt = time.perf_counter() - t0
        rows.append({"seed": s, "cand_mi": round(float(cand_mi), 4), "best_base": bname,
                     "best_base_mi": round(float(base_mi), 4),
                     "lift": round(float(cand_mi - base_mi), 4), "wall_s": round(dt, 4)})
    lifts = [r["lift"] for r in rows]
    return {"scenario": name, "is_tp": is_tp, "mean_cand_mi": round(float(np.mean([r["cand_mi"] for r in rows])), 4),
            "mean_best_base_mi": round(float(np.mean([r["best_base_mi"] for r in rows])), 4),
            "mean_lift": round(float(np.mean(lifts)), 4), "min_lift": round(float(np.min(lifts)), 4),
            "mean_wall_s": round(float(np.mean([r["wall_s"] for r in rows])), 4), "per_seed": rows}


def _gate_select_cand(X, cols):
    c = np.asarray(X["c"], dtype=np.float64)
    a, b = np.asarray(X["a"], dtype=np.float64), np.asarray(X["b"], dtype=np.float64)
    return np.where(c > 0.0, a, b)


def _gate_mask_cand(X, cols):
    c = np.asarray(X["c"], dtype=np.float64)
    a = np.asarray(X["a"], dtype=np.float64)
    return (c > 0.0).astype(float) * a


def _argmax_cand(X, cols):
    stk = np.stack([np.asarray(X[cn], dtype=np.float64) for cn in cols], axis=1)
    return np.argmax(stk, axis=1).astype(float)


def _concordance_cand(X, cols):
    a, b, c = (np.asarray(X[cn], dtype=np.float64) for cn in ("a", "b", "c"))
    return ((a > b).astype(int) + (b > c).astype(int) + (a > c).astype(int)).astype(float)


def main():
    results = {"gate": [], "chain": [], "controls": []}

    results["gate"].append(_run_family("gate_select", make_gate_select, gate_baselines, None, is_tp=True))
    results["gate"].append(_run_family("gate_mask", make_gate_mask, gate_baselines, None, is_tp=True))

    results["chain"].append(_run_family("argmax_row", make_argmax_row, chain_baselines, None, is_tp=True))
    results["chain"].append(_run_family("concordance3", make_concordance, chain_baselines, None, is_tp=True))

    # controls: run BOTH candidate operators; neither should beat existing on a non-matching target.
    for cn, builder, bfn in (("gate_select", _gate_select_cand, gate_baselines),
                             ("gate_mask", _gate_mask_cand, gate_baselines),
                             ("argmax", _argmax_cand, chain_baselines),
                             ("concordance", _concordance_cand, chain_baselines)):
        for ctrl_name, ctrl_gen in (("smooth", make_smooth_control), ("noise", make_noise_control),
                                    ("ordinary_mul", make_ordinary_interaction)):
            results["controls"].append(_run_family(f"{cn}__on__{ctrl_name}", ctrl_gen, bfn, builder, is_tp=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"frontier_candidates_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("=== TP (candidate should WIN: lift > 0) ===")
    for grp in ("gate", "chain"):
        for r in results[grp]:
            print(f"  {r['scenario']:16s} cand_mi={r['mean_cand_mi']:.3f} best_base={r['mean_best_base_mi']:.3f} "
                  f"lift={r['mean_lift']:+.3f} min_lift={r['min_lift']:+.3f} ({r['per_seed'][0]['best_base']})")
    print("=== CONTROLS (candidate should NOT win: lift <= ~0) ===")
    for r in results["controls"]:
        print(f"  {r['scenario']:28s} lift={r['mean_lift']:+.3f} (cand {r['mean_cand_mi']:.3f} vs {r['mean_best_base_mi']:.3f})")
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
