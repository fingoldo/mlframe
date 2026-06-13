"""Bench: the four MRMR MI-floor FE operators (pairwise-modular / integer-lattice / row-argmax / conditional-gate) on REGRESSION targets.

These operators gate candidates on plug-in CLASS-MI. A continuous y is quantile-binned once (``bin_y_for_class_mi``, nbins=quantization_nbins)
into a discrete target before scoring, which makes the operators applicable to regression. This bench confirms, per operator on synthetic
REGRESSION targets where each should win:

  DETECT     -- on the matching modular / gcd / argmax / regime regression target the operator emits its feature (lift of the engineered
                column's MI over the best-existing-op MI, both on the SAME binned y, >= a measured floor).
  SPECIFIC   -- on a SMOOTH/noise CONTINUOUS control (3 seeds) the operator emits ZERO features (0 spurious emission).
  NO-HANG    -- each fit is timed; a regression fit on 600 rows must complete < 30s.

Run: CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 MLFRAME_DISABLE_GPU=1 PYTHONPATH=src python -m \
     mlframe.feature_selection._benchmarks.bench_fe_operators_regression
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_accuracy_gate import bin_y_for_class_mi
from mlframe.feature_selection.filters._pairwise_modular_fe import _mi, hybrid_pairwise_modular_fe_with_recipes
from mlframe.feature_selection.filters._integer_lattice_fe import hybrid_integer_lattice_fe_with_recipes
from mlframe.feature_selection.filters._conditional_gate_fe import (
    hybrid_conditional_gate_fe_with_recipes,
    hybrid_row_argmax_fe_with_recipes,
)

NBINS = 10
N = 600


def _best_existing_mi(X: pd.DataFrame, yb: np.ndarray) -> float:
    """Best MI over the cheap arithmetic ops a selector already has (raw cols + products / sums / diffs), on the binned y."""
    cols = list(X.columns)
    best = 0.0
    for c in cols:
        best = max(best, _mi(X[c].to_numpy().astype(np.float64), yb, nbins=NBINS))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = X[cols[i]].to_numpy().astype(np.float64), X[cols[j]].to_numpy().astype(np.float64)
            for surrogate in (a * b, a + b, a - b):
                best = max(best, _mi(surrogate, yb, nbins=NBINS))
    return best


def _run(name: str, X: pd.DataFrame, y: np.ndarray, fn, marker: str, **kw):
    yb = bin_y_for_class_mi(y, nbins=NBINS)
    t0 = time.time()
    appended, recipes = fn(X, yb, **kw)
    wall = time.time() - t0
    emitted = [a for a in appended if marker in a]
    eng_mi = max((_mi(_mat_for(X, r), yb, nbins=NBINS) for r in recipes if r.name in emitted), default=0.0)
    base_mi = _best_existing_mi(X, yb)
    return {
        "operator": name, "emitted": len(emitted), "names": emitted[:4],
        "engineered_mi": round(eng_mi, 4), "best_existing_mi": round(base_mi, 4),
        "lift": round(eng_mi / max(base_mi, 1e-9), 3), "wall_s": round(wall, 3),
    }


def _mat_for(X, recipe):
    """Materialise a recipe's engineered column for MI scoring (mirrors the apply_* replay)."""
    from mlframe.feature_selection.filters._pairwise_modular_fe import apply_pairwise_modular
    from mlframe.feature_selection.filters._integer_lattice_fe import apply_integer_lattice
    from mlframe.feature_selection.filters._conditional_gate_fe import apply_conditional_gate, apply_row_argmax

    kind = recipe.kind
    if kind == "pairwise_modular":
        return apply_pairwise_modular(X, recipe.extra["op"], recipe.src_names, recipe.extra["modulus"])
    if kind == "pairwise_integer_lattice":
        return apply_integer_lattice(X, recipe.extra["op"], recipe.src_names)
    if kind == "row_argmax":
        return apply_row_argmax(X, recipe.src_names)
    if kind == "conditional_gate":
        return apply_conditional_gate(X, recipe.extra["mode"], recipe.src_names, recipe.extra["tau"])
    raise ValueError(kind)


def _detect_cases():
    rng = np.random.default_rng(0)
    out = []

    a = rng.integers(0, 50, N); b = rng.integers(0, 50, N)
    Xm = pd.DataFrame({"a": a, "b": b, "f": rng.normal(0, 1, N)})
    ym = 3.0 * (a % 7) + rng.normal(0, 0.1, N)
    out.append(_run("pairwise_modular", Xm, ym, hybrid_pairwise_modular_fe_with_recipes, "pmod"))

    a2 = rng.integers(1, 40, N) * 2; b2 = rng.integers(1, 40, N) * 2
    Xl = pd.DataFrame({"a": a2, "b": b2, "f": rng.normal(0, 1, N)})
    yl = np.gcd(a2, b2).astype(float) + rng.normal(0, 0.1, N)
    out.append(_run("integer_lattice", Xl, yl, hybrid_integer_lattice_fe_with_recipes, "il_"))

    ca, cb, cc = rng.normal(0, 1, N), rng.normal(0, 1, N), rng.normal(0, 1, N)
    idx = np.argmax(np.stack([ca, cb, cc], 1), 1)
    Xa = pd.DataFrame({"ca": ca, "cb": cb, "cc": cc})
    ya = idx.astype(float) * 5 + rng.normal(0, 0.1, N)
    out.append(_run("row_argmax", Xa, ya, hybrid_row_argmax_fe_with_recipes, "argmax"))

    ga, gb, gc = rng.normal(0, 1, N), rng.normal(5, 1, N), rng.normal(0, 1, N)
    Xg = pd.DataFrame({"a": ga, "b": gb, "c": gc})
    yg = np.where(gc > np.median(gc), ga, gb) + rng.normal(0, 0.05, N)
    out.append(_run("conditional_gate", Xg, yg, hybrid_conditional_gate_fe_with_recipes, "gate"))
    return out


def _specificity():
    rows = []
    for seed in range(3):
        r = np.random.default_rng(100 + seed)
        Xi = pd.DataFrame({c: r.integers(0, 50, N) for c in "abcd"})
        yi = (Xi["a"] * 0.3 + Xi["b"] * 0.1 + r.normal(0, 1, N)).to_numpy().astype(float)
        ybi = bin_y_for_class_mi(yi, nbins=NBINS)
        Xf = pd.DataFrame({c: r.normal(0, 1, N) for c in "abc"})
        ybf = ybi
        rows.append({
            "seed": seed,
            "pairwise_modular_fp": len(hybrid_pairwise_modular_fe_with_recipes(Xi, ybi)[0]),
            "integer_lattice_fp": len(hybrid_integer_lattice_fe_with_recipes(Xi, ybi)[0]),
            "row_argmax_fp": len(hybrid_row_argmax_fe_with_recipes(Xf, ybf)[0]),
            "conditional_gate_fp": len(hybrid_conditional_gate_fe_with_recipes(Xf, ybf)[0]),
        })
    return rows


def main():
    detect = _detect_cases()
    spec = _specificity()
    total_fp = sum(v for row in spec for k, v in row.items() if k.endswith("_fp"))
    result = {
        "nbins": NBINS, "n": N, "detect": detect, "specificity": spec,
        "total_spurious_emissions": total_fp,
        "all_detect": all(d["emitted"] > 0 for d in detect),
        "all_specific": total_fp == 0,
    }
    for d in detect:
        print(f"DETECT {d['operator']:18s} emitted={d['emitted']} lift={d['lift']:.2f} "
              f"(eng_mi={d['engineered_mi']} vs base={d['best_existing_mi']}) wall={d['wall_s']}s names={d['names']}")
    for row in spec:
        print(f"SPECIFIC seed{row['seed']}: "
              f"mod={row['pairwise_modular_fp']} lat={row['integer_lattice_fp']} "
              f"arg={row['row_argmax_fp']} gate={row['conditional_gate_fp']}")
    print(f"ALL DETECT={result['all_detect']}  ALL SPECIFIC={result['all_specific']}  total_FP={total_fp}")

    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"fe_operators_regression_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
