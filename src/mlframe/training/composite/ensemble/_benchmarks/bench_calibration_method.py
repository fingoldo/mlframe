"""Cross-target output-calibration method shootout: isotonic vs sigmoid vs linear.

Question (qual-15): is ``cross_target_calibration_method="isotonic"`` the right DEFAULT for the cross-target ensemble's
post-hoc output recalibration, given the calibrator is fit on a SMALL OOF holdout (the default ``oof_holdout_frac=0.2``
of train, often only a few hundred rows)?

Known-ML hypothesis (Niculescu-Mizil & Caruana 2005): free-form isotonic (PAV, ~n step parameters) overfits a small
calibration set; the 2-parameter Platt/sigmoid map generalises better at small n. So the accurate DEFAULT may depend on
OOF size: sigmoid at small OOF, isotonic only once OOF is large enough for the step map not to overfit.

Method (honest holdout, per CLAUDE.md "most accurate default first; majority over seeds+scenarios"):
  * Build a biased 1-D ensemble-output surface (raw blend p) whose miscalibration vs truth y is a known monotone
    distortion: SCENARIO A = strong S-shape (logistic squashing), SCENARIO B = saturating (sqrt-like compression).
  * Split into OOF (fit calibrator) + an INDEPENDENT honest holdout (score). Sweep OOF size n_oof in {60, 150, 400, 1500}.
  * For each method fit on OOF, predict on holdout, score holdout RMSE. >=8 seeds per cell.
  * Report per-cell median holdout RMSE + win counts; aggregate verdict by majority.

Run:  python -m mlframe.training.composite.ensemble._benchmarks.bench_calibration_method
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from mlframe.training.composite.ensemble._calibration import OutputCalibrator

METHODS = ("isotonic", "sigmoid", "linear")
N_OOF_GRID = (60, 150, 400, 1500)
N_HOLDOUT = 4000
SEEDS = tuple(range(8))


def _make_surface(scenario: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_blend_pred, truth) with a known monotone miscalibration + noise."""
    # latent signal the ensemble ranks correctly
    s = rng.normal(0.0, 1.0, size=n)
    # truth is a clean monotone function of the latent signal (+ noise)
    y = 2.0 * s + rng.normal(0.0, 0.3, size=n)
    if scenario == "s_shape":
        # raw blend squashes the extremes (classic S-miscalibration a least-squares blend leaves behind)
        p = 3.0 * np.tanh(0.8 * s) + rng.normal(0.0, 0.1, size=n)
    elif scenario == "saturating":
        # raw blend saturates on the high side (sqrt-like compression)
        p = np.sign(s) * np.sqrt(np.abs(s)) * 2.0 + rng.normal(0.0, 0.1, size=n)
    else:
        raise ValueError(scenario)
    return p.astype(np.float64), y.astype(np.float64)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run() -> dict:
    results: dict = {}
    for scenario in ("s_shape", "saturating"):
        for n_oof in N_OOF_GRID:
            cell = {m: [] for m in METHODS}
            cell["raw"] = []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * (1 + N_OOF_GRID.index(n_oof)) + seed + hash(scenario) % 97)
                p_oof, y_oof = _make_surface(scenario, n_oof, rng)
                p_hold, y_hold = _make_surface(scenario, N_HOLDOUT, rng)
                cell["raw"].append(_rmse(p_hold, y_hold))
                for m in METHODS:
                    cal = OutputCalibrator(method=m).fit(p_oof, y_oof)
                    pred = cal.predict(p_hold)
                    cell[m].append(_rmse(pred, y_hold))
            key = f"{scenario}|n_oof={n_oof}"
            summary = {m: {"median": float(np.median(cell[m])), "mean": float(np.mean(cell[m]))} for m in (*METHODS, "raw")}
            # per-seed winner among the three methods
            arr = {m: np.asarray(cell[m]) for m in METHODS}
            iso_beats_sig = int(np.sum(arr["isotonic"] < arr["sigmoid"]))
            sig_beats_iso = int(np.sum(arr["sigmoid"] < arr["isotonic"]))
            summary["iso_beats_sig"] = iso_beats_sig
            summary["sig_beats_iso"] = sig_beats_iso
            summary["n_seeds"] = len(SEEDS)
            results[key] = summary
    return results


def main() -> None:
    res = run()
    out_dir = Path(__file__).resolve().parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "calibration_method_shootout.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"{'cell':<28} {'raw':>8} {'isotonic':>9} {'sigmoid':>9} {'linear':>8}  iso>sig sig>iso")
    for key, s in res.items():
        print(
            f"{key:<28} {s['raw']['median']:>8.4f} {s['isotonic']['median']:>9.4f} "
            f"{s['sigmoid']['median']:>9.4f} {s['linear']['median']:>8.4f}"
            f"  {s['sig_beats_iso']:>7}/{s['n_seeds']} {s['iso_beats_sig']:>3}/{s['n_seeds']}"
        )


if __name__ == "__main__":
    main()
