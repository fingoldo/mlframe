"""Sigmoid-OPTION fix shootout (qual-16): maximum-likelihood Platt fit vs the legacy OLS-on-centred-logit map.

qual-15 found that ``OutputCalibrator(method="sigmoid")`` was NOT a Platt fit -- it regressed ``logit(t_norm)`` on the
normalised prediction by ORDINARY least squares, an objective dominated by the near-asymptote clipping, so the map
barely improved on (and at small OOF was WORSE than) the raw blend. The default therefore stayed ``isotonic``.

qual-16 fixes the sigmoid OPTION itself: ``sigmoid_fit="platt"`` (new default) fits ``sigmoid(A*z+B)`` by minimising the
weighted binomial cross-entropy of the [0,1]-normalised truth -- the maximum-likelihood objective Platt scaling is
actually defined by. The legacy surrogate stays reachable as ``sigmoid_fit="ols_logit"`` for byte-identical replay.

This bench measures, on honest DISJOINT holdouts across multiple seeds + scenarios, that the MLE Platt fit beats both
the legacy OLS-logit map AND the raw blend on the probability-calibration targets where a sigmoid is the right tool,
and is never worse than the OLS-logit map even on the regression S-shape where isotonic ultimately wins.

Run:  python -m mlframe.training.composite.ensemble._benchmarks.bench_sigmoid_platt_vs_ols_logit
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlframe.training.composite.ensemble._calibration import OutputCalibrator

N_OOF_GRID = (60, 150, 400)
N_HOLDOUT = 4000
SEEDS = tuple(range(8))


def _surface(scenario: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_blend_pred, truth) for one scenario."""
    if scenario == "logistic_miscal":
        # Probability target; raw blend is a logistic of the latent signal with the WRONG slope/offset.
        s = rng.normal(0.0, 1.5, size=n)
        ptrue = 1.0 / (1.0 + np.exp(-s))
        y = (rng.uniform(size=n) < ptrue).astype(np.float64)
        raw = 1.0 / (1.0 + np.exp(-(0.4 * s - 0.5)))
        return raw.astype(np.float64), y
    if scenario == "overconfident":
        # Probability target; raw blend is over-sharp (temperature too low) -- needs softening.
        s = rng.normal(0.0, 1.2, size=n)
        ptrue = 1.0 / (1.0 + np.exp(-s))
        y = (rng.uniform(size=n) < ptrue).astype(np.float64)
        raw = 1.0 / (1.0 + np.exp(-1.8 * s))
        return raw.astype(np.float64), y
    if scenario == "regression_s_shape":
        # Regression target with a tanh-squash miscalibration (isotonic's turf; Platt must not be worse than OLS-logit).
        s = rng.normal(0.0, 1.0, size=n)
        y = 2.0 * s + rng.normal(0.0, 0.3, size=n)
        raw = 3.0 * np.tanh(0.8 * s) + rng.normal(0.0, 0.1, size=n)
        return raw.astype(np.float64), y.astype(np.float64)
    raise ValueError(scenario)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run() -> dict:
    results: dict = {}
    for scenario in ("logistic_miscal", "overconfident", "regression_s_shape"):
        for n_oof in N_OOF_GRID:
            platt, ols, raw = [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(7919 * (1 + N_OOF_GRID.index(n_oof)) + seed + hash(scenario) % 101)
                p_oof, y_oof = _surface(scenario, n_oof, rng)
                p_hold, y_hold = _surface(scenario, N_HOLDOUT, rng)
                raw.append(_rmse(p_hold, y_hold))
                cp = OutputCalibrator(method="sigmoid", sigmoid_fit="platt").fit(p_oof, y_oof)
                co = OutputCalibrator(method="sigmoid", sigmoid_fit="ols_logit").fit(p_oof, y_oof)
                platt.append(_rmse(cp.predict(p_hold), y_hold))
                ols.append(_rmse(co.predict(p_hold), y_hold))
            pa, oa, ra = np.asarray(platt), np.asarray(ols), np.asarray(raw)
            results[f"{scenario}|n_oof={n_oof}"] = {
                "platt_median": float(np.median(pa)),
                "ols_logit_median": float(np.median(oa)),
                "raw_median": float(np.median(ra)),
                "platt_beats_ols": int(np.sum(pa < oa)),
                "platt_beats_raw": int(np.sum(pa < ra)),
                "n_seeds": len(SEEDS),
            }
    return results


def main() -> None:
    res = run()
    out_dir = Path(__file__).resolve().parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sigmoid_platt_vs_ols_logit.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"{'cell':<32} {'raw':>8} {'ols_logit':>10} {'platt':>8}  P<OLS  P<raw")
    for key, s in res.items():
        print(
            f"{key:<32} {s['raw_median']:>8.4f} {s['ols_logit_median']:>10.4f} {s['platt_median']:>8.4f}"
            f"  {s['platt_beats_ols']:>3}/{s['n_seeds']}  {s['platt_beats_raw']:>3}/{s['n_seeds']}"
        )


if __name__ == "__main__":
    main()
