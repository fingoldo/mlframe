"""Fit a CatBoost regressor on the null-MI table and report which fingerprint fields drive the threshold.

The table maps (n rows, bins, target entropy, candidates N) to the 95th percentile of the best-of-N null MI. This script fits a
gradient-boosted model on those rows, reports CatBoost's feature importance, and checks how well the model reproduces the table on
rows it did not see (grouped by n, so an unseen row count is a real test of extrapolation). The printed importance says which
fields the oracle must carry; if the model is accurate enough it can replace the nearest-neighbour lookup in
``filters/_null_mi_threshold.py``. Usage::

    python -m mlframe.feature_selection._benchmarks.fit_null_threshold_booster --table src/mlframe/feature_selection/filters/data/null_mi_table.json --out src/mlframe/feature_selection/filters/data/null_mi_booster.cbm
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _features(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix (log n, bins, entropy, log N, class count) and the log-threshold target."""
    names = ["log_n", "bins", "entropy", "log_candidates", "n_classes"]
    X = np.array(
        [[math.log(e["n"]), e["bins"], e["entropy"], math.log(e["candidates"]), len(e["prior"])] for e in entries],
        dtype=np.float64,
    )
    y = np.array([math.log(max(e["threshold"], 1e-12)) for e in entries], dtype=np.float64)
    return X, y, names


def _make_model(kind: str, seed: int):
    """The regressor for the log-threshold: a scaled, regularised linear model by default, or CatBoost on request."""
    if kind == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, loss_function="RMSE", random_seed=seed, verbose=0)
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))


def _importance(model, names: list[str], kind: str) -> dict:
    """Per-field importance: CatBoost's feature importance, or the absolute standardised coefficients of the linear model."""
    if kind == "catboost":
        return dict(zip(names, [round(float(v), 4) for v in model.get_feature_importance()]))
    coef = model[-1].coef_
    return dict(zip(names, [round(float(abs(v)), 4) for v in coef]))


def fit(table_path: Path, out_path: Path | None, seed: int = 0, kind: str = "ridge") -> dict:
    """Train on all n except the largest held out, report importance and the held-out error, and save the model when asked."""
    import joblib

    entries = list(json.loads(table_path.read_text(encoding="utf-8"))["entries"].values())
    X, y, names = _features(entries)
    n_values = np.array([e["n"] for e in entries])
    held = n_values == n_values.max()
    model = _make_model(kind, seed)
    model.fit(X[~held], y[~held])
    pred_held = model.predict(X[held])
    rmse_held = float(np.sqrt(np.mean((pred_held - y[held]) ** 2)))
    model.fit(X, y)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "catboost":
            model.save_model(str(out_path))
        else:
            joblib.dump(model, out_path)
    return {"model": kind, "importance": _importance(model, names, kind), "rmse_log_heldout_n": round(rmse_held, 4), "n_rows": len(entries)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=("ridge", "catboost"), default="ridge")
    args = ap.parse_args(argv)
    report = fit(args.table, args.out, args.seed, args.model)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
