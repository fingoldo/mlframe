"""Render every composable chart type to PNG so the figures can be LOOKED at, not just unit-tested.

The reporting suite asserts on FigureSpec contents, which is the right thing to assert on and is blind to
the whole class of defect a reader sees first: a legend covering the curves it labels, a rotated tick
label landing on the panel below, a three-line title pushing into its neighbour, a diverging colour scale
whose midpoint sits nowhere near zero. Those need pixels.

Inputs are deliberately ordinary rather than adversarial -- a few thousand rows, six to nine
classes/labels/models, and identifier-shaped names of the length this codebase really produces (a
``categorical_feature_8_with_long_name`` is not a stress test, it is Tuesday). A defect that shows up here
shows up for a user on defaults.

Run: ``python profiling/render_every_chart.py --out <dir>`` then open the directory.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd

from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save

N = 3000


def _binary(rng):
    """Labels plus a separable-but-imperfect score."""
    y = (rng.random(N) < 0.3).astype(int)
    return y, np.clip(0.25 * y + rng.random(N) * 0.7, 0, 1)


def build_specs(rng) -> Dict[str, Callable[[], object]]:
    """Map chart name -> a zero-arg builder returning a FigureSpec."""
    from mlframe.reporting.charts.binary import compose_binary_figure
    from mlframe.reporting.charts.category_discriminability import compose_category_discriminability_figure
    from mlframe.reporting.charts.ltr import compose_ltr_figure
    from mlframe.reporting.charts.model_comparison import compose_model_comparison_figure
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure
    from mlframe.reporting.charts.multilabel import compose_multilabel_figure
    from mlframe.reporting.charts.prediction_stability import compose_prediction_stability_figure
    from mlframe.reporting.charts.quantile import compose_quantile_figure
    from mlframe.reporting.charts.regression import compose_regression_figure
    from mlframe.reporting.charts.temporal import compose_target_acf_figure
    from mlframe.reporting.charts.training_curve import compose_training_curve_figure

    y, score = _binary(rng)

    k = 8
    classes = [f"segment_{i}_high_value" for i in range(k)]
    yt_idx = rng.integers(0, k, N)
    logit = rng.normal(size=(N, k))
    logit[np.arange(N), yt_idx] += 1.5
    proba = np.exp(logit)
    proba /= proba.sum(1, keepdims=True)
    yt_named = np.array([classes[i] for i in yt_idx])

    n_labels = 6
    labels = [f"label_{i}_long_descriptive_name" for i in range(n_labels)]
    yt_ml = (rng.random((N, n_labels)) < 0.3).astype(int)
    proba_ml = np.clip(yt_ml * 0.4 + rng.random((N, n_labels)) * 0.6, 0, 1)

    y_reg = rng.normal(1000, 300, N)
    y_hat = y_reg + rng.normal(0, 120, N)
    alphas = [0.1, 0.5, 0.9]
    preds = np.stack([y_reg - 150, y_reg, y_reg + 150], 1) + rng.normal(0, 50, (N, 3))

    groups = np.repeat(np.arange(300), 10)
    rel = rng.integers(0, 4, N)

    per_model = {
        f"{m}_v{v}": {"y_true": y, "y_score": np.clip(0.25 * y + rng.random(N) * 0.7, 0, 1)}
        for m in ("lightgbm_dart", "catboost_lossguide", "xgboost_hist", "random_forest_balanced")
        for v in (1, 2)
    }

    cat_X = pd.DataFrame({f"categorical_feature_{i}_with_long_name": rng.integers(0, 12, N).astype(str) for i in range(9)})
    history = {
        "train": {"logloss": list(np.linspace(0.7, 0.35, 120)), "auc": list(np.linspace(0.6, 0.9, 120))},
        "valid": {"logloss": list(np.linspace(0.7, 0.45, 120)), "auc": list(np.linspace(0.6, 0.83, 120))},
    }

    return {
        "binary": lambda: compose_binary_figure(y, score, suptitle="LightGBM | fold 3 | holdout 2026-Q2"),
        "multiclass": lambda: compose_multiclass_figure(yt_named, proba, classes, suptitle="CatBoost multiclass"),
        "multilabel": lambda: compose_multilabel_figure(yt_ml, proba_ml, labels, suptitle="Multilabel"),
        "regression": lambda: compose_regression_figure(y_reg, y_hat, suptitle="XGBoost regression", metrics_str="RMSE=119.8 MAE=95.2 R2=0.84"),
        "quantile": lambda: compose_quantile_figure(y_reg, preds, alphas, suptitle="Quantile"),
        "ltr": lambda: compose_ltr_figure(rel, rel + rng.normal(0, 1, N), groups, suptitle="LambdaMART"),
        "model_comparison": lambda: compose_model_comparison_figure(per_model, "binary_classification", suptitle="Model comparison"),
        "temporal": lambda: compose_target_acf_figure(np.cumsum(rng.normal(size=2000)), suptitle="Target ACF"),
        "training_curve": lambda: compose_training_curve_figure(history, es_iteration=95, suptitle="Training curve"),
        "prediction_stability": lambda: compose_prediction_stability_figure(rng.random((N, 7)), y_true=y, suptitle="Prediction stability"),
        "category_discriminability": lambda: compose_category_discriminability_figure(cat_X, y, list(cat_X.columns), suptitle="Category discriminability"),
    }


def main() -> None:
    """Render every chart, printing per-chart compose and render wall time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the PNGs into")
    ap.add_argument("--backend", default="matplotlib[png]", help="plot-output DSL clause")
    args = ap.parse_args()

    output = parse_plot_output_dsl(args.backend)
    rng = np.random.default_rng(0)
    for name, build in build_specs(rng).items():
        try:
            t0 = time.perf_counter()
            spec = build()
            t_compose = time.perf_counter() - t0
            t0 = time.perf_counter()
            render_and_save(spec, output, f"{args.out}/{name}")
            print(f"OK   {name:26} compose={t_compose * 1000:7.1f}ms render={(time.perf_counter() - t0) * 1000:8.1f}ms")
        except Exception as exc:  # noqa: PERF203 -- a broken chart must not stop the sweep; the point is to see them all
            print(f"FAIL {name:26} {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
