"""End-to-end ``train_mlframe_models_suite`` profile on 1M-row inputs.

Acknowledges the explicit gap noted in the dummy_baselines wave: smoke
tests + unit tests run on 5K rows; nothing exercised the suite end-to-
end at production shapes (1M+ rows). Per the mlframe CLAUDE.md profile
rule.

Builds a synthetic 1M-row frame per target_type (regression / binary /
multiclass) and runs ``train_mlframe_models_suite`` under cProfile.
Reports per-combo wall time + cProfile top-N cumulative.

Usage::

    python -m mlframe.training._profile_fuzz_1m
    python -m mlframe.training._profile_fuzz_1m --target regression
    python -m mlframe.training._profile_fuzz_1m --n-rows 500000 --top 50
    python -m mlframe.training._profile_fuzz_1m --models cb,xgb,lgb
    python -m mlframe.training._profile_fuzz_1m --save-charts  # surface kaleido cost

== Findings (2026-05-10, n_rows=1M, regression x lgb) ==

Without chart saving (--save-charts NOT set):

  Total wall: 21.6s
  - LGB engine.train + Booster.update:    14.3s  (library bound)
  - numba JIT compilation cold-start:      4.6s  (one-time per process;
                                                  amortized across targets)
  - report_regression_model_perf:          0.1s  (standalone direct-bench
                                                  on 1M rows; the e2e attribution
                                                  noise inflates this to ~2s
                                                  via cProfile deep-stack overhead)
  - All other suite overhead:              2.6s  (split / preprocess /
                                                  pipeline / save / metadata)

With chart saving (--save-charts, plot_outputs=plotly[html,png] default):

  Total wall: 98.4s  (76s extra over the no-chart baseline)
  Dominant cost: kaleido PNG export -- each plotly figure triggers a
  Chromium ``page.reload()`` that takes 12-15s while plotly.js re-mounts
  in the headless browser. On a 4-model x val+test x N-ensemble suite
  this can balloon to MINUTES of pure chart-export wall-time.

  matplotlib backend init (one-time): 20.5s  (Qt backend)

== Mitigation ==

  plot_outputs='matplotlib[png]'  -- 10-20x faster PNG (no Chromium)
  plot_outputs='plotly[html]'     -- HTML only, no PNG, no kaleido,
                                     instant; HTML is interactive in
                                     jupyter and shareable as a file
  plot_inline_display=False       -- skip inline render in jupyter (env
                                     var or ReportingConfig knob)

The suite warns at startup when ``save_charts=True AND
plot_outputs`` contains both ``plotly`` and ``png`` (see core.py
``[reporting] plot_outputs=...`` warning).

== cProfile attribution noise calibration ==

cProfile inflates pandas / sklearn / matplotlib / plotly deep-stack call
timings ~10-13x vs standalone wall-time microbench. When this harness
flags a function as a hotspot at 2-3s cumtime, cross-check by isolating
the function in a direct microbench (mostly the cumulative time turns
out to be 100-300ms standalone, not seconds).

The HONEST mlframe-side hotspots on 1M-row regression are: numba JIT
cold-start (one-time, amortized) + the chart-export path (only when
plot_outputs has plotly+png). Everything else is library-bounded.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _make_synthetic_frame(target_type: str, n_rows: int, seed: int = 42):
    """Build a synthetic 1M-row pandas frame (8 numeric features +
    target). Polars input would exercise more code paths but pandas
    keeps the harness self-contained without polars edge cases."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    cols = {
        f"x{i}": rng.normal(size=n_rows).astype("float32")
        for i in range(6)
    }
    # Add a low-card categorical (int) and a moderate-card group_id-like
    # column to stress the cat / group paths.
    cols["c_low"] = rng.integers(0, 5, n_rows).astype("int32")
    cols["c_mid"] = rng.integers(0, 50, n_rows).astype("int32")

    if target_type == "regression":
        y = (
            2.0 * cols["x0"]
            - 1.5 * cols["x1"]
            + 0.5 * cols["x2"] * cols["x3"]
            + rng.normal(0, 0.5, n_rows).astype("float32")
        )
        cols["y"] = y.astype("float32")
    elif target_type == "binary_classification":
        logit = 1.5 * cols["x0"] - 0.8 * cols["x1"] + 0.3 * cols["x2"]
        prob = 1.0 / (1.0 + np.exp(-logit))
        cols["y"] = (rng.uniform(0, 1, n_rows) < prob).astype("int32")
    elif target_type == "multiclass_classification":
        scores = np.column_stack([
            1.5 * cols["x0"] + rng.normal(0, 0.3, n_rows),
            -1.0 * cols["x1"] + rng.normal(0, 0.3, n_rows),
            0.5 * cols["x2"] + rng.normal(0, 0.3, n_rows),
            -0.5 * cols["x3"] + rng.normal(0, 0.3, n_rows),
        ])
        cols["y"] = scores.argmax(axis=1).astype("int32")
    elif target_type == "multilabel_classification":
        K = 4
        for k in range(K):
            logit = (
                rng.uniform(-1, 1) * cols["x0"]
                + rng.uniform(-1, 1) * cols["x1"]
                + rng.normal(0, 0.3, n_rows)
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            cols[f"y_{k}"] = (rng.uniform(0, 1, n_rows) < prob).astype("int32")
    else:
        raise ValueError(f"unsupported target_type {target_type!r}")
    return pd.DataFrame(cols)


def _run_suite_profiled(
    target_type: str,
    n_rows: int,
    models: tuple[str, ...],
    seed: int,
    top_n: int,
    save_charts: bool = False,
) -> tuple[float, bool, str, str]:
    """Returns (wall_seconds, ok, suite_status, profile_text)."""
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        TargetTypes, BaselineDiagnosticsConfig, DummyBaselinesConfig,
        OutputConfig, ReportingConfig, CompositeTargetDiscoveryConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = _make_synthetic_frame(target_type, n_rows, seed=seed)
    print(f"  built frame: {len(df):_} rows x {len(df.columns)} cols")

    if target_type == "regression":
        target_col = "y"
        fte_kwargs = dict(regression_targets=["y"])
        _tt = TargetTypes.REGRESSION
    elif target_type == "binary_classification":
        target_col = "y"
        fte_kwargs = dict(classification_exact_values={"y": 1})
        _tt = TargetTypes.BINARY_CLASSIFICATION
    elif target_type == "multiclass_classification":
        target_col = "y"
        fte_kwargs = dict(classification_targets=["y"])
        _tt = TargetTypes.MULTICLASS_CLASSIFICATION
    elif target_type == "multilabel_classification":
        # Multilabel needs a different FTE setup; out of scope for
        # SimpleFeaturesAndTargetsExtractor's defaults — skip from
        # this profile harness for now (TODO: extend when needed).
        return 0.0, False, "MULTILABEL_FTE_SETUP_OOS", ""
    else:
        return 0.0, False, "UNSUPPORTED_TARGET_TYPE", ""

    fte = SimpleFeaturesAndTargetsExtractor(**fte_kwargs)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    status = "OK"
    try:
        train_mlframe_models_suite(
            df=df,
            target_name=target_col,
            model_name="prof",
            features_and_targets_extractor=fte,
            mlframe_models=list(models),
            verbose=0,
            output_config=OutputConfig(
                data_dir=("data" if save_charts else ""),
                models_dir=("models" if save_charts else ""),
                save_charts=save_charts,
            ),
            # Disable expensive auxiliary diagnostics so we measure
            # the SUITE path proper, not the addons:
            composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
            baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
            dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            reporting_config=ReportingConfig(
                # plotly[html,png] is the prod default that tripped the
                # user's 5M run with kaleido cycles dominating. Use that
                # when --save-charts is set; otherwise matplotlib (no
                # kaleido cost).
                plot_outputs=("plotly[html,png]" if save_charts else "matplotlib[png]"),
                plot_inline_display=False,
            ),
        )
    except Exception as e:
        status = f"{type(e).__name__}: {e}"[:120]
    finally:
        profiler.disable()
    wall = time.perf_counter() - t0

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    return wall, status == "OK", status, s.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rows", type=int, default=1_000_000)
    p.add_argument("--target", default="all",
                   choices=("all", "regression", "binary_classification",
                            "multiclass_classification"))
    p.add_argument("--models", default="cb",
                   help="Comma-separated model list (cb,xgb,lgb,linear). "
                        "Default 'cb' to bound per-combo wall time.")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-charts", action="store_true",
                   help="Enable chart saving (default off — measures core suite, "
                        "not chart export). Use to surface plotly+kaleido cost.")
    args = p.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())

    targets: list[str] = (
        ["regression", "binary_classification", "multiclass_classification"]
        if args.target == "all" else [args.target]
    )

    print(f"# 1M-row e2e profile (n_rows={args.n_rows:_}, models={models}, "
          f"save_charts={args.save_charts})")
    summary: list[tuple[str, float, str]] = []
    for tt in targets:
        label = f"{tt} x {','.join(models)}"
        print(f"\n=== {label} ===")
        wall, ok, status, prof = _run_suite_profiled(
            tt, args.n_rows, models, args.seed, args.top,
            save_charts=args.save_charts,
        )
        summary.append((label, wall, status))
        print(f"  wall: {wall:.1f}s  status: {status}")
        print(prof[:6000])

    print("\n# Wall-time summary:")
    for label, t, status in summary:
        print(f"  {label:<55} {t:>7.1f}s  {status}")


if __name__ == "__main__":
    main()
