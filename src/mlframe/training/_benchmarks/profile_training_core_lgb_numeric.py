"""Discovery cProfile of the TRAINING-CORE hot path on a NUMERIC + LIGHT-CAT
frame at ~30 columns, two shapes (small ~2k, prod-ish ~100k), CPU-only, lgb/hgb.

Context: the drift-PSI array-cell value_counts blowup (the prior dominant
hotspot) was removed in 55067dc3. This harness re-profiles the now-visible
NEXT mlframe-side hotspot. Unlike ``profile_training_core_hotpath`` (which uses
the seeded fuzz frame capped at ~12 cols and seed-11's object/embedding cell
column), this builds a clean wide ~30-col numeric + a couple of low/mid-card
categorical columns -- the requested "lgb/hgb on numeric+light-cat" config.

cProfile attribution caveat: cProfile inflates deep pandas/sklearn/polars stacks
~10-13x vs standalone wall microbench. A cumulative-hot function may be pure
attribution noise -- ALWAYS confirm a lead with an isolated wall microbench
before declaring it optimizable.

Run (CPU-only):
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.profile_training_core_lgb_numeric

Output -> sibling ``_results/training_core_lgb_numeric.json``.
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Native DLL-load-order guard (same as sibling harness): pre-import pipeline so
# the conflicting native runtimes load in the safe order before training.core.
import mlframe.training.pipeline as _pipeline_preimport  # noqa: E402,F401

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _make_numeric_cat_frame(n_rows: int, n_numeric: int = 26, seed: int = 7):
    """Wide numeric + light-cat pandas frame: ``n_numeric`` float32 features,
    3 low/mid-card categorical (object) columns, 1 informative regression target.
    Total cols ~= n_numeric + 4 (~30 with default n_numeric=26)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    cols: dict[str, Any] = {}
    X = rng.normal(size=(n_rows, n_numeric)).astype("float32")
    for i in range(n_numeric):
        cols[f"x{i}"] = X[:, i]
    # Light categoricals (string/object) -- low + mid cardinality.
    cols["c_low"] = rng.integers(0, 4, size=n_rows).astype("int32")
    cols["c_mid"] = rng.integers(0, 40, size=n_rows).astype("int32")
    cols["cat_low"] = np.asarray([f"g{v}" for v in rng.integers(0, 5, size=n_rows)], dtype=object)
    # Informative linear-ish target from the first few numeric cols + noise.
    y = (1.7 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.5, size=n_rows)).astype("float32")
    cols["y"] = y
    return pd.DataFrame(cols)


def _profile_one(n_rows: int, models: tuple[str, ...], seed: int, top_n: int) -> dict[str, Any]:
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, DummyBaselinesConfig, OutputConfig,
        ReportingConfig, FeatureSelectionConfig, CompositeTargetDiscoveryConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = _make_numeric_cat_frame(n_rows, seed=seed)
    n_cols = len(df.columns)
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    fs_cfg = FeatureSelectionConfig(use_mrmr_fs=False, use_boruta_shap=False)

    profiler = cProfile.Profile()
    status = "OK"
    t0 = time.perf_counter()
    profiler.enable()
    try:
        train_mlframe_models_suite(
            df=df,
            target_name="y",
            model_name="prof",
            features_and_targets_extractor=fte,
            mlframe_models=list(models),
            use_mlframe_ensembles=False,
            feature_selection_config=fs_cfg,
            composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
            baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
            dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            output_config=OutputConfig(data_dir="", models_dir="", save_charts=False),
            reporting_config=ReportingConfig(plot_outputs="matplotlib[png]", plot_inline_display=False),
            verbose=0,
        )
    except Exception as e:  # noqa: BLE001
        status = f"{type(e).__name__}: {e}"[:300]
    finally:
        profiler.disable()
    wall = time.perf_counter() - t0

    def _table(sort_key: str) -> str:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(sort_key).print_stats(top_n)
        return s.getvalue()

    return {
        "n_rows": n_rows, "n_cols": n_cols, "models": list(models), "seed": seed,
        "wall_s": round(wall, 3), "status": status,
        "cumulative_top": _table("cumulative"), "tottime_top": _table("tottime"),
    }


def main(top_n: int = 25) -> dict[str, Any]:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shapes = [(2_000, 7), (100_000, 7)]
    results = []
    for n, seed in shapes:
        print(f"=== profiling regression n_rows={n:_} seed={seed} (hgb, numeric+light-cat, CPU-only) ===", flush=True)
        r = _profile_one(n, models=("hgb",), seed=seed, top_n=top_n)
        print(f"  wall={r['wall_s']}s status={r['status']} cols={r['n_cols']}", flush=True)
        print(r["cumulative_top"], flush=True)
        print("--- tottime ---", flush=True)
        print(r["tottime_top"], flush=True)
        results.append(r)

    out = {"label": "profile-lgb-numeric", "results": results}
    out_path = _RESULTS_DIR / "training_core_lgb_numeric.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
