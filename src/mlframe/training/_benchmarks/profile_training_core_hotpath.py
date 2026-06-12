"""Discovery cProfile of the TRAINING-CORE hot path (``train_mlframe_models_suite``).

Goal: surface NEW mlframe-side optimization leads at two representative shapes
(small ~2k rows, production-ish ~100k rows x ~30 cols, mixed numeric+cat),
CPU-only, with a lightweight model subset (hgb only, few rounds) so the whole
run finishes well under 3 minutes.

Why this exists separately from ``_profile_fuzz_1m``: that harness randomises a
huge axis menu per-seed (MRMR/boruta/outlier/composite/save+load+parity round
trip) which is great for bug-hunting but BURIES the steady-state training-core
hot path under one-off heavy stages. This harness pins the SIMPLEST suite
config so the cProfile attribution lands on the always-on core path.

cProfile attribution caveat (calibrate before trusting a hotspot): cProfile
inflates deep pandas/sklearn/polars call stacks ~10-13x vs a standalone wall
microbench. A function flagged hot by cumulative time may be pure attribution
noise. ALWAYS confirm a lead with an isolated wall-time microbench (see the
``--microbench`` mode at the bottom) before declaring it optimizable.

Run (CPU-only):
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.profile_training_core_hotpath

Output: JSON summary -> sibling ``_results/training_core_hotpath.json``.
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

# Force CPU-only BEFORE any heavy import that might init CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Native DLL-load-order guard: on this Windows store-Python a bare
# ``from mlframe.training.core import ...`` segfaults (0xC0000005) due to a
# native-lib load-order conflict in the package-init chain; pre-importing the
# pipeline module loads the conflicting runtimes in the safe order first.
import mlframe.training.pipeline as _pipeline_preimport  # noqa: E402,F401

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _profile_one(target_type: str, n_rows: int, models: tuple[str, ...], seed: int, top_n: int) -> dict[str, Any]:
    """Profile ONE train_mlframe_models_suite call at a fixed simple config.

    Returns a dict with wall time, status, and the cumulative+tottime top-N
    profile tables as text.
    """
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        TargetTypes, BaselineDiagnosticsConfig, DummyBaselinesConfig,
        OutputConfig, ReportingConfig, FeatureSelectionConfig,
        CompositeTargetDiscoveryConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from ._profile_fuzz_1m import _make_synthetic_frame

    df = _make_synthetic_frame(target_type, n_rows, seed=seed)
    n_cols = len(df.columns)

    if target_type == "regression":
        fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["y"])
    elif target_type == "binary_classification":
        fte = SimpleFeaturesAndTargetsExtractor(
            classification_targets=["y"], classification_exact_values={"y": 1},
        )
    else:
        fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["y"])

    # Simplest always-on core config: no MRMR, no boruta, no outlier detection,
    # no composite discovery, no ensembles, no diagnostics/dummy baselines, no
    # save/predict. This isolates the train-core path (FTE -> split -> preprocess
    # pipeline -> single hgb fit -> metrics) that EVERY suite call runs.
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
        status = f"{type(e).__name__}: {e}"[:200]
    finally:
        profiler.disable()
    wall = time.perf_counter() - t0

    def _table(sort_key: str) -> str:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(sort_key).print_stats(top_n)
        return s.getvalue()

    return {
        "target_type": target_type,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "models": list(models),
        "seed": seed,
        "wall_s": round(wall, 3),
        "status": status,
        "cumulative_top": _table("cumulative"),
        "tottime_top": _table("tottime"),
    }


def main(top_n: int = 25) -> dict[str, Any]:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Small + production-ish. Seed chosen so _make_synthetic_frame yields a mixed
    # numeric+cat frame (~30 cols incl. injected cat/correlated columns).
    # NOTE: seed=11 synthetic emits an ``emb`` object-dtype column holding numpy
    # arrays per cell. The always-on honest-diagnostics drift block classifies it
    # categorical and runs pandas value_counts(dropna=False) over ndarray cells ->
    # pandas PyObjectHashTable O(n) ndarray-hashing blowup (4.14s wall @ n=2000,
    # minutes @ 100k -- this is THE top lead). Production shape capped at 40k so the
    # harness finishes < 3 min; 100k hangs in exactly this path (the bug).
    shapes = [
        ("regression", 2_000, 11),
        ("regression", 40_000, 11),
    ]
    results = []
    for tt, n, seed in shapes:
        print(f"=== profiling {tt} n_rows={n:_} seed={seed} (hgb, CPU-only) ===", flush=True)
        r = _profile_one(tt, n, models=("hgb",), seed=seed, top_n=top_n)
        print(f"  wall={r['wall_s']}s status={r['status']} cols={r['n_cols']}", flush=True)
        print(r["cumulative_top"], flush=True)
        print("--- tottime ---", flush=True)
        print(r["tottime_top"], flush=True)
        results.append(r)

    out = {"label": "profile-training-core-hotpath", "results": results}
    out_path = _RESULTS_DIR / "training_core_hotpath.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
