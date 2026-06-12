"""Discovery cProfile of the CatBoost + categorical-heavy + MULTILABEL train path.

Goal: surface NEW mlframe-side optimization leads on a code path the numeric-LGB
``profile_training_core_hotpath`` harness never touches: CatBoost on a frame with
several string/category columns + some numeric, in a MULTILABEL target config
(``target`` = pandas object column of K-int lists). This exercises:
  - native CB categorical-feature resolution + Pool build (skip_categorical_encoding),
  - the CB ``text_processing`` occurrence_lower_bound calibration (compute_cb_text_processing),
  - the multilabel eval_set carve in the splitter / _setup_eval_set,
  - the multilabel metrics + report path (per-label binarised reports).

Shapes: small ~2k + production-ish ~50k rows, CPU-only, few CB iterations so the
whole run finishes well under 3 min.

cProfile attribution caveat (calibrate before trusting a hotspot): cProfile inflates
deep pandas/sklearn/polars call stacks ~10-13x vs a standalone wall microbench. A
function flagged hot by cumulative time may be pure attribution noise -- ALWAYS
confirm a lead with an isolated wall-time microbench before declaring it optimizable.
Only mlframe-side (or pyutilz, which IS ours) hotspots count; external catboost /
sklearn C code is not ours.

Run (CPU-only):
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.profile_cb_multilabel_cat

Output: JSON summary -> sibling ``_results/cb_multilabel_cat.json``.
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Force CPU-only BEFORE any heavy import that might init CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Native DLL-load-order guard (mirrors profile_training_core_hotpath): pre-import
# the pipeline module so conflicting native runtimes load in the safe order first.
import mlframe.training.pipeline as _pipeline_preimport  # noqa: E402,F401

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"
# The proven multilabel-capable test mock FTE lives under tests/training/shared.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TESTS_DIR = _REPO_ROOT / "tests" / "training"


def _make_cat_heavy_multilabel_frame(n_rows: int, *, n_labels: int = 4, seed: int = 0):
    """pandas frame: several string/category cols + some numeric + a multilabel
    ``target`` object column (each cell a list of K ints).

    The labels are learnable from the numeric + (encoded) cat columns so CB has
    real signal and the eval_set / metrics path runs on non-degenerate scores.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Numeric block.
    f0 = rng.normal(size=n_rows).astype("float32")
    f1 = rng.normal(size=n_rows).astype("float32")
    f2 = rng.normal(size=n_rows).astype("float32")
    data: dict = {"f0": f0, "f1": f1, "f2": f2}

    # String-categorical block (the cat-heavy axis): low / mid / high card +
    # one pandas category-dtype column, so cat resolution + Pool build are stressed.
    lvl5 = np.array(["A", "B", "C", "D", "E"], dtype=object)
    data["cat_low"] = lvl5[rng.integers(0, 5, n_rows)]
    lvl50 = np.array([f"M{j:02d}" for j in range(50)], dtype=object)
    data["cat_mid"] = lvl50[rng.integers(0, 50, n_rows)]
    lvl300 = np.array([f"H{j:03d}" for j in range(300)], dtype=object)
    data["cat_high"] = lvl300[rng.integers(0, 300, n_rows)]

    df = pd.DataFrame(data)
    # One genuine pandas category dtype column (the other strings stay object).
    df["cat_cat"] = pd.Series(lvl5[rng.integers(0, 5, n_rows)]).astype("category")

    # Cat-driven signal: map a couple cat levels to a logit bump per label.
    cat_low_code = pd.Categorical(df["cat_low"], categories=list(lvl5)).codes.astype("float32")

    labels = np.empty((n_rows, n_labels), dtype=np.int8)
    for k in range(n_labels):
        logit = (
            rng.uniform(-1, 1) * f0
            + rng.uniform(-1, 1) * f1
            + 0.4 * (cat_low_code - 2.0)
            + rng.normal(0, 0.3, n_rows)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        labels[:, k] = (rng.uniform(0, 1, n_rows) < prob).astype(np.int8)
    df["target"] = pd.Series(list(labels), dtype=object)
    return df


def _profile_one(n_rows: int, *, n_labels: int, iterations: int, seed: int, top_n: int) -> dict[str, Any]:
    """Profile ONE CB multilabel cat-heavy train_mlframe_models_suite call."""
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        TargetTypes, BaselineDiagnosticsConfig, DummyBaselinesConfig,
        OutputConfig, ReportingConfig, FeatureSelectionConfig,
        CompositeTargetDiscoveryConfig,
    )

    # Import the proven multilabel-capable test mock FTE.
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config  # type: ignore

    df = _make_cat_heavy_multilabel_frame(n_rows, n_labels=n_labels, seed=seed)
    n_cols = len(df.columns)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target", regression=False,
        target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
    )
    # No MRMR / boruta / composite / diagnostics: isolate the CB cat+multilabel core path.
    fs_cfg = FeatureSelectionConfig(use_mrmr_fs=False, use_boruta_shap=False)

    profiler = cProfile.Profile()
    status = "OK"
    t0 = time.perf_counter()
    profiler.enable()
    try:
        train_mlframe_models_suite(
            df=df,
            target_name="cb_ml",
            model_name="cb_ml",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config=get_cpu_config("cb", iterations),
            use_ordinary_models=True,
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
        import traceback
        status = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
    finally:
        profiler.disable()
    wall = time.perf_counter() - t0

    def _table(sort_key: str) -> str:
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats(sort_key).print_stats(top_n)
        return s.getvalue()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_labels": n_labels,
        "iterations": iterations,
        "seed": seed,
        "wall_s": round(wall, 3),
        "status": status,
        "cumulative_top": _table("cumulative"),
        "tottime_top": _table("tottime"),
    }


def main(top_n: int = 25) -> dict[str, Any]:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shapes = [
        (2_000, 4, 30),
        (50_000, 4, 20),
    ]
    results = []
    for n, n_labels, iters in shapes:
        print(f"=== profiling CB multilabel cat-heavy n_rows={n:_} K={n_labels} iters={iters} (CPU-only) ===", flush=True)
        r = _profile_one(n, n_labels=n_labels, iterations=iters, seed=0, top_n=top_n)
        print(f"  wall={r['wall_s']}s status={r['status']} cols={r['n_cols']}", flush=True)
        print(r["cumulative_top"], flush=True)
        print("--- tottime ---", flush=True)
        print(r["tottime_top"], flush=True)
        results.append(r)

    out = {"label": "profile-cb-multilabel", "results": results}
    out_path = _RESULTS_DIR / "cb_multilabel_cat.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
