"""Profile a SPECIFIC fuzz combo (by short_id) under cProfile.

Faster-iteration counterpart to ``profile_fuzz_chains.py`` -- once a
hotspot is identified in the fuzz-chain run, this script lets you
re-run the same combo before/after a fix to verify the speedup.

Usage:
    python profiling/profile_one_combo.py --combo c0042 --rows 100000 --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import io
import logging
import os
import pstats
import sys
import tempfile
import time
import traceback

logging.basicConfig(level=logging.WARNING)
for noisy in ("sklearn", "lightgbm", "xgboost", "catboost", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# 2026-05-08: force matplotlib to Agg in the profiler. Without this,
# pyplot's first ``subplots()`` call probes the Qt backend on Windows
# and fires ~820 ``activateWindow`` calls (~1.45s wasted on c0088).
# Profile is for measuring training cost, not GUI overhead.
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.training._fuzz_combo import (  # noqa: E402
    build_frame_for_combo, enumerate_combos,
)
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402
from tests.training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402
from mlframe.training.configs import OutputConfig, TargetTypes  # noqa: E402


def _build_fte(combo):
    target_type_map = {
        "regression": TargetTypes.REGRESSION,
        "binary_classification": TargetTypes.BINARY_CLASSIFICATION,
        "multiclass_classification": TargetTypes.MULTICLASS_CLASSIFICATION,
        "multilabel_classification": TargetTypes.MULTILABEL_CLASSIFICATION,
        "learning_to_rank": TargetTypes.LEARNING_TO_RANK,
    }
    tt = target_type_map[combo.target_type]
    target_col = "target_reg" if combo.target_type == "regression" else (
        "relevance" if combo.target_type == "learning_to_rank" else "target"
    )
    return SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        target_type=tt,
        group_field="qid" if combo.target_type == "learning_to_rank" else None,
        weight_schemas=combo.weight_schemas,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combo", required=True, help="combo short_id, e.g. c0042")
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--combo-pool", type=int, default=150)
    p.add_argument("--master-seed", type=int, default=2026_04_22,
                   help="Seed for enumerate_combos. Different seed = different "
                        "150-combo space, useful for sampling fresh hotspots.")
    p.add_argument("--save-stats", type=str, default=None,
                   help="Optional path to write the .prof file (snakeviz-compatible).")
    args = p.parse_args()

    print("Pre-warming numba JIT cache...", flush=True)
    try:
        from mlframe.metrics import prewarm_numba_cache
        prewarm_numba_cache()
    except Exception:
        pass

    combos = enumerate_combos(target=args.combo_pool, master_seed=args.master_seed)
    matches = [c for c in combos if c.short_id() == args.combo]
    if not matches:
        print(f"!! combo {args.combo!r} not found", flush=True)
        sys.exit(1)
    combo = dataclasses.replace(matches[0], n_rows=args.rows)
    print(
        f"Running {combo.short_id()}: models={combo.models} target={combo.target_type} "
        f"rows={combo.n_rows:,} cats={combo.cat_feature_count} input={combo.input_type}",
        flush=True,
    )

    df, _, _ = build_frame_for_combo(combo)
    fte = _build_fte(combo)

    profiler = cProfile.Profile()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            profiler.enable()
            train_mlframe_models_suite(
                df=df,
                target_name=combo.short_id(),
                model_name=f"profile_{combo.short_id()}",
                features_and_targets_extractor=fte,
                target_type=fte._resolve_target_type(),
                mlframe_models=list(combo.models),
                hyperparams_config={"iterations": max(combo.iterations, 30)},
                output_config=OutputConfig(data_dir=tmpdir, models_dir="models"),
                use_mlframe_ensembles=combo.use_ensembles,
                # Force CPU so CatBoost / XGBoost / LightGBM don't trip on
                # missing CUDA in profiling environments. We're profiling
                # the mlframe-side overhead, not GPU vs CPU trade-offs.
                behavior_config={"prefer_gpu_configs": False},
                verbose=0,
            )
        except Exception as e:
            print(f"!! suite error ({type(e).__name__}): {e}", flush=True)
            traceback.print_exc(limit=3)
        finally:
            profiler.disable()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

    if args.save_stats:
        profiler.dump_stats(args.save_stats)
        print(f"Stats written to {args.save_stats}", flush=True)

    stream = io.StringIO()
    s = pstats.Stats(profiler, stream=stream)
    s.sort_stats("cumulative")
    s.print_stats(args.top)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
