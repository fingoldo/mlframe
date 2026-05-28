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
from mlframe.training.configs import (  # noqa: E402
    FeatureSelectionConfig, OutputConfig, TargetTypes,
)


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
    p.add_argument("--mrmr-interactions-max-order", type=int, default=1,
                   help="MRMR k-way interaction depth (default 1 = 1-way). "
                        "Set to 2 / 3 to enable pair / triplet discovery. "
                        "Only applied when the combo has use_mrmr_fs=True.")
    p.add_argument("--mrmr-fe-max-steps", type=int, default=None,
                   help="MRMR numeric-FE chain depth (default = MRMR default). "
                        "Set to 2 / 3 to enable multi-step FE -- each step "
                        "appends discovered features and refits. Only applied "
                        "when the combo has use_mrmr_fs=True.")
    args = p.parse_args()

    print("Pre-warming numba JIT cache...", flush=True)
    try:
        from mlframe.metrics.core import prewarm_numba_cache
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
                # 2026-05-23 iter185: floor lowered 30 -> 10. Original 30
                # forced every profile to 30 epochs / boost rounds regardless of
                # combo.iterations (3 or 15), so MLP regressor / LTR ranker on
                # 200k rows took ~80s for 2 fits with the bulk going into
                # Lightning's per-epoch validation + checkpoint cycle. 10 still
                # triggers ES paths (default patience 5-10) and exercises the
                # multi-epoch boost loop; 3x faster MLP profile, ~40s saved on
                # MLP-heavy 200k combos (c0033 regression). Aligns with user
                # instruction "у всех моделей сделай поменьше дефолтное число
                # итераций" for fuzz / profile-time runs.
                hyperparams_config={"iterations": max(combo.iterations, 10)},
                # 2026-05-12 Wave 31: save_charts=False gives a 7x speedup on
            # multiclass combos by skipping wasted chart rendering to a
            # temp dir. Profiler measures training cost, not chart-render cost.
            output_config=OutputConfig(data_dir=tmpdir, models_dir="models", save_charts=False),
                use_mlframe_ensembles=combo.use_ensembles,
                # 2026-05-11: thread combo.use_mrmr_fs through. Previously
                # the script reported `mrmr=True` in the header but never
                # passed it to the suite -- so MRMR didn't appear in the
                # profile despite being in the combo definition.
                # ``args.mrmr_interactions_max_order`` (default 1) lets
                # users compare 1-way vs k-way interaction discovery cost.
                feature_selection_config=FeatureSelectionConfig(
                    use_mrmr_fs=combo.use_mrmr_fs,
                    mrmr_kwargs=(
                        {
                            k: v for k, v in {
                                "interactions_max_order": (
                                    args.mrmr_interactions_max_order
                                    if args.mrmr_interactions_max_order > 1 else None
                                ),
                                "fe_max_steps": args.mrmr_fe_max_steps,
                                # 2026-05-12: cap MRMR at 5 min so profiles
                                # don't run away on order=3 + 1M combos.
                                # Production users keep the default (unlimited).
                                "max_runtime_mins": 5,
                                # n_jobs=1 runs FE in the MAIN process so
                                # cProfile sees the actual FE kernel cost.
                                # Default (n_jobs=-1) fans out to joblib
                                # workers whose work is invisible to cProfile.
                                "n_jobs": getattr(args, "mrmr_n_jobs", None) or 1,
                            }.items() if v is not None
                        }
                        if combo.use_mrmr_fs and (
                            args.mrmr_interactions_max_order > 1
                            or args.mrmr_fe_max_steps is not None
                        )
                        else (
                            {"max_runtime_mins": 5, "n_jobs": getattr(args, "mrmr_n_jobs", None) or 1}
                            if combo.use_mrmr_fs else None
                        )
                    ),
                ),
                # Force CPU so CatBoost / XGBoost / LightGBM don't trip on
                # missing CUDA in profiling environments. We're profiling
                # the mlframe-side overhead, not GPU vs CPU trade-offs.
                behavior_config={"prefer_gpu_configs": False},
                verbose=0,
            )
        except Exception as e:
            # Full stack (no limit). Previously capped at limit=3 which hid
            # the actual raise site below the suite -> process_model
            # dispatcher: a c0030_beb1dc9b @200k regression run surfaced
            # "TypeError: iteration over a 0-d array" with the deepest
            # visible frame being _phase_train_one_target_body.py:740, i.e.
            # the call site, not the raiser. Always emit the full chain so
            # the next profile run is debuggable on its own.
            print(f"!! suite error ({type(e).__name__}): {e}", flush=True)
            traceback.print_exc()
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
