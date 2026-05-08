"""Profile randomly-picked fuzz combos under cProfile on mid-size frames.

Workflow:
1. ``enumerate_combos`` to get the deterministic 150-combo space.
2. Sample K combos at random (different seed per invocation).
3. For each: bump ``n_rows`` to a profiling target (~100k-300k rows
   so the suite touches enough data for hotspots to surface), build
   the frame via ``build_frame_for_combo``, and run
   ``train_mlframe_models_suite`` under ``cProfile``.
4. Print the top-N hotspots (cumulative time) per combo + an
   aggregated cross-combo top-N (sum of cumtime across combos that
   touched the function).

Caller can loop this script externally; the script itself runs ONE
batch of K combos and exits.

Usage:
    python profiling/profile_fuzz_chains.py
        --combos 5
        --rows-target 200000
        --seed 42
        --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import io
import logging
import os
import pstats
import random
import sys
import tempfile
import time
import traceback
from typing import Optional

# Force unbuffered stdout so progress is visible when run in
# background / piped through tee. Mirrors the project's
# ``feedback_pytest_unbuffered`` rule -- never run a long Python job
# blind to its progress.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Quiet down noisy loggers so the cProfile output dominates the screen.
logging.basicConfig(level=logging.WARNING)
for noisy in ("sklearn", "lightgbm", "xgboost", "catboost", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.training._fuzz_combo import (  # noqa: E402
    FuzzCombo, build_frame_for_combo, enumerate_combos,
)
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402
from tests.training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402
from mlframe.training.configs import TargetTypes, OutputConfig  # noqa: E402


def _resize_combo(combo: FuzzCombo, n_rows: int) -> FuzzCombo:
    """Return a copy of ``combo`` with bumped ``n_rows`` for profiling.

    Dataclass is frozen; use ``dataclasses.replace``.
    """
    return dataclasses.replace(combo, n_rows=n_rows)


def _build_fte_from_combo(combo: FuzzCombo) -> SimpleFeaturesAndTargetsExtractor:
    """Build a SimpleFeaturesAndTargetsExtractor with the right target_type
    + group_field / weight_schemas matching the combo."""
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


def _profile_one_combo(
    combo: FuzzCombo, top: int = 20, save_dir: Optional[str] = None,
) -> Optional[pstats.Stats]:
    """Run one combo under cProfile + print top hotspots. Returns the
    pstats object so an outer aggregator can sum across combos."""
    print(
        f"\n{'='*80}\n"
        f"Combo {combo.short_id()}\n"
        f"  models={combo.models} target={combo.target_type} "
        f"rows={combo.n_rows:,} cats={combo.cat_feature_count} "
        f"input={combo.input_type}\n"
        f"  outlier={combo.outlier_detection} ensembles={combo.use_ensembles} "
        f"weight_schemas={combo.weight_schemas} mrmr={combo.use_mrmr_fs}\n"
        f"{'='*80}"
    )

    try:
        df, target_col, _ = build_frame_for_combo(combo)
    except Exception as e:
        print(f"  ! frame-build error ({type(e).__name__}): {e}")
        return None

    fte = _build_fte_from_combo(combo)

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
                target_type=_build_fte_from_combo(combo)._resolve_target_type(),
                mlframe_models=list(combo.models),
                hyperparams_config={"iterations": max(combo.iterations, 30)},
                output_config=OutputConfig(data_dir=tmpdir, models_dir="models"),
                use_mlframe_ensembles=combo.use_ensembles,
                behavior_config={"prefer_gpu_configs": False},
                verbose=0,
            )
        except Exception as e:
            print(f"  ! suite error ({type(e).__name__}): {e}")
            traceback.print_exc(limit=3)
        finally:
            profiler.disable()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.2f}s")
    if save_dir:
        prof_path = os.path.join(save_dir, f"{combo.short_id()}.prof")
        profiler.dump_stats(prof_path)
        print(f"Stats saved to {prof_path}")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    print(f"\nTop {top} hotspots (cumulative):")
    print("-" * 80)
    stream = io.StringIO()
    stats_for_print = pstats.Stats(profiler, stream=stream)
    stats_for_print.sort_stats("cumulative")
    stats_for_print.print_stats(top)
    print(stream.getvalue())
    return stats


def _aggregate_hotspots(
    all_stats: list[pstats.Stats], top: int = 30,
):
    """Aggregate cumtime across all collected stats; print top-N."""
    aggregate: dict = {}
    for s in all_stats:
        for func, (cc, nc, tt, ct, callers) in s.stats.items():
            if func in aggregate:
                ag = aggregate[func]
                aggregate[func] = (
                    ag[0] + cc, ag[1] + nc, ag[2] + tt, ag[3] + ct, None,
                )
            else:
                aggregate[func] = (cc, nc, tt, ct, None)
    rows = sorted(
        aggregate.items(), key=lambda kv: kv[1][3], reverse=True,
    )[:top]
    print(f"\n{'='*80}\nAGGREGATED TOP {top} (cumtime summed across combos)\n{'='*80}")
    print(f"{'cumtime':>12} {'tottime':>12} {'ncalls':>10}  function")
    for func, (cc, nc, tt, ct, _) in rows:
        fname = f"{func[0]}:{func[1]}({func[2]})"
        print(f"{ct:>12.3f} {tt:>12.3f} {nc:>10}  {fname}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combos", type=int, default=3,
                   help="Number of fuzz combos to profile in this batch.")
    p.add_argument("--rows-target", type=int, default=150_000,
                   help="Override n_rows on each combo (~150k = ~50-100MB).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for which combos are picked from the 150-combo space.")
    p.add_argument("--top", type=int, default=20,
                   help="How many hotspots to print per combo.")
    p.add_argument("--combo-pool", type=int, default=150,
                   help="Size of the combo space to sample from.")
    p.add_argument("--prefer-models", type=str, default="lgb,xgb,cb",
                   help="Comma-separated subset of models to prefer when sampling. "
                        "Combos with at least one model from this list pass the filter.")
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, write one .prof file per combo here for later "
                        "aggregation via aggregate_prof.py.")
    args = p.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print("Pre-warming numba JIT cache...")
    try:
        from mlframe.metrics import prewarm_numba_cache
        prewarm_numba_cache()
    except Exception:
        pass
    print("done.\n")

    combos = enumerate_combos(target=args.combo_pool)
    rng = random.Random(args.seed)
    if args.prefer_models:
        prefer = set(m.strip() for m in args.prefer_models.split(","))
        combos = [c for c in combos if set(c.models) & prefer]
    rng.shuffle(combos)
    sample = combos[: args.combos]

    print(f"Profiling {len(sample)} combos (rows-target={args.rows_target:,})...")
    all_stats = []
    for combo in sample:
        resized = _resize_combo(combo, args.rows_target)
        s = _profile_one_combo(resized, top=args.top, save_dir=args.save_dir)
        if s is not None:
            all_stats.append(s)

    if all_stats:
        _aggregate_hotspots(all_stats, top=args.top * 2)


if __name__ == "__main__":
    main()
