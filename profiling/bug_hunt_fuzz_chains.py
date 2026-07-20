"""Run many randomly-picked fuzz combos on SMALL frames (fast iteration) with no cProfile overhead, to surface
correctness bugs (crashes / uncaught exceptions) rather than hotspots.

Mirrors ``profile_fuzz_chains.py``'s combo machinery (``enumerate_combos`` / ``build_frame_for_combo`` /
``train_mlframe_models_suite``) but skips ``cProfile`` entirely and defaults to a 5k-row frame, so a batch of
many combos runs in a fraction of the time a single 200k-300k profiling run takes -- the point here is coverage
of the combo space, not per-combo timing.

Usage:
    python profiling/bug_hunt_fuzz_chains.py --combos 20 --rows-target 5000 --seed 1
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import sys
import tempfile
import traceback

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.WARNING)
for noisy in ("sklearn", "lightgbm", "xgboost", "catboost", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.training._fuzz_combo import (  # noqa: E402
    FuzzCombo, build_frame_for_combo, enumerate_combos,
)
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402
from tests.training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402
from mlframe.training.configs import TargetTypes, OutputConfig, FeatureSelectionConfig  # noqa: E402

# Re-use the profiling harness's combo-config helpers verbatim (identical FS-config / target-type / frame-resize
# logic) so a bug found here reproduces byte-for-byte under profile_fuzz_chains.py at the profiling row count too.
from profiling.profile_fuzz_chains import (  # noqa: E402
    _build_fte_from_combo, _fs_config_from_combo, _resize_combo,
)


def _run_one_combo(combo: FuzzCombo, save_charts: bool = True) -> bool:
    """Run one combo with no cProfile. Returns True on a clean run, False on a frame-build or suite error
    (already printed with a traceback for the user to act on)."""
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
        traceback.print_exc(limit=6)
        return False

    fte = _build_fte_from_combo(combo, target_col)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            train_mlframe_models_suite(
                df=df,
                target_name=combo.short_id(),
                model_name=f"bughunt_{combo.short_id()}",
                features_and_targets_extractor=fte,
                target_type=fte._resolve_target_type(),
                mlframe_models=list(combo.models),
                hyperparams_config={"iterations": max(combo.iterations, 30)},
                output_config=OutputConfig(data_dir=tmpdir, models_dir="models", save_charts=save_charts),
                feature_selection_config=_fs_config_from_combo(combo),
                use_mlframe_ensembles=combo.use_ensembles,
                behavior_config={
                    "prefer_gpu_configs": False,
                    "use_caruana_weights_in_ensemble": bool(
                        getattr(combo, "use_caruana_weights_in_ensemble_cfg", False) and combo.use_ensembles
                    ),
                    "extra_ensembling_methods": (
                        ("rank_average",)
                        if (getattr(combo, "ens_rank_average_cfg", False) and combo.use_ensembles)
                        else ()
                    ),
                },
                verbose=0,
            )
        except Exception as e:
            print(f"  ! suite error ({type(e).__name__}): {e}")
            traceback.print_exc(limit=6)
            return False
    print("  OK")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combos", type=int, default=20, help="Number of fuzz combos to run in this batch.")
    p.add_argument("--rows-target", type=int, default=5_000, help="Override n_rows on each combo (small = fast iteration).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for which combos are picked from the enumerated combo space.")
    p.add_argument("--master-seed", type=int, default=20260422, help="Seed for enumerate_combos itself.")
    p.add_argument("--combo-pool", type=int, default=150, help="Size of the combo space to enumerate + sample from.")
    p.add_argument("--prefer-models", type=str, default="lgb,xgb,cb", help="Comma-separated whitelist of models.")
    p.add_argument("--no-charts", action="store_true", help="Disable diagnostic-chart rendering (save_charts=False).")
    args = p.parse_args()

    print("Pre-warming numba JIT cache...")
    try:
        from mlframe.metrics.core import prewarm_numba_cache
        prewarm_numba_cache()
    except Exception:
        pass
    print("done.\n")

    combos = enumerate_combos(target=args.combo_pool, master_seed=args.master_seed)
    rng = random.Random(args.seed)
    if args.prefer_models:
        prefer = set(m.strip() for m in args.prefer_models.split(",") if m.strip())
        combos = [c for c in combos if set(c.models).issubset(prefer)]
    rng.shuffle(combos)
    sample = combos[: args.combos]

    print(f"Bug-hunting {len(sample)} combos (rows-target={args.rows_target:,}, no cProfile)...")
    n_ok = 0
    n_fail = 0
    for combo in sample:
        resized = _resize_combo(combo, args.rows_target)
        ok = _run_one_combo(resized, save_charts=not args.no_charts)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*80}\nBug hunt done: {n_ok} clean / {n_fail} errored (out of {len(sample)})\n{'='*80}")


if __name__ == "__main__":
    main()
