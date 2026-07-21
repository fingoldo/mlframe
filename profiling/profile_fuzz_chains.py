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
from typing import Any, Optional

# Force unbuffered stdout so progress is visible when run in
# background / piped through tee. Mirrors the project's
# ``feedback_pytest_unbuffered`` rule -- never run a long Python job
# blind to its progress.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# A fuzz combo's Unicode category value (e.g. accented / Cyrillic text under input=polars_utf8) can end up inside a
# caught exception's message; printing it on a cp1251 Windows console then raises UnicodeEncodeError, killing the
# whole profiling run on an otherwise-handled error. errors="replace" makes any unprintable character a "?" instead
# of crashing the harness.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

# Quiet down noisy loggers so the cProfile output dominates the screen.
logging.basicConfig(level=logging.WARNING)
for noisy in ("sklearn", "lightgbm", "xgboost", "catboost", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# 2026-05-08: force matplotlib Agg in the profiler so we measure
# mlframe-side cost, not Qt-backend probe overhead.
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=False)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.training._fuzz_combo import (  # noqa: E402
    FuzzCombo, build_frame_for_combo, enumerate_combos,
)
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402
from tests.training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402
from mlframe.training.configs import TargetTypes, OutputConfig, FeatureSelectionConfig  # noqa: E402


def _fs_config_from_combo(combo):
    """Build a FeatureSelectionConfig that threads the combo's MRMR + FE-engineering axes, so the profile exercises the
    actual feature-selection / feature-engineering kernels (MI candidate scoring, polynomial / Hermite eval, k-way
    interactions) -- NOT just the always-on reporting/bootstrap layer. n_jobs=1 keeps the FE kernels in the MAIN process
    so cProfile attributes them (n_jobs=-1 fans out to joblib workers invisible to the profiler); max_runtime_mins caps
    a runaway FE search. Returns None when the combo does not use MRMR (no FS overhead added)."""
    if not getattr(combo, "use_mrmr_fs", False):
        return None
    _g = lambda name, d=None: getattr(combo, name, d)
    mrmr_kwargs: dict[str, Any] = {
        "verbose": 0, "n_jobs": 1, "max_runtime_mins": 5,
        "quantization_nbins": 5, "use_simple_mode": True,
        # k-way interaction discovery (1 = off; 2/3 = pair/triplet) -- the interaction-MI kernels.
        "interactions_max_order": max(1, int(_g("mrmr_interactions_max_order_cfg", 1) or 1)),
    }
    # Feature-engineering search knobs -> the FE candidate/polynomial/Hermite kernels. Only meaningful when fe_ntop>0.
    _fe_ntop = int(_g("mrmr_fe_ntop_features_cfg", 0) or 0)
    if _fe_ntop > 0:
        mrmr_kwargs.update({
            "fe_ntop_features": _fe_ntop,
            "fe_npermutations": int(_g("mrmr_fe_npermutations_cfg", 0) or 0),
            "fe_unary_preset": _g("mrmr_fe_unary_preset_cfg", "minimal") or "minimal",
            "fe_binary_preset": _g("mrmr_fe_binary_preset_cfg", "minimal") or "minimal",
            "fe_smart_polynom_iters": int(_g("mrmr_fe_smart_polynom_iters_cfg", 0) or 0),
            "fe_min_polynom_degree": int(_g("mrmr_fe_min_polynom_degree_cfg", 3) or 3),
            "fe_max_polynom_degree": int(_g("mrmr_fe_max_polynom_degree_cfg", 3) or 3),
        })
        _fe_steps = _g("mrmr_fe_max_steps_cfg", None)
        if _fe_steps:
            mrmr_kwargs["fe_max_steps"] = int(_fe_steps)
    return FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs=mrmr_kwargs)


def _resize_combo(combo: FuzzCombo, n_rows: int) -> FuzzCombo:
    """Return a copy of ``combo`` with bumped ``n_rows`` for profiling.

    Dataclass is frozen; use ``dataclasses.replace``.
    """
    return dataclasses.replace(combo, n_rows=n_rows)


def _build_fte_from_combo(
    combo: FuzzCombo, target_col: str,
) -> SimpleFeaturesAndTargetsExtractor:
    """Build a SimpleFeaturesAndTargetsExtractor with the right target_type
    + group_field / weight_schemas matching the combo.

    ``target_col`` is the authoritative column name returned by
    ``build_frame_for_combo``. It also disambiguates multi_target_regression:
    that path emits a 2-D ``target`` column only when every model natively
    handles a 2-D continuous target, else it downgrades the frame to a 1-D
    ``target_reg`` column (see frame_builder ``_NATIVE_MTR_MODELS``). A
    downgraded MTR combo is REGRESSION at the data level, so we mirror the
    pytest suite's disambiguation instead of a fixed target-type map that
    KeyErrors on MTR.
    """
    _effective_target_type = combo.target_type
    if combo.target_type == "multi_target_regression" and target_col != "target":
        _effective_target_type = "regression"
    target_type_map = {
        "regression": TargetTypes.REGRESSION,
        "binary_classification": TargetTypes.BINARY_CLASSIFICATION,
        "multiclass_classification": TargetTypes.MULTICLASS_CLASSIFICATION,
        "multilabel_classification": TargetTypes.MULTILABEL_CLASSIFICATION,
        "learning_to_rank": TargetTypes.LEARNING_TO_RANK,
        "multi_target_regression": TargetTypes.MULTI_TARGET_REGRESSION,
        "quantile_regression": TargetTypes.QUANTILE_REGRESSION,
    }
    tt = target_type_map[_effective_target_type]
    return SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        target_type=tt,
        group_field="qid" if combo.target_type == "learning_to_rank" else None,
        weight_schemas=combo.weight_schemas,
    )


def _profile_one_combo(
    combo: FuzzCombo, top: int = 20, save_dir: Optional[str] = None,
    save_charts: bool = True,
) -> Optional[pstats.Stats]:
    """Run one combo under cProfile + print top hotspots. Returns the
    pstats object so an outer aggregator can sum across combos.

    ``save_charts=False`` skips the matplotlib/plotly diagnostic-panel
    rendering (which otherwise dominates ~90% of the wall at 300k), so the
    training / feature-selection / metric code paths surface as the hotspots.
    """
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

    fte = _build_fte_from_combo(combo, target_col)

    profiler = cProfile.Profile()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            profiler.enable()
            train_mlframe_models_suite(
                df=df,
                target_name=combo.short_id(),
                model_name=f"profile_{combo.short_id()}",
                features_and_targets_extractor=fte,  # type: ignore[arg-type]  # tests.training.shared.SimpleFeaturesAndTargetsExtractor is a distinct duck-typed test double, not a FeaturesAndTargetsExtractor subclass
                target_type=fte._resolve_target_type(),
                mlframe_models=list(combo.models),
                hyperparams_config={"iterations": max(combo.iterations, 30)},
                output_config=OutputConfig(data_dir=tmpdir, models_dir="models", save_charts=save_charts),
                feature_selection_config=_fs_config_from_combo(combo),
                use_mlframe_ensembles=combo.use_ensembles,
                # Thread the combo's ensembling axes so the profiler exercises the same Caruana-weights / rank_average
                # blend paths the pytest fuzz suite does (mirrors _fuzz_suite_helpers._configs_for_combo); canon-collapsed
                # to OFF when ensembles are disabled, so this is a no-op for non-ensemble combos.
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
        for func, (cc, nc, tt, ct, callers) in s.stats.items():  # type: ignore[attr-defined]  # pstats.Stats.stats is a real runtime attribute missing from typeshed's stub
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
    p.add_argument("--rows-target", type=int, default=300_000,
                   help="Override n_rows on each combo (~300k = production profiling shape).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for which combos are picked from the enumerated combo space.")
    p.add_argument("--master-seed", type=int, default=20260422,
                   help="Seed for enumerate_combos itself: DIFFERENT master seeds enumerate DISJOINT combo spaces "
                        "(0 overlap), so vary this to reach genuinely fresh combos rather than re-profiling the same "
                        "pool with a different --seed shuffle. Mirrors profile_one_combo.py --master-seed.")
    p.add_argument("--top", type=int, default=20,
                   help="How many hotspots to print per combo.")
    p.add_argument("--combo-pool", type=int, default=150,
                   help="Size of the combo space to enumerate + sample from.")
    p.add_argument("--prefer-models", type=str, default="",
                   help="Comma-separated whitelist of models. Combos pass only "
                        "when ALL of their models are in this list. Empty (default): "
                        "no filter -- combos keep the enumerator's own random "
                        "model-subset selection across the full MODELS universe "
                        "(cb, xgb, lgb, hgb, linear, mlp). NOTE: mlp is a known "
                        "MLP+CUDA Windows access-violation risk (see the fuzz "
                        "combo runner's crash-recovery loop); pass e.g. "
                        "'cb,xgb,lgb,hgb,linear' to exclude it if that becomes noisy.")
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, write one .prof file per combo here for later "
                        "aggregation via aggregate_prof.py.")
    p.add_argument("--no-charts", action="store_true",
                   help="Disable diagnostic-chart rendering (save_charts=False). The "
                        "matplotlib/plotly panel export dominates ~90%% of the wall at "
                        "300k and buries the training / feature-selection / metric code "
                        "paths; turn it off to profile those instead.")
    args = p.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

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
        # SUBSET filter: all combo models must be in the whitelist. The
        # previous semantics (intersection only) let MLP-bearing combos
        # through whenever they also had cb/lgb/xgb -- and the
        # MLP+CUDA path on Windows can fault with an access violation
        # mid-fit, killing the whole profile process before any combo
        # completes. The whitelist behaviour is closer to what the flag
        # name suggests anyway.
        combos = [c for c in combos if set(c.models).issubset(prefer)]
    rng.shuffle(combos)
    sample = combos[: args.combos]

    print(f"Profiling {len(sample)} combos (rows-target={args.rows_target:,})...")
    all_stats = []
    for combo in sample:
        resized = _resize_combo(combo, args.rows_target)
        s = _profile_one_combo(resized, top=args.top, save_dir=args.save_dir, save_charts=not args.no_charts)
        if s is not None:
            all_stats.append(s)

    if all_stats:
        _aggregate_hotspots(all_stats, top=args.top * 2)


if __name__ == "__main__":
    main()
