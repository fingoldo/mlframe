"""Profile a SPECIFIC fuzz combo (by short_id) under cProfile.

Faster-iteration counterpart to ``profile_fuzz_chains.py`` -- once a
hotspot is identified in the fuzz-chain run, this script lets you
re-run the same combo before/after a fix to verify the speedup.

Usage:
    python profiling/profile_one_combo.py --combo c0042 --rows 300000 --top 30
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import io
import logging
import os
import pathlib
import pstats
import sys
import tempfile
import time
import traceback
from typing import Any, cast

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
    FuzzCombo, build_frame_for_combo, enumerate_combos,
)
from tests.training._fuzz_suite_helpers import (  # noqa: E402
    _config_for_models,
    _configs_for_combo,
    _custom_pre_pipelines_for_combo,
    _feature_selection_config_for_combo,
    _maybe_to_parquet,
    _outlier_detector_for_combo,
    _preprocessing_for_combo,
    _randomize_hyperparams_dict,
    _recurrent_config_for_combo,
    _recurrent_sequences_for_combo,
)
from tests.training.fuzz.test_fuzz_suite import (  # noqa: E402
    _build_fhc,
    _build_linear_cfg,
    _build_ltr_models_and_config,
    _build_precomputed,
    _build_quantile_cfg,
    _resolve_combo_target_type,
)
from mlframe.training.core import train_mlframe_models_suite  # noqa: E402
from tests.training.shared import SimpleFeaturesAndTargetsExtractor  # noqa: E402
from mlframe.training.configs import (  # noqa: E402
    FeatureSelectionConfig, OutlierDetectionConfig, OutputConfig,
)
from mlframe.training.fs_params.configs import MRMRConfig  # noqa: E402
from mlframe.training.extractors import FeaturesAndTargetsExtractor  # noqa: E402


def _build_fte(combo: FuzzCombo, target_col: str) -> SimpleFeaturesAndTargetsExtractor:
    """FTE built exactly like the pytest fuzz suite's, so target type, ts_field, group_field, weights and extra targets all follow the combo."""
    is_ltr = combo.target_type == "learning_to_rank"
    return SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=_resolve_combo_target_type(combo, target_col),
        ts_field=("ts" if combo.with_datetime_col else None),
        group_field=("qid" if is_ltr else None),
        target_carrier=combo.target_carrier,
        weight_schemas=combo.weight_schemas,
        extra_targets=combo.extra_targets,
    )


def _hyperparams_for_combo(combo: FuzzCombo) -> dict:
    """Every booster / RFECV axis of the combo, as the pytest fuzz suite passes them, with the profiling iteration floor applied."""
    return _randomize_hyperparams_dict(_config_for_models(
        combo.models,
        combo.n_rows,
        # Floor of 10: short enough to keep MLP/LTR profiles fast, long enough to exercise early stopping and the multi-round boost loop.
        iterations=max(combo.iterations, 10),
        early_stopping_rounds=combo.early_stopping_rounds_cfg,
        mlp_predict_batch_size=combo.mlp_predict_batch_size_cfg,
        lgb_feature_fraction=combo.lgb_feature_fraction_cfg,
        lgb_num_leaves=combo.lgb_num_leaves_cfg,
        xgb_max_depth=combo.xgb_max_depth_cfg,
        xgb_colsample_bynode=combo.xgb_colsample_bynode_cfg,
        cb_border_count=combo.cb_border_count_cfg,
        hgb_max_leaf_nodes=combo.hgb_max_leaf_nodes_cfg,
        rfecv_cv_n_splits=combo.rfecv_cv_n_splits_cfg,
        rfecv_votes_aggregation=combo.rfecv_votes_aggregation_cfg,
        rfecv_search_method=combo.rfecv_search_method_cfg,
        lgb_boosting_type=combo.lgb_boosting_type_cfg,
        lgb_dart_drop_rate=combo.lgb_dart_drop_rate_cfg,
        lgb_goss_top_rate=combo.lgb_goss_top_rate_cfg,
        xgb_tree_method=combo.xgb_tree_method_cfg,
        xgb_hist_max_bin=combo.xgb_hist_max_bin_cfg,
        cb_bootstrap_type=combo.cb_bootstrap_type_cfg,
        cb_bayesian_bagging_temperature=combo.cb_bayesian_bagging_temperature_cfg,
        cb_bernoulli_subsample=combo.cb_bernoulli_subsample_cfg,
        cb_grow_policy=combo.cb_grow_policy_cfg,
        cb_lossguide_max_leaves=combo.cb_lossguide_max_leaves_cfg,
    ), combo)


def _feature_selection_for_profile(combo: FuzzCombo, args: argparse.Namespace) -> FeatureSelectionConfig:
    """The combo's own FeatureSelectionConfig, with profiling-only MRMR overrides layered on top when the combo enables MRMR."""
    rfecv_on = combo._canonical_rfecv_estimator() is not None
    fs: FeatureSelectionConfig = _feature_selection_config_for_combo(combo, rfecv_on, _custom_pre_pipelines_for_combo(combo))
    if fs.mrmr is None:
        return fs
    overrides = {
        "interactions_max_order": args.mrmr_interactions_max_order if args.mrmr_interactions_max_order > 1 else None,
        "fe_max_steps": args.mrmr_fe_max_steps,
        # Caps runaway order-3 / 1M-row profiles; production keeps the unlimited default.
        "max_runtime_mins": 5,
        # Main-process FE so cProfile sees the kernel cost; joblib workers are invisible to it.
        "n_jobs": 1,
    }
    # Field values via getattr, not model_dump: dumping would flatten nested sub-config objects (cat_fe_config, ...) into plain dicts.
    explicit = {k: getattr(fs.mrmr, k) for k in fs.mrmr.model_fields_set}
    mrmr_kwargs = {**explicit, **{k: v for k, v in overrides.items() if v is not None}}
    return fs.model_copy(update={"mrmr": MRMRConfig(**mrmr_kwargs)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combo", required=True, help="combo short_id, e.g. c0042")
    p.add_argument("--rows", type=int, default=300_000)
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
    p.add_argument("--save-charts", action="store_true",
                   help="Enable diagnostic-chart rendering (OutputConfig.save_charts=True). "
                        "Default False (7x speedup, measures training not chart-render cost); "
                        "pass this to profile reporting/charts hotspots specifically.")
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

    df, target_col, _ = build_frame_for_combo(combo)
    fte = _build_fte(combo, target_col)
    is_ltr = combo.target_type == "learning_to_rank"

    profiler = cProfile.Profile()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            df_input = _maybe_to_parquet(combo, df, pathlib.Path(tmpdir))
            combo_configs = _configs_for_combo(combo)
            # Force CPU so boosters don't trip on missing CUDA in profiling environments; this measures mlframe overhead, not GPU vs CPU.
            combo_configs["behavior_config"] = combo_configs["behavior_config"].model_copy(update={"prefer_gpu_configs": False})
            ltr_models, ltr_ranking_config = _build_ltr_models_and_config(combo, is_ltr)
            suite_kwargs: dict[str, Any] = {}
            # Mirrors the pytest suite: the implicit default allowlist is what triggers gated_outlier point-mass auto-detection.
            if is_ltr or combo._canonical_mlframe_models_explicit():
                suite_kwargs["mlframe_models"] = ltr_models
            if is_ltr:
                suite_kwargs["target_type"] = fte._resolve_target_type()
                suite_kwargs["ranking_config"] = ltr_ranking_config.model_copy(update={"assume_comparable_scales": combo.ltr_assume_comparable_scales_cfg})
            optional_configs = {
                "quantile_regression_config": _build_quantile_cfg(combo, is_ltr),
                "linear_model_config": _build_linear_cfg(combo),
                "feature_handling_config": _build_fhc(combo),
                "precomputed": _build_precomputed(combo, df_input, df),
            }
            suite_kwargs.update({k: v for k, v in optional_configs.items() if v is not None})
            recurrent_model = combo._canonical_recurrent_model()

            profiler.enable()
            train_mlframe_models_suite(
                df=df_input,
                target_name=combo.short_id(),
                model_name=f"profile_{combo.short_id()}",
                # Duck-typed test extractor, same object the pytest fuzz suite passes.
                features_and_targets_extractor=cast(FeaturesAndTargetsExtractor, fte),
                **suite_kwargs,
                hyperparams_config=_hyperparams_for_combo(combo),
                preprocessing_config=_preprocessing_for_combo(combo),
                use_ordinary_models=True,
                use_mlframe_ensembles=combo.use_ensembles,
                outlier_detection_config=OutlierDetectionConfig(
                    detector=_outlier_detector_for_combo(combo),
                    apply_to_val=combo.apply_outlier_to_val_cfg,
                ),
                feature_selection_config=_feature_selection_for_profile(combo, args),
                # Charts off by default (7x faster on multiclass): the profile targets training cost, not chart rendering.
                output_config=OutputConfig(data_dir=tmpdir, models_dir="models", save_charts=args.save_charts),
                recurrent_models=([recurrent_model] if recurrent_model is not None else None),
                sequences=_recurrent_sequences_for_combo(combo, df=df_input),
                recurrent_config=_recurrent_config_for_combo(combo),
                enable_target_distribution_analyzer=combo.enable_target_distribution_analyzer_cfg,
                verbose=0,
                **combo_configs,
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
