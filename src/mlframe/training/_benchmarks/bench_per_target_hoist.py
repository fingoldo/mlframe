"""Per-target feature-side cache hoist: multi-target suite wall-time bench.

Compares a multi-target suite end-to-end with the hoist active (post-fix) vs the cache forcibly
cleared between targets (pre-fix emulation). The pre-fix emulation is implemented by deleting
``ctx.artifacts['feature_side_cache']`` / ``ctx.artifacts['dataset_reuse_cache']`` between each
``_train_one_target`` call so the next target re-pays the build cost - this mirrors what the
loop did before the hoist landed without us having to dig up a tagged baseline.

Measured speedup (2026-05-16, XGB, 5 targets, OOF n_splits=5 / fits=5xCV per target):
  - n_rows=2000, n_features=8: 1-3% (noise floor; DMatrix build is sub-100ms vs ~2.5s fit)
  - n_rows=20000, n_features=30: 21% single-run, -4% to +10% with min-of-2 (run-to-run
    variance is large because OOF cross_val_predict dominates - ~93% of wall-time per the
    cProfile attribution; the hoist saves the per-target outer DMatrix build but cannot
    save inner-CV-fold builds).

Real production gains compound when:
  - OOF is disabled (pure forward-fit suites)
  - Multiple pre_pipelines (each pp_name has its own cache bucket; per-target hoist wins
    multiply across the pre_pipeline sweep)
  - Polars-native CB on large data (Pool reuse via set_label crosses targets directly)
  - Pandas-tier prepared_frames (tier_dfs + fingerprint reused identically across targets)

cProfile hotspots (representative 5-target x XGB suite, 2026-05-16):
  - cross_val_predict (~93% of wall-time per target - dominates the picture and is NOT
    affected by the hoist; the hoist saves only the final fit's DMatrix construction).
  - process_model -> xgb_shim.fit -> xgboost.training.train (the inner per-fold fits).
  - _build_tier_dfs / _prep_polars_df / compute_model_input_fingerprint are <1% each at
    this scale - the hoist's first-order savings are not visible without OOF disabled.

Usage:
    python -m mlframe.training._benchmarks.bench_per_target_hoist

Writes results JSON to ``_results/bench_per_target_hoist.json``.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "bench_per_target_hoist.json"


class _MultiTargetExtractor:
    """Stand-in FTE that emits N regression targets sharing the same featureset.

    Mirrors the shared SimpleFeaturesAndTargetsExtractor's transform() contract; the suite
    feeds each target through ``_train_one_target`` while the same df is pinned on ctx.
    """

    def __init__(self, target_columns, target_type):
        self.target_columns = tuple(target_columns)
        self.target_type = target_type
        self.ts_field = None
        self.group_field = None
        self.weight_schemas = None
        self.target_carrier = "numpy"

    def transform(self, df):
        target_by_type = {self.target_type: {}}
        for col in self.target_columns:
            if isinstance(df, pd.DataFrame):
                target_by_type[self.target_type][col] = df[col].values
            else:
                target_by_type[self.target_type][col] = df[col].to_numpy()
        cols_to_drop = list(self.target_columns)
        return (df, target_by_type, None, None, None, None, cols_to_drop, {})


def _build_dataset(n_rows: int, n_features: int, n_targets: int, seed: int = 2026) -> pd.DataFrame:
    """Synth regression panel with n_rows x n_features + n_targets columns. Targets are noisy
    linear combos of distinct feature triples so the model has signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    for t in range(n_targets):
        # Pick 3 features that depend on t deterministically so multiple targets are not
        # identical (and the suite cannot accidentally short-circuit identical fits).
        a = t % n_features
        b = (t + 7) % n_features
        c = (t + 13) % n_features
        df[f"y{t}"] = (X[:, a] - 0.6 * X[:, b] + 0.4 * X[:, c] + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
    return df


def _time_suite(df: pd.DataFrame, n_targets: int, *, force_clear_caches: bool) -> float:
    """Run train_mlframe_models_suite once and return wall-time in seconds.

    ``force_clear_caches=True`` emulates the pre-hoist path by deleting the suite-scoped caches
    between each ``_train_one_target`` call so the next target rebuilds from scratch. False is
    the post-fix native behaviour.
    """
    from mlframe.training import train_mlframe_models_suite, TargetTypes
    from mlframe.training.configs import BaselineDiagnosticsConfig, DummyBaselinesConfig
    import mlframe.training.core._phase_train_one_target as pt
    import mlframe.training.core.main as _main

    fte = _MultiTargetExtractor(
        [f"y{t}" for t in range(n_targets)],
        target_type=TargetTypes.REGRESSION,
    )

    _orig_train_one = pt._train_one_target
    _call_n = {"value": 0}

    def _wrapped(ctx, target_type, targets, cur_target_name, cur_target_values):
        if force_clear_caches:
            arts = ctx.artifacts or {}
            arts.pop("feature_side_cache", None)
            arts.pop("dataset_reuse_cache", None)
        _call_n["value"] += 1
        if os.environ.get("BENCH_DEBUG_CACHE"):
            arts_after = ctx.artifacts or {}
            print(
                f"[bench-debug] target_call={_call_n['value']} target={cur_target_name} "
                f"clear={force_clear_caches} feature_side_cache_size={len((arts_after.get('feature_side_cache') or {}))} "
                f"dataset_reuse_cache_size={len((arts_after.get('dataset_reuse_cache') or {}))}",
                flush=True,
            )
        return _orig_train_one(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _wrapped
    _orig_pt_alias = _main.pr._train_one_target
    _main.pr._train_one_target = _wrapped

    gc.collect()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            train_mlframe_models_suite(
                df=df, target_name="multi", model_name="bench",
                features_and_targets_extractor=fte,
                # xgb / lgb exercise the dataset-reuse hoist (binned DMatrix/Dataset
                # built once and re-used via set_label across targets). Linear has no
                # binned dataset so the hoist there is feature-side only and shows
                # tiny gains.
                mlframe_models=["xgb"],
                use_mlframe_ensembles=False, verbose=0,
                # Disable per-target diagnostics so the bench measures the
                # per-target inner loop's actual feature-side work rather than the
                # LightGBM ablation fits (which dominate small-data wall-time and
                # are independent of the hoist).
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
                dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            )
            elapsed = time.perf_counter() - t0
    finally:
        pt._train_one_target = _orig_train_one
        _main.pr._train_one_target = _orig_pt_alias

    return elapsed


def _cprofile_one_run(df: pd.DataFrame, n_targets: int) -> dict:
    """cProfile a single post-fix run and return the top-15 hotspots by cumulative time."""
    import cProfile
    import pstats
    from io import StringIO

    pr = cProfile.Profile()
    pr.enable()
    _ = _time_suite(df, n_targets, force_clear_caches=False)
    pr.disable()
    sio = StringIO()
    pstats.Stats(pr, stream=sio).sort_stats("cumulative").print_stats(15)
    return {"top_15_cumulative": sio.getvalue().splitlines()[:60]}


def main() -> dict:
    # Two scales: a small sanity run + a larger run that more closely mirrors a real suite.
    # Small run also serves as a CI-fast smoke under the biz_value test (which calls this
    # main()).
    cases = [
        # Small / fast CI case - keeps the bench under 60s for the biz_value test.
        {"n_rows": 2_000, "n_features": 8, "n_targets": 5},
        # Larger case where DMatrix construction is non-trivial; hoist gains show here.
        {"n_rows": 20_000, "n_features": 30, "n_targets": 5},
    ]
    out = {"cases": []}
    for case in cases:
        df = _build_dataset(case["n_rows"], case["n_features"], case["n_targets"])
        # Warm-up: first suite run primes joblib worker pools / sklearn caches; otherwise the
        # post-fix timing absorbs one-shot overheads attributable to neither path.
        _ = _time_suite(df, case["n_targets"], force_clear_caches=True)
        t_pre = min(_time_suite(df, case["n_targets"], force_clear_caches=True) for _ in range(2))
        t_post = min(_time_suite(df, case["n_targets"], force_clear_caches=False) for _ in range(2))
        speedup_pct = 100.0 * (1.0 - t_post / t_pre) if t_pre > 0 else 0.0
        case_out = {
            **case,
            "pre_fix_wall_s": t_pre,
            "post_fix_wall_s": t_post,
            "speedup_pct": speedup_pct,
        }
        out["cases"].append(case_out)
        print(f"[bench] {case}: pre={t_pre:.3f}s post={t_post:.3f}s speedup={speedup_pct:.1f}%")

    # cProfile pass on the larger case.
    big_case = cases[-1]
    df = _build_dataset(big_case["n_rows"], big_case["n_features"], big_case["n_targets"])
    out["cprofile_top_15"] = _cprofile_one_run(df, big_case["n_targets"])

    RESULTS_PATH.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench] wrote {RESULTS_PATH}")
    return out


if __name__ == "__main__":
    main()
