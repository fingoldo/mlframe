"""biz_value: per-target feature-side cache hoist must NOT regress wall-time.

Runs a small multi-target XGBoost suite with the hoist active vs forcibly cleared between
targets (pre-fix emulation). Asserts non-regression: post-fix wall-time may not exceed
pre-fix by more than the noise floor.

The "must improve by X%" floor is intentionally absent: the suite's wall-time on a 2k-row
fast-CI case is dominated by the actual xgb.train fit (~2.5s per target on a single CPU
core); DMatrix build is sub-100ms on this scale. The hoist's measured wall-time speedup
becomes material only at n_rows >= 50k. For CI we assert the hoist does not regress and
the caches do populate (proven separately in
test_per_target_feature_side_cache_reuse.py). The actual scaling speedup is captured in
``bench_per_target_hoist.json`` (5-15% on the 20k case, larger on multi-strategy suites).

NOISE_PCT_FLOOR is set generously to keep this test stable on a shared CI host where 5-10%
run-to-run variance is normal.
"""

from __future__ import annotations

import gc
import time
import warnings

import numpy as np
import pandas as pd
import pytest


NOISE_PCT_FLOOR = 25.0  # Post-fix wall-time may not exceed pre-fix by >25%.


class _MultiTargetExtractor:
    """Groups tests covering multi target extractor."""
    def __init__(self, target_columns, target_type):
        self.target_columns = tuple(target_columns)
        self.target_type = target_type
        self.ts_field = None
        self.group_field = None
        self.weight_schemas = None
        self.target_carrier = "numpy"

    def transform(self, df):
        """Transform."""
        target_by_type = {self.target_type: {}}
        for col in self.target_columns:
            if isinstance(df, pd.DataFrame):
                target_by_type[self.target_type][col] = df[col].values
            else:
                target_by_type[self.target_type][col] = df[col].to_numpy()
        cols_to_drop = list(self.target_columns)
        return (df, target_by_type, None, None, None, None, cols_to_drop, {})


def _make_panel(n_rows: int, n_features: int, n_targets: int, seed: int = 2026) -> pd.DataFrame:
    """Make panel."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    for t in range(n_targets):
        a, b, c = t % n_features, (t + 7) % n_features, (t + 13) % n_features
        df[f"y{t}"] = (X[:, a] - 0.6 * X[:, b] + 0.4 * X[:, c] + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
    return df


def _time_suite(df: pd.DataFrame, n_targets: int, *, force_clear: bool) -> float:
    """Time suite."""
    from mlframe.training import train_mlframe_models_suite, TargetTypes
    from mlframe.training.configs import BaselineDiagnosticsConfig, DummyBaselinesConfig
    import mlframe.training.core._phase_train_one_target as pt
    import mlframe.training.core._phase_runners as pr_mod

    fte = _MultiTargetExtractor(
        [f"y{t}" for t in range(n_targets)],
        target_type=TargetTypes.REGRESSION,
    )
    _orig = pt._train_one_target

    def _wrapped(ctx, target_type, targets, cur_target_name, cur_target_values):
        """Wrapped."""
        if force_clear:
            arts = ctx.artifacts or {}
            arts.pop("feature_side_cache", None)
            arts.pop("dataset_reuse_cache", None)
        return _orig(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _wrapped
    _orig_alias = pr_mod._train_one_target
    pr_mod._train_one_target = _wrapped

    gc.collect()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            train_mlframe_models_suite(
                df=df,
                target_name="multi",
                model_name="biz_val",
                features_and_targets_extractor=fte,
                mlframe_models=["xgb"],
                use_mlframe_ensembles=False,
                verbose=0,
                baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
                dummy_baselines_config=DummyBaselinesConfig(enabled=False),
            )
            elapsed = time.perf_counter() - t0
    finally:
        pt._train_one_target = _orig
        pr_mod._train_one_target = _orig_alias
    return elapsed


@pytest.mark.slow
def test_per_target_hoist_does_not_regress_wall_time():
    """Sentinel: the hoist must NOT make the suite slower than the pre-fix path.

    Floor is the non-regression band (25% slack to absorb CI noise). The actual speedup
    on production-scale (n_rows>=50k, multi-model) is documented in the benchmark JSON.
    """
    pytest.importorskip("xgboost")
    df = _make_panel(2_000, 8, 5)
    n_targets = 5

    # Warm-up: prime the joblib / sklearn cache pools so neither path absorbs first-run
    # cost.
    _ = _time_suite(df, n_targets, force_clear=True)

    t_pre = min(_time_suite(df, n_targets, force_clear=True) for _ in range(2))
    t_post = min(_time_suite(df, n_targets, force_clear=False) for _ in range(2))

    overhead_pct = 100.0 * (t_post - t_pre) / t_pre
    print(f"[biz_val per-target hoist] pre={t_pre:.2f}s post={t_post:.2f}s overhead={overhead_pct:+.1f}% (floor: <={NOISE_PCT_FLOOR}%)")
    assert overhead_pct <= NOISE_PCT_FLOOR, (
        f"per-target hoist regressed wall-time by {overhead_pct:.1f}% (pre={t_pre:.2f}s, post={t_post:.2f}s); floor is {NOISE_PCT_FLOOR}%"
    )
