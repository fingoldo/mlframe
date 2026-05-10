"""Profile harness for composite-target hotfixes 1/2/3 + R3.18 multilabel.

Profiles:
  A. Raw-y baseline gate (_tiny_cv_rmse_raw_y + tiny_model_rerank gate logic)
  B. Hint precompute + per-target config clone (model_copy)
  C. Multilabel expansion (2-D -> k 1-D in target_by_type)
  D. Plot helpers (per_fold_rmse, per_family_disagreement, alpha_stability,
     predictions_vs_actual)
  E. corr-threshold filter + filter_drops bookkeeping

Output: cumulative-time top-30 functions per code path.
"""
import cProfile
import pstats
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


def _make_tvt_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10, scale=3, size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * f1 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"base": base, "f1": f1, "f2": f2, "y": y})


def code_path_a_raw_baseline_gate():
    """Discovery with raw-y gate enabled (default). The gate adds
    one tiny LGBM CV fit per family on the same screening sample.
    """
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    df = _make_tvt_data(n=4000, seed=0)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="hybrid",
        mi_sample_n=2000, tiny_model_sample_n=1500,
        tiny_model_n_estimators=60, tiny_model_cv_folds=3,
        eps_mi_gain=-1.0,
        require_beats_raw_baseline=True,
        raw_baseline_tolerance=1.05,
        random_state=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, target_col="y", feature_cols=["base", "f1", "f2"],
             train_idx=np.arange(3200))


def code_path_a_no_gate():
    """Same as A but with require_beats_raw_baseline=False to isolate
    the cost of the gate alone."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    df = _make_tvt_data(n=4000, seed=0)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="hybrid",
        mi_sample_n=2000, tiny_model_sample_n=1500,
        tiny_model_n_estimators=60, tiny_model_cv_folds=3,
        eps_mi_gain=-1.0,
        require_beats_raw_baseline=False,  # skip gate
        random_state=0,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, target_col="y", feature_cols=["base", "f1", "f2"],
             train_idx=np.arange(3200))


def code_path_b_hint_precompute():
    """Suite-level hint precompute: BaselineDiagnostics inline +
    per-target config clone via model_copy. Synthesised here (no
    full suite) by directly calling BaselineDiagnostics + Pydantic
    model_copy in a tight loop."""
    from mlframe.training.baseline_diagnostics import (
        BaselineDiagnostics,
    )
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, CompositeTargetDiscoveryConfig,
    )
    df = _make_tvt_data(n=4000, seed=0)
    bd_cfg = BaselineDiagnosticsConfig(enabled=True, sample_n=4000)
    base_cfg = CompositeTargetDiscoveryConfig(
        enabled=True, use_baseline_diagnostics_hint=True,
    )
    # Run BD once; then clone N times to reflect a multi-target suite.
    bd = BaselineDiagnostics(bd_cfg)
    report = bd.fit_and_report(
        train_df=df, train_target=df["y"].to_numpy(),
        feature_cols=["base", "f1", "f2"],
        target_type="regression", target_name="y",
        cat_features=None,
    )
    diag = report.to_dict()
    ablation = sorted(
        diag.get("ablation", []),
        key=lambda e: -float(e.get("delta_pct", 0.0)),
    )
    hint_cols = [e["feature"] for e in ablation[:3]]
    # Clone N times.
    for _ in range(20):
        _ = base_cfg.model_copy(update={"dominant_features_hint": hint_cols})


def code_path_c_multilabel_expansion():
    """Suite-level multilabel expansion: 2-D y entry -> k 1-D
    entries in target_by_type."""
    rng = np.random.default_rng(0)
    n = 4_000_000  # production-like row count
    k = 5  # 5-output regression
    y_2d = rng.normal(size=(n, k))
    target_by_type = {"regression": {"multi_y": y_2d}}
    # Mirror the suite-level expansion code in core.py:3370.
    expanded = dict(target_by_type["regression"])
    for tn, tv in list(target_by_type["regression"].items()):
        arr = np.asarray(tv)
        if arr.ndim == 2 and arr.shape[1] >= 1:
            for j in range(arr.shape[1]):
                expanded[f"{tn}_out{j}"] = arr[:, j]
            expanded.pop(tn, None)
    target_by_type["regression"] = expanded


def code_path_d_plots():
    """Cost of the 4 new plot helpers on realistic-size inputs."""
    import matplotlib
    matplotlib.use("Agg")
    from mlframe.training.composite_diagnostics import (
        plot_per_fold_tiny_rmse, plot_per_family_disagreement,
        plot_alpha_stability, plot_predictions_vs_actual,
    )
    rng = np.random.default_rng(0)
    fig1 = plot_per_fold_tiny_rmse(
        {f"spec_{i}": rng.normal(loc=1.0, scale=0.1, size=10).tolist()
         for i in range(8)},
        raw_baseline=1.5,
    )
    fig2 = plot_per_family_disagreement(
        {f: rng.normal(size=20).tolist()
         for f in ["lgb", "xgb", "cb", "linear"]},
        spec_names=[f"s{i}" for i in range(20)],
    )
    fig3 = plot_alpha_stability(
        rng.normal(loc=0.95, scale=0.02, size=50).tolist(),
        expected_alpha=0.95,
    )
    y_true = rng.normal(size=100_000)
    y_preds = {
        f"spec_{i}": y_true + rng.normal(scale=0.1, size=100_000)
        for i in range(4)
    }
    fig4 = plot_predictions_vs_actual(y_true, y_preds, sample_n=5000)
    import matplotlib.pyplot as plt
    plt.close("all")


def code_path_e_filter_drops():
    """corr-threshold filter on a wide feature_cols list with mixed
    survivability. Tests the cost of filter_drops bookkeeping."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    rng = np.random.default_rng(0)
    n = 100_000
    n_features = 200
    y = rng.normal(size=n)
    cols = {}
    for i in range(n_features):
        if i % 7 == 0:
            cols[f"target_enc_f{i}"] = y * 0.5 + rng.normal(scale=0.1, size=n)
        elif i % 11 == 0:
            cols[f"y_smooth_f{i}"] = y + rng.normal(scale=1e-6, size=n)
        else:
            cols[f"f{i}"] = rng.normal(size=n)
    cols["y"] = y
    df = pd.DataFrame(cols)
    feature_cols = [c for c in df.columns if c != "y"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="mi", mi_sample_n=2000,
        eps_mi_gain=-100.0,
        base_candidates="auto", auto_base_top_k=2,
        transforms=["diff"],
    )
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, target_col="y", feature_cols=feature_cols,
             train_idx=np.arange(80_000))


def profile_one(name, fn, top_n=30, sort_by="cumulative"):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    fn()
    pr.disable()
    elapsed = time.perf_counter() - t0
    stats = pstats.Stats(pr).sort_stats(sort_by)
    print(f"  WALL TIME: {elapsed:.3f}s")
    stats.print_stats(top_n)


if __name__ == "__main__":
    # Warmup imports so first-import time doesn't dominate the
    # profiles. Imports are NOT what we want to measure; we want the
    # algorithmic hot spots.
    import lightgbm  # noqa: F401
    from mlframe.training.composite import CompositeTargetDiscovery  # noqa: F401
    from mlframe.training.composite_diagnostics import plot_predictions_vs_actual  # noqa: F401
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics  # noqa: F401
    print("--- WARMUP DONE ---\n")

    profile_one("A. Raw-y baseline gate (with gate ON)",
                code_path_a_raw_baseline_gate)
    profile_one("A'. Same discovery WITHOUT gate (compare)",
                code_path_a_no_gate)
    profile_one("B. Hint precompute (BD inline + 20x model_copy)",
                code_path_b_hint_precompute)
    profile_one("C. Multilabel 4M-row x 5-output expansion",
                code_path_c_multilabel_expansion)
    profile_one("D. 4 plot helpers on realistic inputs",
                code_path_d_plots)
    profile_one("E. corr-threshold filter on 200 features x 100K rows",
                code_path_e_filter_drops)
