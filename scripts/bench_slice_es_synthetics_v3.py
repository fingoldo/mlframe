"""Wave-3 bench: regimes I claimed slice-stable MIGHT help but didn't test.

Targets my own hedging from the previous summary:
  - heavy-tail noise (Cauchy / Student-t df=2): median_minus_mad / quantile aggregators
    should outperform mean because outlier shards distort the mean penalty;
  - tiny n_train (~200): severely overfittable -> ES timing matters more;
  - classification with rare positives: noisy AUC/log-loss on a small val;
  - non-LGB boosters: CB regression to test if the temporal-K5-mean win generalizes.

Five constructions x four aggregators x 30 paired seeds. Per-config wall ~ 30-90s.
Run from mlframe root.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
from scipy.stats import wilcoxon

from mlframe.training._callbacks import LightGBMCallback
from mlframe.training._data_helpers import _setup_eval_set
from mlframe.training._slice_helpers import build_slice_eval_sets


# ---------------------------------------------------------------------------
# Heavy-tail constructions
# ---------------------------------------------------------------------------

def _gen_heavy_tail_tiny(seed: int, n_train: int = 200, n_val: int = 80,
                          n_test: int = 2500, d: int = 4) -> tuple:
    """Tiny train + Student-t df=2 noise (Cauchy-like heavy tail) on all splits."""
    rng = np.random.default_rng(seed)
    def gen(n):
        X = rng.uniform(0, 1, (n, d))
        # Student-t df=2 has infinite variance; scale by 0.3 keeps signal-to-noise sensible
        noise = rng.standard_t(df=2, size=n) * 0.3
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + noise
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(n_train), gen(n_val), gen(n_test)


def _gen_heavy_tail_temporal(seed: int, n_train: int = 600, n_val: int = 120,
                              n_test: int = 2500, d: int = 4) -> tuple:
    """Train + test from clean dist; val time-ordered with Student-t df=2 noise ramp."""
    rng = np.random.default_rng(seed)
    def gen_clean(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    X_tr, y_tr = gen_clean(n_train, 0.3)
    X_val = rng.uniform(0, 1, (n_val, d))
    t_val = np.linspace(0, 1, n_val)
    # Heavy-tail noise scale ramps along time
    noise_scale = 0.2 + 1.2 * t_val
    y_val = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.standard_t(df=2, size=n_val) * noise_scale
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])
    X_te, y_te = gen_clean(n_test, 0.3)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


# ---------------------------------------------------------------------------
# Classification rare-class
# ---------------------------------------------------------------------------

def _gen_rare_class(seed: int, n_train: int = 2000, n_val: int = 300,
                     n_test: int = 3000, d: int = 8, p_pos: float = 0.05) -> tuple:
    rng = np.random.default_rng(seed)
    def gen(n):
        X = rng.normal(0, 1, (n, d))
        # Logit driven by first 3 features + noise
        logit = X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.5, n)
        # Shift threshold to get rare-class rate ~ p_pos
        thr = np.quantile(logit, 1.0 - p_pos)
        y = (logit > thr).astype(int)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(n_train), gen(n_val), gen(n_test)


# ---------------------------------------------------------------------------
# Fitter (LGB regression and LGB binary classification)
# ---------------------------------------------------------------------------

def _fit_one_lgb_regression(
    seed: int, scenario: str, slice_k: int, source: str = "random",
    aggregate: str = "mean", confidence: float = 0.5, quantile_level: float = 0.5,
    correlation_inflation: float = 1.0, n_estimators: int = 1500,
    learning_rate: float = 0.03, num_leaves: int = 31, patience: int = 30,
) -> float:
    if scenario == "heavy_tail_tiny":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _gen_heavy_tail_tiny(seed)
        t_val = None
    elif scenario == "heavy_tail_temporal":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_heavy_tail_temporal(seed)
    else:
        raise ValueError(scenario)

    fit_params: dict = {}
    if slice_k > 0:
        shards = build_slice_eval_sets(
            X_val, y_val, source=source, k=slice_k, min_rows_per_shard=6,
            random_state=seed,
            time_values=t_val if source == "temporal" else None,
        )
        if not shards:
            slice_k = 0

    cb = LightGBMCallback(
        patience=patience, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="l2", mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=confidence,
        slice_aggregate_quantile_level=quantile_level,
        slice_correlation_inflation=correlation_inflation,
        slice_persist_history=False, verbose=0,
    )
    fit_params["callbacks"] = [cb]
    if slice_k > 0:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val,
                        model_category="lgb", extra_eval_sets=shards)
    else:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val, model_category="lgb")

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, verbose=-1,
        num_leaves=num_leaves, random_state=seed,
    )
    model.fit(X_tr, y_tr, **fit_params)
    best_it = cb.best_iter or model.n_estimators
    preds = model.predict(X_te, num_iteration=best_it)
    # Use MAE (robust to heavy-tail test residuals — RMSE dominated by outliers)
    return float(np.mean(np.abs(preds - y_te)))


def _fit_one_lgb_classif(
    seed: int, slice_k: int, source: str = "random",
    aggregate: str = "mean", confidence: float = 0.5, quantile_level: float = 0.5,
    n_estimators: int = 1000, learning_rate: float = 0.05, num_leaves: int = 31,
    patience: int = 30, k_used: int = 3,
) -> float:
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _gen_rare_class(seed)

    fit_params: dict = {}
    if slice_k > 0:
        shards = build_slice_eval_sets(
            X_val, y_val, source=source, k=k_used,
            min_rows_per_shard=20,  # need at least a couple positives per shard
            random_state=seed,
        )
        if not shards:
            slice_k = 0

    cb = LightGBMCallback(
        patience=patience, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="binary_logloss", mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=confidence,
        slice_aggregate_quantile_level=quantile_level,
        slice_correlation_inflation=1.0,
        slice_persist_history=False, verbose=0,
    )
    fit_params["callbacks"] = [cb]
    if slice_k > 0:
        _setup_eval_set("LGBMClassifier", fit_params, X_val, y_val,
                        model_category="lgb", extra_eval_sets=shards)
    else:
        _setup_eval_set("LGBMClassifier", fit_params, X_val, y_val, model_category="lgb")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, verbose=-1,
        num_leaves=num_leaves, random_state=seed,
    )
    model.fit(X_tr, y_tr, **fit_params)
    best_it = cb.best_iter or model.n_estimators
    probs = model.predict_proba(X_te.values, num_iteration=best_it)[:, 1]
    # Test log-loss (smaller is better)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    ll = -float(np.mean(y_te * np.log(probs) + (1 - y_te) * np.log(1 - probs)))
    return ll


# ---------------------------------------------------------------------------
# CB regression (on winning heteroscedastic-temporal scenario from wave-2)
# ---------------------------------------------------------------------------

def _gen_winning_temporal(seed: int) -> tuple:
    """Same scenario as OT_temporal_k5_mean -- LGB headline winner."""
    rng = np.random.default_rng(seed)
    d = 5
    def gen_clean(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    X_tr, y_tr = gen_clean(800, 0.3)
    n_val = 120
    X_val = rng.uniform(0, 1, (n_val, d))
    t_val = np.linspace(0, 1, n_val)
    sigma_val = 0.2 + 1.8 * t_val
    y_val = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.normal(0, sigma_val)
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])
    X_te, y_te = gen_clean(2500, 0.3)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


def _fit_one_cb_regression(
    seed: int, slice_k: int, source: str = "temporal", aggregate: str = "mean",
) -> float:
    try:
        from catboost import CatBoostRegressor
    except Exception:
        return float("nan")
    from mlframe.training._callbacks import CatBoostCallback

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_winning_temporal(seed)

    fit_params: dict = {}
    if slice_k > 0:
        shards = build_slice_eval_sets(
            X_val, y_val, source=source, k=slice_k, min_rows_per_shard=8,
            random_state=seed, time_values=t_val if source == "temporal" else None,
        )
        if not shards:
            slice_k = 0

    cb_cb = CatBoostCallback(
        patience=30, min_delta=0.0,
        monitor_dataset="validation_0", monitor_metric="RMSE", mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=0.5,
        slice_correlation_inflation=1.0,
        slice_persist_history=False, verbose=0,
    )
    fit_params["callbacks"] = [cb_cb]
    if slice_k > 0:
        _setup_eval_set("CatBoostRegressor", fit_params, X_val, y_val,
                        model_category="cb", extra_eval_sets=shards)
    else:
        _setup_eval_set("CatBoostRegressor", fit_params, X_val, y_val, model_category="cb")

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.04, depth=6, verbose=False,
        random_seed=seed, allow_writing_files=False,
        # Strip native ES so our callback owns the decision
        early_stopping_rounds=None,
    )
    model.fit(X_tr, y_tr, **fit_params)
    _ = getattr(cb_cb, "best_iter", None)  # CB integration: use whatever model converged to
    preds = model.predict(X_te)
    return float(np.sqrt(np.mean((preds - y_te) ** 2)))


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def _wilcoxon(baseline, slice_es):
    diffs = np.array(baseline) - np.array(slice_es)
    if np.all(diffs == 0):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue)


def _bench_lgb_reg(name, scenario, slice_kwargs, n_seeds=30):
    t0 = time.perf_counter()
    baseline = [_fit_one_lgb_regression(seed, scenario, slice_k=0) for seed in range(n_seeds)]
    slice_es = [_fit_one_lgb_regression(seed, scenario, **slice_kwargs) for seed in range(n_seeds)]
    t1 = time.perf_counter()
    return _format(name, scenario, slice_kwargs, baseline, slice_es, n_seeds, t1 - t0)


def _bench_lgb_classif(name, slice_kwargs, n_seeds=30):
    t0 = time.perf_counter()
    baseline = [_fit_one_lgb_classif(seed, slice_k=0) for seed in range(n_seeds)]
    slice_es = [_fit_one_lgb_classif(seed, **slice_kwargs) for seed in range(n_seeds)]
    t1 = time.perf_counter()
    return _format(name, "rare_class_lgb", slice_kwargs, baseline, slice_es, n_seeds, t1 - t0)


def _bench_cb_reg(name, slice_kwargs, n_seeds=30):
    t0 = time.perf_counter()
    baseline = [_fit_one_cb_regression(seed, slice_k=0) for seed in range(n_seeds)]
    slice_es = [_fit_one_cb_regression(seed, **slice_kwargs) for seed in range(n_seeds)]
    t1 = time.perf_counter()
    return _format(name, "cb_winning_temporal", slice_kwargs, baseline, slice_es, n_seeds, t1 - t0)


def _format(name, scenario, slice_kwargs, baseline, slice_es, n_seeds, wall):
    p_value = _wilcoxon(baseline, slice_es)
    med_b, med_s = float(np.median(baseline)), float(np.median(slice_es))
    gap_pct = (med_b - med_s) / med_b * 100.0
    wins = int(np.sum(np.array(slice_es) < np.array(baseline)))
    return dict(name=name, scenario=scenario, slice_kwargs=slice_kwargs, n_seeds=n_seeds,
                median_baseline=round(med_b, 5), median_slice=round(med_s, 5),
                gap_pct=round(gap_pct, 3), p_value=round(p_value, 4), wins=wins,
                wall_s=round(wall, 1))


def main():
    n_seeds = 30
    results = []
    Path("benchmarks").mkdir(exist_ok=True)
    out_path = Path("benchmarks/slice_es_wave3_bench.json")

    def _flush():
        out_path.write_text(json.dumps(results, indent=2))

    cfgs = [
        # Heavy-tail tiny (n_train=200, Student-t df=2) -- best case for median_minus_mad / quantile
        ("HT_tiny_mean_random_k5",       "heavy_tail_tiny", dict(slice_k=5, source="random",
                                                                  aggregate="mean")),
        ("HT_tiny_tlcb09_random_k5",     "heavy_tail_tiny", dict(slice_k=5, source="random",
                                                                  aggregate="t_lcb", confidence=0.9,
                                                                  correlation_inflation=1.0)),
        ("HT_tiny_mmad_random_k5",       "heavy_tail_tiny", dict(slice_k=5, source="random",
                                                                  aggregate="median_minus_mad")),
        ("HT_tiny_q60_random_k5",        "heavy_tail_tiny", dict(slice_k=5, source="random",
                                                                  aggregate="quantile", quantile_level=0.6)),
        # Heavy-tail temporal (val time-ordered + t-df=2 noise ramp)
        ("HT_temp_mean_k5",              "heavy_tail_temporal", dict(slice_k=5, source="temporal",
                                                                       aggregate="mean")),
        ("HT_temp_tlcb09_k5",            "heavy_tail_temporal", dict(slice_k=5, source="temporal",
                                                                       aggregate="t_lcb", confidence=0.9,
                                                                       correlation_inflation=1.0)),
        ("HT_temp_mmad_k5",              "heavy_tail_temporal", dict(slice_k=5, source="temporal",
                                                                       aggregate="median_minus_mad")),
    ]
    print(f"Wave-3 LGB regression bench: {len(cfgs)} configs x {n_seeds} seeds\n")
    for name, scenario, kwargs in cfgs:
        print(f"  >>> {name} ... ", end="", flush=True)
        r = _bench_lgb_reg(name, scenario, kwargs, n_seeds=n_seeds)
        results.append(r); _flush()
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"gap={r['gap_pct']:+.2f}%  p={r['p_value']:.3f} {sig}  wins={r['wins']}/{n_seeds}  ({r['wall_s']:.0f}s)")

    # Rare-class classification (LGB binary, K=3 because n_val=300 with ~15 positives is thin)
    print("\nWave-3 LGB classification (rare-class, K=3):")
    classif_cfgs = [
        ("RC_mean_random_k3",        dict(slice_k=3, source="random", aggregate="mean", k_used=3)),
        ("RC_tlcb09_random_k3",      dict(slice_k=3, source="random", aggregate="t_lcb",
                                            confidence=0.9, k_used=3)),
    ]
    for name, kwargs in classif_cfgs:
        print(f"  >>> {name} ... ", end="", flush=True)
        r = _bench_lgb_classif(name, kwargs, n_seeds=n_seeds)
        results.append(r); _flush()
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"gap={r['gap_pct']:+.2f}%  p={r['p_value']:.3f} {sig}  wins={r['wins']}/{n_seeds}  ({r['wall_s']:.0f}s)")

    # CB regression on the LGB-winning scenario
    print("\nWave-3 CB regression on the LGB-winning heteroscedastic-temporal scenario:")
    cb_cfgs = [
        ("CB_mean_temporal_k5",      dict(slice_k=5, source="temporal", aggregate="mean")),
        ("CB_tlcb09_temporal_k5",    dict(slice_k=5, source="temporal", aggregate="t_lcb")),
    ]
    for name, kwargs in cb_cfgs:
        print(f"  >>> {name} ... ", end="", flush=True)
        r = _bench_cb_reg(name, kwargs, n_seeds=15)  # CB slower, fewer seeds
        results.append(r)
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"gap={r['gap_pct']:+.2f}%  p={r['p_value']:.3f} {sig}  wins={r['wins']}/{r['n_seeds']}  ({r['wall_s']:.0f}s)")

    print("\n=== Summary (sorted by p_value asc) ===")
    for r in sorted(results, key=lambda x: x["p_value"]):
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"  {r['name']:30s} gap={r['gap_pct']:+6.2f}%  p={r['p_value']:.4f}  wins={r['wins']}/{r['n_seeds']}  {sig}")

    _flush()
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
