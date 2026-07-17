"""biz_value tests for slice-stable ES.

These are *value-of-the-feature* tests: each construction is a synthetic dataset designed to
expose a failure mode of single-val ES that slice-stable ES is supposed to fix. They are
marked ``slow`` because they fit many real boosters across many seeds and use ``pytest -m
slow`` (or ``--run-slow``) to gate them in CI. The fast verification config (5 seeds) runs
by default so a regression in the value-prop surfaces fast.

Per the project rule "every new feature: unit + biz_value + cProfile": if the value isn't
*measurable* on a constructed case where the failure mode is real, the feature shouldn't ship
as default-on. The default config (``enabled=True, source="temporal", aggregate="mean", K=5``)
is calibrated against the headline test below: ``OT_temporal_k5_mean`` showed
``gap=+0.55%, p=0.006***, wins=9/30`` over 30 paired seeds on a heteroscedastic-by-time val.

### Honest mechanism note
The bench-validated gain at LGB-regressor scale is **dominated by the effective_patience
auto-bump** (``patience * (1 + 1/sqrt(K-1))`` -- ``x1.5`` for K=5), NOT by the aggregator's
variance penalty. ``aggregate="mean"`` over a K-shard partition is mathematically identical
to ``mean(full val)``; the only behavioural difference vs single-val ES is the longer patience
tolerance baked into ``effective_patience``. The variance-aware aggregators (``t_lcb``
confidence>=0.6, ``quantile`` level>=0.7, ``median_minus_mad``) hurt at the benched LGB scale --
they introduce more decision noise than they remove. Those modes are knobs for noisier
regimes (heavy-tail residuals, n_train < ~500, severely under-regularized boosters) that need
their own validation studies before being made default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _gen_heteroscedastic_temporal(seed: int, n_train: int = 800, n_val: int = 120, n_test: int = 2500, d: int = 5) -> tuple:
    """Headline construction: temporal val with sigma ramping from 0.2 to 2.0 across rows."""
    rng = np.random.default_rng(seed)

    def gen_clean(n, sigma):
        """Gen clean."""
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y

    X_tr, y_tr = gen_clean(n_train, 0.3)

    X_val = rng.uniform(0, 1, (n_val, d))
    t_val = np.linspace(0, 1, n_val)
    sigma_val = 0.2 + 1.8 * t_val
    y_val = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.normal(0, sigma_val)
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])

    X_te, y_te = gen_clean(n_test, 0.3)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


def _fit_one(
    seed: int,
    slice_k: int = 0,
    source: str = "temporal",
    aggregate: str = "mean",
    n_estimators: int = 1000,
    learning_rate: float = 0.04,
    num_leaves: int = 31,
    patience: int = 25,
) -> float:
    """Fit one LGB on the heteroscedastic-temporal synthetic; return test RMSE."""
    import lightgbm as lgb
    from mlframe.training.callbacks._callbacks import LightGBMCallback
    from mlframe.training._data_helpers import _setup_eval_set
    from mlframe.training.slicing._slice_helpers import build_slice_eval_sets

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_heteroscedastic_temporal(seed)

    fit_params: dict = {}
    extra_eval_sets = None
    if slice_k > 0:
        extra_eval_sets = build_slice_eval_sets(
            X_val,
            y_val,
            source=source,
            k=slice_k,
            min_rows_per_shard=8,
            random_state=seed,
            time_values=t_val if source == "temporal" else None,
        )
        if not extra_eval_sets:
            slice_k = 0

    cb = LightGBMCallback(
        patience=patience,
        min_delta=0.0,
        monitor_dataset="valid_0",
        monitor_metric="l2",
        mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=0.5,  # zero LCB penalty -> behaviour from patience only
        slice_correlation_inflation=1.0,
        slice_persist_history=False,
        verbose=0,
    )
    fit_params["callbacks"] = [cb]
    if slice_k > 0:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val, model_category="lgb", extra_eval_sets=extra_eval_sets)
    else:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val, model_category="lgb")

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbose=-1,
        num_leaves=num_leaves,
        random_state=seed,
    )
    model.fit(X_tr, y_tr, **fit_params)
    best_it = cb.best_iter or model.n_estimators
    preds = model.predict(X_te, num_iteration=best_it)
    return float(np.sqrt(np.mean((preds - y_te) ** 2)))


def _wilcoxon_one_sided(baseline: list[float], slice_es: list[float]) -> float:
    """Wilcoxon one sided."""
    from scipy.stats import wilcoxon

    diffs = np.array(baseline) - np.array(slice_es)
    if np.all(diffs == 0):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue)


def test_biz_value_headline_smoke_5_seeds() -> None:
    """5-seed smoke for the headline temporal-K5-mean config: feature shouldn't catastrophically regress.

    The strict statistical validation runs under ``@pytest.mark.slow``.
    """
    pytest.importorskip("lightgbm")
    baseline = [_fit_one(seed, slice_k=0) for seed in range(5)]
    slice_es = [_fit_one(seed, slice_k=5, source="temporal", aggregate="mean") for seed in range(5)]
    med_b = float(np.median(baseline))
    med_s = float(np.median(slice_es))
    assert med_s < 3.0 * med_b, f"slice-stable catastrophic regression: median(slice)={med_s:.4f} vs median(baseline)={med_b:.4f}"


@pytest.mark.slow
def test_biz_value_lgb_temporal_K5_mean_observed_gap_documented() -> None:
    """Observability-only: the LGB+temporal+K5+mean win observed on the bench.

    On the calibration bench this configuration gave ``gap=+0.55%, p=0.006***, wins=9/30``
    over 30 paired seeds. **The win does NOT generalise** -- wave-3 ran the same configuration
    on CatBoost regression (same scenario) and saw ``gap=-0.50%, p=0.996`` and on every other
    LGB regime tested (heavy-tail tiny, heavy-tail temporal, rare-class classification) the
    slice-stable variants either tied or lost. Reading the +0.55% as a LGB-specific
    idiosyncrasy of the multi-eval-set registration + ``effective_patience`` interaction is
    the honest take.

    This test runs the bench and PRINTS the numbers, but asserts only against catastrophic
    regression (gap >= -5%) -- it is **not** a value-prop gate. The strict gate was removed
    after the empirical study revealed the win was overfit to a single (booster, scenario,
    aggregator) triple.
    """
    pytest.importorskip("lightgbm")
    pytest.importorskip("scipy")

    baseline = [_fit_one(seed, slice_k=0) for seed in range(30)]
    slice_es = [_fit_one(seed, slice_k=5, source="temporal", aggregate="mean") for seed in range(30)]
    med_b, med_s = float(np.median(baseline)), float(np.median(slice_es))
    p_value = _wilcoxon_one_sided(baseline, slice_es)
    gap_pct = (med_b - med_s) / med_b * 100.0
    wins = int(np.sum(np.array(slice_es) < np.array(baseline)))
    print(
        f"\nObservability (n=30): median baseline={med_b:.4f}, median slice={med_s:.4f}, "
        f"gap={gap_pct:+.2f}%, p_value={p_value:.4f}, wins={wins}/30 "
        f"(NOT a value-prop gate; see test docstring)"
    )
    # Catastrophic-regression guard only.
    assert gap_pct >= -5.0, (
        f"slice-stable catastrophic regression on the LGB-temporal-K5-mean scenario: baseline median={med_b:.4f}, slice median={med_s:.4f}, gap={gap_pct:+.2f}%"
    )


def test_biz_value_default_config_is_disabled() -> None:
    """The default ``SliceStableESConfig`` is OFF (post-empirical-study honesty).

    The infrastructure ships intact; the empirical 27-config study did not find a
    generalisable value-prop and the default reverted accordingly.
    """
    from mlframe.training.configs import SliceStableESConfig

    cfg = SliceStableESConfig()
    assert cfg.enabled is False
    assert cfg.diagnostic_only is False
    # source / aggregate / pareto_plot_enabled defaults remain for operators who opt-in.
    assert cfg.source == "temporal"
    assert cfg.aggregate == "mean"
    assert cfg.k == 5
