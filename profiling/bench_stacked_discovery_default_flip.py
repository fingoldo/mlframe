"""T3#20 2026-05-18: ``use_stacked_discovery`` default-flip evaluation.

Background
----------
The default for ``CompositeTargetDiscoveryConfig.use_stacked_discovery``
is False (single-pass discovery). Pack #3 added the 2-pass variant
(``fit_stacked``) which appends pass-1 OOF predictions as new feature
columns for pass 2.

Question
--------
Should the default flip to True per the "Accuracy/perf over legacy"
rule? This benchmark answers it on a controlled problem with residual-
of-residual structure (where stacking SHOULD help):

    y = 1.5 * x_a + 2.0 * x_b + noise

Decision rule
-------------
Flip the default if all three are true on the residual-of-residual
benchmark:

* stacked discovers >= 1 more spec than single-pass, AND
* stacked OOF holdout RMSE <= single-pass RMSE (no harm), AND
* stacked OOF holdout RMSE < single-pass RMSE by >= 5%

Otherwise leave the default False.

Findings (2026-05-18 measure)
-----------------------------
Run this script with ``python profiling/bench_stacked_discovery_default_flip.py``
and inspect stdout. The CHANGELOG.md captures the verdict after a measured
benchmark; do not flip the default without re-running this script first.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def _build_residual_of_residual_problem(n: int = 4000, seed: int = 17):
    """HIGH#5 2026-05-18: TRUE residual-of-residual structure.

    Pre-fix this used ``y = 1.5*x_a + 2*x_b + noise`` (purely additive
    linear) which pass-1 linear-residual discovery already captures fully -
    a zero-lift result was uninformative because the benchmark itself was
    badly designed.

    True residual-of-residual: pass 1 captures the LINEAR part on x_a;
    the remaining residual has a NON-LINEAR structure on x_b (here:
    ``sin(2*pi*x_b/scale)``) that pass-1 linear-residual cannot model
    directly. Pass 2 should then discover the non-linear composite on
    the augmented feature set (OOF preds of pass 1 + raw cols).
    """
    rng = np.random.default_rng(seed)
    x_a = rng.normal(50.0, 10.0, n)
    x_b = rng.normal(0.0, 3.0, n)
    noise = rng.normal(0.0, 0.3, n)
    # Linear-on-x_a + sinusoidal-on-x_b. The sin part is invisible to
    # pass-1 linear_residual on either base; pass 2 should pick it up
    # after the linear-on-x_a component is OOF-projected out.
    y = 1.5 * x_a + 3.0 * np.sin(x_b) + noise

    df = pd.DataFrame({
        "x_a": x_a, "x_b": x_b,
        "noise1": rng.standard_normal(n),
        "noise2": rng.standard_normal(n),
        "y": y,
    })
    return df, y


def _holdout_rmse(disc, df, train_idx, holdout_idx):
    """Approximate OOF holdout RMSE: take the best spec, fit a linear residual
    model on train, predict the holdout, return RMSE."""
    from sklearn.linear_model import LinearRegression

    if not disc.specs_:
        return float("nan")

    # Just use first spec for simplicity: train its base column linear model on train.
    spec = disc.specs_[0]
    train_x = df.iloc[train_idx][spec.base_column].values.reshape(-1, 1)
    train_y = df.iloc[train_idx]["y"].values
    holdout_x = df.iloc[holdout_idx][spec.base_column].values.reshape(-1, 1)
    holdout_y = df.iloc[holdout_idx]["y"].values
    model = LinearRegression().fit(train_x, train_y)
    pred = model.predict(holdout_x)
    return float(np.sqrt(np.mean((pred - holdout_y) ** 2)))


def main() -> int:
    from mlframe.training.composite_discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    df, y = _build_residual_of_residual_problem(n=4000, seed=17)
    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    holdout_idx = np.arange(int(0.7 * n), n)
    feature_cols = ["x_a", "x_b", "noise1", "noise2"]

    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_sample_n=1500,
        composite_skip_when_raw_dominates_ratio=0.0,
    )

    print("=" * 70)
    print("use_stacked_discovery default-flip evaluation (T3#20)")
    print("=" * 70)
    print()
    print(f"Problem: y = 1.5*x_a + 3.0*sin(x_b) + noise (n={n})")
    print(f"  (true residual-of-residual: linear on x_a + non-linear on x_b)")
    print(f"Feature cols: {feature_cols}")
    print()

    # Single-pass.
    t0 = time.perf_counter()
    disc_off = CompositeTargetDiscovery(config=cfg).fit(
        df=df, target_col="y", feature_cols=feature_cols, train_idx=train_idx,
    )
    elapsed_off = time.perf_counter() - t0
    n_specs_off = len(disc_off.specs_)
    rmse_off = _holdout_rmse(disc_off, df, train_idx, holdout_idx)
    print("[OFF (single-pass)]")
    print(f"  wall: {elapsed_off:.2f}s")
    print(f"  specs discovered: {n_specs_off}")
    print(f"  holdout RMSE (best spec, linear inner): {rmse_off:.4f}")
    print()

    # Feature-stack.
    t0 = time.perf_counter()
    disc_on = CompositeTargetDiscovery(config=cfg).fit_stacked(
        df=df, target_col="y", feature_cols=feature_cols, train_idx=train_idx,
        n_oof_folds=3, max_pass1_specs_to_stack=3,
    )
    elapsed_on = time.perf_counter() - t0
    n_specs_on = len(disc_on.specs_)
    rmse_on = _holdout_rmse(disc_on, df, train_idx, holdout_idx)
    print("[ON (feature-stack)]")
    print(f"  wall: {elapsed_on:.2f}s")
    print(f"  specs discovered: {n_specs_on}")
    print(f"  holdout RMSE (best spec, linear inner): {rmse_on:.4f}")
    print()

    print("-" * 70)
    spec_delta = n_specs_on - n_specs_off
    rmse_delta_pct = (
        100.0 * (rmse_off - rmse_on) / rmse_off
        if rmse_off and not np.isnan(rmse_off) and not np.isnan(rmse_on)
        else float("nan")
    )
    print(f"Specs delta (ON - OFF): {spec_delta:+d}")
    print(f"RMSE improvement (OFF -> ON): {rmse_delta_pct:+.2f}%")
    print()

    # Decision rule.
    spec_ok = spec_delta >= 1
    no_harm = (not np.isnan(rmse_on) and not np.isnan(rmse_off)
               and rmse_on <= rmse_off + 1e-6)
    improvement_ok = (not np.isnan(rmse_delta_pct)
                      and rmse_delta_pct >= 5.0)

    if spec_ok and no_harm and improvement_ok:
        print("VERDICT: flip default to True (all three criteria met)")
        return 0
    print("VERDICT: keep default False")
    print(f"  spec_ok={spec_ok}, no_harm={no_harm}, "
          f"improvement_ok={improvement_ok}")
    print(f"  required: >=1 extra spec, RMSE no worse, RMSE >= 5% better")
    return 1


if __name__ == "__main__":
    sys.exit(main())
