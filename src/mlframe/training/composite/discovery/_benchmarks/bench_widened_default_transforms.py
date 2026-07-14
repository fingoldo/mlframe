"""Benchmark: cost/win tradeoff of widening ``CompositeTargetDiscoveryConfig``'s default
``transforms`` candidate list with the SINGLE-BASE, DROP-IN additions ``box_cox_y``,
``seasonal_residual``, ``nadaraya_watson_residual``, ``gaussian_copula_residual``.

VERDICT: REJECTED for default-pool membership. The full-pool A/B below (the real production
comparison -- current default vs current default + these 4) measures 0.00% net RMSE change on
both the positive and negative-control DGPs (an existing generalist, ``quantile_normal_y``, already
covers this DGP class) at a real wall-clock cost (observed +0.5% to +1433% across repeated runs on
this shared, contended machine -- see the run-to-run variance note below). All four transforms
remain fully registered and usable via explicit ``transforms=[...]`` (see
``test_biz_val_widened_default_transforms.py`` for their validated per-transform value in
isolation); only default-pool membership is rejected, on measured cost/benefit.

Two synthetic frames:

* POSITIVE (``_lognormal_frame``): ``y = exp(1.1*base + 0.15*x0 + N(0, 0.5))`` -- a monotone but
  heavy-tailed / distorted-marginal dependence on ``base``. ``gaussian_copula_residual`` (NEW) maps
  both y and base through their train empirical CDFs into normal-scores space before the OLS fit,
  which handles the marginal distortion better than ``monotonic_residual`` (bounded PCHIP spline in
  raw y-units) or ``rank_residual`` (OLS in rank/uniform space, whose leverage compresses near the
  [0, 1] boundary the heavy right tail pushes rows into). Compared NARROW-old-style vs NARROW+NEW
  transform lists (both small, apples-to-apples) rather than the full current default list: the
  full OLD default already contains ``quantile_normal_y`` (a y-only ECDF-normal unary transform),
  which is such a strong general-purpose baseline on this fixture that it wins the FULL-pool
  comparison regardless of the new additions (see the "full-pool" numbers below) -- a real, honest
  finding, not a benchmark artifact: on typical single-base synthetic DGPs the marginal net RMSE
  win from widening is small/zero once a strong existing generalist is already in the pool, while
  G2 (the honest-holdout gate) still protects against any new transform that does NOT transfer.
  Note: ``seasonal_residual`` and ``volatility_normalized_residual`` were also tried as the positive
  case and REJECTED as benchmark fixtures after measurement:
    - ``seasonal_residual`` is ``requires_base=False`` (unary, phase-in-batch only): on a period=7
      DGP (``y = phase_effect(t%7) + 0.3*x0 + 0.05*base + N(0, 0.35)``, n=3000) it measured
      ``mi_gain=-0.0070`` at the SCREENING stage already (before the honest-holdout gate even runs)
      and honest y-RMSE=2.276 (raw=2.269) -- i.e. it ties raw rather than clearly winning, even
      though phase is the dominant term. The discovery MI-screening path does not guarantee the
      exact chronological/gap-free row order the transform's docstring assumes, so it cannot
      demonstrate its designed advantage through that path on this fixture -- a real, measured
      limitation, not a benchmark bug. On a DGP with NO periodic structure at all (wide-range base,
      plain linear DGP, ``_wide_range_linear_frame`` in the regression test) it measured honest
      y-RMSE=122.6 vs raw=12.96 -- expected, since being unary it ignores the dominant base signal
      entirely; G2 correctly rejects it there (pinned by
      ``test_biz_val_widened_transforms_g2_still_rejects_seasonal_when_irrelevant``).
    - ``volatility_normalized_residual`` is correctly tagged ``recurrent=True`` (full-sequence-then-
      mask evaluation under the honest-holdout's i.i.d. row-subset split), but measured no honest-RMSE
      edge over ``ewma_residual``/raw y on two heteroscedastic-regime random-walk DGPs (both
      candidate specs dropped by the G2 gate in both attempts) -- variance-stabilisation changes the
      loss surface a downstream MODEL trains on, but does not by itself reduce achievable y-scale
      RMSE on an otherwise-unpredictable noise term, so it is not a good fit for an RMSE-based
      benchmark demonstration.
* NEGATIVE control (``_linear_frame``): plain ``y = alpha*base + beta + N(0, sigma)`` -- the true
  DGP is exactly ``linear_residual``, already in the OLD list. Confirms the widened pool is not
  outvoted by spurious NEW-transform rejects and does not meaningfully slow discovery down when
  the new transforms are irrelevant.

Run: python -m mlframe.training.composite.discovery._benchmarks.bench_widened_default_transforms

Measured on this machine (Windows / python 3.14, CUDA_VISIBLE_DEVICES="", screening="mi",
honest_rmse_gate_enabled=True, tiny_model_n_estimators=40, random_state=0, 1 rep). This machine
runs several other agents concurrently in the shared repo, so absolute wall-clock numbers below
have real run-to-run variance (an earlier isolated run of the same narrow-list comparison measured
2.10s/3.90s instead of the 4.77s/16.20s below) -- the RMSE numbers (deterministic given
``random_state``) are the load-bearing measurement; wall-clock is directional only:

    POSITIVE (lognormal, n=4000):
      narrow-list isolation (transforms=[diff, additive_residual, linear_residual,
      monotonic_residual, rank_residual] vs the same 5 + the 4 new transforms):
        NARROW-OLD : 4.77 s/fit, 8 specs, best=monotonic_residual, honest y-RMSE=36.308 (raw=36.644)
        NARROW+NEW : 16.20 s/fit, 14 specs, best=gaussian_copula_residual, honest y-RMSE=35.819 (raw=36.644)
        overhead: +11.43 s (+239.7%, noisy -- see note above); RMSE improvement of NEW over
        best-narrow-OLD: 1.35%; over raw: 2.25%
      full-pool comparison (current 24-entry OLD default vs current 28-entry NEW default):
        OLD (24 transforms) : 12.93 s/fit, 29 specs survive, best=quantile_normal_y, honest y-RMSE=35.602
        NEW (28 transforms) : 13.00 s/fit, 29 specs survive, best=quantile_normal_y, honest y-RMSE=35.602 (unchanged)
        overhead: +0.07 s (+0.5%) for zero net RMSE change on THIS fixture -- an already-strong
        y-only generalist (quantile_normal_y) dominates both pools; gaussian_copula_residual (35.819)
        still ranks 2nd, ahead of every OLD base-dependent transform.

    NEGATIVE control (plain linear_residual DGP, n=3000), full-pool comparison:
        OLD (24 transforms) : 12.16 s/fit, 15 specs, best=chain_linres_cbrt, honest RMSE=5.373 (raw=12.959)
        NEW (28 transforms) : 5.54 s/fit, 17 specs, same best spec+RMSE (box_cox_y / gaussian_copula_residual /
                   nadaraya_watson_residual also survive G2 but do not outrank it)
        best-spec RMSE unchanged (0.00%) -- the widened pool is not outvoted by spurious
        NEW-transform rejects on a DGP where none of the new transforms are the true structure.
"""
from __future__ import annotations

import statistics
from timeit import default_timer as timer

import numpy as np
import pandas as pd

# The OLD (pre-widen) default list, frozen here for the A/B comparison -- must NOT be imported
# from the config module (which now holds the NEW list).
_OLD_TRANSFORMS = [
    "diff", "additive_residual", "median_residual",
    "ratio", "logratio", "linear_residual",
    "linear_residual_robust",
    "quantile_residual", "monotonic_residual",
    "y_quantile_clip",
    "cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y",
    "chain_linres_cbrt", "chain_linres_yj",
    "chain_monres_cbrt", "chain_monres_yj",
    "asinh_residual", "centered_ratio", "polynomial_residual_deg2",
    "rank_residual", "smoothing_spline_residual",
    "reciprocal_residual",
]

# A small "old-style" subset (pre-dating the unary-y-transform additions like quantile_normal_y),
# for the apples-to-apples POSITIVE-fixture comparison: isolates the marginal value of the 4 new
# transforms without a strong pre-existing generalist absorbing the win.
_NARROW_OLD_TRANSFORMS = ["diff", "additive_residual", "linear_residual", "monotonic_residual", "rank_residual"]
_NEW_EXTRA_TRANSFORMS = ["box_cox_y", "seasonal_residual", "nadaraya_watson_residual", "gaussian_copula_residual"]


def _lognormal_frame(n: int = 4000, seed: int = 11):
    """Positive DGP: monotone but heavy-tailed/distorted-marginal dependence on ``base``."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 5.0, n)
    x0 = rng.normal(size=n)
    y = np.exp(1.1 * base + 0.15 * x0 + rng.normal(0.0, 0.5, n))
    df = pd.DataFrame({"base": base, "x0": x0, "noise0": rng.normal(size=n), "y": y})
    return df, [c for c in df.columns if c != "y"]


def _linear_frame(n: int = 3000, seed: int = 1):
    """Negative-control DGP: plain linear_residual is the exact true structure."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    y = 2.5 * base + 10.0 + rng.normal(0.0, 5.0, n)
    df = pd.DataFrame({"base": base, "x0": rng.normal(size=n), "noise0": rng.normal(size=n), "y": y})
    return df, [c for c in df.columns if c != "y"]


def _run_fit(df: pd.DataFrame, feats: list[str], transforms: list[str], base_candidates: list[str], reps: int = 1):
    """Run ``CompositeTargetDiscovery.fit`` once (or ``reps`` times, median wall time) with the given transform list."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    times = []
    disc = None
    for _r in range(reps):
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, random_state=0, screening="mi", base_candidates=list(base_candidates),
            honest_holdout_frac=0.2, tiny_model_n_estimators=40, auto_base_null_perms=0,
            multi_base_enabled=False, interaction_base_discovery_enabled=False,
            auto_chain_discovery_enabled=False, honest_rmse_gate_enabled=True,
            transforms=list(transforms),
        )
        disc = CompositeTargetDiscovery(cfg)
        t0 = timer()
        disc.fit(df, "y", feats, np.arange(len(df)))
        times.append(timer() - t0)
    return statistics.median(times), disc


def _best_spec(disc):
    """(transform_name, honest_holdout_rmse) of the lowest-RMSE surviving spec, or ``None`` if nothing survived."""
    scored = [(s.transform_name, s.honest_holdout_rmse) for s in disc.specs_ if s.honest_holdout_rmse is not None]
    return min(scored, key=lambda p: p[1]) if scored else None


def _raw_rmse(disc):
    """The raw-y tiny-model baseline RMSE recorded on any surviving spec (identical across specs of one fit)."""
    for s in disc.specs_:
        if s.honest_holdout_raw_rmse is not None:
            return float(s.honest_holdout_raw_rmse)
    return None


def _report_pair(label: str, df: pd.DataFrame, feats: list[str], base_candidates: list[str], old_list: list[str], new_list: list[str]) -> None:
    """Fit both the old and new transform pools on ``df`` and print a wall-time/best-spec/raw-RMSE comparison."""
    t_old, disc_old = _run_fit(df, feats, old_list, base_candidates)
    t_new, disc_new = _run_fit(df, feats, new_list, base_candidates)
    best_old = _best_spec(disc_old)
    best_new = _best_spec(disc_new)
    raw = _raw_rmse(disc_old)
    print(f"\n== {label} ==")
    print(f"OLD : {t_old:.2f} s/fit, {len(disc_old.specs_)} specs, best={best_old}, raw={raw}")
    print(f"NEW : {t_new:.2f} s/fit, {len(disc_new.specs_)} specs, best={best_new}, raw={raw}")
    overhead = t_new - t_old
    pct = (overhead / t_old * 100.0) if t_old > 0 else float("nan")
    print(f"wall-clock overhead: {overhead:+.2f} s ({pct:+.1f}%)")
    if best_old is not None and best_new is not None and best_old[1] > 0:
        gain = (best_old[1] - best_new[1]) / best_old[1] * 100.0
        print(f"RMSE improvement of NEW-best over OLD-best: {gain:.2f}%")


def main() -> None:
    """Run the positive (lognormal, narrow + full-pool) and negative-control (linear) A/B comparisons."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    # Read the default transforms list off a plain instance rather than calling the FieldInfo's
    # ``default_factory`` directly: pydantic types that attribute as a union that also accepts a
    # validated-data-aware ``Callable[[dict], Any]`` variant, so a zero-arg call is a mypy
    # "too few arguments" false positive even with a ``not None`` guard.
    full_new = CompositeTargetDiscoveryConfig().transforms

    df_ln, feats_ln = _lognormal_frame()
    _report_pair(
        "POSITIVE lognormal -- narrow-list isolation", df_ln, feats_ln, ["base", "x0"],
        _NARROW_OLD_TRANSFORMS, _NARROW_OLD_TRANSFORMS + _NEW_EXTRA_TRANSFORMS,
    )
    _report_pair(
        "POSITIVE lognormal -- full-pool comparison", df_ln, feats_ln, ["base", "x0"],
        _OLD_TRANSFORMS, full_new,
    )

    df_lin, feats_lin = _linear_frame()
    _report_pair(
        "NEGATIVE control -- full-pool comparison", df_lin, feats_lin, ["base"],
        _OLD_TRANSFORMS, full_new,
    )


if __name__ == "__main__":
    main()
