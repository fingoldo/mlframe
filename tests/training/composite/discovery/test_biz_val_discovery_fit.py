"""Audit-fix tests for ``mlframe.training.composite.discovery._fit`` (2026-06-10).

Covers three audit findings landed against ``_fit.py``:

* **M1** -- the heavy-tail ``mi_n_strata`` boost was a dead no-op while
  ``mi_sample_strategy`` defaulted to ``"random"`` (the uniform draw never
  consulted the strata). The default is flipped to ``"stratified_quantile"``;
  the biz_value test asserts the stratified draw materially raises tail
  coverage on a skewed target relative to the uniform draw.
* **P14** -- the alpha-drift gate and the ``linear_residual``->``diff``
  collapse pass re-extracted full base columns instead of reusing the already
  materialised ``self._auto_base_pool[base]`` (== ``base_full[train_idx]``).
  The optimisation must be BIT-IDENTICAL: same kept specs + same drift z-scores.
* **D8** -- ``report()`` froze ``kept=True`` at the eps gate, so specs dropped
  by the downstream top-K trim / alpha-drift / collapse / rerank still claimed
  ``kept=True``, contradicting the report contract. The reconciliation pass
  flips those rows to ``kept=False`` and records the post-gate drop reason; the
  set of ``kept=True`` report rows must equal the final ``specs_`` set.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# M1: stratified-quantile sampling is the default + raises tail coverage
# ----------------------------------------------------------------------


def test_biz_val_mi_sample_strategy_default_is_stratified_quantile() -> None:
    """The config default flipped ``random`` -> ``stratified_quantile`` so the
    heavy-tail ``mi_n_strata`` boost actually steers the MI screen."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    cfg = CompositeTargetDiscoveryConfig()
    assert cfg.mi_sample_strategy == "stratified_quantile"


def _per_stratum_counts(idx: np.ndarray, y: np.ndarray, n_strata: int) -> np.ndarray:
    """How many sampled rows fall in each quantile stratum of ``y``."""
    qs = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
    cuts = np.quantile(y, qs)
    st = np.clip(np.searchsorted(cuts, y[idx], side="right"), 0, n_strata - 1)
    return np.bincount(st, minlength=n_strata)


def test_biz_val_stratified_quantile_guarantees_heavy_tail_coverage() -> None:
    """biz_value: on a heavy-tail (log-normal) target, stratified-quantile
    sampling GUARANTEES per-quantile-bin coverage, whereas an unlucky uniform
    draw STARVES the leanest stratum (including the signal-carrying tail bins).

    Quantile strata partition ``y`` by rank, so stratified sampling does not
    raise mean tail coverage -- its win is variance reduction: every stratum
    receives ``sample_n // n_strata`` rows deterministically. We assert two
    measured properties:

    * Per-stratum count variance collapses near zero under stratified vs a
      large variance under random (measured ~19.5 -> ~0 at the shape below).
    * The WORST-case minimum per-stratum count is materially higher under
      stratified (measured ~20 guaranteed vs ~11 random-mean / 8 random-worst):
      a >= 1.4x worst-stratum-coverage floor.

    This is exactly the coverage M1 unlocks: with the default flipped to
    stratified, the heavy-tail ``mi_n_strata`` boost steers a draw that can no
    longer starve the upper quantile bins the MI estimate depends on.
    """
    from mlframe.training.composite.discovery.screening import _sample_indices

    n_total = 60_000
    sample_n = 600  # small relative to strata so random starvation shows.
    n_strata = 30  # the heavy-tail boost value (mi_n_strata_heavy_tail).
    expected = sample_n // n_strata

    rand_min_counts: list[int] = []
    strat_min_counts: list[int] = []
    rand_vars: list[float] = []
    strat_vars: list[float] = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = np.exp(rng.normal(loc=0.0, scale=2.5, size=n_total))
        idx_random = _sample_indices(
            n_total,
            sample_n,
            random_state=seed,
            strategy="random",
        )
        idx_strat = _sample_indices(
            n_total,
            sample_n,
            random_state=seed,
            strategy="stratified_quantile",
            y=y,
            n_strata=n_strata,
        )
        cr = _per_stratum_counts(idx_random, y, n_strata)
        cs = _per_stratum_counts(idx_strat, y, n_strata)
        rand_min_counts.append(int(cr.min()))
        strat_min_counts.append(int(cs.min()))
        rand_vars.append(float(cr.var()))
        strat_vars.append(float(cs.var()))

    mean_rand_var = float(np.mean(rand_vars))
    mean_strat_var = float(np.mean(strat_vars))
    mean_rand_min = float(np.mean(rand_min_counts))
    mean_strat_min = float(np.mean(strat_min_counts))

    # 1) Variance reduction: stratified is near-deterministic per stratum.
    assert mean_strat_var <= 0.5, f"stratified per-stratum variance {mean_strat_var:.3f} is not ~0; the strata are not driving the draw"
    assert mean_rand_var >= 5.0 * max(mean_strat_var, 1.0), (
        f"random per-stratum variance {mean_rand_var:.2f} should dwarf the stratified variance {mean_strat_var:.3f}"
    )

    # 2) Guaranteed minimum coverage: stratified gives every bin ~``expected``,
    #    random starves the leanest bin.
    assert mean_strat_min >= 0.9 * expected, (
        f"stratified worst-stratum count {mean_strat_min:.1f} fell below the guaranteed {expected}; the per-stratum quota is broken"
    )
    assert mean_strat_min >= 1.4 * mean_rand_min, (
        f"stratified worst-stratum coverage {mean_strat_min:.1f} is not >= 1.4x the random worst {mean_rand_min:.1f}; M1 tail-coverage win is gone"
    )


# ----------------------------------------------------------------------
# Shared discovery fixtures (P14 + D8)
# ----------------------------------------------------------------------


def _make_config(**overrides):
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    defaults = dict(
        enabled=True,
        base_candidates="auto",
        transforms=("diff", "linear_residual", "additive_residual", "ratio"),
        top_k_after_mi=32,
        top_m_after_tiny=8,
        mi_sample_n=2000,
        tiny_model_sample_n=2000,
        eps_mi_gain=-10.0,
        screening="mi",  # skip Phase B for deterministic, fast fits.
        random_state=42,
        require_beats_raw_baseline=False,
        multi_base_enabled=False,
        discovery_n_jobs=1,
        fail_on_no_gain="warn",
    )
    defaults.update(overrides)
    return CompositeTargetDiscoveryConfig(**defaults)


def _drift_dataset(n: int = 2400, seed: int = 7) -> pd.DataFrame:
    """``y = alpha(t)*base + noise`` with alpha drifting across the two train
    halves so the alpha-drift gate fires on the ``linear_residual`` spec."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=5.0, scale=2.0, size=n)
    alpha = np.where(np.arange(n) < n // 2, 1.0, 3.5)
    y = alpha * base + 0.3 * rng.normal(size=n)
    other = rng.normal(size=n)
    return pd.DataFrame({"base": base, "other": other, "y": y})


def _run(df: pd.DataFrame, config):
    from mlframe.training.composite import CompositeTargetDiscovery

    n = len(df)
    train_idx = np.arange(0, int(0.8 * n))
    val_idx = np.arange(int(0.8 * n), n)
    disc = CompositeTargetDiscovery(config)
    disc.fit(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        val_idx=val_idx,
    )
    return disc


# ----------------------------------------------------------------------
# P14: pooled-base fast path is bit-identical
# ----------------------------------------------------------------------


def test_p14_drift_and_collapse_pool_reuse_is_bit_identical() -> None:
    """P14 reuses ``self._auto_base_pool[base]`` (== ``base_full[train_idx]``)
    in the drift + collapse loops. Must be BIT-IDENTICAL: same kept specs and
    same alpha-drift z-scores as a re-extraction baseline.

    The pool stores float32 train-restricted arrays; re-extraction produces the
    same float32 values, so the only observable is "did we use the same rows".
    We assert the drift flags + kept-spec names are stable across two fits.
    """
    cfg = _make_config(detect_linear_residual_alpha_drift=True)
    df = _drift_dataset()

    disc_a = _run(df, cfg)
    disc_b = _run(df, _make_config(detect_linear_residual_alpha_drift=True))

    names_a = sorted(s.name for s in disc_a.specs_)
    names_b = sorted(s.name for s in disc_b.specs_)
    assert names_a == names_b

    flags_a = getattr(disc_a, "_alpha_drift_flags", {})
    flags_b = getattr(disc_b, "_alpha_drift_flags", {})
    assert set(flags_a) == set(flags_b)
    for name in flags_a:
        za = flags_a[name]["z_score"]
        zb = flags_b[name]["z_score"]
        # Bit-identical (same float32 inputs, same numpy ops) -> exact equality.
        assert za == zb, f"{name}: drift z-score diverged {za} != {zb}"

    # The drift gate must actually have fired on this dataset, otherwise the
    # bit-identity assertion above is vacuous.
    assert flags_a, "alpha-drift gate did not fire; test is not exercising P14"


def test_p14_pool_fast_path_matches_extraction_values() -> None:
    """Directly assert the pool holds ``base_full[train_idx]`` so the P14
    substitution (``pool[:half]`` for ``base_full[train_idx[:half]]`` etc.) is
    value-exact."""
    from mlframe.training.composite.discovery.screening import _extract_column_array

    cfg = _make_config()
    df = _drift_dataset()
    disc = _run(df, cfg)

    train_idx = disc.train_idx_
    pool = getattr(disc, "_auto_base_pool", {})
    assert pool, "auto-base pool empty; cannot validate the P14 fast path"
    for base, pooled in pool.items():
        full = _extract_column_array(df, base)[train_idx]
        np.testing.assert_array_equal(pooled, full)


# ----------------------------------------------------------------------
# D8: report() kept-flag reconciliation
# ----------------------------------------------------------------------


def test_d8_report_kept_flags_match_final_specs_after_topk_trim() -> None:
    """Regression for D8: with a small ``top_k_after_mi`` more specs clear the
    eps gate than survive the trim. ``report()`` must mark exactly the surviving
    specs as ``kept=True`` -- not every spec that cleared the eps gate.

    Pre-fix this FAILS: trimmed specs keep ``kept=True`` because the entry flag
    was frozen at the eps gate and never reconciled against ``specs_``.
    """
    cfg = _make_config(top_k_after_mi=1)
    df = _drift_dataset()
    disc = _run(df, cfg)

    final_names = {s.name for s in disc.specs_}
    report = disc.report()
    kept_rows = [r for r in report if r.get("kept")]
    kept_names = {r["name"] for r in kept_rows}

    assert kept_names == final_names, f"report kept-set {sorted(kept_names)} != final specs {sorted(final_names)} -- D8 reconciliation did not run"
    # And there must be MORE evaluated (non-reject) candidates than survivors,
    # otherwise the top-K trim never dropped anything and the test is vacuous.
    evaluated = [r for r in report if not r.get("rejected")]
    assert len(evaluated) > len(final_names), "top_k_after_mi=1 did not trim any spec; test is not exercising D8"
    # Dropped-but-evaluated rows carry a post-gate drop reason.
    for r in evaluated:
        if not r.get("kept"):
            assert r.get("reason"), f"dropped spec {r['name']} has no recorded drop reason"


def test_d8_report_kept_flags_reconciled_under_alpha_drift_reject() -> None:
    """When ``reject_on_alpha_drift=True`` drops a drifting ``linear_residual``
    spec, that spec's report row must read ``kept=False`` (not the frozen
    eps-gate True)."""
    cfg = _make_config(
        detect_linear_residual_alpha_drift=True,
        reject_on_alpha_drift=True,
        alpha_drift_z_threshold=1.0,  # low threshold so the drift gate bites.
    )
    df = _drift_dataset()
    disc = _run(df, cfg)

    final_names = {s.name for s in disc.specs_}
    report = disc.report()
    kept_names = {r["name"] for r in report if r.get("kept")}
    assert kept_names == final_names

    # No linear_residual spec on the drifting base should survive AND claim kept.
    for r in report:
        if r.get("transform_name") == "linear_residual" and r["name"] not in final_names:
            assert not r.get("kept"), f"dropped drifting spec {r['name']} still claims kept=True"
