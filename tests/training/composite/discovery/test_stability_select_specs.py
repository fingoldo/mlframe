"""Unit + biz_value tests for stability_select_specs (spec stability selection)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._stability import (
    StabilityResult,
    stability_select_specs,
)


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDiscovery:
    """Deterministic fake: ``genuine`` always selected; noise specs selected
    only when the subsample's first row index is even (~half the replicates) so
    their selection frequency lands well below the threshold."""

    def __init__(self) -> None:
        self.specs_ = []

    def fit(self, df, target, feature_cols, train_idx):  # noqa: D401
        names = ["genuine"]
        # Noise inclusion keyed on a property of the subsample -> varies across
        # replicates but is deterministic given the subsample. The subsample
        # sum's parity / mod is effectively a coin flip across random draws, so
        # each noise spec lands near ~50% selection -- below the 0.6 threshold.
        s = int(np.asarray(train_idx).sum()) % (2 ** 32)
        coin = np.random.default_rng(s)
        if coin.random() < 0.4:
            names.append("noise_a")
        if coin.random() < 0.4:
            names.append("noise_b")
        self.specs_ = [_Spec(n) for n in names]
        return self


def _df_and_idx(n: int = 200):
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.standard_normal(n), "x": rng.standard_normal(n)})
    return df, np.arange(n)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_frequencies_in_unit_interval_and_genuine_is_one():
    df, idx = _df_and_idx()
    res = stability_select_specs(
        _FakeDiscovery, df, "y", ["x"], idx,
        n_replicates=8, subsample_frac=0.5, freq_threshold=0.6,
    )
    assert isinstance(res, StabilityResult)
    assert res.frequencies
    for f in res.frequencies.values():
        assert 0.0 <= f <= 1.0
    # The genuine spec is selected in every replicate.
    assert res.frequencies["genuine"] == 1.0


def test_stable_subset_is_high_frequency_only():
    df, idx = _df_and_idx()
    res = stability_select_specs(
        _FakeDiscovery, df, "y", ["x"], idx,
        n_replicates=8, subsample_frac=0.5, freq_threshold=0.6,
    )
    assert res.stable_specs == ["genuine"]
    # Noise specs exist in the frequency table but did not clear the threshold.
    for noise in ("noise_a", "noise_b"):
        assert res.frequencies.get(noise, 0.0) < res.freq_threshold
    # Representative spec objects round-trip for the stable subset.
    assert [s.name for s in res.stable_spec_objects] == ["genuine"]


def test_deterministic_with_fixed_seed():
    df, idx = _df_and_idx()
    kw = dict(n_replicates=6, subsample_frac=0.5, freq_threshold=0.6, random_state=42)
    r1 = stability_select_specs(_FakeDiscovery, df, "y", ["x"], idx, **kw)
    r2 = stability_select_specs(_FakeDiscovery, df, "y", ["x"], idx, **kw)
    assert r1.frequencies == r2.frequencies
    assert r1.stable_specs == r2.stable_specs


def test_different_seed_changes_subsamples_not_genuine():
    df, idx = _df_and_idx()
    r1 = stability_select_specs(_FakeDiscovery, df, "y", ["x"], idx, random_state=1)
    r2 = stability_select_specs(_FakeDiscovery, df, "y", ["x"], idx, random_state=2)
    # Genuine survives regardless of seed; noise frequencies may differ.
    assert "genuine" in r1.stable_specs and "genuine" in r2.stable_specs


def test_n_replicates_respected():
    df, idx = _df_and_idx()
    res = stability_select_specs(
        _FakeDiscovery, df, "y", ["x"], idx, n_replicates=5,
    )
    assert res.n_replicates == 5
    assert res.n_successful == 5
    # All frequencies are multiples of 1/5.
    for f in res.frequencies.values():
        assert abs(f * 5 - round(f * 5)) < 1e-9


def test_failed_replicate_excluded_from_denominator():
    class _Flaky:
        calls = {"n": 0}

        def fit(self, df, target, feature_cols, train_idx):
            _Flaky.calls["n"] += 1
            if _Flaky.calls["n"] == 1:
                raise ValueError("boom")
            self.specs_ = [_Spec("genuine")]
            return self

    df, idx = _df_and_idx()
    res = stability_select_specs(_Flaky, df, "y", ["x"], idx, n_replicates=4)
    assert res.n_replicates == 4
    assert res.n_successful == 3  # one raised
    assert res.frequencies["genuine"] == 1.0  # 3/3 successful


def test_duplicate_spec_within_run_counted_once():
    class _Dup:
        def fit(self, df, target, feature_cols, train_idx):
            self.specs_ = [_Spec("g"), _Spec("g")]
            return self

    df, idx = _df_and_idx()
    res = stability_select_specs(_Dup, df, "y", ["x"], idx, n_replicates=3)
    assert res.frequencies["g"] == 1.0  # never exceeds 1.0


def test_subsample_is_index_subset_no_frame_copy():
    seen = {}

    class _Spy:
        def fit(self, df, target, feature_cols, train_idx):
            seen["idx"] = np.asarray(train_idx)
            seen["df_is"] = df
            self.specs_ = [_Spec("g")]
            return self

    df, idx = _df_and_idx(n=100)
    stability_select_specs(_Spy, df, "y", ["x"], idx, subsample_frac=0.5)
    assert seen["df_is"] is df  # same frame object, no copy
    assert seen["idx"].size == 50  # half the rows
    assert set(seen["idx"]).issubset(set(idx))


# ---------------------------------------------------------------------------
# biz_value: genuine strong spec wins, noise specs dropped (real discovery)
# ---------------------------------------------------------------------------

def test_biz_val_stability_keeps_genuine_drops_noise():
    """On data with ONE genuine composite base (a near-copy of y whose residual
    ``y - base`` still carries LEARNABLE signal from another feature) plus pure-
    noise features, the genuine composite spec attains a high selection frequency
    across subsample replicates while specs built on noise features fall below the
    threshold and are dropped. Because the residual is genuinely learnable the
    near-copy increment-learnability precheck EXEMPTS the base from the near-copy-
    of-y drop, so the genuine composite legitimately survives. Floor 0.8 on the
    genuine frequency; measured ~1.0."""
    pd = pytest.importorskip("pandas")
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(7)
    n = 4_000
    base = np.cumsum(rng.standard_normal(n)).astype(np.float64)
    # GENUINE composite: y = base + 3*x_signal + noise, so the residual y - base = 3*x_signal + noise
    # has REAL learnable signal from ``x_signal`` (a feature) -- the precheck detects it and keeps the base.
    x_signal = rng.standard_normal(n).astype(np.float64)
    y = base + 3.0 * x_signal + rng.standard_normal(n) * 0.2
    feats = {f"noise{j}": rng.standard_normal(n) for j in range(6)}
    feats["x_signal"] = x_signal
    feats["genuine_base"] = base
    df = pd.DataFrame({"y": y, **feats})
    feature_cols = [c for c in df.columns if c != "y"]
    train_idx = np.arange(n)

    def factory():
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, screening="mi", mi_estimator="bin",
            base_candidates="auto", transforms=("diff", "linear_residual"),
        )
        return CompositeTargetDiscovery(config=cfg)

    res = stability_select_specs(
        factory, df, "y", feature_cols, train_idx,
        n_replicates=5, subsample_frac=0.5, freq_threshold=0.6,
    )

    assert res.n_successful >= 4, "discovery should run on most replicates"
    # At least one stable spec built on the genuine base survives.
    genuine_freqs = [
        f for name, f in res.frequencies.items() if "genuine_base" in name
    ]
    assert genuine_freqs, f"no genuine-base spec discovered; got {res.frequencies}"
    assert max(genuine_freqs) >= 0.8, (
        f"genuine-base spec frequency {max(genuine_freqs):.2f} below 0.8 floor"
    )
    # Every stable spec references the genuine base, never a pure-noise column.
    for name in res.stable_specs:
        assert not any(f"noise{j}" in name for j in range(6)), (
            f"noise-based spec {name!r} should not be stable; freqs={res.frequencies}"
        )
