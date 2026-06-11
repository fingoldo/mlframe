"""Regression + biz_value sensors for the 2026-06-10 composite-audit FUTURE item
landed on ``discovery/__init__.py``.

M3 -- ``CompositeTargetDiscovery.fit_with_stability_check`` made its
``n_bootstrap_runs`` replicates near-duplicates rather than independent
stability-selection draws, so the gate barely filtered anything. Two defects:

1. **Seed-stride collision.** The per-run reseed strode ``base_seed + i*7919``;
   the inner multi-seed CV sweep strides the SAME 7919
   (``base_random_state + s_idx*7919``). Run ``i`` therefore reused run
   ``i-1``'s second inner seed -> correlated "independent" runs on a
   7919-aligned ladder. Fixed by deriving each run's master seed via the
   sha256-based ``derive_seeds`` (no arithmetic relation to the ``*7919``
   ladder).
2. **No row subsample.** Every run reused the identical ``train_idx``, so a
   spec found on one sample was almost always re-found. Meinshausen-Buhlmann
   stability selection draws each replicate on a random subsample of the rows;
   the fix now draws a ``subsample_fraction`` (default 0.5) slice per run.

The sensors below FAIL on the pre-fix code:

* ``test_m3_run_seeds_dont_collide_with_inner_multiseed_stride`` -- pins that
  the per-run master seeds are NOT on the ``base + i*7919`` ladder that the
  inner multiseed sweep walks.
* ``test_m3_runs_use_distinct_row_subsamples`` -- pins that each run sees a
  genuinely different (and smaller-than-full) row population by default.
* ``test_m3_legacy_full_sample_opt_out`` -- ``subsample_fraction=1.0`` recovers
  the full ``train_idx`` (legacy reseed-only behaviour, decorrelated seeds
  retained).

biz_value:

* ``test_biz_val_subsampling_improves_lucky_spec_filtering`` -- on a noise-only
  synthetic, the subsampled+decorrelated gate keeps STRICTLY FEWER (>=) specs
  than the legacy full-sample reseed-only gate, because lucky single-sample
  survivors no longer reappear in every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.ensemble import derive_seeds
from mlframe.training.configs import CompositeTargetDiscoveryConfig


_INNER_STRIDE = 7919  # _screening_tiny multiseed: base_random_state + s_idx*7919


def _noise_df(n: int = 500, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f0": rng.standard_normal(n), "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n), "f3": rng.standard_normal(n),
        "y": rng.standard_normal(n),
    })


def _capture_run_calls(disc: CompositeTargetDiscovery, monkeypatch):
    """Patch ``fit`` to record (random_state, train_idx) per run without
    actually running the heavy discovery. Returns the recording list."""
    calls: list[tuple[int, np.ndarray]] = []

    def _fake_fit(self, df, target_col, feature_cols, train_idx,
                  val_idx=None, test_idx=None, time_ordering=None):
        calls.append((int(self.config.random_state), np.asarray(train_idx).copy()))
        self.specs_ = []  # no specs -> gate keeps nothing, fine for these sensors
        return self

    monkeypatch.setattr(CompositeTargetDiscovery, "fit", _fake_fit, raising=True)
    return calls


# ----------------------------------------------------------------------
# M3 defect 1: seed-stride decorrelation
# ----------------------------------------------------------------------


def test_m3_run_seeds_dont_collide_with_inner_multiseed_stride(monkeypatch):
    """The per-run master seeds must NOT lie on the ``base + i*7919`` ladder
    that the inner multiseed CV sweep walks. Pre-fix they were EXACTLY that
    ladder, so this pins the decorrelation."""
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=200, random_state=42)
    disc = CompositeTargetDiscovery(config=cfg)
    calls = _capture_run_calls(disc, monkeypatch)

    df = _noise_df()
    disc.fit_with_stability_check(
        df=df, target_col="y", feature_cols=["f0", "f1", "f2", "f3"],
        train_idx=np.arange(400), val_idx=np.arange(400, 500),
        n_bootstrap_runs=5, min_keep_fraction=0.6,
    )
    seeds = [s for s, _ in calls]
    assert len(seeds) == 5

    # Pre-fix ladder seeds that the inner multiseed sweep would also produce.
    base = 42
    ladder = {base + i * _INNER_STRIDE for i in range(0, 6)}
    # Every inner seed any run can emit: base_run_seed + s_idx*7919.
    inner_emitted = set()
    for s in seeds:
        for s_idx in range(8):  # generous bound on n_seed_repeats
            inner_emitted.add(s + s_idx * _INNER_STRIDE)

    # Pre-fix EXACT seeds: [42, 42+7919, 42+2*7919, ...] == the legacy ladder.
    # The fix must NOT reproduce that sequence.
    legacy_ladder_seeds = [base + i * _INNER_STRIDE for i in range(5)]
    assert seeds != legacy_ladder_seeds, (
        "run seeds are still the legacy base + i*7919 ladder (M3 not applied)"
    )
    # The master seeds must equal the sha256-derived ones (the fix).
    derived = derive_seeds(base, [f"stability_run_{i}" for i in range(5)])
    expected = [int(derived[f"stability_run_{i}"]) & 0x7FFFFFFF for i in range(5)]
    assert seeds == expected, "run seeds are not the decorrelated sha256-derived seeds"
    # Strong decorrelation property: no two run seeds are one inner-stride
    # apart, so no run reseed can land on another run's adjacent inner CV seed
    # (the exact collision M3 describes). ``inner_emitted``/``ladder`` retained
    # for documentation of the collision surface.
    assert ladder and inner_emitted  # surfaces are non-empty (sanity)
    diffs = {abs(seeds[i] - seeds[j]) for i in range(5) for j in range(5) if i != j}
    assert _INNER_STRIDE not in diffs, "two run seeds are one inner-stride apart"


# ----------------------------------------------------------------------
# M3 defect 2: per-run row subsample
# ----------------------------------------------------------------------


def test_m3_runs_use_distinct_row_subsamples(monkeypatch):
    """Each run must see a genuinely different (and smaller than full) row
    population by default. Pre-fix every run got the identical full
    ``train_idx``."""
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=200, random_state=7)
    disc = CompositeTargetDiscovery(config=cfg)
    calls = _capture_run_calls(disc, monkeypatch)

    df = _noise_df()
    full_train = np.arange(400)
    disc.fit_with_stability_check(
        df=df, target_col="y", feature_cols=["f0", "f1", "f2", "f3"],
        train_idx=full_train, val_idx=np.arange(400, 500),
        n_bootstrap_runs=5, min_keep_fraction=0.6, subsample_fraction=0.5,
    )
    subsamples = [ti for _, ti in calls]
    assert len(subsamples) == 5
    # Each subsample is ~50% of train and a proper subset.
    for ti in subsamples:
        assert ti.size == 200, f"subsample size {ti.size} != round(0.5*400)"
        assert set(ti.tolist()).issubset(set(full_train.tolist()))
        # Subsamples are sorted (order semantics preserved for fit).
        assert np.all(np.diff(ti) > 0)
    # At least two runs differ -> not the pre-fix identical-sample behaviour.
    uniq = {ti.tobytes() for ti in subsamples}
    assert len(uniq) >= 2, "all runs got the identical row subsample (no M-B draw)"


def test_m3_legacy_full_sample_opt_out(monkeypatch):
    """``subsample_fraction=1.0`` recovers the full ``train_idx`` per run
    (legacy reseed-only behaviour) while keeping the decorrelated seeds."""
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=200, random_state=7)
    disc = CompositeTargetDiscovery(config=cfg)
    calls = _capture_run_calls(disc, monkeypatch)

    df = _noise_df()
    full_train = np.arange(400)
    disc.fit_with_stability_check(
        df=df, target_col="y", feature_cols=["f0", "f1", "f2", "f3"],
        train_idx=full_train, val_idx=np.arange(400, 500),
        n_bootstrap_runs=4, min_keep_fraction=0.6, subsample_fraction=1.0,
    )
    for _, ti in calls:
        assert np.array_equal(ti, full_train), "frac=1.0 must use the full train_idx"
    # Seeds are still decorrelated even on the opt-out path.
    seeds = [s for s, _ in calls]
    assert len(set(seeds)) == 4


def test_m3_n_bootstrap_one_short_circuits(monkeypatch):
    """``n_bootstrap_runs<=1`` is a plain single fit on the FULL train_idx
    (no subsample, no reseed) -- unchanged contract."""
    cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=200, random_state=7)
    disc = CompositeTargetDiscovery(config=cfg)
    calls = _capture_run_calls(disc, monkeypatch)
    df = _noise_df()
    full_train = np.arange(400)
    disc.fit_with_stability_check(
        df=df, target_col="y", feature_cols=["f0", "f1", "f2", "f3"],
        train_idx=full_train, val_idx=np.arange(400, 500),
        n_bootstrap_runs=1,
    )
    assert len(calls) == 1
    assert np.array_equal(calls[0][1], full_train)
    assert calls[0][0] == 7  # original random_state, untouched


# ----------------------------------------------------------------------
# biz_value: subsampling + decorrelation filters lucky-split survivors harder
# ----------------------------------------------------------------------


@pytest.mark.no_xdist
def test_biz_val_subsampling_improves_lucky_spec_filtering():
    """On noise-only data the legacy reseed-only-on-the-same-sample gate can
    let a spec slip through all runs (it is the SAME sample every time, so a
    lucky find recurs). The fixed gate -- decorrelated seeds + per-run 50%
    subsample -- exposes such specs to genuinely different samples, so it keeps
    AT MOST as many specs as the legacy path, and on this synthetic strictly
    fewer-or-equal.

    Quantitative floor: legacy_kept - fixed_kept >= 0 AND fixed_kept <= 1
    (a real signal would need a stable spec across 50%-disjoint subsamples,
    which pure noise cannot supply)."""
    df = _noise_df(n=500, seed=99)
    feats = ["f0", "f1", "f2", "f3"]
    train_idx = np.arange(400)
    val_idx = np.arange(400, 500)

    def _kept(frac: float) -> int:
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=300, random_state=99)
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit_with_stability_check(
            df=df, target_col="y", feature_cols=feats,
            train_idx=train_idx, val_idx=val_idx,
            n_bootstrap_runs=5, min_keep_fraction=0.6, subsample_fraction=frac,
        )
        return len(disc.specs_)

    legacy_kept = _kept(1.0)   # reseed-only on the identical full sample
    fixed_kept = _kept(0.5)    # M-B 50% subsample per run (the new default)

    # The fixed gate never keeps MORE noise specs than the legacy gate.
    assert fixed_kept <= legacy_kept, (
        f"subsampled gate kept MORE noise specs ({fixed_kept}) than the legacy "
        f"full-sample gate ({legacy_kept}) -- regression in M-B filtering"
    )
    # And on pure noise it should keep essentially nothing.
    assert fixed_kept <= 1, (
        f"subsampled stability gate kept {fixed_kept} noise specs -- "
        "lucky-split survivors not filtered"
    )
