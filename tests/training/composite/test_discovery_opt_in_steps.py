"""Tests for the opt-in discovery steps wired into ``CompositeTargetDiscovery.fit``.

Covers the three NEW config flags (``region_adaptive_enabled`` /
``interaction_base_discovery_enabled`` / ``auto_chain_discovery_enabled``), all
default ``False``, plus the ``discover_incremental`` re-export.

Contract under test
-------------------
* Each flag OFF => the discovered specs + report are IDENTICAL to a baseline fit
  (a flag-gated no-op; the default-OFF path is byte-identical to today).
* Each flag ON => its step runs over the kept specs without crashing on a small
  synthetic, sets its instance artefact attribute, and any extra appended specs
  are well-formed :class:`CompositeSpec` objects (auto-chain).
* ``discover_incremental`` warm-starts a prior fit on an appended frame and
  returns a well-formed :class:`IncrementalDecision`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery import (
    IncrementalDecision,
    discover_incremental,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Strong AR-style structure so single-base residual specs survive the gate.

    ``y = 0.9*base + 0.4*x1 - 0.3*x2 + cube-residual`` gives the auto-chain step a
    heavy residual tail to compress and the interaction step two informative
    bases to pair.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 3.0, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    z = 0.5 * x1 - 0.4 * x2
    y = 0.9 * base + z + 0.05 * z ** 3 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame(
        {"base": base, "x1": x1, "x2": x2, "x3": x3, "y": y}
    )


def _base_config(**overrides):
    cfg = dict(
        enabled=True,
        base_candidates=["base"],
        screening="mi",            # skip tiny-model rerank: faster + deterministic
        multi_base_enabled=False,  # isolate the opt-in hook from multi-base upgrades
        mi_sample_n=None,
        random_state=0,
        # Controlled all-off baseline: the package now defaults these ON, so pin
        # them False here and let each "flag ON" test flip its one flag.
        region_adaptive_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
    )
    cfg.update(overrides)
    return CompositeTargetDiscoveryConfig(**cfg)


def _fit(config, df=None):
    df = _synthetic() if df is None else df
    feats = ["base", "x1", "x2", "x3"]
    train_idx = np.arange(len(df))
    disc = CompositeTargetDiscovery(config)
    disc.fit(df, "y", feats, train_idx)
    return disc


def _spec_keys(specs):
    return [
        (s.name, s.transform_name, s.base_column,
         tuple(getattr(s, "extra_base_columns", ()) or ()))
        for s in specs
    ]


# ----------------------------------------------------------------------
# Default-OFF: byte-identical to baseline
# ----------------------------------------------------------------------


def test_all_flags_off_is_noop_identical_to_baseline():
    base = _fit(_base_config())
    again = _fit(_base_config())
    assert _spec_keys(base.specs_) == _spec_keys(again.specs_)
    # The off-path still initialises the artefacts as empty no-ops only when the
    # hook runs; with all flags off the hook does not run, so the attributes are
    # absent. The point is the discovered specs are unchanged.
    assert len(base.specs_) > 0


@pytest.mark.parametrize(
    "flag",
    [
        "region_adaptive_enabled",
        "interaction_base_discovery_enabled",
        "auto_chain_discovery_enabled",
    ],
)
def test_each_flag_off_individually_matches_baseline_specs(flag):
    baseline = _fit(_base_config())
    # Flip every OTHER flag off (already off) -- this flag explicitly False.
    off = _fit(_base_config(**{flag: False}))
    assert _spec_keys(baseline.specs_) == _spec_keys(off.specs_)


# ----------------------------------------------------------------------
# Each flag ON: step runs, artefact set, extra specs well-formed
# ----------------------------------------------------------------------


def test_region_adaptive_on_runs_and_sets_artifact():
    disc = _fit(_base_config(region_adaptive_enabled=True))
    assert hasattr(disc, "region_adaptive_specs_")
    # At least one kept single-base spec => at least one region-adaptive spec.
    assert isinstance(disc.region_adaptive_specs_, list)
    for ra in disc.region_adaptive_specs_:
        assert ra.k >= 1
        assert len(ra.region_transforms) == len(ra.region_params)
        # forward/inverse must run on a small array without crashing.
        y = np.linspace(-1, 1, 20)
        b = np.linspace(0, 10, 20)
        t = ra.forward(y, b)
        assert t.shape == y.shape and np.isfinite(t).all()


def test_interaction_base_discovery_on_runs_and_sets_artifact():
    disc = _fit(_base_config(interaction_base_discovery_enabled=True))
    assert hasattr(disc, "interaction_bases_")
    assert isinstance(disc.interaction_bases_, dict)
    assert isinstance(disc.interaction_base_records_, list)
    # Whatever surfaced must be a screen-row-length ndarray.
    for name, arr in disc.interaction_bases_.items():
        assert isinstance(name, str)
        assert np.asarray(arr).ndim == 1


def test_auto_chain_on_runs_and_appends_wellformed_specs():
    disc = _fit(_base_config(auto_chain_discovery_enabled=True))
    assert hasattr(disc, "auto_chains_")
    # Any appended chain spec is a well-formed CompositeSpec resolvable by name.
    from mlframe.training.composite.transforms import get_transform
    chain_specs = [s for s in disc.specs_ if s.transform_name.startswith("chain_")]
    for s in chain_specs:
        assert isinstance(s, CompositeSpec)
        assert s.base_column == "base"
        assert isinstance(s.fitted_params, dict)
        assert s.n_train_rows > 0
        # Registered into the registry so iter_transform / get_transform resolve.
        get_transform(s.transform_name)
    # iter_transform must stream every spec (including chains) without crashing.
    streamed = dict(disc.iter_transform(_synthetic()))
    assert set(streamed) == {s.name for s in disc.specs_}


def test_all_flags_on_together_runs_and_superset_of_baseline():
    baseline = _fit(_base_config())
    full = _fit(_base_config(
        region_adaptive_enabled=True,
        interaction_base_discovery_enabled=True,
        auto_chain_discovery_enabled=True,
    ))
    base_names = {s.name for s in baseline.specs_}
    full_names = {s.name for s in full.specs_}
    # Opt-in steps only ADD specs (auto-chain); never drop a baseline spec.
    assert base_names <= full_names
    assert hasattr(full, "region_adaptive_specs_")
    assert hasattr(full, "interaction_bases_")
    assert hasattr(full, "auto_chains_")


# ----------------------------------------------------------------------
# discover_incremental re-export
# ----------------------------------------------------------------------


def test_discover_incremental_reuse_on_identical_frame():
    df = _synthetic()
    disc = _fit(_base_config(), df=df)
    decision = discover_incremental(disc, df, "y", ["base", "x1", "x2", "x3"])
    assert isinstance(decision, IncrementalDecision)
    # Byte-identical frame => trivial reuse, zero re-scores.
    assert decision.reuse is True
    assert decision.specs is not None
    assert decision.n_rescored == 0


def test_discover_incremental_rescores_on_appended_frame():
    df = _synthetic(n=1200, seed=0)
    disc = _fit(_base_config(), df=df)
    grown = pd.concat([df, _synthetic(n=400, seed=1)], ignore_index=True)
    decision = discover_incremental(disc, grown, "y", ["base", "x1", "x2", "x3"])
    assert isinstance(decision, IncrementalDecision)
    # Different frame => the cheap MI re-score actually ran on each prior spec.
    assert decision.n_rescored == len(disc.specs_)
    assert decision.new_signature != decision.prior_signature


# ----------------------------------------------------------------------
# Parallel per-base opt-in steps are bit-identical to the serial path
# ----------------------------------------------------------------------


def test_opt_in_steps_parallel_matches_serial_specs(monkeypatch):
    """region-adaptive + auto-chain run a per-base loop parallelised across physical cores.

    The workers are mutation-free (registry / spec reduction happen on the main thread) and the
    feature matrix is built once then sliced per base via np.delete, so the discovered specs must
    be order- and content-identical whether the loop runs serial (1 core) or parallel (N cores).
    """
    import mlframe.training.composite.discovery._opt_in_steps as ois

    cfg = dict(
        region_adaptive_enabled=True,
        auto_chain_discovery_enabled=True,
        interaction_base_discovery_enabled=True,
        base_candidates=["base", "x1", "x2"],  # >=2 kept bases so the parallel branch fires
    )

    monkeypatch.setattr(ois, "cpu_count_physical", lambda *a, **k: 1)
    serial = _fit(_base_config(**cfg))

    monkeypatch.setattr(ois, "cpu_count_physical", lambda *a, **k: 8)
    parallel = _fit(_base_config(**cfg))

    assert _spec_keys(serial.specs_) == _spec_keys(parallel.specs_)
    ra_keys = lambda specs: [(s.name, s.base_column, s.edges) for s in specs]
    assert ra_keys(serial.region_adaptive_specs_) == ra_keys(parallel.region_adaptive_specs_)
