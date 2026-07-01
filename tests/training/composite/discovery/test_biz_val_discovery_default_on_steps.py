"""Biz-value: the three opt-in discovery steps are default-ON and actually FIRE.

``region_adaptive_enabled`` / ``interaction_base_discovery_enabled`` /
``auto_chain_discovery_enabled`` all default ``True`` in
:class:`CompositeTargetDiscoveryConfig`. Per the "corrective mechanisms
default-ON" rule a default-ON feature that is SILENTLY a no-op is a bug: every
user thinks they have it but the artefact never populates.

The pre-existing ``test_discovery_opt_in_steps.py`` only shape-checks the
artefacts (it iterates ``for ra in disc.region_adaptive_specs_`` etc., which
passes VACUOUSLY when the list is empty), so a regression that gates any step
off -- a default flipped back to ``False``, an early-return gate gone too
strict, a missing wire from ``fit`` -- would not be caught. These tests assert
the artefact is NON-EMPTY on a synthetic where the step must clearly win, so a
silent no-op FAILS the test.

Each test relies on the package DEFAULT for its flag (does NOT set it), so a
regression flipping the constructor default to ``False`` trips the assertion.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _fit(df, feats, **overrides):
    cfg = dict(
        enabled=True,
        base_candidates="auto",
        screening="mi",
        multi_base_enabled=False,
        mi_sample_n=None,
        random_state=overrides.pop("random_state", 0),
    )
    cfg.update(overrides)
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(**cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc.fit(df, "y", feats, np.arange(len(df)))
    return disc


def test_biz_val_region_adaptive_when_enabled_fires_nonempty():
    """region_adaptive WHEN EXPLICITLY ENABLED -> >=1 region_adaptive_spec on multi-base data.

    region_adaptive is a committed-but-rejected research prototype and now defaults OFF (it burned
    ~7.5 min on a prod run fitting specs that collapsed at deploy), so this test opts in explicitly to
    exercise the step's business value. Two informative bases ``a`` / ``b`` each survive the gate, so
    the per-distinct-base region-adaptive step must fit at least one spec when enabled.
    """
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(5, 2, n)
    b = rng.normal(3, 1.5, n)
    c = rng.normal(size=n)
    y = 0.9 * a + 0.6 * b + 0.5 * c + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": y})
    disc = _fit(df, ["a", "b", "c"], region_adaptive_enabled=True)
    assert disc.region_adaptive_specs_, (
        "region_adaptive (explicitly enabled) produced ZERO specs on multi-base data (silent no-op)"
    )
    for ra in disc.region_adaptive_specs_:
        assert ra.k >= 1
        assert len(ra.region_transforms) == len(ra.region_params)


def test_biz_val_interaction_base_default_on_fires_on_pure_interaction():
    """interaction_base default-ON -> >=1 synthetic on a pure ``y~a*b`` target.

    Marginals MI(y,a) ~ MI(y,b) ~ 0 but MI(y, a*b) is large, so the synergy gate
    MUST surface the ``a__mul__b`` synthetic. Empty interaction_bases_ here means
    the step silently no-oped (default off / pool <2 / gate broken) -> FAIL.
    """
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = a * b + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": y})
    disc = _fit(df, ["a", "b", "c"])
    assert disc.interaction_bases_, (
        "interaction-base default-ON surfaced ZERO synthetics on a pure "
        "interaction y=a*b target (silent no-op)"
    )
    assert len(disc.interaction_base_records_) == len(disc.interaction_bases_)
    for name, arr in disc.interaction_bases_.items():
        assert isinstance(name, str)
        assert np.asarray(arr).ndim == 1


def test_biz_val_auto_chain_default_on_appends_chain_specs():
    """auto_chain default-ON -> >=1 chain on a heavy cube-residual AR base.

    ``y = 0.9*base + z + 0.05*z**3`` leaves a heavy residual tail a
    residual x tail-unary chain compresses, so the auto-chain step must surface
    >=1 chain (auto_chains_ non-empty AND a chain_* spec appended to specs_).
    Empty here = silent no-op (default off / gate too strict) -> FAIL.
    """
    rng = np.random.default_rng(0)
    n = 2000
    base = rng.normal(10, 3, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = 0.5 * x1 - 0.4 * x2
    y = 0.9 * base + z + 0.05 * z ** 3 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    disc = _fit(df, ["base", "x1", "x2"])
    assert disc.auto_chains_, (
        "auto-chain default-ON surfaced ZERO chains on a heavy cube-residual "
        "target (silent no-op)"
    )
    chain_specs = [s for s in disc.specs_ if s.transform_name.startswith("chain_")]
    assert chain_specs, "auto_chains_ populated but no chain_* spec appended to specs_"
    from mlframe.training.composite.transforms import get_transform

    for s in chain_specs:
        assert s.n_train_rows > 0
        get_transform(s.transform_name)  # registered -> resolvable by name


def test_biz_val_default_config_opt_in_step_flags():
    """Opt-in discovery flag defaults: interaction_base / auto_chain ON; region_adaptive OFF.

    region_adaptive is a committed-but-rejected research prototype (heavy + collapses at deploy), so
    it defaults OFF; the other two carry test-confirmed business value and stay ON.
    """
    cfg = CompositeTargetDiscoveryConfig(enabled=True)
    assert cfg.region_adaptive_enabled is False
    assert cfg.interaction_base_discovery_enabled is True
    assert cfg.auto_chain_discovery_enabled is True
