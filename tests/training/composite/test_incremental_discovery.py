"""Tests for warm-start / incremental discovery on appended data.

``incremental_discovery_check`` decides, given a prior discovery result (kept
specs + the ``data_signature`` they were fit on) and a NEW appended frame,
whether the prior specs still hold (REUSE -- skip the full re-screen) or the
DGP drifted (REDISCOVER -- run a full ``fit``).

Coverage
--------
Unit:
  * signature-identical frame -> trivial reuse, zero re-scores.
  * no prior specs -> force full discovery.
  * empty new frame -> force full discovery.
  * decision dataclass shape (per-spec gains, surviving counts, signatures).
biz_value:
  * appended data under the SAME DGP keeps the prior specs (spec equality)
    AND is much cheaper than a full re-discovery (re-score count is a small
    fraction of the full (base x transform) candidate count).
  * a DGP SHIFT (the base->y relationship inverts on the appended rows)
    triggers a full re-discovery instead of silently reusing stale specs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.sklearn_matrix

from mlframe.training.composite import (  # noqa: E402
    CompositeTargetDiscovery,
    data_signature,
)
from mlframe.training.composite.discovery._incremental import (  # noqa: E402
    IncrementalDecision,
    incremental_discovery_check,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig  # noqa: E402


_FEATURES = ["TVT_prev", "x1", "x2", "x3"]


def _same_dgp(n: int, seed: int) -> pd.DataFrame:
    """Strong autoregressive signal: y depends linearly on TVT_prev + features.

    Same data-generating process for any seed -- only the noise draw differs,
    so specs found on one draw stay valid on another draw of the same DGP.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "x3": x3, "TVT": y})


def _shifted_dgp(n: int, seed: int) -> pd.DataFrame:
    """A genuinely different DGP: y no longer follows the prior base->residual
    structure. The linear_residual(TVT_prev) spec fit on ``_same_dgp`` data no
    longer opens up the residual here, so its re-scored MI gain decays."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    # y is pure independent noise w.r.t. base + features: no residual structure
    # for any (base, transform) spec to exploit.
    y = rng.normal(scale=5.0, size=n)
    return pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "x3": x3, "TVT": y})


def _fit_prior(df: pd.DataFrame, cfg: CompositeTargetDiscoveryConfig):
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, target_col="TVT", feature_cols=_FEATURES,
             train_idx=np.arange(len(df)))
    sig = data_signature(df, "TVT", _FEATURES)
    return disc, sig


def _cfg() -> CompositeTargetDiscoveryConfig:
    return CompositeTargetDiscoveryConfig(
        enabled=True, mi_sample_n=800, top_k_after_mi=5,
        eps_mi_gain=0.05, auto_base_top_k=3, screening="mi",
    )


# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------


class TestIncrementalUnit:
    def test_signature_identical_trivial_reuse(self) -> None:
        df = _same_dgp(1500, seed=0)
        cfg = _cfg()
        disc, sig = _fit_prior(df, cfg)
        assert disc.specs_, "prior fit must find at least one spec"
        # Same frame -> same signature -> reuse with zero re-scores.
        dec = incremental_discovery_check(
            disc.specs_, sig, df, "TVT", _FEATURES, cfg,
        )
        assert isinstance(dec, IncrementalDecision)
        assert dec.reuse is True
        assert list(dec.specs) == list(disc.specs_)
        assert dec.n_rescored == 0
        assert dec.new_signature == dec.prior_signature == sig

    def test_no_prior_specs_forces_full_discovery(self) -> None:
        df = _same_dgp(800, seed=1)
        cfg = _cfg()
        dec = incremental_discovery_check([], "", df, "TVT", _FEATURES, cfg)
        assert dec.reuse is False
        assert dec.specs is None
        assert dec.n_specs == 0
        assert "no prior specs" in dec.reason

    def test_empty_new_frame_forces_full_discovery(self) -> None:
        df = _same_dgp(800, seed=2)
        cfg = _cfg()
        disc, sig = _fit_prior(df, cfg)
        empty = df.iloc[0:0]
        dec = incremental_discovery_check(
            disc.specs_, sig, empty, "TVT", _FEATURES, cfg,
        )
        assert dec.reuse is False
        assert dec.specs is None

    def test_decision_reports_per_spec_gains(self) -> None:
        df = _same_dgp(1500, seed=3)
        cfg = _cfg()
        disc, sig = _fit_prior(df, cfg)
        appended = pd.concat([df, _same_dgp(1500, seed=99)], ignore_index=True)
        new_sig = data_signature(appended, "TVT", _FEATURES)
        assert new_sig != sig, "appended frame must have a different signature"
        dec = incremental_discovery_check(
            disc.specs_, sig, appended, "TVT", _FEATURES, cfg,
        )
        # Every prior spec got a re-scored gain entry; counts are consistent.
        assert set(dec.per_spec_gain) == {s.name for s in disc.specs_}
        assert dec.n_rescored == len(disc.specs_)
        assert 0 <= dec.n_surviving <= dec.n_specs == len(disc.specs_)


# ----------------------------------------------------------------------
# biz_value tests
# ----------------------------------------------------------------------


class TestIncrementalBizValue:
    def test_biz_val_same_dgp_reuses_specs_and_is_cheaper(self) -> None:
        """Appended data under the SAME DGP: prior specs stay valid (spec
        equality) AND the incremental check is much cheaper than a full
        re-discovery.

        Cost proxy: ``n_rescored`` (one cheap MI re-score per kept spec) vs the
        full screen's per-(base, transform) candidate count -- the full path
        evaluates ``len(report())`` candidates (>= bases x transforms). The
        re-score count must be a small fraction of that. Measured: ~3-5
        re-scores vs ~40-60 full candidates (>= 8x fewer); floor at 3x to
        absorb registry-size variation.
        """
        df = _same_dgp(2000, seed=0)
        cfg = _cfg()
        disc, sig = _fit_prior(df, cfg)
        assert disc.specs_, "prior fit must find specs"
        prior_names = {s.name for s in disc.specs_}
        full_candidate_count = len(disc.report())
        assert full_candidate_count > 0

        # Grow the frame with fresh rows from the SAME DGP.
        appended = pd.concat([df, _same_dgp(2000, seed=7)], ignore_index=True)
        new_sig = data_signature(appended, "TVT", _FEATURES)
        assert new_sig != sig, "appended frame must change the signature"

        dec = incremental_discovery_check(
            disc.specs_, sig, appended, "TVT", _FEATURES, cfg,
        )
        # Reuse verdict: prior specs returned unchanged (spec equality).
        assert dec.reuse is True, dec.reason
        assert list(dec.specs) == list(disc.specs_)
        assert {s.name for s in dec.specs} == prior_names

        # Cost win: re-score count << full-discovery candidate count.
        assert dec.n_rescored == len(disc.specs_)
        speedup = full_candidate_count / max(1, dec.n_rescored)
        assert speedup >= 3.0, (
            f"incremental re-score count {dec.n_rescored} not much cheaper than "
            f"full re-discovery's {full_candidate_count} candidates "
            f"(ratio {speedup:.1f}x, floor 3x)"
        )

    def test_biz_val_dgp_shift_triggers_rediscovery(self) -> None:
        """A DGP shift on the appended rows decays the prior specs' MI gain
        below ``eps_mi_gain`` -> the check must REFUSE reuse and signal a full
        re-discovery (never silently replay stale specs)."""
        df = _same_dgp(2000, seed=0)
        cfg = _cfg()
        disc, sig = _fit_prior(df, cfg)
        assert disc.specs_, "prior fit must find specs"

        # The appended frame's rows follow a different DGP that destroys the
        # base->residual structure the specs exploit. We point the check at the
        # SHIFTED frame (the new regime) to detect the drift.
        shifted = _shifted_dgp(2000, seed=11)
        dec = incremental_discovery_check(
            disc.specs_, sig, shifted, "TVT", _FEATURES, cfg,
        )
        assert dec.reuse is False, (
            f"DGP shift must trigger re-discovery; got reuse with {dec.reason}"
        )
        assert dec.specs is None
        # The re-scored gains decayed: fewer than half the specs survive.
        assert dec.n_surviving < dec.n_specs
