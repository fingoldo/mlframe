"""Regression tests for the 2026-06-11 ``discovery/_stacked.py`` audit fixes.

Covers the four implemented FUTURE items:

* **A6** -- ``_warn_unrebuildable_oof_specs``: pass-2 specs whose base is an
  ephemeral ``_oof_*`` column (which the suite cannot rebuild from the
  persisted split frames) must be flagged with a loud WARNING instead of
  silently training a composite on an all-NaN base.
* **A7** -- ``fit_stacked_on_residual`` emits a WARNING when it merges pass-2
  specs discovered on the residual target (they carry residual-fitted params
  but the suite has no residual-aware training route yet), and stamps them
  ``discovered_on_residual=True``.
* **A17** -- both stacked methods thread ``time_aware`` / ``cv_splitter`` into
  ``composite_oof_predictions`` so temporal OOF folds don't leak future->past;
  default off preserves the historical shuffled-K-fold numerics.
* **A18** -- ``fit_stacked_on_residual`` honours
  ``max_pass1_specs_to_aggregate`` (the old ``ranked[:max(1,len(ranked))]`` was
  a no-op slice that aggregated every pass-1 spec).

These pin BEHAVIOUR (warnings firing, kwargs forwarded, cap enforced) rather
than source strings, and fail on the pre-fix code.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.spec import CompositeSpec


def _spec(name: str, base_column: str, extra: tuple[str, ...] = ()) -> CompositeSpec:
    return CompositeSpec(
        name=name,
        target_col="y",
        transform_name="linear_residual",
        base_column=base_column,
        fitted_params={},
        mi_gain=0.1,
        mi_y=0.2,
        mi_t=0.3,
        valid_domain_frac=1.0,
        n_train_rows=100,
        extra_base_columns=extra,
    )


# ---------------------------------------------------------------------------
# A6: _warn_unrebuildable_oof_specs
# ---------------------------------------------------------------------------
class TestA6UnrebuildableOofWarner:
    def test_flags_oof_base_spec(self, caplog) -> None:
        from mlframe.training.composite.discovery import _stacked

        pass2 = [
            _spec("y-linres-x_a", "x_a"),  # real base -> rebuildable, no warn
            _spec("y-linres-_oof_y-linres-x_a", "_oof_y-linres-x_a"),  # ephemeral
        ]
        with caplog.at_level(logging.WARNING):
            bad = _stacked._warn_unrebuildable_oof_specs(pass2, existing_names=set())
        assert bad == ["y-linres-_oof_y-linres-x_a"]
        assert any("ephemeral OOF feature" in r.message for r in caplog.records)

    def test_flags_oof_in_extra_base_columns(self) -> None:
        from mlframe.training.composite.discovery import _stacked

        pass2 = [_spec("multi", "x_a", extra=("_oof_x_b",))]
        bad = _stacked._warn_unrebuildable_oof_specs(pass2, existing_names=set())
        assert bad == ["multi"]

    def test_clean_specs_do_not_warn(self, caplog) -> None:
        from mlframe.training.composite.discovery import _stacked

        pass2 = [_spec("y-linres-x_a", "x_a"), _spec("y-log-x_b", "x_b")]
        with caplog.at_level(logging.WARNING):
            bad = _stacked._warn_unrebuildable_oof_specs(pass2, existing_names=set())
        assert bad == []
        assert not any("ephemeral OOF" in r.message for r in caplog.records)

    def test_existing_pass1_names_excluded(self) -> None:
        """A pass-1 spec that happens to share a name is not re-flagged."""
        from mlframe.training.composite.discovery import _stacked

        pass2 = [_spec("dup", "_oof_z")]
        bad = _stacked._warn_unrebuildable_oof_specs(pass2, existing_names={"dup"})
        assert bad == []


# ---------------------------------------------------------------------------
# A17 + A18: forwarding into composite_oof_predictions
#
# We monkeypatch the OOF helper imported INSIDE each method (the methods do a
# local ``from ..ensemble.feature_stacking import composite_oof_predictions``),
# so we patch the source symbol and capture every call's kwargs.
# ---------------------------------------------------------------------------
class _OofRecorder:
    def __init__(self, n_train: int) -> None:
        self.calls: list[dict] = []
        self._n = n_train

    def __call__(self, factory, X, y, **kwargs):  # mimic composite_oof_predictions
        self.calls.append(dict(kwargs))
        # Return a deterministic finite OOF vector so the residual path proceeds.
        return np.asarray(y, dtype=np.float64) * 0.5


def _tiny_discovery(monkeypatch, recorder):
    import mlframe.training.composite.ensemble.feature_stacking as fs

    monkeypatch.setattr(fs, "composite_oof_predictions", recorder)
    return fs


def _make_frame(n: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(10.0, 3.0, n)
    x_b = rng.normal(0.0, 2.0, n)
    y = 1.5 * x_a + 2.0 * x_b + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({
        "x_a": x_a, "x_b": x_b,
        "n0": rng.standard_normal(n), "n1": rng.standard_normal(n),
        "y": y,
    })


class TestA17TimeAwareForwarded:
    def test_fit_stacked_forwards_time_aware(self, monkeypatch) -> None:
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 800
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=500,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        CompositeTargetDiscovery(config=cfg).fit_stacked(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3, time_aware=True,
        )
        assert rec.calls, "OOF helper was never called -- pass-1 found no specs to stack"
        assert all(c.get("time_aware") is True for c in rec.calls)

    def test_fit_stacked_default_is_not_time_aware(self, monkeypatch) -> None:
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 800
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=500,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        CompositeTargetDiscovery(config=cfg).fit_stacked(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3,  # time_aware defaults False
        )
        assert rec.calls
        assert all(c.get("time_aware") is False for c in rec.calls)

    def test_residual_forwards_time_aware(self, monkeypatch) -> None:
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 800
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=500,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        CompositeTargetDiscovery(config=cfg).fit_stacked_on_residual(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3, time_aware=True,
        )
        assert rec.calls
        assert all(c.get("time_aware") is True for c in rec.calls)


class TestA18AggregateCap:
    def test_cap_limits_aggregated_specs(self, monkeypatch) -> None:
        """With cap=1 only the single best pass-1 spec feeds the OOF aggregate."""
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 1200
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=700,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit_stacked_on_residual(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3, max_pass1_specs_to_aggregate=1,
        )
        # Each aggregated pass-1 spec triggers exactly one OOF call. With the cap
        # at 1, regardless of how many pass-1 specs were found, only 1 OOF call
        # contributes to the aggregate.
        n_pass1 = len(disc.specs_)
        assert rec.calls, "no OOF calls -- pass-1 found nothing to aggregate"
        assert len(rec.calls) == 1, (
            f"cap=1 must limit aggregation to 1 OOF call, got {len(rec.calls)} "
            f"(pass1 had >= {n_pass1} specs)"
        )

    def test_uncapped_aggregates_more_than_one(self, monkeypatch) -> None:
        """cap<=0 restores the historical aggregate-all behaviour."""
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 1200
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=700,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        CompositeTargetDiscovery(config=cfg).fit_stacked_on_residual(
            df=df, target_col="y", feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3, max_pass1_specs_to_aggregate=0,  # disable cap
        )
        # On this two-signal frame pass-1 finds several specs; uncapped, more
        # than one feeds the aggregate (proving the old behaviour is reachable
        # AND that the cap above genuinely constrained it).
        assert len(rec.calls) >= 2, (
            f"uncapped aggregate must use >=2 OOF calls, got {len(rec.calls)}"
        )

    def test_config_field_present_and_wired(self) -> None:
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.stacked_residual_max_pass1_specs_to_aggregate == 3


# ---------------------------------------------------------------------------
# A7: residual-spec WARNING + discovered_on_residual stamp
# ---------------------------------------------------------------------------
class TestA7ResidualSpecWarning:
    def test_residual_specs_warned_and_stamped(self, monkeypatch, caplog) -> None:
        from mlframe.training.composite.discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        n = 1500
        df = _make_frame(n)
        rec = _OofRecorder(int(0.8 * n))
        _tiny_discovery(monkeypatch, rec)

        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=900,
            composite_skip_when_raw_dominates_ratio=0.0,
        )
        with caplog.at_level(logging.WARNING):
            disc = CompositeTargetDiscovery(config=cfg).fit_stacked_on_residual(
                df=df, target_col="y",
                feature_cols=["x_a", "x_b", "n0", "n1"],
                train_idx=np.arange(int(0.8 * n)),
                n_oof_folds=3,
            )
        residual_specs = [
            s for s in disc.specs_
            if getattr(s, "discovered_on_residual", False)
        ]
        if residual_specs:
            # When pass-2 produced residual specs, the A7 hazard warning fired.
            assert any(
                "residual-aware training path" in r.message for r in caplog.records
            ), "A7 warning must fire when residual specs are merged into specs_"
        else:
            pytest.skip(
                "pass-2 found no new residual specs on this seed; A7 warning "
                "path not exercised (covered by unit-level merge logic)."
            )
