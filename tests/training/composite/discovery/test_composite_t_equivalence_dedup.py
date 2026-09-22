"""Redundant / degenerate composite specs must be dropped (or repaired) before the expensive per-spec training."""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.discovery._t_equivalence import find_equivalent_composite_specs
from mlframe.training.composite.transforms import get_transform
from mlframe.training.core._phase_composite_discovery_dedup import prune_equivalent_composite_specs


def _data(n=5000, seed=0):
    """Positive target ``y`` and a correlated base column."""
    rng = np.random.default_rng(seed)
    base = rng.normal(10, 3, n)
    y = 2.0 * base + rng.gamma(2.0, 2.0, n)
    return y, base


def test_affine_equivalents_are_dropped_and_distinct_specs_kept() -> None:
    """Specs whose T is an affine map of raw y or of a better-ranked spec are dropped; distinct specs are kept."""
    y, base = _data()
    t = {
        "diff": y - base,
        "addres_slope1": y - 1.0 * base - 3.7,  # additive residual whose fitted slope is 1 == diff up to a shift
        "linres_zero_slope": y - 1e-9 * base - 5.0,  # base contribution ~0 -> raw y
        "logY": np.log(y - y.min() + 1.0),  # monotone but NOT affine: genuinely different loss geometry
        "linres": y - 2.0 * base,
    }
    drops = find_equivalent_composite_specs(y, t, list(t))
    assert set(drops) == {"addres_slope1", "linres_zero_slope"}, drops
    assert "diff" in drops["addres_slope1"]
    assert "raw y" in drops["linres_zero_slope"]


def test_constant_t_is_dropped() -> None:
    """A spec whose T is constant is dropped."""
    y, _ = _data()
    drops = find_equivalent_composite_specs(y, {"const": np.full_like(y, 380.0)}, ["const"])
    assert "constant" in drops["const"]


def test_prune_removes_from_pending_and_metadata_and_logs(caplog) -> None:
    """Pruning removes the dropped specs from pending and metadata and logs each drop."""
    y, base = _data()
    specs = [types.SimpleNamespace(name=n, fitted_params={}) for n in ("diff", "addres", "linres")]
    t = {"diff": y - base, "addres": y - base + 1.0, "linres": y - 2.0 * base}
    pending = [{"tt": "regression", "name": n, "values": v, "gain": g} for (n, v), g in zip(t.items(), (0.2, 0.1, 0.3))]
    metadata = {"composite_target_specs": {"regression": {"y": [{"name": n, "fitted_params": {}} for n in t]}}}
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_composite_discovery"):
        drops = prune_equivalent_composite_specs(
            specs=specs, t_by_name=t, y_full=y, train_idx=np.arange(4000), pending=pending, metadata=metadata,
            target_type="regression", target_name="y",
        )
    assert set(drops) == {"addres"}  # lower gain than its twin 'diff'
    assert [p["name"] for p in pending] == ["diff", "linres"]
    assert [s["name"] for s in metadata["composite_target_specs"]["regression"]["y"]] == ["diff", "linres"]
    assert any("dropped redundant composite 'addres'" in r.getMessage() for r in caplog.records)
    # Every surviving spec carries the exact T-train envelope for the end-of-target wrapper.
    assert "t_train_envelope_low" in specs[0].fitted_params
    assert "t_train_envelope_high" in metadata["composite_target_specs"]["regression"]["y"][0]["fitted_params"]


def test_quantile_residual_zero_inflated_target_has_sane_t_scale() -> None:
    """The quantile-residual transform on a zero-inflated target yields a finite, sensibly scaled T."""
    rng = np.random.default_rng(1)
    n = 4000
    base = rng.normal(0, 1, n)
    y = np.where(rng.random(n) < 0.7, 0.0, rng.gamma(1.0, 20.0, n))  # >50% zeros -> IQR == 0 in every bin
    tr = get_transform("quantile_residual")
    params = tr.fit(y, base)
    t = np.asarray(tr.forward(y, base, params))
    # A 1e-6 IQR floor made T ~ y * 1e6; the T column must stay on a scale comparable to y's spread.
    assert np.max(np.abs(t)) < 1e3, float(np.max(np.abs(t)))
    np.testing.assert_allclose(tr.inverse(t, base, params), y, atol=1e-8)


def test_from_fitted_inner_uses_discovery_t_envelope() -> None:
    """from_fitted_inner reuses the T envelope recorded at discovery time."""
    rng = np.random.default_rng(2)
    n = 500
    base = rng.normal(0, 1, n)
    y = 5.0 + base + rng.normal(0, 0.1, n)
    tr = get_transform("quantile_residual")
    params = dict(tr.fit(y, base))
    t = np.asarray(tr.forward(y, base, params))
    params["t_train_envelope_low"], params["t_train_envelope_high"] = float(t.min()) - 1.0, float(t.max()) + 1.0
    inner = DummyRegressor().fit(np.zeros((n, 1)), t)
    w = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner, transform_name="quantile_residual", base_column="b", transform_fitted_params=params, y_train=y,
    )
    assert w.fitted_params_["t_clip_low"] == pytest.approx(params["t_train_envelope_low"])
    assert w.fitted_params_["t_clip_high"] == pytest.approx(params["t_train_envelope_high"])


def test_a_spec_dropped_before_training_leaves_the_metadata_spec_list():
    """A capped (or below-floor) spec is removed from ``composite_target_specs`` and recorded in the failures with the reason."""
    from mlframe.training.core._phase_composite_discovery_dedup import forget_untrained_specs

    md = {"composite_target_specs": {"regression": {"y": [{"name": "y-linres-b"}, {"name": "y-diff-b"}]}}}
    forget_untrained_specs(md, [{"tt": "regression", "target": "y", "name": "y-diff-b"}], "global cap max_total_composite_targets=1")
    assert [s["name"] for s in md["composite_target_specs"]["regression"]["y"]] == ["y-linres-b"]
    assert md["composite_target_failures"]["regression"]["y"] == [
        {"name": "y-diff-b", "kept": False, "rejected": True, "reason": "global cap max_total_composite_targets=1"}
    ]
