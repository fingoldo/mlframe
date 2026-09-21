"""A spec a gate cannot evaluate is rejected, not waved through.

Both collapse gates kept a spec whenever its forward, fit or inverse raised on the gate's rows (logged at DEBUG), and the
tiny rerank compared only finite scores against the raw baseline, so a spec whose CV failed in every family sorted last and
still shipped with fewer than top_m survivors. The forward or inverse that raised on those rows raises again at predict.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_rmse_gate import apply_honest_rmse_gate
from mlframe.training.composite.discovery._tiny_rerank import _reject_unscored_specs
from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate
from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY

from .test_biz_val_discovery_yscale_holdout_gate import _grouped_frame, _make_gate_ctx, _spec
from .test_biz_val_honest_rmse_gate import _additive_dominant_frame, _dummy_spec, _mi_config


@pytest.fixture
def raising_inverse(monkeypatch):
    """``linear_residual`` whose inverse raises, as a transform does on rows outside what it can invert."""

    def _raise(*_a, **_k):
        raise ValueError("cannot invert these rows")

    monkeypatch.setitem(_TRANSFORMS_REGISTRY, "linear_residual", dataclasses.replace(_TRANSFORMS_REGISTRY["linear_residual"], inverse=_raise))


def _stages(disc, stage: str) -> list[dict]:
    """Ledger rows of one stage."""
    return [r for r in disc.rejection_ledger if r["stage"] == stage]


def test_the_yscale_gate_rejects_a_spec_whose_inverse_raises(raising_inverse):
    """A raising inverse on the unseen-group rows is a rejection with a ledger reason, not a survivor."""
    df, groups, y = _grouped_frame()
    disc = _make_gate_ctx(groups)
    out = apply_yscale_holdout_gate(disc, df, "y", [_spec("y-linres-base-unit", alpha=1.0)], ["base", "x1", "x2"], np.arange(len(df)), y)
    assert out == []
    rows = _stages(disc, "yscale_holdout")
    assert rows and "raised ValueError" in rows[0]["reason"]


def test_the_honest_rmse_gate_rejects_a_spec_whose_inverse_raises(raising_inverse):
    """The honest gate rejects the same spec instead of letting it keep its MI verdict."""
    df, y = _additive_dominant_frame(n=1500, seed=7)
    disc = CompositeTargetDiscovery(_mi_config(tiny_model_n_estimators=25))
    perm = np.random.default_rng(0).permutation(len(df))
    out = apply_honest_rmse_gate(disc, df, "y", [_dummy_spec(alpha=1.0)], ["base", "x0", "x1"], np.sort(perm[:1200]), np.sort(perm[1200:]), y)
    assert out == []
    rows = _stages(disc, "honest_rmse")
    assert rows and "raised ValueError" in rows[0]["reason"]


def test_a_spec_over_an_unregistered_transform_is_rejected_by_both_gates():
    """A transform missing from the registry cannot be served, so neither gate may keep its spec."""
    df, groups, y = _grouped_frame()
    ghost = dataclasses.replace(_spec("y-ghost-base", alpha=1.0), transform_name="no_such_transform")
    disc = _make_gate_ctx(groups)
    assert apply_yscale_holdout_gate(disc, df, "y", [ghost], ["base", "x1", "x2"], np.arange(len(df)), y) == []
    df2, y2 = _additive_dominant_frame(n=1500, seed=7)
    disc2 = CompositeTargetDiscovery(_mi_config(tiny_model_n_estimators=25))
    ghost2 = dataclasses.replace(_dummy_spec(alpha=1.0), transform_name="no_such_transform")
    perm = np.random.default_rng(0).permutation(len(df2))
    assert apply_honest_rmse_gate(disc2, df2, "y", [ghost2], ["base", "x0", "x1"], np.sort(perm[:1200]), np.sort(perm[1200:]), y2) == []
    assert _stages(disc, "yscale_holdout") and _stages(disc2, "honest_rmse")


def test_a_healthy_spec_still_survives_both_gates():
    """Control: the stricter error handling must not reject a spec the gates can evaluate."""
    df, groups, y = _grouped_frame()
    disc = _make_gate_ctx(groups)
    assert len(apply_yscale_holdout_gate(disc, df, "y", [_spec("y-linres-base-unit", alpha=1.0)], ["base", "x1", "x2"], np.arange(len(df)), y)) == 1


def test_an_unscored_spec_is_rejected_before_the_raw_baseline_gate():
    """A spec whose tiny CV failed in every family (NaN or inf aggregate) is dropped with a ledger entry."""
    disc = CompositeTargetDiscovery(_mi_config())
    specs = [_spec(f"s{i}", alpha=1.0) for i in range(3)]
    kept, scores = _reject_unscored_specs(disc, specs, [1.5, float("nan"), float("inf")])
    assert [s.name for s in kept] == ["s0"] and scores == [1.5]
    rows = _stages(disc, "tiny_rerank_threshold")
    assert sorted(r["spec_name"] for r in rows) == ["s1", "s2"]
    assert all(math.isfinite(sc) for sc in scores)
