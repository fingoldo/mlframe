"""Which targets get a HurdleRegressor, and that an injected model only trains on the targets it was chosen for."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.composite._hurdle_dispatch import (
    ZERO_INFLATION_FRACTION_THRESHOLD,
    maybe_inject_hurdle_for_zero_inflated,
    zero_inflated_atom,
    zero_inflated_regression_targets,
)
from mlframe.training.core._setup_helpers import _entry_not_for_target


def _zero_inflated(n=2000, zero_frac=0.7, seed=0):
    """Mostly-zero amount with a lognormal magnitude on the event rows."""
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(3.0, 1.0, n))
    y[rng.random(n) < zero_frac] = 0.0
    return y


class TestZeroInflatedAtom:
    """The atom must be modal, heavy enough, at the minimum, and not the whole target."""

    def test_zero_inflated_amount_is_detected(self):
        """The canonical shape: 70% exact zeros below a positive magnitude."""
        assert zero_inflated_atom(_zero_inflated()) == 0.0

    def test_atom_below_threshold_is_not_detected(self):
        """A modest share of zeros is an ordinary target."""
        assert zero_inflated_atom(_zero_inflated(zero_frac=ZERO_INFLATION_FRACTION_THRESHOLD - 0.2)) is None

    def test_interior_atom_is_not_a_hurdle(self):
        """A point mass in the middle of the range is not "no event": a hurdle's atom is the floor."""
        y = _zero_inflated()
        y[y == 0.0] = 5.0
        y[:50] = -1.0
        assert zero_inflated_atom(y) is None

    def test_nonzero_floor_is_reported_as_the_atom(self):
        """The floor need not be 0 (an amount measured from a baseline): its value is returned for zero_value."""
        assert zero_inflated_atom(_zero_inflated() + 10.0) == 10.0

    def test_constant_target_is_not_a_hurdle(self):
        """No magnitude to model."""
        assert zero_inflated_atom(np.zeros(500)) is None

    def test_continuous_target_is_not_detected(self):
        """No exact atom at all."""
        assert zero_inflated_atom(np.random.default_rng(0).normal(size=2000)) is None

    def test_too_few_rows_is_undecidable(self):
        """Below the point-mass gate's row floor the modal share is noise."""
        assert zero_inflated_atom(_zero_inflated(n=50)) is None


def test_only_zero_inflated_regression_targets_are_selected():
    """Classification targets and ordinary regression targets are not candidates; train rows decide."""
    from mlframe.training._configs_base import TargetTypes

    rng = np.random.default_rng(1)
    targets = {
        TargetTypes.REGRESSION: {"charge": _zero_inflated(), "price": rng.normal(size=2000)},
        TargetTypes.BINARY_CLASSIFICATION: {"churn": (rng.random(2000) < 0.2).astype(float)},
    }
    assert zero_inflated_regression_targets(targets, np.arange(1500)) == {0.0: ["charge"]}


class TestTargetRestriction:
    """The per-target loop trains every entry on every target unless the entry says otherwise."""

    def test_unrestricted_entries_train_everywhere(self):
        """User-requested models (string tags, plain estimators) are never skipped by this rule."""
        assert not _entry_not_for_target("cb", "anything")
        assert not _entry_not_for_target(object(), "anything")

    @pytest.mark.parametrize("wrap", [lambda e: e, lambda e: ("hurdle", e)])
    def test_restricted_entry_skips_other_targets(self, wrap):
        """Both an estimator entry and a ``(name, estimator)`` entry honour the restriction."""
        est = SimpleNamespace(_mlframe_only_targets=frozenset({"charge"}))
        assert not _entry_not_for_target(wrap(est), "charge")
        assert _entry_not_for_target(wrap(est), "churn")


def test_injection_adds_one_restricted_hurdle_and_registers_it():
    """The injected entry is a HurdleRegressor at the detected atom, limited to its targets, and trainable by the loop."""
    from mlframe.training._configs_base import TargetTypes
    from mlframe.training.composite.hurdle import HurdleRegressor

    ctx = SimpleNamespace(strategy_by_model={}, sorted_mlframe_models=None, mlframe_models=None)
    metadata: dict = {}
    targets = {TargetTypes.REGRESSION: {"charge": _zero_inflated(), "price": np.random.default_rng(2).normal(size=2000)}}
    out = maybe_inject_hurdle_for_zero_inflated(ctx, metadata, ["lgb"], targets, None, SimpleNamespace())
    assert out[0] == "lgb" and len(out) == 2
    label, est = out[1]
    assert label == "hurdle" and isinstance(est, HurdleRegressor) and est.zero_value == 0.0
    assert est._mlframe_only_targets == frozenset({"charge"})
    assert ctx.mlframe_models is out and id(out[1]) in ctx.strategy_by_model
    assert metadata["hurdle_for_zero_inflated"]["hurdle"]["targets"] == ["charge"]


def test_injection_respects_the_flag():
    """``hurdle_for_zero_inflated=False`` leaves the model list untouched."""
    from mlframe.training._configs_base import TargetTypes

    targets = {TargetTypes.REGRESSION: {"charge": _zero_inflated()}}
    models = ["lgb"]
    out = maybe_inject_hurdle_for_zero_inflated(SimpleNamespace(), {}, models, targets, None, SimpleNamespace(hurdle_for_zero_inflated=False))
    assert out is models


def test_distribution_driven_estimator_is_restricted_to_regression_targets():
    """The E3 estimator is a regressor; before the restriction the loop also fit it on every classification target."""
    pytest.importorskip("lightgbm")
    import pandas as pd

    from mlframe.training._configs_base import TargetTypes
    from mlframe.training.composite._estimator_dispatch import maybe_inject_distribution_driven_estimator

    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    targets = {
        TargetTypes.REGRESSION: {"amount": 2 * x + rng.standard_t(2, 500)},
        TargetTypes.BINARY_CLASSIFICATION: {"flag": (x > 0).astype(float)},
    }
    ctx = SimpleNamespace(strategy_by_model={}, sorted_mlframe_models=None, mlframe_models=None)
    out = maybe_inject_distribution_driven_estimator(
        ctx, {"target_distribution_report": {"pathologies": ["heavy_tail(excess_kurt=20.0)"]}}, ["lgb"], targets, None,
        pd.DataFrame({"x": x}), SimpleNamespace(distribution_driven_estimator=True),
    )
    injected = out[-1]
    assert not _entry_not_for_target(injected, "amount")
    assert _entry_not_for_target(injected, "flag")
