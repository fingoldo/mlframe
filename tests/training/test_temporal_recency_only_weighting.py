"""On a temporal split, fit the recency schema only -- uniform doubles the work for a superseded assumption.

A production run reported ``Using 2 weighting schema(s) from extractor: ['uniform', 'recency']`` on a split
with a real time axis, and spent 26 minutes fitting both. Recency weighting exists precisely because older rows
matter less; running uniform alongside it produces a second model built on the assumption the time axis has
already contradicted.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.core._phase_train_one_target_schema import (
    _drop_uniform_on_temporal_data,
    _resolve_weight_schemas_and_warn_val_placement,
)


def _ctx(temporal: bool = True, opt_out: bool = False):
    """A minimal context: timestamps decide temporality, behavior_config carries the opt-out."""
    return SimpleNamespace(
        timestamps=np.arange(100, dtype=np.int64) if temporal else None,
        behavior_config=SimpleNamespace(temporal_recency_only_weighting=not opt_out),
        _sw_log_emitted=False,
        _val_placement_warn_emitted=False,
    )


BOTH = {"uniform": None, "recency": np.ones(100)}


class TestTheDropRule:
    """When uniform goes and when it stays."""

    def test_uniform_is_dropped_on_temporal_data(self):
        """The case from the log: both offered, a time axis present."""
        schemas, dropped = _drop_uniform_on_temporal_data(dict(BOTH), _ctx(temporal=True))
        assert dropped is True
        assert list(schemas) == ["recency"]

    def test_kept_without_a_time_axis(self):
        """No timestamps means no basis for preferring recency; nothing is removed."""
        schemas, dropped = _drop_uniform_on_temporal_data(dict(BOTH), _ctx(temporal=False))
        assert dropped is False and "uniform" in schemas

    def test_kept_when_no_recency_schema_is_offered(self):
        """Uniform is the only thing on offer; dropping it would leave nothing to fit."""
        schemas, dropped = _drop_uniform_on_temporal_data({"uniform": None}, _ctx(temporal=True))
        assert dropped is False and list(schemas) == ["uniform"]

    def test_opt_out_keeps_both(self):
        """A caller that wants the comparison must be able to have it."""
        schemas, dropped = _drop_uniform_on_temporal_data(dict(BOTH), _ctx(temporal=True, opt_out=True))
        assert dropped is False and set(schemas) == {"uniform", "recency"}

    @pytest.mark.parametrize("name", ["recency", "recency_weights", "time_decay", "exponential_decay"])
    def test_every_recency_flavoured_name_triggers_it(self, name):
        """Extractors spell this differently; all of them supersede uniform for the same reason."""
        schemas, dropped = _drop_uniform_on_temporal_data({"uniform": None, name: np.ones(3)}, _ctx())
        assert dropped is True and list(schemas) == [name]

    def test_an_unrelated_second_schema_does_not_trigger_it(self):
        """A per-class or per-group weighting is not a recency schema and does not supersede uniform."""
        schemas, dropped = _drop_uniform_on_temporal_data({"uniform": None, "class_balance": np.ones(3)}, _ctx())
        assert dropped is False and "uniform" in schemas


class TestThroughTheResolver:
    """End to end, including what the operator is told."""

    def test_resolver_returns_recency_only_and_says_why(self, caplog):
        """A silent halving of the fit count would be worse than the waste it removes."""
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_train_one_target"):
            schemas = _resolve_weight_schemas_and_warn_val_placement(
                sample_weights=dict(BOTH), split_config=SimpleNamespace(val_placement="forward"), ctx=_ctx()
            )
        assert list(schemas) == ["recency"]
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "Dropping the 'uniform' weighting schema" in text
        assert "temporal_recency_only_weighting" in text

    def test_no_weights_still_defaults_to_uniform(self):
        """An extractor supplying nothing must still get a fittable schema."""
        schemas = _resolve_weight_schemas_and_warn_val_placement(sample_weights=None, split_config=SimpleNamespace(val_placement="forward"), ctx=_ctx())
        assert list(schemas) == ["uniform"]
