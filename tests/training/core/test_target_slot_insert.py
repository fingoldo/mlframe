"""A composite spec whose name is already a target must not take over that target's slot (INT-18)."""

from __future__ import annotations

import numpy as np

from mlframe.training.core._target_slots import insert_composite_targets


def _pending(name, values, gain=0.1):
    """A kept pending-composite entry for the regression target ``y``."""
    return {"tt": "regression", "target": "y", "name": name, "values": np.asarray(values, dtype=float), "gain": gain}


def test_a_free_slot_takes_the_composite_values():
    """The normal path: the slot is filled and the entry is returned as inserted."""
    slots, metadata = {"regression": {"y": np.zeros(3)}}, {}
    kept = insert_composite_targets(slots, [_pending("y_lr_b", [1.0, 2.0, 3.0])], metadata)
    assert [k["name"] for k in kept] == ["y_lr_b"]
    np.testing.assert_allclose(slots["regression"]["y_lr_b"], [1.0, 2.0, 3.0])


def test_a_colliding_name_leaves_the_existing_target_alone_and_unships_the_spec():
    """The occupied slot keeps its values, the spec leaves the export and a failure records the collision."""
    raw = np.arange(3, dtype=float)
    slots = {"regression": {"y": raw}}
    metadata = {"composite_target_specs": {"regression": {"y": [{"name": "y"}, {"name": "y_lr_b"}]}}}
    kept = insert_composite_targets(slots, [_pending("y", [9.0, 9.0, 9.0]), _pending("y_lr_b", [1.0, 2.0, 3.0])], metadata)

    np.testing.assert_allclose(slots["regression"]["y"], raw), "the existing target's values were overwritten"
    assert [k["name"] for k in kept] == ["y_lr_b"]
    assert [s["name"] for s in metadata["composite_target_specs"]["regression"]["y"]] == ["y_lr_b"]
    failures = metadata["composite_target_failures"]["regression"]["y"]
    assert [f["name"] for f in failures] == ["y"] and "already belongs to another target" in failures[0]["reason"]
