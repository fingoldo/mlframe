"""Audit 2026-11-20 finding 35-F10: trimming a capped ensemble must keep the per-component means predict uses for dropouts.

The trim branch of ``cap_inference_components`` dropped ``_component_col_means`` and the meta model without a trace, so a capped
linear_stack lost its dropout stand-in and served a forecast halved by a dropped component.
"""

import logging

import numpy as np

from mlframe.training.composite.ensemble._cross_target import CompositeCrossTargetEnsemble


class _Const:
    """Component stub returning a constant prediction."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        """Return the constant for every row."""
        return np.full(len(X), self.value)


def test_trim_keeps_component_means_for_kept_components_and_warns_on_meta_drop(caplog):
    """After a trim to the top-2 components, the stored means are those of the kept components, and a meta model drop is logged."""
    ens = CompositeCrossTargetEnsemble(
        component_models=[_Const(1.0), _Const(2.0), _Const(3.0)],
        component_names=["a", "b", "c"],
        weights=np.array([0.1, 0.5, 0.4]),
        strategy="linear_stack",
        notes={},
    )
    ens._component_col_means = [10.0, 20.0, 30.0]
    ens._meta_model = object()
    with caplog.at_level(logging.WARNING):
        capped = ens.cap_inference_components(2)
    assert capped._component_col_means == [20.0, 30.0]
    assert capped._meta_model is None
    assert any("dropping the meta model" in rec.getMessage() for rec in caplog.records)
