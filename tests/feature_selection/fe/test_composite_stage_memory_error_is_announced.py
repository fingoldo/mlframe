"""A MemoryError in the composite-aggregate stage is announced with the memory wording and the stage returns the input frame."""
import logging

import pandas as pd

import mlframe.feature_selection.filters._composite_group_agg_fe as comp
from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_cascade_mid_a as stage


def test_memory_error_is_logged_with_the_memory_wording(monkeypatch, caplog):
    """The stage catches MemoryError, logs the memory wording (it names the fit as WITHOUT the columns) and returns the frame unchanged."""
    X = pd.DataFrame({"g1": [0, 1, 0, 1], "g2": [0, 0, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]})

    def boom(*args, **kwargs):
        """Raise the allocation failure the composite stage hits on large inputs."""
        raise MemoryError("Unable to allocate 1.19 GiB")

    monkeypatch.setattr(comp, "hybrid_composite_group_agg_fe", boom)

    class _Owner:
        """The estimator the stage reads: nothing beyond the call arguments."""

    with caplog.at_level(logging.WARNING, logger=stage.logger.name):
        out = stage._stage_composite_group_agg(_Owner(), lambda name, default=False: True, X, X["x"].to_numpy(), None, False)
    assert out is X
    assert any("out of memory" in r.getMessage() and "WITHOUT composite-aggregate" in r.getMessage() for r in caplog.records)
