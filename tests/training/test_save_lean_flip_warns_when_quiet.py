"""The save path's pre-flip to a lean bundle drops prediction fields, so it must announce itself even at verbosity 0."""

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.training._io_save import _save_mlframe_model_flip_must_swapped_cheaper


def _big_model():
    """A namespace whose in-memory size is well above a 0.001 MB threshold."""
    return SimpleNamespace(preds=np.zeros(200_000), probs=np.zeros(200_000), model=None)


def test_lean_pre_flip_logs_warning_at_verbose_zero(caplog):
    """verbose=0 must not hide a decision that drops fields from the saved file."""
    with caplog.at_level(logging.WARNING):
        lean = _save_mlframe_model_flip_must_swapped_cheaper(False, True, 0.001, _big_model(), 0, "x.dump")
    assert lean is True
    assert any("flipping lean=True" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
