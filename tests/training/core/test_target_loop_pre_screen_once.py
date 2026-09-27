"""The unsupervised pre-screen runs once, before any target, and its drops survive every target's frame scope.

It used to run inside the first target's ``_train_one_target``, inside ``target_scoped_frames``. When a suite had
per-target supervised columns the scope restored its saved frames on exit, putting the screened-out columns back,
while ``_pre_screen_done`` stayed latched: every target after the first trained on columns the screen had dropped.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.core import _main_train_suite_target_loop as loop
from mlframe.training.pipeline import _per_target_supervised_fe as ptsfe


def _ctx(frame):
    """A minimal suite context carrying ``frame`` as the train frame, with unsupervised pre-screening on."""
    return SimpleNamespace(
        feature_selection_config=SimpleNamespace(pre_screen_unsupervised=True, pre_screen_variance_threshold=0.0,
                                                 pre_screen_null_fraction_threshold=0.99),
        _pre_screen_done=False, _pre_screen_dropped_cols=[], target_by_type={"regression": {"y1": None, "y2": None}},
        cat_features=[], verbose=0, metadata={}, slug_to_original_target_type={},
        filtered_train_df=frame, filtered_val_df=None, train_df_pd=frame, val_df_pd=None, test_df_pd=None,
        train_df_polars=None, val_df_polars=None, test_df_polars=None, calib_df=frame.copy(),
    )


def test_every_target_trains_without_the_screened_out_column(monkeypatch):
    """The pre-screen runs once, and every target trains without the column it dropped, even after per-target frame swaps."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=200), "const": 1.0, "sup_y1": rng.normal(size=200), "sup_y2": rng.normal(size=200)})
    ctx = _ctx(frame)
    # Each target hides the other's supervised column, which makes target_scoped_frames save and restore the frames.
    monkeypatch.setattr(ptsfe, "foreign_columns", lambda md, tt, name: ["sup_y2"] if name == "y1" else ["sup_y1"])
    seen = {}

    def _train_one_target(ctx_, target_type, targets, name, values):
        """Record the columns each target trained on."""
        seen[name] = set(ctx_.train_df_pd.columns)

    y = rng.normal(size=200)
    loop.train_every_target(ctx, {"regression": {"y1": y, "y2": y}}, {}, SimpleNamespace(_train_one_target=_train_one_target))
    assert "const" not in seen["y1"] and "const" not in seen["y2"]
    assert "const" not in ctx.train_df_pd.columns
    assert "const" not in ctx.calib_df.columns, "the calibration slice must lose the columns the models were trained without"
