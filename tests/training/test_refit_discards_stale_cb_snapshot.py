"""The RMSE refit of a degenerate CatBoost Huber fit does not resume from the degenerate fit's snapshot.

The GPU budget guard turns on ``save_snapshot`` for the guarded fit, and the loss-fallback refit runs inside it. CatBoost
found the first fit's snapshot, tried to resume from it with a different loss and refused ("Saved model's params are
different from current model's params"), so a production target kept its best_iter=2 Huber fit (R2=-1.28).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

catboost = pytest.importorskip("catboost")

from mlframe.training._training_loop_refit import _maybe_refit_on_degenerate_best_iter


def test_refit_replaces_the_huber_fit_despite_an_armed_snapshot(tmp_path, caplog):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = 150.0 * x[:, 0] + rng.normal(size=400)
    snap = str(tmp_path / "snap.cbsnapshot")
    model = catboost.CatBoostRegressor(
        iterations=100, loss_function="Huber:delta=1.345", eval_metric="Huber:delta=1.345", verbose=False,
        save_snapshot=True, snapshot_file=snap, train_dir=str(tmp_path), random_seed=0, thread_count=1,
    )
    eval_set = [(x[300:], y[300:])]
    model.fit(x[:300], y[:300], eval_set=eval_set)
    assert os.path.exists(snap), "setup: the first fit must leave a snapshot behind, as the guarded GPU fit does"
    with caplog.at_level(logging.WARNING):
        new_best = _maybe_refit_on_degenerate_best_iter(
            model_obj=model, model_type_name="CatBoostRegressor", best_iter=1, train_df=x[:300], train_target=y[:300], fit_params={"eval_set": eval_set},
            logger_=logging.getLogger("t"),
        )
    assert not any("refit rejected" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert new_best is not None
    assert model.get_params()["loss_function"] == "RMSE"
