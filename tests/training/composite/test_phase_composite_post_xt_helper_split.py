"""Wave 12a monolith-split sensor for ``mlframe.training.core._phase_composite_post``.

Carve pattern: the per-target body of the cross-target ensemble loop was lifted out to ``_phase_composite_post_xt_ensemble._build_cross_target_ensemble_for_target``. The parent's ``run_composite_post_processing`` now calls the helper for each ``(_tt_e, _orig_tname)`` pair; the helper mutates ``models``, ``metadata``, and ``_train_pred_cache`` in place (same contract as the pre-carve inline body).

Sensors:
1. parent + sibling import cleanly
2. helper symbol resolves
3. parent LOC drops below 400 (was 938)
4. behavioural smoke: ``run_composite_post_processing`` runs end-to-end on a synthetic with no composite specs (the early-out path) without raising
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.training.core import _phase_composite_post

    return _phase_composite_post


@pytest.fixture(scope="module")
def xt_sibling():
    from mlframe.training.core import _phase_composite_post_xt_ensemble

    return _phase_composite_post_xt_ensemble


def test_helper_symbol_resolves(xt_sibling):
    assert hasattr(xt_sibling, "_build_cross_target_ensemble_for_target")


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 400, f"facade is {n_lines} LOC, expected < 400 after Wave 12 cross-target carve"


def test_lag_predict_still_exported(parent_module):
    """Sanity: prior Wave 11 carve still resolves through parent re-export."""
    from mlframe.training.core import _phase_composite_post_lag_predict

    assert parent_module._LagPredictDeployableModel is _phase_composite_post_lag_predict._LagPredictDeployableModel


def test_run_composite_post_no_specs_path(parent_module):
    """Behavioural smoke: when no composite specs exist AND discovery is disabled, the function returns (models, metadata) unchanged with the dummy-baselines summary appended. Exercises the post-carve call site without needing a full suite fixture."""
    models = {"regression": {"y0": [SimpleNamespace(model=None, model_name="dummy", metrics={})]}}
    metadata = {"composite_target_specs": {}, "dummy_baselines": {}}
    target_by_type = {"regression": {"y0": np.array([0.0, 1.0, 2.0, 3.0])}}
    cfg = SimpleNamespace(
        enabled=False,
        cross_target_ensemble_strategy="off",
        always_build_ct_ensemble_for_raw=False,
        skip_wrap_pass_predict=True,
        enable_wrap_pass_watchdog=False,
    )
    dummy_cfg = SimpleNamespace(enabled=False)
    rpt_cfg = SimpleNamespace(plot_outputs=None, plot_dpi=None)
    train_idx = np.array([0, 1])
    val_idx = np.array([2])
    test_idx = np.array([3])
    df = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})

    out_models, out_metadata = parent_module.run_composite_post_processing(
        models=models,
        metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=cfg,
        target_name="y0",
        model_name="dummy",
        filtered_train_df=df.iloc[train_idx],
        filtered_val_df=df.iloc[val_idx],
        test_df_pd=df.iloc[test_idx],
        filtered_train_idx=train_idx,
        filtered_val_idx=val_idx,
        test_idx=test_idx,
        train_df_pd=df.iloc[train_idx],
        val_df_pd=df.iloc[val_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        dummy_baselines_config=dummy_cfg,
        reporting_config=rpt_cfg,
        plot_file=None,
        verbose=False,
    )
    assert out_models is models
    assert out_metadata is metadata
