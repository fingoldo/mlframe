"""Regression: the honest-OOF refine's train-RMSE fallback skips the ensemble only on its two documented conditions.

The fallback used to return ``proceed=False`` unconditionally after its warning, so a target with scored components was
never ensembled and the drop-unscored / ``_oof_rmses`` steps were dead code.
"""

from types import SimpleNamespace

import numpy as np

import mlframe.training.core._phase_composite_post_xt_ensemble as ens


def _refine(monkeypatch, rmse_proxy, drop_fn):
    """Run _build_cross_target_honest_oof_refine on the train-RMSE fallback path with stubbed collaborators."""
    monkeypatch.setattr(ens, "_build_cross_target_e_step1_spec_base_matrix", lambda *a, **k: (["a", "b", "c"], ["ca", "cb", "cc"]))
    monkeypatch.setattr(ens, "_build_cross_target_e_step2_oof_residuals_redundancy", lambda *a, **k: None)
    st = SimpleNamespace(
        _oof_pred_matrix=None,
        _oof_y_holdout=None,
        _defer_train_proxy=True,
        _rmse_arr=None,
        _oof_components=None,
        _oof_names=None,
        _oof_rmses=None,
        _oof_frac=0.0,
        _surviving=[],
    )
    out = ens._build_cross_target_honest_oof_refine(
        st, ["ca", "cb", "cc"], ["a", "b", "c"], [], None, None, None, None, None, None, None, None, None, None, None,
        "tgt", None, {}, None, lambda: rmse_proxy, drop_fn,
    )
    return out, st


def test_fallback_with_scored_components_proceeds_and_keeps_scored_rmse(monkeypatch):
    """Two scored components survive the drop: the ensemble proceeds and _oof_rmses is the scored-subset RMSE."""
    filtered = np.array([0.5, 0.7])

    def drop(rmse):
        """Drop the scored component named in the test, leaving the rest in place."""
        return ["ca", "cc"], ["a", "c"], filtered

    (names, comps, proceed), st = _refine(monkeypatch, np.array([0.5, np.nan, 0.7]), drop)

    assert proceed is True
    assert names == ["a", "b", "c"]
    assert comps == ["ca", "cb", "cc"]
    assert st._oof_rmses is filtered


def test_fallback_with_no_scored_component_skips_ensemble(monkeypatch):
    """No component scored on train: the ensemble is skipped (proceed=False)."""
    (_, _, proceed), _ = _refine(monkeypatch, np.array([np.nan, np.nan, np.nan]), lambda rmse: ([], [], rmse))

    assert proceed is False


def test_fallback_with_fewer_than_two_scored_components_skips_ensemble(monkeypatch):
    """Only one component survives the drop: the ensemble is skipped (proceed=False)."""

    def drop(rmse):
        """Drop the scored component named in the test, leaving the rest in place."""
        return ["ca"], ["a"], np.array([0.5])

    (_, _, proceed), _ = _refine(monkeypatch, np.array([0.5, np.nan, np.nan]), drop)

    assert proceed is False
