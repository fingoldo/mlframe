"""The ensemble chooser reports which split the winning flavour was picked on."""

from __future__ import annotations

from types import SimpleNamespace

from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour, _choose_ensemble_flavour_and_surface


def _ens(auc: float, split: str):
    """One flavour's result carrying a roc_auc on ``split`` only."""
    return SimpleNamespace(metrics={split: {"roc_auc": auc}})


def test_surface_is_val_when_no_oof_metrics_exist():
    """With only val metrics the winner is reported as chosen on val (the early-stopping surface), not on an honest surface."""
    d = {"arithm": _ens(0.70, "val"), "harm": _ens(0.75, "val")}
    assert _choose_ensemble_flavour_and_surface(d) == ("harm", "val")


def test_surface_is_oof_when_oof_metrics_exist():
    """OOF metrics take precedence over val and are reported as the surface."""
    d = {
        "arithm": SimpleNamespace(metrics={"oof": {"roc_auc": 0.8}, "val": {"roc_auc": 0.1}}),
        "harm": SimpleNamespace(metrics={"oof": {"roc_auc": 0.7}, "val": {"roc_auc": 0.9}}),
    }
    assert _choose_ensemble_flavour_and_surface(d) == ("arithm", "oof")


def test_surface_none_on_first_flavour_fallback_and_wrapper_agrees():
    """No ranking metric anywhere gives surface None, and ``_choose_ensemble_flavour`` returns the same winner name."""
    d = {"arithm": SimpleNamespace(metrics={}), "harm": SimpleNamespace(metrics={})}
    assert _choose_ensemble_flavour_and_surface(d) == ("arithm", None)
    assert _choose_ensemble_flavour(d) == "arithm"
