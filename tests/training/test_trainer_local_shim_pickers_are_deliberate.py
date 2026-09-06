"""trainer's four shim pickers are a deliberate copy of the factory's, not a carve leftover.

`_model_factories` and `trainer` each define `_xgb_classifier_cls` / `_xgb_regressor_cls` /
`_lgb_classifier_cls` / `_lgb_regressor_cls` with the same logic, differing only in local import aliases.
A duplicate-function scan reports them, and the obvious "fix" is to have trainer delegate to the factory.

That would break documented behaviour. Each picker reads `USE_*_SHIM` from ITS OWN module, so
`monkeypatch.setattr(trainer, "USE_LGB_DATASET_REUSE_SHIM", False)` flips the trainer's dispatch -- which is
how the toggle is documented and how the existing shim tests drive it. The factory's constant is a separate
binding and deliberately does not follow.

These tests pin both halves, so a consolidation fails here rather than silently detaching the toggle.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "flag,picker",
    [
        ("USE_LGB_DATASET_REUSE_SHIM", "_lgb_classifier_cls"),
        ("USE_LGB_DATASET_REUSE_SHIM", "_lgb_regressor_cls"),
        ("USE_XGB_DMATRIX_REUSE_SHIM", "_xgb_classifier_cls"),
        ("USE_XGB_DMATRIX_REUSE_SHIM", "_xgb_regressor_cls"),
    ],
)
def test_the_trainer_flag_drives_the_trainer_picker(flag, picker, monkeypatch):
    """The documented toggle: flipping the constant on `trainer` changes what `trainer` dispatches to."""
    from mlframe.training import trainer as tr_mod

    if not getattr(tr_mod, flag):
        pytest.skip(f"{flag} is already off in this build; the shim class may be unavailable")

    shimmed = getattr(tr_mod, picker)()
    monkeypatch.setattr(tr_mod, flag, False)
    vanilla = getattr(tr_mod, picker)()

    assert shimmed is not vanilla, f"flipping trainer.{flag} did not change trainer.{picker}(); the picker is not reading this module's binding"


@pytest.mark.parametrize(
    "flag,picker",
    [
        ("USE_LGB_DATASET_REUSE_SHIM", "_lgb_classifier_cls"),
        ("USE_XGB_DMATRIX_REUSE_SHIM", "_xgb_classifier_cls"),
    ],
)
def test_the_factory_picker_does_not_follow_the_trainer_flag(flag, picker, monkeypatch):
    """The half that makes the duplication deliberate rather than accidental.

    If trainer's picker were consolidated to delegate to the factory, the two modules would share one
    binding and this assertion would stop holding -- which is exactly the detachment of the documented
    toggle that the duplication exists to avoid.
    """
    from mlframe.training import _model_factories as fac_mod
    from mlframe.training import trainer as tr_mod

    if not getattr(tr_mod, flag):
        pytest.skip(f"{flag} is already off in this build; the shim class may be unavailable")

    before = getattr(fac_mod, picker)()
    monkeypatch.setattr(tr_mod, flag, False)
    after = getattr(fac_mod, picker)()

    assert before is after, (
        f"flipping trainer.{flag} changed _model_factories.{picker}(); the two modules now share one binding, "
        "so the trainer-local toggle no longer means what its docstring says"
    )
