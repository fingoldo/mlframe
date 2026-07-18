"""Regression sensor: ``_flat_torch_module.on_train_end`` references
``ModelCheckpoint`` and ``os`` at function-body scope. When the W10E
monolith carve moved this method from the parent ``flat.py`` into the
sibling, both imports were dropped silently because Python lazy-resolves
name lookups at call time -- a bare smoke ``import`` of the module looks
healthy. The bug only surfaces when an MLP trainer actually exercises
``load_best_weights_on_train_end=True`` on a real ModelCheckpoint
callback.

Per CLAUDE.md "Monolith split: AST-audit sibling for unresolved names",
this sensor pins the names that must be resolvable so a future carve /
refactor that drops the import fails at collection rather than at fit
time deep in a Lightning hook.
"""

from __future__ import annotations

import importlib


def test_flat_torch_module_on_train_end_names_are_resolvable():
    """``ModelCheckpoint`` and ``os.path`` must be importable inside
    ``_flat_torch_module``'s ``on_train_end`` body."""
    mod = importlib.import_module("mlframe.training.neural._flat_torch_module")
    # ``os`` and ``ModelCheckpoint`` are module-level names the method
    # references; if the carve dropped the import, ``getattr`` returns
    # ``None`` (no attribute) instead of a callable / module object.
    os_mod = getattr(mod, "os", None)
    assert os_mod is not None and hasattr(os_mod, "path"), "_flat_torch_module no longer imports os; on_train_end will raise NameError on checkpoint-load path."
    model_checkpoint_cls = getattr(mod, "ModelCheckpoint", None)
    assert model_checkpoint_cls is not None and isinstance(
        model_checkpoint_cls, type
    ), "_flat_torch_module no longer imports ModelCheckpoint; on_train_end will raise NameError on callback scan."


def test_flat_torch_module_mlp_class_exposes_on_train_end():
    """Smoke: the on_train_end hook is defined on MLPTorchModel."""
    from mlframe.training.neural._flat_torch_module import MLPTorchModel

    on_train_end = getattr(MLPTorchModel, "on_train_end", None)
    assert callable(on_train_end), "MLPTorchModel dropped on_train_end"
