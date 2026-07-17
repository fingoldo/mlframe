"""Wave 103 (2026-05-21): split training/neural/recurrent.py
(1347 lines) into recurrent.py (now 963 lines) + new
_recurrent_torch_model.py (518 lines).

Moved to the sibling file: the ``RecurrentTorchModel`` PyTorch Lightning
model class (~390 lines). The two wrappers
(``RecurrentClassifierWrapper`` / ``RecurrentRegressorWrapper``) and
their shared base (``_RecurrentWrapperBase``) stay in the parent.

Original re-exports the moved class so existing imports keep working.
"""

from __future__ import annotations

from pathlib import Path


def test_recurrent_torch_model_importable_from_facade() -> None:
    from mlframe.training.neural.recurrent import RecurrentTorchModel

    assert RecurrentTorchModel is not None
    # The class is a LightningModule subclass; check via base-class name string
    # so the test works whether the codebase uses `lightning` or `pytorch_lightning`
    # (two distinct package names that resolve to different LightningModule
    # base classes in different installs).
    base_names = {cls.__name__ for cls in RecurrentTorchModel.__mro__}
    assert "LightningModule" in base_names


def test_wrappers_still_importable() -> None:
    from mlframe.training.neural.recurrent import (
        _RecurrentWrapperBase,
        RecurrentClassifierWrapper,
        RecurrentRegressorWrapper,
        extract_sequences,
        extract_sequences_chunked,
    )

    assert _RecurrentWrapperBase is not None
    assert RecurrentClassifierWrapper is not None
    assert RecurrentRegressorWrapper is not None
    assert callable(extract_sequences)
    assert callable(extract_sequences_chunked)


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent.parent / "src" / "mlframe" / "training" / "neural"
    facade = root / "recurrent.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"recurrent.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_class() -> None:
    """Identity: facade and sibling expose the SAME class object."""
    from mlframe.training.neural import recurrent, _recurrent_torch_model

    assert recurrent.RecurrentTorchModel is _recurrent_torch_model.RecurrentTorchModel
