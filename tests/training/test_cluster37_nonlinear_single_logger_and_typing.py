"""Cluster 37 regression: nonlinear.py defines its logger and TYPE_CHECKING import once (audit 2026-11-20, finding 37-F35).

The module carried a duplicate ``logger`` assignment and a duplicate ``TYPE_CHECKING`` block. Each now appears once.
"""

from pathlib import Path

from mlframe.training.composite.transforms import nonlinear


def test_logger_defined_once():
    """The module defines its logger exactly once (37-F35)."""
    src = Path(nonlinear.__file__).resolve().read_text(encoding="utf-8")
    assert src.count('logger = logging.getLogger("mlframe.training.composite_transforms")') == 1


def test_type_checking_block_once():
    """The TYPE_CHECKING import block appears exactly once (37-F35)."""
    src = Path(nonlinear.__file__).resolve().read_text(encoding="utf-8")
    assert src.count("if TYPE_CHECKING:") == 1
