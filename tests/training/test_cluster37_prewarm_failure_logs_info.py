"""Cluster 37 regression: a failed numba warm-up is logged at INFO, not DEBUG (audit 2026-11-20, finding 37-F41).

The warm-up is optional, but when it fails the first fit pays the JIT cost; an operator should see that at INFO level, not only at DEBUG.
"""

from pathlib import Path

from mlframe.training.core import _phase_config_setup as setup


def test_warmup_failure_is_logged_at_info():
    """The warm-up failure message is emitted at INFO (37-F41)."""
    src = Path(setup.__file__).resolve().read_text(encoding="utf-8")
    assert 'logger.info("numba kernel warm-up failed, first real call will pay JIT cost' in src
    assert 'logger.debug("numba kernel warm-up failed' not in src
