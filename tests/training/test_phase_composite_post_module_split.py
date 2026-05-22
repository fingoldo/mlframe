"""Wave 100 (2026-05-21): split _phase_composite_post.py (1157 lines)
into _phase_composite_post.py (now 773 lines) + new
_phase_composite_wrapping.py (419 lines).

Moved to the sibling file: ``_run_composite_target_wrapping`` (the
~390-line function that wraps fitted T-scale models in
``CompositeTargetEstimator``).

Original re-exports the symbol so existing imports continue to work.
"""
from __future__ import annotations

from pathlib import Path


def test_wrapping_symbol_still_importable_from_facade() -> None:
    from mlframe.training.core._phase_composite_post import _run_composite_target_wrapping
    assert callable(_run_composite_target_wrapping)


def test_other_phase_post_symbols_still_importable() -> None:
    from mlframe.training.core._phase_composite_post import (
        recover_composite_y_scale_metrics,
        _run_suite_end_dummy_baselines_summary,
        run_composite_post_processing,
    )
    for fn in (
        recover_composite_y_scale_metrics,
        _run_suite_end_dummy_baselines_summary,
        run_composite_post_processing,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training" / "core"
    facade = root / "_phase_composite_post.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_phase_composite_post.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_symbol() -> None:
    """Identity: facade and sibling expose the SAME function object."""
    from mlframe.training.core import _phase_composite_post, _phase_composite_wrapping
    assert _phase_composite_post._run_composite_target_wrapping is _phase_composite_wrapping._run_composite_target_wrapping
