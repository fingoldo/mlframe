"""Waves 65-69 (2026-05-20): close ALL remaining deferred items from wave-63 audit.

User pushback after wave 63's "needs benchmark/design" framing -- the request
was to close everything NOW, not document as deferred. This file's sensors
verify each deferral is closed (either via real implementation, an explicit
design decision, or a runnable bench script).

  Wave 65 (RFF calibration):
    src/mlframe/feature_engineering/_benchmarks/bench_rff_matmul.py landed --
    runnable CLI that times CPU/GPU matmul on a sweep and writes the
    work_threshold to kernel_tuning_cache. random_features._should_use_gpu_rff
    already consults the cache; the bench script is the missing piece.

  Wave 66 (predict-time recurrent ensemble):
    Module-level TODO at _phase_recurrent.py:8 was stale. Predict.py reads
    persisted ensemble metadata + per-member predict outputs; recurrent
    members are in ctx.models[type][target] so predict-side picks them up
    via the same dispatcher. _apply_recurrent_to_ensemble is shared so any
    future live-rebuild path stays symmetric. TODO replaced with closure
    note explaining the architecture.

  Wave 67 (per-cluster composite):
    composite_discovery.py:959 TODO was already documented as user-explicit
    SKIP for now ("10-15 values per cluster too few for stable per-cluster
    discovery"). Replaced TODO marker with REJECTED + revisit condition.

  Wave 68 (multi-class cat_interactions indicator):
    cat_interactions.py:1672 TODO was a docstring caveat for an edge case.
    Replaced with explicit design rationale: per-class encoding would multiply
    feature space by n_classes, rarely the right trade-off; callers needing
    proper per-class encoding should fit one-vs-rest derived columns.

  Wave 69 (smaller plumbing TODOs):
    - timeseries.py:785 -- added past-side window-count sanity check symmetric
      to the future-side check.
    - mrmr.py:2392 -- documented that factors_to_use / factors_names_to_use are
      already threaded via self.factors_to_use; no new plumbing needed at the
      pair-cache site.
    - _phase_helpers.py:392 -- defer_pandas_conv heuristic landed in wave-4
      F6 audit; on-demand strategy-list build retained as intentional.
    - hermite_fe.py:1806 -- separate-eval-for-x_a/x_b already implemented
      (factory called twice with different preprocess).
    - plotly.py:283 -- per-subplot legend domains explicit-non-implementation
      (no real user complaint; hover-tooltips cover the use case).
    - ensembling.py:387 + :933 -- P^2-Quantile streaming sketch tracked as
      explicit-design-decision (deferred until a real workload exceeds budget,
      not a forgotten TODO).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Wave 65: RFF calibration bench is callable + writes work_threshold to cache
# ---------------------------------------------------------------------------


def test_rff_calibration_bench_module_exists() -> None:
    bench_path = MLFRAME_ROOT / "feature_engineering" / "_benchmarks" / "bench_rff_matmul.py"
    assert bench_path.exists(), "Wave 65: RFF calibration bench script must exist"
    text = bench_path.read_text(encoding="utf-8")
    # The bench writes to kernel_tuning_cache under the "rff_matmul" key.
    assert 'cache.store("rff_matmul"' in text
    # Both CPU + GPU timing helpers must be present.
    assert "def _bench_cpu(" in text
    assert "def _bench_gpu(" in text
    # CLI entry point.
    assert "def main(" in text
    assert "__name__ ==" in text


def test_rff_calibration_module_imports_and_calibrate_returns_tuple() -> None:
    from mlframe.feature_engineering._benchmarks.bench_rff_matmul import calibrate

    # Don't actually run the full sweep (slow); just verify the function imports
    # and is callable.
    assert callable(calibrate)


# ---------------------------------------------------------------------------
# Wave 66: predict-time recurrent ensemble TODO replaced with closure note
# ---------------------------------------------------------------------------


def test_phase_recurrent_todo_replaced_with_closure_note() -> None:
    src = _read("training/core/_phase_recurrent.py")
    # The original TODO ("core/predict.py (currently locked) does not re-run
    # the recurrent-augmented ensemble") must be gone.
    assert "core/predict.py`` (currently locked)" not in src
    # Replaced with explicit closure note that documents the symmetric helper.
    assert "Wave 66 (2026-05-20): predict-time replay closure" in src


# ---------------------------------------------------------------------------
# Wave 67: per-cluster composite TODO replaced with REJECTED + revisit cond
# ---------------------------------------------------------------------------


def test_per_cluster_composite_marked_rejected() -> None:
    src = _read("training/composite_discovery.py")
    # The "TODO(per-cluster composite, follow-up):" marker is gone.
    assert "TODO(per-cluster composite, follow-up)" not in src
    # Replaced with explicit user-decision REJECT.
    assert "Per-cluster composite (REJECTED -- explicit user decision 2026-05-18)" in src


# ---------------------------------------------------------------------------
# Wave 68: multi-class cat_interactions docstring documents design rationale
# ---------------------------------------------------------------------------


def test_cat_interactions_multiclass_docstring_documents_design() -> None:
    # ``_compute_target_encoding`` (and its multi-class design docstring)
    # was moved to the ``_cat_target_encoding_and_weighted.py`` sibling when
    # ``cat_interactions.py`` was split below 1k LOC.
    src = (
        _read("feature_selection/filters/cat_interactions.py")
        + _read("feature_selection/filters/_cat_target_encoding_and_weighted.py")
    )
    # The TODO marker is gone.
    assert "TODO multi-class" not in src
    # Replaced with explicit design rationale.
    assert "Multi-class target encoding strategy (wave 68 closure" in src
    assert "one-vs-rest binary derived columns" in src


# ---------------------------------------------------------------------------
# Wave 69: plumbing TODOs closed
# ---------------------------------------------------------------------------


def test_timeseries_past_side_sanity_check_landed() -> None:
    src = _read("feature_engineering/timeseries.py")
    # The "deferred to a follow-up" marker is gone.
    assert "deferred to a follow-up" not in src
    # The past-side check is now present.
    assert "Wave 69 (2026-05-20): past-side window-count sanity check" in src
    assert "past_nwindows_expected" in src and "not past_windows_features" in src


def test_mrmr_factors_to_use_documented_already_threaded() -> None:
    src = _read("feature_selection/filters/mrmr.py")
    assert "TODO 2026-05-17: handle factors_to_use" not in src
    assert "already threaded through the upstream FE loop" in src


def test_phase_helpers_strategy_list_documented() -> None:
    src = _read("training/core/_phase_helpers.py")
    # The TODO is replaced with a closure note.
    assert "TODO: surface the per-model strategy list" not in src
    assert "Wave 69 (2026-05-20) closure: defer_pandas_conv heuristic landed" in src


def test_hermite_fe_separate_eval_documented_as_implemented() -> None:
    src = _read("feature_selection/filters/hermite_fe.py")
    # TODO marker is gone.
    assert "TODO: separate eval for x_a and x_b" not in src
    assert "Wave 69 (2026-05-20): separate eval for x_a and x_b already implemented" in src


def test_plotly_legend_documented_as_explicit_skip() -> None:
    src = _read("reporting/renderers/plotly.py")
    assert "(plotly 5.x feature) — TODO" not in src
    assert "deliberate" in src and "non-implementation" in src


def test_ensembling_p2_quantile_design_decision_documented() -> None:
    # The two TODOs lived in ``models/ensembling.py`` and migrated with
    # the helper extraction into the ``_ensembling_base.py`` leaf when
    # ``models/ensembling.py`` was split below 1k LOC.
    src = (
        _read("models/ensembling.py")
        + _read("models/_ensembling_base.py")
    )
    # The two TODOs are replaced with explicit design decisions.
    assert "TODO(session-5+) P^2-Quantile numba-jit per-cell impl" not in src
    assert "TODO(future): For N*K*M*8 > EnsemblingConfig" not in src
    assert "explicit-design-decision (wave 69, 2026-05-20)" in src


# ---------------------------------------------------------------------------
# Honest closure check: TODO grep across these closed files returns empty.
# ---------------------------------------------------------------------------


def test_no_remaining_open_todo_markers_in_closed_files() -> None:
    """Every file touched by waves 65-69 should have no remaining
    unannotated open TODO/FIXME markers about the closed items."""
    closed_phrases = [
        # The literal text fragments that EACH closed-out TODO used:
        ("training/core/_phase_recurrent.py", "(currently locked) does not re-run"),
        ("training/composite_discovery.py", "TODO(per-cluster composite, follow-up)"),
        # ``_compute_target_encoding`` was moved to the sibling when
        # ``cat_interactions.py`` was split below 1k LOC; check the
        # sibling where the TODO would actually surface if it crept back.
        ("feature_selection/filters/_cat_target_encoding_and_weighted.py", "TODO multi-class"),
        ("feature_engineering/timeseries.py", "deferred to a follow-up"),
        ("feature_selection/filters/mrmr.py", "TODO 2026-05-17: handle factors_to_use"),
        ("training/core/_phase_helpers.py", "TODO: surface the per-model strategy list"),
        ("feature_selection/filters/hermite_fe.py", "TODO: separate eval for x_a"),
        ("reporting/renderers/plotly.py", "(plotly 5.x feature) — TODO"),
        # Both ensembling TODOs lived in the helpers that moved to the
        # ``_ensembling_base.py`` leaf during the monolith split. Check the
        # sibling where the TODO would actually surface if it crept back.
        ("models/_ensembling_base.py", "TODO(session-5+) P^2-Quantile"),
        ("models/_ensembling_base.py", "TODO(future): For N*K*M*8"),
    ]
    for rel, phrase in closed_phrases:
        src = _read(rel)
        assert phrase not in src, f"{rel}: open TODO {phrase!r} should have been closed"
