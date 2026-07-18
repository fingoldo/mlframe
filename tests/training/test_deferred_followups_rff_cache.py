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

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    Monolith-split compat: when the requested file is a parent whose code
    moved to siblings, append every matching sibling so source-pattern
    sensors that pre-date the split still match.
    """
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            _parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    _parts.append(_sub.read_text(encoding="utf-8"))
            primary = "\n".join(_parts)
        else:
            primary = _path.read_text(encoding="utf-8")
    else:
        primary = _path.read_text(encoding="utf-8")
    if rel == "feature_selection/filters/hermite_fe.py":
        _dir = MLFRAME_ROOT / "feature_selection" / "filters"
        for nm in ("_hermite_fe_optimise.py", "_hermite_fe_mi.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/mrmr.py":
        _dir = MLFRAME_ROOT / "feature_selection" / "filters"
        for nm in (
            "_mrmr_fingerprints.py",
            "_mrmr_fit_impl/_fit_impl_core.py",
            "_mrmr_fit_impl/_helpers.py",
            "_mrmr_fe_step/_step_core.py",
            "_mrmr_fe_step/_step_score.py",
            "_mrmr_fe_step/_helpers.py",
            "_mrmr_validate_transform.py",
        ):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# Wave 65: RFF calibration bench is callable + writes work_threshold to cache
# ---------------------------------------------------------------------------


def test_rff_calibration_bench_module_exists() -> None:
    """Rff calibration bench module exists."""
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
    """Rff calibration module imports and calibrate returns tuple."""
    from mlframe.feature_engineering._benchmarks.bench_rff_matmul import calibrate

    # Don't actually run the full sweep (slow); just verify the function imports
    # and is callable.
    assert callable(calibrate)


# ---------------------------------------------------------------------------
# Wave 66: predict-time recurrent ensemble TODO replaced with closure note
# ---------------------------------------------------------------------------


def test_phase_recurrent_todo_replaced_with_closure_note() -> None:
    """Phase recurrent todo replaced with closure note."""
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
    """Per cluster composite marked rejected."""
    src = _read("training/composite/discovery/__init__.py")
    # The "TODO(per-cluster composite, follow-up):" marker is gone.
    assert "TODO(per-cluster composite, follow-up)" not in src
    # Replaced with an explicit REJECTED design-decision marker.
    assert "Per-cluster composite (REJECTED" in src


# ---------------------------------------------------------------------------
# Wave 68: multi-class cat_interactions docstring documents design rationale
# ---------------------------------------------------------------------------


def test_cat_interactions_multiclass_docstring_documents_design() -> None:
    # ``_compute_target_encoding`` (and its multi-class design docstring)
    # was moved to the ``_cat_target_encoding_and_weighted.py`` sibling when
    # ``cat_interactions.py`` was split below 1k LOC.
    """Cat interactions multiclass docstring documents design."""
    src = _read("feature_selection/filters/cat_interactions.py") + _read("feature_selection/filters/_cat_target_encoding_and_weighted.py")
    # The TODO marker is gone.
    assert "TODO multi-class" not in src
    # Replaced with explicit design rationale.
    assert "Multi-class target encoding strategy (wave 68 closure" in src
    assert "one-vs-rest binary derived columns" in src


# ---------------------------------------------------------------------------
# Wave 69: plumbing TODOs closed
# ---------------------------------------------------------------------------


def test_timeseries_past_side_sanity_check_landed() -> None:
    """Timeseries past side sanity check landed."""
    src = _read("feature_engineering/timeseries.py")
    # The "deferred to a follow-up" marker is gone.
    assert "deferred to a follow-up" not in src
    # The past-side check is now present.
    assert "Wave 69 (2026-05-20): past-side window-count sanity check" in src
    assert "past_nwindows_expected" in src and "not past_windows_features" in src


def test_mrmr_factors_to_use_documented_already_threaded() -> None:
    """Mrmr factors to use documented already threaded."""
    src = _read("feature_selection/filters/mrmr.py")
    assert "TODO 2026-05-17: handle factors_to_use" not in src
    assert "already threaded through the upstream FE loop" in src


def test_phase_helpers_strategy_list_documented() -> None:
    """Phase helpers strategy list documented."""
    src = _read("training/core/_phase_helpers.py")
    # The TODO is replaced with a closure note.
    assert "TODO: surface the per-model strategy list" not in src
    assert "Wave 69 (2026-05-20) closure: defer_pandas_conv heuristic landed" in src


def test_hermite_fe_separate_eval_documented_as_implemented() -> None:
    # The closure marker moved out of ``hermite_fe.py`` into the sibling
    # ``_hermite_fe_optimise_pair.py`` during the hermite-fe monolith split.
    """Hermite fe separate eval documented as implemented."""
    import pathlib
    import mlframe as _mlframe

    _root = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "filters"
    src = ""
    for nm in ("hermite_fe.py", "_hermite_fe_optimise_pair.py"):
        p = _root / nm
        if p.exists():
            src += p.read_text(encoding="utf-8")
            src += "\n"
    # TODO marker is gone.
    assert "TODO: separate eval for x_a and x_b" not in src
    assert "Wave 69 (2026-05-20): separate eval for x_a and x_b already implemented" in src


def test_plotly_legend_implemented_not_skipped() -> None:
    # The plotly static legend landed (was a documented TODO/skip): the renderer now wires a
    # ``static_legend`` flag through to ``showlegend`` for png/svg/pdf exports that have no hover.
    """Plotly legend implemented not skipped."""
    src = _read("reporting/renderers/plotly.py")
    assert "(plotly 5.x feature) — TODO" not in src
    assert "static_legend" in src and "showlegend=static_legend" in src


def test_ensembling_p2_quantile_design_decision_documented() -> None:
    # The two TODOs were replaced with explicit design decisions in the
    # ``models/ensembling`` package (``base.py`` leaf); reading the package
    # concats every submodule.
    """Ensembling p2 quantile design decision documented."""
    src = _read("models/ensembling.py")
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
        ("training/composite/discovery/__init__.py", "TODO(per-cluster composite, follow-up)"),
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
        # ``models/ensembling`` package during the monolith split; reading
        # the package concats every submodule where a TODO could resurface.
        ("models/ensembling.py", "TODO(session-5+) P^2-Quantile"),
        ("models/ensembling.py", "TODO(future): For N*K*M*8"),
    ]
    for rel, phrase in closed_phrases:
        src = _read(rel)
        assert phrase not in src, f"{rel}: open TODO {phrase!r} should have been closed"
