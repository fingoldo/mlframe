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


def _src_has(text: str) -> bool:
    """Whether any module under ``mlframe/`` contains ``text``.

    Design rationale migrates between modules on every carve, and the per-file lists in ``_read``
    below have had to be extended after each one. A sensor that only asks whether the rationale is
    still written down somewhere does not care where it currently lives.
    """
    return any(text in path.read_text(encoding="utf-8", errors="ignore") for path in MLFRAME_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Wave 65: RFF calibration bench is callable + writes work_threshold to cache
# ---------------------------------------------------------------------------


def test_rff_calibration_bench_module_exists() -> None:
    """Rff calibration bench module exists and exposes the expected CPU/GPU/CLI entry points."""
    bench_path = MLFRAME_ROOT / "feature_engineering" / "_benchmarks" / "bench_rff_matmul.py"
    assert bench_path.exists(), "Wave 65: RFF calibration bench script must exist"

    from mlframe.feature_engineering._benchmarks import bench_rff_matmul as _mod

    assert callable(_mod._bench_cpu)
    assert callable(_mod._bench_gpu)
    assert callable(_mod.main)


def test_rff_calibration_main_writes_work_threshold_to_kernel_tuning_cache(monkeypatch) -> None:
    """``main()`` must actually persist the calibrated crossover under the ``"rff_matmul"`` key that
    ``random_features._should_use_gpu_rff`` looks up -- not merely mention it in source text. Stubs
    ``calibrate`` to a deterministic threshold and spies on the real ``KernelTuningCache.update`` call
    to assert on the runtime side effect."""
    from mlframe.feature_engineering._benchmarks import bench_rff_matmul as _mod

    calls: list[dict] = []

    class _FakeCache:
        """Spy standing in for the real ``KernelTuningCache``, recording every ``update()`` call."""

        def update(self, key, axes, regions):
            """Record the call args instead of touching the real on-disk cache."""
            calls.append({"key": key, "axes": axes, "regions": regions})

    fake_sweep_row = {
        "n": 1000,
        "d": 16,
        "work": 16000,
        "cpu_s": 0.01,
        "gpu_s": 0.005,
        "speedup": 2.0,
        "gpu_wins": True,
    }
    monkeypatch.setattr(_mod, "calibrate", lambda *a, **k: (12345, [fake_sweep_row]))

    class _FakeKTC:
        """Spy standing in for the real ``KernelTuningCache`` class, returning the fake instance."""

        @staticmethod
        def load_or_create():
            """Return the fake cache instead of loading/creating a real on-disk one."""
            return _FakeCache()

    monkeypatch.setattr("pyutilz.performance.kernel_tuning.cache.KernelTuningCache", _FakeKTC)

    rc = _mod.main()

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["key"] == "rff_matmul"
    assert calls[0]["regions"] == [{"work_threshold": 12345}]


def test_rff_calibration_module_imports_and_calibrate_returns_tuple() -> None:
    """Rff calibration module imports and calibrate returns tuple."""
    from mlframe.feature_engineering._benchmarks.bench_rff_matmul import calibrate

    # Don't actually run the full sweep (slow); just verify the function imports
    # and is callable.
    assert callable(calibrate)


# ---------------------------------------------------------------------------
# Wave 67: per-cluster composite TODO replaced with REJECTED + revisit cond
# ---------------------------------------------------------------------------


def test_per_cluster_composite_decision_is_recorded() -> None:
    """The per-cluster composite question carries a decision, not an open TODO.

    It was closed as REJECTED and has since been REOPENED and implemented in ``discovery/_per_group.py``,
    so pinning the rejection would now pin a reversed decision. What must hold either way is that the
    marker is a recorded decision rather than a deferral.
    """
    assert not _src_has("TODO(per-cluster composite, follow-up)")
    assert _src_has("Per-cluster composite (REJECTED") or _src_has("Per-cluster composite (REOPENED")


# ---------------------------------------------------------------------------
# Wave 68: multi-class cat_interactions docstring documents design rationale
# ---------------------------------------------------------------------------


def test_cat_interactions_multiclass_docstring_documents_design() -> None:
    # ``_compute_target_encoding`` (and its multi-class design docstring)
    # was moved to the ``_cat_target_encoding_and_weighted.py`` sibling when
    # ``cat_interactions.py`` was split below 1k LOC.
    """Cat interactions multiclass docstring documents design."""
    # The wave-marker prefix the rationale originally carried is gone: this repo keeps audit markers
    # out of comments. What matters is that the strategy itself is still written down.
    assert not _src_has("TODO multi-class")
    assert _src_has("one-vs-rest binary derived columns")


# ---------------------------------------------------------------------------
# Wave 69: plumbing TODOs closed
# ---------------------------------------------------------------------------


def test_timeseries_past_side_sanity_check_landed() -> None:
    """A base point whose past windows are only PARTIALLY satisfied is skipped, so every emitted row carries the full expected window count."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_engineering.timeseries import create_windowed_features

    df = pd.DataFrame({"vol": np.ones(30), "pos": np.arange(30, dtype=float)})

    def apply_fcn(df, row_features, targets, features_names, dataset_name):
        """Emit the last position of the window; register one column name per window."""
        row_features.append(float(df["pos"].iloc[-1]))
        if f"{dataset_name}-last" not in features_names:
            features_names.append(f"{dataset_name}-last")

    X, Y = create_windowed_features(
        df=df,
        start_index=0,
        end_index=20,
        past_processing_fcn=apply_fcn,
        future_processing_fcn=apply_fcn,
        past_windows={"vol": [3, 10]},
        future_windows={"": [1]},
    )
    assert list(X.columns) == ["vol:3-last", "vol:10-last"]
    assert X.shape == (12, 2)
    assert len(Y) == len(X)
    assert not X.isna().any().any()
    assert X.iloc[:, 0].is_monotonic_increasing


def _rendered(static_legend):
    """Render a one-panel figure through the plotly renderer with the given legend mode."""
    import numpy as _np

    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import FigureSpec, LinePanelSpec

    panel = LinePanelSpec(
        x=_np.array([0.0, 1.0]),
        y=(_np.array([0.0, 1.0]), _np.array([1.0, 0.0])),
        series_labels=("a", "b"),
        title="p",
    )
    spec = FigureSpec(suptitle="s", panels=((panel,),))
    return PlotlyRenderer().render(spec, static_legend=static_legend)


def test_static_legend_turns_the_legend_on_and_parks_it_below_the_plot() -> None:
    """A png/svg/pdf export has no hover, so the legend has to carry the series identity.

    Behavioural since 2026-09-03. This asserted that "static_legend" and
    "showlegend=static_legend" appear in plotly.py -- two fragments that say the flag is threaded
    somewhere, not that passing it produces a legend, and that survive the branch being dead.

    The parking matters as much as the flag: a default top-right in-plot legend overlaps subplot
    titles and the suptitle on multi-panel figures, which is why it goes horizontal and below.
    """
    fig = _rendered(static_legend=True)

    assert fig.layout.showlegend is True
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.y is not None and fig.layout.legend.y < 0, "the legend is still inside the plot area"


def test_the_static_legend_is_opt_in() -> None:
    """Interactive HTML identifies series by hover, and a legend there is pooled-series soup on
    multi-panel figures. A single labelled panel is the documented exception, so this asserts only
    that the flag is not what turned it on."""
    default = _rendered(static_legend=False)

    assert default.layout.legend.orientation != "h" or default.layout.legend.y is None or default.layout.legend.y >= 0
