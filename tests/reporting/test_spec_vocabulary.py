"""New LinePanelSpec / AnnotationPanelSpec vocabulary + sampling helper + INV-4 show().

Every additive spec field (vlines, vspans, markers / lines+markers line_styles, band, x_is_time,
AnnotationPanelSpec) must render on BOTH backends without error. Also pins
``subsample_preserving_extremes`` determinism + all-extremes-present + size cap, and the
MatplotlibRenderer.show() IPython-display path (INV-4) under a faked __IPYTHON__ / IPython.display.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mlframe.reporting.charts._sampling import _finite_argmin_argmax, subsample_preserving_extremes
from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec


BACKENDS = ["matplotlib", "plotly"]


def _render_both(panel):
    """Helper: Render both."""
    spec = FigureSpec(panels=((panel,),), figsize=(6, 4))
    for backend in BACKENDS:
        fig = get_renderer(backend).render(spec)
        assert fig is not None
    return spec


# ----------------------------------------------------------------------------
# New spec vocabulary renders on both backends.
# ----------------------------------------------------------------------------


class TestSpecVocabularyRenders:
    """Groups tests for: TestSpecVocabularyRenders."""
    def test_vlines(self):
        """Vlines."""
        x = np.arange(20)
        _render_both(LinePanelSpec(x=x, y=x.astype(float), vlines=((5.0, "red", "split"), (12.0, "gray", ""))))

    def test_vspans(self):
        """Vspans."""
        x = np.arange(20)
        _render_both(LinePanelSpec(x=x, y=x.astype(float), vspans=((3.0, 7.0, "orange", 0.2), (10.0, 15.0, "green", 0.15))))

    def test_marker_only_series(self):
        """Marker only series."""
        x = np.arange(20)
        _render_both(LinePanelSpec(x=x, y=(x.astype(float), x.astype(float) * 0.5), line_styles=("markers", "-"), series_labels=("observed", "fit")))

    def test_lines_plus_markers(self):
        """Lines plus markers."""
        x = np.arange(20)
        _render_both(LinePanelSpec(x=x, y=x.astype(float), line_styles=("lines+markers",)))

    def test_band(self):
        """Band."""
        x = np.arange(20).astype(float)
        center = np.sin(x / 3.0)
        _render_both(LinePanelSpec(x=x, y=center, band=(center - 0.3, center + 0.3), band_label="+-std"))

    def test_x_is_time(self):
        """X is time."""
        x = np.arange(20).astype(float)
        _render_both(LinePanelSpec(x=x, y=x, x_is_time=True))

    def test_annotation_panel(self):
        """Annotation panel."""
        _render_both(AnnotationPanelSpec(text="metric unavailable\n(K<3)", title="PIT"))

    def test_marker_series_plotly_mode(self):
        """A markers-only series must emit a markers-mode trace on plotly (not lines)."""
        x = np.arange(20)
        spec = FigureSpec(panels=((LinePanelSpec(x=x, y=x.astype(float), line_styles=("markers",)),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        modes = [t.mode for t in fig.data if t.type in ("scatter", "scattergl")]
        assert "markers" in modes

    def test_annotation_panel_text_present_matplotlib(self):
        """Annotation panel text present matplotlib."""
        spec = FigureSpec(panels=((AnnotationPanelSpec(text="unavailable"),),), figsize=(4, 3))
        fig = get_renderer("matplotlib").render(spec)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert "unavailable" in texts


# ----------------------------------------------------------------------------
# subsample_preserving_extremes — determinism, all-extremes-present, size cap.
# ----------------------------------------------------------------------------


class TestSubsamplePreservingExtremes:
    """Groups tests for: TestSubsamplePreservingExtremes."""
    def test_returns_all_indices_when_small(self):
        """Returns all indices when small."""
        x = np.arange(100)
        idx = subsample_preserving_extremes(x, sample_size=500)
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_deterministic_default_seed(self):
        """Deterministic default seed."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100_000)
        y = rng.standard_normal(100_000)
        a = subsample_preserving_extremes(x, y, sample_size=5_000)
        b = subsample_preserving_extremes(x, y, sample_size=5_000)
        np.testing.assert_array_equal(a, b)

    def test_size_cap_respected(self):
        """Size cap respected."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100_000)
        idx = subsample_preserving_extremes(x, sample_size=5_000)
        assert len(idx) <= 5_000

    def test_sorted_ascending_and_unique(self):
        """Sorted ascending and unique."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(50_000)
        idx = subsample_preserving_extremes(x, sample_size=2_000)
        assert np.all(np.diff(idx) > 0), "indices must be strictly ascending (sorted + unique)"

    def test_argmin_argmax_always_included(self):
        """Argmin argmax always included."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(100_000)
        y = rng.standard_normal(100_000)
        idx = subsample_preserving_extremes(x, y, sample_size=5_000)
        s = set(idx.tolist())
        for arr in (x, y):
            assert int(np.argmin(arr)) in s
            assert int(np.argmax(arr)) in s

    def test_top_k_extreme_values_included(self):
        """The k largest-|extreme_values| indices (e.g. MaxError residuals) survive the subsample."""
        rng = np.random.default_rng(4)
        n = 100_000
        x = rng.standard_normal(n)
        resid = rng.standard_normal(n)
        # Plant 10 clearly-largest |residual| points.
        planted = rng.choice(n, size=10, replace=False)
        resid[planted] = 1e6 + np.arange(10)
        idx = subsample_preserving_extremes(x, sample_size=5_000, extreme_values=resid, k_extremes=10)
        s = set(idx.tolist())
        assert all(p in s for p in planted), "the k largest-|residual| points must be kept"

    def test_nan_extreme_values_do_not_displace_real_extremes(self):
        """Nan extreme values do not displace real extremes."""
        rng = np.random.default_rng(5)
        n = 50_000
        x = rng.standard_normal(n)
        resid = rng.standard_normal(n)
        resid[:100] = np.nan  # NaN must NOT be treated as the top extreme
        planted = rng.choice(np.arange(100, n), size=5, replace=False)
        resid[planted] = 1e6
        idx = subsample_preserving_extremes(x, sample_size=2_000, extreme_values=resid, k_extremes=5)
        s = set(idx.tolist())
        assert all(p in s for p in planted)

    def test_mismatched_lengths_raise(self):
        """Mismatched lengths raise."""
        with pytest.raises(ValueError, match="same length"):
            subsample_preserving_extremes(np.arange(10), np.arange(9), sample_size=5)

    def test_no_arrays_raise(self):
        """No arrays raise."""
        with pytest.raises(ValueError, match="at least one array"):
            subsample_preserving_extremes(sample_size=5)

    def test_nonpositive_sample_size_raises(self):
        """Nonpositive sample size raises."""
        with pytest.raises(ValueError, match="positive"):
            subsample_preserving_extremes(np.arange(10), sample_size=0)


class TestFiniteArgMinMaxParity:
    """The plain-argmin/argmax fast path must equal the nan-aware reference on every NaN configuration."""

    @staticmethod
    def _reference(arr):
        """Helper: Reference."""
        if np.issubdtype(arr.dtype, np.floating):
            try:
                return [int(np.nanargmin(arr)), int(np.nanargmax(arr))]
            except ValueError:
                return []
        return [int(np.argmin(arr)), int(np.argmax(arr))]

    def test_no_nan(self):
        """No nan."""
        x = np.random.default_rng(0).standard_normal(10_000)
        assert _finite_argmin_argmax(x) == self._reference(x)

    def test_nan_scattered(self):
        """Nan scattered."""
        x = np.random.default_rng(1).standard_normal(10_000)
        x[::97] = np.nan
        assert _finite_argmin_argmax(x) == self._reference(x)

    def test_nan_at_min_position(self):
        """Nan at min position."""
        x = np.random.default_rng(2).standard_normal(10_000)
        x[int(np.argmin(x))] = np.nan  # NaN exactly at the plain-argmin index
        assert _finite_argmin_argmax(x) == self._reference(x)

    def test_all_nan_returns_empty(self):
        """All nan returns empty."""
        x = np.full(100, np.nan)
        assert _finite_argmin_argmax(x) == []

    def test_int_dtype(self):
        """Int dtype."""
        x = np.random.default_rng(3).integers(-50, 50, 10_000)
        assert _finite_argmin_argmax(x) == self._reference(x)


# ----------------------------------------------------------------------------
# INV-4 — MatplotlibRenderer.show() uses IPython.display under a kernel.
# ----------------------------------------------------------------------------


class TestMatplotlibShowIPython:
    """Groups tests for: TestMatplotlibShowIPython."""
    def test_show_triggers_display_under_faked_ipython(self, monkeypatch):
        """Under a faked IPython kernel, show() routes to IPython.display.display(fig) -- never plt.figure(.number)
        which would raise (the renderer builds figures without a pyplot manager)."""
        displayed = []

        # Fake IPython module with get_ipython() returning a truthy shell, plus IPython.display.display.
        fake_ipython = types.ModuleType("IPython")
        fake_ipython.get_ipython = lambda: object()
        fake_display_mod = types.ModuleType("IPython.display")
        fake_display_mod.display = lambda fig: displayed.append(fig)
        fake_ipython.display = fake_display_mod

        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display_mod)

        from mlframe.reporting.spec import ScatterPanelSpec

        spec = FigureSpec(panels=((ScatterPanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0])),),), figsize=(4, 3))
        renderer = get_renderer("matplotlib")
        fig = renderer.render(spec)
        renderer.show(fig)
        assert len(displayed) == 1 and displayed[0] is fig

    def test_show_no_ipython_is_non_raising(self, monkeypatch):
        """Outside a kernel show() must not raise even with no display (headless)."""
        monkeypatch.delitem(sys.modules, "IPython", raising=False)
        from mlframe.reporting.spec import ScatterPanelSpec

        spec = FigureSpec(panels=((ScatterPanelSpec(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0])),),), figsize=(4, 3))
        renderer = get_renderer("matplotlib")
        fig = renderer.render(spec)
        renderer.show(fig)  # must return cleanly
