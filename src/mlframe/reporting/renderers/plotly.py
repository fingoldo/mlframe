"""plotly renderer.

Builds a plotly ``go.Figure`` from a ``FigureSpec``. Multi-panel figures
use ``plotly.subplots.make_subplots`` with row_heights / column_widths
matching the matplotlib gridspec.

Save formats:
- ``html``: ``write_html`` (interactive, includes plotly.js)
- ``json``: ``to_json`` (data + layout, embed-friendly)
- ``png/svg/pdf``: ``write_image`` (requires kaleido package; falls back to
  html with WARN if missing)
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Tuple

import numpy as np

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec,
    HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec, ScatterPanelSpec,
    ViolinPanelSpec,
)

logger = logging.getLogger(__name__)

# Renderer-level safety nets for specs carrying raw large-n data. Builders are expected to
# pre-sample / pre-bin, but the renderer is public API: above these thresholds a raw spec would
# embed n values into the HTML (37 MB / 73 MB per panel at 2M, browser-freezing).
_HIST_PREBIN_THRESHOLD = 50_000
_SCATTER_MAX_POINTS = 50_000
# WebGL traces render large scatters orders of magnitude faster than SVG-mode go.Scatter.
_SCATTER_WEBGL_THRESHOLD = 10_000
_SCATTER_DOWNSAMPLE_WARNED = False


def _warn_scatter_downsample(n: int) -> None:
    global _SCATTER_DOWNSAMPLE_WARNED
    if not _SCATTER_DOWNSAMPLE_WARNED:
        logger.warning(
            "[plotly-render] scatter panel carries %d raw points; downsampled to %d "
            "(extremes preserved) to keep the figure responsive. Pre-sample at the spec "
            "builder to silence this. Fires once per process.",
            n, _SCATTER_MAX_POINTS,
        )
        _SCATTER_DOWNSAMPLE_WARNED = True


# Process-singleton: track whether the persistent kaleido sync server
# is up so we don't pay the ~10-15s Chromium-spawn cost on every PNG /
# SVG / PDF write. Verified empirically (2026-05-08) that kaleido 1.x
# default ``fig.write_image()`` calls spawn a fresh Chromium process
# per call (~13s each); persistent server reuses one process and drops
# subsequent calls to ~0.13s. On c0114 (lgb+xgb multiclass, 100k rows,
# 32 PNG calls), this saved 32 * ~13s = ~7 minutes of wall time.
_KALEIDO_SERVER_STARTED = False

# 2026-05-08 v2 hardening: process-wide flag that the persistent path
# has burned itself on a JS error / hang. Once True, all subsequent
# saves in this process go straight to HTML fallback (skipping kaleido
# entirely because plotly's oneshot also routes through the SAME
# broken sync server -- verified empirically on c0031). Reset only on
# interpreter exit.
_KALEIDO_PERSISTENT_BURNED = False

# Count of consecutive timeouts/errors. Burn after this many.
# v3 (single-burn-on-first-failure) was paying 12 x 30s timeouts in
# c0031 because each timeout fires once before "burning" via the
# next-call path. Decrement: cap at 2 consecutive failures (~60s
# wasted before HTML fallback takes over). Single-failure runs (rare
# transient) still get a retry; pathological combos burn fast.
_KALEIDO_PERSISTENT_FAIL_COUNT = 0
_KALEIDO_PERSISTENT_FAIL_THRESHOLD = 2

# 2026-05-10: idempotency guard for the "Failed to start kaleido sync
# server" warning. Pre-2026-05-10, the warn fired on EVERY call when
# the kaleido binary lacked ``start_sync_server`` (e.g. kaleido 0.x
# wheels) -- 32+ times per suite call on plot-heavy regression suites,
# polluting the log. Once True we suppress repeats; the suite-end
# wall-share log will still surface the cumulative oneshot-fallback
# cost so the user notices the missing fast path.
_KALEIDO_START_WARN_EMITTED = False
# Counter of fallback PNG/SVG/PDF writes that took the slow oneshot
# path. Reported in the suite-end summary so the reader sees ROI for
# upgrading kaleido.
_KALEIDO_ONESHOT_CALL_COUNT = 0
_KALEIDO_ONESHOT_TOTAL_WALL_S = 0.0


def get_kaleido_oneshot_stats() -> Tuple[int, float]:
    """Returns (n_oneshot_calls, total_wall_seconds) so suite-end
    reporting can quote concrete numbers. Cleared via ``reset_kaleido_oneshot_stats``."""
    return _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S


def reset_kaleido_oneshot_stats() -> None:
    global _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S
    _KALEIDO_ONESHOT_CALL_COUNT = 0
    _KALEIDO_ONESHOT_TOTAL_WALL_S = 0.0


def record_kaleido_oneshot_call(wall_s: float) -> None:
    global _KALEIDO_ONESHOT_CALL_COUNT, _KALEIDO_ONESHOT_TOTAL_WALL_S
    _KALEIDO_ONESHOT_CALL_COUNT += 1
    _KALEIDO_ONESHOT_TOTAL_WALL_S += wall_s

# Hard ceiling on a single persistent write_fig_sync call. Beyond this,
# the call is treated as hung; we leave the worker thread to die on
# process exit (the kaleido server holds an asyncio loop so we can't
# safely cancel it from outside). Empirical normal cost after warmup:
# 0.13s/call. Cold persistent warmup: ~8s. 30s is well above both,
# while still bounding c0031-style hangs to 30s instead of infinity.
_KALEIDO_PERSISTENT_TIMEOUT_S = 30.0


def _is_kaleido_persistent_burned() -> bool:
    return _KALEIDO_PERSISTENT_BURNED


def _record_kaleido_persistent_failure() -> bool:
    """Increment failure counter; return True if we just crossed the
    threshold (caller should burn the persistent path)."""
    global _KALEIDO_PERSISTENT_FAIL_COUNT, _KALEIDO_PERSISTENT_BURNED
    _KALEIDO_PERSISTENT_FAIL_COUNT += 1
    if _KALEIDO_PERSISTENT_FAIL_COUNT >= _KALEIDO_PERSISTENT_FAIL_THRESHOLD:
        _KALEIDO_PERSISTENT_BURNED = True
        return True
    return False


def _mark_kaleido_persistent_burned() -> None:
    """Force-burn the persistent path (legacy entry, kept for tests
    and external callers)."""
    global _KALEIDO_PERSISTENT_BURNED
    _KALEIDO_PERSISTENT_BURNED = True


def _restart_kaleido_server() -> bool:
    """Stop + restart the kaleido sync server. Used after a JS error
    poisons the async task chain so subsequent calls don't deadlock.
    Idempotent / no-op when the server isn't started.

    2026-05-11: a successful restart clears the persistent-failure
    counter AND the burned flag. The counter exists to catch
    "this process's kaleido is fundamentally broken" -- a clean
    restart proves otherwise. Without this, prior failures
    accumulated across two unrelated callsites (e.g. two separate
    test_kaleido_recovery tests) would cross the burned threshold
    and force HTML fallback forever.
    """
    global _KALEIDO_SERVER_STARTED
    global _KALEIDO_PERSISTENT_FAIL_COUNT, _KALEIDO_PERSISTENT_BURNED
    try:
        import kaleido
    except ImportError:
        return False
    if _KALEIDO_SERVER_STARTED:
        try:
            kaleido.stop_sync_server(silence_warnings=True)
        except Exception:
            pass
        _KALEIDO_SERVER_STARTED = False
    started = _ensure_kaleido_server_started()
    if started:
        # Successful restart: the persistent path is now usable again.
        # Clear cumulative-failure state so callers don't keep burning
        # after a recoverable hiccup.
        _KALEIDO_PERSISTENT_FAIL_COUNT = 0
        _KALEIDO_PERSISTENT_BURNED = False
    return started


def _ensure_kaleido_server_started() -> bool:
    """Start the kaleido sync server (idempotent). Returns True if the
    server is up after the call, False if kaleido isn't installed.
    """
    global _KALEIDO_SERVER_STARTED
    if _KALEIDO_SERVER_STARTED:
        return True
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False
    try:
        # silence_warnings=True so the "already started" message doesn't
        # spam logs if some other caller already started the server.
        kaleido.start_sync_server(silence_warnings=True)
        _KALEIDO_SERVER_STARTED = True
        # Register cleanup so the Chromium subprocess gets a clean exit
        # rather than the "Resorting to unclean kill browser" warning
        # at interpreter shutdown.
        import atexit
        def _stop():
            try:
                kaleido.stop_sync_server(silence_warnings=True)
            except Exception:
                pass
        atexit.register(_stop)
        return True
    except Exception as e:
        global _KALEIDO_START_WARN_EMITTED
        if not _KALEIDO_START_WARN_EMITTED:
            logger.warning(
                "[plotly-render] kaleido sync server unavailable (%s); will "
                "use the slower per-call oneshot path. This warning fires "
                "ONCE per process; check the suite-end wall-share summary "
                "for cumulative oneshot cost. To enable the fast path, "
                "upgrade kaleido (>=1.x ships ``start_sync_server``).", e,
            )
            _KALEIDO_START_WARN_EMITTED = True
        return False


class PlotlyRenderer:
    backend = "plotly"

    def render(self, spec: FigureSpec, *, static_legend: bool = False) -> Any:
        """Build a plotly figure from the spec.

        ``static_legend`` enables a figure-level legend. The interactive HTML output identifies series via
        hover tooltips, so legends stay off there; a STATIC export (png/svg/pdf) has no hover, so when the
        save-format set includes one the caller passes ``static_legend=True`` to make the export readable.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = len(spec.panels)
        cols = max((len(r) for r in spec.panels), default=0)
        if rows == 0 or cols == 0:
            raise ValueError("FigureSpec has no panels")

        # Per-panel subplot spec: heatmap needs no shared axes; default
        # ``xy`` works for everything else.
        sub_specs: List[List[dict]] = []
        for r, row in enumerate(spec.panels):
            row_specs: List[dict] = []
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    row_specs.append({})  # empty cell
                else:
                    row_specs.append({"type": "xy"})
            sub_specs.append(row_specs)

        # Build subplot titles list (row-major). Plotly's subplot_titles
        # are HTML annotations — newlines need ``<br>`` instead of ``\n``,
        # otherwise the linebreak character is dropped and 2-line panel
        # titles render as one long string that bleeds into adjacent
        # subplots horizontally (visible bug 2026-05-09 on regression
        # chart with hypothesis + heteroscedasticity multiline titles).
        subplot_titles = []
        for row in spec.panels:
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    subplot_titles.append("")
                else:
                    raw_title = getattr(row[c], "title", "") or ""
                    subplot_titles.append(raw_title.replace("\n", "<br>"))

        subplots_kwargs = dict(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=spec.sharex,
            shared_yaxes=spec.sharey,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )
        if spec.row_height_ratios is not None:
            total = sum(spec.row_height_ratios)
            subplots_kwargs["row_heights"] = [r / total for r in spec.row_height_ratios]
        if spec.col_width_ratios is not None:
            total = sum(spec.col_width_ratios)
            subplots_kwargs["column_widths"] = [c / total for c in spec.col_width_ratios]

        fig = make_subplots(**subplots_kwargs)

        # 2026-05-09: shrink subplot-title font from plotly default (16)
        # to match matplotlib's ~11. With three columns at the typical
        # (18in, 4in) figsize, default-16 titles overflow horizontally
        # into adjacent subplots (e.g. regression chart left-panel title
        # "MAE=... R2=..." bleeds into middle-panel "Residuals (skew=...)").
        for ann in fig.layout.annotations:
            ann.font = dict(size=11)

        for r, row in enumerate(spec.panels, start=1):
            for c in range(1, cols + 1):
                if c - 1 >= len(row) or row[c - 1] is None:
                    continue
                self._render_panel(fig, row[c - 1], r, c)

        # Figure-level layout.
        if spec.suptitle:
            fig.update_layout(title=dict(text=spec.suptitle,
                                         font=dict(size=spec.suptitle_fontsize)))
        fig.update_layout(
            width=int(spec.figsize[0] * 80),   # ~80px per matplotlib inch
            height=int(spec.figsize[1] * 80),
            margin=dict(l=60, r=40, t=80 if spec.suptitle else 50, b=50),
            # Interactive HTML identifies series via hover tooltips, so the legend defaults off there to avoid
            # the multi-subplot soup (precision/recall/F1 mixed with reliability lines). A static export has no
            # hover; the caller flips ``static_legend`` when a png/svg/pdf is in the save set so it stays readable.
            showlegend=static_legend,
        )
        if static_legend:
            fig.update_layout(legend=dict(font=dict(size=9), itemsizing="constant",
                                          bgcolor="rgba(255,255,255,0.6)"))
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        fmt = fmt.lower()
        if fmt == "html":
            fig.write_html(path, include_plotlyjs="cdn", auto_open=False)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(fig.to_json())
        elif fmt in ("png", "svg", "pdf"):
            # Persistent-server fast path: write_fig_sync reuses one
            # Chromium subprocess across all calls in the process,
            # dropping per-call cost from ~13s (oneshot) to ~0.13s.
            # Falls through to plotly's default write_image when the
            # sync server can't be started (kaleido missing, etc.).
            #
            # 2026-05-08 brittleness fix: a single figure that triggers
            # a JS error inside kaleido (e.g. "Error 525: Cannot read
            # properties of undefined (reading 'v')") asyncio-cancels
            # the persistent server's task chain and leaves
            # ``write_fig_sync`` blocked on ``await asyncio.gather``
            # FOREVER -- killing the whole suite. Real reproducer:
            # c0031_c58ed4fc (lgb+xgb+hgb multiclass + recency
            # weights). Recovery strategy:
            #   1. Try persistent server.
            #   2. On any kaleido / asyncio error, stop+restart the
            #      sync server (clears the broken async-task chain).
            #   3. Retry once via ONESHOT path (slower but isolated).
            #   4. If that also fails, fall back to HTML so the suite
            #      doesn't lose the diagnostic entirely.
            #
            # 2026-05-08 hardening v2: even with restart-on-error (above),
            # certain figure shapes (multiclass + recency in c0031)
            # produce JS errors that DON'T raise from write_fig_sync --
            # they cancel the asyncio task chain INSIDE the server, but
            # write_fig_sync waits forever on the queue. Mitigation: run
            # write_fig_sync in a worker thread with a hard timeout. If
            # it doesn't return in PERSISTENT_TIMEOUT_SECONDS, mark the
            # persistent path "burned" for the rest of the process and
            # forever-after use oneshot. We also do NOT retry the
            # persistent path on failures within ONE process -- once
            # any call fails, every later call goes oneshot directly.
            persistent_failed = _is_kaleido_persistent_burned()
            # ``server_hung`` distinguishes "persistent server is dead /
            # hung" (timeout — skip oneshot retry, go straight to HTML
            # because plotly's oneshot would re-enter the same dead queue)
            # from "single persistent call raised an exception" (the
            # server is probably still alive; oneshot retry may succeed).
            # Pre-2026-05-11 the code treated both the same and went to
            # HTML on any persistent failure, which surfaced as the
            # test_kaleido_persistent_failure_falls_back_to_oneshot
            # regression — the test injects a synthetic exception into
            # ``write_fig_sync`` (not a hang), so the oneshot retry SHOULD
            # produce a PNG. Now: exception -> restart server -> oneshot;
            # timeout -> HTML directly.
            server_hung = False
            burned_now = False
            if not persistent_failed and _ensure_kaleido_server_started():
                import kaleido as _kal
                import threading

                _result: list = [None]
                _exc: list = [None]

                def _do_persistent():
                    try:
                        _kal.write_fig_sync(fig, path, opts={"format": fmt})
                    except Exception as ee:
                        _exc[0] = ee

                th = threading.Thread(target=_do_persistent, daemon=True)
                th.start()
                th.join(timeout=_KALEIDO_PERSISTENT_TIMEOUT_S)
                if th.is_alive():
                    persistent_failed = True
                    server_hung = True
                    burned_now = _record_kaleido_persistent_failure()
                    logger.warning(
                        "kaleido persistent write_fig_sync(%s) did not "
                        "return in %.0fs%s.",
                        fmt, _KALEIDO_PERSISTENT_TIMEOUT_S,
                        "; persistent path BURNED for this process -- subsequent "
                        "saves write HTML directly" if burned_now else
                        " (will retry persistent up to threshold before burning)",
                    )
                    # Don't try to restart the server -- a hung server may
                    # have stop_sync_server() ALSO blocking. Just leave it
                    # behind; it'll get cleaned up at process exit.
                elif _exc[0] is not None:
                    persistent_failed = True
                    burned_now = _record_kaleido_persistent_failure()
                    logger.warning(
                        "kaleido persistent save(%s) raised %s%s.",
                        fmt, type(_exc[0]).__name__,
                        "; persistent path BURNED" if burned_now else
                        " (will retry up to threshold)",
                    )
                    # 2026-05-11: server is likely still alive but its async
                    # task chain may be poisoned -- restart it so the
                    # oneshot retry below uses a clean queue. Best-effort;
                    # if restart hangs we'll fall through to the HTML
                    # fallback anyway via the burned-or-hung gate below.
                    try:
                        _restart_kaleido_server()
                    except Exception:
                        pass
                else:
                    return  # success

            # 2026-05-08 v3: when the persistent server is HUNG or BURNED,
            # ``fig.write_image`` routes through plotly's wrapper which
            # talks to the SAME broken kaleido sync server. Verified
            # empirically: c0031 v4 hung past 30 minutes because the
            # oneshot fallback re-entered the same dead queue. In those
            # cases write HTML directly -- a different code path
            # (write_html doesn't touch kaleido) that always works.
            #
            # 2026-05-11 split: a single ``write_fig_sync`` exception is
            # NOT proof that the server is dead; after restarting the
            # sync server (see ``_restart_kaleido_server`` call above),
            # the oneshot retry can succeed. Only go straight to HTML
            # when the server is genuinely hung (timeout) or has crossed
            # the burned threshold.
            if persistent_failed and (server_hung or burned_now or _is_kaleido_persistent_burned()):
                from os.path import splitext
                root, _ = splitext(path)
                try:
                    fig.write_html(root + ".html", include_plotlyjs="cdn", auto_open=False)
                    logger.warning(
                        "kaleido burned -- wrote interactive HTML instead of %s "
                        "to %s (PNG/SVG/PDF unavailable for the rest of this "
                        "process; restart Python to retry persistent kaleido).",
                        fmt, root + ".html",
                    )
                except Exception as e:
                    logger.error(
                        "All save paths failed for %s (%s); diagnostic chart "
                        "lost but suite continues. Last error: %s",
                        path, fmt, e,
                    )
                return

            # Persistent path skipped (kaleido never started or unavailable);
            # use plotly oneshot. Catch ALL exceptions for HTML fallback.
            # 2026-05-10: instrument oneshot wall-time + call count so the
            # suite-end summary surfaces the cumulative cost (e.g. on the
            # 2026-05-09 prod log: 32 oneshots x ~13s = 7+ minutes that
            # disappear when kaleido is upgraded to a build with
            # ``start_sync_server``).
            import time as _time
            _t0 = _time.time()
            try:
                try:
                    fig.write_image(path, format=fmt)
                except Exception as e:
                    logger.warning(
                        "plotly write_image(%s) oneshot failed (%s: %s); "
                        "falling back to .html.",
                        fmt, type(e).__name__, e,
                    )
                    from os.path import splitext
                    root, _ = splitext(path)
                    try:
                        fig.write_html(root + ".html", include_plotlyjs="cdn", auto_open=False)
                    except Exception as e2:
                        logger.error(
                            "All save paths failed for %s (%s); diagnostic chart "
                            "lost but suite continues. Last error: %s",
                            path, fmt, e2,
                        )
            finally:
                # Record cumulative oneshot wall regardless of success /
                # exception path; the suite-end summary uses this to
                # quote concrete savings of upgrading kaleido.
                record_kaleido_oneshot_call(_time.time() - _t0)
        else:
            raise ValueError(
                f"plotly doesn't support format {fmt!r}; "
                "supported: html/png/svg/pdf/json"
            )

    def show(self, fig: Any) -> None:
        try:
            fig.show()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, fig, panel, row: int, col: int) -> None:
        if isinstance(panel, ScatterPanelSpec):
            self._scatter(fig, panel, row, col)
        elif isinstance(panel, HistogramPanelSpec):
            self._histogram(fig, panel, row, col)
        elif isinstance(panel, HeatmapPanelSpec):
            self._heatmap(fig, panel, row, col)
        elif isinstance(panel, BarPanelSpec):
            self._bar(fig, panel, row, col)
        elif isinstance(panel, LinePanelSpec):
            self._line(fig, panel, row, col)
        elif isinstance(panel, ViolinPanelSpec):
            self._violin(fig, panel, row, col)
        elif isinstance(panel, NetworkPanelSpec):
            self._network(fig, panel, row, col)
        elif isinstance(panel, AnnotationPanelSpec):
            self._annotation(fig, panel, row, col)
        else:
            raise TypeError(f"unknown panel type: {type(panel).__name__}")

    def _annotation(self, fig, p: AnnotationPanelSpec, row: int, col: int) -> None:
        fig.add_annotation(text=p.text.replace("\n", "<br>"), x=0.5, y=0.5,
                           xref="x domain", yref="y domain", showarrow=False,
                           font=dict(size=p.fontsize), align="center",
                           row=row, col=col)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def _scatter(self, fig, p: ScatterPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        x = np.asarray(p.x)
        y = np.asarray(p.y)
        n = len(x)

        # Per-point size / color arrays must follow the SAME row subset as x/y when downsampling.
        size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
        color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

        if n > _SCATTER_MAX_POINTS:
            _warn_scatter_downsample(n)
            from mlframe.reporting.charts._sampling import subsample_preserving_extremes
            idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
            x, y = x[idx], y[idx]
            if size_arr is not None and len(size_arr) == n:
                size_arr = size_arr[idx]
            if color_arr is not None and len(color_arr) == n:
                color_arr = color_arr[idx]

        marker = dict(opacity=p.point_alpha)
        # ScatterPanelSpec.point_size follows matplotlib's ``s=`` (area in pt^2); plotly marker.size is the
        # DIAMETER in px. Without conversion large mpl areas blow up to giant circles and the auto-axis range
        # goes haywire. Conversion: plotly_diameter_px = sqrt(mpl_area_pt2) * 1.33.
        if size_arr is not None:
            marker["size"] = (np.sqrt(np.maximum(np.asarray(size_arr, dtype=float), 0.0)) * 1.33).tolist()
        else:
            marker["size"] = float(math.sqrt(max(float(p.point_size), 0.0)) * 1.33)
        if color_arr is not None:
            marker["color"] = np.asarray(color_arr)
            marker["colorscale"] = _mpl_to_plotly_cmap(p.colormap)
            marker["showscale"] = bool(p.colorbar_label)
            if p.colorbar_label:
                marker["colorbar"] = dict(title=p.colorbar_label)
        elif p.point_color is not None:
            marker["color"] = p.point_color

        # Hover labels for inline_labels (population annotations). Only valid when no downsample reordered rows.
        text = None
        if p.inline_labels and len(p.inline_labels) == n and n <= _SCATTER_MAX_POINTS:
            text = [t[2] for t in p.inline_labels]

        # WebGL renders large scatters orders of magnitude faster than SVG-mode go.Scatter; ndarrays pass
        # through to plotly natively (faster + smaller than .tolist()).
        trace_cls = go.Scattergl if n > _SCATTER_WEBGL_THRESHOLD else go.Scatter
        fig.add_trace(
            trace_cls(x=x, y=y,
                      mode="markers+text" if text else "markers",
                      marker=marker,
                      text=text,
                      textposition="top center" if text else None,
                      textfont=dict(size=8),
                      name=p.legend_label or "",
                      showlegend=bool(p.legend_label)),
            row=row, col=col,
        )

        if p.perfect_fit_line and n > 0:
            # Span the y=x line over the UNION of both axes so it stays the true diagonal even when prediction
            # collapse (constant y) makes the y-range a single point; scaleanchor squares the panel so y=x is 45deg.
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            fig.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi],
                           mode="lines",
                           line=dict(color="green", dash="dash"),
                           name="Perfect fit", showlegend=True),
                row=row, col=col,
            )
            fig.update_yaxes(scaleanchor=_axis_ref(fig, row, col), scaleratio=1.0,
                             range=[lo, hi], row=row, col=col)
            fig.update_xaxes(range=[lo, hi], row=row, col=col)

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _histogram(self, fig, p: HistogramPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        # ``overlay_x_lo/hi`` anchors the Normal-overlay grid. When we pre-bin (here or upstream) they come from
        # the bin EDGES, avoiding two extra full-n min/max passes over raw values (PERF-18).
        overlay_x_lo = overlay_x_hi = None
        bin_centers = p.bin_centers
        heights = None
        if bin_centers is None and len(np.asarray(p.values)) > _HIST_PREBIN_THRESHOLD:
            # Raw spec with n above the embed-hazard ceiling: bin once with numpy instead of shipping n values
            # into the HTML (37 MB / browser-freezing at 2M).
            from mlframe.reporting.charts._sampling import prebin_histogram
            heights, centers, width0 = prebin_histogram(np.asarray(p.values), p.bins, p.density)
            if heights is not None:
                bin_centers = centers

        if bin_centers is not None:
            if heights is None:
                heights = np.asarray(p.values)
                width = float(p.bin_width or (
                    (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0))
            else:
                width = float(width0)
            colors_kw = dict(color=p.color)
            if p.bar_colors is not None:
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = dict(
                    color=np.asarray(p.bar_colors),
                    colorscale=_mpl_to_plotly_cmap(p.colormap),
                )
            fig.add_trace(
                go.Bar(x=np.asarray(bin_centers), y=np.asarray(heights),
                       width=width,
                       marker=dict(line=dict(color="white", width=0.5), **colors_kw),
                       showlegend=False),
                row=row, col=col,
            )
            if len(bin_centers) > 0:
                overlay_x_lo = float(bin_centers[0] - width / 2.0)
                overlay_x_hi = float(bin_centers[-1] + width / 2.0)
        else:
            fig.add_trace(
                go.Histogram(x=np.asarray(p.values),
                             nbinsx=p.bins,
                             histnorm="probability density" if p.density else "",
                             marker=dict(color=p.color, line=dict(color="white", width=0.4)),
                             opacity=0.6, showlegend=False),
                row=row, col=col,
            )

        if p.overlay_normal is not None:
            mu, sigma = p.overlay_normal
            if sigma > 0:
                if overlay_x_lo is None:
                    vals = np.asarray(p.values)
                    overlay_x_lo, overlay_x_hi = float(np.min(vals)), float(np.max(vals))
                x_grid = np.linspace(overlay_x_lo, overlay_x_hi, 200)
                normal_pdf = (
                    1 / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                )
                label = p.overlay_label or f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                fig.add_trace(
                    go.Scatter(x=x_grid, y=normal_pdf,
                               mode="lines",
                               line=dict(color="red", dash="dash", width=1.4),
                               name=label, showlegend=True),
                    row=row, col=col,
                )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid,
                         type="log" if p.yscale == "log" else "linear")

    def _heatmap(self, fig, p: HeatmapPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        fig.add_trace(
            go.Heatmap(z=p.matrix.tolist(),
                       x=list(p.col_labels), y=list(p.row_labels),
                       colorscale=_mpl_to_plotly_cmap(p.colormap),
                       colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None,
                       showscale=True),
            row=row, col=col,
        )
        # 2026-05-09: per-cell text via add_annotation instead of
        # plotly's built-in ``text`` + ``texttemplate`` (which uses
        # one global font color and produces white-on-yellow
        # invisibility on viridis high-end / RdYlBu high-end). Per-cell
        # ``auto_text_color`` flips by perceived luminance.
        if p.cell_text is not None:
            from mlframe.reporting.colors import auto_text_color
            mat = p.matrix
            vmin = float(np.nanmin(mat))
            vmax = float(np.nanmax(mat))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    text_color = auto_text_color(
                        float(mat[i, j]), p.colormap, vmin=vmin, vmax=vmax,
                    )
                    fig.add_annotation(
                        text=format(p.cell_text[i, j], p.text_format),
                        x=p.col_labels[j], y=p.row_labels[i],
                        showarrow=False,
                        font=dict(color=text_color, size=10),
                        row=row, col=col,
                    )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-45)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col,
                         autorange="reversed")  # match matplotlib top-to-bottom row order

    def _bar(self, fig, p: BarPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        from mlframe.reporting.colors import line_color
        if isinstance(p.values, tuple):
            for i, series in enumerate(p.values):
                lbl = p.series_labels[i] if p.series_labels else f"series {i}"
                # 2026-05-09: default plotly's qualitative palette
                # ('Plotly': bright violet/red/green) is too saturated
                # and clashes with matplotlib's tab10 in the same
                # figure. Fall back to ``line_color(i)`` (tab10) when
                # the spec doesn't pin colors -- cross-backend parity
                # with matplotlib's bar default.
                if p.colors is not None and i < len(p.colors):
                    color = p.colors[i]
                else:
                    color = line_color(i)
                col_kw = {"marker": dict(color=color)}
                fig.add_trace(
                    go.Bar(x=list(p.categories), y=series.tolist(),
                           name=lbl, showlegend=True, **col_kw),
                    row=row, col=col,
                )
            fig.update_layout(barmode="group")
        else:
            col_kw = dict(marker=dict(color=p.colors[0] if p.colors else "steelblue"))
            fig.add_trace(
                go.Bar(x=list(p.categories),
                       y=np.asarray(p.values).tolist(),
                       showlegend=False, **col_kw),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-p.xtick_rotation if p.xtick_rotation else 0)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _line(self, fig, p: LinePanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors or tuple(line_color(i) for i in range(len(ys)))
        x = np.asarray(p.x) if isinstance(p.x, np.ndarray) else p.x
        # matplotlib linestyle tokens -> plotly dash; "markers" / "lines+markers" select the trace mode.
        _STYLE_MAP = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

        if p.band is not None:
            lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
            band_color = p.band_color or cols[0]
            fig.add_trace(
                go.Scatter(x=np.concatenate([x, x[::-1]]),
                           y=np.concatenate([upper, lower[::-1]]),
                           fill="toself", fillcolor=_rgba(band_color, 0.2),
                           line=dict(width=0), hoverinfo="skip",
                           name=p.band_label or "band", showlegend=bool(p.band_label)),
                row=row, col=col,
            )

        for i, y in enumerate(ys):
            token = styles[i % len(styles)]
            if token == "markers":
                mode, dash = "markers", "solid"
            elif token == "lines+markers":
                mode, dash = "lines+markers", "solid"
            else:
                mode, dash = "lines", _STYLE_MAP.get(token, "solid")
            yv = np.asarray(y) if isinstance(y, np.ndarray) else y
            fig.add_trace(
                go.Scatter(x=x, y=yv,
                           mode=mode,
                           line=dict(color=cols[i % len(cols)], dash=dash),
                           marker=dict(color=cols[i % len(cols)], size=5),
                           name=labels[i] if i < len(labels) else None,
                           showlegend=any(labels)),
                row=row, col=col,
            )

        for vx0, vx1, vcolor, valpha in (p.vspans or ()):
            fig.add_vrect(x0=vx0, x1=vx1, fillcolor=_rgba(vcolor, valpha),
                          line_width=0, layer="below", row=row, col=col)
        for vx, vcolor, vlabel in (p.vlines or ()):
            fig.add_vline(x=vx, line=dict(color=vcolor, dash="dot", width=1.2),
                          annotation_text=vlabel or None, annotation_position="top",
                          row=row, col=col)

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid,
                         tickangle=-30 if p.x_is_time else 0)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _violin(self, fig, p: ViolinPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import line_color

        for i, group in enumerate(p.groups):
            # tab10 cycle for cross-backend parity (plotly default
            # 'Plotly' qualitative is over-saturated next to mpl bars).
            color = line_color(i)
            fig.add_trace(
                go.Violin(y=np.asarray(group).tolist(),
                          name=p.group_labels[i],
                          box_visible=p.show_box,
                          meanline_visible=False,
                          line_color=color,
                          fillcolor=color,
                          opacity=0.6,
                          showlegend=False),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-30)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    # Cap on directed-edge arrow annotations. Each arrow is one layout
    # annotation; beyond this the topology still renders (lines + nodes) but
    # arrowheads are skipped so a large opt-in graph doesn't bloat the layout.
    _NETWORK_MAX_ARROWS = 500

    def _network(self, fig, p: NetworkPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        node_x = np.asarray(p.node_x, dtype=float)
        node_y = np.asarray(p.node_y, dtype=float)
        e_src = np.asarray(p.edge_src, dtype=np.int64)
        e_dst = np.asarray(p.edge_dst, dtype=np.int64)
        weights = np.asarray(p.edge_weight, dtype=float)

        if e_src.size:
            wmin, wmax = float(weights.min()), float(weights.max())
            wspan = (wmax - wmin) or 1.0
            lo, hi = p.edge_width_range
            colorscale = _mpl_to_plotly_cmap(p.colormap)
            # Bin edges by MI into a handful of width/color buckets: one Scattergl
            # line trace per non-empty bucket keeps trace count O(bins) regardless
            # of edge count (a single line trace can't vary width/color per segment).
            n_bins = min(8, max(1, e_src.size))
            bin_idx = np.minimum(((weights - wmin) / wspan * n_bins).astype(int), n_bins - 1)
            from plotly.colors import sample_colorscale
            for b in range(n_bins):
                mask = bin_idx == b
                if not mask.any():
                    continue
                frac = (b + 0.5) / n_bins
                width = lo + frac * (hi - lo)
                color = sample_colorscale(colorscale, [frac])[0]
                xs: List = []
                ys: List = []
                for a, d in zip(e_src[mask], e_dst[mask]):
                    xs.extend([node_x[a], node_x[d], None])
                    ys.extend([node_y[a], node_y[d], None])
                fig.add_trace(
                    go.Scattergl(x=xs, y=ys, mode="lines",
                                 line=dict(width=width, color=color),
                                 hoverinfo="skip", showlegend=False),
                    row=row, col=col,
                )

            # Invisible marker trace at edge midpoints carries the continuous MI
            # colorbar and a per-edge hover readout without cluttering the plot.
            mid_x = (node_x[e_src] + node_x[e_dst]) / 2.0
            mid_y = (node_y[e_src] + node_y[e_dst]) / 2.0
            fig.add_trace(
                go.Scattergl(
                    x=mid_x.tolist(), y=mid_y.tolist(), mode="markers",
                    marker=dict(size=0.1, color=weights.tolist(), colorscale=colorscale,
                                showscale=True,
                                colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None),
                    text=[f"MI={w:.4f}" for w in weights],
                    hoverinfo="text", showlegend=False),
                row=row, col=col,
            )

            # Directed-edge arrowheads via data-space annotations. Axis refs are
            # derived from the subplot grid so multi-panel figures stay correct;
            # any failure falls back to no arrows (lines already convey topology).
            directed = p.edge_directed
            if np.isscalar(directed):
                directed = np.full(e_src.shape, bool(directed))
            else:
                directed = np.asarray(directed, dtype=bool)
            if directed.any() and int(directed.sum()) <= self._NETWORK_MAX_ARROWS:
                try:
                    n_cols = len(fig._grid_ref[0])
                    idx = (row - 1) * n_cols + col
                    suffix = "" if idx == 1 else str(idx)
                    xref, yref = f"x{suffix}", f"y{suffix}"
                    for a, d, dirn in zip(e_src, e_dst, directed):
                        if dirn:
                            fig.add_annotation(
                                x=node_x[d], y=node_y[d], ax=node_x[a], ay=node_y[a],
                                xref=xref, yref=yref, axref=xref, ayref=yref,
                                showarrow=True, arrowhead=2, arrowsize=1.2,
                                arrowwidth=1.0, arrowcolor="rgba(80,80,80,0.6)",
                                standoff=6, startstandoff=6,
                            )
                except Exception:
                    logger.debug("network arrows skipped (subplot axis-ref resolution failed)",
                                 exc_info=True)

        # Nodes: one marker trace. size follows matplotlib ``scatter(s=)`` area
        # semantics; convert to plotly's pixel diameter (sqrt(area) * 1.33).
        sizes = np.sqrt(np.maximum(np.asarray(p.node_size, dtype=float), 0.0)) * 1.33
        hovertext = list(p.node_hovertext) if p.node_hovertext else list(p.node_label)
        fig.add_trace(
            go.Scattergl(
                x=node_x.tolist(), y=node_y.tolist(),
                mode="markers+text",
                marker=dict(size=sizes.tolist(), color=list(p.node_color),
                            line=dict(width=0.5, color="black")),
                text=list(p.node_label), textposition="top center", textfont=dict(size=8),
                hovertext=hovertext, hoverinfo="text", showlegend=False),
            row=row, col=col,
        )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col,
                         showgrid=False, zeroline=False, showticklabels=False)


# Map matplotlib colormap names to plotly's named colorscales (best-effort).
# plotly supports many of the standard mpl names directly via lowercase.
_MPL_TO_PLOTLY = {
    "RdYlBu": "RdYlBu",
    "RdYlBu_r": "RdYlBu_r",
    "RdBu": "RdBu",
    "RdBu_r": "RdBu_r",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "Blues": "Blues",
    "Greens": "Greens",
}


def _axis_ref(fig, row: int, col: int) -> str:
    """x-axis reference string for the subplot at (row, col), e.g. ``"x"`` / ``"x4"`` — for scaleanchor."""
    try:
        n_cols = len(fig._grid_ref[0])
        idx = (row - 1) * n_cols + col
    except Exception:
        idx = 1
    return "x" if idx == 1 else f"x{idx}"


def _rgba(color: str, alpha: float) -> str:
    """Best-effort named/hex color -> rgba() string with the given alpha; leaves rgb()/rgba() as-is."""
    c = str(color)
    if c.startswith("rgba(") or c.startswith("rgb("):
        return c
    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(c)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"
    except Exception:
        return c


def _mpl_to_plotly_cmap(name: str) -> str:
    """Map a matplotlib colormap name to a plotly colorscale name.

    Falls back to 'Viridis' for unknown names with a one-time WARN.
    """
    if name in _MPL_TO_PLOTLY:
        return _MPL_TO_PLOTLY[name]
    logger.warning(
        "Unknown colormap %r; falling back to plotly 'Viridis'. "
        "Add the mapping to mlframe.reporting.renderers.plotly._MPL_TO_PLOTLY "
        "to silence this warning.",
        name,
    )
    return "Viridis"


__all__ = ["PlotlyRenderer"]
