"""W2-B after-fix bench: plotly histogram + scatter renderer paths at n=2M.

Re-runs the two PERF-4 / PERF-5 renderer hazards measured in bench_viz.py (before-numbers in performance.md:
hist 6.65s / 37.30 MB, scatter 14.60s / 73.08 MB at 2M) now that the renderer pre-bins histograms above
50k and Scattergl-downsamples scatters above 50k. Also a cProfile pass on each 2M render+save.

Run from repo root:  python "audit/viz_audit_2026_06_10/bench_viz_w2b_after.py"
ASCII-only output (Windows cp1251). MPLBACKEND=Agg.
"""
import os
import sys
import time
import cProfile
import pstats
import io

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "bench_out")
os.makedirs(OUT_DIR, exist_ok=True)


def fsize_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None


def make_regression(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y_true = 3.0 * x + rng.normal(scale=1.0, size=n)
    y_pred = 3.0 * x + rng.normal(scale=0.7, size=n)
    return y_true.astype(np.float64), y_pred.astype(np.float64)


def bench_hist(n):
    from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    y_true, y_pred = make_regression(n)
    resid = y_true - y_pred
    spec = FigureSpec(panels=((HistogramPanelSpec(values=resid, bins=50, title="resid",
                                                  overlay_normal=(0.0, float(resid.std()))),),),
                      figsize=(8, 5))
    r = PlotlyRenderer()
    t0 = time.perf_counter()
    fig = r.render(spec)
    path = os.path.join(OUT_DIR, f"w2b_hist_{n}.html")
    r.save(fig, path, "html")
    t = time.perf_counter() - t0
    return t, fsize_mb(path), path, spec, r


def bench_scatter(n):
    from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec
    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    y_true, y_pred = make_regression(n)
    spec = FigureSpec(panels=((ScatterPanelSpec(x=y_pred, y=y_true, title="scatter",
                                                perfect_fit_line=True),),), figsize=(8, 5))
    r = PlotlyRenderer()
    t0 = time.perf_counter()
    fig = r.render(spec)
    path = os.path.join(OUT_DIR, f"w2b_scatter_{n}.html")
    r.save(fig, path, "html")
    t = time.perf_counter() - t0
    return t, fsize_mb(path), path, spec, r


def profile_render_save(spec, renderer, fmt, path, label):
    pr = cProfile.Profile()
    pr.enable()
    fig = renderer.render(spec)
    renderer.save(fig, path, fmt)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(f"\n===== cProfile: {label} =====")
    print(s.getvalue())
    sys.stdout.flush()


def main():
    n = 2_000_000
    print("W2-B after-fix bench, python", sys.version.split()[0], "MPLBACKEND=", os.environ.get("MPLBACKEND"))

    th, mbh, ph, spec_h, r_h = bench_hist(n)
    print(f"plotly histogram render+save @2M: {th:8.3f}s  html={mbh:.3f}MB  (before: 6.650s / 37.30 MB)")
    sys.stdout.flush()

    ts, mbs, ps_, spec_s, r_s = bench_scatter(n)
    print(f"plotly scatter   render+save @2M: {ts:8.3f}s  html={mbs:.3f}MB  (before: 14.603s / 73.08 MB)")
    sys.stdout.flush()

    profile_render_save(spec_h, r_h, "html", os.path.join(OUT_DIR, "w2b_hist_prof.html"),
                        "2M histogram render+save")
    profile_render_save(spec_s, r_s, "html", os.path.join(OUT_DIR, "w2b_scatter_prof.html"),
                        "2M scatter render+save")


if __name__ == "__main__":
    main()
