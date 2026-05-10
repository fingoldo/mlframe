"""A/B bench: sequential vs ThreadPoolExecutor render_and_save across
matplotlib + plotly backends on the canonical regression scatter spec.

Wave 9 c0089 profile attributed 28.4 s to render_and_save on 34 calls
(836 ms / call), split as matplotlib (618 ms / call) + plotly (218 ms /
call) in series. Both backends release the GIL during their heavy
C-level work (Agg rasterization, JSON serialization) and use isolated
per-Figure state, so running them in parallel collapses the per-call
wall to ``max(matplotlib, plotly) = 618 ms``, saving ~7 s on c0089.

Usage:
    python -m mlframe.profiling.bench_parallel_render
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time

import numpy as np

from mlframe.reporting.charts.regression import build_regression_panel_spec
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers.base import get_renderer
from mlframe.reporting.renderers.save import render_and_save


def _render_and_save_sequential(spec, output, base_path):
    """Pre-Wave-9 sequential variant kept inline so the bench reports
    a fair OLD vs NEW delta."""
    multi_output = (len(output.backends) > 1) or any(
        len(fmts) > 1 for _, fmts in output.backends
    )
    for backend, fmts in output.backends:
        renderer = get_renderer(backend)
        fig = renderer.render(spec)
        for fmt in fmts:
            if multi_output:
                path = f"{base_path}.{backend}.{fmt}"
            else:
                path = f"{base_path}.{fmt}"
            renderer.save(fig, path, fmt)
        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass


class _DummyAudit:
    """Duck-typed ResidualAudit stand-in for the bench."""
    mean = 0.0
    std = 1.0
    skew = 0.1
    excess_kurt = 0.2
    hypothesis = "Gaussian noise"
    suggested_loss = "L2"
    hetero_significant = False
    hetero_spearman = 0.05


def main() -> None:
    rng = np.random.default_rng(42)
    N = 1_000_000
    y_true = rng.normal(size=N).astype(np.float64)
    y_pred = y_true + rng.normal(0, 0.3, N)

    spec = build_regression_panel_spec(
        y_true, y_pred,
        audit=_DummyAudit(),
        header_str="bench header",
        metrics_str="MAE=0.24 RMSE=0.30 R2=0.91",
    )
    output_dual = parse_plot_output_dsl("plotly[html] + matplotlib[png]")

    def _save_parallel():
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "bench")
            render_and_save(spec, output_dual, base, interactive=False)

    def _save_sequential():
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "bench")
            _render_and_save_sequential(spec, output_dual, base)

    # Warm both paths (matplotlib font cache, plotly's first call).
    _save_parallel()
    _save_sequential()

    def bench(fn, label, n_repeat=5):
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        m = statistics.mean(times)
        s = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  {label:<55} {m*1000:>7.1f} ms +/- {s*1000:>5.1f} ms")

    print(f"# regression scatter spec, N={N:_}, plotly[html]+matplotlib[png]")
    bench(_save_sequential, "OLD: sequential (matplotlib -> plotly)")
    bench(_save_parallel,   "NEW: parallel (ThreadPoolExecutor)")


if __name__ == "__main__":
    main()
