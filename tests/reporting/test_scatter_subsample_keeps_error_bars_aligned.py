"""A scatter subsampled past the cap keeps its error bars aligned with the points (audit 2026-11-20, finding 21-F4)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.renderers._matplotlib_scatter import _SCATTER_MAX_POINTS, _scatter
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import ScatterPanelSpec


def test_error_bars_follow_a_subsampled_scatter():
    """Past the point cap, y_err (symmetric) and an asymmetric (lower, upper) pair must be sliced with the points, not left at full length."""
    rng = np.random.default_rng(0)
    n = _SCATTER_MAX_POINTS + 2000
    x = rng.standard_normal(n)
    y = x + rng.standard_normal(n)
    err = np.full(n, 0.1)
    for spec in (
        ScatterPanelSpec(x=x, y=y, y_err=err),
        ScatterPanelSpec(x=x, y=y, y_err=(err, err)),
    ):
        fig, ax = plt.subplots()
        _scatter(MatplotlibRenderer(), ax, spec, fig)
        plt.close(fig)
