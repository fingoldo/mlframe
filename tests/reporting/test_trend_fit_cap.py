"""Visual-equivalence + cost contract for the robust trend-line fit cap.

``robust_fit_endpoints`` fits a Theil-Sen line as a VISUAL GUIDE on the pred-vs-actual overlay. Its cost is bounded by
``_TREND_FIT_CAP`` (points fed to the fit). Theil-Sen's slope is dominated by its ``max_subpopulation`` stochastic pair
sample, not by how many rows those pairs are drawn from, so the drawn line is essentially cap-insensitive above a few
thousand points. These tests pin that contract so a future cap change (up or down) cannot silently move the rendered
line beyond a small fraction of the y-range, and that the cap actually bounds the fit input.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers._trend import robust_fit_endpoints, _TREND_FIT_CAP


def test_trend_fit_cap_is_bounded():
    """The cap must stay small enough to keep the Theil-Sen fit cheap on multi-million-row clouds (the fit was one of
    the largest single reporting costs at ~786ms/call before the cap was lowered)."""
    assert _TREND_FIT_CAP <= 5_000, f"_TREND_FIT_CAP={_TREND_FIT_CAP} too large; Theil-Sen fit cost is ~linear in it"


def test_trend_line_visually_equivalent_to_full_data_fit():
    """The capped fit's endpoints must stay within a small fraction of the y-range of a fit on many more points, across
    diverse heteroscedastic + outlier-laden clouds -- i.e. the cost cut does not visibly move the drawn line."""
    worst_rel = 0.0
    for s in range(15):
        r = np.random.default_rng(s)
        n = int(r.integers(20_000, 80_000))
        x = r.standard_normal(n) * r.uniform(0.5, 3.0)
        y = r.uniform(0.3, 1.5) * x + r.uniform(-1.0, 1.0) + r.standard_normal(n) * r.uniform(0.2, 0.8)
        n_out = int(n * r.uniform(0.0, 0.05))
        if n_out:
            y[:n_out] += r.standard_normal(n_out) * r.uniform(3.0, 10.0)

        # Reference: fit on a much larger sample than the production cap by monkeying the cap up for one call.
        import mlframe.reporting.renderers._trend as _t

        cap_prod = _t._TREND_FIT_CAP
        (_, ylo_c), (_, yhi_c) = robust_fit_endpoints(x, y, "theil-sen")
        _t._TREND_FIT_CAP = 20_000
        try:
            (_, ylo_r), (_, yhi_r) = robust_fit_endpoints(x, y, "theil-sen")
        finally:
            _t._TREND_FIT_CAP = cap_prod
        y_range = max(float(y.max() - y.min()), 1e-9)
        worst_rel = max(worst_rel, abs(ylo_c - ylo_r) / y_range, abs(yhi_c - yhi_r) / y_range)

    assert worst_rel <= 0.03, (
        f"capped trend endpoints shifted {worst_rel:.2%} of the y-range vs a 20k-point fit; the visual guide moved "
        f"more than the ~2% Theil-Sen sampling-variance budget"
    )


def test_trend_endpoints_anchor_x_range():
    """Endpoints must always span the true [x_min, x_max] (the cap subsampling keeps the extremes)."""
    r = np.random.default_rng(0)
    x = r.standard_normal(50_000)
    y = 0.9 * x + r.standard_normal(50_000) * 0.3
    (xlo, _), (xhi, _) = robust_fit_endpoints(x, y, "theil-sen")
    assert xlo == pytest.approx(float(x.min()))
    assert xhi == pytest.approx(float(x.max()))
