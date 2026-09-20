"""Every ICE path must bin the same way, so the metric that drives early stopping is the metric in the report.

``fast_ice_only`` was pinned to uniform bins while ``fast_calibration_report`` resolves ``"auto"`` to equal-population
bins below a 10% base rate, under a docstring calling the two bit-exact: on a 4.6%-positive bed they returned -0.3114
and -0.3442. The batched per-class kernel bins uniformly and cannot express the quantile grid, so the metric must fall
back to the per-class loop whenever the shared resolver asks for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error
from mlframe.metrics.calibration import resolve_binning_strategy
from mlframe.metrics.classification import fast_calibration_report
from mlframe.metrics.core import fast_ice_only


def _bed(kind: str, n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.random(n) if kind == "balanced" else np.clip(rng.beta(1.0, 22.0, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p).astype(np.int8)
    return np.ascontiguousarray(y), np.ascontiguousarray(p)


@pytest.mark.parametrize("kind", ["balanced", "rare"])
def test_fast_ice_only_matches_the_report(kind):
    y, p = _bed(kind)
    report_ice = float(fast_calibration_report(y_true=y, y_pred=p, show_plots=False, plot_file=None).ice)
    assert float(fast_ice_only(y, p)) == pytest.approx(report_ice, rel=0, abs=0)


def test_rare_class_actually_takes_the_quantile_grid():
    """Pins that the two grids differ on this bed, so the test above is not vacuous."""
    y, p = _bed("rare")
    assert resolve_binning_strategy(y, "auto") == "quantile"
    assert float(fast_ice_only(y, p, binning_strategy="uniform")) != pytest.approx(float(fast_ice_only(y, p)), rel=1e-6)


@pytest.mark.parametrize("kind, kernel_expected", [("balanced", True), ("rare", False)])
def test_batched_uniform_kernel_is_skipped_when_bins_must_be_quantile(monkeypatch, kind, kernel_expected):
    import mlframe.metrics._ice_metric as ice_mod

    used = []
    original = ice_mod._ice_kernel_dispatch

    def _spy(*args, **kwargs):
        used.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ice_mod, "_ice_kernel_dispatch", _spy)
    y, p = _bed(kind)
    compute_probabilistic_multiclass_error(y_true=y, y_score=np.stack([1.0 - p, p], axis=1), method="multicrit")
    assert bool(used) is kernel_expected
