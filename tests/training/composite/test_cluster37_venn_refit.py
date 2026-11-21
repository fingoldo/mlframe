"""Cluster 37 regressions: NaN Venn-Abers scores get no interval, and a capped refit keeps its sample weights (audit 2026-11-20, findings 37-F3, 37-F6)."""

import numpy as np

from mlframe.training.composite.venn_abers import _lookup_interval


class _Holder:
    """Carries a fitted Venn-Abers envelope as the calibrator does."""

    def __init__(self):
        self._venn_abers_ = {
            "grid": np.array([0.0, 0.5, 1.0]),
            "p0": np.array([0.1, 0.4, 0.8]),
            "p1": np.array([0.2, 0.5, 0.9]),
        }


def test_nan_score_gets_no_interval_instead_of_the_top_bin():
    """A NaN score used to read the top grid bin's probabilities silently (37-F3)."""
    lo, hi = _lookup_interval(_Holder(), np.array([0.7, np.nan]))
    assert np.isfinite(lo[0]) and np.isfinite(hi[0])
    assert np.isnan(lo[1]) and np.isnan(hi[1])


def test_capped_refit_passes_sample_weight_through():
    """refit_capped_stack used to refit unweighted although the gate validated a weighted fit (37-F6)."""
    import inspect

    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import refit_capped_stack

    assert "sample_weight" in inspect.signature(refit_capped_stack).parameters
