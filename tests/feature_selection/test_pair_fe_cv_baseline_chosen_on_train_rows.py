"""The trivial baseline in validate_pair_fe_cv is chosen on train rows and scored on held-out rows."""

import numpy as np

import mlframe.feature_selection.filters.composition as comp
import mlframe.feature_selection.filters.fe_baselines as fb


def test_the_baseline_is_selected_on_train_rows(monkeypatch):
    """Selecting it on the held-out rows made the honest-uplift ratio divide by a winner's-curse maximum."""
    chosen_on = {}
    real = fb.best_trivial_pair

    def spy(x_a, x_b, y, **kw):
        chosen_on["n"] = len(y)
        return real(x_a, x_b, y, **kw)

    monkeypatch.setattr(fb, "best_trivial_pair", spy)
    rng = np.random.default_rng(0)
    x_a, x_b = rng.normal(size=100), rng.normal(size=100)
    y = (x_a + x_b > 0).astype(np.int64)
    tr, va = np.arange(80), np.arange(80, 100)
    mi = comp._heldout_trivial_mi(x_a, x_b, y, tr, va, discrete_target=True, mi_estimator="plugin", plugin_n_bins=10)
    assert chosen_on["n"] == 80, "the baseline must be chosen on the 80 train rows"
    assert np.isfinite(mi) and mi >= 0.0
