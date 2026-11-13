"""The host wavelet leg selection rebuilds only the admitted legs instead of keeping every surviving leg array."""
import numpy as np

from mlframe.feature_selection.filters import _wavelet_basis_fe as W


def _data(n=6000, seed=0):
    """A feature with one localised band that drives the label."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = ((x > 0.55) & (x < 0.7)).astype(int) ^ (rng.random(n) < 0.05)
    return x, y


def test_arrays_match_the_tuple_form_and_the_leg_builder():
    """``return_arrays=True`` admits the same legs as the tuple form and each array is the leg builder's output."""
    x, y = _data()
    lo, span = float(x.min()), float(x.max() - x.min())
    plain = W._select_wavelet_legs(x, y, lo, span)
    with_arrays = W._select_wavelet_legs(x, y, lo, span, return_arrays=True)
    assert plain and [(j, k) for j, k, _ in with_arrays] == plain
    z = np.clip((x - lo) / span, 0.0, 1.0)
    for j, k, leg in with_arrays:
        assert np.array_equal(leg, W._dyadic_haar_leg(z, j, k))


def test_leg_builder_runs_once_per_candidate_plus_once_per_admitted_leg(monkeypatch):
    """No per-leg array is cached: the builder is called for each candidate and again only for the admitted legs."""
    x, y = _data()
    lo, span = float(x.min()), float(x.max() - x.min())
    calls = []
    real = W._dyadic_haar_leg
    monkeypatch.setattr(W, "_dyadic_haar_leg", lambda *a, **k: calls.append(1) or real(*a, **k))
    n_plain = len(W._select_wavelet_legs(x, y, lo, span)) and len(calls)
    calls.clear()
    admitted = W._select_wavelet_legs(x, y, lo, span, return_arrays=True)
    assert len(calls) == n_plain + len(admitted)
