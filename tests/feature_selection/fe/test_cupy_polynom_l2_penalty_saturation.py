"""The device polynomial search must apply ``l2_penalty_saturation`` exactly like the numba kernel does."""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.feature_selection.filters import _cupy_polynom_optimizer as cpo


def _numpy_as_cupy() -> types.SimpleNamespace:
    """A numpy namespace standing in for cupy, with ``asnumpy`` as identity."""
    ns = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    ns.asnumpy = np.asarray
    return ns


@pytest.mark.parametrize("saturation", [0.25, 1.0, 4.0])
def test_device_penalty_uses_the_requested_saturation(monkeypatch: pytest.MonkeyPatch, saturation: float) -> None:
    """With zero MI the best score is minus the penalty, so it pins the penalty formula."""
    from mlframe.feature_selection.filters import _fe_batched_mi

    monkeypatch.setattr(cpo, "_bf_apply_checked", lambda cp, k, HA, HB: (HA + HB, np.ones(HA.shape[1], dtype=bool)))
    monkeypatch.setattr(cpo, "_rank_bin_batched_gpu", lambda cp, C, n_bins: C)
    monkeypatch.setattr(_fe_batched_mi, "binned_mi_from_codes_gpu", lambda codes, *a, **k: np.zeros(codes.shape[1]))
    Ca = np.array([[1.0, 2.0], [0.5, 0.5]])
    Cb = np.array([[0.0, 1.0], [2.0, 0.0]])
    Ba = np.ones((5, 2))
    Bb = np.ones((5, 2))
    scores, _, _ = cpo._score_generation_gpu(_numpy_as_cupy(), Ba, Bb, Ca, Cb, np.zeros(5, dtype=np.int64), 2, 4, 0.5, False, ["add"], saturation)
    s = (Ca * Ca).sum(axis=1) + (Cb * Cb).sum(axis=1)
    np.testing.assert_allclose(scores, -(0.5 * s / (s + saturation)))


def test_device_penalty_is_linear_for_non_positive_saturation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive saturation means the raw ``l2 * s`` penalty, like ``_l2_penalty_njit``."""
    from mlframe.feature_selection.filters import _fe_batched_mi

    monkeypatch.setattr(cpo, "_bf_apply_checked", lambda cp, k, HA, HB: (HA + HB, np.ones(HA.shape[1], dtype=bool)))
    monkeypatch.setattr(cpo, "_rank_bin_batched_gpu", lambda cp, C, n_bins: C)
    monkeypatch.setattr(_fe_batched_mi, "binned_mi_from_codes_gpu", lambda codes, *a, **k: np.zeros(codes.shape[1]))
    Ca = np.array([[1.0, 2.0]])
    Cb = np.array([[0.0, 1.0]])
    scores, _, _ = cpo._score_generation_gpu(
        _numpy_as_cupy(), np.ones((5, 2)), np.ones((5, 2)), Ca, Cb, np.zeros(5, dtype=np.int64), 2, 4, 0.5, False, ["add"], 0.0
    )
    np.testing.assert_allclose(scores, -0.5 * ((Ca * Ca).sum(axis=1) + (Cb * Cb).sum(axis=1)))
