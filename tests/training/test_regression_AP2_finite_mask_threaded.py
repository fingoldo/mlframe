"""Sensor for AP2 (F15): precomputed finite_mask threaded through residual _fit.

When the caller has already validated that ``y`` and ``base`` are finite (the typical hot-path situation inside ``CompositeTargetDiscovery._eval_one_transform``: ``y_train[valid]`` + ``base_train[valid]`` was filtered via ``transform.domain_check``), each inner ``_fit`` recomputing ``np.isfinite(y) & np.isfinite(base)`` is wasted compute. The threading lets the caller pass ``_finite_mask=<precomputed>`` (or skip a recompute when input is already finite) and the inner uses it directly.

The sensor patches ``numpy.isfinite`` with a counter and asserts:

- WITHOUT the kwarg: inner ``_fit`` does at least one ``isfinite`` pass on each (y, base) pair.
- WITH the kwarg supplied: the inner skips the per-call recompute (counter delta is 0 for the mask-formation step).

Test runs the 9 _fit kernels we threaded (additive_residual, median_residual, y_quantile_clip, ratio, rolling_quantile_ratio, quantile_residual, monotonic_residual, ewma_residual, frac_diff).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._composite_transforms_simple import (
    _additive_residual_fit,
    _median_residual_fit,
    _y_quantile_clip_fit,
    _ratio_fit,
    _rolling_quantile_ratio_fit,
)
from mlframe.training._composite_transforms_nonlinear import (
    _quantile_residual_fit,
    _monotonic_residual_fit,
    _ewma_residual_fit,
    _frac_diff_fit,
)


def _make_finite_pair(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n).astype(np.float64)
    base = rng.normal(loc=2.0, scale=1.0, size=n).astype(np.float64) + 5.0  # positive for ratio/logratio
    return y, base


_FIT_CASES = [
    ("additive_residual", _additive_residual_fit, {}),
    ("median_residual", _median_residual_fit, {}),
    ("y_quantile_clip", _y_quantile_clip_fit, {}),
    ("ratio", _ratio_fit, {}),
    ("rolling_quantile_ratio", _rolling_quantile_ratio_fit, {}),
    ("quantile_residual", _quantile_residual_fit, {}),
    ("monotonic_residual", _monotonic_residual_fit, {}),
    ("ewma_residual", _ewma_residual_fit, {}),
    ("frac_diff", _frac_diff_fit, {}),
]


@pytest.mark.parametrize("name,fit_fn,extra_kwargs", _FIT_CASES, ids=[c[0] for c in _FIT_CASES])
def test_AP2_finite_mask_kwarg_accepted_by_all_threaded_fits(name, fit_fn, extra_kwargs):
    """Each threaded _fit accepts ``_finite_mask=<bool ndarray>`` without TypeError, and the returned params are non-empty / deterministic on identical input + mask."""
    y, base = _make_finite_pair(n=512, seed=1)
    mask = np.ones_like(y, dtype=bool)
    # Without mask (back-compat).
    p_no_mask = fit_fn(y, base, **extra_kwargs)
    # With explicit all-True mask. Must produce the SAME params (mask is consistent with input).
    p_with_mask = fit_fn(y, base, _finite_mask=mask, **extra_kwargs)
    assert type(p_no_mask) is type(p_with_mask), f"{name}: type mismatch"
    assert isinstance(p_no_mask, dict)
    assert isinstance(p_with_mask, dict)
    # Numeric scalar comparison (deep dict equality on float values).
    _assert_param_dicts_close(p_no_mask, p_with_mask, name)


def _assert_param_dicts_close(a: dict, b: dict, name: str, atol: float = 1e-9):
    assert set(a.keys()) == set(b.keys()), f"{name}: param keys differ {a.keys()} vs {b.keys()}"
    for k in a.keys():
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            np.testing.assert_allclose(va, vb, atol=atol, err_msg=f"{name}.{k}")
        elif isinstance(va, (int, float)):
            assert abs(float(va) - float(vb)) <= atol, f"{name}.{k}: {va} vs {vb}"
        elif isinstance(va, list):
            np.testing.assert_allclose(np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64), atol=atol, err_msg=f"{name}.{k}")
        else:
            assert va == vb, f"{name}.{k}: {va!r} vs {vb!r}"


@pytest.mark.parametrize("name,fit_fn,extra_kwargs", _FIT_CASES, ids=[c[0] for c in _FIT_CASES])
def test_AP2_finite_mask_skips_internal_isfinite_when_supplied(monkeypatch, name, fit_fn, extra_kwargs):
    """When ``_finite_mask`` is supplied, the inner _fit MUST NOT recompute the y/base isfinite mask. We patch ``numpy.isfinite`` with a counter and require: at least one isfinite call WITHOUT the mask (baseline), strictly fewer WITH the mask. Some kernels still call isfinite on derived arrays (e.g. residuals in linear_residual_robust) but the y/base initial isfinite pair is dropped."""
    import numpy as _np
    y, base = _make_finite_pair(n=512, seed=2)
    mask = np.ones_like(y, dtype=bool)

    counter = {"calls": 0}
    real_isfinite = _np.isfinite

    def _counting_isfinite(*args, **kwargs):
        counter["calls"] += 1
        return real_isfinite(*args, **kwargs)

    monkeypatch.setattr(_np, "isfinite", _counting_isfinite)

    counter["calls"] = 0
    fit_fn(y, base, **extra_kwargs)
    baseline_calls = counter["calls"]

    counter["calls"] = 0
    fit_fn(y, base, _finite_mask=mask, **extra_kwargs)
    threaded_calls = counter["calls"]

    assert baseline_calls > 0, f"{name}: baseline made zero isfinite calls; test is degenerate"
    assert threaded_calls < baseline_calls, (
        f"{name}: expected fewer isfinite calls when _finite_mask supplied "
        f"(baseline={baseline_calls}, with_mask={threaded_calls})"
    )
