"""Registry-driven property legs every transform must satisfy; a new registry entry is covered automatically.

The max-error round trip across scale, offset and size lives in ``test_composite_transforms_registry_contract.py``. This
module adds the legs that contract does not reach: degenerate inputs (a constant base, a constant y, missing group
labels), an over-restrictive y domain, params that grow with the training rows, and two names sharing one function
triple.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

from .test_composite_transforms_registry_contract import _base_for, _call_domain, _call_fit, _call_forward, _call_inverse, _grid_data

_NAMES = sorted(TRANSFORMS_REGISTRY)


def _groups(n: int) -> np.ndarray:
    """Four contiguous integer groups."""
    return (np.arange(n) * 4 // n).astype(np.int64)


def _fit_on_domain(t, name: str, y: np.ndarray, base: np.ndarray, base2: np.ndarray, groups: np.ndarray):
    """Params fitted on the rows ``domain_check`` accepts, with the domain mask and the base the transform takes."""
    b = _base_for(name, base, base2)
    domain = np.asarray(_call_domain(t, y, b), dtype=bool)
    return _call_fit(t, y[domain], b[domain] if t.requires_base else b, groups[domain]), domain, b


@pytest.mark.parametrize("name", [n for n in _NAMES if TRANSFORMS_REGISTRY[n].requires_base])
def test_a_constant_train_base_keeps_the_inverse_near_the_train_level(name: str):
    """Fitted on a constant base, the mean-T inverse at a base 10% off stays within 25% of the train y level.

    A constant base carries no information about y, so nothing fitted from it may amplify a small predict-time base move:
    ``volatility_normalized_residual`` divided by an absolute 1e-12 volatility floor and returned y ~ 2.5e12 against a
    train mean of 100. Fixed-form transforms (``1/y - 1/base``) move with the base by construction, within this band.
    """
    t = TRANSFORMS_REGISTRY[name]
    n = 300
    y = 100.0 + np.random.default_rng(0).standard_normal(n)
    g = _groups(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, domain, b = _fit_on_domain(t, name, y, np.full(n, 50.0), np.full(n, 20.0), g)
        t_mean = float(np.nanmean(_call_forward(t, y[domain], b[domain], params, g[domain])))
        for factor in (0.9, 1.0, 1.1):
            b_pred = _base_for(name, np.full(n, 50.0 * factor), np.full(n, 20.0 * factor))
            y_hat = _call_inverse(t, np.full(n, t_mean), b_pred, params, g)
            assert np.all(np.isfinite(y_hat)), f"{name}: non-finite inverse at base x{factor} after a constant-base fit"
            dev = float(np.max(np.abs(y_hat - 100.0)))
            assert dev <= 25.0, f"{name}: base x{factor} after a constant-base fit moved y_hat {dev:.4g} from the train level 100"


@pytest.mark.parametrize("name", _NAMES)
def test_a_constant_y_inverts_a_small_perturbation_back_to_the_constant(name: str):
    """Fitted on a constant y, ``inverse(forward(y) + 1e-6)`` stays at the constant: a zero spread must not blow a perturbation up."""
    t = TRANSFORMS_REGISTRY[name]
    n = 300
    rng = np.random.default_rng(0)
    y = np.full(n, 7.0)
    g = _groups(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, domain, b = _fit_on_domain(t, name, y, np.linspace(1.0, 10.0, n), rng.uniform(1.0, 5.0, n), g)
        b_fit = b[domain] if t.requires_base else b
        y_back = _call_inverse(t, _call_forward(t, y[domain], b_fit, params, g[domain]) + 1e-6, b_fit, params, g[domain])
    assert np.all(np.isfinite(y_back)), f"{name}: non-finite inverse on a constant y"
    err = float(np.max(np.abs(y_back - 7.0))) if y_back.size else 0.0
    assert err <= 1e-3, f"{name}: a 1e-6 perturbation of T moved a constant y by {err:.4g}"


@pytest.mark.parametrize("name", [n for n in _NAMES if TRANSFORMS_REGISTRY[n].requires_groups])
def test_missing_and_unseen_group_labels_give_finite_predictions(name: str):
    """Groups ``["a", None, "b", nan, "zz"]`` after a fit on string groups must not raise or produce non-finite rows."""
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2, _ = _grid_data(300, 1.0, 0.0)
    b = _base_for(name, base, base2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = _call_fit(t, y, b, np.array(["a", "b", "c"] * 100, dtype=object))
        g_pred = np.array(["a", None, "b", np.nan, "zz"], dtype=object)
        b_pred = b[:5] if t.requires_base else b
        y_back = _call_inverse(t, _call_forward(t, y[:5], b_pred, params, g_pred), b_pred, params, g_pred)
    assert np.all(np.isfinite(y_back)), f"{name}: missing or unseen group labels produced non-finite predictions"


@pytest.mark.parametrize("name", _NAMES)
def test_the_y_domain_rejects_only_rows_the_transform_cannot_round_trip(name: str):
    """A row with an accepted base whose y the domain rejects must fail the round trip; otherwise the domain is over-restrictive.

    Only y-side rejections are probed: a rejected base (non-positive for a geometric mean, zero for a ratio) can still
    divide and multiply back exactly while T means nothing, so its round trip proves nothing.
    """
    t = TRANSFORMS_REGISTRY[name]
    n = 400
    rng = np.random.default_rng(3)
    base = np.linspace(1.0, 10.0, n)
    y = 0.5 * base - 2.5 + rng.standard_normal(n)  # crosses zero
    b = _base_for(name, base, rng.uniform(1.0, 5.0, n))
    base_ok = np.asarray(_call_domain(t, None, b), dtype=bool) if t.requires_base else np.ones(n, dtype=bool)
    domain = np.asarray(_call_domain(t, y, b), dtype=bool)
    rejected = base_ok & ~domain & np.isfinite(y)
    if domain.sum() < 50 or not rejected.any():
        return
    g = _groups(n)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        params = _call_fit(t, y[domain], b[domain] if t.requires_base else b, g[domain])
        b_rej = b[rejected] if t.requires_base else b
        y_back = _call_inverse(t, _call_forward(t, y[rejected], b_rej, params, g[rejected]), b_rej, params, g[rejected])
    exact = np.isfinite(y_back) & (np.abs(y_back - y[rejected]) <= 1e-8 * max(1.0, float(np.max(np.abs(y[rejected])))))
    assert not exact.all(), f"{name}: all {int(rejected.sum())} y-rejected rows round-trip exactly; the domain is over-restrictive"


@pytest.mark.parametrize("name", _NAMES)
def test_fitted_params_do_not_grow_with_the_training_rows(name: str):
    """The pickled params at 1e5 rows are at most 2x their size at 1e4: a model must not carry its training data.

    ECDF knot tables (``rank_ecdf_residual``, ``gaussian_copula_residual``) and the smoothing spline's raw bucket means
    grew 10x, to 3.2 MB, before they were bounded.
    """
    t = TRANSFORMS_REGISTRY[name]
    sizes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (10_000, 100_000):
            y, base, base2, groups = _grid_data(n, 1.0, 0.0)
            params, _, _ = _fit_on_domain(t, name, y, base, base2, groups)
            sizes.append(len(pickle.dumps(params)))
    assert sizes[1] <= 2 * sizes[0], f"{name}: pickled params grew from {sizes[0]} B at 1e4 rows to {sizes[1]} B at 1e5"


def test_no_two_registry_names_share_one_function_triple():
    """Two names over the same ``(fit, forward, inverse)`` are one transform scored twice under different names."""
    seen: dict[tuple, str] = {}
    duplicates = []
    for name in _NAMES:
        t = TRANSFORMS_REGISTRY[name]
        key = (t.fit, t.forward, t.inverse)
        if key in seen:
            duplicates.append((seen[key], name))
        seen.setdefault(key, name)
    assert not duplicates, f"registry names sharing one function triple: {duplicates}"


def _fit_and_forward(name: str):
    """Fitted params, the forward T on the fit rows, and those rows' base and groups."""
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2, groups = _grid_data(400, 1.0, 0.0)
    params, domain, b = _fit_on_domain(t, name, y, base, base2, groups)
    b_fit = b[domain] if t.requires_base else b
    return t, params, _call_forward(t, y[domain], b_fit, params, groups[domain]), b_fit, groups[domain]


@pytest.mark.parametrize("name", [n for n in _NAMES if TRANSFORMS_REGISTRY[n].additive_in_t])
def test_a_transform_declared_additive_in_t_moves_y_one_for_one_with_t(name: str):
    """``additive_in_t`` promises ``inverse(T + d) - inverse(T) == d``; the wrap-pass watchdog relies on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, params, t_fit, b_fit, g = _fit_and_forward(name)
        shift = _call_inverse(t, t_fit + 0.3, b_fit, params, g) - _call_inverse(t, t_fit, b_fit, params, g)
    np.testing.assert_allclose(shift, 0.3, rtol=0, atol=1e-9, err_msg=f"{name} declares additive_in_t but T+0.3 does not move y by 0.3")


@pytest.mark.parametrize("name", [n for n in _NAMES if TRANSFORMS_REGISTRY[n].linear_in_base])
def test_a_transform_declared_linear_in_base_is_additive_and_linear_in_its_base(name: str):
    """``linear_in_base`` promises a y that moves linearly with the base; the soft base shrink relies on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, params, t_fit, b_fit, g = _fit_and_forward(name)
        assert t.additive_in_t, f"{name}: linear_in_base implies additive_in_t"
        y0 = _call_inverse(t, t_fit, b_fit, params, g)
        y1 = _call_inverse(t, t_fit, b_fit + 0.2, params, g)
        y2 = _call_inverse(t, t_fit, b_fit + 0.4, params, g)
    np.testing.assert_allclose(y2 - y0, 2.0 * (y1 - y0), rtol=0, atol=1e-9, err_msg=f"{name} declares linear_in_base but is not linear in it")


@pytest.mark.parametrize("name", ["quantile_residual", "ratio", "log_y", "rank_ecdf_residual"])
def test_a_non_additive_transform_does_not_declare_additive_in_t(name: str):
    """The flag is not set where the property fails: ``quantile_residual`` inverts as ``T * IQR + median``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, params, t_fit, b_fit, g = _fit_and_forward(name)
        shift = _call_inverse(t, t_fit + 0.3, b_fit, params, g) - _call_inverse(t, t_fit, b_fit, params, g)
    assert not np.allclose(shift, 0.3, atol=1e-9), f"{name} is additive in T on this data; pick a non-additive probe"
    assert not t.additive_in_t


def test_the_wrap_watchdog_checks_the_t_error_invariant_only_where_it_holds():
    """The ``MAE_T == MAE_y`` watchdog runs for additive transforms and never for ``quantile_residual`` or an OOF-forward transform."""
    from mlframe.training.core._phase_composite_wrapping import _is_additive_in_t

    assert _is_additive_in_t("linear_residual") and _is_additive_in_t("monotonic_residual") and _is_additive_in_t("causal_anchor_residual")
    assert not _is_additive_in_t("quantile_residual")
    assert not _is_additive_in_t("target_encoding_residual")
    assert not _is_additive_in_t("no_such_transform")


def test_the_soft_base_shrink_covers_every_linear_in_base_transform():
    """``causal_anchor_residual`` (``T + alpha * base``) extrapolates linearly in the base, so the shrink must guard it."""
    from mlframe.training.composite.estimator import _soft_shrink

    assert "causal_anchor_residual" in _soft_shrink.ADDITIVE_BASE_TRANSFORMS
    assert _soft_shrink.ADDITIVE_BASE_TRANSFORMS == {n for n, t in TRANSFORMS_REGISTRY.items() if t.linear_in_base}
