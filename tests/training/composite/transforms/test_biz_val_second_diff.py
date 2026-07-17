"""Unit + biz_value tests for the ``second_diff`` composite transform.

``second_diff`` fits nothing: ``T = y - 2*b1 + b2`` with ``b1`` the lag-1 anchor
(``base_prev``) and ``b2`` the lag-2 anchor (``base_prev2``), supplied as a
``linear_residual_multi``-style ``(n, 2)`` base. It cancels the level AND linear
drift of a doubly-integrated (I(2)) series that a single ``diff`` leaves trending.

biz_value assertion pins the measured win with margin (per CLAUDE.md): on a
random-walk-of-random-walk (I(2)) synthetic the second-difference reconstruction
RMSE crushes plain ``diff`` (measured ratio ~0.017; floor 0.10).
"""

from __future__ import annotations

import warnings

import numpy as np
import orjson
import pytest

warnings.filterwarnings("ignore")

from mlframe.training.composite.transforms import (
    TRANSFORMS_REGISTRY,
    compose_target_name,
    get_transform,
    is_composite_target_name,
)


# ---------------------------------------------------------------------------
# Unit: registry wiring, round-trip, base contract, domain, serialisation.
# ---------------------------------------------------------------------------


def test_second_diff_registered_contract():
    t = get_transform("second_diff")
    assert t is TRANSFORMS_REGISTRY["second_diff"]
    assert t.requires_base is True
    assert t.requires_groups is False
    assert not t.recurrent


def test_second_diff_not_in_default_transform_list():
    """Reachable ONLY via explicit config, mirroring causal_anchor_residual."""
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    cfg = CompositeTargetDiscoveryConfig()
    assert "second_diff" not in cfg.transforms


def test_second_diff_round_trip_identity_two_column_base():
    """inverse(forward(y)) == y exactly (pure additive inverse, no fitted state)."""
    rng = np.random.default_rng(0)
    n = 1500
    b1 = rng.standard_normal(n) * 3.0
    b2 = rng.standard_normal(n) * 3.0
    y = rng.standard_normal(n)
    base = np.column_stack([b1, b2])
    t = get_transform("second_diff")
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    y_back = t.inverse(T, base, p)
    assert np.allclose(y_back, y, atol=1e-12)


def test_second_diff_forward_algebra_matches_definition():
    rng = np.random.default_rng(1)
    n = 200
    b1 = rng.standard_normal(n)
    b2 = rng.standard_normal(n)
    y = rng.standard_normal(n)
    base = np.column_stack([b1, b2])
    t = get_transform("second_diff")
    T = t.forward(y, base, t.fit(y, base))
    np.testing.assert_allclose(T, y - 2.0 * b1 + b2, atol=1e-12)


def test_second_diff_one_d_base_degenerates_to_single_lag():
    """A 1-D base (no lag-2) => T = y - 2*b1 with exact additive inverse."""
    rng = np.random.default_rng(2)
    b1 = rng.standard_normal(800)
    y = rng.standard_normal(800)
    t = get_transform("second_diff")
    p = t.fit(y, b1)
    T = t.forward(y, b1, p)
    np.testing.assert_allclose(T, y - 2.0 * b1, atol=1e-12)
    np.testing.assert_allclose(t.inverse(T, b1, p), y, atol=1e-12)


def test_second_diff_ignores_extra_base_columns():
    """Only the first two columns are consulted; extras must not change T."""
    rng = np.random.default_rng(3)
    n = 300
    b1, b2, junk = (rng.standard_normal(n) for _ in range(3))
    y = rng.standard_normal(n)
    t = get_transform("second_diff")
    base2 = np.column_stack([b1, b2])
    base3 = np.column_stack([b1, b2, junk])
    p = t.fit(y, base3)
    np.testing.assert_array_equal(t.forward(y, base2, p), t.forward(y, base3, p))


def test_second_diff_params_serialize_empty():
    """Parameter-free: fit returns an empty JSON-serialisable dict."""
    rng = np.random.default_rng(4)
    base = np.column_stack([rng.standard_normal(50), rng.standard_normal(50)])
    y = rng.standard_normal(50)
    p = get_transform("second_diff").fit(y, base)
    assert p == {}
    assert orjson.loads(orjson.dumps(p, option=orjson.OPT_SORT_KEYS)) == {}


def test_second_diff_domain_rejects_non_finite():
    t = get_transform("second_diff")
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    base = np.column_stack(
        [
            np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            np.array([1.0, 2.0, 3.0, np.inf, 5.0]),
        ]
    )
    mask = t.domain_check(y, base)
    assert mask.tolist() == [True, False, False, False, True]
    # Predict-time (y is None) gates only the base side.
    assert t.domain_check(None, base).tolist() == [True, True, False, False, True]


def test_second_diff_domain_ignores_nan_in_extra_column():
    """A NaN in an IGNORED (3rd) base column must not drop an otherwise-valid row."""
    t = get_transform("second_diff")
    y = np.array([1.0, 2.0, 3.0])
    base = np.column_stack(
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([np.nan, np.nan, np.nan]),
        ]
    )
    assert t.domain_check(y, base).tolist() == [True, True, True]


def test_second_diff_single_row():
    t = get_transform("second_diff")
    base = np.array([[2.0, 1.0]])
    y = np.array([5.0])
    p = t.fit(y, base)
    T = t.forward(y, base, p)
    assert T.tolist() == [5.0 - 4.0 + 1.0]
    np.testing.assert_allclose(t.inverse(T, base, p), y, atol=1e-12)


def test_second_diff_fit_does_not_mutate_inputs():
    rng = np.random.default_rng(5)
    base = np.column_stack([rng.standard_normal(200), rng.standard_normal(200)])
    y = rng.standard_normal(200)
    b_snap, y_snap = base.copy(), y.copy()
    get_transform("second_diff").fit(y, base)
    np.testing.assert_array_equal(base, b_snap)
    np.testing.assert_array_equal(y, y_snap)


def test_second_diff_composite_name_recognised():
    name = compose_target_name("y", "second_diff", "lag")
    assert name == "y-d2-lag"
    assert is_composite_target_name(name)


# ---------------------------------------------------------------------------
# biz_value: I(2) synthetic -> second_diff reconstruction crushes plain diff.
# ---------------------------------------------------------------------------


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _gen_i2(n: int, seed: int):
    """Random-walk-of-random-walk (I(2)) level with an innovation driven by a
    generalisable feature ``f``: the second difference IS ``innov`` = 0.6*f+noise,
    which a base-agnostic downstream can predict; the first difference is still a
    non-stationary random walk it cannot."""
    r = np.random.default_rng(seed)
    f = r.standard_normal(n)
    innov = 0.6 * f + r.standard_normal(n) * 0.4
    v = np.cumsum(innov)
    y = np.cumsum(v)
    return f, y


def test_biz_val_second_diff_beats_diff_on_doubly_integrated():
    sd = get_transform("second_diff")
    diff = get_transform("diff")
    sd_rmses, diff_rmses = [], []
    for s in range(6):
        f, y = _gen_i2(4000, s)
        b1 = np.empty_like(y)
        b1[0] = 0.0
        b1[1:] = y[:-1]
        b2 = np.empty_like(y)
        b2[:2] = 0.0
        b2[2:] = y[:-2]
        base = np.column_stack([b1, b2])
        ntr = 2000
        ftr, fte = f[2:ntr], f[ntr:]
        ytr, yte = y[2:ntr], y[ntr:]
        btr, bte = base[2:ntr], base[ntr:]
        # second_diff reconstruction.
        p = sd.fit(ytr, btr)
        T = sd.forward(ytr, btr, p)
        c = _ols(ftr, T)
        yhat = sd.inverse(c[0] * fte + c[1], bte, p)
        sd_rmses.append(_rmse(yhat, yte))
        # diff reconstruction (single lag-1 base).
        b1tr, b1te = b1[2:ntr], b1[ntr:]
        pd = diff.fit(ytr, b1tr)
        Td = diff.forward(ytr, b1tr, pd)
        cd = _ols(ftr, Td)
        ydhat = diff.inverse(cd[0] * fte + cd[1], b1te, pd)
        diff_rmses.append(_rmse(ydhat, yte))
    sd_rmse = float(np.median(sd_rmses))
    diff_rmse = float(np.median(diff_rmses))
    assert sd_rmse <= diff_rmse * 0.10, f"second_diff RMSE {sd_rmse:.4f} should be <=0.10x diff RMSE {diff_rmse:.4f} (ratio {sd_rmse / diff_rmse:.4f})"
