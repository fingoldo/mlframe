"""Unit tests for structure-aware calib/conformal carving (`_conformal_split.py`).

Pins the exchangeable-unit discipline: iid -> disjoint random; temporal -> forward blocks with a
purge gap (conformal = most recent); grouped -> no group straddles fit/calib/conformal.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._conformal_split import (
    carve_calib_conformal,
    carve_calib_conformal_grouped,
    carve_calib_conformal_iid,
    carve_calib_conformal_temporal,
)


def _disjoint(fit, calib, conf):
    """Disjoint."""
    s_fit, s_cal, s_conf = set(fit.tolist()), set(calib.tolist()), set(conf.tolist())
    return not (s_fit & s_cal) and not (s_fit & s_conf) and not (s_cal & s_conf)


def test_iid_carve_disjoint_and_sized():
    """Iid carve disjoint and sized."""
    train_idx = np.arange(1000)
    fit, calib, conf = carve_calib_conformal_iid(train_idx, 0.1, 0.1, seed=0)
    assert _disjoint(fit, calib, conf)
    assert fit.size + calib.size + conf.size == 1000
    assert calib.size == 100 and conf.size == 100


def test_temporal_carve_forward_order_and_purge():
    # train_idx not chronological; time_values give the order. Conformal must be the most-recent block.
    """Temporal carve forward order and purge."""
    n = 1000
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    train_idx = np.arange(n)[perm]
    time_values = np.arange(n)[perm]  # same permutation -> idx value == time
    purge = 20
    fit, calib, conf = carve_calib_conformal_temporal(train_idx, 0.1, 0.1, time_values=time_values, purge=purge)
    assert _disjoint(fit, calib, conf)
    # Conformal is the latest in time, fit the earliest; gaps exist.
    assert fit.max() < calib.min()
    assert calib.max() < conf.min()
    assert calib.min() - fit.max() - 1 >= purge
    assert conf.min() - calib.max() - 1 >= purge


def test_grouped_carve_no_group_straddles():
    """Grouped carve no group straddles."""
    n = 1200
    groups = np.repeat(np.arange(120), 10)  # 120 groups of 10 rows
    train_idx = np.arange(n)
    fit, calib, conf = carve_calib_conformal_grouped(train_idx, 0.2, 0.2, group_values=groups, seed=1)
    assert _disjoint(fit, calib, conf)
    g_fit = set(groups[fit].tolist())
    g_cal = set(groups[calib].tolist())
    g_conf = set(groups[conf].tolist())
    assert not (g_fit & g_cal) and not (g_fit & g_conf) and not (g_cal & g_conf)


def test_dispatch_routes_by_structure():
    """Dispatch routes by structure."""
    train_idx = np.arange(600)
    groups = np.repeat(np.arange(60), 10)
    f1, c1, k1 = carve_calib_conformal(train_idx, 0.1, 0.1, structure="iid", seed=0)
    assert _disjoint(f1, c1, k1)
    f2, c2, k2 = carve_calib_conformal(
        train_idx,
        0.1,
        0.1,
        structure="grouped",
        group_values=groups,
        seed=0,
    )
    g_all = [set(groups[a].tolist()) for a in (f2, c2, k2)]
    assert not (g_all[0] & g_all[1]) and not (g_all[0] & g_all[2]) and not (g_all[1] & g_all[2])
    f3, c3, k3 = carve_calib_conformal(
        train_idx,
        0.1,
        0.1,
        structure="temporal",
        time_values=train_idx,
        purge=5,
    )
    assert f3.max() < c3.min() < k3.min()


def test_grouped_requires_group_values():
    """Grouped requires group values."""
    with pytest.raises(ValueError):
        carve_calib_conformal(np.arange(100), 0.1, 0.1, structure="grouped")


def test_carve_raises_when_no_fit_rows_left():
    """Carve raises when no fit rows left."""
    with pytest.raises(ValueError):
        carve_calib_conformal_iid(np.arange(100), 0.6, 0.6, seed=0)
