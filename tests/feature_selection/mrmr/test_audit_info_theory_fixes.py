"""Regression tests for the info-theory findings of the mrmr audit fix wave (see audits/mrmr_audit_2026-07-25/).

Covers the cardinality-cap corrections (the per-pair gate must bound joint_card * n_classes_y, the PID / BUR /
JMIM / RelaxMRMR entry points must reject an over-budget dense joint), the FE class-code dtype auto-widen (an
int32-disc column with nbins > 32767 must not be forced into a wrapping int16), and the fastmi seed contract.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_batch_pair_mi_gate_bounds_n_classes_y_factor():
    """batch_pair_mi_prange must skip-and-sentinel (score 0.0) a pair whose joint_card is under the 64M cap but
    whose joint_card * n_classes_y is far over it, instead of attempting a ~100 GiB (joint_card, n_classes_y) alloc."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import MAX_JOINT_CARDINALITY, batch_pair_mi_prange

    n = 4
    factors_data = np.zeros((n, 2), dtype=np.int32)  # tiny codes; the gate fires before any per-row work
    nb_a, nb_b = 8000, 7999  # joint_card 63.99M < 64M cap, but *200 classes is ~1.3e10
    assert nb_a * nb_b < MAX_JOINT_CARDINALITY
    nbins = np.array([nb_a, nb_b], dtype=np.int64)
    pair_a = np.array([0], dtype=np.int64)
    pair_b = np.array([1], dtype=np.int64)
    n_classes_y = 200
    classes_y = np.zeros(n, dtype=np.int64)
    freqs_y = np.full(n_classes_y, 1.0 / n_classes_y, dtype=np.float64)
    out = batch_pair_mi_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
    assert out[0] == 0.0, "high-n_classes_y pair over the real (joint_card, n_classes_y) budget must be sentinel-skipped"


def test_check_joint_cardinality_caps_dense_product():
    """check_joint_cardinality must raise once the staged product exceeds the cap, and accept a within-budget product."""
    from mlframe.feature_selection.filters.info_theory._batch_kernels import check_joint_cardinality

    check_joint_cardinality(10, 10, 10, what="ok")  # 1000 cells, fine
    with pytest.raises(ValueError, match="exceeds cap"):
        check_joint_cardinality(5000, 5000, 10, what="pid")  # 2.5e8 > 64M
    with pytest.raises(ValueError, match="exceeds cap"):
        check_joint_cardinality(300000, 300000, what="pair")  # 9e10 would wrap a naive int64 multiply


def test_bur_term_rejects_oversized_joint():
    """bur_term must reject a candidate whose (nbins_x, nbins_y) dense joint blows the cardinality cap rather than OOM."""
    from mlframe.feature_selection.filters._bur_term import bur_term

    n = 8
    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds cap"):
        bur_term(x, [], y, nbins_x=9000, nbins_selected=[], nbins_y=9000)  # 8.1e7 > 64M


def test_fe_classes_dtype_widens_past_int16_for_high_nbins():
    """_fe_classes_dtype must widen an int32-disc column with nbins > 32767 to int32 (not the wrapping int16), while
    keeping the tight int16 for lower bin counts and honoring an already-narrow disc dtype."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import _fe_classes_dtype

    assert _fe_classes_dtype(np.dtype(np.int32), np.array([1000], dtype=np.int64)) is np.int16
    assert _fe_classes_dtype(np.dtype(np.int32), np.array([40000], dtype=np.int64)) is np.int32
    assert _fe_classes_dtype(np.dtype(np.int16), np.array([40000], dtype=np.int64)) is np.int16


def test_fastmi_default_seed_is_deterministic_zero():
    """fastmi's MISE sub-sample must be seedable: the default call maps to seed 0 (reproducible) and an explicit seed round-trips to the same value."""
    from mlframe.feature_selection.filters._fastmi import fastmi

    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = x * 0.8 + rng.normal(size=n) * 0.6  # correlated so MISE has real signal
    v_default = fastmi(x, y)
    v_seed0 = fastmi(x, y, random_seed=0)
    assert np.isfinite(v_default) and v_default >= 0.0
    assert v_default == v_seed0, "default fastmi must equal the explicit seed-0 call (deterministic default)"
    v_seed7 = fastmi(x, y, random_seed=7)
    assert np.isfinite(v_seed7) and v_seed7 >= 0.0  # a different seed still yields a valid estimate
