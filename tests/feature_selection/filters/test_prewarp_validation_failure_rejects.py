"""A prewarp whose held-out validation raises must be REJECTED, not accepted (mrmr_audit_2026-09-14 NUM-14).

``_prewarp_generalises`` exists to prove, on held-out rows, that an ALS operand warp generalises; an
overfit-on-noise warp otherwise gets engineered and absorbs a genuine feature. On ANY exception it returned
``True`` ("accept the warp") at debug level -- promoting an unvalidated operand exactly when the check
could not run. It now rejects and warns.
"""

import logging

import numpy as np

from mlframe.feature_selection.filters import hermite_fe
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import _fit_prewarp_and_gate_med

_N = 120  # >= 60 so the held-out CV path is active


def _call(monkeypatch, fake_als):
    """Run the prewarp setup for one prospective pair with a controlled ALS fitter."""
    monkeypatch.setattr(hermite_fe, "fit_pair_prewarp_als", fake_als)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(_N, 2))
    return _fit_prewarp_and_gate_med(
        prospective_pairs={((0, 1), "mul"): None},
        prewarp_enable=True,
        prewarp_y=rng.normal(size=_N),
        prewarp_y_continuous=None,
        prewarp_basis="hermite",
        prewarp_max_degree=3,
        prewarp_min_val_corr=0.08,  # > 0 -> held-out validation enabled
        fe_gate_med_enable=False,
        original_cols=[0, 1],
        _use_subsample=False,
        _full_n_rows=_N,
        _sample_idx=None,
        _extval_raw_col=lambda v: data[:, v],
    )


def test_a_validation_that_raises_does_not_register_the_warp(monkeypatch, caplog):
    """The ALS fit raises on the TRAIN slice (inside validation) but would succeed on full n.

    Pre-fix the validator returned True, so the full-n fit ran and both operands were registered. Post-fix
    the validator rejects, so neither operand gets a warp spec.
    """

    def _fake_als(a, b, y, basis=None, max_degree=None):
        """Fail only on the shorter train slice the held-out validation fits on."""
        if len(a) < _N:
            raise RuntimeError("synthetic ALS failure on the train slice")
        return {"spec": "a"}, {"spec": "b"}

    with caplog.at_level(logging.WARNING):
        _active, spec_by_var, _gm_active, _gm_medians = _call(monkeypatch, _fake_als)

    assert spec_by_var == {}, f"an unvalidated warp was registered: {spec_by_var!r}"
    assert any("REJECTING the warp" in r.message for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_validation_that_succeeds_still_registers_the_warp(monkeypatch):
    """Control: when validation can run and the fit is healthy, the warp must still be admitted.

    Without this, the test above would also pass if the setup had simply stopped registering anything.
    """

    def _fake_als(a, b, y, basis=None, max_degree=None):
        """Always succeed; the real validator then decides from the held-out correlation."""
        return {"spec": "a"}, {"spec": "b"}

    def _identity_warp(x, spec):
        """Pass the operand through so the reconstruction correlates with a y built from it."""
        return np.asarray(x, dtype=np.float64)

    monkeypatch.setattr(hermite_fe, "apply_operand_prewarp", _identity_warp)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(_N, 2))
    y = data[:, 0] * data[:, 1]  # genuine product synergy -> the held-out reconstruction tracks y
    monkeypatch.setattr(hermite_fe, "fit_pair_prewarp_als", _fake_als)
    _active, spec_by_var, _gm_active, _gm_medians = _fit_prewarp_and_gate_med(
        prospective_pairs={((0, 1), "mul"): None},
        prewarp_enable=True,
        prewarp_y=y,
        prewarp_y_continuous=None,
        prewarp_basis="hermite",
        prewarp_max_degree=3,
        prewarp_min_val_corr=0.08,
        fe_gate_med_enable=False,
        original_cols=[0, 1],
        _use_subsample=False,
        _full_n_rows=_N,
        _sample_idx=None,
        _extval_raw_col=lambda v: data[:, v],
    )
    assert set(spec_by_var) == {0, 1}
