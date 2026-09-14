"""A failed order-2 maxT floor must be distinguishable from the deliberate no-op (mrmr_audit_2026-09-14 NUM-12/NUM-13).

``compute_pair_maxt_floor`` returns ``0.0`` both when it deliberately self-gates (narrow pools, to stay
byte-identical) and when it FAILED -- and ``0.0`` is the literal "gate off" sentinel its consumers act on
(``_step_pairs_rank``: "No-op when floor==0.0"). So one transient fault silently removed best-of-p
chance-max noise-pair rejection with nothing in the data to say so. NUM-13 is the same shape one level in:
the circuit breaker that makes the fault recoverable had its own failure swallowed to ``debug`` + ``pass``.
"""

import logging

import numpy as np

from mlframe.feature_selection.filters import _mrmr_fe_step_helpers as H

# fe_pair_maxt_min_pairs defaults to 30, so the pool must clear it or the whole block self-gates away.
_N_VARS = 10
_N_PAIRS = _N_VARS * (_N_VARS - 1) // 2  # 45


class _SelfStub:
    """The knobs ``compute_pair_maxt_floor`` reads off the estimator."""

    fe_pair_maxt_null_permutations = 25
    fe_pair_maxt_min_pairs = 30
    fe_pair_maxt_null_quantile = 0.95
    fe_mm_debias_prevalence = False
    random_seed = 0


def _run(stub, *, n_vars=_N_VARS, n_pairs=_N_PAIRS, n_rows=200, k_bins=5, k_y=2):
    """Call with the shapes the real kernel expects: DISCRETISED codes, per-column bin counts, per-ROW y codes.

    ``classes_y`` is the per-row ordinal target code (length n), not the list of distinct classes, and
    ``nbins`` is the per-column bin-count vector. Passing the wrong shapes here indexes the njit kernel out
    of bounds and segfaults rather than raising.
    """
    rng = np.random.default_rng(0)
    return H.compute_pair_maxt_floor(
        stub,
        numeric_vars_to_consider=list(range(n_vars)),
        n_pairs=n_pairs,
        data=rng.integers(0, k_bins, size=(n_rows, n_vars)).astype(np.int64),
        nbins=np.full(n_vars, k_bins, dtype=np.int64),
        classes_y=rng.integers(0, k_y, size=n_rows).astype(np.int64),
        freqs_y=np.full(k_y, 1.0 / k_y, dtype=np.float64),
        verbose=0,
    )


def test_a_failed_floor_is_recorded_on_the_estimator(monkeypatch, caplog):
    """The returned 0.0 is ambiguous, so the failure must be visible as state, not only as a log line."""
    from mlframe.feature_selection.filters import _permutation_null

    def _boom(*a, **k):
        """Stand in for the CPU floor kernel and fail, driving the outer handler."""
        raise RuntimeError("synthetic floor failure")

    monkeypatch.setattr(_permutation_null, "pooled_pair_permutation_null_joint_mi_floor", _boom)

    stub = _SelfStub()
    with caplog.at_level(logging.WARNING):
        floor, bias = _run(stub)

    assert floor == 0.0 and bias == {}
    assert stub._pair_maxt_floor_failed_ is True, "a failed floor must not look like the deliberate no-op"
    assert any("DISABLED" in r.message for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_healthy_floor_is_not_flagged_as_failed():
    """The happy path must leave the flag clear, otherwise the flag says nothing."""
    stub = _SelfStub()
    floor, _bias = _run(stub)
    assert isinstance(floor, float)
    assert stub._pair_maxt_floor_failed_ is False


def test_the_failure_flag_is_reset_per_call(monkeypatch):
    """A stale True from an earlier FE step must not mark a later healthy step as failed."""
    stub = _SelfStub()
    stub._pair_maxt_floor_failed_ = True
    _run(stub)
    assert stub._pair_maxt_floor_failed_ is False


def test_the_deliberate_no_op_is_not_flagged_as_a_failure():
    """A narrow pool floors at 0.0 by design; that must stay distinguishable from the failure path."""
    stub = _SelfStub()
    floor, _bias = _run(stub, n_vars=2, n_pairs=1, n_rows=50)
    assert floor == 0.0
    assert stub._pair_maxt_floor_failed_ is False


def test_breaker_trip_failure_is_warned_not_swallowed(monkeypatch, caplog):
    """NUM-13: the breaker that makes the GPU fault recoverable must not fail more quietly than the fault.

    Drives the real branch: force the resident-GPU path on, make it fault, then make the breaker's own trip
    raise. The CPU floor must still be computed (the degradation is by design) while the failed trip is
    reported at WARNING naming the exception type -- it used to be ``logger.debug`` + a bare ``pass``.
    """
    from mlframe.feature_selection.filters import _permutation_null_pair_resident as RES

    def _gpu_fault(*a, **k):
        """Stand in for the resident-GPU floor and fault, as a poisoned CUDA context would."""
        raise RuntimeError("synthetic cupy launch failure")

    def _breaker_fault(*a, **k):
        """Make the circuit-breaker trip itself fail -- the NUM-13 branch under test."""
        raise ValueError("synthetic breaker failure")

    monkeypatch.setattr(RES, "pair_maxt_perm_null_gpu_enabled", lambda *a, **k: True)
    monkeypatch.setattr(RES, "pooled_pair_permutation_null_joint_mi_floor_cupy", _gpu_fault)
    monkeypatch.setattr(RES, "trip_pair_maxt_gpu_circuit_breaker", _breaker_fault)

    stub = _SelfStub()
    with caplog.at_level(logging.WARNING):
        floor, _bias = _run(stub)

    warnings_text = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "circuit breaker" in warnings_text, f"breaker-trip failure was not warned; got: {warnings_text!r}"
    assert "ValueError" in warnings_text, "the warning must name the exception type"
    # The CPU njit floor still ran: this is a degradation, not a failure, so a real float comes back.
    assert isinstance(floor, float)
    assert stub._pair_maxt_floor_failed_ is False, "the CPU floor succeeded, so this is not a failed floor"
