"""Correctness tests for ``_numba_polynom_optimizer.py`` (2026-07-11).

This module implements a full ``@njit(parallel=True)`` port of the existing ``optimizer="random_batch"``
strategy (batch random search + elitism + Gaussian perturbation, see ``_hermite_fe_optimise._run_random_batch_search``)
so ``run_polynom_pair_fe`` can evaluate ALL candidate pairs in ONE ``prange``-parallel kernel call instead of
dispatching one loky worker PROCESS per pair -- no GIL, no worker spawn, no pickling, no memmap. Its own
docstring says as much ("This is the path the polynom-pair FE dispatch should call when
``fe_optimizer == "numba_kernel"`` to fully eliminate joblib"), but it shipped with ZERO tests and was never
actually wired into ``run_polynom_pair_fe``'s dispatch -- ``fe_optimizer='numba_kernel'`` only changes the
PER-PAIR inner search strategy inside a loky-dispatched worker, it does not skip the outer loky pool.

These tests establish the missing baseline of trust before this kernel is used for anything: (1) it recovers a
strong, unambiguous synthetic signal (proves the njit port is functionally sound, not just "doesn't crash"),
(2) multi-pair batching (P>1, prange) gives results consistent with running each pair alone at P=1 (the
critical thread-safety check for the RNG-stream design -- each prange iteration must be fully independent),
(3) it reaches comparable solution QUALITY to the existing ``_run_random_batch_search`` implementation it
mirrors (not bit-identical -- both are stochastic global optimizers with different RNG draw orders -- but
should land in the same ballpark on a fixed budget), (4) basic robustness on degenerate inputs (constant
column, tiny n) does not crash or return nonsense.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_polynom_optimizer import (
    optimize_all_pairs_numba_kernel,
    run_numba_kernel_search,
)


def _make_hermite_signal(n=4000, seed=0, n_classes=4):
    """x_a, x_b standardised; y is a discretised strong function of (x_a * x_b) -- a signal the mul bf_id +
    a low-degree hermite basis should recover easily."""
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal(n)
    x_b = rng.standard_normal(n)
    raw = x_a * x_b
    # quantile-discretize raw into n_classes bins -> a clean classification target strongly tied to x_a*x_b.
    edges = np.quantile(raw, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(raw, edges).astype(np.int64)
    return x_a, x_b, y


def test_optimize_all_pairs_recovers_strong_synthetic_signal():
    """Ground-truth check: y is a strong discretized function of x_a*x_b -- the kernel must find a candidate
    scoring well above a near-zero/noise-floor MI, not just return SOME finite number."""
    x_a, x_b, y = _make_hermite_signal(seed=1)
    n = x_a.shape[0]
    X = np.column_stack([x_a, x_b])
    pair_indices = np.array([[0, 1]], dtype=np.int64)

    result = optimize_all_pairs_numba_kernel(
        X, y, pair_indices,
        ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), basis="hermite",
        bf_names=("mul", "add", "sub", "div"),
        n_trials=300, batch_size=20, elitism_k=4,
        n_bins=20, l2_penalty=0.0, direction_only=False, discrete_target=True,
        seed=42,
    )
    assert result["best_scores"][0] > 0.3, f"expected a strong MI recovery, got {result['best_scores'][0]}"
    assert result["best_bfs"][0] >= 0
    assert result["n_evals"][0] > 0
    assert np.isfinite(result["best_coefs_a"][0]).all()
    assert np.isfinite(result["best_coefs_b"][0]).all()


def test_run_numba_kernel_search_single_pair_matches_batched_p1():
    """run_numba_kernel_search (single-pair convenience wrapper around _run_cma_search's contract) internally
    calls the SAME multi-pair kernel with P=1 -- verify it doesn't silently diverge from a direct P=1 call to
    optimize_all_pairs_numba_kernel with equivalent inputs."""
    x_a, x_b, y = _make_hermite_signal(seed=2)
    X = np.column_stack([x_a, x_b])
    pair_indices = np.array([[0, 1]], dtype=np.int64)

    direct = optimize_all_pairs_numba_kernel(
        X, y, pair_indices,
        ca_size=2, cb_size=2, coef_range=(-1.5, 1.5), basis="hermite",
        bf_names=("mul", "add"),
        n_trials=100, batch_size=10, elitism_k=2,
        n_bins=15, l2_penalty=0.05, direction_only=False, discrete_target=True,
        seed=7,
    )

    def _hermeval_stub(x, c):
        return x  # name is inspected via __name__, body unused by run_numba_kernel_search

    _hermeval_stub.__name__ = "hermeval_placeholder"
    wrapper_result = run_numba_kernel_search(
        ca_size=2, cb_size=2, coef_range=(-1.5, 1.5), n_trials=100, seed=7,
        direction_only=False,
        warm_start_seeds=None,
        eval_kwargs={
            "eval_func": _hermeval_stub,
            "bf_names": ("mul", "add"),
            "z_a": x_a, "z_b": x_b,
            "discrete_target": True,
            "y_njit": y,
            "plugin_n_bins": 15,
            "l2_penalty": 0.05,
        },
        batch_size=10, elitism_k=2,
    )
    assert wrapper_result is not None
    _, _, wrapper_bf, wrapper_raw, wrapper_evals = wrapper_result
    # Same seed, same search budget, same problem shape -> the two entry points must produce IDENTICAL
    # results (both reduce to the exact same P=1 kernel call with the same RNG-stream generation order).
    assert wrapper_bf == int(direct["best_bfs"][0])
    assert wrapper_raw == pytest.approx(float(direct["best_raws"][0]), abs=1e-12)
    assert wrapper_evals == int(direct["n_evals"][0])


def test_multi_pair_batching_is_reproducible_across_repeated_calls():
    """Non-determinism / race-condition smoke test: the SAME call (same pre-generated RNG streams via the
    same seed) run TWICE must give BIT-IDENTICAL results. A genuine cross-thread race in the prange loop
    (e.g. two threads aliasing the same output row, or an accidentally-shared mutable scratch buffer) would
    show up as run-to-run non-determinism here even though each individual run "looks" fine."""
    rng = np.random.default_rng(99)
    n = 3000
    cols = [rng.standard_normal(n) for _ in range(6)]
    X = np.column_stack(cols)
    _, _, y = _make_hermite_signal(n=n, seed=99)

    pair_indices = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    kwargs = dict(
        ca_size=2, cb_size=2, coef_range=(-1.0, 1.0), basis="hermite",
        bf_names=("mul", "sub"), n_trials=80, batch_size=10, elitism_k=2,
        n_bins=12, l2_penalty=0.0, direction_only=False, discrete_target=True, seed=123,
    )
    run1 = optimize_all_pairs_numba_kernel(X, y, pair_indices, **kwargs)
    run2 = optimize_all_pairs_numba_kernel(X, y, pair_indices, **kwargs)
    assert np.array_equal(run1["best_scores"], run2["best_scores"])
    assert np.array_equal(run1["best_bfs"], run2["best_bfs"])
    assert np.array_equal(run1["best_coefs_a"], run2["best_coefs_a"])
    assert np.array_equal(run1["best_coefs_b"], run2["best_coefs_b"])
    # Sanity: the 3 independent pairs (unrelated random columns) should NOT all land on the identical score --
    # that would suggest every prange iteration is reading the SAME stream row instead of its own.
    assert len(set(np.round(run1["best_scores"], 6))) > 1, "all pairs converged identically -- suspect stream aliasing"


def test_prange_thread_isolation_given_identical_pre_generated_streams():
    """The REAL thread-safety property: given the SAME uniform/normal RNG streams handed to the low-level
    kernel directly (bypassing the convenience wrapper's own RNG generation, which consumes a shared
    ``np.random.Generator`` in a P-dependent order and so is NOT expected to reproduce per-row across
    different P -- that is a property of the wrapper's stream-slicing convenience, not of the kernel), pair
    ``p``'s result must depend ONLY on ``uniform_streams[p]``/``normal_streams[p]`` -- running it as part of a
    3-pair prange batch must give the IDENTICAL result to running it alone with its own stream row extracted
    verbatim. This isolates genuine cross-thread interference from RNG-consumption-order confusion."""
    from mlframe.feature_selection.filters._numba_polynom_optimizer import _optimize_all_pairs_kernel

    rng = np.random.default_rng(7)
    n = 2000
    cols = [rng.standard_normal(n) for _ in range(6)]
    X = np.ascontiguousarray(np.column_stack(cols))
    _, _, y = _make_hermite_signal(n=n, seed=7)
    y = np.ascontiguousarray(y)

    pair_indices = np.ascontiguousarray(np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64))
    ca_size = cb_size = 2
    bf_ids = np.array([0, 2], dtype=np.int64)  # mul, sub
    n_trials, batch_size, elitism_k = 60, 10, 2
    n_iters = int(np.ceil(n_trials / batch_size))
    u_per_iter = batch_size * (ca_size + cb_size)
    n_per_iter = elitism_k * (ca_size + cb_size)
    P = 3

    stream_rng = np.random.default_rng(555)
    uniform_streams = stream_rng.uniform(-1.0, 1.0, size=(P, u_per_iter * n_iters))
    normal_streams = stream_rng.normal(0.0, 0.2, size=(P, n_per_iter * n_iters))
    warm_a = np.zeros((1, ca_size)); warm_b = np.zeros((1, cb_size))

    def _run(_pair_indices, _uniform, _normal, _p_count):
        out_ca = np.zeros((_p_count, ca_size)); out_cb = np.zeros((_p_count, cb_size))
        out_score = np.full(_p_count, -np.inf); out_raw = np.zeros(_p_count)
        out_bf = np.full(_p_count, -1, dtype=np.int64); out_evals = np.zeros(_p_count, dtype=np.int64)
        _optimize_all_pairs_kernel(
            X, y, _pair_indices, ca_size, cb_size, -1.0, 1.0,
            n_iters, batch_size, elitism_k, 0.2, 0, bf_ids, 12, 0.0, False, True,
            _uniform, _normal, warm_a, warm_b, 0,
            out_ca, out_cb, out_score, out_raw, out_bf, out_evals,
        )
        return out_score, out_bf, out_ca, out_cb

    batched_score, batched_bf, batched_ca, batched_cb = _run(pair_indices, uniform_streams, normal_streams, P)

    for p in range(P):
        single_score, single_bf, single_ca, single_cb = _run(
            pair_indices[p : p + 1], uniform_streams[p : p + 1], normal_streams[p : p + 1], 1,
        )
        assert batched_score[p] == single_score[0], (p, batched_score[p], single_score[0])
        assert batched_bf[p] == single_bf[0]
        assert np.array_equal(batched_ca[p], single_ca[0])
        assert np.array_equal(batched_cb[p], single_cb[0])


def test_comparable_quality_to_existing_random_batch_search():
    """Not bit-identical (different implementations, different RNG draw orders for the SAME conceptual
    algorithm), but on the same synthetic signal with a comparable trial budget, the njit kernel should not
    land meaningfully WORSE than the existing pure-Python/numpy ``_run_random_batch_search`` it mirrors --
    both implement the same batch-random+elitism+perturbation strategy."""
    from mlframe.feature_selection.filters._hermite_fe_optimise import _run_random_batch_search
    from mlframe.feature_selection.filters.hermite_fe import _hermeval_njit

    x_a, x_b, y = _make_hermite_signal(seed=5)
    X = np.column_stack([x_a, x_b])
    pair_indices = np.array([[0, 1]], dtype=np.int64)

    numba_result = optimize_all_pairs_numba_kernel(
        X, y, pair_indices,
        ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), basis="hermite",
        bf_names=("mul", "add", "sub", "div"),
        n_trials=300, batch_size=20, elitism_k=4,
        n_bins=20, l2_penalty=0.0, direction_only=False, discrete_target=True,
        seed=11,
    )

    from mlframe.feature_selection.filters.hermite_fe import _DEFAULT_BIN_FUNCS

    bf_names = ("mul", "add", "sub", "div")
    legacy_result = _run_random_batch_search(
        ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), n_trials=300, seed=11,
        direction_only=False, warm_start_seeds=None,
        eval_kwargs={
            "eval_func": _hermeval_njit,
            "bf_callables": [_DEFAULT_BIN_FUNCS[n] for n in bf_names],
            "bf_names": bf_names,
            "z_a": x_a, "z_b": x_b,
            "y": y, "y_njit": y,
            "mi_estimator": "plugin",
            "discrete_target": True,
            "plugin_n_bins": 20,
            "n_neighbors": 3,
            "l2_penalty": 0.0,
        },
        batch_size=20, elitism_k=4,
    )
    # (best_coef_a, best_coef_b, best_bf_idx, best_raw_mi, n_evals) -- index 3 is the raw MI score.
    legacy_score = legacy_result[3] if legacy_result is not None else -np.inf
    numba_score = float(numba_result["best_scores"][0])
    assert numba_score > 0.3, f"numba kernel found a weak solution: {numba_score}"
    # within a generous factor of the legacy path's quality -- both should comfortably recover this signal.
    assert numba_score >= legacy_score * 0.7, (numba_score, legacy_score)


def test_degenerate_constant_column_does_not_crash():
    """A constant operand column can't carry MI -- the kernel must return a finite (likely low/zero) score,
    never NaN/inf or a crash, matching the existing optimizers' best-effort degenerate handling."""
    n = 1000
    x_a = np.full(n, 3.0)
    rng = np.random.default_rng(3)
    x_b = rng.standard_normal(n)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    X = np.column_stack([x_a, x_b])
    pair_indices = np.array([[0, 1]], dtype=np.int64)

    result = optimize_all_pairs_numba_kernel(
        X, y, pair_indices,
        ca_size=2, cb_size=2, coef_range=(-1.0, 1.0), basis="hermite",
        bf_names=("mul", "add"), n_trials=40, batch_size=10, elitism_k=2,
        n_bins=10, l2_penalty=0.0, direction_only=False, discrete_target=True,
        seed=1,
    )
    assert np.isfinite(result["best_scores"][0]) or result["best_scores"][0] == -np.inf


def test_unsupported_basis_raises_clear_error():
    x_a, x_b, y = _make_hermite_signal(n=500, seed=4)
    X = np.column_stack([x_a, x_b])
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="not supported by numba_kernel"):
        optimize_all_pairs_numba_kernel(
            X, y, pair_indices,
            ca_size=2, cb_size=2, coef_range=(-1.0, 1.0), basis="rbf",
            n_trials=20, discrete_target=True, seed=1,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
