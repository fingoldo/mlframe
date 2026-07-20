"""Correctness tests for ``_cupy_polynom_optimizer.run_cupy_kernel_search`` (mrmr_audit_2026-07-20
test_coverage.md #13). This GPU-resident device twin of ``_numba_polynom_optimizer`` shipped with zero
tests. The triage flagged a real reachable bug: when ``elitism_k >= batch_size``, ``n_perturb =
batch_size - elitism_k - max(1, batch_size // 4)`` goes negative, and ``rng.integers(0, elitism_k,
size=n_perturb)`` raises ``ValueError`` on a negative size on the very first generation -- fixed by
clamping ``elitism_k`` below ``batch_size`` and floor-clamping ``n_perturb`` at 0."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._cupy_polynom_optimizer import run_cupy_kernel_search


def _hermite_eval():
    """A dummy eval_func whose __name__ lets run_cupy_kernel_search infer basis='hermite'."""

    def hermite_eval(*args, **kwargs):
        """Never actually invoked -- run_cupy_kernel_search only reads its __name__."""
        raise AssertionError("eval_func body must never be called by run_cupy_kernel_search")

    return hermite_eval


def _make_signal(n=2000, seed=0, n_classes=4):
    """x_a, x_b standardised; y is a discretised strong function of x_a*x_b (mirrors the numba
    optimizer's own synthetic-signal fixture)."""
    rng = np.random.default_rng(seed)
    x_a = rng.standard_normal(n)
    x_b = rng.standard_normal(n)
    raw = x_a * x_b
    edges = np.quantile(raw, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(raw, edges).astype(np.int64)
    return x_a, x_b, y


def _eval_kwargs(x_a, x_b, y, l2_penalty=0.0):
    """The eval_kwargs dict run_cupy_kernel_search expects, wired for the hermite/plugin-MI path."""
    return dict(
        eval_func=_hermite_eval(),
        bf_names=("mul", "add", "sub", "div"),
        mi_estimator="plugin",
        discrete_target=True,
        z_a=x_a,
        z_b=x_b,
        y_njit=y,
        plugin_n_bins=20,
        l2_penalty=l2_penalty,
    )


class TestElitismKBatchSizeCrash:
    """The confirmed bug: elitism_k >= batch_size must not crash rng.integers with a negative size."""

    def test_elitism_k_equal_to_batch_size_does_not_raise(self):
        """elitism_k == batch_size zeroes out n_perturb entirely before the fix; must still return a result."""
        x_a, x_b, y = _make_signal(seed=1)
        result = run_cupy_kernel_search(
            ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), n_trials=40, seed=42,
            direction_only=False, warm_start_seeds=None, eval_kwargs=_eval_kwargs(x_a, x_b, y),
            batch_size=10, elitism_k=10,
        )
        assert result is not None
        assert np.isfinite(result[4])

    def test_elitism_k_greater_than_batch_size_does_not_raise(self):
        """elitism_k > batch_size is the confirmed crash case (negative n_perturb pre-fix)."""
        x_a, x_b, y = _make_signal(seed=2)
        result = run_cupy_kernel_search(
            ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), n_trials=40, seed=42,
            direction_only=False, warm_start_seeds=None, eval_kwargs=_eval_kwargs(x_a, x_b, y),
            batch_size=10, elitism_k=50,
        )
        assert result is not None
        assert np.isfinite(result[4])


class TestSignalRecovery:
    """Ground-truth check: the kernel must find a candidate scoring well above a near-zero MI floor."""

    def test_recovers_strong_synthetic_signal(self):
        """A normal (non-crash-triggering) elitism_k/batch_size pair must still find a strong signal."""
        x_a, x_b, y = _make_signal(seed=1)
        result = run_cupy_kernel_search(
            ca_size=3, cb_size=3, coef_range=(-2.0, 2.0), n_trials=300, seed=42,
            direction_only=False, warm_start_seeds=None, eval_kwargs=_eval_kwargs(x_a, x_b, y),
            batch_size=20, elitism_k=4,
        )
        assert result is not None
        coef_a, coef_b, bf_idx, _raw_mi, score = result
        assert score > 0.3, f"expected a strong MI recovery, got {score}"
        assert bf_idx >= 0
        assert np.isfinite(coef_a).all() and np.isfinite(coef_b).all()
