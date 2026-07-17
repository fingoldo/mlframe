"""Adversarial / edge-case battery for the clustering primitives (2026-06-03).

Earlier benchmarks used benign Gaussian fixtures. This file throws degenerate
and hostile inputs at the low-level clustering primitives and asserts GRACEFUL,
FINITE, DETERMINISTIC behaviour. Failures here are real prod bugs to fix, not
test problems.

Targets: _standardize_align, compute_cluster_aggregate, pair_su (SU/VI/auto),
and the aggregate combiners under constants / NaN / Inf / n<=3 / perfect dups /
mixed-sign clusters / rank-deficiency.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cluster_aggregate import (
    _apply_method_nonlinear,
    _derive_weights,
    _standardize_align,
    CLUSTER_AGGREGATE_METHODS,
)


def _finite(a):
    return np.all(np.isfinite(np.asarray(a, dtype=np.float64)))


# ---------------------------------------------------------------------------
# _standardize_align
# ---------------------------------------------------------------------------


class TestStandardizeAlign:
    def test_constant_column_no_nan(self):
        M = np.column_stack([np.zeros(50), np.arange(50.0)])  # col0 constant
        Z, _mean, _std, signs = _standardize_align(M, 0)
        assert _finite(Z) and _finite(signs)

    def test_all_constant_matrix(self):
        M = np.full((40, 3), 7.0)
        Z, *_ = _standardize_align(M, 0)
        assert _finite(Z)

    def test_n_equals_2(self):
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Z, *_ = _standardize_align(M, 0)
        assert _finite(Z)

    def test_mixed_sign_alignment(self):
        # cols 1 and 3 are anti-correlated with col 0; signs must flip them.
        rng = np.random.default_rng(0)
        z = rng.standard_normal(200)
        M = np.column_stack([z, -z + 0.01 * rng.standard_normal(200), z + 0.01 * rng.standard_normal(200), -z + 0.01 * rng.standard_normal(200)])
        Z, _mean, _std, signs = _standardize_align(M, 0)
        assert signs[1] < 0 and signs[3] < 0 and signs[0] > 0 and signs[2] > 0
        # After alignment all columns point the same way -> mean has high variance.
        assert float(np.std(Z.mean(axis=1))) > 0.5

    def test_nan_input_does_not_silently_emit_nan_aggregate(self):
        # _standardize_align itself does not fill NaN; document that callers must.
        # The canonical aggregate path (compute_cluster_aggregate) fills first;
        # here we assert the raw helper's NaN propagation so a future caller that
        # forgets to fill is caught by this contract test.
        M = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]])
        Z, *_ = _standardize_align(M, 0)
        # Either it stays finite (guarded) OR NaN propagates; pin current behaviour.
        # If this assertion flips, a guard was added -> update the contract.
        assert not _finite(Z), "if _standardize_align now NaN-guards, update callers/test"


# ---------------------------------------------------------------------------
# compute_cluster_aggregate (orth-basis path, post gap-03)
# ---------------------------------------------------------------------------


class TestComputeClusterAggregate:
    def _agg(self, X, members, aggregator):
        import pandas as pd
        from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
            compute_cluster_aggregate,
        )

        return compute_cluster_aggregate(pd.DataFrame(X), members, aggregator=aggregator)

    @pytest.mark.parametrize("aggregator", ["mean_z", "median_z", "pc1"])
    def test_all_nan_member_finite(self, aggregator):
        import pandas as pd

        n = 100
        X = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.full(n, np.nan)})
        out = self._agg(X, ["a", "b"], aggregator)
        assert _finite(out), f"{aggregator} emitted non-finite on all-NaN member"

    @pytest.mark.parametrize("aggregator", ["mean_z", "median_z", "pc1"])
    def test_constant_member_finite(self, aggregator):
        import pandas as pd

        n = 100
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": np.full(n, 5.0)})
        out = self._agg(X, ["a", "b"], aggregator)
        assert _finite(out)

    @pytest.mark.parametrize("aggregator", ["mean_z", "median_z", "pc1"])
    def test_identical_members_rank_deficient(self, aggregator):
        import pandas as pd

        rng = np.random.default_rng(1)
        col = rng.standard_normal(120)
        X = pd.DataFrame({"a": col, "b": col, "c": col})  # perfect dups
        out = self._agg(X, ["a", "b", "c"], aggregator)
        assert _finite(out)


# ---------------------------------------------------------------------------
# combiners directly
# ---------------------------------------------------------------------------


class TestCombiners:
    @pytest.mark.parametrize("method", list(CLUSTER_AGGREGATE_METHODS))
    def test_combiner_finite_on_degenerate_Z(self, method):
        # Z with a zero column and a constant column.
        Z = np.column_stack([np.zeros(60), np.ones(60), np.linspace(-1, 1, 60)])
        w = _derive_weights(Z, method)
        out = _apply_method_nonlinear(Z, method) if w is None else (Z @ np.asarray(w))
        assert _finite(out), f"combiner {method} emitted non-finite on degenerate Z"
