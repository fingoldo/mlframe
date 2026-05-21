"""CRITICAL #2 regression: ``check_prospective_fe_pairs`` memory dispatcher.

Pre-fix the function always allocated a single hoisted ``(n, max_n_combs *
|binary|)`` float32 scratch buffer. On n=4M with the medium preset this is
~17.6 GiB and the suite crashed with numpy.core._exceptions._ArrayMemoryError.

Post-fix it estimates the required buffer, checks psutil.virtual_memory()
.available, and falls back to a recompute-from-metadata path (1D scratch +
on-demand survivor rebuild) when RAM is tight. The two paths must produce
identical survivors -- the test below forces the fallback by setting the
budget ratio to 0.0 and asserts byte-identical output vs the fast path.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import feature_engineering as fe_mod
from mlframe.feature_selection.filters.feature_engineering import (
    _can_hoist_shared_buffer,
    _estimate_fe_shared_buffer_bytes,
    check_prospective_fe_pairs,
    create_binary_transformations,
    create_unary_transformations,
)


@pytest.fixture
def synthetic_pair_inputs():
    """Build a minimal but realistic input set: 200 rows, 3 columns, one prospective pair
    and a binary target derived from a deterministic threshold on col-0."""
    from mlframe.feature_selection.filters.info_theory import merge_vars
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "a": rng.uniform(0.5, 5.0, n).astype(np.float32),
        "b": rng.uniform(-2.0, 2.0, n).astype(np.float32),
        "c": rng.uniform(0.1, 1.0, n).astype(np.float32),
    })
    data = np.column_stack([
        discretize_array(df[c].to_numpy(), n_bins=4, method="quantile", dtype=np.int32)
        for c in ("a", "b", "c")
    ])
    target_col = (df["a"].to_numpy() > df["a"].mean()).astype(np.int32)
    data = np.column_stack([data, target_col])
    nbins = np.array([4, 4, 4, 2], dtype=np.int64)
    target_indices = np.array([3], dtype=np.int64)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=data, vars_indices=target_indices,
        var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
    )
    return {
        "df": df,
        "classes_y": classes_y,
        "classes_y_safe": classes_y.copy(),
        "freqs_y": freqs_y,
        "original_cols": {0: 0, 1: 1, 2: 2},
        "prospective_pairs": {((0, 1), 1.0): 1.5},
        "cols_names": ["a", "b", "c"],
    }


def _run_check(inputs, *, unary_preset: str, binary_preset: str):
    unary = create_unary_transformations(preset=unary_preset)
    binary = create_binary_transformations(preset=binary_preset)
    times_spent: dict = defaultdict(float)
    return check_prospective_fe_pairs(
        prospective_pairs=inputs["prospective_pairs"],
        X=inputs["df"],
        unary_transformations=unary,
        binary_transformations=binary,
        classes_y=inputs["classes_y"],
        classes_y_safe=inputs["classes_y_safe"],
        freqs_y=inputs["freqs_y"],
        num_fs_steps=0,
        cols=inputs["cols_names"],
        original_cols=inputs["original_cols"],
        fe_max_steps=2,  # >1 so transformed_vals buffer is materialised
        fe_npermutations=1,
        fe_max_pair_features=4,
        fe_print_best_mis_only=True,
        fe_min_nonzero_confidence=0.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_good_to_best_feature_mi_threshold=0.5,
        fe_max_external_validation_factors=0,
        numeric_vars_to_consider=[0, 1, 2],
        quantization_nbins=4,
        quantization_method="quantile",
        quantization_dtype=np.int32,
        times_spent=times_spent,
        verbose=0,
    )


class TestDispatcher:
    def test_buffer_estimate_matches_byte_count(self):
        # float32 = 4 bytes; ensure the helper returns n*K*|binary|*4.
        assert _estimate_fe_shared_buffer_bytes(100, 64, 18) == 100 * 64 * 18 * 4

    def test_can_hoist_returns_false_when_buffer_overshoots(self):
        # 100 TiB request never fits; result must be False regardless of host RAM.
        big = 100 * (2 ** 40)
        can, bb, av = _can_hoist_shared_buffer(big)
        assert can is False
        assert bb == big

    def test_can_hoist_returns_true_on_zero_byte_request(self):
        # Trivially within any budget.
        can, _bb, _av = _can_hoist_shared_buffer(0)
        assert can is True

    def test_dispatcher_picks_fallback_under_tight_budget(self, synthetic_pair_inputs, monkeypatch):
        """When budget ratio is zero, the hoist path can never be taken; the
        recompute fallback must run and still produce well-formed output."""
        monkeypatch.setattr(fe_mod, "_FE_BUFFER_RAM_BUDGET_RATIO", 0.0)
        res = _run_check(synthetic_pair_inputs, unary_preset="minimal", binary_preset="minimal")
        assert (0, 1) in res
        this_pair_features, transformed_vals, new_cols, _new_nbins, _msgs = res[(0, 1)]
        assert len(this_pair_features) >= 1, "Fallback path produced empty survivor set"
        assert transformed_vals is not None, "fe_max_steps>1 must materialise transformed_vals"
        assert transformed_vals.shape == (len(synthetic_pair_inputs["df"]), len(this_pair_features))
        # No NaN/Inf in the recomputed survivor columns.
        assert np.all(np.isfinite(transformed_vals))


class TestSubsampleMode:
    """subsample_n > 0 forces the MI sweep onto a row subset but survivor
    columns must still be produced at full n (mrmr.py contract)."""

    def _run_with_subsample(self, inputs, *, subsample_n: int):
        unary = create_unary_transformations(preset="minimal")
        binary = create_binary_transformations(preset="minimal")
        times_spent: dict = defaultdict(float)
        return check_prospective_fe_pairs(
            prospective_pairs=inputs["prospective_pairs"],
            X=inputs["df"],
            unary_transformations=unary,
            binary_transformations=binary,
            classes_y=inputs["classes_y"],
            classes_y_safe=inputs["classes_y_safe"],
            freqs_y=inputs["freqs_y"],
            num_fs_steps=0,
            cols=inputs["cols_names"],
            original_cols=inputs["original_cols"],
            fe_max_steps=2,
            fe_npermutations=1,
            fe_max_pair_features=4,
            fe_print_best_mis_only=True,
            fe_min_nonzero_confidence=0.0,
            fe_min_engineered_mi_prevalence=0.0,
            fe_good_to_best_feature_mi_threshold=0.5,
            fe_max_external_validation_factors=0,
            numeric_vars_to_consider=[0, 1, 2],
            quantization_nbins=4,
            quantization_method="quantile",
            quantization_dtype=np.int32,
            times_spent=times_spent,
            verbose=0,
            subsample_n=subsample_n,
        )

    def test_subsample_returns_full_n_survivor_columns(self, synthetic_pair_inputs):
        """Subsample n=120 of full n=200 -- the transformed_vals returned must still
        have shape[0]==200 because the caller (mrmr.py) appends to its full-n data
        array."""
        full_n = len(synthetic_pair_inputs["df"])
        assert full_n == 200, "synthetic fixture sanity"
        # Subsample fraction matters for MI estimation: too small and the noisy
        # MI estimate fails the engineered_mi_prevalence gate. 120/200 (60%) is
        # plenty for the synthetic strong-signal fixture.
        res = self._run_with_subsample(synthetic_pair_inputs, subsample_n=120)
        assert (0, 1) in res
        this_pair_features, transformed_vals, _new_cols, _new_nbins, _msgs = res[(0, 1)]
        assert len(this_pair_features) >= 1, "subsample path produced empty survivor set"
        assert transformed_vals is not None, "fe_max_steps>1 must materialise transformed_vals"
        # The critical contract: subsample is for MI sweep ONLY; output rows match FULL X.
        assert transformed_vals.shape[0] == full_n, (
            f"subsample mode must return full-n columns; got shape={transformed_vals.shape}"
        )
        # And the recompute must produce finite values (NaN -> 0 sanitisation runs).
        assert np.all(np.isfinite(transformed_vals))

    def test_subsample_disabled_when_n_geq_full_n(self, synthetic_pair_inputs):
        """subsample_n >= len(X) is a no-op: legacy full-data path runs."""
        full_n = len(synthetic_pair_inputs["df"])
        res_sub_full = self._run_with_subsample(synthetic_pair_inputs, subsample_n=full_n + 1)
        res_no_sub = self._run_with_subsample(synthetic_pair_inputs, subsample_n=0)
        # Same survivors regardless of the > full_n knob.
        for key in res_no_sub:
            assert key in res_sub_full
            cols_no_sub = sorted(res_no_sub[key][2])
            cols_sub_full = sorted(res_sub_full[key][2])
            assert cols_no_sub == cols_sub_full


class TestFastVsFallbackEquivalence:
    """Same inputs through both paths -> same survivor set and same column data."""

    def _normalise_features(self, res_pair):
        this_pair_features, transformed_vals, new_cols, _nbins, _msgs = res_pair
        # Use new_cols as the deterministic key; columns rebuilt from the same
        # (a_key, b_key, bin_func_name) metadata must match bit-for-bit up to
        # the float32 rounding of the binary ufunc (which is deterministic).
        return sorted(new_cols), transformed_vals

    @pytest.mark.fast
    def test_minimal_preset_paths_produce_identical_output(self, synthetic_pair_inputs, monkeypatch):
        # Fast path
        monkeypatch.setattr(fe_mod, "_FE_BUFFER_RAM_BUDGET_RATIO", 1.0)
        fast = _run_check(synthetic_pair_inputs, unary_preset="minimal", binary_preset="minimal")
        # Fallback path
        monkeypatch.setattr(fe_mod, "_FE_BUFFER_RAM_BUDGET_RATIO", 0.0)
        slow = _run_check(synthetic_pair_inputs, unary_preset="minimal", binary_preset="minimal")

        assert set(fast.keys()) == set(slow.keys()), "Different pair keys returned"
        for k in fast:
            fast_cols, fast_vals = self._normalise_features(fast[k])
            slow_cols, slow_vals = self._normalise_features(slow[k])
            assert fast_cols == slow_cols, f"{k}: survivor name set differs"
            # Compare contents column-by-column under the canonical name order
            # (both buffers may pack in a different idx order; align via new_cols).
            fast_names = fast[k][2]
            slow_names = slow[k][2]
            fast_lookup = {nm: fast_vals[:, idx] for idx, nm in enumerate(fast_names)}
            slow_lookup = {nm: slow_vals[:, idx] for idx, nm in enumerate(slow_names)}
            for nm in fast_lookup:
                np.testing.assert_array_equal(
                    fast_lookup[nm], slow_lookup[nm],
                    err_msg=f"{k}/{nm}: fast vs fallback column differs",
                )
