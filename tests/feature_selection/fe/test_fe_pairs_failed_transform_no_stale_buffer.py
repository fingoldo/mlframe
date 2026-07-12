"""Regression: a binary transform that RAISES for one candidate column must not leave a
prior column's data in that buffer slot, and must not be recorded as a scored candidate.

Before the fix, ``_compute_one_fe_chunk`` (numpy-fallback path) logged the error and
``continue``-d without touching ``chunk_buffer[:, col]``, so the shared (cross-chunk-reused)
buffer slot kept whatever a previous column / chunk had written -- silently scoring stale
data if the slot were ever read by index. The fix nulls the slot (``= np.nan``) and excludes
the failed column from the candidate list.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunks import (
    _compute_one_fe_chunk,
)


def _stub_discretize_2d_quantile_batch(arr, *, n_bins, dtype, parallel, assume_finite):
    # codes don't matter for this test; just a shape-correct int array
    return np.zeros(arr.shape, dtype=dtype)


def _stub_batch_mi_kernel(*args, **kwargs):
    # Unused: the real MI dispatch is monkeypatched out below; this is only a placeholder
    # to satisfy the required kwarg.
    raise AssertionError("placeholder kernel must not be invoked")


def test_failed_transform_nulls_slot_and_excludes_candidate(monkeypatch):
    # Replace the heavy njit MI dispatch with a shape-correct stub: this test exercises the
    # numpy-fallback materialise/buffer logic, not the MI kernel.
    import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunks as _chunks

    monkeypatch.setattr(
        _chunks, "_dispatch_batch_mi_with_noise_gate",
        lambda *, disc_2d, **kw: np.zeros(disc_2d.shape[1], dtype=np.float64),
    )

    n = 64
    rng = np.random.default_rng(0)
    # transformed_vars: two operand columns referenced by the comb below.
    transformed_vars = rng.standard_normal((n, 2)).astype(np.float32)
    # Operand keys are (raw_var, unary_name) tuples; ``tp[i][1]`` is the unary name.
    op_a = ("a", "id")
    op_b = ("b", "id")
    vars_transformations = {op_a: 0, op_b: 1}

    # ONE pair, ONE valid comb -> candidate order is (comb x bin_func).
    raw_vars_pair = ("a", "b")
    transformations_pair = (op_a, op_b)
    pair_valid_combs = {raw_vars_pair: [transformations_pair]}

    good_value = np.full(n, 7.0, dtype=np.float32)

    def good_op(pa, pb):
        return good_value

    def bad_op(pa, pb):
        raise RuntimeError("simulated transform failure")

    # "good" first (col 0, succeeds), "bad" LAST (col 1, raises -> slot 1 never overwritten).
    # Custom names (not in the njit op registry) force the numpy-fallback branch under test.
    binary_transformations = {"goodxx": good_op, "badxx": bad_op}

    # Shared buffer pre-seeded with a COPY of the good column in slot 1: this is exactly the
    # stale-data scenario (a prior chunk's good column sitting in the slot the failing op targets).
    chunk_buffer = np.empty((n, 2), dtype=np.float32)
    chunk_buffer[:, 0] = -1.0
    chunk_buffer[:, 1] = good_value  # stale copy of a good column

    out = _compute_one_fe_chunk(
        chunk_pairs=[raw_vars_pair],
        pair_valid_combs=pair_valid_combs,
        chunk_buffer=chunk_buffer,
        vars_transformations=vars_transformations,
        transformed_vars=transformed_vars,
        binary_transformations=binary_transformations,
        quantization_nbins=4,
        quantization_dtype=np.int8,
        classes_y=None,
        classes_y_safe=None,
        freqs_y=None,
        fe_npermutations=0,
        fe_min_nonzero_confidence=0.0,
        batch_mi_kernel=_stub_batch_mi_kernel,
        use_su=False,
        prewarp_unary="__prewarp__",
        logger=__import__("logging").getLogger(__name__),
        discretize_2d_quantile_batch=_stub_discretize_2d_quantile_batch,
        serial_main_thread=True,
        # "goodxx"/"badxx" are not in the njit op registry -> the caller-hoisted op-code table for
        # this pool is None (forces the numpy-fallback branch under test, same as before this was a
        # caller-hoisted parameter instead of an internal per-chunk ``_njit_binary_op_codes`` call).
        op_code_arr=None,
        gpu_mat_enabled=True,
    )

    candidates, _fe_mi, _times = out[raw_vars_pair]

    # The failing op must NOT appear as a scored candidate.
    cand_op_names = {bin_func_name for (_tp, bin_func_name, _col, _pw) in candidates}
    assert "badxx" not in cand_op_names, "failed transform must not be recorded as a candidate"
    assert "goodxx" in cand_op_names, "good transform should still be recorded"

    # Exactly one candidate, at buffer column 0.
    assert len(candidates) == 1
    assert candidates[0][2] == 0

    # The failed slot (column 1) must NOT still hold the stale copy of the good column.
    # Post-fix it was written to NaN, then the vectorised nan_to_num scrubbed it to 0.0;
    # either way it must differ from the stale good-column data it would have retained pre-fix.
    failed_slot = chunk_buffer[:, 1]
    assert not np.array_equal(failed_slot, good_value), (
        "failed column slot still holds a copy of a good column's data (stale buffer not nulled)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
