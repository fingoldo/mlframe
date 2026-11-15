"""The auto scorer's batched bootstrap values do not depend on how the columns are cut into blocks (the scorers are column-separable)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _orth_auto_scorer_fe as A


def _case(n=3000, k=7):
    """Finite float columns, an integer target, three bootstrap resamples."""
    rng = np.random.default_rng(0)
    columns = [rng.normal(size=n) for _ in range(k)]
    y = (columns[0] + rng.normal(size=n) > 0).astype(np.int64)
    boot_idx = [rng.integers(0, n, size=n) for _ in range(3)]
    hyper = {"nbins": 10, "n_neighbors": 3, "copula_n_bins": 20, "dcor_n_sample": 200, "n_jobs": 1}
    return columns, y, boot_idx, hyper


@pytest.mark.parametrize("width", [1, 3])
def test_column_blocks_give_the_values_of_one_call(monkeypatch, width):
    """With a block of one or three columns the values of every scorer equal the single-call ones."""
    columns, y, boot_idx, hyper = _case()
    whole = A._batched_bootstrap_values(columns, y, boot_idx, 0, hyper)
    assert whole is not None
    monkeypatch.setattr(A, "_BOOTSTRAP_BLOCK_BYTES", width * len(boot_idx[0]) * int(np.dtype(A._crit_np_dtype()).itemsize))
    blocked = A._batched_bootstrap_values_in_column_blocks(columns, y, boot_idx, 0, hyper)
    assert blocked is not None and set(blocked) == set(whole)
    for scorer, values in whole.items():
        np.testing.assert_allclose(blocked[scorer], values, rtol=1e-6, atol=1e-9, err_msg=scorer)
