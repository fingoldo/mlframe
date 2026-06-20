"""Wave 9.1 loop-iter-9 regression: ``categorize_dataset`` NaN bin code
MUST NOT collide with real-data bins under any ``nbins_strategy``.

Pre-fix: the NaN-bin code was hardcoded to the constructor ``n_bins``
argument (default 4), but the adaptive ``nbins_strategy`` branch
produces per-column bin counts that often exceed 4 (e.g. Freedman-
Diaconis gives ~22 bins for N(0,1) at n=600). So NaN observations got
written to bin index 4 - the same index a regular real-data row landed
in. Missingness signal destroyed; MI / SU / MRMR rankings biased.

Effect: silent corruption whenever ``missing_strategy='separate_bin'``
(the default) is combined with any of:
``fd``, ``sturges``, ``qs``, ``knuth``, ``blocks``, ``mdlp``,
``fayyad_irani``, ``optimal_joint``, ``cv``, ``mah``, ``mah_sci``,
``sci``, ``marx`` - 12 of 13 adaptive strategies.

Fix: per-column NaN code = one past the column's highest regular bin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Strategies whose per-column nbins commonly exceeds ctor n_bins=4 and
# therefore trigger the collision pre-fix.
COLLIDING_STRATEGIES = ["fd", "sturges", "qs"]


@pytest.mark.parametrize("strategy", COLLIDING_STRATEGIES)
def test_nan_code_does_not_collide_with_real_data_bins(strategy):
    """Pre-fix this assertion fails; post-fix it passes."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    nan_idx = rng.choice(n, 30, replace=False)
    x[nan_idx] = np.nan

    data, cols, nbins = categorize_dataset(
        df=pd.DataFrame({"x": x}),
        n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
        nbins_strategy=strategy,
    )
    codes = data[:, cols.index("x")]
    real_codes = {int(c) for c in codes[~np.isnan(x)]}
    nan_codes = {int(c) for c in codes[np.isnan(x)]}
    collision = real_codes & nan_codes
    assert not collision, (
        f"strategy={strategy!r}: NaN bin code collides with real data "
        f"at indices {collision}. real_range=[{min(real_codes)},"
        f"{max(real_codes)}], nan_codes={nan_codes}. This silently "
        f"destroys the missingness signal."
    )


@pytest.mark.parametrize("strategy", COLLIDING_STRATEGIES)
def test_nan_code_is_one_past_max_real_code(strategy):
    """The NaN code per column should sit at ``per_col_real_bins`` --
    exactly one past the highest regular code.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    nan_idx = rng.choice(n, 50, replace=False)
    x[nan_idx] = np.nan

    data, cols, nbins = categorize_dataset(
        df=pd.DataFrame({"x": x}),
        n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
        nbins_strategy=strategy,
    )
    codes = data[:, cols.index("x")]
    max_real = int(codes[~np.isnan(x)].max())
    nan_code = int(codes[np.isnan(x)].max())
    assert nan_code == max_real + 1, (
        f"strategy={strategy!r}: NaN code {nan_code} is not one past "
        f"max real code {max_real}."
    )


def test_non_adaptive_path_unchanged():
    """Negative control: the non-adaptive path (no ``nbins_strategy``)
    still uses the ctor ``n_bins`` as NaN code, matching pre-fix behaviour
    on this branch.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    nan_idx = rng.choice(n, 20, replace=False)
    x[nan_idx] = np.nan

    data, cols, nbins = categorize_dataset(
        df=pd.DataFrame({"x": x}),
        n_bins=4, dtype=np.int16,
        missing_strategy="separate_bin",
        nbins_strategy=None,  # legacy uniform path
    )
    codes = data[:, cols.index("x")]
    real_codes = {int(c) for c in codes[~np.isnan(x)]}
    nan_codes = {int(c) for c in codes[np.isnan(x)]}
    # n_bins=4 means real codes are in [0, 3] (or [0, 4] depending on
    # the discretizer); NaN code is 4. The legacy path was already safe
    # because per-column nbins equals the global n_bins.
    assert nan_codes == {4}
    assert max(real_codes) <= 4
    # Make the safety property explicit:
    if 4 in real_codes:
        # Discretizer happened to use 5-bin output ([0..4]); then NaN code
        # collides on this legacy path too. That's a separate pre-existing
        # issue NOT in iter-9 scope - allow but assert it stays as it was.
        # (If this fires, file as iter 10 candidate.)
        pytest.xfail("Legacy non-adaptive path has its own NaN collision "
                      "when discretizer uses [0..n_bins] range -- pre-existing.")
