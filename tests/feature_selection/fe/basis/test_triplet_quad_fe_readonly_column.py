"""Regression: triplet/quadruplet cross-basis FE must not crash on a read-only input column.

``generate_triplet_cross_basis_features`` / ``generate_quadruplet_cross_basis_features`` pulled each leg via
``np.asarray(X[col].to_numpy(), dtype=_dt)``, which is a NO-OP (no copy) when the source column already has
dtype ``_dt`` -- e.g. a zero-copy Arrow-backed pandas view, or a frozen MRMR-fit-cache array reused via
``fe_append_columns`` from an earlier FE stage. The subsequent in-place ``np.copyto(x, ...)`` NaN-repair then
raised ``ValueError: assignment destination is read-only`` on any such column containing a NaN/Inf, degrading
the WHOLE triplet/quadruplet stage (the surrounding try/except in ``_mrmr_fit_impl`` swallows it and logs
"continuing without triplet-FE columns"). Fixed by switching to ``np.array(..., dtype=_dt)`` (always copies),
matching the sibling pair-cross (``_orth_pair_cross_fe.py``) and GPU-resident (``_gpu_resident_cross_basis.py``)
generators, which already used this pattern.

Surfaced by profiling/bug_hunt_fuzz_chains.py on a ``cats=1, input=polars_utf8, mrmr=True`` combo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_quadruplet_fe import (
    generate_quadruplet_cross_basis_features,
)
from mlframe.feature_selection.filters._orthogonal_triplet_fe import (
    generate_triplet_cross_basis_features,
)


def _readonly_nan_bearing_frame(n_cols: int, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A DataFrame whose float32 columns are backed by read-only ndarrays and contain one NaN each.

    float32 matches ``_crit_np_dtype()``'s default so ``np.asarray(..., dtype=_dt)`` is a true no-op (no copy)
    on these columns -- the exact condition that exposed the bug.
    """
    rng = np.random.default_rng(seed)
    arrs = {f"f{i}": rng.standard_normal(n).astype(np.float32) for i in range(n_cols)}
    arrs["f0"][5] = np.nan
    for a in arrs.values():
        a.flags.writeable = False
    return pd.DataFrame(arrs, copy=False)


def test_triplet_cross_basis_survives_readonly_nan_column():
    """Triplet FE on a read-only, NaN-bearing float32 column must not raise."""
    X = _readonly_nan_bearing_frame(n_cols=3)
    assert X["f0"].to_numpy().flags.writeable is False

    out = generate_triplet_cross_basis_features(X, triplets=[("f0", "f1", "f2")], basis="hermite", max_degree=2)

    assert out.shape[0] == len(X)
    assert out.shape[1] > 0, "the triplet stage produced no columns -- it silently dropped on the read-only NaN column"


def test_quadruplet_cross_basis_survives_readonly_nan_column():
    """Quadruplet FE on a read-only, NaN-bearing float32 column must not raise."""
    X = _readonly_nan_bearing_frame(n_cols=4)
    assert X["f0"].to_numpy().flags.writeable is False

    out = generate_quadruplet_cross_basis_features(X, quadruplets=[("f0", "f1", "f2", "f3")], basis="hermite", max_degree=1)

    assert out.shape[0] == len(X)
    assert out.shape[1] > 0, "the quadruplet stage produced no columns -- it silently dropped on the read-only NaN column"
