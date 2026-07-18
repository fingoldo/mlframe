"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``_cheap_mi_with_y`` (the GROUP pre-selection cheap-MI prescreen
in ``_binned_numeric_agg_fe.py``) previously re-uploaded the target's y-codes via a raw ``cp.asarray`` on
EVERY candidate group column, and AGAIN via a separate raw upload in the survivor-stage device gate
(``local_mi_gate_binagg_resident``, ``_binned_numeric_agg_resident.py``) under a DIFFERENT role
(``"y_mi_classif"``). Both now route through ``resident_operand`` under the SAME role string, so identical
y-code content shares ONE resident device buffer instead of two-plus separate uploads.

Skips when cupy is unavailable (CI without a GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    SUPPORTED_STATS,
    _cheap_mi_with_y,
    fit_binned_numeric_agg,
)
from mlframe.feature_selection.filters._binned_numeric_agg_resident import local_mi_gate_binagg_resident
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def test_cheap_mi_with_y_dedups_ycodes_upload_across_candidates(monkeypatch):
    """Two _cheap_mi_with_y calls scoring DIFFERENT candidate group columns against the SAME y_codes
    (the realistic gcands loop in binned_numeric_agg_with_recipes) must upload the y-codes ONLY ONCE."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_FE_CHEAP_MI_GPU", "1")

    rng = np.random.default_rng(5)
    n = 4000
    y = rng.integers(0, 3, size=n)
    _, y_codes = np.unique(np.asarray(y, dtype=np.float64), return_inverse=True)
    y_codes = y_codes.astype(np.int64)

    col_a = rng.normal(size=n)
    col_b = rng.normal(size=n)

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        # Match on ACTUAL CONTENT (not just shape/dtype): a shape/dtype-only match would also count the
        # (n,) int64 fold_ids upload the resident gate does under a DIFFERENT role ("binagg_foldids"),
        # which happens to share y_codes' (n,) int64 shape but not its values.
        """Counting asarray."""
        if isinstance(arr, np.ndarray) and arr.shape == y_codes.shape and arr.dtype == y_codes.dtype and np.array_equal(arr, y_codes):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    mi_a = _cheap_mi_with_y(col_a, y_codes, nbins=10)
    mi_b = _cheap_mi_with_y(col_b, y_codes, nbins=10)

    assert upload_calls["n"] == 1, f"y-codes upload called {upload_calls['n']} times across 2 _cheap_mi_with_y calls (expected 1)"
    assert mi_a >= 0.0 and mi_b >= 0.0


def test_cheap_mi_and_resident_gate_share_one_ycodes_upload(monkeypatch):
    """_cheap_mi_with_y's yc_d (role 'y_mi_classif') and local_mi_gate_binagg_resident's y_gpu (SAME role)
    must share ONE resident upload for a classification target: both reduce y to the identical
    np.unique(y_as_float64, return_inverse=True) class codes (confirmed byte-identical for int y)."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_FE_CHEAP_MI_GPU", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")

    rng = np.random.default_rng(9)
    n = 4000
    g = rng.uniform(0.0, 1.0, n)
    aux = rng.normal(size=n)
    X = pd.DataFrame({"g": g, "aux": aux})
    y = rng.integers(0, 3, size=n)  # classification target -> low-cardinality codes at BOTH call sites

    _, y_codes = np.unique(np.asarray(y, dtype=np.float64), return_inverse=True)
    y_codes = y_codes.astype(np.int64)

    feat_df, recipes = fit_binned_numeric_agg(
        X,
        y,
        group_num_cols=["g"],
        agg_num_cols=["aux"],
        stats=SUPPORTED_STATS,
        nbins_base=10,
        n_folds=5,
        random_state=0,
    )
    assert feat_df.shape[1] > 0

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        # Match on ACTUAL CONTENT (not just shape/dtype): a shape/dtype-only match would also count the
        # (n,) int64 fold_ids upload the resident gate does under a DIFFERENT role ("binagg_foldids"),
        # which happens to share y_codes' (n,) int64 shape but not its values.
        """Counting asarray."""
        if isinstance(arr, np.ndarray) and arr.shape == y_codes.shape and arr.dtype == y_codes.dtype and np.array_equal(arr, y_codes):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    _mi = _cheap_mi_with_y(X["g"].to_numpy(), y_codes, nbins=10)
    keep = local_mi_gate_binagg_resident(feat_df, y, raw_X=X, recipes=recipes, n_folds=5, random_state=0)

    assert keep is not None, "resident gate returned None (GPU path unavailable under STRICT)"
    assert upload_calls["n"] == 1, (
        f"y-codes upload called {upload_calls['n']} times across _cheap_mi_with_y + "
        "local_mi_gate_binagg_resident on the same target (expected 1 -- cross-function share)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
