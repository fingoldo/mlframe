"""Phase-B audit regression tests for the mlframe.training subpackage.

Covers:
- splitting.make_train_test_split partition invariant
- remove_constant_columns keeps varying columns (NaN-preserving semantics)
- process_infinities leaves NaN intact (NaN preservation on mixed ±inf/NaN)
- create_linear_model("ridge", ...) smoke + ridge classifier variant
- is_linear_model / is_neural_model type detection
- save/load mlframe model round-trip (np + dict + SimpleNamespace)
- trusted_root path-escape guards on joblib.load sites (trainer + core metadata)
- preprocess_dataframe does not pass invalid `parallel="columns"` down to scan_parquet
"""

from __future__ import annotations

import os
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hypothesis import HealthCheck, given, settings, strategies as st


# ----------------------------------------------------------------------
# Hypothesis: splitting partition invariant
# ----------------------------------------------------------------------
@given(
    n=st.integers(min_value=20, max_value=400),
    test_size=st.floats(min_value=0.05, max_value=0.4),
    val_size=st.floats(min_value=0.05, max_value=0.4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_split_partition_invariant(n, test_size, val_size, seed):
    from mlframe.training.splitting import make_train_test_split

    df = pd.DataFrame({"x": np.arange(n)})
    tr, va, te, *_ = make_train_test_split(
        df,
        test_size=test_size,
        val_size=val_size,
        shuffle_val=True,
        shuffle_test=True,
        random_seed=seed,
    )
    # No overlap
    assert set(tr).isdisjoint(set(va))
    assert set(tr).isdisjoint(set(te))
    assert set(va).isdisjoint(set(te))
    # Union ⊆ [0, n)
    union = set(tr) | set(va) | set(te)
    assert union.issubset(set(range(n)))
    # Totals don't exceed input
    assert len(tr) + len(va) + len(te) <= n


# ----------------------------------------------------------------------
# Hypothesis: remove_constant_columns keeps varying columns
# ----------------------------------------------------------------------
@given(
    values=st.lists(st.floats(allow_nan=True, allow_infinity=False, width=32),
                    min_size=3, max_size=40),
)
@settings(max_examples=25, deadline=None)
def test_remove_constant_keeps_varying(values):
    from mlframe.training.utils import remove_constant_columns

    # Construct a DataFrame with one varying column and one all-equal column.
    arr = np.array(values, dtype=np.float32)
    varying = arr.copy()
    # Force variation: flip sign on second half unless values are all zeros / NaN.
    if np.all(np.isnan(arr)) or np.nanstd(arr) == 0:
        varying = np.linspace(0, 1, len(arr), dtype=np.float32)
    constant = np.full_like(arr, 7.0, dtype=np.float32)

    df = pd.DataFrame({"varying": varying, "constant_col": constant})
    out = remove_constant_columns(df, verbose=0)
    assert "varying" in out.columns
    assert "constant_col" not in out.columns


def test_remove_constant_numeric_with_nans_pandas():
    """[1.0, NaN, 1.0] should be flagged constant (Polars min==max semantics)."""
    from mlframe.training.utils import remove_constant_columns

    df = pd.DataFrame({
        "mixed": pd.Series([1.0, np.nan, 1.0], dtype="float64"),
        "ok": [1.0, 2.0, 3.0],
    })
    out = remove_constant_columns(df, verbose=0)
    assert "mixed" not in out.columns
    assert "ok" in out.columns


# ----------------------------------------------------------------------
# process_infinities: NaN preservation in mixed polars columns
# ----------------------------------------------------------------------
def test_process_infinities_polars_mixed():
    from mlframe.training.utils import process_infinities

    df = pl.DataFrame({"a": [1.0, float("inf"), float("nan"), float("-inf")]})
    out = process_infinities(df, fill_value=-999.0, verbose=0)
    col = out["a"].to_list()
    # ±inf replaced; NaN preserved (caller should run process_nans first if wanted).
    assert col[0] == 1.0
    assert col[1] == -999.0
    # NaN preservation:
    assert col[2] != col[2]  # NaN != NaN
    assert col[3] == -999.0


# ----------------------------------------------------------------------
# Linear model factory
# ----------------------------------------------------------------------
def test_create_linear_model_ridge_classifier():
    from mlframe.training.models import create_linear_model
    from mlframe.training.configs import LinearModelConfig
    from sklearn.linear_model import RidgeClassifier, Ridge

    cfg = LinearModelConfig(alpha=0.1, max_iter=500)
    reg = create_linear_model("ridge", cfg, use_regression=True)
    clf = create_linear_model("ridge", cfg, use_regression=False)
    assert isinstance(reg, Ridge)
    assert isinstance(clf, RidgeClassifier)


def test_is_linear_tree_neural_model():
    from mlframe.training.models import is_linear_model, is_tree_model, is_neural_model

    assert is_linear_model("Ridge")
    assert is_linear_model("lasso")
    assert not is_linear_model("cb")

    assert is_tree_model("cb")
    assert is_tree_model("LGB")
    assert not is_tree_model("ridge")

    assert is_neural_model("mlp")
    assert is_neural_model("NN")
    assert not is_neural_model("xgb")


# ----------------------------------------------------------------------
# Save/load mlframe model round-trip
# ----------------------------------------------------------------------
def test_save_load_mlframe_model(tmp_path):
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    obj = SimpleNamespace(
        arr=np.arange(10, dtype=np.float64),
        meta={"k": 1, "v": [1, 2, 3], "nested": {"a": "b"}},
        label="hello",
    )
    path = str(tmp_path / "m.zst")
    assert save_mlframe_model(obj, path, verbose=0) is True
    # dill's numpy-array reconstructor lives in `dill._dill`, which is intentionally
    # NOT on the _SafeUnpickler allowlist. For this round-trip we opt out of safe mode.
    loaded = load_mlframe_model(path, safe=False)
    assert loaded is not None
    assert isinstance(loaded, SimpleNamespace)
    np.testing.assert_array_equal(loaded.arr, obj.arr)
    assert loaded.meta == obj.meta
    assert loaded.label == obj.label


# ----------------------------------------------------------------------
# trusted_root guards
# ----------------------------------------------------------------------
def test_trainer_validate_trusted_path_rejects_escape(tmp_path):
    from mlframe.training.trainer import _validate_trusted_path

    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "other" / "evil.pkl"
    outside.parent.mkdir()
    outside.write_bytes(b"x")

    with pytest.raises(ValueError):
        _validate_trusted_path(str(outside), str(root))
    with pytest.raises(ValueError):
        _validate_trusted_path(str(outside), None)

    # Inside root: no raise
    inside = root / "ok.pkl"
    inside.write_bytes(b"x")
    _validate_trusted_path(str(inside), str(root))


def test_core_validate_trusted_path_rejects_escape(tmp_path):
    from mlframe.training.core import _validate_trusted_path

    root = tmp_path / "ok"
    root.mkdir()
    outside = tmp_path / "nope" / "meta.joblib"
    outside.parent.mkdir()
    outside.write_bytes(b"x")

    with pytest.raises(ValueError):
        _validate_trusted_path(str(outside), str(root))
    with pytest.raises(ValueError):
        _validate_trusted_path(str(outside), None)


# ----------------------------------------------------------------------
# preprocessing: scan_parquet no longer receives parallel="columns"
# ----------------------------------------------------------------------
def test_preprocess_does_not_pass_parallel_columns_to_scan(tmp_path):
    """Regression: `parallel='columns'` is only valid for eager read_parquet; forwarding it
    to `scan_parquet` previously raised TypeError. Writing a tiny parquet and loading via the
    scan path (no columns / no n_rows selector) must succeed."""
    from mlframe.training.preprocessing import load_and_prepare_dataframe, preprocess_dataframe
    from mlframe.training.configs import PreprocessingConfig

    # Build a small DataFrame with a constant + varying col, write to parquet.
    pdf = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "const": [7.0, 7.0, 7.0, 7.0]})
    p = tmp_path / "tiny.parquet"
    pdf.write_parquet(str(p))

    cfg = PreprocessingConfig()  # no columns / no n_rows → hits scan_parquet branch
    loaded = load_and_prepare_dataframe(str(p), cfg, verbose=0)
    assert isinstance(loaded, pl.DataFrame)
    out = preprocess_dataframe(loaded, cfg, verbose=0)
    assert "a" in out.columns
