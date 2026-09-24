"""The OOF holdout returns the same rows, in the same order, for a pandas and a polars training frame.

On the time-sorted holdout path the polars branch built X with a boolean mask (row order) while y followed the time order
(index order), so every holdout prediction was matched with another row's target (max |pred - y| 39 against 0 for pandas).
An identity component makes the check exact: its prediction is the row's own y, so any misalignment is nonzero.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import compute_oof_holdout_predictions


class _Identity(BaseEstimator, RegressorMixin):
    """Predicts the frame's ``y_copy`` column: the prediction of a row is its own target."""

    def fit(self, X, y, **_):
        """Nothing to learn."""
        return self

    def predict(self, X):
        """The row's own target."""
        return np.asarray(X["y_copy"].to_numpy() if hasattr(X, "to_numpy") else X["y_copy"], dtype=np.float64)


@pytest.mark.parametrize("carrier", ["pandas", "polars"])
@pytest.mark.parametrize("order", ["monotone", "reversed", "shuffled"])
@pytest.mark.parametrize("kfold", [1, 3])
def test_the_holdout_rows_of_X_match_the_holdout_targets(carrier, order, kfold):
    """For every carrier and time order, the identity component's OOF prediction equals the returned holdout y exactly."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(0.0, 10.0, n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "y_copy": y})
    X = pl.from_pandas(frame) if carrier == "polars" else frame
    t = {"monotone": np.arange(n), "reversed": np.arange(n)[::-1], "shuffled": rng.permutation(n)}[order].astype(float)
    P, y_h, names = compute_oof_holdout_predictions(
        component_models=[_Identity()], component_names=["id"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, time_ordering=t, kfold=kfold,
    )
    assert names == ["id"] and P.shape[0] == y_h.shape[0] > 0
    np.testing.assert_array_equal(P[:, 0], y_h)


def _row_slicers():
    """Every row-subsetting helper that branches on the frame type, as ``name -> fn(frame, idx)``."""
    from mlframe.training.composite.row_level_average_importance import _subset_rows
    from mlframe.training.core._phase_composite_post_xt_ensemble import _slice_frame_rows
    from mlframe.training.slicing._slice_helpers import _row_select

    return {"_subset_rows": _subset_rows, "_slice_frame_rows": _slice_frame_rows, "_row_select": _row_select}


@pytest.mark.parametrize("name", ["_subset_rows", "_slice_frame_rows", "_row_select"])
@pytest.mark.parametrize("order", ["monotone", "reversed", "shuffled"])
def test_row_slicers_return_the_same_rows_in_the_same_order_for_pandas_and_polars(name, order):
    """A polars frame gives the pandas rows, in the index's order (a boolean-mask filter returned them in row order)."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": np.arange(50, dtype=float), "b": rng.normal(size=50)})
    idx = {"monotone": np.arange(0, 50, 3), "reversed": np.arange(0, 50, 3)[::-1], "shuffled": rng.permutation(50)[:20]}[order]
    fn = _row_slicers()[name]
    got_pd = np.asarray(fn(frame, idx)["a"])
    got_pl = np.asarray(fn(pl.from_pandas(frame), idx)["a"].to_numpy())
    np.testing.assert_array_equal(got_pd, idx.astype(float))
    np.testing.assert_array_equal(got_pl, got_pd)


@pytest.mark.parametrize("carrier", ["pandas", "polars"])
@pytest.mark.parametrize("order", ["monotone", "shuffled"])
def test_the_external_holdout_rows_match_their_targets(carrier, order):
    """``oof_holdout_source='external_val'`` predicts on the caller's val frame; the returned y is that frame's, row for row.

    The K-fold and train-tail sources are covered above. This is the third source, where the holdout is a separate frame:
    a polars val frame gathered in a different order from its targets would misalign every weight the stack fits on it.
    """
    rng = np.random.default_rng(3)
    n, n_val = 200, 80
    y = rng.normal(0.0, 10.0, n)
    y_val = rng.normal(0.0, 10.0, n_val)
    train = pd.DataFrame({"x": rng.normal(size=n), "y_copy": y})
    val = pd.DataFrame({"x": rng.normal(size=n_val), "y_copy": y_val})
    if order == "shuffled":
        perm = rng.permutation(n_val)
        val, y_val = val.iloc[perm].reset_index(drop=True), y_val[perm]
    X, X_val = (pl.from_pandas(train), pl.from_pandas(val)) if carrier == "polars" else (train, val)
    P, y_h, names = compute_oof_holdout_predictions(
        component_models=[_Identity()], component_names=["id"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, external_holdout_X=X_val, external_holdout_y=y_val,
    )
    assert names == ["id"] and P.shape[0] == n_val
    np.testing.assert_array_equal(y_h, y_val)
    np.testing.assert_array_equal(P[:, 0], y_h)


# ---------------------------------------------------------------------------
# Every (frame, idx) helper that branches on polars is registered and checked; the shared order-losing scan finds none.
# ---------------------------------------------------------------------------


def _first_of_split(df, idx):
    """``create_split_dataframes``' train frame for ``idx`` (the val and test frames go through the same slicer)."""
    from mlframe.training.preprocessing import create_split_dataframes

    return create_split_dataframes(df, idx, idx, idx)[0]


# ``path::function`` (under src/mlframe/training) -> "module:attr" of a ``fn(frame, idx)`` to check, or a reason it is not one.
FRAME_ROW_SLICERS = {
    "_data_helpers.py::_subset_dataframe": "mlframe.training._data_helpers:_subset_dataframe",
    "composite/bagging.py::_take_rows": "mlframe.training.composite.bagging:_take_rows",
    "composite/classification_discovery.py::_take_rows": "mlframe.training.composite.classification_discovery:_take_rows",
    "composite/highlevel.py::_select_rows": "mlframe.training.composite.highlevel:_select_rows",
    "composite/meta.py::_row_subset": "mlframe.training.composite.meta:_row_subset",
    "composite/row_level_average_importance.py::_subset_rows": "mlframe.training.composite.row_level_average_importance:_subset_rows",
    "core/_phase_composite_post_xt_ensemble/_phase_composite_post_xt_mtr_oof.py::_slice_rows_by_idx":
        "mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof:_slice_rows_by_idx",
    "diagnostics/learning_curve.py::_take_rows": "mlframe.training.diagnostics.learning_curve:_take_rows",
    "preprocessing.py::create_split_dataframes": "tests.training.composite.ensemble.test_frame_carrier_parity:_first_of_split",
    "slicing/_slice_helpers.py::_row_select": "mlframe.training.slicing._slice_helpers:_row_select",
}
_NOT_ROW_SLICERS = {
    "_dataset_cache_fingerprint.py::_row_sample_hash": "hashes a row sample for a cache key; it returns no rows",
    "composite/_canonical_hash.py::row_order_fingerprint": "fingerprints the row order itself; it returns no rows",
    "neural/data.py::_extract": "a torch Dataset method: plain positional indexing into a tensor or array, no mask branch",
    "trainer.py::_row": "a closure (not importable); .iloc for pandas, plain positional df[idx] otherwise, no mask branch",
}
_FRAME_PARAM = re.compile(r"^(X|X_rows|df|frame|data|X_all|features)$")
_IDX_PARAM = re.compile(r"(^|_)(idx|rows|row_idx|indices|positions|sel)$")


def _frame_idx_helpers() -> set[str]:
    """Every function whose first two parameters are a frame and an index and whose body branches on polars."""
    import ast
    from pathlib import Path

    import mlframe

    root = Path(mlframe.__file__).resolve().parent / "training"
    out = set()
    for path in sorted(root.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        for f in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)):
            params = [a.arg for a in f.args.args if a.arg != "self"]
            if len(params) >= 2 and _FRAME_PARAM.match(params[0]) and _IDX_PARAM.search(params[1]):
                if re.search(r"polars|pl\.DataFrame|_is_polars|is_polars", ast.unparse(f)):
                    out.add(f"{path.relative_to(root).as_posix()}::{f.name}")
    return out


def test_every_frame_idx_helper_is_registered():
    """A new ``(frame, idx)`` helper with a polars branch joins the parity check below, or says why it is not a row slicer."""
    found = _frame_idx_helpers()
    listed = set(FRAME_ROW_SLICERS) | set(_NOT_ROW_SLICERS)
    assert found == listed, f"unregistered: {sorted(found - listed)}; stale: {sorted(listed - found)}"


@pytest.mark.parametrize("key", sorted(FRAME_ROW_SLICERS))
@pytest.mark.parametrize("order", ["monotone", "reversed", "shuffled"])
def test_every_registered_slicer_keeps_the_index_order_on_both_carriers(key, order):
    """pandas gives the rows in index order and polars gives the same rows in the same order."""
    import importlib

    mod, attr = FRAME_ROW_SLICERS[key].split(":")
    fn = getattr(importlib.import_module(mod), attr)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": np.arange(50, dtype=float), "b": rng.normal(size=50)})
    idx = {"monotone": np.arange(0, 50, 3), "reversed": np.arange(0, 50, 3)[::-1], "shuffled": rng.permutation(50)[:20]}[order]
    got_pd = np.asarray(fn(frame, idx)["a"], dtype=float)
    got_pl = np.asarray(fn(pl.from_pandas(frame), idx)["a"].to_numpy(), dtype=float)
    np.testing.assert_array_equal(got_pd, idx.astype(float))
    np.testing.assert_array_equal(got_pl, got_pd)


def test_no_function_selects_rows_by_a_mask_built_from_its_positional_index():
    """py_ci_shared.order_losing_filters over src/mlframe: no function pairs ``.iloc[idx]`` with a mask built from ``idx``.

    On the tree before the fix it flags the five helpers repaired then; it also found fit_stacked / fit_stacked_on_residual.
    """
    from pathlib import Path

    olf = pytest.importorskip("py_ci_shared.order_losing_filters")
    import mlframe

    root = Path(mlframe.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "_benchmarks" not in p.parts)
    found = olf.find_order_losing_filters(files, root.parent.parent)
    assert not found, found
