"""Regression tests for audits/full_audit_2026-07-21/core_infra_a.md (F1-F10).

PR1-PR3 (test-coverage proposals) are satisfied by the tests below; PR4 (comment clarity) and PR5
(DiskCache key-safety docs) are documentation-only, folded into the F2/F3 code comments. PR6
(composite_similarity biz_val) is out of scope (no live finding, conditionally phrased in the report).
PR7 (default_fingerprint cardinality-loop vectorization) was benchmarked and REJECTED -- see the module's
own bench note; no code change, nothing to test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.core.arrays import topk_by_partition
from mlframe.core.binning import apply_bin_smoother, bin_smooth, fit_bin_smoother
from mlframe.core.helpers import get_model_best_iter
from mlframe.core.matrix_seriation import spectral_seriation
from mlframe.core.set_similarity import jaccard
from mlframe.estimators.custom import MyDecorrelator
from mlframe.estimators.pipelines import optimize_pipeline_by_gridsearch, replay_cv_results
from mlframe.utils.disk_cache import DiskCache
from mlframe.utils.misc import get_pipeline_last_element, hygienic_fit

# ----------------------------------------------------------------------
# F1 (P0) -- fit_bin_smoother/apply_bin_smoother/bin_smooth on a constant column.
# ----------------------------------------------------------------------


def test_f1_bin_smoother_constant_column_no_crash():
    """F1: bin smoother constant column no crash."""
    x = np.array([5.0] * 20)
    sm = fit_bin_smoother(x, n_bins=10, binning="quantile")
    out = apply_bin_smoother(x, sm, strategy="mean")
    assert np.allclose(out, 5.0)


@pytest.mark.parametrize("strategy", ["mean", "median", "boundary"])
def test_f1_bin_smoother_constant_column_all_strategies(strategy):
    """F1: bin smoother constant column all strategies."""
    x = np.array([3.5] * 15)
    sm = fit_bin_smoother(x, n_bins=10)
    out = apply_bin_smoother(x, sm, strategy=strategy)
    assert np.allclose(out, 3.5)


def test_f1_bin_smoother_constant_column_preserves_nan_passthrough():
    """F1: bin smoother constant column preserves nan passthrough."""
    x = np.array([5.0] * 18 + [np.nan, np.nan])
    sm = fit_bin_smoother(x, n_bins=10)
    out = apply_bin_smoother(x, sm)
    assert np.allclose(out[:18], 5.0)
    assert np.all(np.isnan(out[18:]))


def test_f1_bin_smooth_convenience_wrapper_constant_column():
    """F1: bin smooth convenience wrapper constant column."""
    x = np.array([-2.0] * 12)
    out = bin_smooth(x, n_bins=10)
    assert np.allclose(out, -2.0)


def test_f1_bin_smoother_uniform_binning_constant_column():
    """F1: bin smoother uniform binning constant column."""
    x = np.array([1.0] * 10)
    sm = fit_bin_smoother(x, n_bins=5, binning="uniform")
    out = apply_bin_smoother(x, sm)
    assert np.allclose(out, 1.0)


def test_f1_bin_smoother_non_constant_column_unaffected():
    """Regression guard: the constant-column special case must not change behaviour for real data."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    sm = fit_bin_smoother(x, n_bins=10)
    assert len(sm["edges"]) > 2
    out = apply_bin_smoother(x, sm)
    assert np.isfinite(out).all()


# ----------------------------------------------------------------------
# F2 -- topk_by_partition with axis=None on a multi-dimensional array.
# ----------------------------------------------------------------------


def test_f2_topk_by_partition_axis_none_2d_descending():
    """F2: topk by partition axis none 2d descending."""
    arr = np.array([[5.0, 1.0, 9.0], [3.0, 8.0, 2.0]])
    ind, val = topk_by_partition(arr, 3, axis=None, ascending=False)
    expected = np.sort(arr.ravel())[::-1][:3]
    assert np.allclose(val, expected)
    assert np.allclose(arr.ravel()[ind], val)


def test_f2_topk_by_partition_axis_none_2d_ascending():
    """F2: topk by partition axis none 2d ascending."""
    arr = np.array([[5.0, 1.0, 9.0], [3.0, 8.0, 2.0]])
    ind, val = topk_by_partition(arr, 4, axis=None, ascending=True)
    expected = np.sort(arr.ravel())[:4]
    assert np.allclose(val, expected)
    assert np.allclose(arr.ravel()[ind], val)


def test_f2_topk_by_partition_axis_0_and_1_still_correct():
    """Regression guard: the axis=None fix must not disturb the (already-correct) axis=0/1 paths."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(6, 5))
    _ind0, val0 = topk_by_partition(arr, 3, axis=0, ascending=False)
    for col in range(5):
        assert np.allclose(val0[:, col], np.sort(arr[:, col])[::-1][:3])
    _ind1, val1 = topk_by_partition(arr, 2, axis=1, ascending=False)
    for row in range(6):
        assert np.allclose(val1[row], np.sort(arr[row])[::-1][:2])


# ----------------------------------------------------------------------
# F3 -- DiskCache path-traversal-safe keys.
# ----------------------------------------------------------------------


def test_f3_disk_cache_rejects_path_traversal_key(tmp_path):
    """F3: disk cache rejects path traversal key."""
    sub_dir = tmp_path / "cache"
    cache = DiskCache(sub_dir)
    with pytest.raises(ValueError):
        cache.put("../evil", {"x": 1})
    assert not (tmp_path / "evil.pkl").exists()


def test_f3_disk_cache_rejects_absolute_path_key(tmp_path):
    """F3: disk cache rejects absolute path key."""
    sub_dir = tmp_path / "cache"
    cache = DiskCache(sub_dir)
    evil_target = str(tmp_path / "outside")
    with pytest.raises(ValueError):
        cache.put(evil_target, {"x": 1})


def test_f3_disk_cache_normal_hex_key_still_works(tmp_path):
    """F3: disk cache normal hex key still works."""
    cache = DiskCache(tmp_path / "cache")
    cache.put("abc123deadbeef", {"x": 1})  # pragma: allowlist secret -- hex-looking cache key, not a credential
    assert cache.get("abc123deadbeef") == {"x": 1}  # pragma: allowlist secret


# ----------------------------------------------------------------------
# F4 -- optimize_pipeline_by_gridsearch writes a sidecar replay_cv_results can load.
# ----------------------------------------------------------------------


def test_f4_gridsearch_dump_round_trips_through_replay(tmp_path):
    """F4: gridsearch dump round trips through replay."""
    def fake_cv_func(X, Y, title, **constants):
        """Fake CV function returning canned results for the dump/replay round trip."""
        return {"results": {"cv_results": {"model_a": {"metrics": {"root_mean_squared_error": [0.1, 0.2, 0.15]}}}}}

    optimize_pipeline_by_gridsearch(X=None, Y=None, title="audit_t", cv_func=fake_cv_func, output_dir=str(tmp_path))
    fname = tmp_path / "cv_results-audit_t.dump"
    assert fname.exists()
    assert (tmp_path / (fname.name + ".sha256")).exists(), "write_sidecar was not called after joblib.dump"
    out = replay_cv_results(str(fname))
    assert "results" in out


# ----------------------------------------------------------------------
# F5 -- get_model_best_iter falls through on a bad field value instead of re-raising.
# ----------------------------------------------------------------------


class _BadBestIterModel:
    """Bad Best Iter Model."""
    best_iteration_ = "not-an-int"
    tree_count_ = 42


class _AllBadModel:
    """All Bad Model."""
    best_iteration_ = "n/a"


def test_f5_get_model_best_iter_falls_through_to_tree_count():
    """F5: get model best iter falls through to tree count."""
    assert get_model_best_iter(_BadBestIterModel()) == 42


def test_f5_get_model_best_iter_returns_none_when_nothing_usable():
    """F5: get model best iter returns none when nothing usable."""
    assert get_model_best_iter(_AllBadModel()) is None


def test_f5_get_model_best_iter_normal_case_unaffected():
    """F5: get model best iter normal case unaffected."""
    class _GoodModel:
        """Good Model."""
        best_iteration_ = 17

    assert get_model_best_iter(_GoodModel()) == 17


# ----------------------------------------------------------------------
# F6 -- MyDecorrelator: fit/transform type mismatch no longer silently keeps all columns.
# ----------------------------------------------------------------------


def _correlated_frame():
    """Correlated frame."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.01, size=200)
    c = rng.normal(size=200)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_f6_decorrelator_fit_dataframe_transform_ndarray():
    """F6: decorrelator fit dataframe transform ndarray."""
    df = _correlated_frame()
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(df)
    assert len(dec.correlated_features_) == 1
    out = dec.transform(df.to_numpy())
    assert out.shape[1] == 2, f"expected 1 correlated column dropped, got shape {out.shape}"


def test_f6_decorrelator_fit_ndarray_transform_dataframe():
    """F6: decorrelator fit ndarray transform dataframe."""
    df = _correlated_frame()
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(df.to_numpy())
    out = dec.transform(df)
    assert out.shape[1] == 2


def test_f6_decorrelator_fit_transform_dataframe_both_ways():
    """F6: decorrelator fit transform dataframe both ways."""
    df = _correlated_frame()
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(df)
    out = dec.transform(df)
    assert list(out.columns) == ["a", "c"]


def test_f6_decorrelator_shape_mismatch_raises():
    """F6: decorrelator shape mismatch raises."""
    df = _correlated_frame()
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(df)
    with pytest.raises(ValueError):
        dec.transform(df.to_numpy()[:, :2])


# ----------------------------------------------------------------------
# F7 -- set_similarity._counts requires BOTH inputs boolean before taking the mask branch.
# ----------------------------------------------------------------------


def test_f7_mixed_bool_and_nonbool_falls_through_to_set_semantics():
    """F7: mixed bool and nonbool falls through to set semantics."""
    mask = np.array([True, False, True, True])
    ids = np.array([5, 10, 15, 20])
    result = jaccard(mask, ids)
    assert result == 0.0  # set({True, False}) & set({5,10,15,20}) is empty


def test_f7_both_boolean_masks_still_use_fast_path():
    """F7: both boolean masks still use fast path."""
    a = np.array([True, False, True])
    b = np.array([True, True, False])
    result = jaccard(a, b)
    assert abs(result - (1.0 / 3.0)) < 1e-9


def test_f7_mismatched_length_boolean_masks_still_raise():
    """F7: mismatched length boolean masks still raise."""
    a = np.array([True, False, True])
    b = np.array([True, True])
    with pytest.raises(ValueError):
        jaccard(a, b)


# ----------------------------------------------------------------------
# F8 -- get_pipeline_last_element on an empty Pipeline raises a clear error.
# ----------------------------------------------------------------------


class _FakeEmptyPipeline:
    """Fake Empty Pipeline."""
    named_steps: dict = {}


class _FakePipeline:
    """Fake Pipeline."""
    def __init__(self, steps):
        self.named_steps = steps


def test_f8_empty_pipeline_raises_clear_value_error():
    """F8: empty pipeline raises clear value error."""
    with pytest.raises(ValueError, match="empty"):
        get_pipeline_last_element(_FakeEmptyPipeline())


def test_f8_nonempty_pipeline_returns_last_step():
    """F8: nonempty pipeline returns last step."""
    pipe = _FakePipeline({"a": "step_a", "b": "step_b", "c": "step_c"})
    assert get_pipeline_last_element(pipe) == "step_c"


# ----------------------------------------------------------------------
# F9 -- hygienic_fit restores column schema for polars DataFrames too, not just pandas.
# ----------------------------------------------------------------------


def test_f9_hygienic_fit_restores_pandas_schema():
    """F9: hygienic fit restores pandas schema."""
    class Sel:
        """Minimal selector stub used to check schema restoration after fit()."""

        @hygienic_fit
        def fit(self, X):
            """No-op / recording stub matching the estimator's fit() signature."""
            X["engineered"] = 1
            return self

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    Sel().fit(df)
    assert list(df.columns) == ["a", "b"]


def test_f9_hygienic_fit_restores_polars_schema():
    """F9: hygienic fit restores polars schema."""
    pl = pytest.importorskip("polars")

    class Sel:
        """Minimal selector stub used to check schema restoration after fit()."""

        @hygienic_fit
        def fit(self, X):
            """No-op / recording stub matching the estimator's fit() signature."""
            X.hstack([pl.Series("engineered", [9, 9, 9])], in_place=True)
            return self

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    Sel().fit(df)
    assert df.columns == ["a", "b"], f"leaked columns into caller's polars frame: {df.columns}"


# ----------------------------------------------------------------------
# F10 -- spectral_seriation raises a clear ValueError on NaN/Inf input.
# ----------------------------------------------------------------------


def test_f10_spectral_seriation_rejects_nan():
    """F10: spectral seriation rejects nan."""
    M = np.eye(4)
    M[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        spectral_seriation(M)


def test_f10_spectral_seriation_rejects_inf():
    """F10: spectral seriation rejects inf."""
    M = np.eye(4)
    M[0, 1] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        spectral_seriation(M)


def test_f10_spectral_seriation_valid_matrix_unaffected():
    """F10: spectral seriation valid matrix unaffected."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    perm = spectral_seriation(M)
    assert perm.shape == (5,)
    assert set(perm.tolist()) == set(range(5))
