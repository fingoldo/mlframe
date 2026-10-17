"""Regression tests for the h3 code-audit fixes (failure sentinels, explicit parameters, radix pass refactor)."""
import numpy as np
import pandas as pd
import pytest


def test_radix_sort_f32_matches_numpy_sort():
    """The refactored 4-pass radix sort must still equal np.sort, including negatives and zeros."""
    from mlframe.feature_selection.filters.discretization._quantile_radix import _radix_sort_f32

    rng = np.random.default_rng(0)
    col = np.concatenate([rng.normal(size=500), [0.0, -0.0, -1e30, 1e30]]).astype(np.float32)
    expected = np.sort(col)
    n = col.shape[0]
    work = col.copy()
    _radix_sort_f32(work, np.empty(n, np.uint32), np.empty(n, np.uint32), np.zeros(256, np.int64))
    assert np.array_equal(work, expected)


def test_knockoffs_strict_lam_min_raises_on_collinear_input():
    """strict_lam_min=True is now a real parameter and raises on a near-singular design; the default only warns."""
    from mlframe.feature_selection.wrappers._knockoffs import make_gaussian_knockoffs

    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    X = np.column_stack([a, a, rng.normal(size=200)])
    with pytest.raises(ValueError, match="strict_lam_min"):
        make_gaussian_knockoffs(X, random_state=0, strict_lam_min=True)
    assert make_gaussian_knockoffs(X, random_state=0).shape == X.shape


def test_ransac_min_samples_is_a_config_field():
    """ransac_min_samples is read from LinearModelConfig and reaches the RANSAC estimator."""
    from mlframe.training._model_configs import LinearModelConfig
    from mlframe.training.models import _build_ransac_regressor

    assert _build_ransac_regressor(LinearModelConfig()).min_samples == 0.5
    assert _build_ransac_regressor(LinearModelConfig(ransac_min_samples=0.8)).min_samples == 0.8


def test_missing_indicator_group_kind_without_group_col_raises_once_for_all():
    """Group-kind columns without a group_col raise one ValueError listing every offender (not an assert on the first)."""
    from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation

    fit_state = {"group_col": None, "columns": {"a": {"kind": "group"}, "b": {"kind": "group"}}}
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
        apply_missing_indicator_imputation(df, fit_state)


def test_reporting_field_default_failure_not_cached(monkeypatch):
    """A failed lookup returns None without pinning None for the process; a later call succeeds."""
    from mlframe.training.reporting import _reporting_common as rc

    rc._reporting_field_default.__dict__.pop("no_such_field", None)
    assert rc._reporting_field_default("no_such_field") is None
    assert "no_such_field" not in rc._reporting_field_default.__dict__


def test_default_inner_spaces_returns_none_on_probe_failure():
    """A failed get_params probe is None (distinguishable from an empty space)."""
    from mlframe.training.composite.hpo import _default_inner_spaces

    class _Bad:
        """Estimator whose get_params raises."""

        def get_params(self):
            """Raise."""
            raise RuntimeError("boom")

    assert _default_inner_spaces(_Bad()) is None


def test_auto_base_top_k_override_is_a_parameter():
    """_auto_base accepts top_k_override so the cap no longer rewrites self.config."""
    import inspect

    from mlframe.training.composite.discovery._auto_base import _auto_base

    assert "top_k_override" in inspect.signature(_auto_base).parameters
