"""Cluster 28 regressions: NaT binning, the unpickle denylist, the drift cache key, the automl label default, quantile NaN rows, TTA NaN std (audit 2026-11-20, findings 28-F1, 28-F5, 28-F8, 28-F9, 28-F10, 28-F15)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.evaluation import _fixed_freq_bin_slices
from mlframe.training.feature_drift_report import _drift_invariant_cache_key
from mlframe.training.quantile_postproc import fix_quantile_crossing
from mlframe.training._tta import tta_predict
from mlframe.training.io import _resolve_denylisted_class
from mlframe.training._model_configs import AutoMLConfig


def test_fixed_freq_bins_ignore_nat_rows():
    """A NaT (int64 minimum) used to wrap every bin offset; NaT rows are dropped and the rest bin normally (28-F1)."""
    day = 86_400 * 10**9
    ts = np.array([0, day, np.iinfo(np.int64).min, day], dtype=np.int64)
    order, labels, starts, ends = _fixed_freq_bin_slices(ts, day)
    assert set(order.tolist()) == {0, 1, 3}
    assert len(labels) == 2
    assert (ends - starts).sum() == 3


def test_denylist_refuses_write_methods_on_dotted_names():
    """polars.DataFrame.write_* is a file writer and must be refused on the older loader path (28-F8, 28-F9)."""
    import dill

    with pytest.raises(dill.UnpicklingError):
        _resolve_denylisted_class("polars", "DataFrame.write_csv", lambda m, n: None)


def test_drift_cache_key_distinguishes_same_shape_frames_with_different_interiors():
    """Two frames that differ only in interior rows must not share one cached drift key (28-F15)."""
    a = pd.DataFrame({"x": np.arange(100.0)})
    b = a.copy()
    b.loc[50, "x"] = -7.0
    key_a = _drift_invariant_cache_key(a, a, a, None, 3.0)
    key_b = _drift_invariant_cache_key(b, a, a, None, 3.0)
    assert key_a != key_b


def test_automl_target_label_defaults_to_none_so_the_run_target_is_used():
    """The default used to be the literal "target", which overrode the real target name (28-F10)."""
    assert AutoMLConfig().automl_target_label is None


def test_quantile_sort_keeps_nan_rows_verbatim():
    """A NaN row used to be sorted into a finite order; NaN rows are returned as they were (28-F4)."""
    preds = np.array([[0.3, 0.1, 0.2], [np.nan, 0.5, 0.4]])
    alphas = np.array([0.1, 0.5, 0.9])
    with pytest.warns(RuntimeWarning):
        out = fix_quantile_crossing(preds, alphas, mode="sort")
    np.testing.assert_allclose(out[0], [0.1, 0.2, 0.3])
    assert np.isnan(out[1, 0])
    np.testing.assert_allclose(out[1, 1:], [0.5, 0.4])


def test_tta_jitter_stays_finite_when_a_column_holds_nan():
    """A plain std made the whole column NaN for every jittered pass; nanstd keeps the passes finite (28-F5)."""
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0], [6.0, 7.0]])
    out = tta_predict(lambda Z: np.nan_to_num(Z).sum(axis=1), X, n=4, seed=0)
    assert np.isfinite(np.asarray(out)).all()
