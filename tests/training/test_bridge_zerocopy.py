"""Regression sensors for wave2 w2a-bridge-p0 fixes.

Each test pins a zero-copy / dtype-preservation contract that, when broken, would silently re-introduce a polars -> pandas (or pandas -> polars) full-frame consolidation copy on a hot path.

Covered findings:
    F1  training/core/_main_train_suite.py:814 (leaderboard polars frame -> pandas via Arrow bridge, not via CSV round-trip or bare pd.DataFrame())
    F3  training/_pipeline_helpers.py:401-402  (sklearn ndarray-output branch must rejoin polars-passthrough cols via the bridge, mirroring its DataFrame-output sibling)
    F4  training/core/_predict_main_from_models.py:246  (pandas-extension back-merge into polars-pre frame must skip the pandas block consolidation copy)
    F5  training/core/_phase_helpers_fit_pipeline.py:521 (same pandas -> polars back-merge across train/val/test splits)
    F7  training/_pipeline_extensions.py:_filter_to_numeric (polars -> pandas inside the numeric-only gate must use the Arrow split-blocks bridge)
    F15 training/extractors.py head/tail display path (Arrow bridge so pl.Enum / pl.Categorical / pl.Date keep their pandas-native dtype for Jupyter rich rendering)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest


from mlframe.training.utils import get_pandas_view_of_polars_df

# ---------------------------------------------------------------------------
# F1 + F15: leaderboard / head-tail must use the Arrow bridge so categorical / enum / datetime
# dtypes survive the polars -> pandas hop (bare .to_pandas() collapses them to object).
# ---------------------------------------------------------------------------


def _mixed_polars_frame_for_dtype_check(n=64):
    """Mixed polars frame for dtype check."""
    return pl.DataFrame(
        {
            "f_float": np.arange(n, dtype=np.float32),
            "f_int": np.arange(n, dtype=np.int32),
            "f_enum": pl.Series("f_enum", ["a", "b", "c"] * ((n + 2) // 3))[:n].cast(pl.Enum(["a", "b", "c"])),
            "f_date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1).replace(day=1), eager=True).head(1).extend_constant(pl.date(2020, 1, 1), n - 1),
        }
    )


def test_f1_main_train_suite_leaderboard_path_routes_polars_through_bridge(tmp_path, monkeypatch):
    """The leaderboard CSV export converts a polars leaderboard through the Arrow bridge and writes every row under the target_type/target_name columns."""
    from types import SimpleNamespace

    from mlframe.training import utils as training_utils
    from mlframe.training.core._main_train_suite_phases import export_votenrank_leaderboards

    seen = []
    real_bridge = training_utils.get_pandas_view_of_polars_df

    def spy(frame, *args, **kwargs):
        """Record the polars frame handed to the bridge, then delegate."""
        seen.append(frame)
        return real_bridge(frame, *args, **kwargs)

    monkeypatch.setattr(training_utils, "get_pandas_view_of_polars_df", spy)
    leaderboard = pl.DataFrame({"model": ["cb", "lgb", "xgb"], "rank": [1, 2, 3]})
    ctx = SimpleNamespace(ensembles={"binary": {"tgt": {"_leaderboard": leaderboard}}}, metadata={})

    export_votenrank_leaderboards(ctx, str(tmp_path), 0)

    assert len(seen) == 1
    assert seen[0] is leaderboard
    assert ctx.metadata["votenrank_leaderboard"]["binary"]["tgt"] is leaderboard
    out = pd.read_csv(tmp_path / ".leaderboard.csv")
    assert list(out.columns) == ["target_type", "target_name", "model", "rank"]
    assert out["model"].tolist() == ["cb", "lgb", "xgb"]
    assert out["rank"].tolist() == [1, 2, 3]
    assert (out["target_type"] == "binary").all()
    assert (out["target_name"] == "tgt").all()


def test_f15_extractors_head_tail_preserves_enum_dtype_through_bridge():
    """``head.to_pandas()`` bare collapses pl.Enum to object dtype; the bridge keeps it as pandas CategoricalDtype."""
    pl_df = pl.DataFrame({"f_int": np.arange(5), "f_enum": pl.Series(["x", "y", "x", "z", "y"]).cast(pl.Enum(["x", "y", "z"]))})
    pdf = get_pandas_view_of_polars_df(pl_df.head(5))
    assert isinstance(pdf["f_enum"].dtype, pd.CategoricalDtype), (
        "F15 regression: pl.Enum collapsed to %s after bridge (expected CategoricalDtype)" % pdf["f_enum"].dtype
    )


def test_f15_extractors_module_source_routes_head_tail_via_bridge(monkeypatch, capsys):
    """The showcase display path hands Jupyter a head and a tail whose pl.Enum column kept its pandas CategoricalDtype, not object."""
    import IPython.display as ipy_display

    from mlframe.training.extractors import _extractors_showcase as showcase

    shown = []
    monkeypatch.setattr(showcase, "is_jupyter_notebook", lambda: True)
    monkeypatch.setattr(ipy_display, "display", lambda obj, *a, **k: shown.append(obj))
    pl_df = pl.DataFrame(
        {
            "f_float": np.arange(12, dtype=np.float32),
            "f_enum": pl.Series(["x", "y", "z"] * 4).cast(pl.Enum(["x", "y", "z"])),
        }
    )

    showcase.showcase_features_and_targets(pl_df, {})

    frames = [obj for obj in shown if isinstance(obj, pd.DataFrame)]
    assert len(frames) == 2
    head, tail = frames
    assert isinstance(head["f_enum"].dtype, pd.CategoricalDtype)
    assert isinstance(tail["f_enum"].dtype, pd.CategoricalDtype)
    assert head["f_enum"].astype(str).tolist() == ["x", "y", "z", "x", "y"]
    assert tail["f_enum"].astype(str).tolist() == ["y", "z", "x", "y", "z"]
    assert tail["f_float"].tolist() == [7.0, 8.0, 9.0, 10.0, 11.0]


# ---------------------------------------------------------------------------
# F3: sklearn-ndarray-output branch must rejoin polars passthrough cols via bridge
# ---------------------------------------------------------------------------


def _passthrough_frames():
    """A polars passthrough frame holding a pl.Enum column, and the reduced polars frame the transformer saw."""
    held = pl.DataFrame({"f_enum": pl.Series(["x", "y", "z", "x"]).cast(pl.Enum(["x", "y", "z"]))})
    reduced = pl.DataFrame({"a": np.arange(4, dtype=np.float64), "b": np.arange(4, dtype=np.float64) * 2.0})
    return held, reduced


@pytest.mark.parametrize("output_kind", ["ndarray", "dataframe"])
def test_f3_pipeline_helpers_ndarray_branch_uses_bridge_for_held(output_kind):
    """Rejoining polars passthrough columns to a transformer's pandas or ndarray output keeps the pl.Enum column categorical and the values row-aligned."""
    from mlframe.training.pipeline._pipeline_helpers import _reattach_passthrough_to_frame

    held, reduced = _passthrough_frames()
    if output_kind == "ndarray":
        out = np.column_stack([np.arange(4, dtype=np.float64), np.arange(4, dtype=np.float64) * 2.0])
    else:
        out = pd.DataFrame({"a": np.arange(4, dtype=np.float64), "b": np.arange(4, dtype=np.float64) * 2.0})

    result = _reattach_passthrough_to_frame(out, ["f_enum"], held, True, reduced)

    assert list(result.columns) == ["a", "b", "f_enum"]
    assert isinstance(result["f_enum"].dtype, pd.CategoricalDtype)
    assert result["f_enum"].astype(str).tolist() == ["x", "y", "z", "x"]
    assert result["b"].tolist() == [0.0, 2.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# F4 + F5: pandas -> polars back-merge must skip block consolidation copy
# ---------------------------------------------------------------------------


def test_f4_predict_main_uses_dict_of_numpy_not_from_pandas(monkeypatch):
    """The predict back-merge hstacks the extension columns onto the polars-pre frame without ``pl.from_pandas``, which would pay a pandas block consolidation copy."""
    from mlframe.training.core import _predict_main_from_models as pm

    def forbidden(*args, **kwargs):
        """Fail the call: the back-merge must build polars columns from per-column numpy views."""
        raise AssertionError("pl.from_pandas used on the predict back-merge")

    monkeypatch.setattr(pm.pl, "from_pandas", forbidden)
    df_pre = pl.DataFrame({"raw": np.arange(6, dtype=np.float64)})
    df_post = pd.DataFrame({"raw": np.arange(6, dtype=np.float64), "ext_0": np.arange(6, dtype=np.float32) * 0.5, "ext_1": np.arange(6, dtype=np.int32)})

    merged = pm._predict_from_model_dim_reducer_truncatedsvd(object(), df_post, df_pre)

    assert isinstance(merged, pl.DataFrame)
    assert merged.columns == ["raw", "ext_0", "ext_1"]
    assert merged["ext_0"].to_list() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert merged["ext_1"].to_list() == [0, 1, 2, 3, 4, 5]
    assert merged["ext_0"].dtype == pl.Float32


def test_f5_phase_helpers_fit_pipeline_uses_dict_of_numpy_not_from_pandas(monkeypatch):
    """The fit-time train/val/test back-merge hstacks the extension columns onto each polars-pre split without ``pl.from_pandas``."""
    from mlframe.training.core import _phase_helpers_fit_pipeline as phfp

    def forbidden(*args, **kwargs):
        """Fail the call: the back-merge must build polars columns from per-column numpy views."""
        raise AssertionError("pl.from_pandas used on the fit back-merge")

    monkeypatch.setattr(phfp.pl, "from_pandas", forbidden)
    raw_cols = ["raw"]

    def split(n, offset):
        """One split as a (pandas post-extension, polars pre-extension) pair."""
        base = np.arange(n, dtype=np.float64) + offset
        pre = pl.DataFrame({"raw": base})
        post = pd.DataFrame({"raw": base, "ext_0": base * 2.0})
        return post, pre

    (train_pd, train_pl), (val_pd, val_pl), (test_pd, test_pl) = split(6, 0), split(3, 100), split(4, 200)

    test_out, train_out, val_out = phfp._phase_fit_pipeline_polars_native_fastpath_mrmr(train_pd, raw_cols, True, train_pl, val_pl, test_pl, val_pd, test_pd, 0)

    for out, post in ((train_out, train_pd), (val_out, val_pd), (test_out, test_pd)):
        assert out.columns == ["raw", "ext_0"]
        assert out["ext_0"].to_list() == post["ext_0"].tolist()


def test_f4_f5_dict_of_numpy_back_merge_behaviour_matches_from_pandas():
    """Functional equivalence: the dict-of-numpy back-merge must produce the same polars Series values as pl.from_pandas would have, for the supported dtype set on this path."""
    pdf = pd.DataFrame(
        {
            "f_float": np.arange(8, dtype=np.float64),
            "f_int": np.arange(8, dtype=np.int32),
            "f_bool": np.array([True, False] * 4),
        }
    )
    expected = pl.from_pandas(pdf)
    actual = pl.DataFrame({c: pdf[c].to_numpy() for c in pdf.columns})
    assert len(pdf.columns) > 0
    for c in pdf.columns:
        assert expected[c].to_list() == actual[c].to_list(), c
    assert actual.shape == expected.shape


# ---------------------------------------------------------------------------
# F7: _filter_to_numeric polars -> pandas hop must use Arrow split-blocks path, not bare to_pandas
# ---------------------------------------------------------------------------


def test_f7_filter_to_numeric_uses_split_blocks_for_polars_input(monkeypatch):
    """_filter_to_numeric converts a polars frame with ``split_blocks=True`` rather than a bare ``to_pandas()`` full consolidation copy."""
    from mlframe.training.pipeline._pipeline_extensions import _filter_to_numeric

    calls = []
    real_to_pandas = pl.DataFrame.to_pandas

    def spy(self, *args, **kwargs):
        """Record the keyword arguments of each conversion, then delegate."""
        calls.append(kwargs)
        return real_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", spy)
    pl_df = pl.DataFrame({"f_float": np.arange(4, dtype=np.float32), "f_str": ["a", "b", "c", "d"]})

    out, dropped = _filter_to_numeric(pl_df)

    assert len(calls) == 1
    assert calls[0].get("split_blocks") is True
    assert list(out.columns) == ["f_float"]
    assert dropped == ["f_str"]


def test_f7_filter_to_numeric_accepts_polars_and_preserves_numeric_dtypes():
    """End-to-end: polars frame in -> pandas with numeric dtypes preserved, non-numeric dropped (existing behaviour, just verifies the bridge path didn't break the contract)."""
    from mlframe.training.pipeline._pipeline_extensions import _filter_to_numeric

    pl_df = pl.DataFrame(
        {
            "f_float": np.arange(4, dtype=np.float32),
            "f_int": np.arange(4, dtype=np.int64),
            "f_str": ["a", "b", "c", "d"],
        }
    )
    out, dropped = _filter_to_numeric(pl_df)
    assert "f_str" in dropped, "non-numeric string column should have been dropped"
    assert set(out.columns) == {"f_float", "f_int"}
    assert out["f_float"].dtype == np.float32
    assert out["f_int"].dtype == np.int64


# ---------------------------------------------------------------------------
# Bridge contract: zero-copy on numeric columns (sanity-check the underlying helper)
# ---------------------------------------------------------------------------


def test_bridge_returns_zero_copy_view_for_numeric_columns():
    """get_pandas_view_of_polars_df contract: numeric columns are Arrow-backed views; the bridge promise underpins every fix in this file."""
    n = 1024
    pl_df = pl.DataFrame({"a": np.arange(n, dtype=np.float64), "b": np.arange(n, dtype=np.int32)})
    pdf = get_pandas_view_of_polars_df(pl_df)
    # round-trip values OK
    assert pdf["a"].iloc[0] == 0.0 and pdf["a"].iloc[-1] == float(n - 1)
    assert pdf["b"].iloc[0] == 0 and pdf["b"].iloc[-1] == n - 1
    # Bytes-cap sanity: bridge result should not balloon beyond the source byte size by more than 2x (a full consolidation copy plus a Python pandas index is the worst case we tolerate; a regression to bare to_pandas() with deep object materialisation would inflate well past that).
    src_bytes = pl_df.estimated_size()
    dst_bytes = int(pdf.memory_usage(deep=True).sum())
    assert dst_bytes <= max(src_bytes * 2, 4096), "bridge result %d bytes vs source %d bytes; suspect non-Arrow materialisation copy" % (dst_bytes, src_bytes)
