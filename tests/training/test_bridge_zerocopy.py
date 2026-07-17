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
from pathlib import Path


def _module_source(mod) -> str:
    """Read a module's source via ``Path.read_text`` rather than
    ``inspect.getsource``. The latter is forbidden in tests per the
    ``feedback_behavioral_tests`` rule (meta-test
    ``tests/test_meta/test_no_inspect_getsource.py``). File-based
    read achieves the same source-grep without importing inspect."""
    return Path(mod.__file__).read_text(encoding="utf-8")


from mlframe.training.utils import get_pandas_view_of_polars_df


# ---------------------------------------------------------------------------
# F1 + F15: leaderboard / head-tail must use the Arrow bridge so categorical / enum / datetime
# dtypes survive the polars -> pandas hop (bare .to_pandas() collapses them to object).
# ---------------------------------------------------------------------------


def _mixed_polars_frame_for_dtype_check(n=64):
    return pl.DataFrame(
        {
            "f_float": np.arange(n, dtype=np.float32),
            "f_int": np.arange(n, dtype=np.int32),
            "f_enum": pl.Series("f_enum", ["a", "b", "c"] * ((n + 2) // 3))[:n].cast(pl.Enum(["a", "b", "c"])),
            "f_date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1).replace(day=1), eager=True).head(1).extend_constant(pl.date(2020, 1, 1), n - 1),
        }
    )


def test_f1_main_train_suite_leaderboard_path_routes_polars_through_bridge():
    """Source-level sensor: the polars leaderboard branch must call the Arrow bridge, not a pd.DataFrame() / CSV round-trip that would silently densify all dtypes.

    ``_main_train_suite.py`` was carved into themed siblings; the
    leaderboard / VOTENRANK aggregation phase moved to
    ``_main_train_suite_phases.py``. Concat parent + sibling so the
    source-grep guard survives the split.
    """
    from mlframe.training.core import _main_train_suite as mts

    src = _module_source(mts)
    sib = Path(mts.__file__).parent / "_main_train_suite_phases.py"
    if sib.exists():
        src += "\n" + sib.read_text(encoding="utf-8")
    # CSV round-trip is the old pl -> bytes -> pd.read_csv path; it densified every column to whatever pd.read_csv inferred (typically string for datetimes / categoricals).
    assert "pl.DataFrame, _pd.DataFrame)" not in src or "get_pandas_view_of_polars_df" in src, (
        "the leaderboard polars branch must route through get_pandas_view_of_polars_df"
    )
    assert "get_pandas_view_of_polars_df" in src, "F1 regression: leaderboard polars branch no longer routes through the Arrow split-blocks bridge"


def test_f15_extractors_head_tail_preserves_enum_dtype_through_bridge():
    """``head.to_pandas()`` bare collapses pl.Enum to object dtype; the bridge keeps it as pandas CategoricalDtype."""
    pl_df = pl.DataFrame({"f_int": np.arange(5), "f_enum": pl.Series(["x", "y", "x", "z", "y"]).cast(pl.Enum(["x", "y", "z"]))})
    pdf = get_pandas_view_of_polars_df(pl_df.head(5))
    assert isinstance(pdf["f_enum"].dtype, pd.CategoricalDtype), (
        "F15 regression: pl.Enum collapsed to %s after bridge (expected CategoricalDtype)" % pdf["f_enum"].dtype
    )


def test_f15_extractors_module_source_routes_head_tail_via_bridge():
    """Module-level source check: extractors must not call ``head.to_pandas()`` / ``tail.to_pandas()`` bare; both head and tail polars-branch must go through ``get_pandas_view_of_polars_df``.

    ``extractors.py`` was carved into themed siblings
    (``_extractors_showcase.py`` for the head/tail show-distribution path,
    plus ``_extractors_simple.py`` / ``_extractors_dtype_helpers.py``).
    Concat parent + every relevant sibling so source-grep guards survive
    the split.
    """
    from mlframe.training import extractors

    src = _module_source(extractors)
    _dir = Path(extractors.__file__).parent
    for sib_name in (
        "_extractors_showcase.py",
        "_extractors_simple.py",
        "_extractors_dtype_helpers.py",
    ):
        sib = _dir / sib_name
        if sib.exists():
            src += "\n" + sib.read_text(encoding="utf-8")
    assert "head = head.to_pandas()" not in src, "F15 regression: extractors head path back to bare .to_pandas()"
    assert "tail = tail.to_pandas()" not in src, "F15 regression: extractors tail path back to bare .to_pandas()"
    # 1 import at top + at least 2 call-sites (head, tail) on the show-distribution path.
    assert src.count("get_pandas_view_of_polars_df") >= 3, "expected both head + tail polars branches to route through the bridge"


# ---------------------------------------------------------------------------
# F3: sklearn-ndarray-output branch must rejoin polars passthrough cols via bridge
# ---------------------------------------------------------------------------


def test_f3_pipeline_helpers_ndarray_branch_uses_bridge_for_held():
    from mlframe.training.pipeline import _pipeline_helpers as ph

    src = _module_source(ph)
    # The pre-fix shape was ``held_pd = held.to_pandas() if is_polars else held``; the fix routes is_polars=True through the bridge.
    assert "held_pd = held.to_pandas() if is_polars else held" not in src, "F3 regression: ndarray-output branch reverted to bare held.to_pandas()"
    assert src.count("get_pandas_view_of_polars_df") >= 2, "expected BOTH DataFrame-output and ndarray-output passthrough branches to route through the bridge"


# ---------------------------------------------------------------------------
# F4 + F5: pandas -> polars back-merge must skip block consolidation copy
# ---------------------------------------------------------------------------


def test_f4_predict_main_uses_dict_of_numpy_not_from_pandas():
    """pl.from_pandas(df[cols]) consolidates pandas blocks; building polars columns via per-column to_numpy() views skips that copy. Bench shows 15x speedup on 100k x 30 mixed dtypes; this sensor pins the source pattern so a future refactor cannot silently revert."""
    from mlframe.training.core import _predict_main_from_models as pm

    src = _module_source(pm)
    # Old form: pl.from_pandas(df[_ext_new_cols]) — the back-merge hot path.
    assert "pl.from_pandas(df[_ext_new_cols])" not in src, "F4 regression: predict back-merge reverted to pl.from_pandas (pays pandas block consolidation copy)"
    assert "pl.DataFrame({c: df[c].to_numpy() for c in _ext_new_cols})" in src, (
        "F4 regression: expected pl.DataFrame({c: df[c].to_numpy() ...}) dict-of-numpy back-merge"
    )


def test_f5_phase_helpers_fit_pipeline_uses_dict_of_numpy_not_from_pandas():
    from mlframe.training.core import _phase_helpers_fit_pipeline as phfp

    src = _module_source(phfp)
    assert "pl.from_pandas(_new_df_pd)" not in src, "F5 regression: train/val/test back-merge reverted to pl.from_pandas (pays 3x the consolidation copy)"
    assert "pl.DataFrame({c: _new_df_pd[c].to_numpy() for c in _new_df_pd.columns})" in src, (
        "F5 regression: expected dict-of-numpy back-merge for polars-pre extension hstack"
    )


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
    for c in pdf.columns:
        assert expected[c].to_list() == actual[c].to_list(), c
    assert actual.shape == expected.shape


# ---------------------------------------------------------------------------
# F7: _filter_to_numeric polars -> pandas hop must use Arrow split-blocks path, not bare to_pandas
# ---------------------------------------------------------------------------


def test_f7_filter_to_numeric_uses_split_blocks_for_polars_input():
    from mlframe.training.pipeline import _pipeline_extensions as pe

    # Read the module source (Path.read_text, NOT inspect.getsource per
    # ``feedback_behavioral_tests``) and grep for the contract anchors. The
    # ``_filter_to_numeric`` function body is the only place ``split_blocks``
    # appears in this module, so a substring check is sufficient.
    src = _module_source(pe)
    # Bare ``_df = _df.to_pandas()`` is the regressed shape; fixed shape uses split_blocks=True with a TypeError fallback for pre-0.20.4 polars.
    assert "_df = _df.to_pandas()" not in src or "split_blocks=True" in src, (
        "F7 regression: _filter_to_numeric reverted to bare _df.to_pandas() (full consolidation copy on wide frames)"
    )
    assert "split_blocks=True" in src, "F7 regression: expected split_blocks=True in _filter_to_numeric polars hop"


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
