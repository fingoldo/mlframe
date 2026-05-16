"""Regression: ``_phase_pandas_conversion_and_cat_prep`` must gate ``defer_pandas_conv``
on ``polars_pipeline_applied`` in addition to ``was_polars_input``.

Pre-fix the gate was:

    defer_pandas_conv = was_polars_input and not recurrent_models and not _has_rfecv

The ``polars_pipeline_applied`` flag (set in ``_phase_fit_pipeline`` to capture whether
the polars-aware pipeline actually fitted on the polars frame) was only consumed by a log
line. When the polars-input branch ran but the pipeline never produced a polars-side
fitted pipeline (e.g. ``prefer_polarsds=False`` or ``pipeline is None``), the downstream
state lives only in pandas representation; deferring the pandas conversion would silently
strip that state from the lazy-pandas fastpath consumers.

Post-fix the gate threads the flag:

    defer_pandas_conv = (
        was_polars_input
        and polars_pipeline_applied
        and not recurrent_models
        and not _has_rfecv
    )
"""
from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")


def _make_toy_polars_df(n_rows: int = 8):
    return pl.DataFrame({"x": list(range(n_rows)), "y": [float(i) for i in range(n_rows)]})


def _invoke_helper(
    *,
    was_polars_input: bool,
    polars_pipeline_applied: bool,
    recurrent_models: list,
    rfecv_models: list,
) -> bool:
    """Drive ``_phase_pandas_conversion_and_cat_prep`` with a minimal valid input set and
    return its resolved ``defer_pandas_conv`` (12th tuple element, index 11)."""
    from mlframe.training.core import _phase_helpers as ph

    train_df = _make_toy_polars_df(8) if was_polars_input else None
    val_df = _make_toy_polars_df(4) if was_polars_input else None
    test_df = _make_toy_polars_df(4) if was_polars_input else None
    polars_pre = train_df if was_polars_input else None
    val_polars_pre = val_df if was_polars_input else None
    test_polars_pre = test_df if was_polars_input else None

    out = ph._phase_pandas_conversion_and_cat_prep(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_df_polars_pre=polars_pre,
        val_df_polars_pre=val_polars_pre,
        test_df_polars_pre=test_polars_pre,
        cat_features=[],
        was_polars_input=was_polars_input,
        all_models_polars_native=True,
        needs_polars_pre_clone=False,
        mlframe_models=[],
        recurrent_models=recurrent_models,
        rfecv_models=rfecv_models,
        baseline_rss_mb=0.0,
        df_size_mb=0.0,
        verbose=False,
        polars_pipeline_applied=polars_pipeline_applied,
    )
    # Per the return signature, ``defer_pandas_conv`` sits at index 11.
    return bool(out[11])


def test_defer_pandas_conv_false_when_polars_pipeline_not_applied():
    """The headline post-fix case: polars input was provided but the polars-aware pipeline
    never fitted (``polars_pipeline_applied=False``). Pre-fix the gate returned True (only
    ``was_polars_input`` mattered); post-fix it must return False so the lazy-pandas
    fastpath is suppressed and the pandas-side pipeline state is preserved."""
    assert _invoke_helper(
        was_polars_input=True,
        polars_pipeline_applied=False,
        recurrent_models=[],
        rfecv_models=[],
    ) is False


def test_defer_pandas_conv_true_when_polars_pipeline_applied_and_no_blockers():
    """When all three real gates are satisfied (polars input, polars pipeline actually
    fitted, no recurrent / no RFECV), the fastpath stays on. This guards against an
    over-eager fix that would also strip the True branch."""
    assert _invoke_helper(
        was_polars_input=True,
        polars_pipeline_applied=True,
        recurrent_models=[],
        rfecv_models=[],
    ) is True


def test_defer_pandas_conv_false_when_input_was_pandas():
    """``was_polars_input`` is the hard gate; even with ``polars_pipeline_applied=True``
    (which cannot actually happen in production when the input was pandas, but the gate
    must still short-circuit), the fastpath must be off."""
    assert _invoke_helper(
        was_polars_input=False,
        polars_pipeline_applied=True,
        recurrent_models=[],
        rfecv_models=[],
    ) is False


def test_defer_pandas_conv_false_with_recurrent_models():
    """Recurrent models still block the fastpath regardless of the new gate."""
    assert _invoke_helper(
        was_polars_input=True,
        polars_pipeline_applied=True,
        recurrent_models=["lstm"],
        rfecv_models=[],
    ) is False


def test_defer_pandas_conv_false_with_rfecv_models():
    """RFECV still blocks the fastpath regardless of the new gate."""
    assert _invoke_helper(
        was_polars_input=True,
        polars_pipeline_applied=True,
        recurrent_models=[],
        rfecv_models=["cb_num_rfecv"],
    ) is False
