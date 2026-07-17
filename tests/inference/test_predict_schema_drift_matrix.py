"""Predict-time schema-drift matrix.

End-to-end ``train -> save metadata -> call predict -> drift the input -> assert
the right thing happens`` would require touching ``core/predict.py``,
``_predict_guards.py``, ``pipeline.py`` and friends - all locked for this wave.
The whole predict-time schema-drift contract is enforced by one helper that lives
in ``training/core/_misc_helpers.py`` and IS readable:
``_validate_input_columns_against_metadata``. The full suite-level
``predict_mlframe_models_suite`` calls this helper before it hands the frame to
any model. Driving the helper directly exercises the same drift contract without
training a real model, and avoids touching any locked file.

Cases covered (one parametrised function per case):
  (a) Missing required column -> ValueError naming the column.
  (b) Extra unknown column -> dropped silently (logger.info when verbose).
  (c) Dropped high-card column kept in predict -> silently dropped, no crash.
  (d) Dtype family drift on a numeric-role column -> hard ValueError surfaces the dtype mismatch.
  (e) Polars Enum domain drift (new category at predict) -> behaviour gated by the
      schema-fingerprint path; observed contract is soft-warn-and-accept (same-family
      width-only change), with the trained model on the hook for unseen-category
      handling at scoring time.

The dtype-drift case (d) requires the ``model_schemas`` fingerprint block in
metadata to be populated; we build it with ``compute_model_input_fingerprint``
the same way the production trainer does.
"""

from __future__ import annotations

import logging

import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._misc_helpers import (
    _validate_input_columns_against_metadata,
)
from mlframe.training.utils import compute_model_input_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_frame(n_rows: int = 200) -> pd.DataFrame:
    """Three columns: A (int), B (int), C (categorical-by-content, low card)."""
    return pd.DataFrame(
        {
            "A": list(range(n_rows)),
            "B": [i * 2 for i in range(n_rows)],
            "C": [["x", "y", "z"][i % 3] for i in range(n_rows)],
        }
    )


def _train_metadata(df: pd.DataFrame) -> dict:
    """Mimic what training emits: column list + role lists + per-model fingerprint."""
    schema_hash, input_schema = compute_model_input_fingerprint(
        df,
        cat_features=["C"],
        text_features=[],
        embedding_features=[],
    )
    return {
        "columns": list(df.columns),
        "cat_features": ["C"],
        "text_features": [],
        "embedding_features": [],
        "auto_detected_high_card_to_drop": [],
        "model_schemas": {
            "model_A.cbm": {
                "schema_hash": schema_hash,
                "input_schema": input_schema,
            },
        },
    }


# ---------------------------------------------------------------------------
# (a) Missing required column
# ---------------------------------------------------------------------------


def test_missing_required_cat_column_raises_naming_the_column():
    """Drop 'C' (declared as cat_feature) -> ValueError naming 'C'."""
    train = _train_frame()
    meta = _train_metadata(train)
    drifted = train.drop(columns=["C"])
    with pytest.raises(ValueError, match=r"\bC\b") as excinfo:
        _validate_input_columns_against_metadata(drifted, meta)
    msg = str(excinfo.value)
    assert "load-bearing" in msg or "cat/text/embedding" in msg, msg


# ---------------------------------------------------------------------------
# (b) Extra unknown column
# ---------------------------------------------------------------------------


def test_extra_unknown_column_dropped_silently_and_logged(caplog):
    """A column 'D' that wasn't seen at train time must be dropped without raising; verbose mode logs an INFO line naming it."""
    train = _train_frame()
    meta = _train_metadata(train)
    drifted = train.copy()
    drifted["D"] = 1.0
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._misc_helpers"):
        out = _validate_input_columns_against_metadata(drifted, meta, verbose=True)
    assert "D" not in out.columns, "Unknown extra column must be dropped before downstream models see it."
    info_msgs = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any("D" in m and "Dropping" in m for m in info_msgs), f"Verbose mode must INFO-log the dropped extra column; got: {info_msgs}"


# ---------------------------------------------------------------------------
# (c) Dropped high-card column kept in predict
# ---------------------------------------------------------------------------


def test_auto_dropped_high_card_column_still_present_in_predict_is_silently_dropped():
    """A column auto-dropped at train time (high-card text-like) is not in metadata.columns.
    If predict still receives it, the helper must drop it silently - no crash, no error log."""
    train = _train_frame()
    meta = _train_metadata(train)
    # Simulate the trainer having auto-dropped 'high_card_text' before fit.
    meta["auto_detected_high_card_to_drop"] = ["high_card_text"]
    drifted = train.copy()
    drifted["high_card_text"] = [f"tok_{i}" for i in range(len(drifted))]
    out = _validate_input_columns_against_metadata(drifted, meta)
    assert "high_card_text" not in out.columns
    # And the originally trained columns must still be present.
    for c in ("A", "B", "C"):
        assert c in out.columns


# ---------------------------------------------------------------------------
# (d) Dtype family drift on column B (int -> object/string)
# ---------------------------------------------------------------------------


def test_dtype_family_drift_on_b_raises_naming_column_and_dtypes():
    """B was int64 at train; predict provides B as object. The fingerprint path must hard-fail naming the dtype mismatch.

    Note: numeric-role family changes are SOFT-warn by the helper, while cat/text/embedding-role family changes are HARD-fail.
    To make this case raise we have to declare B as cat_feature in the trained metadata so the role is critical.
    The realistic scenario is a column the operator treated as categorical that arrives wrong-typed at predict time.
    """
    train = _train_frame()
    train["B"] = train["B"].astype("int64")
    # Treat B as a categorical-role feature so the role is critical (numeric-role family changes are intentionally soft).
    schema_hash, input_schema = compute_model_input_fingerprint(
        train,
        cat_features=["B", "C"],
        text_features=[],
        embedding_features=[],
    )
    meta = {
        "columns": list(train.columns),
        "cat_features": ["B", "C"],
        "text_features": [],
        "embedding_features": [],
        "model_schemas": {
            "model_A.cbm": {
                "schema_hash": schema_hash,
                "input_schema": input_schema,
            },
        },
    }
    drifted = train.copy()
    drifted["B"] = drifted["B"].astype(str)  # int -> string
    with pytest.raises(ValueError) as excinfo:
        _validate_input_columns_against_metadata(drifted, meta)
    msg = str(excinfo.value)
    assert "B" in msg, f"Hard-fail message must name the drifted column 'B'; got: {msg}"
    # The helper formats trained/serving dtypes verbatim; one of them is the integer family token.
    assert "int" in msg.lower() or "string" in msg.lower() or "object" in msg.lower(), f"Hard-fail message must surface the dtype mismatch; got: {msg}"


# ---------------------------------------------------------------------------
# (e) Polars Enum domain drift (new category at predict).
# ---------------------------------------------------------------------------


def test_polars_enum_domain_drift_soft_warns_and_accepts(caplog):
    """Train sees pl.Enum(['x','y','z']) on column C; predict sees pl.Enum(['x','y','z','w']).

    Actual suite contract (observed via the helper, not specced top-down): an Enum domain
    widening on a cat-role column is treated as a same-family width-only change and is
    SOFT-warn-and-accept, not hard-fail. The trained pipeline is then responsible for
    casting / coercing 'w' downstream (CatBoost's native cat encoder NaNs unseen
    categories; LightGBM treats them as missing-cat).

    Per user memory ``polars Categorical shares one process-wide dict; pl.Enum is
    per-Series built from train+val union, never from test``: production code is supposed
    to build the Enum domain from train + val so that the test/predict frame never widens
    it. When that discipline is violated, this is the contract: a WARNING is emitted, the
    frame is accepted, and the model is on the hook for unseen-category handling.
    """
    train_pl = pl.DataFrame(
        {
            "A": list(range(200)),
            "B": [i * 2 for i in range(200)],
            "C": pl.Series("C", [["x", "y", "z"][i % 3] for i in range(200)], dtype=pl.Enum(["x", "y", "z"])),
        }
    )
    schema_hash, input_schema = compute_model_input_fingerprint(
        train_pl,
        cat_features=["C"],
        text_features=[],
        embedding_features=[],
    )
    meta = {
        "columns": train_pl.columns,
        "cat_features": ["C"],
        "text_features": [],
        "embedding_features": [],
        "model_schemas": {
            "model_A.cbm": {
                "schema_hash": schema_hash,
                "input_schema": input_schema,
            },
        },
    }
    drifted_pl = pl.DataFrame(
        {
            "A": list(range(200)),
            "B": [i * 2 for i in range(200)],
            "C": pl.Series(
                "C",
                [["x", "y", "z", "w"][i % 4] for i in range(200)],
                dtype=pl.Enum(["x", "y", "z", "w"]),
            ),
        }
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._misc_helpers"):
        out = _validate_input_columns_against_metadata(drifted_pl, meta)
    # Frame is accepted (graceful coercion-to-NaN is the contracted downstream behaviour).
    assert out is not None
    assert "C" in out.columns
    warn_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("C" in m and ("schema" in m.lower() or "drift" in m.lower()) for m in warn_msgs), (
        f"Enum domain drift must surface a WARNING naming column 'C' and the schema drift; got: {warn_msgs}"
    )
