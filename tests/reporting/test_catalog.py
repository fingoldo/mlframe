"""Tests for the panel-token catalogue (reporting/catalog.py, INV-58).

cProfile note: the catalogue is an O(tokens) dict-lookup over ~45 tokens; a wall-time microbench is sub-millisecond,
so there is no actionable hotspot to optimize -- this is a pure metadata helper, not a hot path.
"""

from __future__ import annotations

import io

import pytest

from mlframe.reporting.catalog import available_panels, describe_available_panels
from mlframe.reporting.charts.binary import ALLOWED_BINARY_PANEL_TOKENS
from mlframe.reporting.charts.ltr import ALLOWED_LTR_PANEL_TOKENS
from mlframe.reporting.charts.multiclass import ALLOWED_MULTICLASS_PANEL_TOKENS
from mlframe.reporting.charts.multilabel import ALLOWED_MULTILABEL_PANEL_TOKENS
from mlframe.reporting.charts.quantile import ALLOWED_QUANTILE_PANEL_TOKENS
from mlframe.reporting.charts.regression import ALLOWED_REGRESSION_PANEL_TOKENS
from mlframe.reporting.charts.temporal import ALLOWED_TEMPORAL_PANEL_TOKENS

_EXPECTED = {
    "binary_classification": ALLOWED_BINARY_PANEL_TOKENS,
    "multiclass_classification": ALLOWED_MULTICLASS_PANEL_TOKENS,
    "multilabel_classification": ALLOWED_MULTILABEL_PANEL_TOKENS,
    "learning_to_rank": ALLOWED_LTR_PANEL_TOKENS,
    "quantile_regression": ALLOWED_QUANTILE_PANEL_TOKENS,
    "regression": ALLOWED_REGRESSION_PANEL_TOKENS,
    "temporal": ALLOWED_TEMPORAL_PANEL_TOKENS,
}


def test_available_panels_covers_every_task_type():
    cat = available_panels()
    assert set(cat) == set(_EXPECTED)


def test_available_panels_lists_exactly_the_frozenset_tokens():
    cat = available_panels()
    for task, frozen in _EXPECTED.items():
        listed = {tok for tok, _desc in cat[task]}
        assert listed == set(frozen), f"{task} catalogue tokens diverged from ALLOWED_*_PANEL_TOKENS"


def test_every_token_has_a_real_description():
    """No token may fall back to the (no description) sentinel -- a new chart token must add a blurb here."""
    cat = available_panels()
    missing = [(task, tok) for task, rows in cat.items() for tok, desc in rows if desc == "(no description)"]
    assert not missing, f"tokens missing a catalogue description: {missing}"


def test_descriptions_are_ascii():
    cat = available_panels()
    for rows in cat.values():
        for _tok, desc in rows:
            desc.encode("ascii")  # raises on non-ascii (console safety)


def test_describe_prints_and_returns_mapping():
    buf = io.StringIO()
    ret = describe_available_panels(file=buf)
    text = buf.getvalue()
    assert "binary_classification" in text
    assert "CONFUSED_PAIRS" in text
    assert "QUANTILE_CROSSING" in text
    # Return value is the same structured mapping.
    assert ret == available_panels()


def test_describe_output_is_ascii():
    buf = io.StringIO()
    describe_available_panels(file=buf)
    buf.getvalue().encode("ascii")


def test_standalone_diagnostics_listed_and_described():
    from mlframe.reporting.catalog import standalone_diagnostics

    rows = standalone_diagnostics()
    names = {name for name, _desc in rows}
    for expected in ("pdp_ice", "model_comparison", "slice_finder", "decision_curve", "calibration_drift", "shap_panels", "learning_curve", "combined_html"):
        assert expected in names, f"{expected} missing from standalone diagnostics catalogue"
    for name, desc in rows:
        assert desc and desc != "(no description)", f"{name} has no description"
        desc.encode("ascii")


def test_describe_includes_standalone_section():
    buf = io.StringIO()
    describe_available_panels(file=buf)
    text = buf.getvalue()
    assert "standalone diagnostics" in text
    assert "pdp_ice" in text and "shap_panels" in text
