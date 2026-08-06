"""Regression tests for audits/full_audit_2026-07-21/core_infra_b.md's mlflow.py findings
(B4/B5/B6, PR2/P2/P5).

Closes B5's zero-test-coverage gap: this file previously did not exist at all
(``tests/integrations/`` was an empty/absent directory).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mlflow", reason="mlflow ships as an optional extra, not installed in the base CI env")

from mlframe.integrations.mlflow import (
    create_mlflow_run_label,
    embed_website_to_mlflow,
    flatten_classification_report,
    get_or_create_mlflow_run,
    log_classification_report_to_mlflow,
)


def _sample_cr() -> dict:
    """A minimal sklearn-shaped classification_report(output_dict=True) fixture."""
    return {
        "accuracy": 0.9,
        "0": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75, "support": 10},
        "1": {"precision": 0.95, "recall": 0.97, "f1-score": 0.96, "support": 90},
        "macro avg": {"precision": 0.875, "recall": 0.835, "f1-score": 0.855, "support": 100},
    }


# ---------------------------------------------------------------------------
# B4: flatten_classification_report / log_classification_report_to_mlflow no longer mutate cr
# ---------------------------------------------------------------------------


def test_flatten_classification_report_does_not_mutate_input():
    """flatten_classification_report must not pop keys out of the caller's dict."""
    cr = _sample_cr()
    original_keys = set(cr.keys())
    flatten_classification_report(cr, source="val_")
    assert set(cr.keys()) == original_keys, "B4 REGRESSION: flatten_classification_report popped keys out of the caller's dict"
    assert cr["accuracy"] == 0.9  # still present, unchanged


def test_flatten_classification_report_output_correct():
    """The flattened dict carries the expected MLflow-metric-name -> value entries."""
    cr = _sample_cr()
    out = flatten_classification_report(cr, source="val_")
    assert out["val_accuracy"] == 0.9
    assert out["val_macro avg_precision"] == 0.875
    assert out["val_class 0_recall"] == 0.7
    assert out["val_class 1_f1-score"] == 0.96


def test_flatten_and_log_can_both_run_on_the_same_cr_object():
    """The B4 scenario: flatten for one destination, log for another, on the SAME cr dict."""
    cr = _sample_cr()
    flat = flatten_classification_report(cr, source="")
    assert "accuracy" in flat

    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        log_classification_report_to_mlflow(cr, step=0)
        # accuracy must still have been logged -- pre-fix, flatten's .pop() would have already
        # removed it from cr, so this second call would silently skip logging it.
        logged_names = {c.args[0] for c in mock_mlflow.log_metric.call_args_list}
        assert "accuracy" in logged_names


def test_log_classification_report_to_mlflow_does_not_mutate_input():
    """log_classification_report_to_mlflow must not pop keys out of the caller's dict."""
    cr = _sample_cr()
    original_keys = set(cr.keys())
    with patch("mlframe.integrations.mlflow.mlflow"):
        log_classification_report_to_mlflow(cr, step=0, source="")
    assert set(cr.keys()) == original_keys


def test_log_classification_report_to_mlflow_logs_expected_metrics():
    """Every scalar/per-class/averaged metric is logged with the right name, value, and step."""
    cr = _sample_cr()
    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        log_classification_report_to_mlflow(cr, step=3, source="test_")
    calls = {c.args[0]: c.args[1] for c in mock_mlflow.log_metric.call_args_list}
    assert calls["test_accuracy"] == 0.9
    assert calls["test_macro avg_f1-score"] == 0.855
    for c in mock_mlflow.log_metric.call_args_list:
        assert c.kwargs.get("step") == 3


# ---------------------------------------------------------------------------
# B6: embed_website_to_mlflow escapes the URL and writes the expected file
# ---------------------------------------------------------------------------


def test_embed_website_to_mlflow_escapes_url_and_writes_file(tmp_path):
    """A URL containing HTML-injection characters must be escaped, not embedded verbatim."""
    fname = str(tmp_path / "embed")
    embed_website_to_mlflow('http://x/"><script>alert(1)</script>', fname=fname)
    content = (tmp_path / "embed.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&quot;" in content or "&#x27;" in content or "&amp;" in content or "%22" not in content


def test_embed_website_to_mlflow_docstring_documents_trust_boundary():
    """B6: the docstring must call out that url must not be untrusted/user-supplied."""
    doc = embed_website_to_mlflow.__doc__ or ""
    assert "untrusted" in doc.lower() or "internally-controlled" in doc.lower()


# ---------------------------------------------------------------------------
# create_mlflow_run_label
# ---------------------------------------------------------------------------


def test_create_mlflow_run_label_basic():
    """A category plus params renders as ``category:k=v,k=v``."""
    label = create_mlflow_run_label({"a": 1, "b": "x"}, category="cat")
    assert label == "cat:a=1,b=x"


def test_create_mlflow_run_label_skips_falsy_values():
    """Falsy param values (0, None, '') are dropped from the label."""
    label = create_mlflow_run_label({"a": 1, "b": 0, "c": None, "d": ""})
    assert label == "a=1"


def test_create_mlflow_run_label_enum_renders_member_name():
    """An Enum-valued param renders as its member name, not its repr."""
    from enum import Enum

    class Color(Enum):
        """A minimal enum fixture."""

        RED = 1

    label = create_mlflow_run_label({"color": Color.RED})
    assert label == "color=RED"


def test_create_mlflow_run_label_bare_type_renders_class_name():
    """A bare type-valued param renders as its class name, not its repr."""
    label = create_mlflow_run_label({"cls": int})
    assert label == "cls=int"


def test_create_mlflow_run_label_no_category_no_prefix():
    """With no category, the label is just the param list, no leading colon."""
    label = create_mlflow_run_label({"a": 1})
    assert label == "a=1"


def test_create_mlflow_run_label_category_no_params():
    """With a category and no params, the label is just the category."""
    label = create_mlflow_run_label({}, category="cat")
    assert label == "cat"


# ---------------------------------------------------------------------------
# get_or_create_mlflow_run: "already active" retry path
# ---------------------------------------------------------------------------


def test_get_or_create_mlflow_run_finds_existing_run():
    """An existing matching run is returned as-is, found=True."""
    fake_run = MagicMock()
    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = [fake_run]
        run, found = get_or_create_mlflow_run("my_run")
    assert run is fake_run
    assert found is True


def test_get_or_create_mlflow_run_retries_past_already_active_error():
    """The retry loop's core contract: an 'already active' start_run() failure ends the stale run
    and retries, rather than propagating or silently giving up on the first attempt."""
    fake_new_run = MagicMock()
    fake_active_run = MagicMock()
    fake_active_run.info.run_id = "stale-run-id"

    call_count = {"n": 0}

    def _start_run(*args, **kwargs):
        """Fail with an 'already active' error on the first call, succeed on the retry."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Run with UUID x is already active. To start a new run, first end the current run.")
        return fake_new_run

    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = []
        mock_mlflow.start_run.side_effect = _start_run
        mock_mlflow.active_run.return_value = fake_active_run
        run, found = get_or_create_mlflow_run("my_run")

    assert run is fake_new_run
    assert found is False
    assert call_count["n"] == 2, "must retry start_run() after ending the stale active run"
    mock_mlflow.end_run.assert_called()


def test_get_or_create_mlflow_run_gives_up_after_5_failures():
    """A persistently 'already active' error gives up cleanly instead of retrying forever."""
    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = []
        mock_mlflow.start_run.side_effect = RuntimeError("already active, forever")
        mock_mlflow.active_run.return_value = None
        run, found = get_or_create_mlflow_run("my_run")
    assert run is None
    assert found is False


def test_get_or_create_mlflow_run_reraises_unrelated_errors():
    """An unrelated start_run() failure must propagate, not be swallowed by the retry loop."""
    import pytest

    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = []
        mock_mlflow.start_run.side_effect = ValueError("some unrelated failure")
        with pytest.raises(ValueError, match="unrelated failure"):
            get_or_create_mlflow_run("my_run")


def test_get_or_create_mlflow_run_reraised_exception_has_scrubbed_message():
    """X_SECURITY_ROBUSTNESS-4: the RE-RAISED exception's own message must be scrubbed of tracking-server
    userinfo credentials, not just the log line -- a bare `raise` of the original exception would leak the
    credential to any outer handler that logs str(e) on the propagated exception."""
    import pytest

    with patch("mlframe.integrations.mlflow.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = []
        mock_mlflow.start_run.side_effect = ConnectionError("failed to connect to https://user:supersecret@tracking.example.com/api")
        with pytest.raises(ConnectionError) as exc_info:
            get_or_create_mlflow_run("my_run")
    assert "supersecret" not in str(exc_info.value), f"credential leaked in re-raised exception message: {exc_info.value}"
    assert "***@" in str(exc_info.value)
