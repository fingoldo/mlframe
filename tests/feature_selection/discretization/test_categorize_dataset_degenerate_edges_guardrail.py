"""Dispatcher-level silent-degenerate-edges guardrail (2026-07-19 audit round).

Both ``per_feature_edges`` (``_adaptive_nbins.py``) and ``categorize_dataset`` (this package) previously used
whatever edges a chosen ``nbins_strategy`` returned with NO validation: an empty-edges column silently collapses
to a single degenerate bin (every row gets code 0) with no observable signal anywhere in the call chain -- exactly
how the MDLP ``3.0**n_classes`` overflow bug (acceptance check always False -> empty edges) slipped through
undetected in production. These tests pin the new guardrail: a WARNING is logged (from the correct module, with
the offending column identified) whenever a strategy collapses a real-variance column to empty edges, at BOTH the
per-column dispatch layer and the top-level ``categorize_dataset`` entry point.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges
from mlframe.feature_selection.filters.discretization import categorize_dataset


def _fake_empty_edges_method(monkeypatch, module, target_name: str):
    """Monkeypatch a real edge-builder to unconditionally return empty edges, simulating exactly the
    MDLP-overflow failure mode (a method that silently degrades to 0 edges on real-variance input)
    without touching the actual MDLP code (owned by another concurrent session)."""
    def _always_empty(*args, **kwargs):
        """Stand-in edge-builder that unconditionally returns empty edges."""
        return np.array([], dtype=np.float64)
    monkeypatch.setattr(module, target_name, _always_empty)


def test_per_feature_edges_warns_on_empty_edges_for_real_variance_column(monkeypatch, caplog):
    """A method that silently collapses a real-variance column to 0 edges must trigger the systemic
    guardrail WARNING at the per_feature_edges dispatch layer, regardless of WHICH strategy it is --
    simulated here via 'qs' (in this audit's own scope) so the guardrail is proven generic, not
    hardcoded to one method name."""
    import mlframe.feature_selection.filters._adaptive_nbins as mod

    _fake_empty_edges_method(monkeypatch, mod, "edges_qs")
    rng = np.random.default_rng(0)
    # Enough distinct values to bypass the low_card_cap midpoint shortcut (default 32) so the column
    # actually reaches the (now-stubbed) edges_qs call.
    X = rng.normal(size=(2000, 1)).astype(np.float64)
    X[:, 0] += np.arange(2000) * 1e-3  # inflate distinct-value count comfortably above low_card_cap

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._adaptive_nbins"):
        edges_list = per_feature_edges(X, method="qs")

    assert edges_list[0].size == 0  # the stub really did produce empty edges
    assert any("EMPTY bin edges" in rec.message for rec in caplog.records), (
        "expected the per_feature_edges systemic guardrail WARNING when a strategy silently " "collapses a real-variance column to empty edges"
    )


def test_per_feature_edges_no_warning_when_edges_are_healthy():
    """Sanity control: the guardrail must NOT fire spuriously on ordinary, non-degenerate data (no
    false positives that would train users to ignore the warning)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3)).astype(np.float64)
    import logging as _logging

    logger = _logging.getLogger("mlframe.feature_selection.filters._adaptive_nbins")
    records: list = []
    handler = _logging.Handler()
    handler.emit = lambda rec: records.append(rec)  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        per_feature_edges(X, method="qs")
    finally:
        logger.removeHandler(handler)
    assert not any("EMPTY bin edges" in r.getMessage() for r in records)


def test_categorize_dataset_warns_and_names_degenerate_columns(monkeypatch, caplog):
    """Top-level guardrail: ``categorize_dataset`` (the actual MRMR.fit entry point) must surface a
    WARNING naming the specific column(s) that silently collapsed to a degenerate single bin under a
    supervised nbins_strategy -- this is the exact call site the background MDLP bug report flagged as
    having NO observability at all."""
    import mlframe.feature_selection.filters._adaptive_nbins as mod

    _fake_empty_edges_method(monkeypatch, mod, "edges_qs")
    rng = np.random.default_rng(2)
    n = 2000
    real_variance_col = rng.normal(size=n) + np.arange(n) * 1e-3
    df = pd.DataFrame({"feat_a": real_variance_col})

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.discretization"):
        data, _cols, _nbins = categorize_dataset(df, nbins_strategy="qs")

    assert data.shape == (n, 1)
    assert (data[:, 0] == 0).all(), "stubbed empty-edges column should collapse to a single degenerate bin"
    matched = [
        rec for rec in caplog.records
        if "EMPTY bin edges" in rec.message and rec.name == "mlframe.feature_selection.filters.discretization._discretization_dataset"
    ]
    assert matched, "expected categorize_dataset (top-level entry) to warn about the degenerate column"
    assert "feat_a" in matched[0].message


def test_categorize_dataset_no_warning_for_genuinely_constant_column():
    """Sanity control: a genuinely CONSTANT column (zero variance) collapsing to one bin is CORRECT
    behavior, not a bug -- the guardrail must not fire for it (it only should flag columns that HAD
    real variance but still ended up with empty edges)."""
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({"const_col": np.full(n, 5.0), "real_col": rng.normal(size=n)})
    logger = logging.getLogger("mlframe.feature_selection.filters.discretization")
    records: list = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec)  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        categorize_dataset(df, nbins_strategy="qs")
    finally:
        logger.removeHandler(handler)
    assert not any("EMPTY bin edges" in r.getMessage() for r in records)
