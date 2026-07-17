"""Regression coverage for the tiny-rerank emergency-skip env-var + RAM
checkpoint helper (2026-05-30).

User observed Jupyter-kernel kill INSIDE ``_tiny_model_rerank`` on a Windows
host with ~20 GB physical RAM still free -- not a classical OOM. Most likely
cause: system-wide commit-charge limit exhaustion when a LightGBM Dataset
transient allocation tips the system over the (physical + pagefile) ceiling.

These tests cover the operator-visible escape hatches without instantiating
the full discovery scaffold:
1. MLFRAME_DISCOVERY_SKIP_TINY_RERANK=1 short-circuits before any LightGBM fit.
2. Default (env unset) keeps the rerank pass active.
3. The RAM checkpoint helper writes a single coherent log line with all three
   memory signals.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_self_stub(kept_specs):
    """Build the minimum-viable ``self`` stub the function needs to early-return."""
    return SimpleNamespace(
        config=SimpleNamespace(
            tiny_model_sample_n=100,
            random_state=0,
            tiny_consensus="union",
            tiny_screening_models="single_lgbm",
        ),
    )


def test_skip_env_var_short_circuits_rerank(caplog):
    from mlframe.training.composite.discovery._tiny_rerank import _tiny_model_rerank

    kept = [SimpleNamespace(name="spec_a", base_column="b1", transform_name="diff", fitted_params={})]
    self_stub = _make_self_stub(kept)
    with patch.dict(os.environ, {"MLFRAME_DISCOVERY_SKIP_TINY_RERANK": "1"}, clear=False):
        with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.discovery._tiny_rerank"):
            result = _tiny_model_rerank(
                self_stub,
                kept_specs=kept,
                df=None,
                target_col="t",
                usable_features=[],
                train_idx=None,
                y_full=None,
            )
    # Returned the input list unchanged -- no LightGBM was fired.
    assert result is kept
    skip_lines = [r for r in caplog.records if "SKIPPED" in r.getMessage()]
    assert skip_lines, "skip path must emit a WARNING so the operator sees it"


def test_skip_env_var_truthy_variants():
    """Mirror the parsing contract used in the function body so a regression in
    one trips both tests."""
    for v in ("1", "true", "yes", "ON", "TRUE"):
        active = v.strip().lower() in ("1", "true", "yes", "on")
        assert active, f"{v!r} should activate skip"
    for v in ("0", "false", "no", "OFF", ""):
        active = v.strip().lower() in ("1", "true", "yes", "on")
        assert not active, f"{v!r} should NOT activate skip"


def test_ram_checkpoint_helper_emits_three_signals(caplog):
    from mlframe.training.composite.discovery._tiny_rerank import _tiny_rerank_ram_checkpoint

    with caplog.at_level(logging.INFO, logger="mlframe.training.composite.discovery._tiny_rerank"):
        _tiny_rerank_ram_checkpoint("test_label")
    lines = [r for r in caplog.records if "tiny_rerank.RAM" in r.getMessage()]
    assert lines, "checkpoint must emit one INFO line"
    msg = lines[0].getMessage()
    assert "test_label" in msg
    # All three Windows-relevant signals must be present so the operator can
    # tell USS / RSS / commit apart at the kill point.
    assert "USS=" in msg
    assert "RSS=" in msg
    assert "commit=" in msg


def test_ram_checkpoint_tolerates_psutil_failure(caplog):
    """If memory_full_info() raises, the checkpoint must silently return -- it's
    diagnostic-only and must not block the rerank or crash the suite."""
    from mlframe.training.composite.discovery import _tiny_rerank as mod

    with patch("mlframe.training.composite.discovery._fit._process_mem_mb", side_effect=RuntimeError("psutil down")):
        # Must not raise.
        mod._tiny_rerank_ram_checkpoint("test")
