"""The honest calibration block refuses to pair OOF probabilities with a target of another length and writes a safe plot name (API5-3, API5-4)."""

from __future__ import annotations

import logging
import types

import numpy as np

from mlframe.training.honest_diagnostics import _calibration_block


def _entry(n_probs: int, n_target: int, seed: int = 3):
    """An entry whose OOF probabilities and OOF target have the given row counts."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, size=max(n_probs, n_target))
    y = (rng.uniform(0, 1, size=raw.shape[0]) < raw).astype(np.int64)
    probs = np.column_stack([1.0 - raw, raw])[:n_probs]
    return types.SimpleNamespace(model=types.SimpleNamespace(oof_probs=probs), oof_probs=probs, oof_target=y[:n_target])


def test_a_length_mismatch_is_skipped_not_truncated(caplog):
    """100 OOF rows against a 90-row target return status skipped with the two counts, and log a warning."""
    with caplog.at_level(logging.WARNING, logger="mlframe.training.honest_diagnostics"):
        out = _calibration_block(_entry(100, 90), "y", None, rng_seed=0)
    assert out["status"] == "skipped"
    assert "100" in out["reason"] and "90" in out["reason"]
    assert any("cannot be paired" in r.getMessage() for r in caplog.records)


def test_equal_lengths_still_run(tmp_path):
    """Matching lengths are not skipped for alignment reasons."""
    out = _calibration_block(_entry(300, 300), "y", None, rng_seed=0)
    assert out["status"] != "skipped" or "not pairable" not in out.get("reason", "")


def test_the_plot_file_name_for_a_hostile_target_name_stays_in_out_dir(tmp_path):
    """A target name with separators and wildcards still puts the calibration plot directly inside out_dir."""
    out_dir = tmp_path / "reports"
    _calibration_block(_entry(300, 300), "price>100/a:b", str(out_dir), rng_seed=0)
    pngs = list(out_dir.rglob("*.png"))
    assert len(pngs) > 0 and all(p.parent == out_dir for p in pngs)
    unsafe = '<>:"/\\|?*'
    assert not any(ch in p.name for p in pngs for ch in unsafe)
