"""Per-model classification report at the tuned decision threshold, next to the 0.5 one printed at train time.

Synthetic: 30% positives with compressed (under-confident) probabilities, so the balanced-accuracy optimum sits well below 0.5 and the two
operating points produce visibly different reports. The threshold is tuned on val only; test is reported at it, never used to pick it.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import numpy as np

from mlframe.metrics.core import format_classification_report
from mlframe.training.configs import TargetTypes
from mlframe.training.core._phase_train_one_target_ensembling import _tune_decision_thresholds

LOGGER = "mlframe.training.core._phase_train_one_target"


def _member(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)

    def split():
        y = (rng.random(n) < 0.3).astype(np.int64)
        p1 = np.clip(0.18 + 0.30 * y + rng.normal(0, 0.07, n), 0.01, 0.99)  # positives centre near 0.48: a 0.5 cut misses most of them
        return y, np.column_stack([1.0 - p1, p1])

    yv, pv = split()
    yt, pt = split()
    m = SimpleNamespace(model_name="cb_demo", val_probs=pv, val_target=yv, test_probs=pt, test_target=yt)
    return m, yv


def _run(caplog, *, target_type=TargetTypes.BINARY_CLASSIFICATION, **cfg):
    member, yv = _member()
    behavior = SimpleNamespace(tune_decision_threshold="auto", tune_decision_threshold_metric="balanced_accuracy", **cfg)
    metadata: dict = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _tune_decision_thresholds(
            ens_models=[member], target_type=target_type, cur_target_name="t", behavior_config=behavior, common_params={},
            metadata=metadata, verbose=1, current_val_target=yv,
        )
    return member, metadata, [r.getMessage() for r in caplog.records if "CLASSIFICATION REPORT at tuned threshold" in r.getMessage()]


def test_biz_val_tuned_threshold_report_second_block_at_tuned_value(caplog):
    """Val and test blocks are emitted at the stamped threshold, labelled optimistic / honest, and differ from the 0.5 report."""
    member, metadata, blocks = _run(caplog)
    thr = metadata["decision_thresholds"]["binary_classification|t|cb_demo"]
    assert 0.2 < thr < 0.4, thr
    assert len(blocks) == 2, blocks
    val_block = next(b for b in blocks if "[VAL]" in b)
    test_block = next(b for b in blocks if "[TEST]" in b)
    assert f"{thr:.4f}" in val_block and f"{thr:.4f}" in test_block
    assert "optimistic" in val_block and "honest" in test_block
    at_half = format_classification_report(member.test_target, (member.test_probs[:, 1] >= 0.5).astype(np.int64), nclasses=2, digits=4)
    assert test_block.split("\n", 1)[1].strip() != at_half.strip()
    recall_tuned = float(re.search(r"\n\s*1\s+\S+\s+(\S+)", test_block).group(1))
    recall_half = float(re.search(r"\n\s*1\s+\S+\s+(\S+)", at_half).group(1))
    assert recall_tuned >= recall_half + 0.5, (recall_tuned, recall_half)  # measured 0.99 vs 0.40 (tuned threshold 0.32)


def test_biz_val_tuned_threshold_report_opt_out(caplog):
    """``report_at_tuned_threshold=False`` keeps the 0.5 block alone; the threshold is still tuned and stamped."""
    _, metadata, blocks = _run(caplog, report_at_tuned_threshold=False)
    assert blocks == []
    assert metadata["decision_threshold_paths"]["binary_classification|t|cb_demo"] == "tuned"


def test_biz_val_tuned_threshold_report_not_emitted_for_multiclass_or_regression(caplog):
    """Only binary classification has a decision threshold: the other target types log no tuned-threshold block."""
    for tt in (TargetTypes.MULTICLASS_CLASSIFICATION, TargetTypes.REGRESSION):
        caplog.clear()
        _, metadata, blocks = _run(caplog, target_type=tt)
        assert blocks == [] and not metadata.get("decision_thresholds")


def test_biz_val_tuned_threshold_report_skipped_when_threshold_not_tuned(caplog):
    """A balanced target under ``auto`` keeps 0.5, where a second block would repeat the first."""
    member, _ = _member()
    y_balanced = (np.arange(member.val_target.shape[0]) % 2).astype(np.int64)
    behavior = SimpleNamespace(tune_decision_threshold="auto", tune_decision_threshold_metric="balanced_accuracy")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _tune_decision_thresholds(
            ens_models=[member], target_type=TargetTypes.BINARY_CLASSIFICATION, cur_target_name="t", behavior_config=behavior,
            common_params={}, metadata={}, verbose=1, current_val_target=y_balanced,
        )
    assert not [r for r in caplog.records if "CLASSIFICATION REPORT at tuned threshold" in r.getMessage()]
