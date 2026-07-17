"""Regression sensor for S05: at level-1, ``_oof_or_train`` silently substitutes
in-sample ``train_probs`` / ``train_preds`` for missing OOF arrays. The level>1
guard in ``_ensembling_score.py:228`` catches the multi-level case, but level-1
is the default-suite path and the "train" branch of every ensemble flavour ends
up evaluated on leaked rows.

The minimum fix is to emit a single per-call ``logger.warning`` when the fallback
is exercised on level-1 so operators can audit suites that quietly produced
in-sample-train ensemble metrics. This sensor pins that warning shape.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np


def _make_regression_member(n: int, seed: int, *, with_oof: bool):
    """Build a SimpleNamespace mimicking ``models_and_predictions`` entries
    for ``score_ensemble``. ``with_oof=False`` is the leak scenario - train_*
    are present but oof_* are explicitly None.
    """
    rng = np.random.default_rng(seed)
    member = SimpleNamespace(
        model=None,
        val_preds=rng.normal(size=n),
        val_probs=None,
        test_preds=rng.normal(size=n),
        test_probs=None,
        train_preds=rng.normal(size=n),
        train_probs=None,
        oof_preds=(rng.normal(size=n) if with_oof else None),
        oof_probs=None,
    )
    return member


def test_score_ensemble_level1_warns_when_oof_missing_and_train_fallback_used(caplog):
    """At ``max_ensembling_level=1`` (the production default), every member
    missing ``oof_preds`` / ``oof_probs`` triggers the silent ``train_*``
    fallback inside ``_oof_or_train``. The post-fix path MUST emit a
    ``logger.warning`` that names the leak so operators see it in suite
    logs; pre-fix path is silent.
    """
    from mlframe.models.ensembling import score_ensemble

    n = 100
    rng = np.random.default_rng(0)
    member_a = _make_regression_member(n, seed=1, with_oof=False)
    member_b = _make_regression_member(n, seed=2, with_oof=False)

    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        score_ensemble(
            models_and_predictions=[member_a, member_b],
            ensemble_name="s05_sensor",
            train_target=rng.normal(size=n),
            val_target=rng.normal(size=n),
            test_target=rng.normal(size=n),
            max_ensembling_level=1,
            verbose=False,
            n_jobs=1,
        )

    # The warning must explicitly name "OOF" or "train_fallback" (operator-greppable token).
    fallback_msgs = [
        rec for rec in caplog.records if rec.levelno == logging.WARNING and ("oof" in rec.getMessage().lower() and "train" in rec.getMessage().lower())
    ]
    assert fallback_msgs, (
        "S05: score_ensemble(max_ensembling_level=1) silently substituted "
        "in-sample train_preds for missing OOF on every member without "
        "emitting a leak-visibility warning. Expected at least one WARN "
        "naming the OOF -> train fallback so operators can audit suites "
        f"running with cross_val_predict disabled. Captured records: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )


def test_score_ensemble_level1_no_warning_when_all_members_have_oof(caplog):
    """When every member exposes a real ``oof_preds`` ndarray, no
    train-fallback fires and no leak-visibility warning is emitted.
    Catches a future regression where the warning gates fire on the
    happy path.
    """
    from mlframe.models.ensembling import score_ensemble

    n = 100
    rng = np.random.default_rng(0)
    member_a = _make_regression_member(n, seed=1, with_oof=True)
    member_b = _make_regression_member(n, seed=2, with_oof=True)

    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        score_ensemble(
            models_and_predictions=[member_a, member_b],
            ensemble_name="s05_no_fallback_sanity",
            train_target=rng.normal(size=n),
            val_target=rng.normal(size=n),
            test_target=rng.normal(size=n),
            max_ensembling_level=1,
            verbose=False,
            n_jobs=1,
        )

    fallback_msgs = [
        rec for rec in caplog.records if rec.levelno == logging.WARNING and ("oof" in rec.getMessage().lower() and "fallback" in rec.getMessage().lower())
    ]
    assert not fallback_msgs, f"S05 sanity: train-fallback warning fired even though every member had oof_preds. Spurious fallback signal: {fallback_msgs}"
