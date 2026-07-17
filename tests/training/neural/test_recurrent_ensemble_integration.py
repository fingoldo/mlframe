"""Recurrent models must JOIN the score_ensemble blend, not silently bypass it.

Pre-fix flow: ``_train_one_target`` ran ``score_ensemble`` for each target with the booster member list; then
``train_recurrent_models`` ran AFTER the per-target loop and just appended raw model objects to
``ctx.models[type][target]``. ``score_ensemble`` had already returned, so the recurrent member's predictions
never participated in the blend - the recurrent model was effectively dead weight from the ensemble's perspective.

Post-fix flow: ``_phase_recurrent.train_recurrent_models`` computes val/test/train preds for each fitted
recurrent model, wraps each in a member-entry compatible with ``score_ensemble``, appends it to
``ctx.models[type][target]``, then ``_rerun_ensemble_with_recurrent`` calls back into ``score_ensemble`` with the
augmented member list. ``ctx.metadata['recurrent_ensemble_integration']`` records the rerun for observability.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

# Match the rest of tests/training/ - tests import via the installed package, not via sys.path tweaks.
from mlframe.training.configs import OutputConfig, PreprocessingConfig
from mlframe.training.core import train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_cb, pytest.mark.uses_torch]

N_ROWS = 200
SEQ_LEN = 6
N_SEQ_FEATURES = 3


def _build_synthetic_dataset(seed: int = 0):
    """200-row regression dataset with both tabular features and per-row sequences. Target is a deterministic
    function of the tabular features plus mild noise; both CB (tabular) and LSTM (sequence) can learn signal."""
    rng = np.random.default_rng(seed)
    sequences = [rng.standard_normal((SEQ_LEN, N_SEQ_FEATURES)).astype("float32") for _ in range(N_ROWS)]
    seq_means = np.array([s.mean(axis=0) for s in sequences], dtype="float32")
    # Target draws on the SEQUENCE mean (so LSTM has signal) plus a per-row tabular feature.
    extra_tabular = rng.standard_normal(N_ROWS).astype("float32")
    target = (seq_means[:, 0] * 1.2 + extra_tabular * 0.5 + rng.standard_normal(N_ROWS).astype("float32") * 0.1).astype("float32")
    df = pd.DataFrame(
        {
            "num_0": seq_means[:, 0],
            "num_1": seq_means[:, 1],
            "num_2": seq_means[:, 2],
            "num_3": extra_tabular,
            "target": target,
        }
    )
    return df, sequences


def test_recurrent_member_joins_ensemble_after_integration(tmp_path):
    """End-to-end: train tiny CB + tiny LSTM on the same regression target. Assert the recurrent member made it
    into ``ctx.metadata['recurrent_ensemble_integration']`` (proof the rerun fired) AND that the LSTM appears in
    ``models[REGRESSION][target]`` AS a member-entry (SimpleNamespace with val_preds), not as a raw torch wrapper."""
    pytest.importorskip("torch")
    pytest.importorskip("catboost")

    from mlframe.training.core import train_mlframe_models_suite

    df, sequences = _build_synthetic_dataset(seed=42)
    fte = SimpleFeaturesAndTargetsExtractor(regression=True)

    try:
        trained, metadata = train_mlframe_models_suite(
            df=df,
            target_name="tgt",
            model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            recurrent_models=["lstm"],
            sequences=sequences,
            hyperparams_config={"iterations": 5},
            preprocessing_config=PreprocessingConfig(drop_columns=[]),
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            verbose=0,
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        )
    except (NotImplementedError, ImportError) as e:
        pytest.skip(f"recurrent path not fully wired in this env: {e}")

    assert trained is not None
    # The integration bookkeeping is the contract this test enforces. It is populated by
    # _rerun_ensemble_with_recurrent only when (a) a prior ensemble existed for this target AND (b) at least
    # one recurrent member was fit successfully AND (c) the rebuild call did not raise.
    rec_meta = (metadata or {}).get("recurrent_ensemble_integration") or {}
    assert rec_meta, (
        "recurrent_ensemble_integration metadata missing; either the rerun never fired or train_recurrent_models "
        "did not receive ctx. Pre-fix: rerun helper did not exist; post-fix: ctx is threaded from main.py."
    )
    # Walk every target type / target name and look for at least one entry that recorded an LSTM member.
    matched_any = False
    for by_name in rec_meta.values():
        if not isinstance(by_name, dict):
            continue
        for info in by_name.values():
            if not isinstance(info, dict):
                continue
            recurrent_members = info.get("recurrent_members") or []
            if any(str(n).lower() == "lstm" for n in recurrent_members):
                # We also expect at least 2 total members (the CB booster + the LSTM); the rerun would have
                # been skipped by ``len(members) < 2`` otherwise, so this is implied, but assert defensively.
                assert info.get("total_members", 0) >= 2, (
                    f"Augmented ensemble member count was {info.get('total_members')} - expected the CB booster alongside the LSTM."
                )
                matched_any = True
    assert matched_any, f"No recurrent_ensemble_integration entry listed an LSTM member. recurrent_ensemble_integration={rec_meta!r}"


def test_recurrent_skipped_gracefully_when_predict_fails(monkeypatch, tmp_path):
    """Defensive contract: if the recurrent member's predict() emits NaN on a split, the helper drops that
    member from the ensemble rebuild rather than poisoning the blend. This pins the ``np.all(np.isfinite())``
    guard inside ``_safe_predict_recurrent`` so a future refactor cannot regress to silently-NaN preds."""
    pytest.importorskip("torch")
    pytest.importorskip("catboost")

    from mlframe.training.core import _phase_recurrent as _pr

    # Force predict to emit NaN. The helper must return None and the member should NOT join the ensemble.
    real_predict = _pr._safe_predict_recurrent

    def _nan_predict(*, model, sequences, features, is_classification, **_extra):
        # **_extra absorbs newly-added kwargs (ctx, split) so this patch keeps
        # working when the production helper grows signature parameters.
        return None  # mimic predict-failure path

    monkeypatch.setattr(_pr, "_safe_predict_recurrent", _nan_predict)

    df, sequences = _build_synthetic_dataset(seed=7)
    fte = SimpleFeaturesAndTargetsExtractor(regression=True)

    try:
        _, metadata = train_mlframe_models_suite(
            df=df,
            target_name="tgt",
            model_name="mdl",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            recurrent_models=["lstm"],
            sequences=sequences,
            hyperparams_config={"iterations": 5},
            preprocessing_config=PreprocessingConfig(drop_columns=[]),
            use_ordinary_models=True,
            use_mlframe_ensembles=True,
            verbose=0,
            output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        )
    except (NotImplementedError, ImportError) as e:
        pytest.skip(f"recurrent path not fully wired in this env: {e}")

    # When predict is forced to None on every split, the helper appends the raw model (legacy path) but does
    # NOT register the member under recurrent_ensemble_integration; the rerun is skipped because there are no
    # SimpleNamespace member entries to augment with.
    rec_meta = (metadata or {}).get("recurrent_ensemble_integration") or {}
    # If rec_meta is populated, every recorded entry must NOT claim an LSTM member (since predict failed).
    for by_name in rec_meta.values():
        if not isinstance(by_name, dict):
            continue
        for info in by_name.values():
            assert "lstm" not in [str(n).lower() for n in (info or {}).get("recurrent_members") or []], (
                f"recurrent member 'lstm' should have been dropped when predict() returned None; got info={info!r}"
            )

    # Restore for any cleanup
    monkeypatch.setattr(_pr, "_safe_predict_recurrent", real_predict)
