"""The composite post-phases predict each wrapped model on each split, and transform each frame, a bounded number of times.

The wrap pass, the per-model hook, the value report and the MoE gate each re-predicted the same wrapper on the same split,
and pre-pipelines were re-applied to frames they had already transformed. One suite run is counted here: y-scale predicts
per (wrapper, split) and ``pre_pipeline.transform`` calls per (pipeline, frame). The ideal is the budget; today's excess
sits in ``_composite_call_budget_baseline.json`` with a note and may only go down.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import orjson
import pytest

pytest.importorskip("lightgbm")
pytestmark = pytest.mark.slow  # one composite suite run (~60 s)

_BASELINE = Path(__file__).resolve().parent / "_composite_call_budget_baseline.json"
# Ideal counts: one y-scale predict per (wrapper, split) serves the hook, the report and the MoE (PRF-13) - two, since the
# wrap pass also scores the pre-clip prediction for the model card (PRF-14); one transform per (pipeline, frame) (PRF-16).
_IDEAL = {"yscale_predicts_per_wrapper_split": 2, "pp_transforms_per_pipeline_frame": 1}


@pytest.fixture(scope="module")
def post_counts(tmp_path_factory):
    """Max counts over (wrapper, split) and (pipeline, frame) during one composite suite run."""
    from sklearn.pipeline import Pipeline

    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite

    from ..test_composite_integration import _LEAN_OUTPUT_CONFIG_KWARGS, _LEAN_REPORTING_CONFIG_KWARGS, _build_minimal_fte, _tvt_dataset

    predicts: Counter = Counter()
    transforms: Counter = Counter()
    real_predict, real_transform = CompositeTargetEstimator._predict_unclipped, Pipeline.transform

    def predict_spy(self, X, *a, **k):
        predicts[(id(self), len(X))] += 1
        return real_predict(self, X, *a, **k)

    def transform_spy(self, X, *a, **k):
        transforms[(id(self), id(X))] += 1
        return real_transform(self, X, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(CompositeTargetEstimator, "_predict_unclipped", predict_spy)
    mp.setattr(Pipeline, "transform", transform_spy)
    tmp = str(tmp_path_factory.mktemp("post_budget"))
    try:
        cfg = CompositeTargetDiscoveryConfig(enabled=True, base_candidates=["TVT_prev"], transforms=["linear_residual"], mi_sample_n=300,
                                             eps_mi_gain=-1.0, max_total_composite_targets=1, min_honest_gain_to_train=None)
        train_mlframe_models_suite(
            df=_tvt_dataset(n=600), target_name="target", model_name="pb", features_and_targets_extractor=_build_minimal_fte(),
            mlframe_models=["linear", "lgb"], output_config={"data_dir": tmp, "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        )
    finally:
        mp.undo()
    assert predicts, "no composite wrapper predicted: the fixture lost its subject"
    return {"yscale_predicts_per_wrapper_split": max(predicts.values()), "pp_transforms_per_pipeline_frame": max(transforms.values(), default=0)}


@pytest.mark.parametrize("primitive", sorted(_IDEAL))
def test_a_post_phase_primitive_stays_within_its_budget(post_counts, primitive: str):
    """At most the ideal, or the recorded excess; an improvement must lower the record."""
    recorded = orjson.loads(_BASELINE.read_text(encoding="utf-8")).get(primitive)
    got, ideal = post_counts[primitive], _IDEAL[primitive]
    if recorded is None:
        assert got <= ideal, f"{primitive}: {got}, over its ideal {ideal}"
        return
    assert got <= recorded["measured"], f"{primitive}: {got}, over the recorded {recorded['measured']} ({recorded['note']})"
    assert got == recorded["measured"], f"{primitive} improved from {recorded['measured']} to {got}: lower the baseline to lock the gain in"


def test_the_post_phase_budgets_are_known_to_the_baseline_check():
    """The shared baseline check accepts these primitives."""
    assert os.path.exists(_BASELINE)
