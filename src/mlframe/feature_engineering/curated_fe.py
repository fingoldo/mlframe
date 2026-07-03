"""Curated allowlist of high-value, benched feature-engineering transformers (Workstream C).

The ``feature_engineering/transformer/`` package holds 100+ shortlist transformers with no handle from
the suite. This module exposes the 5 with the strongest measured evidence + leak-safe OOF discipline,
each wrapped via :class:`ShortlistTransformerAdapter` into a sklearn pipeline ready to drop into
``FeatureSelectionConfig.custom_pre_pipelines`` (already consumed by the suite -- no pipeline-phase
wiring needed). Supervised (need ``y`` at fit); the adapter enforces the per-fold OOF discipline so
``transform`` on held-out rows never leaks the calibration labels.

The 5 (each cites a measured win in its module docstring / biz_value test):
  nn_oof_target_mean       -- 3-baseline embedding + kNN target-encoding (Home-Credit 1st-place feature).
  multi_aux_ensemble       -- LGB / focal-LGB / XGB cross-model disagreement.
  baseline_disagreement    -- 3-baseline disagreement-as-feature.
  trust_score_oof          -- kNN distance to OOF-correct rows (confidence signal).
  y_quintile_baseline_knn  -- quintile-conditional kNN baseline (regression).

Opt-in: nothing is enabled implicitly. Call ``curated_fe_pipelines(...)`` and pass the result as
``custom_pre_pipelines`` to turn specific ones on.

DEFAULT-WIRING VERDICT (2026-07-03, bench_curated_fe_holdout_value, honest holdout, 4-5 seeds, bare LGBM,
FULL metric block via mlframe fused kernels -- a single AUC misled the first pass). Opt-in stays the
deliberate default -- a blanket default is contraindicated -- but the per-task/metric picture is nuanced:
  * REGRESSION (all error metrics agree, 4/5-5/5 seeds): multi_aux +0.075 R2 / -0.23 RMSE, nn_oof +0.064 /
    -0.19, baseline_disagreement +0.062 / -0.18; ALL +0.078 R2. trust_score HURTS; y_quintile mixed.
    On FE-UNFAVORABLE regression (no categorical structure) only multi_aux + baseline_disagreement hold up;
    nn_oof needs the categorical-target-mean signal (hurts without it).
  * BINARY: rank-AUC / hard-accuracy barely move (LGBM already ranks well), but the PROBABILITY metrics
    disagree per transformer -- baseline_disagreement IMPROVES Brier -0.0035 and LogLoss -0.0109 (5/5 seeds)
    and PR-AUC (4/5): genuinely better-calibrated probabilities. multi_aux HURTS every binary metric (adds
    variance to a good classifier). trust_score/y_quintile hurt; nn_oof mild-help on Brier/LogLoss (3/5).
So: no global default flip (the suite's stacking likely subsumes the disagreement signal -- why these are
opt-in, and it costs aux-model fits/fold). Use ``recommended_curated_fe_names(task)`` for the evidence-backed
subset: regression -> multi_aux + baseline_disagreement (robust across favorable/unfavorable); binary ->
baseline_disagreement (when you optimise Brier / LogLoss / calibration, not just ROC-AUC).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional


def _allowlist():
    """Map name -> the ``compute_*`` function (lazy import so a missing optional dep does not break import)."""
    from .transformer.baseline_disagreement import compute_baseline_disagreement_features
    from .transformer.multi_aux_ensemble import compute_multi_aux_features
    from .transformer.nn_oof_target_mean import compute_nn_oof_target_mean_features
    from .transformer.trust_score_oof import compute_trust_score_oof_features
    from .transformer.y_quintile_baseline_knn import compute_y_quintile_baseline_knn_features

    return {
        "nn_oof_target_mean": compute_nn_oof_target_mean_features,
        "multi_aux_ensemble": compute_multi_aux_features,
        "baseline_disagreement": compute_baseline_disagreement_features,
        "trust_score_oof": compute_trust_score_oof_features,
        "y_quintile_baseline_knn": compute_y_quintile_baseline_knn_features,
    }


CURATED_FE_NAMES = (
    "nn_oof_target_mean",
    "multi_aux_ensemble",
    "baseline_disagreement",
    "trust_score_oof",
    "y_quintile_baseline_knn",
)

# Evidence-backed subset per task (full-metric honest-holdout benchmark; see the module DEFAULT-WIRING
# VERDICT). Regression: multi_aux + baseline_disagreement lift every error metric on FE-favorable AND
# FE-unfavorable data. Binary: baseline_disagreement improves the PROBABILITY metrics (Brier / LogLoss /
# PR-AUC, 5/5 seeds) though not rank-AUC -- so it is recommended for binary only when the objective is
# calibration / proper-scoring, not ROC-AUC. nn_oof needs categorical-target-mean structure; trust_score and
# y_quintile did not earn a recommendation.
_RECOMMENDED_BY_TASK = {
    "regression": ("multi_aux_ensemble", "baseline_disagreement"),
    "binary": ("baseline_disagreement",),
}


def recommended_curated_fe_names(task: str = "regression") -> tuple:
    """The evidence-backed curated-FE subset for ``task`` (see the module DEFAULT-WIRING VERDICT). Feed to
    ``curated_fe_pipelines(task=..., names=recommended_curated_fe_names(task))`` to opt into only the
    transformers that measurably helped honest holdout on the FULL metric block. For binary the recommendation
    (baseline_disagreement) helps the PROBABILITY metrics (Brier / LogLoss / PR-AUC), not necessarily ROC-AUC
    or hard accuracy -- match it to your objective. An empty result means raw features already win for that
    task; add nothing."""
    return _RECOMMENDED_BY_TASK.get(task, ())


def curated_fe_pipelines(
    task: str = "regression",
    *,
    names: Optional[Sequence[str]] = None,
    seed: int = 42,
    passthrough: bool = True,
) -> dict[str, Any]:
    """Build ``{name: sklearn Pipeline}`` for the curated FE transformers, ready for ``custom_pre_pipelines``.

    ``task`` is ``"regression"`` or ``"binary"`` (forwarded to each transformer). ``names`` selects a subset
    (default all 5). ``passthrough=True`` keeps the raw columns alongside the engineered ones. Each pipeline
    is a single ``ShortlistTransformerAdapter`` step running the leak-safe per-fold OOF transform.
    """
    from sklearn.pipeline import Pipeline

    from .transformer._suite_adapter import ShortlistTransformerAdapter

    allow = _allowlist()
    chosen = list(names) if names is not None else list(CURATED_FE_NAMES)
    unknown = [n for n in chosen if n not in allow]
    if unknown:
        raise ValueError(f"curated_fe_pipelines: unknown name(s) {unknown}; valid: {sorted(allow)}")

    out: dict[str, Any] = {}
    for name in chosen:
        adapter = ShortlistTransformerAdapter(
            allow[name],
            seed=seed,
            needs_y=True,
            passthrough=passthrough,
            compute_kwargs={"task": task},
        )
        out[name] = Pipeline([(name, adapter)])
    return out
