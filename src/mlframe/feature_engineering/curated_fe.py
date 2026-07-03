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

DEFAULT-WIRING VERDICT (2026-07-03, bench_curated_fe_holdout_value, honest holdout, 4-5 seeds, bare LGBM):
opt-in stays the deliberate default -- a blanket default is contraindicated. Measured delta vs raw-only:
  * REGRESSION, FE-favorable (categorical target-mean signal): ALL +0.078 R2; multi_aux +0.075, nn_oof
    +0.064, baseline_disagreement +0.062 (5/5 seeds); trust_score -0.009.
  * REGRESSION, FE-unfavorable (pure continuous, no cat structure): only multi_aux +0.013 (4/4) and
    baseline_disagreement +0.003 hold up; nn_oof/trust_score/y_quintile HURT (need the categorical structure).
  * BINARY: EVERYTHING mildly hurts (-0.004..-0.015); trust_score with Mode-A -0.006 (pre-Mode-A: -0.42).
Only ``multi_aux_ensemble`` helps robustly, REGRESSION-ONLY, and at a real cost (3 aux-model fits/fold).
Caveat: this is bare-LGBM; the suite already runs stacking that likely subsumes the disagreement signal --
the very reason these are opt-in. So no global default flip; use ``recommended_curated_fe_names(task)`` to
turn on the evidence-backed subset when your data has the structure it needs.
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

# Evidence-backed subset that lifted honest holdout across FE-favorable AND FE-unfavorable data (see the
# module docstring's DEFAULT-WIRING VERDICT). Regression only -- on binary every curated transformer mildly
# hurt a gradient-booster, and nn_oof/trust_score/y_quintile need categorical-target-mean structure to help.
_RECOMMENDED_BY_TASK = {
    "regression": ("multi_aux_ensemble", "baseline_disagreement"),
    "binary": (),
}


def recommended_curated_fe_names(task: str = "regression") -> tuple:
    """The evidence-backed curated-FE subset for ``task`` (empty for binary; see the module DEFAULT-WIRING
    VERDICT). Feed to ``curated_fe_pipelines(task=..., names=recommended_curated_fe_names(task))`` to opt into
    only the transformers that measurably helped honest holdout. Empty means "raw features already win --
    don't add curated FE for this task"."""
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
