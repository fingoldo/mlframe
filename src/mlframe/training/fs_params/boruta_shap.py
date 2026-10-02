"""Strict parameters of `mlframe.feature_selection.boruta_shap:BorutaShap`.

GENERATED from the signature; do not edit by hand. A meta-test compares it with the live signature
(`pyutilz.dev.signature_models.signature_drift`). Regenerate: `python -m mlframe.training.fs_params._generate`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict


class BorutaShapParams(BaseModel):
    """Parameters of `mlframe.feature_selection.boruta_shap:BorutaShap`; an unknown name or a wrong type raises when this is instantiated."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    __signature_overrides__ = ()
    __signature_skip_default__ = ()

    model: Any = None
    importance_measure: str = "gini"
    permutation_n_repeats: int = 5
    classification: bool = True
    percentile: float = 99
    pvalue: float = 0.05
    n_trials: int = 150
    random_state: int = 0
    sample: bool = False
    train_or_test: Literal["train", "test"] = "train"
    resample_holdout_per_trial: bool = False
    premerge_clusters: bool = False
    premerge_corr_thr: float = 0.92
    normalize: bool = True
    verbose: bool = True
    stratify: Any = None
    optimistic: bool = True
    fit_params: Optional[dict] = None
    stability_subsamples: int = 0
    stability_subsample_fraction: float = 0.75
    stability_threshold: float = 1.0
    early_stop_tentative: bool = False
    early_stop_patience: int = 20
    early_stop_margin: float = 0.15
    max_runtime_mins: Optional[float] = None
    stop_file: str = "stop"
    shadow_min_pad: int = 5
