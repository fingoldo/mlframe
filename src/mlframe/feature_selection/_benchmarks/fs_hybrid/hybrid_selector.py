"""HybridSelector was PROMOTED to production: mlframe.feature_selection.hybrid_selector.

This benchmark module now re-exports it so run_experiment.py and the round-3 combine-variant subclasses
(round3_hybrid_refine_bench, round3_hard_bed_bench) keep importing ``HybridSelector`` / ``corr_clusters`` from here
unchanged. The production class still stashes the fit-time X_aug/y on the instance (used by those subclasses'
overrides); __getstate__ drops them on pickle so a fitted estimator stays small.
"""
from mlframe.feature_selection.hybrid_selector import HybridSelector, corr_clusters  # noqa: F401
