"""Feature-selection HYBRID benchmark: MRMR / RFECV / BorutaShap / ShapProxiedFS alone and in
filter-then-wrapper cascades / ensembles, scored by honest-holdout AUC across three downstream model
families (LightGBM / Logistic / kNN) on a controlled synthetic with known ground truth.

Re-run this whenever a selector's internals change to see the effect on hybrid quality + cost.
See README.md for how to run and how to swap in your own dataset.
"""
