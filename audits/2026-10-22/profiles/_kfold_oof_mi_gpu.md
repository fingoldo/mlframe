# _kfold_oof_mi_gpu (k-fold OOF MI scorer folds on the device), 2026-11-03

Where found: utilization baseline of the 100k all-families GPU fit, `score_features_by_kfold_oof_mi` 1.0 s at 1.8 % GPU (host per-column binning and MI
in a fold thread pool).
Device fold: train-row edges by `radix_quantiles_grouped` (numpy's linear quantile at linspace(0, 1, nbins + 1)[1:-1], exact for float32 input),
test-row codes by `_searchsorted_codes` (side="right"; the host deduplicates edges, which only renumbers bins), joint counts by
`_batched_joint_counts2`, Miller-Madow MI with the observed supports (`_mi_from_binned_xy`, clip_zero=False) vectorised in cupy.
Same values up to summation order: 100k rows x (20 raw + 200 engineered) columns, 5 folds: same ranking, max |diff| 6.2e-16 (test_kfold_oof_mi_device).
Speed, warm, alternating: host 1.80-1.84 s, device 0.49-0.50 s. Dispatch: GPU usable, `MLFRAME_KFOLD_OOF_MI_DEVICE` not 0 and the plug-in MI
dispatcher's tuned choice (`lookup_mi_classif_backend`) for the size; any device failure runs every fold on the host pool.
