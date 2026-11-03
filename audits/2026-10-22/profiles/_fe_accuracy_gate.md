# _fe_accuracy_gate

## Engineered-column accuracy gate at 100k (2026-11-03)
100k x 100 GPU fit: 1.4 s of main-thread wait in `_gate_base_columns` (the candidates are already probed on a thread pool). Per candidate
`keep_engineered_over_source` costs ~25.9 ms: subsample 0.16 ms (rng.choice of 5000 of 100000), `measure_feature_uplift` 24.9 ms = 10 folds x
(standardise + Newton logistic + fast AUC); cProfile of 20 probes: `_newton_kernel` 0.30 of 0.54 s (1.5 ms per fold, ~8 iterations of an exp per row
for the gradient and exp + log1p per row for the line-search loss), probe glue 0.05, fold split 0.04. Not changed: skipping the line-search loss when
the full step is accepted would change the iterates' rounding (same optimum) for at most ~0.4 s of fit; a batched device Newton over all candidates and
folds is the larger lever if this gate grows.
