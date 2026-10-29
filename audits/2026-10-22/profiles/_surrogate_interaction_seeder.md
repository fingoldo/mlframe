# Surrogate GBM seeder (`_surrogate_interaction_seeder.py`) profile, 2026-10-29

Structure: `self_gate_reps` real fits plus `self_gate_reps` permuted-y fits (default 3+3), run sequentially with `num_threads=n_threads`.
Each fit draws its own 70/30 split from its seed, so the Dataset cannot be shared between fits without changing results.

Single binary fit, n=20000 (14000 train), p=40, 150 rounds, depth 4, 4 threads (`D:/Temp/seeder_split.py`):
- Dataset construct 0.056 s (about 20%)
- train 0.204 s
- predict 0.008 s

Six fits come to about 1.6 s for the binary case, so the 5.7 s seen in the all-flags fit comes from the multiclass target (one tree per class per round).
Earlier measurements: thread scaling saturates at 4 threads, and concurrent fits on separate thread pools were slower than sequential.
Not changed: no lever left that keeps the permuted-null and per-seed splits intact; the predict and construct shares are too small to matter.

## Multiclass confirmation and refreshed whole-fit numbers (2026-10-29)

Same bench with a multiclass target (`seeder_split.py <n_classes>`, 4 threads): train 0.40 s at 3 classes, 1.40 s at 8 classes (one tree per class per round), so six fits cost
about 2.4 s to 8.4 s. The 5.5 s seen in the all-flags fit (`apply_surrogate_gbm_seeder`) is the multiclass tree cost, not construct or predict.
Shortening the permuted-y fits would change the null the self-gate z-test is built on, so it is not selection-equivalent and stays unapplied.

Whole fit re-profile at HEAD (76.5 s wall, GPU shared): composite group-agg 6.0 -> 2.8 s, accuracy gate 4.9 -> 3.5 s, operand pool 6.1 s (seeder 5.5 s, gradient seeder 0.6 s).
