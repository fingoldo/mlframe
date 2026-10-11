# F2 whole-fit timing with all FE families and the residual rescue (2026-10-11)

Paired A/B, GPU strict-resident mode, the F2 example on fresh data per round, three arms fitted on the same frame back to back with the order rotating (`python -m mlframe.feature_selection._benchmarks.offset_product.ab_fit <n> <rounds>`). `off` = offset product, out-of-fold warp, row statistics, 2-D cell table and residual rescue all off; `families` = the four FE families on; `all` = the constructor default (families + rescue). Quiet box (one GPU job, nothing else heavy), medians of the rounds.

| rows | rounds | off | families | all | paired delta families | paired delta all | same selection as off |
|---|---|---|---|---|---|---|---|
| 100 000 | 3 | 4.54 s | 4.84 s | 5.26 s | +0.36 s (+7.8%) | +0.54 s (+11.9%) | 3 of 3 |
| 300 000 | 3 | 5.62 s | 5.97 s | 6.27 s | +0.53 s (+9.5%) | +0.58 s (+10.4%) | 3 of 3 |
| 1 000 000 | 5 | 11.26 s | 12.13 s | 11.86 s | +0.87 s (+7.7%) | +0.33 s (+3.0%) | 5 of 5 |

At 1M the two deltas are within the noise of the box (about 0.5 s: `all` came out faster than `families`), so the honest statement is: the new operators cost **0.3-0.9 s, 3-8%, on an 11.3 s fit at 1M**, and about 0.5 s at 100k. The F2 selection is identical with the operators on and off in every round (the example is already solved by the older machinery; the operators earn their keep on the targets they were built for, see their own reports).

## Per-stage cost (cProfile cumulative, 2 fits per size; cProfile inflates the whole fit, the stage figures are close to the real ones)

| stage | 100k | 1M |
|---|---|---|
| out-of-fold warp | 0.18 s | 0.18 s |
| offset product | 0.17 s | 0.19 s |
| residual rescue (admission, pair tables, pruning) | 0.12 s | 0.21 s |
| row statistics | 0.07 s | 0.07 s |
| sum | 0.54 s | 0.66 s |

The sum matches the A/B deltas. The residual rescue scans at most 50 000 rows, so it grows only with the conversion of the candidate columns (0.12 s to 0.21 s from 100k to 1M).

## Cold start

In two of five rounds at 1M the first arms with the operators on took 22-26 s (the same arms took about 11 s afterwards; `off` stayed at 10 s). It does not follow the arm, it happens in the first rounds after the warm-up fit, so it is a one-time cost of the first fits on a new row count (kernel and tuning-cache specialisations), not a property of the operators. It is not in the medians above. Open item: find which specialisation compiles at the first 1M fit with the operators on and move it into the warm-up (`mlframe-tune-kernels ensure`).
