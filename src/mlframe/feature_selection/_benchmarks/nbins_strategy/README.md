# nbins_strategy benchmarks

Two benchmarks of the same question, "which `nbins_strategy` should MRMR use", side by side. They answer it with different criteria, which is why they can look like they disagree.

| | `gt_sweep.py` (ground truth) | `linear_oof.py` (out of fold) |
|---|---|---|
| criterion | which columns MRMR **selects** against known relevant / redundant / irrelevant roles: recall, precision, false-positive rate, F1; also RMSE / logloss of a linear model and CatBoost on the selected columns | what a **linear model** does on the selected columns, out of fold, selection made inside each training fold; plus how many of the generating columns the selected features use |
| data | synthetic multi-column problem and a real wellbore frame (5 configs) | `x0*x1 + 0.7*x2 - 0.5*x3**2 + 0.3*x4` signal in 50 normal columns, binary and regression targets, several seeds |
| selection counted as | raw columns | any selected feature, raw or engineered, by the source columns it uses |
| results | `results/mrmr_binning_gt_sweep_2026-07-21.{csv,json}` | printed table, `--out FILE.json` |

Row labels of the 2026-07-21 sweep are the historical ones: `fayyad_irani_*` is `mdlp`. `fayyad_irani_insample_validated` is the **default** `mdlp` (`mdlp_fast_mode=False`), `fayyad_irani_fast` is
`mdlp` with `mdlp_fast_mode=True`, `fayyad_irani_oos_validated` is `mdlp_validated`, `qs_gupta` is `quantile_spacing`, `mah_sci` is `mah`.

Run: `python -m mlframe.feature_selection._benchmarks.nbins_strategy.linear_oof --targets binary,regression --seeds 0,1 --out table.json` (nothing else heavy beside it: the wall of a fold is part of the table).
The sweep: `python -m mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite --full` (the harness lives in `gt_sweep.py`; the suite re-exports it).

See the tracker row `MRMR-NBINS-STRATEGY-LINEAR-OOF-2026-11-22` for the numbers and for how the two sets of results fit together.
