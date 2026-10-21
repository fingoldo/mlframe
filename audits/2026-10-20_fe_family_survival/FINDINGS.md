Written for: mlframe maintainers working on MRMR feature-engineering families.

# End-to-end survival of every MRMR FE family: what the new tests found

The tests are `tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_family_survival.py` and `..._encodings.py` (shared code in `_fe_family_survival_common.py`). For each of 27 families one scenario is built in which only that family can express the target. `MRMR` is fitted with every default-on `fe_*_enable` flag that generates a family switched off (read from the constructor signature) and the family's own flags on. Six checks per family: created and selected, replay of the held-out frame, held-out lift of a downstream model, no column on a permuted target, detection power over a seed panel (14 or 20 seeds), false-positive rate over a seed panel. Power and false-positive rates are Clopper-Pearson 90 percent bounds against policy constants (`POWER_TARGET = 0.8`, `FALSE_POSITIVE_MAX = 0.2`). The constants are policy, not tuned per family.

Result of the first full run (162 test cases): 137 pass, 25 fail. Every replay, permuted-target and false-positive check passes for every family, so no family leaks a column on noise and every selected column replays on a held-out frame. The failures are in "created and selected", "lift" and "power".

Failing checks are not xfailed and no threshold was widened. They are listed in `OPEN_FINDINGS` of the common module and left out of the parametrization until the family is fixed; the registry is the list of what is not covered.

## Families that pass all six checks (12)

cat_num_interaction, conditional_dispersion, conditional_gate, integer_lattice, kfold_target_encoding, missingness_count, offset_product, oof_cell2d, oof_warp, pairwise_modular, row_stat, univariate_basis.

## Open findings

| Family | Check | Observed |
|---|---|---|
| binned_numeric_agg | created, lift, power | seed 0 creates no recipe (selected `g` only); held-out lift present on 2 of 3 seeds (R^2 0.85 against 0.00) and absent on seed 0; detected in 10 of 14 seeds, lower bound 0.46. This fixture has 14 seeds and a smaller n than the one used for the gate work (17 of 20 at 30000 rows), so the power floor of 0.8 is above what a 10-cell permutation test can reach here |
| modular | created, lift, power | no recipe of any kind is created (selected `t` only), 0 of 20 seeds. Traced: the family's own uplift test compares the engineered column's MI with the raw column's. The raw counter `t` has 200 levels and a plug-in MI of 2.294 nats (about ln 10, the ceiling for 10 bins); the best engineered column scores 2.294 (`t__modsin_365`) and the period-7 candidates 2.2 to 2.25, so the uplift is at most 0.9997 against a requirement above 1, and nothing is appended. A raw counter with many levels already holds all the information as a lookup table, so an MI uplift over it cannot exist; the family could only help a model that cannot use 200 levels (a linear one) |
| univariate_fourier | created, power | no recipe whose name carries `__sin` or `__cos` is created; the selection is `a` and `a__L3` (a Legendre column), 0 of 20 seeds. A cubic polynomial column recovers the sine scenario instead |
| pair_cross | created, power | the orthogonal pair-cross recipe is never created; the generic pair search produces `mul(prewarp(x0),prewarp(x1))` instead, detected 5 of 14 seeds, lower bound 0.15 |
| wavelet | lift, power | lift on seeds 1 and 2 is about 0.50 (R^2 0.55 against 0.05) and negative on seed 0; detected in 16 of 20 seeds, lower bound 0.60 |
| hinge | power | 12 of 14 seeds, lower bound 0.61 |
| row_argmax | power | 11 of 14 seeds, lower bound 0.53; this panel also takes 943 s on the benchmark host |
| missingness_indicator | power | 17 of 20, lower bound 0.66 |
| missingness_pattern | power | 14 of 20, lower bound 0.49 |
| grouped_delta | power | 19 of 20, lower bound 0.78 (just under the floor) |
| lagged_diff | power | 18 of 20, lower bound 0.72 |
| pairwise_log_ratio | power | 16 of 20, lower bound 0.60 |
| pairwise_ratio | lift | held-out R^2 with the selection is below the raw columns on all three seeds (0.50, 0.51, 0.49 against 0.55, 0.52, 0.52) although the ratio column is selected |
| count_encoding, frequency_encoding | lift | lifts are small and positive (0.015 and 0.024 on two seeds, about 0 on the third) on a baseline of 0.86 to 0.88: the scenario leaves almost no headroom, so this may be the fixture and not the families |

## What is a finding and what may be the fixture

Hypotheses, not established causes:

- Descriptor errors found and fixed while reading the first run: `pairwise_ratio` and `pairwise_log_ratio` produce recipes of kind `pairwise_ratio` (told apart by the `ratio__` and `log_ratio__` name prefixes), not `div` and `log_div`; the common module now narrows a kind with a name pattern.
- `modular`, `univariate_fourier`, `pair_cross` produce nothing of their own on their scenarios. For `pair_cross` and `univariate_fourier` another mechanism recovers the signal. These are the strongest candidates for real defects (a gate rejecting the family, or a scenario the family cannot express); this has not been traced to a gate.
- The power shortfalls of 0.5 to 0.8 (hinge, row_argmax, missingness, wavelet, lagged_diff, grouped_delta, pairwise_log_ratio) may be real or may be a scenario at the edge of detectability; nothing was done beyond counting.
- `binned_numeric_agg` was fixed for the gate defects found earlier in this series (`audits/2026-10-20_binagg_linear_usability/FINDINGS.md`); the power measured here is lower because of the smaller fixture.

## A pattern behind several "created" findings (hypothesis)

`modular` was traced (above): the MI-uplift test is unreachable when the raw column already carries the information to a binned or lookup estimator. The same holds for the binned-aggregation gate fixed earlier in this series (a column that is a function of the group's cell is redundant for any model that can use the group, not for a linear one). `univariate_fourier` (a sine of a continuous column, which 10 quantile bins of the raw column already describe) and `pair_cross` (a product that the generic pair search finds first) fit the same description: their scenarios are expressible by something cheaper, and the selection prefers it. If that is the cause, the remedy is the one used for the binned aggregation, a held-out linear-usability test for columns the MI test rejects, not a lower threshold. Not verified for `univariate_fourier` and `pair_cross`.

## Cost

The committed checks fit about 2 hours of panels on the benchmark host (`row_argmax` 943 s, `missingness_count` 913 s and 871 s, `missingness_indicator` 847 s, `missingness_pattern` 799 s and 635 s each in one test). The repository default per-test timeout of 900 s (`addopts`) is below that, so the panel tests carry `@pytest.mark.timeout(3600)` and `@pytest.mark.slow`. The table of every family's power can be printed with `tests/feature_selection/mrmr/biz_val/print_fe_family_power_table.py`.

## Not verified

- The open findings are from one run on one host; the seed panels are fixed so the numbers repeat, but the cost and the 14-seed panels were not re-run on CI hardware.
- The three "created" findings were not traced into the gates (rejection ledger) to name the gate.
