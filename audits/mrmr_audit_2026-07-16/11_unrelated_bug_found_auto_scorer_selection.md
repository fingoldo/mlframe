# Unrelated pre-existing bug found during finding #17 regression testing

**Not part of `03_code_quality_design.md`** -- surfaced incidentally while running the broader
`tests/feature_selection/mrmr/` regression slice after the random_seed/random_state rename
(finding #17). Logged here per project convention (fix now or leave a concrete, diagnosed
follow-up) rather than silently dropped.

## Failing test

`tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_auto_scorer_selection.py::TestPlugInWinsOnDiscreteBinned::test_auto_picks_plug_in_on_discrete_x`

Fails deterministically, `0/8` seeds (floor is `>=1/8`), reproducible standalone, unaffected by
any change in this audit pass (`_orth_auto_scorer_fe.py` / `_orthogonal_scorer_auto_fe.py` last
touched in `9e1deff79`, well before this session's work started).

## Root cause (confirmed via direct diagnostic, not guessed)

`select_best_scorer_per_column` (`src/mlframe/feature_selection/filters/_orth_auto_scorer_fe.py`)
picks the best scorer per engineered column by normalizing each scorer's LCB by its own
"headroom" scale (`per_scorer_scale[s] = max(max LCB s achieves across raw source columns,
median of those, 1e-12)`), then argmax-ing the normalized ratio across scorers.

On the discrete-binned fixture (single raw source column `x1`, 3 levels), HSIC's raw-`x1`-vs-`y`
LCB is tiny (~0.006-0.03, since HSIC's RKHS statistic is near its noise floor on a weak discrete
marginal), while HSIC's LCB on the strongly-engineered `x1__He2` column is only mildly larger in
ABSOLUTE terms (~0.03-0.04) -- but because the denominator (HSIC's own raw-column scale) is even
tinier, the RATIO explodes to ~4.0-4.1, dwarfing every other scorer's ratio (plug_in tops out
around ~0.08-0.19). Confirmed with a direct diagnostic across all 5 seeds:

```
seed=1  x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.188, ..., 'hsic': 4.038}
seed=7  x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.125, ..., 'hsic': 4.027}
seed=13 x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.084, ..., 'hsic': 4.090}
seed=42 x1__He2  best_scorer=hsic  lcb_norm={'plug_in': 0.163, ..., 'hsic': 4.078}
seed=101 x1__He2 best_scorer=hsic  lcb_norm={'plug_in': 0.089, ..., 'hsic': 3.984}
```

HSIC wins on every seed on `x1__He2` (the strongest engineered column), so plug_in never gets
picked as `best_scorer` for ANY column on ANY seed -- structurally, not by noise.

This is NOT a simple code bug (off-by-one, wrong sign, etc.) -- it's a calibration weakness in
the ratio-to-own-raw-baseline normalization scheme: when a scorer's raw-column baseline sits near
its own noise floor (as HSIC's does on a weak discrete marginal), any nonzero uplift on an
engineered column inflates its normalized ratio disproportionately versus scorers whose raw
baseline is NOT near their noise floor (plug_in's raw-`x1` MI is ~0.2, comfortably above its
floor). The test's own docstring already flagged this direction of drift when HSIC was added to
`SCORER_NAMES` at Layer 71 (win rate predicted to drop toward 1/8); it has now drifted one seed
further, to 0/8.

## Why not fixed in this pass

Fixing the normalization scheme (e.g., additive/z-score normalization instead of ratio-to-own-
floor, or a stability floor requiring the raw-baseline LCB to itself clear some SNR threshold
before a scorer's ratio is trusted) is a real algorithmic change to
`select_best_scorer_per_column`, used by every auto-scorer-selection caller across the FE
pipeline -- per project convention this needs its own dedicated multi-seed benchmark validating
selection-equivalence before shipping, not a blind tweak mid-way through an unrelated 22-finding
code-quality audit (wrong blast radius / wrong review context).

## Concrete next action

Dedicated follow-up pass on `select_best_scorer_per_column`'s normalization:
1. Try an additive (z-score-like) cross-scorer normalization instead of ratio-to-own-raw-max.
2. Or: gate a scorer's ratio-based "win" on its raw-baseline LCB clearing a minimum absolute
   floor relative to the OTHER scorers' raw baselines (so a scorer sitting at its own noise floor
   can't win purely via ratio inflation).
3. Re-run the full `n_boot`/seed sweep in `test_biz_value_mrmr_auto_scorer_selection.py` (all
   scorer combinations, not just discrete-binned) to confirm the fix doesn't regress HSIC's own
   legitimate wins elsewhere in that file.
