# Pure-form retention: a pair form must beat its better side

`retain_usable_pure_forms` re-attaches linearly usable two-operand forms the MI greedy left out. Its gates (non-separability of the form, whole-form
correlation with the class indicator) cannot tell a form that is good on its own merit from one that is good through ONE of its operands: on
`y = sign(x^2 - 1)` the forms `max(abs(x), sin(noise_k))` correlate with the class through `abs(x)` alone, the noise side only overwrites the rows
where `|x|` is small (where `y` is constant), and three of them displaced `abs(x)` from the support (`test_generic_migreedy_constructor`,
`test_hybrid_fe_stress::test_each_constructor_alone_emits_winner`, failing since before 2026-11-04).

`_fe_pure_form_sides.drop_forms_not_beating_their_sides` runs after the non-separability filter and before the greedy. Each `unary_binary` form is
compared with each side (`unary_a(a)`, `unary_b(b)`, replayed through the recipe's own `_apply_unary_side`) under the retention's scorer and a 3-fold split
(CV log-loss of the scaled logistic model for classification, CV MAE of OLS for regression, `_usability_scorers`), and kept only when its loss is below
0.99 x the better side's (the greedy's own relative margin). Raw passthroughs and forms whose sides cannot be replayed are kept.

Measured on the failing fixture (seed 1): `max(abs(x),sin(noise_c))` 0.0765 vs `abs(x)` 0.0578; `max(sqr(x),sin(noise_c))` 0.0910 vs 0.0643;
`max(sqrt(x),sign(noise_b))` 0.1331 vs 0.0631: every noise variant is dropped. Retention suites (pure-form, usability-aware, raw-column, GPU-resident
parity): 33 passed. Cost: three small CV fits per pair candidate that survives the earlier gates (the pool keeps at most `max_per_pair` = 3 per pair).

`mi_greedy_features_` keeps survivors only, so the tests that pin "the family emitted a |x|-family transform" read the new `mi_greedy_candidates_`
(everything the family appended; `hybrid_orth_candidates_` is the existing counterpart).

Not covered: `test_default_recovers_quadratic_regression[42, 7]` fails for another reason. The univariate bases and hinges of `x` are generated but the final MI
screen drops them, because `y` is a function of the quantile bins of `x` (MI(x__pade2; y | x) = 0), while a LINEAR downstream model needs them. No retention pass
re-attaches single-operand forms.
