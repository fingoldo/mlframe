# How much the Phase 0 design could have seen

The pre-registration recorded `R = 20` seeds as a declared floor rather than a measured one, and said
explicitly that the pilot would still be worth running afterwards to answer whether 20 is *enough*. This is
that answer, and it is measured off the confirmatory run's own cells rather than off a fresh pilot: 2 240
executed cells supply a direct estimate of `tau` for every (scenario, arm, model, K) combination, which is
strictly more information than the 40-cell pilot the pre-registration budgeted for.

Regenerate with:

    python -m mlframe.feature_selection._benchmarks.fs_hybrid._power \
        src/mlframe/feature_selection/_benchmarks/_results/phase0_confirm.jsonl

## What this changes about the kill criterion

The criterion fired: with lightgbm downstream, no arm beat `all-features` on four to seven of the seven
eligible real beds, at every K from k5 to k200. That result stands. What the power analysis adds is the
size of the claim it licenses.

At `R = 20` the minimum detectable effect is **0.013 AUC** on the median contrast and **0.031 AUC** on the
p90 contrast (lightgbm, k10). So the finding is:

> No arm delivered a gain **larger than roughly one to three AUC points** over using every feature.

It is **not** the finding that no arm delivered a gain. A true improvement of 0.005 AUC would need 134 seeds
at the median contrast and 715 at the p90 one to be detected at 80% power — between seven and thirty-six
times the executed design. Phase 0 could not have seen such an effect and did not test for it.

This asymmetry is worth stating plainly because it cuts in the direction *against* the conclusion the run
reached. A null result is only as strong as the effect the design could resolve, and this design resolves
effects that are large by the standards of the FS literature, where a one-point AUC gain is a publishable
claim.

## Why the noisy contrasts matter more than the median

The p90 column is not pessimism. The contrasts that set it are the ones where the arm loses badly and
erratically — `madelon`/`mrmr` moves by `tau = 0.099` per seed around a mean of `-0.194` — and a wide `tau`
there is a real property of the arm, not measurement slop. Sizing a design on the median `tau` means the
cells where selection behaves worst are also the cells where the design is least able to say anything.

Where `tau` is genuinely small the design is strong: at k200 the median contrast resolves 0.0024 AUC, which
is below any effect worth acting on. The problem is concentrated at small K and on the wide beds.

## Reading guide

- `R for d=X` — seeds needed to detect a true difference of `X` at 80% power, two-sided alpha 0.05, using
  the `t` critical value rather than the normal one (the normal approximation understates the requirement by
  one to three seeds in this range, and overstates achieved power at small `m`).
- `detectable at R=20` — the smallest difference the executed design resolves, i.e. the effect at which it
  had 80% power. Any null result in this benchmark must be read against this number.
- Quantiles are over (scenario, arm) contrasts within one (model, K). They are conditional on the
  hand-picked bed roster and do not transfer to a different grid.

## Measured tau and required replicates (estimated from executed cells, not from a pilot assumption)

Paired design, two-sided alpha=0.05, power=80%, metric=roc_auc.
`tau` is the sd of the per-dataset_seed difference `arm - all-features` within one scenario.
Quantiles are taken over (scenario, arm) contrasts, so p90 sizes the design for its harder cells
rather than its quiet ones.

## lightgbm @ k10

Observed tau over 91 (scenario, arm) contrasts: median 0.02049, p75 0.03279, p90 0.04765, max 0.09909

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.02049 | 826 | 134 | 35 | 10 | 2 | 0.01344 |
| p75 = 0.03279 | 2112 | 340 | 87 | 23 | 6 | 0.02152 |
| p90 = 0.04765 | 4457 | 715 | 180 | 47 | 9 | 0.03127 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `mrmr`: tau=0.09909, mean delta=-0.19374, m=20
- `madelon` / `skb-mi`: tau=0.08547, mean delta=-0.16448, m=20
- `madelon` / `knockoffs`: tau=0.07991, mean delta=-0.09822, m=20
- `arcene` / `select-fdr`: tau=0.05467, mean delta=-0.16332, m=20
- `arcene` / `skb-f`: tau=0.05467, mean delta=-0.16332, m=20

## lightgbm @ k100

Observed tau over 78 (scenario, arm) contrasts: median 0.00492, p75 0.01447, p90 0.03261, max 0.09909

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.00492 | 50 | 10 | 2 | 2 | 2 | 0.00323 |
| p75 = 0.01447 | 413 | 68 | 19 | 6 | 2 | 0.00950 |
| p90 = 0.03261 | 2089 | 336 | 86 | 23 | 5 | 0.02140 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `mrmr`: tau=0.09909, mean delta=-0.19374, m=20
- `madelon` / `lars-order`: tau=0.05555, mean delta=-0.20373, m=20
- `arcene` / `variance-sort`: tau=0.05060, mean delta=-0.12384, m=20
- `madelon` / `knockoffs`: tau=0.04737, mean delta=-0.04588, m=20
- `arcene` / `select-fdr`: tau=0.04351, mean delta=-0.05127, m=20

## lightgbm @ k20

Observed tau over 91 (scenario, arm) contrasts: median 0.01628, p75 0.03066, p90 0.04142, max 0.09909

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01628 | 522 | 85 | 23 | 7 | 2 | 0.01068 |
| p75 = 0.03066 | 1847 | 297 | 76 | 21 | 5 | 0.02012 |
| p90 = 0.04142 | 3368 | 541 | 137 | 36 | 8 | 0.02718 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `mrmr`: tau=0.09909, mean delta=-0.19374, m=20
- `madelon` / `knockoffs`: tau=0.07540, mean delta=-0.08284, m=20
- `madelon` / `skb-mi`: tau=0.05794, mean delta=-0.11563, m=20
- `arcene` / `select-fdr`: tau=0.05757, mean delta=-0.13348, m=20
- `arcene` / `skb-f`: tau=0.05757, mean delta=-0.13348, m=20

## lightgbm @ k200

Observed tau over 78 (scenario, arm) contrasts: median 0.00359, p75 0.01633, p90 0.02851, max 0.09909

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.00359 | 27 | 6 | 2 | 2 | 2 | 0.00236 |
| p75 = 0.01633 | 525 | 86 | 23 | 7 | 2 | 0.01071 |
| p90 = 0.02851 | 1597 | 257 | 66 | 18 | 4 | 0.01871 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `mrmr`: tau=0.09909, mean delta=-0.19374, m=20
- `madelon` / `lars-order`: tau=0.05374, mean delta=-0.19485, m=20
- `madelon` / `knockoffs`: tau=0.04068, mean delta=-0.04384, m=20
- `arcene` / `select-fdr`: tau=0.04015, mean delta=-0.03194, m=20
- `arcene` / `skb-f`: tau=0.04015, mean delta=-0.03194, m=20

## lightgbm @ k5

Observed tau over 91 (scenario, arm) contrasts: median 0.02749, p75 0.04373, p90 0.05472, max 0.10169

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.02749 | 1485 | 239 | 61 | 17 | 4 | 0.01804 |
| p75 = 0.04373 | 3754 | 602 | 152 | 40 | 8 | 0.02869 |
| p90 = 0.05472 | 5878 | 942 | 237 | 61 | 12 | 0.03591 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `skb-mi`: tau=0.10169, mean delta=-0.21594, m=20
- `madelon` / `mrmr`: tau=0.09924, mean delta=-0.19520, m=20
- `gisette` / `variance-sort`: tau=0.07752, mean delta=-0.13820, m=20
- `madelon` / `knockoffs`: tau=0.07311, mean delta=-0.15092, m=20
- `Bioresponse` / `variance-sort`: tau=0.06962, mean delta=-0.18231, m=20

## lightgbm @ k50

Observed tau over 91 (scenario, arm) contrasts: median 0.01151, p75 0.02297, p90 0.04058, max 0.09909

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01151 | 262 | 44 | 13 | 4 | 2 | 0.00756 |
| p75 = 0.02297 | 1037 | 168 | 44 | 13 | 2 | 0.01507 |
| p90 = 0.04058 | 3233 | 519 | 131 | 34 | 7 | 0.02663 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `madelon` / `mrmr`: tau=0.09909, mean delta=-0.19374, m=20
- `madelon` / `knockoffs`: tau=0.06378, mean delta=-0.06206, m=20
- `madelon` / `skb-mi`: tau=0.04805, mean delta=-0.07034, m=20
- `madelon` / `lars-order`: tau=0.04694, mean delta=-0.24221, m=20
- `arcene` / `rfecv`: tau=0.04574, mean delta=-0.04737, m=20

## logistic @ k10

Observed tau over 91 (scenario, arm) contrasts: median 0.01641, p75 0.04030, p90 0.06974, max 0.10374

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01641 | 530 | 87 | 23 | 7 | 2 | 0.01077 |
| p75 = 0.04030 | 3189 | 512 | 130 | 34 | 7 | 0.02645 |
| p90 = 0.06974 | 9546 | 1529 | 384 | 98 | 17 | 0.04576 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `hill-valley` / `boruta-shap`: tau=0.10374, mean delta=-0.27883, m=20
- `hill-valley` / `ace`: tau=0.10055, mean delta=-0.26823, m=20
- `arcene` / `variance-sort`: tau=0.09413, mean delta=-0.24156, m=20
- `hill-valley` / `skb-mi`: tau=0.08867, mean delta=-0.33613, m=20
- `arcene` / `univariate-mi`: tau=0.08437, mean delta=-0.16302, m=20

## logistic @ k100

Observed tau over 78 (scenario, arm) contrasts: median 0.01122, p75 0.01538, p90 0.03076, max 0.05614

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01122 | 249 | 42 | 12 | 4 | 2 | 0.00737 |
| p75 = 0.01538 | 467 | 76 | 21 | 7 | 2 | 0.01009 |
| p90 = 0.03076 | 1859 | 299 | 76 | 21 | 5 | 0.02019 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `arcene` / `variance-sort`: tau=0.05614, mean delta=-0.11895, m=20
- `arcene` / `select-fdr`: tau=0.05182, mean delta=-0.09946, m=20
- `arcene` / `skb-f`: tau=0.05182, mean delta=-0.09946, m=20
- `arcene` / `univariate-mi`: tau=0.04530, mean delta=-0.05111, m=20
- `arcene` / `boruta`: tau=0.04355, mean delta=-0.02362, m=20

## logistic @ k20

Observed tau over 91 (scenario, arm) contrasts: median 0.01657, p75 0.03706, p90 0.06479, max 0.11186

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01657 | 541 | 88 | 24 | 8 | 2 | 0.01087 |
| p75 = 0.03706 | 2697 | 433 | 110 | 29 | 6 | 0.02432 |
| p90 = 0.06479 | 8240 | 1320 | 332 | 85 | 15 | 0.04252 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `hill-valley` / `variance-sort`: tau=0.11186, mean delta=-0.30339, m=20
- `hill-valley` / `boruta-shap`: tau=0.09022, mean delta=-0.18863, m=20
- `hill-valley` / `univariate-mi`: tau=0.08973, mean delta=-0.25613, m=20
- `hill-valley` / `ace`: tau=0.07801, mean delta=-0.19432, m=20
- `hill-valley` / `skb-mi`: tau=0.07504, mean delta=-0.24070, m=20

## logistic @ k200

Observed tau over 78 (scenario, arm) contrasts: median 0.01044, p75 0.01419, p90 0.02226, max 0.05423

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01044 | 216 | 36 | 11 | 3 | 2 | 0.00685 |
| p75 = 0.01419 | 397 | 65 | 18 | 6 | 2 | 0.00931 |
| p90 = 0.02226 | 974 | 158 | 41 | 12 | 2 | 0.01460 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `arcene` / `variance-sort`: tau=0.05423, mean delta=-0.09410, m=20
- `arcene` / `select-fdr`: tau=0.04565, mean delta=-0.06667, m=20
- `arcene` / `skb-f`: tau=0.04565, mean delta=-0.06667, m=20
- `arcene` / `univariate-mi`: tau=0.03683, mean delta=-0.02784, m=20
- `scene` / `knockoffs`: tau=0.03221, mean delta=-0.07614, m=20

## logistic @ k5

Observed tau over 91 (scenario, arm) contrasts: median 0.02087, p75 0.04597, p90 0.07414, max 0.09810

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.02087 | 857 | 139 | 36 | 11 | 2 | 0.01369 |
| p75 = 0.04597 | 4149 | 666 | 168 | 44 | 9 | 0.03017 |
| p90 = 0.07414 | 10787 | 1728 | 434 | 110 | 19 | 0.04865 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `arcene` / `mrmr`: tau=0.09810, mean delta=-0.14752, m=20
- `arcene` / `univariate-mi`: tau=0.09604, mean delta=-0.20365, m=20
- `hill-valley` / `ace`: tau=0.08728, mean delta=-0.34759, m=20
- `arcene` / `skb-mi`: tau=0.08633, mean delta=-0.17651, m=20
- `Bioresponse` / `variance-sort`: tau=0.08518, mean delta=-0.11054, m=20

## logistic @ k50

Observed tau over 91 (scenario, arm) contrasts: median 0.01377, p75 0.02335, p90 0.04969, max 0.15674

| planning tau | R for d=0.002 | R for d=0.005 | R for d=0.01 | R for d=0.02 | R for d=0.05 | detectable at R=20 |
|---|---|---|---|---|---|---|
| median = 0.01377 | 374 | 62 | 17 | 6 | 2 | 0.00904 |
| p75 = 0.02335 | 1072 | 173 | 45 | 13 | 2 | 0.01532 |
| p90 = 0.04969 | 4848 | 777 | 196 | 51 | 10 | 0.03261 |

Noisiest contrasts (these set the p90, and are where a null result is least informative):

- `hill-valley` / `select-fdr`: tau=0.15674, mean delta=-0.34412, m=20
- `hill-valley` / `skb-f`: tau=0.15674, mean delta=-0.34412, m=20
- `hill-valley` / `univariate-mi`: tau=0.07149, mean delta=-0.11909, m=20
- `arcene` / `variance-sort`: tau=0.07040, mean delta=-0.16819, m=20
- `arcene` / `select-fdr`: tau=0.06591, mean delta=-0.12314, m=20
