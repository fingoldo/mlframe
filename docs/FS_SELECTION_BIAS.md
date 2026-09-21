# Selection bias in feature selection: the four ways a number here could flatter itself

`docs/SELECTION_BIAS.md` does not exist and never did; the document people reach for under that name in
this repository is about prior shift and PU learning, which is a different subject entirely. This is the
missing one: what "selection bias" means when the thing doing the selecting is a feature selector, and
what this benchmark does about each case.

Every mechanism below has the same shape. Something is chosen by looking at data, and then the same data
is used to say how good the choice was. The number that comes out is not wrong by accident — it is biased
in a direction, and the direction is always flattering.

---

## 1. The selector's own optimum is not an estimate of anything

An arm that evaluates two hundred subsets and reports the best one is reporting a maximum over two hundred
noisy draws. Even when every subset is exactly as good as every other, the best of two hundred looks
better than average by roughly the spread of the noise times the expected maximum of two hundred standard
normals — which is around 2.7 standard deviations, not zero.

This is the winner's curse, and it applies to RFECV's internal cross-validation score, to
`iterative_zero_importance_pruning`'s best-of-rounds set, to the bandit selectors' arm value, and to any
hyper-parameter search reporting its best trial.

**What this benchmark does.** Every cell records the arm's own reported optimum (`selection_score`) next to
the honest holdout score, and `_winners_curse.py` reports the difference as a column. The gap is the bias,
measured rather than argued.

**What it refuses to do.** It will not compare a `selection_score` in one metric against a holdout score in
another. An arm optimising accuracy and scored on AUC produces a difference that is mostly the metric
change, and reporting that as optimism would be a made-up number with a real-looking decimal point. Those
rows are refused, not estimated.

---

## 2. Selecting on all the data and scoring on part of it

The classic form, and the one with the largest effect size in the literature: run the selector on the full
dataset, then cross-validate the *downstream model* on the same data using the selected columns. Ambroise
and McLachlan (2002) measured near-zero apparent error on pure noise this way.

The mechanism is simple. The selection has already seen every row, so the columns it chose are the ones
that happened to look good on the validation rows too.

**What this benchmark does.** The holdout is cut ONCE per `(scenario, dataset_seed)`, before any arm runs,
and the arm is fitted on the training half only. The downstream panel is fitted on the same training half
and scored on the holdout. Nothing that touches the holdout rows feeds back into a selection.

**How it is checked rather than asserted.** `tests/feature_selection/test_fs_hybrid_leakage_and_determinism.py`
plants a column that is pure noise on the training rows and a near-copy of the label on the holdout rows,
alongside genuine signal. An arm that ranks the plant above the real signal has been shown the holdout. It
is a specific, named answer key rather than a general "does it overfit" check.

---

## 3. Choosing the beds after seeing which ones flatter your method

This is the one a technical protocol cannot fix, because it is about the author rather than the code, and
it is the most relevant one here: this benchmark was designed and run by the author of one of the arms it
judges (MRMR).

Three mechanisms bound it, none of which eliminates it:

- **Beds declare what they expect to defeat, before the run.** Each scenario names the arms it is designed
  to break, and those declarations are scored afterwards. Roughly half hold. A suite whose predictions
  always held would be tuned to its author's beliefs; one whose predictions never held would not be
  measuring what it thinks it is.
- **The registry is hashed.** `REGISTRY.lock.json` carries every bed's structural hash. Adding a bed after
  seeing results is allowed — it is often the right response to a surprise — but it bumps the lock, shows
  up in the diff, and is reported as a post-hoc addition rather than blending into the pre-registered set.
  `BENCHMARK_PREREGISTRATION.md` section 2d names every bed added that way and why.
- **The disclaimer is the first paragraph of the atlas**, not a footnote.

**What remains, stated plainly.** The author still chooses which weaknesses to encode as beds. No lock file
detects a weakness nobody wrote a bed for.

---

## 4. Reporting only the cells that finished

An arm that crashes on wide beds and survives on narrow ones has a complete-case average computed over the
beds it finds easy. Since the hardest beds kill the weakest arms, dropping failed cells systematically
flatters exactly the arms that deserve it least. This is survivorship bias with the survivors chosen by
difficulty.

**What this benchmark does.** Every cell writes a record, including the ones that fail: `error`, `timeout`,
`crashed`, `oom`. `reliability_table` reports the completed fraction per `(arm, scenario)` and
`intention_to_treat_mean` scores a failed cell at the base rate rather than omitting it. An arm that cannot
finish is worse than one that returns something mediocre, and the aggregate says so.

---

## The one this benchmark cannot fix, and does not claim to

A benchmark is a sample of problems, and this one's sample is seventeen hand-picked beds plus seven real
datasets. It is not drawn from any population of real problems, so every COUNT in the atlas is a property
of this bed list. The qualitative separations — a marginal filter cannot reach through a collider, a
binned estimator loses resolution when a column has fewer distinct values than it wants bins — are the part
most likely to transfer, because they follow from what the methods are rather than from how many beds were
written.

That distinction is why the atlas reports separations in prose and counts in tables marked as counts.

---

## Reading a null result

One more failure mode, which is not bias but is regularly mistaken for its absence. "No arm beat the null
hypothesis" means two very different things depending on what the design could have detected:

- the difference was smaller than this contrast's minimum detectable effect, in which case the test never
  had a chance and the result is about the design;
- the difference was above that threshold and the test still did not reject, which is evidence about the
  arm.

Every leaderboard row now carries its own minimum detectable effect and the two states are labelled
separately (`underpowered` against `indistinguishable`). On the real leg the threshold is 0.013 to 0.031
AUC, larger than most gains the synthetic legs measured — so most of that leg's silence is the first case.

---

## References

- Ambroise & McLachlan (2002), *Selection bias in gene extraction on the basis of microarray gene-expression
  data*, PNAS 99(10).
- Reisach, Seiler & Drton (2021), *Beware of the Simulated DAG*, NeurIPS — varsortability, and why every
  column in this generator is standardised to unit variance.
- Nogueira & Brown (2016), *Measuring the stability of feature selection* — the stability index used here,
  and the only one with a known null distribution.
