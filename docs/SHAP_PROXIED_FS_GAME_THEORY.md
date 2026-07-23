# ShapProxiedFS game-theory extensions (gt_01–gt_09): final verdict

Nine research plans (`research/gt_01_*.md` … `research/gt_09_*.md`) extended `ShapProxiedFS`
and MRMR with cooperative/non-cooperative game-theory constructs beyond the vanilla
Shapley-value proxy game (`v(S) = -subset_loss(S)`, additive attribution). All nine are
implemented, unit- and biz-value-tested, and merged to `master`. This page is the honest
scorecard: what actually won, by how much, and — just as importantly — where a plausible
idea measurably did **not** help and shipped as an opt-in with the null result documented
rather than as a forced default.

**Bottom line.** Vanilla `ShapProxiedFS` (`proxy_mode="additive"`) was genuinely improved
in the one place that matters most for a default: **gt_08's `proxy_mode="auto"` is now the
constructor default**, because it is the one lever proven to be a majority-win under honest
holdout replication, with a hard overhead cap and bit-identical fallback when its own gate
finds nothing. Every other plan is a validated, real capability — but ships **opt-in**,
because either its win is narrow-niche (a specific interaction shape, a specific ensemble
duplication pattern) or its hypothesis was disproven outright (gt_03).

## Wired-in by default

### gt_08 — `proxy_mode="auto"` (interaction levers, auto-gated)
SNR-gated synergy screen (`su_synergy_screen`) that fires pairwise-interaction handling
**only** when a statistically significant synergy signal is detected between candidate
pairs — replacing the old unconditional `"interaction"` TreeSHAP-tensor mode as the thing
callers get without opting in.
- **Win:** on XOR-interaction beds, `auto_recall == 2/2` operands vs additive's `0/2`;
  `auto_auc >= additive_auc + 0.03`.
- **Safety:** on pure-additive and noise-buried-interaction beds the gate correctly stays
  silent (`n_kept_pairs == 0`), giving **bit-identical** selection to additive — wall-clock
  overhead capped at `<= 1.10x` additive wall.
- **Why it, and not the raw `"interaction"` mode, is the default:** the underlying O(P²)
  TreeSHAP interaction tensor (gated to P≤16) only won 1 of 6 benched regimes outright — a
  true minority-win, correctly rejected as a default and kept opt-in.
- **Weakness:** the auto-gate is a screen, not a guarantee — on a regime where synergy is
  real but below its SNR threshold, it silently falls back to additive (safe, but leaves
  recall on the table until the threshold is loosened by hand).

## Opt-in — real, validated wins on a specific data shape

### gt_01 — Faith-Shap order-2 interaction index (`proxy_mode="faith_interaction"`)
Weighted-ridge estimate of the Faith-Shap interaction index — the *unique* interaction
index that is least-squares-optimal at reconstructing the coalition game from linear +
pairwise terms (Tsai/Yeh/Ravikumar 2023), estimated over sampled coalitions of the same
proxy game rather than the full O(P²) TreeSHAP tensor.
- **Win:** on a pure-XOR bed, `faith_recall == 2/2` vs additive `0/2`; `faith_auc >=
  additive_auc + 0.05`; matches or beats the old TreeSHAP `"interaction"` mode (`faith_auc`
  within `-0.01`) at a fraction of its memory/compute cost.
- **Safety floor:** on a pure-additive bed it must **not** diverge from additive
  (`jaccard >= 0.9`) — a stability guarantee, not a strength in its own right.
- **Weakness:** needs a screened candidate-pair set (never the full k² design — at proxy
  width 112 a full pairwise design would be underdetermined against a ~2048-coalition
  sample); only as good as that upstream screen.

### gt_09 — Two-phase residual attribution (`residual_passes=1`)
A second SHAP pass on the *residuals* of the first model's predictions — boosting applied
to attribution, not to the prediction itself — to recover credit for weak features whose
signal a strong feature already absorbed in pass one.
- **Win:** on a strong+weak weight-mix bed, `residual_weak_recall >= 3` of the weak set,
  `auc_residual >= auc_default - 0.005` (no honest-AUC cost).
- **Guardrail:** on a pure-strong bed with no weak signal to recover, the residual pass adds
  **zero** noise columns (explicitly asserted) — it doesn't manufacture false credit when
  there's nothing left to find.
- **Weakness:** selected-set growth is deliberately capped at `+1` member vs the default
  selection — it recovers, it doesn't bloat; validated only on this specific weight-mix
  synthetic, not a general claim.

### gt_02 — Least-core / nucleolus stability refine (`refine_mode="core"`)
Replaces the greedy `parsimony_tol` member-drop rule with a least-core allocation (every
retained feature's credit is stable against any blocking sub-coalition) to decide which
*weak* features survive the refine step.
- **Win:** on a 6-strong+6-weak weight-mix bed, `core_weak_recall >= 5/6` (measured 6/6 on
  the pinned seed) vs greedy's much lower recall, at `<= 0.005` downstream AUC cost.
- **Safety:** an adversarial threshold (`core_drop_threshold=0.9`) correctly trips an honest
  fallback gate that reproduces greedy's exact selection rather than degrading silently.
- **Weakness:** the flip side of gt_08's plan-literal-fixture finding — core-refine's edge
  over greedy is real but regime-dependent (log-loss saturation can erase the gap on other
  weight/signal combinations); documented plainly in the test rather than hidden.

### gt_07 — FE-generator-family Shapley budgeting (`fe_budget_learning=True`)
Additive-Shapley-style credit (sum of surviving-column importance) attributed to each FE
generator family (triplet, quadruplet, …), feeding an ROI-proportional budget-reallocation
loop that **persists across separate `MRMR.fit()` calls** on the same dataset (keyed by a
column-set fingerprint) — not multiple fits inside one call.
- **Win, with real numbers:** on a useless-quadruplet / useful-triplet bed, quadruplet's
  budget share drops `0.333 -> 0.183` (a 45% cut, asserted `>= 40%`) while triplet's rises
  `0.333 -> 0.600` across repeated fits, with a floor (`>= 10%` of equal-share) keeping a
  temporarily-starved family alive so it can recover after a signal flip.
- **Weakness:** v1 credit is an *additive* Shapley approximation, not true leave-one-family-
  out Shapley; floor-protection trades some recovery latency (2 fits) for that safety.
- **Real economic effect:** the win compounds over repeated calls to `MRMR.fit()` on the
  same dataset (e.g. a daily retraining job) — each call starts from the *previous* call's
  learned budget, not from scratch. It is not multiple internal fits per call.

### gt_05 — Shapley ensemble-member weighting/pruning (`shapley_blend`)
Shapley value over ensemble members as players (`v(C)` = blended OOF score of coalition
`C`), used as a redundancy-stable alternative to NNLS/hill-climb weighting when the
ensemble contains near-duplicate models.
- **Win:** `jaccard_shapley >= jaccard_nnls + 0.10` for pruning-set stability under model
  duplication; identical models get equal Shapley values (within tolerance); a genuinely
  useless (dummy) model's value stays below `5*stderr + 0.02` and gets correctly pruned to
  `< 0.05` weight; incremental-vs-naive Shapley computation is bit-identical (`atol=1e-10`).
- **Weakness:** the efficiency-axiom check (`sum(values) == v(full) - v(empty)`) only holds
  to `abs=0.03` — some approximation slack remains in the value estimates themselves.

### gt_06 — Adversarial reweighting (χ²-ball DRO) + adversarial-validation shift diagnostic
A genuinely non-cooperative two-player game (distributionally-robust reweighting: model vs.
an adversary maximizing weighted loss under a χ² divergence budget) plus a cheap
adversarial-validation diagnostic for detecting train/test distribution shift.
- **Win:** on a simulated subpopulation-shift bed, the shift diagnostic's AUC `>= 0.75` with
  `>= 2/3` of the truly-shifted features surfaced in `top_shift_features`, vs `<= 0.55` on
  an unshifted control; DRO-weighted training beats unweighted by `>= +0.01` AUC.
- **Weakness / honest scope:** at `rho=0` (no robustness budget) weights correctly collapse
  to near-uniform — the mechanism only helps once a real uncertainty budget is set by the
  caller; explicitly scoped as research-exploratory, reusing gt_04's plumbing rather than a
  production-hardened default.

## Opt-in — implemented and validated, but for a use case outside `ShapProxiedFS` itself

### gt_04 — Data Shapley / KNN-Shapley row valuation (`mlframe.data_valuation`)
Not a feature selector: a **row**-valuation module (training *examples* are the players),
using the closed-form KNN-Shapley recursion — no retraining per coalition, unlike Monte
Carlo Data Shapley.
- **Win:** a label-noise detector built on it reaches `AUROC >= 0.85` and puts `>= 90%` of
  flipped-label rows below the clean-row median value; converting values to `sample_weight`
  beats unweighted training by `>= +0.01` AUC.
- **Weakness:** TMC-vs-KNN-Shapley agreement is only moderate (Spearman `>= 0.45`, not a
  tight bound); the widely-quoted "286-438x speedup vs a naive loop" figure lives only in an
  **unasserted benchmark script**, not pinned by any test — treat it as a lead, not a
  guarantee, until a regression test pins it.
- **Integration status:** genuinely not wired into `train_mlframe_models_suite` or any
  training default. The one real, existing seam for it is `_setup_sample_weight`
  (`training/_data_helpers.py:200`) — the single place a precomputed `sample_weight` array
  already flows into `fit_params` before every model's `.fit()` call, independent of model
  type or classification/regression. Wiring gt_04 in means: compute values, turn them into a
  weight vector, hand it to that existing seam — nothing upstream of it needs to change.

## Cautionary tale — an honest negative result, shipped anyway

### gt_03 — Banzhaf semivalue prescreen ranking (`prescreen_ranking="banzhaf"`)
Maximum-Sample-Reuse (MSR) Banzhaf estimator as an alternative to `mean|φ|` for the
prescreen ranking step, on the hypothesis that a semivalue would be more seed-stable than a
raw mean of an already-noisy Shapley estimate.
- **Hypothesis disproven, measured directly.** On a low-SNR bed,
  `test_biz_val_banzhaf_ranking_seed_stability_low_snr` shows Banzhaf is **less** stable
  across seeds, not more: Jaccard ≈ 0.41–0.43 across 4 seeds vs `mean_abs_phi`'s ≈ 0.53 — a
  documented regression (bounded, not hidden, by a `>= -0.20` guard).
- **Root cause, in the test's own words:** MSR-Banzhaf layers a *second* round of
  Monte-Carlo coalition sampling on top of an OOF-SHAP φ that is already variance-reduced —
  adding noise instead of removing it.
- **On clean, high-SNR data** both rankings recall all informative features and AUC
  diverges by `<= 0.005` — no regression there; the MSR-vs-exact-Banzhaf Spearman
  correlation is `>= 0.95` on a small exact-enumeration check (P=10), so the *estimator* is
  correct — it's the *hypothesis* (Banzhaf beats mean-φ for prescreen stability) that
  didn't hold.
- **Shipped anyway, opt-in, default unchanged** (`prescreen_ranking="mean_abs_phi"`) — per
  this repo's convention: a rigorously-measured null result is still a deliverable, not a
  discarded experiment. Anyone curious can flip the option and see the same regression on
  their own low-SNR data.

## What's actually left

- gt_04's `credit="loo"` upgrade for gt_07 (true leave-one-family-out Shapley credit instead
  of the additive v1 approximation) — specified in the gt_07 plan as a v2 item, not built.
- A regression test pinning gt_04's njit-vs-naive KNN-Shapley speedup (currently only in an
  unasserted benchmark script).
- The `_setup_sample_weight` adapter wiring gt_04's row values into
  `train_mlframe_models_suite` as an opt-in config flag — the seam exists, nothing consumes
  it yet.
