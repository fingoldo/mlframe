# Audit: training_feature_handling_targets

**Scope**: `src/mlframe/training/feature_handling/**`, `src/mlframe/training/targets/**`,
`src/mlframe/training/ranking/**`, `src/mlframe/training/strategies/**`
(excludes `src/mlframe/feature_selection/filters/**` and
`src/mlframe/feature_selection/shap_proxied_fs/**`, out of scope per a prior dedicated audit).

**Files reviewed**: 62 (every `.py` file under the four scope directories, including
`_benchmarks/` scripts).

**LOC reviewed**: ~18,040 (feature_handling 8,845 + targets 4,370 + ranking 2,685 +
strategies 2,140, per `wc -l`).

## Summary

This cluster has clearly been through multiple prior hardening passes — nearly every file
carries explicit "round-3 fix" / "wave NN fix" / "audit D P1-N" provenance comments describing a
previously-found-and-fixed bug, and the numerically sensitive paths (target-encoder moments,
residual-audit skew/kurtosis, temporal-audit aggregation) already use the numerically-stable
two-pass `z=(y-mean)/std` pattern rather than the catastrophic raw-moment-expansion formula that
was found buggy elsewhere in the repo this cycle (per project memory). As a result this pass
surfaced a small number of real, previously-unfixed issues rather than a large backlog.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| TRAINING_FEATURE_HANDLING_TARGETS-1 | P2 | `feature_handling/ordered_target_encoder.py:130-139` | `ordered_target_encode_batch` is missing the `noise_count_halflife` parameter that its sibling `ordered_target_encode` exposes; batch callers can't get count-decayed noise. | Add `noise_count_halflife` to the batch signature and apply the same per-column decay as the single-column function. | AST scanner: for any `f`/`f_batch` sibling-function pair, flag when the batch variant's keyword-only parameter set is a strict subset of the singular variant's. |
| TRAINING_FEATURE_HANDLING_TARGETS-2 | P2 | `feature_handling/apply.py:457-527` | `feature_handling_apply`'s target-encoder integration never threads `sample_weight` into `LeakageSafeEncoder.fit`/`fit_transform`, even though the encoder fully supports weighted encoding; `TargetEncodeParams`/`CatHandlerSpec` expose no `sample_weight` slot at all. | Add a `sample_weight` parameter to `feature_handling_apply` (and thread it into `_apply_target_encoder`'s `_fit()` closure) so weighted-training suites get weighted target-encoding instead of silently unweighted. | Grep/AST scanner over the FHC layer: any call to a method whose signature declares `sample_weight` where a `sample_weight` variable is in scope at the call site but not passed. |
| TRAINING_FEATURE_HANDLING_TARGETS-3 | P2 | `targets/_target_temporal_plot.py:111` | `plot_target_over_time` hardcodes `ax.set_ylim(-0.02, 1.05)` unconditionally, but `target_type="regression"` (explicitly supported, `target_rate = mean(y)`) is unbounded — the chart silently clips/hides real data for any regression target audit. `drift_warn_threshold=0.10` is also calibrated for a `[0,1]` rate and is meaningless at regression scale. | Compute `ylim` from the actual `target_rate` range (with small padding) when `result.target_type != "binary_classification"`, and document/scale `drift_warn_threshold` per target_type. | Parametrized test: call `plot_target_over_time` with a regression `TemporalAuditResult` whose `target_rate` values are e.g. in `[1000, 5000]`; assert `ax.get_ylim()` actually contains the data range. |
| TRAINING_FEATURE_HANDLING_TARGETS-4 | P3 | `ranking/_ranker_fs.py:404` | `group_aware_mrmr_select`: `cap = min(n, int(max_features)) if max_features else n` — `max_features=0` is falsy, so passing `max_features=0` (meaning "cap at 0 features") silently falls through to "no cap" instead of selecting nothing. | Use `if max_features is not None else n` instead of the truthy check. | Repo-wide AST grep for `if <param>` / ternaries gating on a numeric parameter whose name matches `max_*`/`min_*`/`n_*`/`*_count` where `0` is a semantically valid value. |
| TRAINING_FEATURE_HANDLING_TARGETS-5 | P2 | `strategies/__init__.py:297-305` | `is_neural_model()`'s string-alias set includes the literal `"recurrent"`, but `MODEL_STRATEGIES` (the actual registry `get_strategy()` reads) has no `"recurrent"` key — only `lstm`/`gru`/`rnn`/`transformer` map to `RecurrentModelStrategy`. Live-verified: `is_neural_model("recurrent") is True` while `get_strategy("recurrent")` emits `UserWarning: Unknown model 'recurrent', defaulting to TreeModelStrategy` and returns the wrong strategy (no scaling/imputation). This is the exact bug class already found and fixed once in this codebase (`_helpers_training_configs.py`, regression-tested in `test_recurrent_alias_builds_mlp_config`), left unfixed in this sibling function. `strategies.is_neural_model` also duplicates (with diverging alias lists) `training.models.is_neural_model`, which is the version every real production call site actually imports — `strategies.is_neural_model` has zero internal callers, is public API (`__all__`), and is the only one carrying this specific bug. | Drop `"recurrent"` from the alias tuple (or add a `"recurrent"` key to `MODEL_STRATEGIES`); longer-term, deduplicate the two `is_neural_model` implementations (e.g. have `strategies.is_neural_model` delegate to `training.models.is_neural_model` for the string-alias branch) so the alias lists cannot drift apart again. | Cross-consistency meta-test: for every `(is_X(name), get_X(name))` classifier pair sharing a registry, iterate every string literal in `is_X`'s own alias set and assert `get_X` resolves it without an "unknown model" warning and to a strategy consistent with `is_X`'s verdict. |
| TRAINING_FEATURE_HANDLING_TARGETS-6 | P3 | `targets/regression_residual_audit.py:182` | `EXCESS_KURT_MILD = 1.5` is declared with a comment implying it is a real threshold ("0.5-1.5 -> mild leptokurtosis") but is never referenced anywhere else in the module; `_diagnose` hardcodes the same `1.5` boundary twice via `EXCESS_KURT_HEAVY` instead. A future edit to "loosen mild leptokurtosis" via `EXCESS_KURT_MILD` would silently do nothing. | Delete `EXCESS_KURT_MILD`, or wire it into the `_diagnose` boundary it's documented to control. | Repo-wide AST scan: module-level UPPER_CASE constant assigned once and never read anywhere else in the same file/package (a class of dead code plain unused-import linters don't catch). |
| TRAINING_FEATURE_HANDLING_TARGETS-7 | P3 | `ranking/ranker_suite.py:123-183` vs `ranking/ranking.py:698-786` | Two independently-implemented "Borda" fusions live in the same package under the same name: `ranker_suite.borda_fuse` computes `score = Σ(group_size - rank)` per member (classic Borda count), while `ranking.ensemble_ranker_scores(method="borda")` computes `score = -Σ(rank)` (no group-size term). For a fixed query (same group size across all members compared) the two induce the same relative ordering, but the two are not the same statistic and a caller mixing raw scores from both entry points (e.g. logging/comparing across differently-sized queries) would get inconsistent numbers under the same "Borda" label. | Either make `ensemble_ranker_scores`'s borda branch call `borda_fuse` directly (passing per-row group sizes derived from `group_starts`), or document explicitly that the two are order-equivalent-only, not value-equivalent. | Property test: for synthetic per-model rank arrays within one query group, assert `ranker_suite.borda_fuse(...)` and `ranking.ensemble_ranker_scores(..., method="borda")` produce identical **orderings** (`np.argsort` equal), flagging any future edit that breaks even order-equivalence. |

## Counts

- P0: 0
- P1: 0
- P2: 4
- P3: 3

## Dimension coverage notes (explicit, per instructions)

- **Computational efficiency**: no actionable inefficiency found. Every hot loop in this cluster
  (target-encoder factorize/bincount paths, `_ranker_fs.py`'s fused njit MI kernels, the
  `ordered_target_encode`/batch cumsum paths, `PipelineCache`'s LRU accounting) is already
  vectorised or njit-fused, with documented bench evidence (`_benchmarks/`) for every
  non-obvious choice. Nothing rose to a reportable finding.
- **OSS/hygiene (comment cruft, mojibake, stale audit markers)**: none found. Comments
  consistently follow the repo's WHY-not-WHAT convention; no dashes-in-prose, no mojibake, no
  stray phase/wave banners beyond the legitimate provenance notes the repo's own convention
  permits (dated bug-fix explanations, not process narration).
- **Mutable-default-argument bugs, broad `except` clauses hiding real errors, silent
  exception-swallowing**: reviewed extensively; every broad `except` seen in this cluster logs
  at `warning`/`debug` with the exception object and a clear rationale for why it's non-fatal
  (e.g. `cache.py`'s disk-write failures, `hf_provider.py`'s CUDA-context-loss recovery). No
  instance of the "silently downgrade to a 100x-slower/wrong path with no log line" bug class
  documented elsewhere in this repo's history was found recurring in this cluster.
- **Data leakage / train-val-test boundary violations**: `LeakageSafeEncoder`'s OOF K-fold path,
  `ordered_target_encode`'s causal/expanding-mean design, and the polars-Enum "train+val union,
  test excluded" category-map construction in `strategies/_cat_levels_shared.py` /
  `hgb.py` / `xgboost.py` are all correctly leak-free and explicitly documented as such; no
  violation found.
- **Reproducibility / unseeded RNG**: `LeakageSafeEncoder`'s default `random_state=42`,
  `ordered_target_encode`'s `SeedSequence`-based per-column spawning, and
  `_target_distribution_analyzer_features.py`'s fixed `random_state=0` sampling are all
  deterministic by default; no unseeded-RNG finding.

## Narrative

### TRAINING_FEATURE_HANDLING_TARGETS-1 — `ordered_target_encode_batch` missing `noise_count_halflife`

`ordered_target_encoder.py`'s single-column `ordered_target_encode` accepts an opt-in
`noise_count_halflife` parameter that decays the injected regularisation noise by each row's
own causal observation count (documented rationale: early, low-confidence expanding-mean
estimates get more noise, well-established categories get less). The batch variant
`ordered_target_encode_batch`, whose entire purpose is to let a caller encode several columns
sharing one `(y, order)` pair without repeating the sort/prior computation, has no such
parameter at all — its signature stops at `noise_std`. I found this by diffing the two
functions' parameter lists side by side (they're otherwise near-identical in structure) and
confirmed no test exercises `noise_count_halflife` on the batch path (grepped
`tests/training/feature_handling/test_biz_val_ordered_target_encoder_batch.py`, no match) —
consistent with the parameter simply not existing to test. A real caller migrating from the
single-column API (e.g. `categorical_powerset_concat`'s `prune_against_target`, cited in the
batch function's own docstring as the intended consumer) to the batch API for its documented
performance win would silently lose the halflife-decay feature with no error.

### TRAINING_FEATURE_HANDLING_TARGETS-2 — target-encoder sample_weight gap in `feature_handling_apply`

`LeakageSafeEncoder.fit`/`.fit_transform` (target_encoders.py) fully support a `sample_weight`
argument — weighted per-category means, weighted WoE cell mass, weighted global prior — with an
explicit contract that `sample_weight=None` reproduces legacy byte-for-byte behaviour. However
`apply.py::_apply_target_encoder`'s `_fit()` closure calls
`enc.fit_transform(train_col, list(train_target))` with no third argument, and neither
`TargetEncodeParams` (handlers.py) nor `CatHandlerSpec` nor `feature_handling_apply`'s own
signature has any `sample_weight` slot to source one from. Per this audit's explicit ML-
correctness checklist ("sample_weight threading gaps"), this means any suite that trains with
per-row weights (a routine mlframe pattern for class-imbalance / temporal-decay weighting) and
also routes categorical columns through the FHC target-encoder path gets **unweighted**
encodings silently — the weighting intent is dropped exactly at this seam. I verified this by
reading every call site between `feature_handling_apply`'s public signature and
`LeakageSafeEncoder.fit_transform`; there is no `sample_weight` anywhere in that chain.

### TRAINING_FEATURE_HANDLING_TARGETS-3 — hardcoded `[0,1]` y-axis breaks the regression temporal-audit chart

`target_temporal_audit.py`'s public `audit_target_over_time`/`audit_targets_over_time` both
explicitly support `target_type="regression"` (`rate = mean(target)`, not a probability — see
`_aggregate_by_time_pandas`'s `else: rate = s.groupby("__bin")[target_col].mean()` branch and
the docstring's own worked example, `("amount_spent", "regression")`). But
`_target_temporal_plot.py::plot_target_over_time` unconditionally sets
`ax.set_ylim(-0.02, 1.05)` — a range that only makes sense for a `[0,1]` binary rate. For a
regression target (e.g. mean order value in the thousands), every real data point renders
outside the visible axis range, so the chart silently shows nothing informative (an empty
band or a flat line pinned at the axis edge) instead of raising or adapting. I confirmed the
target_type/rate contract by reading `_aggregate_by_time_pandas`/`_polars_rate_expr` directly,
and confirmed the `set_ylim` call has no `target_type` guard anywhere nearby in
`_target_temporal_plot.py`. The companion `drift_warn_threshold=0.10` default ("0.10 = a 10pp
swing" per its own docstring) is likewise calibrated only for the `[0,1]` case and produces a
meaningless threshold at regression scale — same root cause (the module was extended to support
regression targets without auditing every `[0,1]`-scale assumption downstream).

### TRAINING_FEATURE_HANDLING_TARGETS-4 — falsy-zero `max_features=0` footgun in `group_aware_mrmr_select`

`ranking/_ranker_fs.py::group_aware_mrmr_select` computes
`cap = min(n, int(max_features)) if max_features else n`. Python's `if max_features` treats
`0` the same as `None` (both falsy), so a caller passing `max_features=0` — a reasonable way to
say "select nothing via this path" — gets the "no cap" branch (`cap = n`) instead, silently
selecting up to all features. This is a low-probability-but-real edge case (most callers pass a
positive int or omit the kwarg entirely, matching `mrmr_kwargs.get("max_features")`'s default
`None`), so P3, but it is a genuine falsy-zero bug class the project's own memory log flags
repeatedly (`select_target`'s `hyperparams_config.model_fields_set` handling nearby in this same
audit cluster shows the team is aware of the None-vs-explicit-value distinction elsewhere) —
this one instance was missed.

### TRAINING_FEATURE_HANDLING_TARGETS-5 — `is_neural_model("recurrent")` disagrees with `get_strategy("recurrent")`

Confirmed live in a Python REPL against the installed package:

```
>>> from mlframe.training.strategies import is_neural_model, get_strategy, MODEL_STRATEGIES
>>> 'recurrent' in MODEL_STRATEGIES
False
>>> is_neural_model('recurrent')
True
>>> get_strategy('recurrent')
UserWarning: Unknown model 'recurrent', defaulting to TreeModelStrategy
<TreeModelStrategy instance>
```

`is_neural_model`'s alias tuple in `strategies/__init__.py` (`("mlp", "recurrent", "ngb", "lstm",
"gru", "rnn", "transformer")`) includes the literal string `"recurrent"`, but the actual
model-name -> strategy registry `MODEL_STRATEGIES` never registers that key — only the four
concrete aliases (`lstm`/`gru`/`rnn`/`transformer`) map to `RecurrentModelStrategy`. This is the
identical bug class already found and fixed once in this exact codebase: `tests/training/
test_training_configs_scope_instance_alias.py`'s docstring states verbatim "`_mlp_in_scope`
matched only literal `"mlp"`/`"recurrent"`. The actual recurrent aliases (lstm/gru/rnn/
transformer) were never matched" — describing a fix applied to
`_helpers_training_configs.get_training_configs`, but not to this sibling function. I traced
every internal call site of `is_neural_model` (`grep -rn is_neural_model src/`) and found the
production dispatch code (`core/_ar_skip.py`, `core/_phase_train_one_target_body.py`,
`core/_phase_config_setup.py`, `composite/_estimator_dispatch.py`) all import
`training.models.is_neural_model` — a *different*, independently-implemented function with its
own `NEURAL_MODEL_TYPES` constant — never `strategies.is_neural_model`. The buggy function in
this cluster therefore has zero internal callers today, but it is exported in
`strategies/__init__.py`'s `__all__` (public API surface) and is directly unit-tested
(`test_is_neural_model_alias_and_instance_parity`, which deliberately excludes `"recurrent"`
from its alias loop — the test author already knew not to trust that literal, but nobody
removed it from the source). Any external caller of the public
`mlframe.training.strategies.is_neural_model` function, or any future internal caller who
reasonably assumes it agrees with `get_strategy`, hits the same misclassification the sibling
bug already proved is a real production hazard (wrong strategy = no scaling/imputation for what
should be a neural model).

### TRAINING_FEATURE_HANDLING_TARGETS-6 — dead `EXCESS_KURT_MILD` constant

`regression_residual_audit.py` declares five kurtosis-band constants at module scope, all
consumed by `_diagnose` except `EXCESS_KURT_MILD` (`1.5`), which appears exactly once — its own
declaration. `_diagnose`'s "mildly leptokurtic" branch (the band the name and comment
("0.5-1.5 -> mild leptokurtosis") describe) is instead bounded by the literal `EXCESS_KURT_HEAVY`
constant reused as an upper bound, which also happens to equal `1.5`. I confirmed via
`grep -n EXCESS_KURT_MILD src/` (single hit, the declaration line) and by reading every branch of
`_diagnose` that references kurtosis thresholds. This is purely a hygiene/dead-code finding —
behaviour is correct today only because the two constants happen to share a value — but it is a
live footgun: editing `EXCESS_KURT_MILD` to "tune" the mild-leptokurtosis band (the name's
entire reason to exist) would have zero effect, and a reviewer skimming the constant block would
reasonably believe otherwise.

### TRAINING_FEATURE_HANDLING_TARGETS-7 — two divergent "Borda" implementations under one name

`ranking/ranker_suite.py::borda_fuse` and `ranking/ranking.py::ensemble_ranker_scores(...,
method="borda")` both claim to implement Borda-count rank fusion for the LTR suite, and both are
exercised in production (`_ranker_suite_train.py` calls the latter directly; `borda_fuse` is a
public top-level export of the `ranking` package intended for callers who already have raw
per-item ranks). Their formulas differ: `borda_fuse` computes the textbook
`Σ(group_size - rank)` (classic Borda count, an item ranked 1st in a 10-item group scores 9);
`ensemble_ranker_scores`'s borda branch computes `-Σ(rank)` with no group-size term at all (the
same item scores -1). Within one fixed-size query group the two are an affine (order-preserving)
transform of each other, so `compute_ranking_summary`'s NDCG/MAP outputs are unaffected either
way — but the two functions are not interchangeable if a caller ever compares or logs raw scores
side by side (e.g. cross-query score comparison, manual sanity-checking against the "textbook
Borda score" the `borda_fuse` docstring explicitly describes). I found this by reading both
implementations back to back while auditing the ranking-ensemble gate logic in
`_ranker_suite_train.py`; there is no shared helper or cross-reference between the two, so a
future edit to one's tie-breaking or weighting behaviour has no guarantee of staying consistent
with the other.

## Cluster-level meta-test ideas (apply beyond this one cluster)

1. **Sibling-function parameter-parity scanner**: for any two functions in the same module whose
   names differ only by a `_batch`/`_bulk`/`_many` suffix (or a documented "N-column variant of
   X" docstring cross-reference), AST-diff their keyword-only parameter sets and flag any
   parameter present on the singular form but absent from the plural form. Catches
   TRAINING_FEATURE_HANDLING_TARGETS-1 and is directly reusable for the several other
   single/batch pairs in this repo (e.g. `_categorical_to_string_array` variants,
   `fingerprint_df` single-vs-multi).
2. **`sample_weight`-drop scanner**: static call-graph check — starting from every public
   training-suite entry point that accepts `sample_weight`, walk calls into any function whose
   own signature declares `sample_weight` and flag call sites that don't forward it. Directly
   reusable across the whole `feature_handling`/`composite`/`core` call chain, not just target
   encoders.
3. **Registry-consistency scanner**: for every `is_X(name)` / `get_X(name)` pair backed by the
   same conceptual registry (this cluster has two instances: `strategies.is_neural_model` vs
   `strategies.get_strategy`/`MODEL_STRATEGIES`, and the previously-fixed
   `_helpers_training_configs._mlp_in_scope` vs the real alias list), generate a property test
   that iterates every literal in the `is_X` alias set and asserts `get_X` resolves it without a
   fallback warning. This bug class has now recurred at least twice in this codebase.
4. **Hardcoded-axis-limit scanner**: grep for `set_ylim(`/`set_xlim(` calls with two numeric
   literal arguments inside any plotting function whose docstring or type signature accepts a
   `target_type`/metric-kind parameter with more than one possible scale; flag for manual review.
