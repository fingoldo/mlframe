# Composite Targets audit - DX / docs / API surface / test gaps (2026-06-10)

Scope: src/mlframe/training/composite/** public surface, README composite section,
CHANGELOG composite entries, docs/examples/composite_targets.md, docstrings, and a
test-gap mapping of tests/training/composite/*.py + tests/training/test_composite_*.py
vs source modules. All file:line refs verified against current code; DX1, DX2, DX4 and the
package-surface claims (no __all__, 96 underscore re-exports, missing TRANSFORMS_REGISTRY
proxy, dead legacy module paths) were additionally confirmed by live execution.

Known-fixed items from tests/composite_discovery_audit_notes.md (Welford, transform
round-trip, MI LCB gate, tiny-rerank seeding) re-checked only where touched; NOT re-reported.

## Test-gap mapping summary

| Source module | Covering tests | Verdict |
|---|---|---|
| cache.py | test_composite_discovery_cache, test_composite_cache_default_caps, test_composite_cache_row_order, test_caching_discovery*, test_cache_lru_eviction, test_cache_atomic_write_fsync | covered |
| diagnostics.py | test_composite_polish_refinement (all 6 plot helpers) | covered |
| post_shim.py | test_composite_post_shim | covered |
| provenance.py | test_composite_provenance | covered |
| spec.py | pervasive | covered |
| streaming.py | test_composite_streaming_alpha, test_composite_low_polish_composite | covered |
| discovery/_auto_base.py | test_composite_gate_and_edges, test_composite_null_prebinning_parity | covered |
| discovery/_eval.py | test_regression_AP2_finite_mask_threaded + via fit tests | covered |
| discovery/_filter.py | test_composite_discovery (filter_drops) | covered |
| discovery/_fit.py | test_composite_discovery, biz_val suites | covered, EXCEPT iter_transform x multi-base (DX1) |
| discovery/_screening_tiny.py | test_composite_screening_tiny_import, test_composite_screening_split | covered |
| discovery/_stacked.py | test_residual_target_stacked_discovery, test_residual_stacked_suite_integration | covered |
| discovery/_tiny_rerank.py | biz_val_training_composite_discovery | covered |
| discovery/auto_detect.py | test_composite_group_detect, test_composite_time_detect, test_composite_auto_detect_biz_value | covered |
| discovery/bayesian.py | test_composite_bayesian_alpha / _conjugate / _bootstrap | covered |
| discovery/forward_stepwise.py | test_composite_forward_stepwise | covered |
| ensemble/__init__.py + _cross_target.py | test_composite_ensemble* (8 files) | covered |
| ensemble/stacking.py | gate covered; residual_dedup_indices has ZERO tests (DX12) | gap |
| ensemble/feature_stacking.py | test_composite_feature_stacking | covered |
| estimator/_estimator.py | test_monolith_split_w11a, test_composite_medium_composite, sklearn-compliance | covered |
| estimator/_predict.py | polish_refinement, quantile_ensemble_per_column; predict_quantile x unary / x grouped untested (DX2, DX3) | gap |
| estimator/_update.py | test_composite_streaming_update | covered |
| estimator/_utils.py | test_monolith_split_w11a (get_booster etc.) | covered |
| transforms/* | unary/extended/registry-contract/biz_val/interaction_bases tests | covered |

---

## DX1 - P1 - bug - src/mlframe/training/composite/discovery/__init__.py:146-170
**iter_transform crashes on multi-base specs that discovery produces by default**

iter_transform extracts only spec.base_column (one 1-D column, line 162) and calls
transform.forward(y, base, params) (167-169). For linear_residual_multi specs the fitted
params carry K alphas, so forward raises "linear_residual_multi: base has 1 columns but
fitted alphas has 2 entries" (reproduced live). Multi-base specs are produced BY DEFAULT:
multi_base_enabled=True (_composite_target_discovery_config.py:115) auto-promotes kept
linear_residual specs via forward-stepwise and stamps extra_base_columns (_fit.py:736-742).
The method also cannot serve grouped specs (linear_residual_grouped.forward raises "groups
kwarg is required", transforms/linear.py:526-529). export_specs (same file, 236-262) was
already fixed for exactly this consumer class; iter_transform was missed.

Fix: build base from (spec.base_column,) + spec.extra_base_columns (2-D matrix when K>1)
and thread groups for requires_groups transforms (or raise a clear NotImplementedError
naming the spec). Regression test:
tests/training/composite/test_composite_discovery.py::test_iter_transform_multi_base_spec
(fit on a 2-base synthetic with auto-promotion forced; assert finite T); verify it fails pre-fix.

## DX2 - P2 - bug - src/mlframe/training/composite/estimator/_predict.py:259-260
**predict_quantile crashes on unary transforms with a misleading multi-base error**

predict_quantile calls self._extract_base_for_transform(X, base_columns) unconditionally.
For unary transforms (requires_base=False, e.g. cbrt_y) _resolve_base_columns() returns an
empty tuple, and _extract_base_matrix raises "base_columns is empty; multi-base transforms
require at least one base column" (reproduced live: fit+predict succeed, predict_quantile
crashes). _predict_unclipped already has the correct requires_base branch (lines 36-44);
predict_quantile missed it. The error text actively misleads (multi-base wording for a
base-less transform).

Fix: mirror _predict_unclipped - when not transform.requires_base, skip extraction and feed
a zeros placeholder sized to t_raw. Test:
test_composite_unary_transforms.py::test_predict_quantile_unary_cbrt_y_round_trip.

## DX3 - P2 - bug - src/mlframe/training/composite/estimator/_predict.py:302,308,317
**predict_quantile never threads groups kwarg -> grouped transform inverse raises**

All three transform.inverse(...) calls in predict_quantile omit groups, so a
linear_residual_grouped wrapper raises "linear_residual_grouped.inverse: groups kwarg is
required" (transforms/linear.py:541-543) on any predict_quantile call, even though predict
works (threads inverse_kwargs["groups"] at _predict.py:64-72). No test covers
predict_quantile x grouped.

Fix: extract inverse_kwargs (groups) exactly as _predict_unclipped does; pass to each
inverse call. Test: test_composite_linear_residual_grouped.py::test_predict_quantile_grouped.

## DX4 - P2 - bug - src/mlframe/training/composite/estimator/_predict.py:232-281
**predict_quantile silently flips quantile ordering for reciprocal_residual**

The docstring quantile-preservation table (233-242) covers only diff/linear_residual/
logratio/ratio, and the code guards only ratio (264-271) and logratio (275-281). But
reciprocal_residual.inverse is y = 1/(T + 1/base) - strictly DECREASING in T (verified live:
T=0.1 -> y=1.667, T=0.3 -> y=1.25 at base=2). Asking for q90 returns the y-scale q10 and
vice versa: silent wrong results, the exact failure mode the ratio guard exists to prevent.
reciprocal_residual IS in the default discovery transforms list (verified live).

Fix: add monotonicity metadata to Transform (inverse_monotone_in_t: inc|dec|mixed);
for "dec" reverse the quantile mapping (alpha -> 1-alpha) or raise NotImplementedError like
ratio. Pin both directions:
test_composite_polish_refinement.py::test_predict_quantile_reciprocal_residual_ordering.

## DX5 - P2 - docs - src/mlframe/training/composite/__init__.py:20-27,51-56
**Package docstring: shipped features declared "Out of scope ... future PR"; Public-surface list covers ~10% of actual surface**

Lines 51-56 say "Out of scope: Discovery ... future PR. Cross-target ensembling: future PR"
- yet the same file re-exports CompositeTargetDiscovery (line 258) and
CompositeCrossTargetEnsemble (line 180). The "Public surface" section (20-27) lists only
Transform/registry/estimator/exceptions, omitting discovery, ensemble, DiscoveryCache,
CompositeProvenance/report_to_markdown, streaming refit, bayesian alpha, forward stepwise,
feature stacking. help(mlframe.training.composite) actively misinforms.

Fix: rewrite the Public-surface section to enumerate the real exported families; delete the
stale Out-of-scope items (classification residuals remain genuinely out of scope).

## DX6 - P2 - usability - src/mlframe/training/composite/__init__.py:99-258; transforms/__init__.py:295
**No __all__; 96 underscore names re-exported; public read-only TRANSFORMS_REGISTRY NOT exported while private mutable _TRANSFORMS_REGISTRY is**

Verified live: the package exposes 96 single-underscore attributes, has no __all__, exports
the private mutable _TRANSFORMS_REGISTRY dict (line 118) but NOT the MappingProxyType
read-only TRANSFORMS_REGISTRY created for callers at transforms/__init__.py:295 ("Read-only
view exported to callers" - unreachable from the package). Tab-completion / star-import
bury the ~20 public names; the one safe registry handle is hidden.

Fix: add __all__ (public names only; keep underscore re-exports importable for back-compat
but out of __all__); re-export TRANSFORMS_REGISTRY and prefer it in docs.

## DX7 - P2 - docs - transforms/naming.py:4-7, transforms/registry.py:8-10, discovery/__init__.py:438-441, streaming.py:1, estimator/__init__.py:1, ensemble/_cross_target.py:5-6
**Docstrings promise back-compat import paths that no longer exist**

E.g. naming.py: historical "from mlframe.training.composite_transforms import
compose_target_name" claimed to resolve transparently; discovery/__init__.py:440: "New code
should prefer the direct sub-module import (mlframe.training.composite_auto_detect)".
Verified live: find_spec("mlframe.training.composite_transforms") is None; no
composite_transforms / composite_estimator / composite_ensemble / composite_auto_detect
modules or aliases exist. Following these docstrings yields ModuleNotFoundError. Several
also reference the long-gone composite.py / _composite_target_estimator.py filenames.

Fix: one pass over composite/** docstrings replacing legacy module names with real subpackage
paths; drop "resolves transparently" claims.

## DX8 - P2 - docs - README.md:125-145
**README composite section shows only the wrapper; discovery / ensemble / cache / kill-switch invisible, no link to the real docs**

The block documents only CompositeTargetEstimator with manual transform_name/base_column.
Headline features - CompositeTargetDiscoveryConfig(enabled=True) inside
train_mlframe_models_suite, the _CT_ENSEMBLE__<target> cross-target ensemble, DiscoveryCache,
BaselineDiagnostics-driven recommendation, MLFRAME_DISABLE_COMPOSITE kill switch - are
unmentioned, and there is no link to docs/examples/composite_targets.md (3-tier recipes) or
docs/composite_targets_tutorial.ipynb. README users will hand-wire wrappers that discovery
automates.

Fix: add 5-8 lines + a Tier-1 snippet (config opt-in + ensemble entry key) and links to both
docs; show the list_transforms() import path.

## DX9 - P2 - docs - docs/examples/composite_targets.md:67,92-94
**Tier-1 doc claims the default tries "all four core transforms"; actual default is a 24-transform list**

"Inside the defaults: ... transforms=[diff, ratio, logratio, linear_residual]" and "all four
core transforms are tried". Verified live: CompositeTargetDiscoveryConfig().transforms has
24 entries (core + additive/median residual, robust, quantile/monotonic residual,
y_quantile_clip, 4 unary, 4 chains, 5 Pack-L incl. reciprocal_residual). Runtime, log
volume, and which specs can win differ materially from the doc promise.

Fix: state the real default set (or point at list_transforms()); keep the 4-core example as
an explicit-override snippet.

## DX10 - P2 - docs - README.md:68
**Corrupted command in install instructions: "pre-commit installgress" (stray Cyrillic tail)**

Line 68 reads "pre-commit install" + a glued Cyrillic fragment in the dev-checkout code
block; copy-paste fails.

Fix: restore plain "pre-commit install".

## DX11 - P2 - docs - src/mlframe/training/composite/estimator/_estimator.py:59-111,406
**Class docstring documents 9 of 13 constructor params; fit has no docstring**

Parameters section omits runtime_stats_callback, auto_variance_stabilise, base_columns,
group_column - documented only as inline __init__ comments (132-175), invisible to help() /
IDE / sphinx. base_columns-vs-base_column precedence and the grouped-transform requirement
are construction-time decisions. fit (line 406) has no docstring: the sample_weight
pass-through and fit_kwargs-to-inner contract are undocumented.

Fix: add the 4 missing Parameters entries (hoist the inline comments); add a short fit
docstring (sample_weight, fit_kwargs, domain-row dropping, returns self).

## DX12 - P2 - test-gap - src/mlframe/training/composite/ensemble/stacking.py:84-137
**residual_dedup_indices is production-wired but has zero tests**

Used by the CT-ensemble build at
src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py:611-617 to drop
near-duplicate ensemble members, but a grep over tests/ returns nothing (only a bench
references it). The greedy keep-best-first logic, min_keep floor (113-114, 133), and
NaN-row guard (115-117) can silently regress.

Fix: tests/training/composite/test_composite_residual_dedup.py with
test_dedup_keeps_lower_rmse_of_redundant_pair, test_min_keep_floor_never_violated,
test_nonfinite_rows_keep_all, test_independent_members_all_kept.

## DX13 - LOW - docs - src/mlframe/training/composite/__init__.py:46-48; estimator/__init__.py:44-49,54-80
**y-clip bounds documented as multiplicative "[Q001/10, Q999*10]" but implemented as span extension**

Three docstrings describe the post-inverse clip as quantile multipliers; the implementation
(_y_train_clip_bounds, estimator/__init__.py:78-79) is low = q_low - 0.9*span,
high = q_high + 9*span. For q_low=10, q_high=100 the documented low bound is 1; the real one
is -71. Operators debugging clip hits from runtime_stats_ will compute wrong expected bounds.

Fix: document the span formula (and why span-based beats multiplicative for negative /
zero-crossing targets) in all three places.

## DX14 - LOW - docs - src/mlframe/training/composite/transforms/__init__.py:1,96
**Module docstring says "11 transforms"; registry has 32. TAG_EXTENDED comment stale**

Registry (transforms/registry.py:194-675) holds 32 entries (15 bivariate, 4 unary, 5 chains,
8 Pack-L). Line 96 TAG_EXTENDED comment "placeholder; future presets may add more" is stale -
the tag is load-bearing on 20+ transforms and is the list_transforms(tags=...) filter axis.

Fix: update the count (or defer to list_transforms() to avoid future drift); rewrite the
TAG_EXTENDED comment.

## DX15 - LOW - usability - src/mlframe/training/composite/estimator/_estimator.py:692-712
**update / get_buffer_state / predict_pre_clip and 5 delegated properties still runtime-bound - invisible to mypy / IDEs**

The fix that gave predict / predict_quantile in-body delegating stubs (lines 177-195,
motivated by "invisible to mypy / IDEs / help()") was not extended to the rest: update,
get_buffer_state, predict_pre_clip, get_booster and the feature_importances_ / coef_ /
intercept_ / booster_ / n_features_in_ properties are assigned at module bottom. Static
checking of est.update(...) fails; help() omits them.

Fix: same stub pattern for the public methods; class-body property wrappers for the
delegated attributes.

## DX16 - LOW - docs - src/mlframe/training/composite/__init__.py:59-72
**Dead monolith-era imports at package top**

contextlib, math, re, warnings, dataclass, field, timer, typing names, numpy, pandas,
BaseEstimator, RegressorMixin, clone are imported and never used (file is pure re-exports
below). Dead weight from the monolith split.

Fix: delete unused imports (keep logging).

## DX17 - LOW - usability - src/mlframe/training/composite/discovery/_eval.py:277; transforms/registry.py:388-391
**Unary spec names embed a meaningless base segment, contradicting the registry comment**

Registry comment: "Composite-target name has no base segment (e.g. y-cbrtY) since there is
no base." Actual: _eval.py:277 always calls compose_target_name(target_col, transform_name,
base), so unary specs are named y-cbrtY-<base loop var> and spec.base_column points at a
column the transform ignores. Confusing in reports/provenance (implies dependence that does
not exist); the name varies with whichever base the unary happened to be evaluated under.

Fix: when not transform.requires_base, compose the name without the base segment and set
base_column="" on the spec; or update the registry comment if current behaviour is kept.

## DX18 - LOW - test-gap - tests/test_docs_examples_smoke.py:24-25
**Docs smoke test pins README symbols only; composite_targets.md / tutorial imports unpinned**

test_readme_composite_estimator_symbol checks just CompositeTargetEstimator, while
docs/examples/composite_targets.md exercises mlframe.training.configs.
CompositeTargetDiscoveryConfig, mlframe.training.composite CompositeSpec and
report_to_markdown, and the MLFRAME_DISABLE_COMPOSITE contract (_phase_config_setup.py:238).
A rename breaks the doc silently - the rot this test exists to catch (its own docstring even
claims it covers the composite tutorial).

Fix: add test_docs_composite_examples_symbols asserting those imports + that the Tier-2
config kwargs in the doc construct a valid CompositeTargetDiscoveryConfig.

## DX19 - LOW - docs/extension - src/mlframe/training/_composite_target_discovery_config.py:17
**94-field config with no user-facing knob reference; dict-config acceptance undocumented**

CompositeTargetDiscoveryConfig has 94 fields (verified live); docs/examples cover ~12. The
rich inline comments are unreachable without opening source. CompositeTargetDiscovery also
accepts a plain dict config (discovery/__init__.py:130-133) - documented nowhere.

Fix: generate docs/composite_targets_config_reference.md (field, type, default, first
comment line) from model_fields via a small scripts/ generator run as a doc-drift test;
mention dict-config in the class docstring.

## DX20 - LOW - usability - src/mlframe/training/composite/estimator/__init__.py:87,114-117
**_extract_base docstring + TypeError advertise structured-ndarray support that does not exist**

Docstring: "Pull base values from X (pandas / polars / structured ndarray)"; the raise says
"pass pandas / polars DataFrame or a structured ndarray with named columns" - but there is
no ndarray branch; a structured ndarray hits the same TypeError that tells you to pass one.
_extract_base_matrix:189-191 routes unknown types "per-column for ndarray-with-names etc."
into the same dead end.

Fix: implement the structured-ndarray branch (X[base_column] when x.dtype.names) or drop the
claim from docstring and message.

## DX21 - LOW - usability - src/mlframe/training/composite/__init__.py:258
**CompositeTargetDiscoveryConfig not importable from the composite package**

The package exports CompositeTargetDiscovery but its required config lives in
mlframe.training.configs - every example needs two import locations; the config is invisible
to dir(mlframe.training.composite).

Fix: re-export CompositeTargetDiscoveryConfig from mlframe.training.composite (lazy at the
bottom, mirroring the dict-config lazy import already inside CompositeTargetDiscovery.__init__).

## DX22 - LOW - test-gap - tests/training/ (layout)
**Composite tests split across two locations (38 in tests/training/composite/, 28 test_composite_* at tests/training/)**

Both layers actively receive files (test_composite_streaming_update.py,
test_composite_multi_base_integration.py sit at top level although composite/ exists).
Running pytest tests/training/composite silently runs only ~60% of the composite suite -
easy to under-test a change locally.

Fix: git mv the top-level test_composite_*.py into tests/training/composite/ (no code
edits), or document that the canonical selector is pytest -k composite.
