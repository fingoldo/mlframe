# core_class — mrmr_audit_2026-09-14

## Scope

| File | LOC |
|---|---|
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py` | 4285 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_fit_helpers.py` | 605 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_config.py` | 532 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_setstate_defaults.py` | 397 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_config_dataclasses.py` | 328 |
| `src/mlframe/feature_selection/filters/_mrmr_partial_fit.py` | 299 |
| `src/mlframe/feature_selection/filters/mrmr/__init__.py` | 262 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_transform.py` | 251 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_param_constants.py` | 99 |
| `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_shared.py` | 27 |

Cross-read for evidence only (other clusters own them): `filters/_mrmr_validate_transform.py`,
`filters/_mrmr_fit_impl/_helpers.py`, `tests/test_meta/`.

---

## Findings

### CORE-1 — `_stability_outer_fit` silently DISCARDS `sample_weight` and `groups`  [P1]
**Where:** `mrmr/_mrmr_class.py:3532-3552` (call) and `mrmr/_mrmr_class_fit_helpers.py:218-300` (callee)
**What:** `_fit_body` forwards both to the stability outer loop:
```python
_stab_result = self._stability_outer_fit(X, y, groups=groups, sample_weight=sample_weight, **fit_params)
```
`_stability_outer_fit(self, X, y, **fit_kwargs)` never references `fit_kwargs` again anywhere in its body
(lines 218-300 — the name appears exactly once, in the signature). `_inner_selector` (line 254) calls
`sub.fit(X_sub_df, y_sub_s)` with no weights, and `_sub_base_params` (241-252) is built from `get_params()`,
which by definition contains no fit-time arguments.
**Why it is wrong / costly:** `MRMR(stability_selection_method="cluster").fit(X, y, sample_weight=w)` returns a
selection computed on UNWEIGHTED data, with no warning at any level. The classic path honours weights via
`_maybe_resample_for_sample_weight` (`_mrmr_class.py:3661-3662`), so the same call with
`stability_selection_method="classic"` gives a different, weight-aware answer. Bites hardest on the exact
workloads weights exist for: class-imbalance reweighting and importance-sampled panels, where the unweighted
resample can rank a majority-class-driven feature first. `groups` is likewise dropped, so the
`strict_groups`/`group_aware_mi` contract (`GroupAwareConfig`, `_check_groups_contract` at
`_mrmr_class_fit_helpers.py:126-141`) is not enforced on this path and `groups_ignored_` is never even set.
**Fix:** Accept `sample_weight=None, groups=None` explicitly; resample once up front via
`self._maybe_resample_for_sample_weight(X, y, sample_weight)` before the bootstrap loop (the bootstrap then
draws from the weight-corrected empirical distribution, which is the correct composition), and either forward
`groups` into `_inner_selector`'s `sub.fit` or raise `NotImplementedError` naming the unsupported combination.
Never silently drop. Also call `self._check_groups_contract(groups)` so `groups_ignored_` is set.
**Test:** `test_stability_selection_honours_sample_weight` — fit the same (X, y) twice under
`stability_selection_method="cluster"`, once with `sample_weight` that up-weights a subpopulation in which a
different feature is dominant, once without; assert `support_` differs. Today both are identical, which is the
regression signature.

### CORE-2 — `_stability_outer_fit` leaves the fitted-attribute surface incomplete; `feature_names_in_` can be non-string  [P1]
**Where:** `mrmr/_mrmr_class_fit_helpers.py:233-235, 293-300`
**What:** The path sets only six attributes (`support_`, `feature_names_in_`, `n_features_in_`, `n_features_`,
`stability_freq_`, `stability_info_`). The two sibling terminal paths set a far larger roster:
`_fit_identity_shortcut` sets 15 including `_feature_names_in_synthesized_`, `fallback_used_`, `mrmr_gains_`,
`dcd_`, `cluster_members_`, `friend_graph_`, `cluster_aggregate_`, `ran_out_of_time_`
(`_mrmr_class_fit_helpers.py:461-506`); `_fit_multioutput` sets `_engineered_features_`,
`_engineered_recipes_`, `_fit_sample_weight_`, `degenerate_columns_`, `provenance_`
(`_mrmr_class_fit_helpers.py:563-595`).
Separately, line 233 is `X_df = X if hasattr(X, "iloc") else pd.DataFrame(np.asarray(X))` — for ndarray input
pandas assigns INTEGER columns `0..p-1`, so `feature_names = list(X_df.columns)` is a list of ints and line 295
stores `feature_names_in_` as an object array of ints.
**Why it is wrong / costly:** two concrete consequences.
(a) `_feature_names_in_synthesized_` is never set, so `get_feature_names_out(input_features=...)` falls into the
back-compat heuristic at `_mrmr_class_transform.py:76-80`, which is exactly the brittle path the sentinel was
introduced to retire (its own comment documents the misclassification of user columns literally named
`feature_<n>`). A stability-fit estimator therefore re-acquires the retired bug.
(b) With ndarray input the saved names are ints; `_placeholder.match(str(n))` on `"0"` fails
(`^(?:f|feature_)\d+$`), so `synthesized` resolves False and any `input_features` the caller passes raises
`ValueError` — whereas the identical ndarray fit under `stability_selection_method="classic"` accepts it.
sklearn also requires `feature_names_in_` to be all-string.
**Fix:** Factor the fitted-attribute roster the three terminal paths share into one
`_seed_default_fitted_attrs(self, X, feature_names)` helper called by all three; in it set
`self._feature_names_in_synthesized_ = not hasattr(X, "columns")` and store
`np.asarray([str(n) for n in feature_names], dtype=object)`.
**Test:** `test_stability_fit_sets_the_same_fitted_attr_roster_as_classic` — fit the same data both ways and
assert `set(a for a in vars(stab) if a.endswith("_")) >= set(a for a in vars(classic) if a.endswith("_"))`,
plus `test_stability_fit_feature_names_in_are_strings` asserting
`all(isinstance(n, str) for n in est.feature_names_in_)` after an ndarray fit.

### CORE-3 — `fit()` never resets the `partial_fit` streaming buffer  [P2]
**Where:** `filters/_mrmr_partial_fit.py:214, 223-227, 234-246`; `mrmr/_mrmr_class.py` (no `_partial_fit_`
assignment anywhere — `grep -n "_partial_fit" _mrmr_class.py` returns only docstring hits)
**What:** "first call" is detected purely as `getattr(self, "_partial_fit_X_buffer_", None) is None`
(line 214). Nothing in `fit`/`_fit_body` clears `_partial_fit_X_buffer_` / `_partial_fit_y_buffer_` /
`_partial_fit_batch_sizes_` / `_partial_fit_n_seen_`.
**Why it is wrong / costly:** the sequence `est.partial_fit(A, ya); est.fit(B, yb); est.partial_fit(C, yc)`
refits on `A + C` (the stale buffer plus the new batch) — `B` is absent and `A`, which the explicit `fit(B)` was
a clear instruction to forget, is silently back in the training set. The decay weights are also computed against
a `batch_sizes` registry that no longer describes the data the user believes is buffered. Bites any
retrain-then-resume-streaming service loop, and there is no observable symptom other than a wrong `support_`.
**Fix:** At the top of `_fit_body`, when the call did not originate from `partial_fit`, reset the four
`_partial_fit_*` attributes. Simplest safe wiring: have `partial_fit` set a private `self._in_partial_fit = True`
around its `self.fit(...)` call and have `_fit_body` clear the buffers when that flag is absent.
**Test:** `test_fit_resets_partial_fit_buffer` — `partial_fit(A); fit(B); partial_fit(C)` then assert
`len(est._partial_fit_X_buffer_) == len(B) + len(C)` (today it is `len(A) + len(C)`).

### CORE-4 — `_fit_sample_weight_` is write-only on MRMR but pickled into every saved model  [P2]
**Where:** `mrmr/_mrmr_class.py:3661-3662`, `mrmr/_mrmr_class_fit_helpers.py:582`; `__getstate__` at
`mrmr/_mrmr_class.py:3253-3269`
**What:** `self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)`
is assigned and then consumed on the very next line as the argument to `_maybe_resample_for_sample_weight`.
A repo-wide grep (`src` + `tests`) finds no other MRMR read: the only other hits are
`wrappers/rfecv/_fit_fold.py:206-215` and `wrappers/rfecv/_fit_init.py:98`, which belong to the unrelated
`RFECV` class. `__getstate__` copies `self.__dict__` and pops only `_fit_reentrancy_lock_`, so the array is
persisted.
**Why it is wrong / costly:** an n-length float64 array rides in every pickle for zero benefit. At the 2M-row
scale this repo routinely profiles that is 16 MB per saved model; the CLAUDE.md 100 GB-frame regime makes it
~800 MB at 100M rows. Under a joblib fan-out the array is also re-serialised to every worker.
**Fix:** Use a local variable instead of an instance attribute at both sites. If some external consumer is
believed to read it (none found — say so explicitly if one turns up), keep the attribute but pop it in
`__getstate__` alongside `_fit_reentrancy_lock_`.
**Test:** `test_fitted_mrmr_pickle_excludes_row_length_arrays` — fit with `sample_weight` on n=50_000 and assert
no value in `est.__getstate__()` is an ndarray of length `n_rows`.

### CORE-5 — `__getstate__` persists `_stability_replay_state_`, an unbounded (n_sub × n_cand) int32 matrix  [P2]
**Where:** `mrmr/_mrmr_class.py:3253-3269` (the `__getstate__` that pops only the lock);
state built at `filters/_mrmr_fit_impl/_helpers.py:333-339`, read at `filters/_mrmr_stability_report.py:99`
**What:** `_stability_replay_state_` is a dict of `cand_codes` (int32 screening-code matrix over the screen
subsample), `y_codes`, `selected_mask`, plus a per-recipe `recipe_replay` dict each carrying up to three more
int32 row-length vectors (`_helpers.py:326-331`). It is a pure REPLAY CACHE for the optional
`selection_stability_report()` accessor — nothing in `transform` or the sklearn protocol reads it.
**Why it is wrong / costly:** exactly this cluster's documented bug class ("a runtime cache attached to the
instance is not excluded from `__getstate__`"), in its size rather than its picklability form. At the screen
default (`_fast_search_default_subsample_n` fallback 90_000, `_mrmr_class_config.py:80-90`) and a 2000-candidate
pool, `cand_codes` alone is 90_000 × 2000 × 4 B ≈ 720 MB added to every saved model and shipped to every joblib
worker, silently. The consumer already tolerates its absence (`getattr(self, "_stability_replay_state_", None)`
at `_mrmr_stability_report.py:99`).
**Fix:** Add a constructor knob (default the memory-safe direction) or a byte-size gate in `__getstate__`:
drop `_stability_replay_state_` from the pickled state when its arrays exceed a threshold, logging at
`warning` which accessor is thereby disabled on reload. Never drop it silently — per this repo's
"substituted value must not disable the check it feeds" rule, the report must then say "replay state was not
persisted", not return an empty report.
**Test:** `test_pickle_size_does_not_scale_with_screen_subsample` — fit at n=20_000 and n=80_000 with the same
column count and assert `len(pickle.dumps(est))` does not grow proportionally to n.

### CORE-6 — the empirically-best scorer is documented but not the default  [P2]
**Where:** `mrmr/_mrmr_class.py:2423` (`fe_hybrid_orth_default_scorer: str = "plug_in"`) vs
`mrmr/_mrmr_class_config.py:216-243` (`recommend_default_scorer`) and
`mrmr/_mrmr_config_dataclasses.py:127` (`default_scorer: str = "plug_in"`)
**What:** `recommend_default_scorer()`'s own docstring states CMIM took "5/7 dataset wins on top-AUC of the
downstream LogReg over the marginal-MI baseline, including all three high-redundancy fixtures", that
"the plug-in default is last on every redundant fixture", and that a later acceleration pass left the
leaderboard unchanged. It returns `"cmim"`. The shipped default is `"plug_in"`, justified at
`_mrmr_param_constants.py` as `"plug_in",  # Layer 21 (default)` — i.e. preserved for byte-for-byte legacy
equivalence.
**Why it is wrong / costly:** CLAUDE.md "Accuracy/performance over legacy/compat" is explicit — "Default knobs
flip to the new path once it measurably wins", "most-accurate-on-the-honest-metric first". A method whose sole
purpose is to tell callers the right value, which no default reads, is the "already-optimized primitive, just
not wired in" pattern this repo has repeatedly logged. Every caller who does not know to call
`recommend_default_scorer()` gets the measured-worst scorer on redundant pools.
**Fix:** Either flip both defaults to `"cmim"` (ctor line 2423 and `HybridOrthScorersConfig.default_scorer`,
which `tests/test_meta/test_config_dataclass_defaults_match_ctor.py` will force to stay in lockstep), keeping an
explicit opt-out, or — if the L83 bake-off is not considered a wide enough benchmark by current standards —
record a `# bench-attempt-rejected` note with the numbers, per REJECTED ≠ DELETED. Do not leave it undecided.
**Test:** `test_default_scorer_matches_recommend_default_scorer` — asserts
`MRMR()._ctor_defaults()["fe_hybrid_orth_default_scorer"] == MRMR.recommend_default_scorer()`, so the two can
never drift again.

### CORE-7 — `HybridOrthConfig` cannot set `fe_hybrid_orth_elasticnet_l1_ratio`  [P2]
**Where:** `mrmr/_mrmr_config_dataclasses.py:163-166` vs `mrmr/_mrmr_class.py:2110-2112`
**What:** The ctor exposes three elasticnet knobs — `fe_hybrid_orth_elasticnet_enable`,
`..._alpha`, `..._l1_ratio` (lines 2110, 2111, 2112). `HybridOrthConfig` declares only `elasticnet_enable` and
`elasticnet_alpha`. Enumerating every `fe_hybrid_orth_*` ctor parameter against
`_HYBRID_ORTH_FIELD_MAP ∪ _HYBRID_ORTH_SCORERS_FIELD_MAP`, `l1_ratio` is the single unmapped one.
**Why it is wrong / costly:** `l1_ratio` is the parameter that decides whether elastic net behaves as lasso or
as ridge — the one knob a caller configuring elastic net most needs. Because `_MRMRSubConfig` sets
`extra="forbid"`, `HybridOrthConfig(elasticnet_l1_ratio=0.9)` raises a `ValidationError` that reads like a typo
rather than a missing feature, so the config user is pushed back to the flat kwarg and gets a config/flat mix —
which `apply_mrmr_config_objects` resolves config-last, the ordering the module docstring itself calls "a caller
error this module does not attempt to reconcile" (`_mrmr_config_dataclasses.py:14-16`).
**Fix:** Add `elasticnet_l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0)` to `HybridOrthConfig`.
**Test:** `test_every_fe_hybrid_orth_flat_param_has_a_config_field` — a meta-test asserting
`{p for p in MRMR._ctor_defaults() if p.startswith("fe_hybrid_orth_")} == set(_HYBRID_ORTH_FIELD_MAP.values()) | set(_HYBRID_ORTH_SCORERS_FIELD_MAP.values())`.
The existing `tests/test_meta/test_config_dataclass_defaults_match_ctor.py` gates the VALUES of mapped fields
but has no coverage gate, which is why this gap is live.

### CORE-8 — the config layer's "typo raises at construction" promise does not hold for 8 string fields  [P2]
**Where:** `mrmr/_mrmr_config_dataclasses.py:6-7` (the claim), fields at lines 85, 90, 112, 123, 126, 127, 135,
183; validator at `filters/_mrmr_validate_transform.py:30-150`
**What:** The module docstring promises "construction-time validation (a typo'd enum value raises immediately,
not minutes into a fit()) instead of MRMR's ad hoc `_validate_string_params` late-validation pass". These fields
are typed bare `str`, not `Literal`:

| field | line | late-validated at fit? |
|---|---|---|
| `DCDConfig.dcd_distance` | 85 | yes — `_mrmr_validate_transform.py:47` |
| `DCDConfig.dcd_swap_method` | 90 | yes — `:48` |
| `HybridOrthScorersConfig.default_scorer` | 127 | yes — `:129-142` |
| `HybridOrthScorersConfig.hsic_kernel` | 112 | **no** |
| `HybridOrthScorersConfig.ensemble_aggregator` | 123 | **no** |
| `HybridOrthScorersConfig.ensemble_scorers` (elements) | 124 | **no** |
| `HybridOrthScorersConfig.meta_force_scorer` | 126 | **no** |
| `HybridOrthConfig.basis` | 135 | **no** |
| `HybridOrthConfig.cluster_basis_aggregator` | 183 | **no** |

The three "yes" rows still miss the advertised construction-time raise — the error surfaces at `fit`, which is
the behaviour the config layer exists to replace. The six "no" rows are never validated at all; allow-lists
already exist for three of them in `_mrmr_param_constants.py`
(`_VALID_DCD_DISTANCES`, `_VALID_DCD_SWAP_METHODS`, `_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS`).
**Why it is wrong / costly:** a typo'd `hsic_kernel="rbg"` or `ensemble_aggregator="borda"` reaches its consumer
(`_mrmr_fit_impl/_hybrid_orth_family_variants/_group3.py:223-251`,
`_group4.py:215`) via `getattr(self, ..., <default>)` and is passed straight through — whether the consumer
raises or silently falls back is that cluster's question, but at THIS surface the value is accepted unchecked,
which is precisely the "typo silently degrades behaviour instead of raising" class the file's own base-class
comment (`_mrmr_config_dataclasses.py:26-27`) claims to have fixed.
**Fix:** Retype each of the eight as `Literal[...]` sourced from the corresponding `_VALID_*` tuple (the
constants module is a leaf with no class refs, so importing it here creates no cycle); add `_VALID_*` tuples for
`hsic_kernel`, `ensemble_aggregator`, `cluster_basis_aggregator` and `basis`, and extend
`_validate_string_params`'s `_checks` table with them so the FLAT kwarg path is covered too. Either fix the
docstring's promise or make it true; do not leave both.
**Test:** `test_subconfig_string_fields_reject_typos` — parametrised over the eight fields, asserts
`pytest.raises(ValidationError)` on a deliberate misspelling; plus
`test_validate_string_params_covers_every_enumerated_ctor_param` asserting every `_VALID_*` constant in
`_mrmr_param_constants.py` is referenced by `_validate_string_params`.

### CORE-9 — `partial_fit` cannot recover from a below-threshold batch after a truncating window  [P2]
**Where:** `filters/_mrmr_partial_fit.py:249-263`
**What:** The rolling window is applied (249-250) and the counters updated (255-256) BEFORE the
`if self._partial_fit_n_since_refit_ < min_recompute: return self` early exit (258-263). When
`partial_fit_window` is smaller than `partial_fit_min_recompute`, every call appends, truncates back down to
`window` rows, increments `_partial_fit_n_since_refit_` — and the refit eventually fires against a buffer that
by construction can never hold more than `window` rows.
**Why it is wrong / costly:** the two knobs are independent and nothing validates their relationship (the only
validation is `window > 0` at 210-211). With `partial_fit_window=50, partial_fit_min_recompute=1000` the user
believes they refit on 1000 rows of history; they refit on 50. The decay weights are then computed over a
`batch_sizes` registry whose oldest entries were trimmed to slivers, so the effective recency profile is also
not the documented `(1 - decay) ** k`.
**Fix:** Validate at the top of `partial_fit`: when `window is not None and window < min_recompute`, raise a
`ValueError` naming both values, or clamp `min_recompute` to `window` with a `logger.warning`. Document the
relationship in the module docstring's DECAY SEMANTICS section.
**Test:** `test_partial_fit_rejects_window_smaller_than_min_recompute` — asserts the raise, and
`test_partial_fit_refit_buffer_length_matches_window` asserting the buffer handed to `fit` is exactly `window`
rows when truncation is active.

### CORE-10 — `fit` mutates constructor parameters; only a `finally` restore keeps the sklearn contract  [P3]
**Where:** `mrmr/_mrmr_class_config.py:133-214` (`_override_if_at_default`, `_apply_default_screen_subsample`,
`_apply_fast_search_profile`), restore driven from `mrmr/_mrmr_class.py:4147-4245`
**What:** `setattr(self, attr, new_value)` on real constructor parameters (`fe_check_pairs_subsample_n`,
`fe_smart_polynom_subsample_n`, every entry of `_FAST_SEARCH_OVERRIDES`) for the duration of a fit, with the
pre-fit values stashed in a `saved` dict restored in a `finally`.
**Why it is wrong / costly:** sklearn forbids `fit` mutating constructor params. The `finally` restore makes it
externally invisible for the normal case, and `_restore_toggles_snapshot_and_raise`
(`_mrmr_class.py:3790`) covers the raising case — so this is a contract nit, not a live bug. It does become
observable to anything that reads the estimator DURING a fit: a `get_params()` from a logging/monitoring hook,
or an `MRMR` nested inside a `Pipeline` whose outer `clone()` races the inner fit. `_safe_restore`
(`_mrmr_class.py:171`) is the mitigation's own acknowledgement that the restore can itself fail.
**Fix:** Carry the profile overrides in a per-fit resolved-config object (a small frozen dataclass built at the
top of `_fit_body` and threaded down) rather than on `self`. This also retires `_safe_restore`, the
`saved`-dict plumbing and the `__env__`/`__fdcap__` sentinel keys (`_mrmr_class_config.py:204, 212`).
**Test:** `test_fit_does_not_mutate_ctor_params_observed_concurrently` — a fit on a background thread with a
main-thread poll of `get_params()`, asserting the polled dict always equals the pre-fit one.

### CORE-11 — `__setstate__`'s ctor-default backfill does NOT cover fitted attributes  [P3]
**Where:** `mrmr/_mrmr_class.py:3296-3333`; roster `mrmr/_mrmr_setstate_defaults.py:17-387`
**What:** Two backfill passes run. The first seeds the hand-maintained roster (3310-3311); the second
(3328-3332) covers "every remaining ctor default the roster did not cover" — sourced from `_ctor_defaults()`,
so it is complete for CONSTRUCTOR params only. Trailing-underscore FITTED attributes are covered only by the
hand-maintained roster (`hybrid_orth_features_`, `wavelet_features_`, `fe_provenance_`, … ).
**Why it is wrong / costly:** the brief asks whether `__setstate__` supplies defaults for every attribute added
since the last release. For ctor params: yes, structurally guaranteed. For fitted attributes: no — a new fitted
attribute added without a roster entry is simply absent from an old pickle. The current code base survives this
because every consumer I checked reads them via `getattr(..., default)` (`_mrmr_class_transform.py:104-105,
125, 138, 152, 214-215`; `_mrmr_stability_report.py:99`), and the roster comment at
`_mrmr_setstate_defaults.py:252-255` records a real past instance where one buffer was omitted and
silently collapsed multi-batch history — so the hazard is demonstrated, not hypothetical. No live instance
found today.
**Fix:** Add a meta-test rather than more hand-maintenance: AST-collect every `self.<name>_ = ` assignment
across the `mrmr` package and assert each name is either in `_SETSTATE_LEGACY_DEFAULTS` or read exclusively via
`getattr` with a default. Grandfather the current set into a baseline.
**Test:** `test_every_fitted_attr_is_setstate_safe` — as described; fails on the next fitted attribute added
without either a roster entry or a defaulted read.

### CORE-12 — `build_setstate_defaults()` deep-copies ~250 keys on every unpickle  [P3]
**Where:** `mrmr/_mrmr_setstate_defaults.py:390-397`, called at `mrmr/_mrmr_class.py:3296`
**What:** `copy.deepcopy(_SETSTATE_LEGACY_DEFAULTS)` on every `__setstate__`. The roster's values are
overwhelmingly immutable scalars/tuples; only about a dozen are mutable (`[]` literals).
**Why it is wrong / costly:** a deep copy walks every key. The sibling caches in this cluster
(`_CTOR_DEFAULTS_CACHE`, `_FRESH_INSTANCE_DEFAULTS_CACHE`) exist precisely because per-unpickle work was found
to matter under joblib fan-out, where each worker unpickles the estimator. I have NOT measured this one and make
no speedup claim. Grepped for `@njit` / `parallel=True` / `prange` / `cuda.jit` / `cupy` /
`KernelTuningCache` across the whole cluster: none present, and none would be applicable — this is pure Python
dict/attr plumbing, so the ladder does not apply.
**Fix:** `dict(_SETSTATE_LEGACY_DEFAULTS)` plus `copy.deepcopy` only on values failing an
`isinstance(v, (list, dict, set))` check — the same predicate `__setstate__` already uses two passes later
(3309, 3332).
**Bench plan (do not skip):** `timeit` `pickle.loads` of one fitted MRMR, best-of-30 warm, before/after, in a
separate process; report both the isolated `build_setstate_defaults()` time and the end-to-end
`pickle.loads` wall. Reject if the e2e delta is inside noise, and record the numbers per REJECTED ≠ DELETED.
**Test:** `test_setstate_defaults_are_not_aliased_across_instances` — unpickle two instances from the same bytes,
append to one's `_engineered_features_`, assert the other is unaffected (pins the invariant any copy-narrowing
must preserve).

### CORE-13 — module-level caches mutated without a lock (already baselined)  [P3]
**Where:** `mrmr/_mrmr_class_config.py:30-34` (five dicts), written at 88-105, 114-131, 375-380, 389-393,
417-426
**What:** Five `dict[type, ...]` process-lifetime caches, read-then-write with no lock. Confirmed present in the
repo's own gate baseline: `tests/test_meta/_unlocked_module_cache_baseline.json` lists
`feature_selection/filters/mrmr/_mrmr_class_config.py:30` through `:34`.
**Why it is (mostly) fine:** every cached value is class/host-constant and idempotent, so a race produces a
duplicate computation, never a wrong value. The one with real cost is `_FRESH_INSTANCE_DEFAULTS_CACHE`, whose
miss path constructs a full ~300-parameter `MRMR()` (line 420) — N concurrently-unpickling threads each pay it.
Reported per the brief's "report every finding" rule; the baseline entry means it is known, not that it is
invisible.
**Fix:** If touched for any other reason, guard the five with one module-level `threading.Lock` (double-checked:
read outside, populate inside) and de-baseline the five lines.
**Test:** `test_fresh_instance_defaults_cache_populates_once_under_threads` — patch `MRMR.__init__` with a
counting wrapper, clear the cache, unpickle from 8 threads, assert the counter is 1.

### CORE-14 — `_engineered_names_cache_` is an instance memo that is pickled  [P3]
**Where:** `mrmr/_mrmr_class_transform.py:104-114`
**What:** `self._engineered_names_cache_ = (_current_recipes, engineered_names)` — a memo keyed by the object
IDENTITY of `_engineered_recipes_`, set from `get_feature_names_out`, and not popped by `__getstate__`.
**Why it is a finding, and why it is only P3:** it is the cluster's bug-class shape (runtime memo not excluded
from `__getstate__`). It happens to be benign: the tuple's first element IS `self._engineered_recipes_`, and
pickle's memo preserves that identity within a single pickle, so the cache remains valid after a round-trip and
adds only the name list to the payload. It would stop being benign the moment the key becomes something
non-picklable or the recipes list starts being mutated in place (the comment at 98-103 asserts it never is;
nothing enforces that).
**Fix:** Pop `_engineered_names_cache_` in `__getstate__` next to `_fit_reentrancy_lock_`; the memo rebuilds on
first use at a cost of one list comprehension.
**Test:** `test_get_feature_names_out_cache_survives_pickle_round_trip` — assert equal names before and after a
round-trip AND that `__getstate__()` contains no `_engineered_names_cache_` key.

### CORE-15 — `__repr__` textually patches sklearn's output on an undocumented format assumption  [P3]
**Where:** `mrmr/_mrmr_class.py:3172-3191`
**What:** `super().__repr__()` is sliced on `r.endswith(")")` to splice in `n_workers=`. The code's own comment
(3178-3182) concedes this is "an untested assumption about its trailing format … rather than a documented public
contract".
**Why it is wrong / costly:** the `N_CHAR_MAX` truncation path in `BaseEstimator.__repr__` emits `...` inside the
parens; the result still ends in `)`, so the splice produces a repr whose parameter list is elided but which
nonetheless advertises an exact `n_workers=` — mildly misleading rather than broken. The `except Exception ->
logger.debug` (3189-3190) is acceptable here: a repr annotation is genuinely cosmetic and the fallback value
(the plain repr) is correct, not a non-neutral substitute.
**Fix:** Override `_get_param_names`/`_changed_params` semantics instead, or simply document `n_workers` in the
class docstring and drop the splice. Alternatively skip the annotation when `"..." in r`.
**Test:** `test_repr_annotation_skipped_when_truncated` — `repr(MRMR())` under a small `N_CHAR_MAX`, assert the
output does not claim an `n_workers=` value alongside an elided parameter list.

### CORE-16 — `partial_fit`'s `and is_first is False` is dead  [P3]
**Where:** `filters/_mrmr_partial_fit.py:284`
**What:** `if sw_new.shape[0] != len(X_df) and is_first is False:` — control flow returned at line 231 for the
`is_first` case, so `is_first` is unconditionally `False` here.
**Why it is wrong / costly:** dead condition; a reader must re-derive the control flow to conclude the
length check is unconditional. No behavioural effect.
**Fix:** Drop the `and is_first is False` clause.
**Test:** covered by CORE-9's tests; no dedicated test warranted for a dead-clause removal.

### CORE-17 — `_mrmr_class.py` at 4285 LOC, 4.3× budget — grandfathered, with a stale LOC figure in the exemption note  [P3]
**Where:** `tests/test_meta/test_no_file_over_1k_loc.py:24-47`
**What:** The file IS grandfathered — the exemption entry at line 47
(`# - src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py`) is live, so the gate does not fail. The
accompanying `FIXME(carve-wave-next)` header at line 24 states "at ~4.76k LOC" and the carve log at line 45 ends
"4497 -> 3544 LOC; still exempt". The file is 4285 LOC today — neither figure matches, so the note is
self-contradictory and cannot be used to tell whether the file is growing or shrinking (the gate's stated
purpose: "no grandfathered one grows").
**Why it matters:** CLAUDE.md's rule is "carve BEFORE ~800-900 LOC; the meta-test is a backstop, not the design".
The exemption note itself already concludes the residual is not logic: "the irreducible residual is `__init__`'s
~2080-line parameter docstring + fit; further LOC drop needs relocating that docstring, not more logic carving."
That diagnosis matches what I read — `__init__` spans 334-3170 and is overwhelmingly per-parameter prose.
**Concrete carve proposal** (repo convention: focused sibling + re-export from the parent facade, with the
post-move AST audit for unresolved `Load`-context names that CLAUDE.md mandates):

1. **`_mrmr_class_params.md` (or `_mrmr_param_docs.py`) — the ~2080-line `__init__` docstring.** The single
   biggest and lowest-risk win, and the one the exemption note already names. Move the per-parameter prose to a
   docs file (or a module-level `_INIT_PARAM_DOC` string in a leaf module assigned to `MRMR.__init__.__doc__`
   at class-binding time in `mrmr/__init__.py`, next to the existing bindings at lines 160-217). Leave a short
   summary docstring plus a pointer in place. Expected: ~4285 → ~2200 LOC, no behaviour change.
   Watch: `tests/test_meta/test_public_docstrings.py` and `test_pydoclint_baseline.py` read this docstring —
   verify both before/after, and keep `numpydoc`-style sections intact wherever they land.
2. **`_mrmr_class_pickle.py`** — `__getstate__`, `__setstate__`, `_MRMR_SCHEMA_VERSION`,
   `_SETSTATE_LEGACY_OVERRIDES` (3201-3333, ~135 LOC). Must be bound onto the class body the same way
   `__setstate__`'s own comment at 3272-3274 requires (it overrides `BaseEstimator.__setstate__`, so a MIXIN
   would be shadowed) — bind as a plain function in `mrmr/__init__.py`, exactly as `set_params` already is at
   `__init__.py:216-217. Pairs naturally with the existing `_mrmr_setstate_defaults.py`.
3. **`_mrmr_fit_restore.py`** — the `finally`-block restore closures (`_restore_toggles_snapshot_and_raise`,
   `_restore_synergy_bonuses`, `_make_fe_budget_restorer`, `_make_default_screen_restorer`,
   `_restore_fast_search_knob`, `_make_fast_search_restorer`, `_restore_fe_auto_flags`,
   `_drop_target_cleanup_columns`, `_refresh_signature_params_post_restore`, plus `_safe_restore` at 171),
   lines ~3790 and 4140-4270, ~250 LOC. Cohesive (all are "undo a fit-scoped mutation") and would be retired
   wholesale by CORE-10's resolved-config refactor — so carve it only if CORE-10 is deferred.
4. **`_mrmr_class_repr.py`** — `__repr__` + the `_VALID_*` class-attribute re-binding block (3172-3215,
   ~45 LOC). Small; fold into (2) rather than making a fourth sibling.

Net after (1)+(2)+(3): roughly 4285 → ~1800 LOC. Still over budget, still exempt, but the residual becomes
`__init__`'s parameter list + `fit`/`_fit_body` — genuinely irreducible without an API change. Update the
exemption note's LOC figure in the same commit so the "does not grow" check has a real anchor.
**Test:** `test_mrmr_class_loc_exemption_figure_is_current` — parse the LOC number out of the FIXME comment and
assert it is within, say, 5 % of the file's actual line count, so the note can never go stale again.

---

## Proposed tests (beyond the per-finding ones)

- `test_clone_round_trips_every_ctor_param` — `sklearn.base.clone(MRMR(**non_default_for_every_param))` and
  assert `clone.get_params() == original.get_params()` across ALL ~300 params, not a sampled subset. The
  config/flat interaction (`invalidate_stale_mrmr_configs`, `_mrmr_config_dataclasses.py:257-285`) is the
  fragile part and is currently exercised only for the flat-vs-config DISAGREEMENT case.
- `test_set_params_with_a_nested_config_then_clone_is_stable` — `set_params(dcd_config=DCDConfig(dcd_enable=False))`
  then `clone()` twice; assert the second clone equals the first. Pins the "config wins, then invalidate" ordering
  at `_mrmr_config_dataclasses.py:298-313` against a future reordering.
- `test_partial_fit_matches_full_fit_on_the_same_rows` — `partial_fit(A); partial_fit(B)` with
  `partial_fit_min_recompute=1, partial_fit_decay=0.0, partial_fit_window=None` must give the same `support_`
  as `fit(concat(A, B))`. This is the incremental path's core contract and I found no test asserting it. The
  uniform-weight short-circuit at `_mrmr_class_fit_helpers.py:325-327` is what should make it hold — if it
  fails, that short-circuit is not firing.
- `test_partial_fit_decay_one_weights_only_the_last_batch` — mutation-resistant assertion on the actual weight
  VECTOR `_decay_weights([10, 10, 10], 1.0)`: first 20 entries `== _WEIGHT_FLOOR`, last 10 `== 1.0`.
- `test_getstate_contains_no_unpicklable_or_oversized_values` — walk `est.__getstate__()` after a real fit and
  assert every value pickles standalone AND that the total is under a stated budget relative to the input size.
  This is the general form of CORE-4/CORE-5/CORE-14 and would have caught all three.
- `test_every_terminal_fit_path_sets_the_same_attribute_roster` — parametrised over classic /
  identity-shortcut / multioutput / stability; the general form of CORE-2.
- `test_recommend_enabled_fe_lists_only_real_ctor_params` — the three hand-written lists at
  `_mrmr_class_config.py:490-514` are pure literals with nothing tying them to the ctor. Assert every name in
  `flip_safe + already_default + flip_risky` is a key of `MRMR._ctor_defaults()`, and that every
  `fe_*_enable` ctor param appears in exactly one of the lists. A renamed or removed flag currently rots there
  invisibly.
- `test_setstate_injects_no_key_absent_from_the_live_class` — assert every key of `_SETSTATE_LEGACY_DEFAULTS`
  is either a current ctor param or a name assigned somewhere in the package, so the 250-key roster cannot
  accumulate injections for parameters that no longer exist.

## Prior-wave findings touching this cluster

| Finding | Status in current source |
|---|---|
| `HybridOrthScorersConfig.ensemble_scorers` default `()` silently emptied the real 5-tuple | **HOLDS FIXED.** `_mrmr_config_dataclasses.py:124` carries the full `("plug_in", "ksg", "copula", "dcor", "hsic")` and matches the ctor at `_mrmr_class.py:2369`. Now gated by `tests/test_meta/test_config_dataclass_defaults_match_ctor.py`, which enumerates BOTH hybrid-orth maps — the regression cannot recur silently. |
| D5 — a `__setstate__` literal drifting from the ctor default (e.g. `cluster_aggregate_mode`) | **HOLDS FIXED.** `_mrmr_class.py:3297-3309` re-sources every shared key from `_ctor_defaults()`, with the divergences enumerated and justified in `_SETSTATE_LEGACY_OVERRIDES` (3226-3239). |
| P0 pickle BC — a ctor param absent from the hand roster raised `AttributeError` on reload | **HOLDS FIXED** for ctor params (`_mrmr_class.py:3312-3332`). Does NOT extend to fitted attributes — see CORE-11. |
| `_partial_fit_batch_sizes_` omitted from the setstate roster, collapsing multi-batch history | **HOLDS FIXED.** `_mrmr_setstate_defaults.py:252-255`, with the incident recorded in the comment. |
| FS-P2-1 — string ctor params validated with an actionable message | **PARTIALLY HOLDS.** The `_VALID_*` allow-lists and `_validate_string_params` are in place, but six hybrid-orth string params were added since and never joined the table — see CORE-8. |
| `get_feature_names_out` ignored `input_features`; the `startswith("feature_")` heuristic misclassified user columns | **HOLDS FIXED on the classic path** (`_mrmr_class_transform.py:60-89`, sentinel `_feature_names_in_synthesized_`). **REGRESSED on the stability path**, which never sets the sentinel — see CORE-2. |
| `check_is_fitted` accepted a half-fit instance | **HOLDS FIXED.** `__sklearn_is_fitted__` at `_mrmr_class_transform.py:173-174` requires both `support_` and `feature_names_in_`. |
| `n_jobs` / `parallel_kwargs` resolved at construction time, baking the constructing host's core count into pickles | **HOLDS FIXED.** Resolved lazily via `_effective_n_jobs` / `_effective_parallel_kwargs` (`_mrmr_class_config.py:258-298`), with the hazard documented at `:400-407`. |
| `_resolve_target_prefix` consumed the process-global numpy RNG | **HOLDS FIXED.** `_mrmr_class_config.py:300-318` uses a local `default_rng` or a PID-based token. |
| `_coerce_target_dtype` truncated int64 targets outside int16 range | **HOLDS FIXED**, and the skip-notice warning is now ungated (`_mrmr_class_config.py:337-345`). |

## Verified-clean

Checked and found genuinely fine — do not re-audit blind:

- **`__getstate__` / lock exclusion.** `_fit_reentrancy_lock_` is the only non-picklable instance attribute in
  the cluster; it is popped (`_mrmr_class.py:3267`) and lazily recreated (3247-3251). Grepped every
  `self.<attr> = ` assignment across all ten files for cache/lock/pool/handle/stream/device/GPU/executor
  patterns: the only hits are `_engineered_names_cache_` (CORE-14, picklable) and the `_partial_fit_*`
  buffers (picklable, and persisted by design per the module docstring at `_mrmr_partial_fit.py:41-48`).
  No cupy array, no device buffer, no open handle, no closure is ever attached to a `MRMR` instance.
- **Schema-version downgrade detection.** `_MRMR_SCHEMA_VERSION` is stamped in `__getstate__` (3268) and a
  newer-than-installed pickle raises a real `UserWarning`, not a `debug` log (3282-3291) — correct per the
  brief's silent-fallback rule.
- **Deep-copy isolation of mutable setstate defaults.** Three independent layers: `build_setstate_defaults()`
  deep-copies the template (`_mrmr_setstate_defaults.py:397`), the ctor overlay deep-copies mutable values
  (`_mrmr_class.py:3309`), and the backfill pass does too (3332). No aliasing across unpickled instances.
- **Nested-config default drift.** Every mapped config field default equals its flat ctor default, and this is
  gated by `tests/test_meta/test_config_dataclass_defaults_match_ctor.py`, which explicitly enumerates BOTH
  hybrid-orth maps and carries justified exemptions only for the four `FastSearchConfig` profile fields.
  I re-verified the mapping construction at `_mrmr_config_dataclasses.py:202-208` by hand against the ctor.
- **`extra="forbid"` + `frozen=True`** on `_MRMRSubConfig` (`_mrmr_config_dataclasses.py:29`) — unknown FIELD
  NAMES do raise at construction and the configs cannot be mutated after the fact. Only the string VALUES are
  unvalidated (CORE-8).
- **`set_params` → `clone` self-consistency.** The `mrmr_set_params` / `invalidate_stale_mrmr_configs` pair
  (`_mrmr_config_dataclasses.py:257-314`) correctly nulls a config that no longer agrees with its flats, which
  is what stops `clone()` raising sklearn's "constructor modifies a parameter" `RuntimeError`. The precedence
  (config applied after flats, in both `__init__` and `set_params`) is consistent between the two paths — I
  checked `_mrmr_class.py:3158-3169` against `_mrmr_config_dataclasses.py:304-313`.
- **`fit` re-entrancy guard.** Non-blocking `acquire` with a `try/finally` release (`_mrmr_class.py:3350-3366`);
  the `_enter_active_fit_scope` / `_exit_active_fit_scope` pair is in its own inner `try/finally`, so a raise in
  `_fit_body` cannot leak either the lock or the in-flight count.
- **`sample_weight` handling on the classic path.** `_maybe_resample_for_sample_weight`
  (`_mrmr_class_fit_helpers.py:303-343`) validates 1-D, length, finiteness, non-negativity and non-zero sum,
  each with a distinct actionable message; short-circuits on uniform weights (326-327), which is what makes
  `partial_fit`'s `np.ones` vector a true no-op; and seeds the draw deterministically from
  `_effective_random_seed()`. No `sum(x^k)`-minus-a-power-of-the-mean construction and no additive-epsilon
  denominator padding anywhere in this cluster — I grepped all ten files for `**2`, `**3`, `**4`, `1e-12` and
  `+ eps` patterns and the only numeric expression of any kind is `_decay_weights`' `(1 - decay) ** age`
  (`_mrmr_partial_fit.py:124`), which is a weight, not a moment.
- **`_apply_rolling_window`.** The de-duplicated single loop (`_mrmr_partial_fit.py:149-158`) is correct: once
  `drop_remaining` hits 0, later iterations append each batch's full size unchanged, and the comment's claim to
  that effect matches the code. `X_trimmed`/`y_trimmed` use the same `drop` offset, so rows and the registry
  stay aligned.
- **`_to_series` multi-output guard** (`_mrmr_partial_fit.py:83-106`) — raises explicitly for 2-D y instead of
  `.ravel()`-flattening into bogus rows, on both the DataFrame and ndarray branches.
- **Performance.** Grepped all ten cluster files for `@njit`, `parallel=True`, `prange`, `cuda.jit`, `cupy`,
  `KernelTuningCache` / `get_or_tune`: no compiled kernels are present and none are applicable — this cluster is
  constructor plumbing, reflection and dict manipulation, with no per-row or per-candidate loop. The reflection
  hot spots that DID matter were already fixed and are correctly cached: `_ctor_defaults`
  (`_mrmr_class_config.py:375-380`), `_fe_enable_attr_names` (389-393), `_resolve_fresh_instance_defaults`
  (417-426), and the two KTC lookups (88-105, 114-131) — all one-shot per class per process. The
  `get_params()` hoist out of the bootstrap replicate loop (`_mrmr_class_fit_helpers.py:241-252`) is done. I
  make no speedup claim anywhere in this document; CORE-12 is the only remaining per-call candidate and it ships
  with a bench plan, not a number.
- **Fallback logging quality.** Every broad `except` in the cluster was inspected against the brief's rule. The
  `debug`-level ones substitute a NEUTRAL value and are acceptable: `_fast_search_default_subsample_n` /
  `_default_screen_subsample_n` fall back to the documented HW-agnostic constant
  (`_mrmr_class_config.py:102-103, 128-129`); `_apply_default_screen_subsample` returns an empty `saved` dict
  and leaves knobs untouched (156-158); `_apply_fast_search_profile` treats all knobs as user-set (187-188);
  `_resolve_fresh_instance_defaults` falls back to the raw signature defaults (422-424); `__repr__` falls back to
  the plain repr (`_mrmr_class.py:3189-3190). The two that could hide something real are correctly louder:
  the stability outer-loop failure raises a `UserWarning` naming the exception type and the likely cause
  (`_mrmr_class.py:3541-3549`), and `recommend_enabled_fe`'s recommender failure logs at `warning`
  (`_mrmr_class_config.py:524-525`). No `except` in this cluster substitutes a non-neutral value that would
  disable a check it feeds.
- **Comment-rule compliance.** Per CLAUDE.md's "no process/audit metadata in code comments": the cluster is
  largely clean, but a handful of dated wave/layer markers survive — `_mrmr_validate_transform.py:46, 54, 124`
  ("2026-05-30 Wave 9", "2026-06-01 Layer 85") and `_mrmr_class_config.py:22` ("perf audit findings #4/#7/#8,
  2026-07-17"). The first file is another cluster's; the second is mine. Flagging rather than filing as a
  separate finding since `tests/test_meta/test_no_audit_metadata_in_comments.py` has a baseline
  (`_audit_metadata_baseline.json`) that presumably grandfathers them — worth confirming the baseline covers
  exactly these and no more.
