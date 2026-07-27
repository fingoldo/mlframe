# FE orchestration / gating / budget infrastructure -- 2026-07-22 audit

**Cluster scope**: the pair-FE transform registries (`feature_engineering.py`), the auto-escalation
proposer for prescreen-surviving pairs with zero/partial admission (`_fe_auto_escalation.py`), the
rung-0 successive-halving screen for the operator search (`_fe_rung_schedule.py`), the Shapley-flavored
per-family compute-budget reallocator (`_fe_family_budget.py`) and its wall-clock ledger
(`_fe_family_timing.py`), the thread-local enrichment-generator deadline (`_fe_deadline.py`), the
shared stratified row-subsampler (`_fe_subsample.py`), the (gated-off, unwired) matrix-native frame
adapter (`_fe_matrix_io.py`), the format-agnostic pandas/polars frame-op seam (`_fe_frame_ops.py`), the
CPU/GPU batch-MI backend dispatcher (`_fe_batch_dispatch.py`), the two-tier local-MI / cross-mechanism
CMI gates for the count/target-encoding/missingness/ratio-delta FE mechanisms (`_unified_fe_gate.py`),
the held-out downstream-uplift accuracy gate (`_fe_accuracy_gate.py`), the tail-concentrated
linear-usability signal shared by three FE gates (`_fe_usability_signal.py`), the cross-fold recipe
stability voter (`_fe_stability_vote.py`), the cross-backend MI binning/tiebreak contract
(`_fe_mi_contract.py`), the "do the raws already linearly explain y" skip-gate for the discrete-structural
operators (`_fe_linear_explainability.py`), and the engineered-vs-engineered retention subsumption guard
(`_fe_retention_subsumption.py`). Together these are the scaffolding around the FE search rather than the
search itself: they decide how much compute each family gets, when to stop, which candidates from a
prescreen/family/recipe survive a cross-cutting gate, and how to keep two backends (CPU/GPU) or two
data reps (pandas/polars) selection-equivalent.

All 17 files were read in full. `mypy --cache-dir=.mlframe_mypy_cache_shared` reports zero issues across
all 17. None of the six "most-P0/P1-fixed" commits cited in the task prompt (741926f8c, 6bca572d7,
68ebd6a29, f067e0d44, 2cc59a6b1, 6a09aa8b0) touch any file in this cluster (`git show --stat <sha> --
src/mlframe/feature_selection/filters/` has zero hits for every one of these 17 filenames), so none of
this cluster's prior findings were incidentally fixed by that batch. Angle 9 (SQL/HTTP/UI): confirmed
N/A -- no DB/network/UI surface in any of these 17 files; the only "artifact" any of them writes is a
local JSON cache file (`_fe_family_budget.persist_budgets`) and log lines, neither browser- nor
network-facing.

Of the 17 files, 6 were in the prior 2026-07-20 audit's `c7d_fe_orchestration.md` (`feature_engineering.py`,
`_fe_rung_schedule.py`, `_fe_family_timing.py`, `_fe_deadline.py`, `_fe_batch_dispatch.py`, and
`_unified_fe_gate.py` only via `fe_expansion.md`'s no-issue GPU-residency note) -- those findings are
cross-referenced below rather than restated. The other 11 files (`_fe_auto_escalation.py`,
`_fe_family_budget.py`, `_fe_subsample.py`, `_fe_matrix_io.py`, `_fe_frame_ops.py`, `_fe_accuracy_gate.py`,
`_fe_usability_signal.py`, `_fe_stability_vote.py`, `_fe_mi_contract.py`, `_fe_linear_explainability.py`,
`_fe_retention_subsumption.py`) do not appear in any prior audit location at all and got a full first-pass
review.

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| FE_ORCH_BUDGET-1 | P1 | bug | `_fe_linear_explainability.py:84` | `raws_linearly_explain_y`'s regression skip-gate scores **in-sample** R^2 of a plain `LinearRegression` over all numeric raw columns with no check on the columns-vs-rows ratio; empirically confirmed (`sklearn` on pure noise, `n=2000` matching the function's own `max_rows` default) that R^2 crosses the gate's default 0.92 threshold once `p` numeric raw columns exceeds ~95% of the (possibly-subsampled) row count purely from overfitting, with **zero real linear signal** -- e.g. `p=1900,n=2000` gives R^2=0.936 on independent noise. MRMR is a wide-`p` feature-selection tool, so a fit with `p` in the high hundreds/low thousands and the default 2000-row subsample cap is a plausible, not exotic, shape. When this fires it silently sets `_discrete_fe_master=False` for the WHOLE fit (`_mrmr_fit_impl/_fit_impl_core.py:3851-3852`), disabling the pairwise-modular/row-argmax/conditional-gate/binned-agg discrete-structural operators even though there is no real linear explanation of `y` at all -- a false "nothing left to find" verdict driven by the p/n ratio, not by genuine signal. No caller-side or gate-side guard on `p` vs `n` exists anywhere on this path, and the function has **zero direct unit-test coverage** (grepped `tests/`; only reachable transitively through a full fit). | NEW -- not in any prior audit (file/function absent from `c7d_fe_orchestration.md`, `fe_expansion.md`, `full_audit_2026-07-21`, and the `mrmr_critique_2026_07` critique). |
| FE_ORCH_BUDGET-2 | P1 | security / thread-safety | `_unified_fe_gate.py:67-91,148-219`; `_fe_accuracy_gate.py:42-44,156-163,163-193` | Four module-level bounded-FIFO memo caches (`_COERCE_Y_MEMO`, `_RAW_MI_FLOOR_MEMO` in `_unified_fe_gate.py`; `_BASELINE_CV_MEMO`, `_INFER_CLS_MEMO` in `_fe_accuracy_gate.py`) all use the unlocked `if len(cache) > N: cache.pop(next(iter(cache)))` eviction idiom with **no lock**, on plain module-level `dict`s consulted on every FE gate call (`raw_mi_noise_floor`, `local_mi_gate`, `measure_feature_uplift`, `infer_classification`). This is the *exact* race class (`popitem`/`__setitem__` on a shared cache under concurrent fits) the codebase already identified and fixed elsewhere in this same package: `_mrmr_fit_impl/_fit_impl_core.py:32-39` introduced `_MRMR_FIT_CACHE_LOCK` specifically because "Concurrent fits -- multi-target discovery, joblib-threading callers, web-service workers -- otherwise race `popitem`/`__setitem__`/`move_to_end`... and can raise KeyError or evict the wrong entry", and this cluster's own `_fe_family_timing.py` guards its wall-time dict with `_LOCK = threading.Lock()` for the identical documented reason. Two threads racing the same eviction step (`next(iter(cache))` returns the same first key to both, one `.pop()`s it, the second `.pop()` on the now-missing key raises `KeyError`) will crash whichever FE gate call hits it, propagating out of `raw_mi_noise_floor`/`local_mi_gate`/etc. uncaught (no try/except wraps the memo internals). Concurrent `MRMR.fit()` calls (multi-target discovery, a service serving multiple fit requests) are a documented, supported usage pattern in this very codebase, so this is a real, reachable crash path, not a hypothetical. | NEW -- not mentioned by any prior audit; grepped all 4 cache names across `audits/` and the critique directory with zero hits. |
| FE_ORCH_BUDGET-3 | P2 | correctness / dead-code | `_fe_matrix_io.py:80,225-236,246,258` | `FeatureMatrix.numeric_column` and `from_feature_matrix` both resolve a column by `columns.index(name)`, which returns only the **first** index for a duplicate column name; the two loops that build `col_objs`/`data`/`out_pd` are also keyed by `name` in a plain dict, so a second column sharing a name silently **overwrites** the first in the round-trip output. pandas explicitly permits duplicate column labels, so a frame with a repeated name loses one of the two columns' data with no error or warning. Currently unreachable in production: the module's own docstring states it is "GATED OFF by default... UN-WIRED -- nothing in the FE path imports this yet", confirmed by grep (only the dedicated `test_fe_matrix_parity.py` imports it, and that suite has no duplicate-column-name test), so this is a real but currently-dormant bug that will surface the first time this P0-of-the-replatform adapter gets wired into a caller that can see duplicate-named input frames. | NEW -- module post-dates 2026-07-20 (absent from all three prior-audit locations). |
| FE_ORCH_BUDGET-4 | P2 | code-quality | `_fe_auto_escalation.py:118-122` (`_candidate_values`), `:290-293` (`_propose_poly`'s `fit_pair_prewarp_als` call), `:397-403` (`_resolve_operand`) | Three `except Exception: return None` blocks swallow the exception with no `logger.debug`/`logger.warning`, unlike every other exception handler in the same file (which all log via `logger.debug(...)`, e.g. lines 255-256, 278-279, 545-546, 673-675, 724, 858-861) and unlike CLAUDE.md's "no silent except-Exception swallowing without logging" convention. Given the file's own stated design ("Never raises... degrades to `[]`") this is low-blast-radius (a genuine bug in `apply_operand_prewarp`/`fit_pair_prewarp_als`/column resolution would just silently suppress an escalation candidate rather than corrupt output), but it is inconsistent with the file's otherwise-careful logging discipline and would make a real regression in those three helpers invisible in logs. | NEW. |
| FE_ORCH_BUDGET-5 | P2 | architecture | `_fe_auto_escalation.py` (871 LOC) | Approaching the repo's "carve before nearing 800-900 LOC" guideline (CLAUDE.md); not over the 1k-LOC hard backstop, but the largest file in this cluster by a wide margin (next is `_unified_fe_gate.py` at 414) and still growing per its own module history. Worth carving `_propose_poly`/`_propose_fourier`/`_fit_fourier_amplitude_spec` (the two proposer families, ~260 LOC) into a sibling before the next addition pushes it over 900. | NEW (module post-dates the prior audit). |
| FE_ORCH_BUDGET-6 | P2 | test_gap | `_fe_deadline.py`, `_fe_family_timing.py`, `_fe_batch_dispatch.py` | Still open, see `c7d_fe_orchestration.md`'s test_gap findings (no dedicated unit test file exists for any of the three as of this audit -- confirmed via `find tests -iname "*fe_deadline*" -o -iname "*fe_family_timing*" -o -iname "*fe_batch_dispatch*"`, zero hits). `_fe_batch_dispatch.py`'s force-env branch (`MLFRAME_FE_VRAM_BACKEND=cpu`) IS now covered by `tests/feature_selection/mrmr/fe/test_fe_batch_parity.py:80-85`, but the STRICT-flag / KTC-crossover / CUDA-downgrade precedence branches the prior report called out remain untested. | Still open (partially narrowed), see `c7d_fe_orchestration.md`. |
| FE_ORCH_BUDGET-7 | P2 | cpu_gpu_parity | `feature_engineering.py:666-668` | `greater`/`less`/`equal` still return `.astype(int)` on CPU (`create_binary_transformations`) vs `.astype(a.dtype)` (float32/float64) on the GPU-resident twin (`engineered_recipes/_recipe_unary_binary_gpu.py:169-173`, confirmed unchanged by re-reading the file) -- same values, divergent dtype by backend. | Already fixed = NO; still open, see `c7d_fe_orchestration.md` P2 finding (unchanged). |
| FE_ORCH_BUDGET-8 | P2 | hygiene | `_fe_family_budget.py:232-266` (`persist_budgets`/`load_budgets`) | `cache_key`/`fingerprint` are concatenated directly into a filename (`f"{cache_key}.{fingerprint}"`) with no sanitisation against path-separator/`..` characters. Not currently exploitable -- grepped every call site (`mrmr/_mrmr_class.py:3240,3859`, the one benchmark script) and confirmed `fingerprint` is always `dataset_fingerprint()`'s sha256 hex digest and `cache_key` is never overridden from its literal default -- but the public function signature accepts an arbitrary string with no defensive `os.path.basename`/character-allowlist, so a future caller passing a user- or column-derived string here would have a path-traversal write primitive into `~/.cache/mlframe/...`. | NEW. |

## Proposals

### (coverage_gap) Direct unit test for `raws_linearly_explain_y`'s p/n behaviour
Add a regression test that constructs pure-noise `X` with `p` numeric columns close to (but below) the
subsampled `n` (e.g. `p=1900, n=2000`, matching the empirical repro in FE_ORCH_BUDGET-1) and asserts the
gate does NOT fire (`raws_linearly_explain_y(...) is False`) despite crossing 0.92 in-sample R^2 on plain
`LinearRegression` -- i.e. pin whatever fix is chosen (a train/held-out split, a `p`-vs-`n` guard capping
`num_cols` before the fit, or an adjusted-R^2 / effective-dof correction) rather than leaving the
in-sample score as the sole signal. Currently there is no test file for this module at all.

### (coverage_gap) Concurrent-fit stress test for the FE gate memo caches
Extend whatever test currently exercises `_MRMR_FIT_CACHE_LOCK`'s concurrency (if one exists) or add a new
`ThreadPoolExecutor`-driven test that calls `raw_mi_noise_floor` / `local_mi_gate` / `measure_feature_uplift`
/ `infer_classification` from >8 distinct-content-keyed threads simultaneously (forcing repeated FIFO
eviction) and asserts no `KeyError` — this is the cheapest way to convert FE_ORCH_BUDGET-2 from "identified
by code inspection" to "reproduced", and to pin whichever lock is added as the fix.

### (design) Route `_fe_family_budget`'s per-family credit through a shared recipe-kind registry
`_RECIPE_KIND_TO_FAMILY` in `_fe_family_budget.py` and the wall-ledger's `@fe_timed(...)` call-site names in
sibling FE-family modules are two independently-maintained string vocabularies that must stay in sync by
hand (the module's own docstring already documents one real, currently-unresolvable drift for
`adaptive_arity`). A single shared enum/registry both sides import would make a THIRD family added to one
side and not the other a type error instead of a silent credit-misattribution.
