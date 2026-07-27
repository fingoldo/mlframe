# Screening/confirmation greedy loop + permutation gates (cluster A)

Files: `screen.py` (170), `_screen_predictors.py` (919), `_screen_predictors_gate.py` (185),
`_screen_predictors_prescreen.py` (164), `_screen_dcd_swap.py` (150), `_confirm_predictor.py` (974),
`_confirm_predictor_engineered.py` (323), `evaluation.py` (997), `_evaluation_driver.py` (461),
`fleuret.py` (337), `_cmi_perm_stop.py` (177). Total ~4857 LOC, all read in full.

This cluster is the mRMR greedy-selection engine's core loop: `screen_predictors` (screen.py /
_screen_predictors.py) walks interaction orders and delegates each single-predictor confirmation
cycle to `confirm_one_predictor` (_confirm_predictor.py), which composes `score_candidates`
(per-candidate conditional-MI gain via `evaluate_candidate`/`evaluate_gain` in evaluation.py /
_evaluation_driver.py) with `confirm_candidate` (permutation-confirmation: a marginal `mi_direct`/
`mi_direct_gpu` bootstrap plus, in complex mode, a Fleuret conditional-MI permutation recheck via
fleuret.py). `_screen_predictors_gate.py` holds the abs/relative/maxT-FDR selection-gate math and
Dynamic Cluster Discovery (DCD) state construction; `_screen_predictors_prescreen.py` holds the
cardinality pre-screen and the maxT gain-floor cache; `_screen_dcd_swap.py` is the DCD
discover/anchor→PC1-swap step; `_confirm_predictor_engineered.py` holds the directed-FE tie-break
and per-signal prefer-engineered substitution; `_cmi_perm_stop.py` is an opt-in CMI-permutation
early-stopping criterion plus the UAED elbow detector.

## Prior-audit cross-reference

`c2_screen_confirm.md` (2026-07-20, 14 findings) covers most of this exact file set. Re-verified
against current `HEAD` (git log / `git show`):

- B-5, B-6, B-7 (confirm_candidate's marginal + Fleuret permutation seeds never reached
  `ctx.random_seed`) — **fixed**, commit `741926f8c` (`_confirm_predictor.py:552,590,608,669,690`
  now fold `hash(X)` + `random_seed` into a per-candidate `base_seed`, threaded to both
  `mi_direct` and `mi_direct_gpu`).
- S-F2 (JMIM confirmed against the wrong/CMIM statistic) — **fixed** per
  `_benchmarks/mrmr_critique_2026_07/_TRACKER.md` ("DONE (sf2)"); confirmed in `fleuret.py`:
  `use_jmim` is now threaded `get_fleuret_criteria_confidence_parallel → parallel_fleuret →
  get_fleuret_criteria_confidence → evaluate_gain`.
- The `evaluation.py:566` sibling of B-5 (evaluate_candidate's *baseline*-relevance
  `mi_direct`/`mi_direct_gpu` calls never received a seed) was **rejected by majority vote in the
  2026-07-20 tracker only on a factual imprecision about the GPU mechanism**, with an explicit
  tracker caveat "treat as effectively confirmed... not a false alarm." It was **not part of the
  741926f8c fix batch** (that commit only touches `_confirm_predictor.py`) and remains open —
  restated below as SCREEN_CONFIRM_A-1 since the prior report's disposition table would otherwise
  make it look closed.
- P2 finding "`should_skip_candidate`'s DCD prune-mask lookup swallows any exception, zero
  logging" (evaluation.py) — **still open**, re-verified at the current line range; restated
  below (SCREEN_CONFIRM_A-6).
- P2 finding "`find_best_partial_gain`'s DCD-prune import guard is a bare
  `except Exception: pass`" — **still open** (SCREEN_CONFIRM_A-7).
- P2 finding "`build_dcd_state` only logs a DCD-init failure when `verbose` is truthy" — **still
  open** (SCREEN_CONFIRM_A-8).
- P1 perf finding "`seed_workers_pool` dead for the still-active Fleuret confirmation path,
  `workers_pool=None` forces a fresh joblib.Parallel per candidate" — **still open**, re-verified:
  `_screen_predictors.py:608` hardcodes `workers_pool = None` (2026-07-19 retirement, correctly
  scoped to the `evaluate_candidates` pool by its own comment), but `ctx.workers_pool` threads
  that same `None` into `confirm_candidate` → `get_fleuret_criteria_confidence_parallel` →
  `fleuret.py:127-129`'s `if workers_pool is None: workers_pool = Parallel(...)`, which still
  fires on every candidate whenever `n_workers > 1`. Restated below (SCREEN_CONFIRM_A-4).
- `mrmr_critique_2026_07` (S-F3 JMIM exponent, N-F1..N-F8) mostly targets `evaluation.py`/
  `permutation.py` numerics already resolved to DOC/DONE per that tracker; nothing new to add.

No P0s were found in this cluster (none carried over as open, none newly discovered).

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| SCREEN_CONFIRM_A-1 | P1 | bug / cpu_gpu_parity | evaluation.py:566,586,611 | `evaluate_candidate`'s baseline-relevance `mi_direct_gpu`/`mi_direct` calls never pass `base_seed`, so `ctx.random_seed` never reaches this per-candidate permutation-null gate: CPU always draws the identical `base_seed=0` stream (byte-identical null for every candidate/round of the whole fit), GPU draws fresh OS-entropy every call (non-reproducible). | Prior finding at evaluation.py:566, marked "rejected" in the 2026-07-20 tracker only on a wording technicality (own caveat: "effectively confirmed"); confirmed NOT fixed by 741926f8c (that commit only touches `_confirm_predictor.py`). Still open. |
| SCREEN_CONFIRM_A-2 | P1 | bug / robustness | _confirm_predictor.py:578-591 | `confirm_candidate`'s GPU marginal-confirmation call (`mi_direct_gpu`) has no `try/except`, unlike the structurally-identical call in `evaluate_candidate` (evaluation.py:565-599) which was explicitly hardened by a 2026-07-09 fix specifically because "a single CUDA fault here... would propagate all the way up and crash the whole MRMR.fit()". `mi_direct_gpu` itself (gpu.py) has no internal circuit breaker around its CUDA kernel launches, and `_fit_impl_core.py`'s `screen_predictors(...)` call site has no surrounding try/except either — a transient CUDA fault during confirmation (driver hiccup, OOM, a poisoned context from an earlier unrelated GPU call) aborts the entire (possibly hours-long) `.fit()` instead of degrading to CPU for that one candidate. | New — not raised by any 2026-07-20/07-21 report. |
| SCREEN_CONFIRM_A-3 | P1 | perf / efficiency | _cmi_perm_stop.py:32-63,92-113 | `_cmi_plugin_njit` (used by the opt-in `cmi_permutation_stop`, wired into `evaluate_candidate` via `get_cmi_perm_stop()`) allocates a DENSE `(K_x, K_y, K_z)` histogram where `K_z` is the PRODUCT of every selected feature's bin cardinality, capped only at 1,000,000 (not bounded earlier). With as few as ~6 ten-bin selected features `K_z` already reaches ~1e6, so `joint = np.zeros((K_x, K_y, K_z))` alone is `K_x*K_y*1e6` float64 cells (hundreds of MB to multiple GB for realistic `K_x`/`K_y`), rebuilt from scratch for the "observed" call AND every one of `n_permutations` (default 100) null draws, for every candidate scored in every greedy round once this stopping rule is enabled — deep into any real fit (10-50+ selected features is routine) this is a near-certain OOM/multi-minute hang for a feature that is silently invisible until it fires. | New — not raised by any prior report; derived from static shape/complexity analysis (not executed, per the read-only mandate), so flagged PLAUSIBLE pending an empirical repro. |
| SCREEN_CONFIRM_A-4 | P1 | perf | _screen_predictors.py:608 / fleuret.py:127-129 | `workers_pool = None` is now hardcoded (2026-07-19 `evaluate_candidates`-pool retirement), and `seed_workers_pool` is accepted but never used to build a real pool. Since `ctx.workers_pool` is always `None`, every `confirm_candidate` call that takes the still-active Fleuret parallel-confirmation branch (`n_workers>1` and `full_npermutations > NMAX_NONPARALLEL_ITERS`) hits `fleuret.py`'s `if workers_pool is None: workers_pool = Parallel(...)` and spawns a brand-new joblib pool per candidate instead of once per screen call. | Same as prior finding #4 in `c2_screen_confirm.md` (P1 perf); re-verified still open at current line numbers. |
| SCREEN_CONFIRM_A-5 | P2 | robustness / logging | evaluation.py:581-585 | `evaluate_candidate`'s GPU-relevance `except Exception` fallback has no circuit breaker: on a persistently broken GPU (bad driver, exhausted VRAM) every single candidate in every round re-attempts the doomed CUDA call and logs a fresh WARNING, instead of falling back to CPU-only for the remainder of the fit after the first fault. | New (closely related to, but distinct from, the already-fixed B-10 in `_ksg.py`, which was about exception-type narrowness, not repeat-cost). |
| SCREEN_CONFIRM_A-6 | P2 | bug / logging | evaluation.py:156-164 | `should_skip_candidate`'s DCD prune-mask lookup is a bare `except Exception: pass` (not even DEBUG) — a genuine `should_be_pruned`/DCDState bug silently degrades to "never pruned" with zero diagnostic trace. | Still open (prior `c2_screen_confirm.md` P2 finding, unchanged). |
| SCREEN_CONFIRM_A-7 | P2 | bug / logging | evaluation.py:958-963 | `find_best_partial_gain`'s DCD-prune import guard is `except Exception: _should_be_pruned = None` with no logging — an import-time bug is indistinguishable from "DCD not configured". | Still open (prior `c2_screen_confirm.md` P2 finding, unchanged). |
| SCREEN_CONFIRM_A-8 | P2 | logging | _screen_predictors_gate.py:67-73 | `build_dcd_state`'s except-block only logs a DCD-init failure `if verbose:` — at the library's own `verbose=0` default a DCD init exception is completely silent. | Still open (prior `c2_screen_confirm.md` P2 finding, unchanged). |
| SCREEN_CONFIRM_A-9 | P2 | architecture / module-size | _screen_predictors.py (919 LOC), _confirm_predictor.py (974 LOC), evaluation.py (997 LOC) | Three of this cluster's files sit at or just under the hard 1000-LOC CI backstop (`test_no_file_over_1k_loc.py`) despite each having ALREADY been the target of a prior carve specifically to stay "well below the 1k-line monolith threshold" (their own module docstrings say so) — none is meaningfully below the CLAUDE.md ~800-900 LOC "carve before" guideline any more; the next feature added to any of the three will need a further carve just to keep passing the hard 1k gate. | New observation (exact line counts weren't called out by name in the prior reports). |
| SCREEN_CONFIRM_A-10 | P2 | dead code / architecture | _evaluation_driver.py:186-461 (evaluate_candidates / _evaluate_candidates_inner, ~275 LOC) | Unreachable in production: their only production call site (`score_candidates`'s `n_workers>1` joblib-pool branch, `_confirm_predictor.py:283-401`) is permanently gated off by `_EVALUATE_CANDIDATES_POOL_ENABLED = False`, a hardcoded module constant with no env var / public parameter to flip it. The functions are still re-exported from `mrmr/__init__.py` and `_legacy.py` as if part of the live parallel-scoring path, and their ~275 lines (thread-local republish/restore, per-worker cache merge-back, JMIM stats) are exercised only by direct unit tests that call them standalone, never by an actual `MRMR.fit(n_workers>1)` run. | New. |
| SCREEN_CONFIRM_A-11 | P2 | dead code | screen.py:25, _screen_predictors.py:62 (`_pool_warmup_noop`) | Both copies are orphaned by the same 2026-07-19 `workers_pool` retirement: neither is called from any production code path (only from a regression test that checks the symbol exists and is callable — the exact "coverage looks fine but no production caller" failure mode `test_dead_helpers.py` exists to catch). Invisible to that scanner because its own docstring explicitly excludes any leading-`_` symbol as "intentionally module-internal". | New. |
| SCREEN_CONFIRM_A-12 | P2 | test coverage | _screen_predictors_prescreen.py (cardinality_prescreen, compute_fdr_gain_floor) | `cardinality_prescreen` and `_screen_predictors_gate.py`'s `compute_selection_gate` have no dedicated unit test file (grep of `tests/` turns up only `.prof` artifacts and an annotation-baseline JSON) despite both being carved out specifically to be independently testable pure functions ("Neither mutates caller locals"); coverage today is whatever an end-to-end `MRMR.fit()` integration test happens to exercise. | New. |
| SCREEN_CONFIRM_A-13 | P2 | test coverage | _cmi_perm_stop.py | The only tests for `cmi_permutation_stop` (`test_cmi_perm_stop_marginal_hoist_identity.py`, `test_cmi_perm_stop_conditional_null.py`) use at most `nbins_selected=[5,4]` (`K_z=20`); nothing exercises the many-selected-features regime where `K_z` grows toward the 1,000,000 cap, so SCREEN_CONFIRM_A-3's scaling blowup is entirely unguarded by tests. | New (paired with SCREEN_CONFIRM_A-3). |
| SCREEN_CONFIRM_A-14 | P2 | dead code | screen.py:90 (`ScreenState`) | Fully unused by the production orchestrator per its own docstring ("Not currently routed through"). Already tracked as an accepted exemption in `tests/test_meta/test_dead_helpers.py`'s allowlist, so no action item — noted for completeness only. | DOC (already tracked). |
| SCREEN_CONFIRM_A-15 | P2 | code quality | _screen_predictors_gate.py:115,141,181 | The pattern `int(y[0]) if hasattr(y, "__len__") else int(y)` is repeated verbatim 3 times in `compute_selection_gate` (once per `# defensive fallback for a caller passing a bare int` comment) instead of being factored into a one-line local helper. | New (minor). |

## Confirmed non-findings (checked, no issue)

- **CPU/GPU parity, confirm_candidate**: post-741926f8c, both `mi_direct` (CPU LCG) and
  `mi_direct_gpu` (CuPy `default_rng`) now receive the SAME derived `base_seed` per candidate;
  the two RNG algorithms are not bit-identical to each other by design (documented in
  `gpu.py`'s `mi_direct_gpu` docstring: "each path is internally reproducible under a seed... not
  cross-backend bit-parity"), which is the correct/intended contract, not a gap.
- **`hash(X)` determinism**: `X` is always a tuple of `int` column indices; Python's hash
  randomization (`PYTHONHASHSEED`) only affects `str`/`bytes`/`datetime`, not `int`/`tuple[int]`
  hashing, so the `_marginal_base_seed`/`_fleuret_base_seed` derivations are reproducible across
  processes as intended.
- **`maxt_floor_cache` key** (`_screen_predictors_prescreen.py:123-133`) omits
  `screen_fdr_target_oversplit_ratio`/`screen_fdr_min_rows_per_joint_cell` from its cache key —
  looked like a correctness gap at first read, but those two params only gate WHETHER the floor
  fires (computed fresh every call, never cached), not the cached floor VALUE itself, which is a
  pure function of the remaining (cached) key fields. Not a bug.
- **`_z_merge_cache`** (`_confirm_predictor_engineered.py`) correctly invalidates across a DCD
  swap: the swap replaces the anchor's index with a new aggregate index, changing `z_key`, so the
  cache misses and recomputes against the (already-updated) `ctx.factors_data` rather than
  silently reusing a stale pre-swap encoding.
- **Edge cases**: `screen_predictors` raises a clear `ValueError` below `n=10`; a fully
  high-cardinality pool degrades gracefully to an empty `selected_vars` (no crash) via
  `cardinality_prescreen`; `_cmi_plugin_njit`/`cmi_permutation_stop` handle `n<=0` and
  `K_x`/`K_y=0` without raising. No new all-NaN / single-row / constant-column crash was found in
  this cluster (all of it operates on already-discretized integer codes; NaN handling lives
  upstream in the discretization module, out of this cluster's scope).
- **Security / SQL / HTTP / UI applicability**: confirmed N/A for this entire cluster — no
  database access, no network I/O, no HTML/report generation, no `eval`/`exec`/`subprocess`/
  `os.system`. The only filesystem touch is `stop_file` (an `os.path.exists` check on a
  caller-supplied path, not attacker-controlled network input). `joblib.Parallel`/`delayed` is
  used purely for in-process compute parallelism, not deserialization of untrusted data.
  Feature names reach only `%s`-style logger arguments, never string-interpolated into anything
  executed or rendered.
- **Module-level mutable state thread-safety**: `_JMIM_CACHE_STATS` (a bounded `deque`) is
  appended to from joblib worker threads; `deque.append`/`popleft` are documented CPython-safe
  under the GIL. The MI caches (`cached_MIs`, `cached_confident_MIs`, `cached_cond_MIs`,
  `cached_jmim_MIs`, `entropy_cache`) are per-fit-local (built fresh in `screen_predictors`,
  threaded via `ScreenContext`), never module-level globals shared across concurrent `.fit()`
  calls.
- **mypy**: `mypy --cache-dir=.mlframe_mypy_cache_shared --ignore-missing-imports` on all 11
  files in this cluster reports "Success: no issues found."

## Proposals

- **Thread a real per-candidate seed into `evaluate_candidate`'s baseline calls** (fixes
  SCREEN_CONFIRM_A-1): mirror `_confirm_predictor.py`'s
  `int(((int(random_seed or 0) * 2654435761) + hash(X)) & 0xFFFFFFFF)` pattern for the
  `mi_direct`/`mi_direct_gpu` calls at evaluation.py:566/586/611, and add a regression test
  asserting `random_seed=1` vs `random_seed=2` (or two distinct candidates at the same round)
  produce different baseline-relevance permutation draws — this would fail today and pass after
  the fix.
- **Wrap `confirm_candidate`'s GPU marginal-confirmation call in the same try/except pattern
  `evaluate_candidate` already uses** (fixes SCREEN_CONFIRM_A-2): fall back to the CPU `mi_direct`
  branch on any GPU exception, with a WARNING log; add a regression test that monkeypatches
  `mi_direct_gpu` to raise and asserts `confirm_one_predictor` still completes (falls back) rather
  than propagating.
- **Bound `_cmi_perm_stop.py`'s conditioning cardinality far below 1,000,000, or switch to a
  sparse/hashed joint representation** (fixes SCREEN_CONFIRM_A-3): either cap `K_z` at a much
  smaller budget (e.g. 10_000, still generous for a conditional-independence test) before the
  dense array is ever allocated, or replace the dense `(K_x, K_y, K_z)` cube with a
  dictionary/np.unique-based sparse joint keyed on the OBSERVED `(x,y,z)` triples (bounded by `n`,
  not by the cardinality product). Add a test with 6+ ten-bin `selected_cols` asserting the call
  completes in bounded time/memory.
- **Add a circuit breaker for `evaluate_candidate`'s GPU relevance fallback** (fixes
  SCREEN_CONFIRM_A-5): after the first GPU exception in a fit, set a `ctx`-scoped flag that routes
  all subsequent candidates straight to CPU for the remainder of that `screen_predictors` call,
  mirroring the pattern already used by `_cmi_cuda.py`'s circuit breaker elsewhere in this
  package.
- **Re-thread `seed_workers_pool` for the still-active Fleuret parallel-confirmation path**
  (fixes SCREEN_CONFIRM_A-4, restates prior proposal): build the pool once per
  `screen_predictors` call when `n_workers>1`, store it on `ctx.workers_pool`, and return it via
  `seed_workers_pool` for the next round — while leaving the (already-retired,
  correctly-retired) `evaluate_candidates` pool off.
- **Delete or explicitly re-enable `evaluate_candidates`/`_evaluate_candidates_inner`**
  (SCREEN_CONFIRM_A-10): either wire `_EVALUATE_CANDIDATES_POOL_ENABLED` to a real
  env-var/parameter so the ~275-line parallel-scoring path is reachable again (if it's still
  wanted for some future workload shape), or remove it and its re-exports now that the isolated
  A/B in the file's own comment found it never wins.
- **Delete the two orphaned `_pool_warmup_noop` copies** (SCREEN_CONFIRM_A-11) and their
  existence-only regression test, or clarify in `test_dead_helpers.py`'s docstring that
  leading-`_` module-level defs are a known blind spot the sensor doesn't cover.
- **Add focused unit tests** for `cardinality_prescreen` and `compute_selection_gate`
  (SCREEN_CONFIRM_A-12): edge cases worth pinning — a pool where every column is refused (n small,
  all-high-cardinality), the MM-gate arithmetic at `interactions_order>1` (currently skipped by
  design), and the maxT-FDR floor branch with `cardinality_bias_correction=False`.
