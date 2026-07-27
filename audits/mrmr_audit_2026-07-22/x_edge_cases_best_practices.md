# Cross-cutting edge cases & world-class best-practices sweep (2026-07-22)

## Scope

This report is a cross-cutting, adversarially-minded pass over the MRMR feature-selection module, targeting the
subpackages that have grown or materially changed since the 2026-07-20 audit and were each already deep-dived by a
dedicated cluster this same wave: `engineered_recipes/` (recipe replay contract), `_feature_engineering_pairs/`
(pair-FE search engine), `_orthogonal_univariate_fe/` (univariate orth-basis FE — flagged by
`x_efficiency_architecture.md` as the one subpackage that fell through every cluster's scope), `_dynamic_cluster_discovery/`
(DCD), `hermite_fe/` (Hermite/Legendre/Chebyshev/Laguerre pair-FE), `discretization/`, `info_theory/`, `_fe_gpu_batch/`
(the new heterogeneous multi-GPU FE-batch executor + CP-SAT packer), and `_gpu_strict_fe/` (the STRICT-resident
scaffold + residency-audit harness). Rather than re-walking ground the 29 other 2026-07-20/2026-07-22 reports already
covered file-by-file, this pass read the actual source of the least-audited corners of that file list directly
(`_orthogonal_univariate_fe/__init__.py`, `_imbalance_mi.py`, `_orth_scoring_memo.py`, `_orth_dedup.py`,
`_fe_gpu_batch/*.py`, `_gpu_strict_fe/*.py`, `_fe_resident_operands.py`, `hermite_fe/_hermite_prewarp.py`,
`engineered_recipes/_recipe_core.py`, `_recipe_dispatch.py`, `_recipe_extract.py`) tracing specific edge-case
scenarios (all-constant/near-zero-baseline raw features fed into a recipe pipeline, extreme cardinality, mixed
DataFrame types, multi-device/multi-process cache sharing, hardcoded vs. resolved random seeds) end-to-end through
the code rather than pattern-matching for missing try/except blocks. Two genuinely new, previously-unflagged P1
findings resulted (a cross-device correctness gap in the brand-new multi-GPU FE-batch executor, and a Layer-27
noise-floor gate that degrades to a near-no-op on narrow candidate pools) plus several smaller P2 items. One
promising early lead — that ~65 `getattr(self, "random_seed", 0)` call sites across the FE pipeline silently ignore
`random_state=`-only callers — was investigated and **disproved** by tracing `mrmr/_mrmr_class.py:3711-3716/3910-3912`'s
documented scoped override (`self.random_seed` is temporarily patched to `self._effective_random_seed()` for the
duration of `_fit_body`, then restored in a `finally`), so it is recorded here only as a confirmed non-finding, not
reported as a bug — the codebase's own random-seed reconciliation is comprehensive.

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| X_EDGE_CASES_BEST_PRACTICES-1 | P1 | bug (cpu_gpu_parity / thread-safety) | `_fe_resident_operands.py:101-163` (`_FE_RESIDENT_OPERANDS`, `resident_operand`); `_fe_gpu_batch/_executor.py:76,104-165` (`gpu_fe_batch_mi`, `multi_gpu_fe_batch_mi`) | The module-level `resident_operand` cache that `gpu_fe_batch_mi` uses to upload the fit-constant `y_codes` "once per fit" is keyed **purely on content** (`(shape, dtype-str, content-hash)`, `_fe_resident_operands.py:154`) with **no device component at all** — by the module's own docstring, "keyed PURELY on a content fingerprint ... NOT on the caller's role discriminator". `multi_gpu_fe_batch_mi` (the new 2026-06-26 heterogeneous multi-GPU FE-batcher, wired into production via `_fe_batch_dispatch.fe_batch_mi` and `_orthogonal_univariate_fe/_orth_mi_backends.py:151-157`) spins up one `ThreadPoolExecutor` worker per physical CUDA device and has EVERY worker call `gpu_fe_batch_mi(..., device=profs[dev_slot].device, ...)` with the SAME `y_codes` content. The first device's thread uploads `y_codes` under `resident_operand(y_host, "gpu_fe_batch_y", ...)` and caches the resulting cupy array; every OTHER device's thread, calling with identical `y_codes` content but a DIFFERENT current `cp.cuda.Device` context, gets a cache HIT and receives back a cupy array physically resident on the FIRST device. Passing a device-N array into a kernel while device-M (N != M) is the active context is a documented cupy error (`ValueError: Array device must be same as the current device`) for the overwhelming majority of cupy operations (no explicit peer-access is enabled anywhere in this codebase) — so on any genuine 2+-GPU host, every device beyond the first hits this on its very first `_plugin_mi_classif_batch_cuda_resident` call. The exception is caught by `fe_batch_mi`'s own outer `except Exception: pass` (`_fe_batch_dispatch.py:91-92`, no logging) and silently falls back to the CPU path — so the user-visible effect is not a crash but a **silently defeated multi-GPU speedup with zero diagnostic trace**, on the exact feature this new subpackage exists to provide. Confirmed by direct code tracing (no multi-GPU hardware available to execute this read-only audit on) — not a hand-waved hypothesis: cupy's device-context contract is unambiguous and every other resident cache in this codebase that IS device-aware (`_gpu_strict_fe/_state.py`'s `ResidentFEState._operands: dict` keyed by `p.device`, built specifically for this multi-device scenario) demonstrates the authors already know how to do this correctly — that class is simply not used by the actual wired `_fe_gpu_batch/_executor.py` path (confirmed dead per `gpu_infra_c.md` GPU_INFRA_C-6). The ONLY test for this path, `tests/feature_selection/mrmr/fe/test_fe_multi_gpu.py::test_multi_gpu_matches_single_gpu`, explicitly injects "two heterogeneous profiles pointing at the SAME physical device 0" (its own docstring) — because no real multi-GPU CI hardware exists — so it structurally cannot exercise the cross-device cache-hit path this finding describes. Also compounding: `_FE_RESIDENT_OPERANDS` (a plain `OrderedDict`, `.get`/`.move_to_end`/`.popitem`) is read/evicted with no lock at all while `multi_gpu_fe_batch_mi`'s `ThreadPoolExecutor(max_workers=g)` calls into it concurrently from every device thread — the same unlocked-cache class flagged repeatedly elsewhere this audit wave (`X_SECURITY_API_PACKAGING-2`), but this specific cache was not in that cross-file inventory. | NEW — not in any of the 29 2026-07-20/2026-07-22 reports (`gpu_infra_a/b/c/d.md` covered `_fe_gpu_batch`'s sibling caches and `_gpu_resident_fe.py`'s operand tables but not `_fe_resident_operands.py` itself or this specific device-vs-content cache-key gap; `x_efficiency_architecture.md`'s cache inventory (X_EFFICIENCY_ARCHITECTURE-5 / X_SECURITY_API_PACKAGING-2) does not list `_FE_RESIDENT_OPERANDS`). |
| X_EDGE_CASES_BEST_PRACTICES-2 | P1 | bug (edge_case, silently-wrong-result) | `_orthogonal_univariate_fe/__init__.py:626-669` (`hybrid_orth_mi_fe`'s noise-aware absolute floor) | The Layer-27 "noise-aware floor" (median + `sigma_thresh * 1.4826 * MAD` over `raw_baselines`/`eng_mis`) — added specifically to stop an all-noise frame from flooding the top-K with false positives — is gated behind `if raw_baselines.size >= 4:` (line 648) / `if eng_mis.size >= 4:` (line 662); below that count both `noise_floor` and `eng_noise_floor` silently default to `0.0`, leaving `abs_floor = max(legacy_floor, 0.0, 0.0)`. `raw_baselines`/`eng_mis` have one entry per SCORED ENGINEERED COLUMN (`len(degrees) * len(surviving raw cols)`, roughly), not per raw source column — so with the default `degrees=(2, 3)` (2 degrees), the guard is bypassed whenever **at most one raw numeric source column** reaches the scan (2 engineered columns < 4). This is directly reachable in three realistic ways: (a) an input frame with genuinely only 1-2 numeric raw columns (e.g. mostly-categorical tabular data with one continuous sensor/measurement column — not an exotic shape); (b) the function's OWN `_dedup_collinear_source_cols` step (default ON, `dedup_corr_threshold=0.999`) collapsing many highly-correlated numeric raw columns down to 1 survivor on a wide, redundant-sensor frame; (c) a direct caller passing an explicit short `cols=` list (this function's own docstring frames direct-call usage as the primary, "NOT wired into MRMR.fit" entry point). When the sole surviving raw column's `baseline_mi` is itself ~0 (weak/irrelevant raw signal — the common case, not just a literally-constant column), `legacy_floor = min_abs_mi_frac * max_raw_baseline` ALSO collapses to ~0, so `abs_floor` is a full no-op; simultaneously `uplift = engineered_mi / (baseline_mi + 1e-12)` becomes enormous for any nonzero (including pure finite-sample noise) `engineered_mi`, trivially clearing the relative `min_uplift=1.05` gate too. Both of the function's OWN documented safety gates (the very ones its inline comments cite the "Layer 27 incident" — an all-noise frame flooding the top-K — to justify) are simultaneously defeated, and a purely noise-driven `He_n`/`T_n`/`L_n`/`L^Lag_n` basis column of the sole raw column is silently admitted into `winners`/`keep` and appended to `X_aug`, then (via `hybrid_orth_mi_fe_with_recipes`, the actual `MRMR.fit`-wired entry point, default `fe_univariate_basis_enable=True`) surfaces as a real `EngineeredRecipe`-backed candidate feature for MRMR's redundancy/relevance gates to reason about. No test in the suite constructs a <=1-numeric-raw-column fixture for this gate (the existing biz_value test for this stage, `test_biz_value_mrmr_univariate_basis_fe.py`, per `_fit_impl_core.py`'s own citation, targets the "univariate nonlinearity recovery" happy path, not this degenerate-pool-size interaction). | NEW — not raised by `mi_greedy_recipes.md` (covers `engineered_recipes/` proper, not this file), `x_efficiency_architecture.md` (which flagged the SAME file's stale "not wired" docstring and the GPU-resident semi-supervised gap, but not this noise-floor interaction), or the 2026-07-20 `edge_cases.md`/`fe_expansion.md` (neither names this function). |
| X_EDGE_CASES_BEST_PRACTICES-3 | P2 | edge_case (unbounded allocation, opt-in) | `_orthogonal_univariate_fe/_imbalance_mi.py:140-144` (`compute_class_weights`) | `n_classes = int(y_arr.max()) + 1` followed by `counts = np.bincount(y_arr, minlength=n_classes)` has no upper-bound sanity check on `y_arr.max()`. This whole module is bench-rejected and default-OFF (`MLFRAME_FE_IMBALANCE_MI` unset ⇒ `_imbalance_mode()` returns `"off"` and `compute_class_weights` short-circuits before reaching this code), so it is not reachable in any default configuration — but once a caller sets `MLFRAME_FE_IMBALANCE_MI=on`/`auto`, an integer-labelled `y` whose max value is large (e.g. a row-id or timestamp column mistakenly typed/passed as a classification target, or genuinely tens of millions of sparse integer "classes") allocates a `counts` array of that size — an attacker/caller-controlled unbounded allocation, the same class of gap `INFO_THEORY_A-2`/`INFO_THEORY_B-7` flagged for sibling dense-histogram builders elsewhere in the info-theory estimator zoo, here in a file none of those reports covered. | NEW — file not named in `info_theory_a.md`/`info_theory_b.md` (different directory) or any other 2026-07-20/2026-07-22 report. |
| X_EDGE_CASES_BEST_PRACTICES-4 | P2 | thread-safety (test-infra only) | `_gpu_strict_fe/_audit.py:47-105` (`residency_audit`) | `residency_audit()` monkeypatches process-wide, module-level `cp.asarray`/`cp.asnumpy`/`cp.ndarray.get` for the duration of its `with` block (lines 91-96) and restores the ORIGINAL functions on exit (lines 100-104) with no lock and no reentrancy/overlap guard. If two `residency_audit()` regions are ever active concurrently on different threads of the same process (e.g. a future parallelized test run, or a test helper invoked from a background thread), the SECOND region's `_orig_asarray = cp.asarray` capture is actually the FIRST region's wrapper (not the true original); whichever region exits FIRST restores `cp.asarray` to its own captured "original" — which, if that region entered second, is the other region's still-active wrapper, and if it entered first, is the true original — either way, the surviving/overlapping region's H2D/D2H byte tally silently stops matching reality for the remainder of its scope, with no error raised. Confined to the test/profiling harness only (the module's own docstring: "Intended for tests / profiling, not the production path"), so blast radius is a silently-wrong residency assertion in CI, not a production bug — but that is exactly the failure mode this harness exists to prevent, and the only existing test (`test_install_cuda_teardown_guard_is_idempotent`-style sequential pattern used elsewhere in this cluster) would not catch a genuine overlap. | NEW — `_audit.py` was not named in `gpu_infra_c.md`'s coverage of `_gpu_strict_fe` (that report examined `_entry.py`/`_state.py` for the stale-docstring finding only) or any other report. |
| X_EDGE_CASES_BEST_PRACTICES-5 | P2 | reproducibility (dead code) | `hermite_fe/_hermite_prewarp.py:449-453` (`_ksg_mi_1d`) | Hardcodes `random_state=42` in both `mutual_info_classif`/`mutual_info_regression` calls — sklearn's kNN-based MI estimator uses random tie-breaking noise, so this hardcodes the exact bug class flagged repeatedly elsewhere this audit wave (`MI_GREEDY_RECIPES-1`, `ORTH_BASIS_A-1`, `GPU_INFRA_D-3`: a hardcoded seed that ignores the estimator's `random_state`/`random_seed`) — except this instance is currently harmless because `_ksg_mi_1d` is dead code: re-exported from `hermite_fe/__init__.py:799` but never called anywhere else in `src/` or `tests/` (grep-confirmed, zero call sites beyond its own definition and the facade re-export). Flagged so a future author wiring this function in (e.g. as a "ksg" `mi_estimator` option for the ALS prewarp objective, which the surrounding module clearly anticipates) does not silently reintroduce the already-fixed bug class. | NEW (dead-code + latent-bug-class observation; not raised by `orth_basis_a.md`, which reviewed this same file for other findings). |
| X_EDGE_CASES_BEST_PRACTICES-6 | P2 (edge_case, currently unreachable) | edge_case | `_fe_gpu_batch/_packer.py:67-77` (`pack_blocks_to_devices`) | `if len(speeds) <= 1: return [0] * len(works)` treats a genuinely EMPTY `speeds` list (`len(speeds) == 0`, i.e. zero visible devices) identically to the single-device case, returning device index `0` for every block — misleading (there is no device 0) though currently unreachable: the sole caller, `multi_gpu_fe_batch_mi`, already special-cases `len(profs) <= 1` (falling back to `gpu_fe_batch_mi` with `device=None`) BEFORE ever calling this function, so `pack_blocks_to_devices` is only ever invoked with `len(speeds) >= 2` in production. A future direct caller (or a refactor that removes the caller-side guard) would silently get a bogus device-0 assignment instead of a clear error for the zero-device case. | NEW (defensive-gap observation on a reusable pure function; not reachable via any current call path). |

## Confirmed non-findings (investigated, no issue)

- **`random_seed`/`random_state` aliasing across the FE pipeline** (the specific angle this brief calls out by name): a
  repo-wide grep found ~65 sites across `_mrmr_fit_impl/_fit_impl_core.py`, `_mrmr_fe_step/*.py`, `_fe_auto_escalation.py`
  reading `getattr(self, "random_seed", 0)` rather than the canonical `self._effective_random_seed()`. Traced to
  `mrmr/_mrmr_class.py:3711-3716`: `_fit_body` computes `_eff_seed = self._effective_random_seed()` and, if it differs
  from the raw ctor attribute, temporarily **overwrites** `self.random_seed = _eff_seed` for the scope of the fit body,
  restoring the original value in a `finally` at lines 3910-3912. Every one of the ~65 sites executes strictly inside
  this scoped-override window, so a user who sets only `random_state=` (never the legacy `random_seed=`) DOES get a
  reproducible, resolved seed threaded through every FE-stage subsample/bootstrap/permutation-null draw. This is a
  deliberate, well-documented, comprehensive mechanism — not a bug. (This lead was pursued specifically because it
  looked, on first grep, like exactly the class of bug the task brief asks to hunt for; recorded here per the "always
  flag alternative readings" / "no hand-wave hypotheses as verdicts" conventions rather than silently dropped.)
- **Mixed pandas/polars/numpy input to the recipe-replay path**: `engineered_recipes/_recipe_extract._extract_column`
  explicitly dispatches on `isinstance(X, pd.DataFrame)` / `isinstance(X, pl.DataFrame)` / `isinstance(X, np.ndarray)`
  (structured-array only) and raises a clear `TypeError`/`KeyError` naming the actual type for anything else — a
  fail-fast, not a silent-wrong-result path. Confirmed clean.
- **`EngineeredRecipe`'s frozen-dataclass-with-mutable-`extra`-dict hazard** (a classic Python footgun: a `dict`/`ndarray`
  field on a `frozen=True` dataclass can still be mutated in place, and mixing a narrow `__hash__` with a wide `__eq__`
  can violate the hash/eq contract for dict/set-key use): `_recipe_core.py`'s `__post_init__` deep-copies `extra`,
  freezes every owned+writable ndarray inside it (`.flags.writeable = False`), and wraps it in `MappingProxyType`; the
  `__hash__` docstring explicitly documents and accepts the narrower-than-`__eq__` hash contract and states the
  concrete caller discipline required (never use a recipe instance itself as a dict/set key when the same `name` can
  carry different `extra` — store by `recipe.name` instead) — already correctly handled, not a fresh finding.
- **Constant/all-NaN/single-unique-value source columns feeding the orth-basis preprocessors**: traced all four
  `_POLY_BASES` fit functions (`_preprocess_zscore`/`_minmax_neg1_1`/`_shift_nonneg` in `hermite_fe/__init__.py`) against
  a literally-constant column — all three degrade to a well-defined constant output (`z=0` / `z=-1` / `z=1e-9`
  respectively, via each function's own `+1e-12` epsilon guard on the scale denominator) with no NaN/Inf ever produced;
  the resulting engineered column is itself constant, so `MI(constant; y) == 0` naturally fails every downstream
  relevance gate. Confirmed safe (this is what pointed toward X_EDGE_CASES_BEST_PRACTICES-2's REAL failure mode — a
  *weakly-informative*, not literally-constant, sole raw column bypassing the noise floor via candidate-COUNT rather
  than candidate-VALUE degeneracy).
- **`_orth_scoring_memo.py` / `_orth_dedup.py` fit-scoped memo caches**: both use `threading.local()` (not a bare module
  global), explicitly documented as "no cross-worker contamination"; confirmed this holds — no shared-mutable-state
  risk under concurrent `.fit()` calls from separate threads, unlike the module-level `OrderedDict` caches flagged
  elsewhere this wave.
- **Extreme class imbalance (rare-class MI estimation)**: `_imbalance_mi.py`'s two-sided gate (`_PRIOR_THRESHOLD=0.30`,
  `_N_RARE_FLOOR=150`) is a deliberately bench-rejected, default-OFF correction (documented net-negative on downstream
  rare-class recall/AP across 120 imbalanced-frame trials) — its own module docstring is an unusually rigorous
  "we tried this and it doesn't help" writeup; no action item beyond X_EDGE_CASES_BEST_PRACTICES-3's unrelated
  allocation-size gap.
- **Documentation staleness** (`_orthogonal_univariate_fe/__init__.py:37-40`'s "NOT wired into MRMR.fit by default"
  claim): independently re-confirmed false (the univariate path is `fe_univariate_basis_enable=True` by default,
  default-on since at least the 2026-06-02 comment in `_fit_impl_core.py`) — but this is the exact claim
  `x_efficiency_architecture.md`'s X_EFFICIENCY_ARCHITECTURE-4 already reports; not re-counted as a new finding here.

## Proposals

1. **Fix X_EDGE_CASES_BEST_PRACTICES-1** by folding the active device index into `resident_operand`'s cache key
   (e.g. `sig = (cp.cuda.Device().id, host.shape, host.dtype.str, _content_hash(host))`), or — cheaper and consistent
   with the codebase's own already-correct pattern — route `_fe_gpu_batch/_executor.py`'s multi-device path through
   `_gpu_strict_fe/_state.py`'s `ResidentFEState` (already device-keyed, currently dead code) instead of the
   content-only `_fe_resident_operands.resident_operand`. Either way, add a `threading.Lock` around
   `_FE_RESIDENT_OPERANDS`'s get/evict/write sequence. Ship a regression test that fakes 2 `DeviceProfile`s pointing at
   2 DIFFERENT real devices when CUDA reports >=2 (skip otherwise), or — runnable on any single-GPU CI box —
   monkeypatch `cp.cuda.Device` to assert `resident_operand` is never called from two different simulated device
   contexts without a fresh upload; at minimum, promote `fe_batch_mi`'s bare `except Exception: pass` (line 91-92) to
   `logger.warning(..., exc_info=True)` so a real multi-GPU deployment doesn't silently lose its whole speedup with
   zero trace in the logs.
2. **Fix X_EDGE_CASES_BEST_PRACTICES-2** by computing the noise-floor guard's activation threshold from the number of
   distinct RAW source columns actually scanned (a quantity already available before the scoring call), not from
   `len(scores)`/`len(eng_mis)` — e.g. skip straight to a `min_abs_mi_frac`-only floor with an explicit `logger.debug`
   warning when fewer than ~4 raw columns are being expanded, or (more robust) always compute the noise floor from
   whatever rows ARE available (median/MAD are well-defined even at n=2-3, just noisier) rather than hard-gating on
   `size>=4`. Add a regression test mirroring the existing Layer-27 all-noise fixture but with exactly 1 raw numeric
   column (2 engineered columns), asserting the noise-driven engineered column is NOT admitted when its source
   baseline MI is near zero.
3. Add an explicit upper-bound check (or clamp to `int32`/require `y.max() < some documented ceiling`) before
   `np.bincount(y_arr, minlength=n_classes)` in `_imbalance_mi.compute_class_weights` (X_EDGE_CASES_BEST_PRACTICES-3) —
   cheap, and this opt-in module already has excellent test coverage (`tests/feature_selection/test_imbalance_mi.py`)
   to extend.
4. Add a `threading.Lock` (or a reentrancy counter) around `residency_audit()`'s monkeypatch/restore pair
   (X_EDGE_CASES_BEST_PRACTICES-4) so two overlapping audit regions on different threads can't silently corrupt each
   other's byte tally; low priority given its test-only scope.
5. When `_ksg_mi_1d` (X_EDGE_CASES_BEST_PRACTICES-5) is ever wired into a real `mi_estimator="ksg"` prewarp path, thread
   a real seed through instead of the hardcoded `42` — flagging now, before it happens, is cheaper than finding it
   after a user reports non-reproducible KSG-mode fits.
6. Give `pack_blocks_to_devices` (X_EDGE_CASES_BEST_PRACTICES-6) an explicit `if not speeds: raise ValueError(...)`
   ahead of the `len(speeds) <= 1` fast path, so a future direct caller with zero visible devices fails loudly instead
   of silently addressing a nonexistent device 0.
