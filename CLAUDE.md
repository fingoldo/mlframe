# mlframe — project conventions

## No project-wide lint/format rewrite without approval (CRITICAL)
Never run a repo-wide `black .` / `ruff format .` / `ruff check --fix` beyond files already being edited — even when asked to "fix everything"; that's consent for the specific fixes, not a mechanical rewrite. Report scope ("N/M files need reformatting") and ask: run now / make the gate advisory / defer. Narrow fixes to files already being edited for a diagnosed reason are fine without asking.
**Why:** a prior unscoped run once rewrote a huge fraction of a repo unasked while the user was away; reverted.
**Excluded Black behaviors:** arg/collection explosion (multi-item one-line → one-per-line, incl. `from x import (...)`) and blank-line insertion — neither configurable via stock Black flags. Enforced via the shared `py_ci_shared.black_filtered_apply` (`--write`/`--check`) — never raw `black`.

## RUF100 unsafe under a narrow `--select` (CRITICAL)
`ruff check --select <narrow> --fix` makes RUF100 judge `# noqa` comments against only the narrow ruleset and silently strip ones load-bearing for the full config (e.g. star-import re-export markers). Never `--fix` with `--select` narrower than the full configured set. To triage one new rule category: list findings with `--select`, fix by hand. `--fix` only with no `--select` override, on files just edited, verified against the real blocking gate (`ruff check . --ignore C901`) afterward. `--ignore` is safe (adds to the ignore list); `--select` is not (replaces the rule set).

## Never pause on obvious/low-stakes questions (CRITICAL)
Don't ask about execution order or "should I also do the deeper fix" — pick the sensible default and do all the work, report after. Pause only for: a genuine accuracy/functionality tradeoff, a destructive/irreversible op, another session's uncommitted WIP, or a hard blocker. Never end a turn with authorized work still pending — a status update must be followed by more work in the same turn, not be the final act.

## Drive every discovery to resolution (CRITICAL)
A bug/gap found while doing other work is a commitment: fix now, or add a concrete next action to the active plan and finish it the same session. Never drop with "context running out" / "not the main goal" / "pre-existing" — pre-existing bugs still get fixed once found. Closure = fixed+tested, or a written plan item; never a hand-wave.

## New code goes in focused submodules from the start (CRITICAL)
Put new non-trivial functionality in a new, well-named sibling module and re-export from the parent facade — don't keep appending to an already-large file. Carve *before* a file nears ~800-900 LOC; `test_no_file_over_1k_loc.py` is a backstop, not the design.

## Prefer top performance in hot paths (CRITICAL)
Take the higher-performance, more complex option over a "safe but partial" one when it's validated rigorously (bit-identity, GPU/CPU parity, selection-equivalence, full suite) — never ship an *unvalidated* selection change, but do ship validated complexity.

## Enable corrective mechanisms by default (CRITICAL)
When a fix closes a real bug class, flip its default ON in the same change — don't keep the old (wrong) behavior for "compatibility". If existing tests assumed the old wrong behavior, fix the tests. Document the flip and keep an explicit opt-out for legacy callers.

## Fuzz/combo tests are bug DETECTORS, not bug hiders (CRITICAL)
Every fuzz-suite failure is a real prod bug unless proven otherwise. Never paper over with: canonicalization rules that collapse combos "because they crash" (only collapse genuinely semantically-equivalent combos), runtime canonicalizations for the same reason, `xfail`/`skip` for anything but a genuine third-party/OS limitation, or a defensive prod guard hiding an upstream bug. Ask "would a real user with these settings hit this?" — if yes, fix the root cause (often retires several band-aids at once).

## Memory / RAM discipline (CRITICAL)
Frames can be 100+ GB — never `.copy()`/`.clone()`/reconstruct a frame to work around a bug; mutate-and-restore (try/finally) or use views.
- Caching: only cheap-to-rebuild artifacts, or the caller's own reference plus a content-hash key — never a whole-frame copy. Never pickle a whole frame for caching.
- Batching: stream/mini-batch when per-element cost is small; don't materialize a full frame when batching suffices.
- Eager format conversion (e.g. polars→torch): gate on byte size (~2GB — eager under, lazy/per-batch over; unknown-size defaults to eager).
- Frame-format conversions (polars↔pandas↔ndarray) are the CALLER's decision, made once at the suite boundary — inner wrappers must never silently down-convert on a hot path.

## `/loop` fuzz-profile-optimize: stop after 100 consecutive rejects (CRITICAL)
Track a REJECT streak (reset on any RESOLVED), not a fixed iteration cap — stop only after 100 in a row.

## REJECTED ≠ DELETED (CRITICAL)
A rejected optimization keeps: the committed bench/prototype (runnable), a tracker row with exact numbers + bench filename, the option itself in prod if it's a tunable (only the *default* stays unchanged), and a `# bench-attempt-rejected` note at any touched call site. Never silent-revert or silent-delete.

## A validated improvement that breaks a test → re-frame the stale test (CRITICAL)
When a real improvement conflicts with a test (especially test-vs-test contradictions), don't default to reverting: bisect to the exact assertion, confirm the real contract still holds (no regression, equal-or-better), look for the codebase already endorsing the new behavior, then re-frame the assertion to the real contract with a comment citing the evidence. Revert only if the new behavior genuinely regresses *and* can't be gated. Diagnose WHAT the assertion now sees, not WHEN it broke — never bisect commits just to assign blame.

## Profile every new feature; optimize hotspots (CRITICAL)
cProfile harness saved in-package (not `/tmp`), sorted by cumtime top 20-30, optimize where it materially helps (njit / prange / cuda.jit / cupy / vectorize / cache), calibrate against cProfile's pandas/sklearn attribution inflation (~10-13x vs standalone wall-time), document "no actionable speedup" with reasoning when that's the conclusion.

## A measured speedup is a LEAD, never dismiss on one hasty look (CRITICAL)
Warm the kernel, run multiple times, sweep the size range (small AND large — crossovers are the whole story), validate end-to-end not just isolated. A hardware-relative win (weak dev GPU making it look neutral) still gets kept as a size/env-gated option, never deleted — promote it to `kernel_tuning_cache` so each host finds its own crossover. Only a written, multi-size, end-to-end measurement is a valid rejection — "felt marginal" is a skipped investigation. GPU kernels: always measure host-input, GPU-resident-input, and GPU-with-H2D separately — a kernel that wins big resident but loses end-to-end needs residency fixed, not the backend reverted.

## A/B validation procedure (CRITICAL)
Warm before timing; best-of-N/median, never one shot; measure BOTH isolated and end-to-end (an isolated win that's flat e2e is a REJECT — find the size where it nets positive, or reject); real baseline via `git show`/a separate process, never from memory; paired/interleaved trials on a noisy box (or switch to `process_time`); separate-process A/B when in-process state (numba cache, warmed JIT, module globals) could contaminate; identity gate alongside speed always — bit-identical, or a documented ~1e-9 FP-reorder delta proven not to move a decision (FE/MRMR exception: selection-equivalence is the bar, not bit-identical MI); cProfile mis-attributes compiled-kernel time to the Python caller frame — microbench the wrapper standalone before trusting a flagged frame.

## GPU profiling traps (CRITICAL)
nvprof per-kernel % is sync-distorted — never pick a target from it; use cProfile cumtime + a full-fit wall A/B. cProfile is blind to async GPU compute; it shows up as tottime at the blocking sync (`.get()`/`copy_to_host`). Isolated-kernel or wrong-shape microbenches lie end-to-end — confirm on the real `(n, K)` shape and the full-fit wall. `nvprof --print-gpu-summary` needs no admin; `--metrics`/`--events` do. Quiet the machine (`nvidia-smi`) and use novel-seed cold fits (MRMR memoizes fits by content-hash). A single-kernel launch-batch tweak is usually 0x on the wall, but fusing away the aggregate launch COUNT pipeline-wide is a real win — always confirm via the full-fit wall and report both launch count and wall.

## joblib threading over GPU-bound work = contention, not parallelism (CRITICAL)
Gate parallel-vs-serial on whether the stage is GPU-bound: GPU stage → serial main-thread (CPU kernels still `prange` internally); CPU-only stage → keep threading. Diagnose via cProfile `print_callers("time.sleep")` pointing at joblib `_retrieve`.

## Never kill a near-done background agent to "free the machine" (CRITICAL)
Paired A/B timing already cancels for machine load — no benchmark reason to serialize by killing. Only stop an agent that's genuinely hung, confirmed on a wrong/superseded path, or editing a file another agent must own — and let it finish emitting its report first.

## Audit hot kernels for wasted per-call work (CRITICAL)
Once a kernel is hot by tottime *and* call count, check every call site: does the caller use the FULL output, or discard part? If it discards, write a pruned fast-path variant for that caller (keep the full kernel for others) — bit-identical by construction, biggest win at the hottest sites. "Converged" isn't a valid verdict until call sites have been audited for discarded work.

## Gate a big win on its safe condition (CRITICAL)
When a speedup is only bit-identical under a detectable predicate, don't reject wholesale or ship unconditionally — gate the fast path to the safe case, exact path elsewhere. ~1e-9 FP-reorder divergence is fine; ~1e-3 selection-altering divergence never ships unconditionally. Verify bit-identity on the UNSAFE case explicitly (tied/discrete data). Ship a test pinning both sides.

## Every ML trick gets a quantitative `biz_value` test (CRITICAL)
A synthetic where the trick should clearly win; threshold set 5-15% below the measured value; compared against the closest baseline; each test <5s. Bad: `assert res is not None`. Good: `assert res.mi >= 0.55`. Naming: `tests/<pkg>/test_biz_val_<class>.py`, one file per class, functions `test_biz_val_<class>_<param>_<scenario>`. Skip for pure refactors, trivial helpers, crash-fix regression tests, docs-only changes.

## Numerical-kernel acceleration ladder
Backends in priority order: numpy/scipy (baseline) → `numba.njit` (wins n≈100-50k) → `njit(parallel=True)+prange` (wins n≈50k-500k; spawn overhead loses at small n) → CUDA (wins n≥500k once transfer is amortized — ranked `cp.RawKernel` > cupy elementwise > `numba.cuda.jit`, the last measured 6-10x slower than RawKernel on this hardware).
- The fastest applicable path MUST be the public API's default — a dispatcher picks the backend; each `_backend` variant stays directly callable; keep a `force_backend=`/env-var escape hatch. Never ship a `_gpu`/`_cuda` name callers must manually wire in.
- Bench all four backends across n∈{500,2k,10k,100k,1M} *before* writing the dispatcher; save to `_benchmarks/`.
- New GPU dispatchers integrate with `pyutilz.system.kernel_tuning_cache` (measured, per-hardware thresholds) — never hardcode a threshold constant.
- Hoist the dispatch decision out of hot loops (~4us/call overhead adds up).
- Skip the whole ladder for kernels called <100x/fit or already <1% of wall.
- fastmath: a full `fastmath=True` kernel + a Python-level `np.isfinite(arr).all()` gate beats a hand-picked partial fastmath flag set (selective fastmath still blocks the SIMD reduction — measured ~14% slower).

## Accuracy/performance over legacy/compat/deps (CRITICAL)
Default knobs flip to the new path once it measurably wins — no feature-flag-for-safety. Extra optional deps are fine for a real speedup/accuracy win. Tighten loose test tolerances to match the new path's actual precision. Do the wide multi-seed benchmark now, don't ship "validated on one fixture" with a follow-up TODO. Variant defaults: most-accurate-on-the-honest-metric first, speed only breaks ties within noise — profile/speed-up the accurate variant before conceding to a faster-but-worse default; a single-seed win doesn't count.

## Every feature: unit + biz_value + cProfile, in order (CRITICAL)
Required for every non-trivial feature/param/branch. Skip clauses: bug fixes (a regression test suffices), default-flip-only changes (existing biz_value tests must still pass), test-infra additions.

## Every bug fix ships a regression test, same commit (CRITICAL)
Reuse the real fixture that surfaced the bug where possible. Verify empirically: fails on pre-fix code (temporarily revert just the fix, run, confirm the real failure signature, restore), passes post-fix. One narrowly-named test per bug (`test_<failure_mode>`, not `test_thing_works`). Applies equally to bugs you introduced, pre-existing bugs you found, and fuzz-caught combos (promote to a named unit test).

## Multi-agent review: every finding gets an explicit disposition (CRITICAL)
RESOLVED / FUTURE (with reason) / DOC / REJECTED (with reason) — never "ignored", "low priority", or silent omission, regardless of N. One running disposition table across review rounds; mark cross-agent duplicates explicitly. PR descriptions include the full rollup.

## polars FE path is already optimal — don't "fix" it again (CRITICAL)
`MRMR.fit` already bridges polars → a zero-copy Arrow-backed pandas view (`get_pandas_view_of_polars_df`) whenever FE runs — no whole-frame copy at any size. Measured: one contiguous plane beats per-column zero-copy views 8.65x at equal memory, so a "native, zero-copy, per-column" rewrite would be strictly worse. Keep the bridge; the format-agnostic seam (`_fe_frame_ops`) is a fallback path, not the fast one.

## Comment style (CRITICAL — repeated complaints)
Line length up to 160 chars — don't hard-wrap at 72-80; one sentence/clause per line up to the limit.
No process/audit metadata in code comments: no phase/wave markers, finding IDs, date stamps, fuzz-seed refs, refactor-history narration ("was 4 star imports, now explicit"), banner separators — that belongs in git history / the PR description.
Default to minimalist comments: write one only when the WHY is non-obvious (hidden constraint, subtle invariant, workaround, surprising behavior) — never restate WHAT the code does or narrate process. No AI-justifying parentheticals ("(idiomatic)", "(elegant)").

## Monolith split: AST-audit the sibling for unresolved names before commit (CRITICAL)
A moved function/class imports clean but can `NameError` at first call if it references a parent-module name with no matching import (name lookup is lazy). Gate: AST-walk every `Load`-context `Name` in the new sibling, flag anything not bound locally/builtin/closure, grep the parent for its home, add an explicit (or lazy, if cycle-prone) import. Smoke-import + `hasattr` isn't proof — exercise a real call path. Any "name X is not defined" WARN in logs is a P0 sibling-split regression; audit every sibling for the same bug.
Flat sibling (`name.py` + `name_helpers.py`) for a single split; convert to a subpackage (`name/__init__.py` + submodules) once a monolith fans out into 2+ siblings — backward-compatible, same AST gate per submodule.

## Never use destructive git to inspect state (CRITICAL)
Multiple parallel agent sessions share this working tree. Banned even "just to peek": `git stash`, `git checkout -- <path>`, `git checkout <ref> -- <path>`, `git reset --hard`, `git restore`, `git worktree remove`. Use instead: `git show <ref>:<path>`, `git diff <ref> -- <path>`, `git log -p`/`git log -S`, `git worktree add <tmpdir> <ref>`. Don't bother distinguishing "pre-existing vs introduced" — just fix what a linter/test surfaces, regardless of origin.
Never `git reset --hard` / `checkout -B` / `branch -f` / `push --force` on a shared branch — can silently discard another session's committed work. To sync: `git fetch` + `git merge` (never reset). Before any history-moving command, check `git log --oneline @{u}..HEAD` — non-empty means a reset would destroy real commits. Recovery: `git reflog` → `git branch recovery-<name> <sha>`, cherry-pick back, surface to the user.
A broad/dir-wide `ruff`/auto-fixer pass is read-only `check`, never `--fix` — except on files just edited, after a commit exists.
Never kill processes without explicit user authorization for that specific action.

## Test pollution: never rebind a module without snapshot/restore (CRITICAL)
`del sys.modules[...]` / `importlib.reload()` on an mlframe module splits class identity between already-loaded and newly-loaded references — breaks class-attribute caches, `isinstance` checks, and idempotent-install markers in unrelated later tests. Don't reach for these to "force a fresh import" — use `monkeypatch`/explicit params instead. If unavoidable, snapshot+restore via an autouse fixture scoped to the affected module prefixes. `reload()` is less destructive than del+reimport but still resets module-level singletons. For true isolation, use a subprocess. An intermittent `isinstance` failure is the cheapest tripwire for a reload polluter upstream.

## Write mypy-clean code from the start (CRITICAL)
Never `param: T = None` (always `Optional[T] = None`); match return annotations to the actual `return` statements; annotate dict/list literals whose later use needs a wider type than line-1 inference; wrap numpy/pandas arithmetic chains in the concrete constructor matching the declared return type; concrete types over `object` for params that only ever hold one concrete class; declare dynamically-set attributes at class scope; never silence an error with pointless extra wrapping — understand the real runtime type, annotate correctly or `cast`. Run mypy on any touched file before calling it done.

## No hand-waving "time constraints" (CRITICAL)
There's no time limit on a turn. If a shallow fix or an unexplored gap is being justified with "given the time/budget constraints," stop and do the full fix instead — trace the value to its ROOT cause via actual runtime testing, don't assume from one read.
