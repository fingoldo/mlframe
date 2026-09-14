# mrmr_audit_2026-09-14 — audit brief (shared by all cluster agents)

Fresh read-only audit of the MRMR module surface (77 files, ~31 440 LOC) against git `57f649fb6`.
Prior waves: `audits/mrmr_audit_2026-07-20/`, `-07-22/`, `-07-25/`. The -07-25 tracker lists its findings
as DONE; part of this wave's job is to confirm whether the ones touching your cluster actually hold in the
CURRENT source, and to find what has regressed or been introduced since.

The owner asked for three things explicitly: **critique the code**, **check algorithmic and code
efficiency**, and **propose additional tests**. Every cluster doc must deliver all three for its files.

## What to check (ALL aspects, for every file in your assigned cluster)

1. **Correctness bugs** — logic errors, off-by-one, wrong default, silent wrong-result, dtype/overflow,
   NaN/inf handling, empty/degenerate/constant/all-null inputs, ragged-shape misalignment, seed and
   determinism, sklearn contract (`clone` / `get_params` / `set_params` / `__getstate__` / `__setstate__`),
   pickle round-trip, `transform()` replaying frozen fit params rather than refitting (leak/drift).
2. **Algorithmic soundness** — is the mRMR/JMIM/relevance/redundancy math actually what the docstring
   claims? Estimator bias (in-screen "winner's curse" vs honest holdout), MI estimator assumptions,
   discretization interacting with the score, tie-breaking, monotonicity invariants, whether a documented
   selection guarantee is really enforced by the code.
3. **Numerical stability** — this repo has a REPEATEDLY CONFIRMED catastrophic-cancellation bug class:
   any `sum(x^k)` minus a power of the mean, for **any k >= 2** (not just skew/kurt — `var = E[x²] − E[x]²`
   is the same bug at k=2). Also additive epsilon padding in a denominator that is merely *small* rather
   than zero (corrupts by 30–100 % with zero cancellation error). Flag every site; say which regime breaks it.
4. **Silent failure / fallback quality** — this repo has TWICE shipped a broad `except` that silently
   downgraded the MI backend ~100× with only a `debug` log. Any handler that substitutes a value and
   returns without logging above `debug` is a finding; so is a fallback whose substituted value is
   non-neutral in the direction that disables the check it feeds.
5. **Performance** — wasted per-call work in hot loops, un-hoisted dispatch decisions, recomputation across
   iterations, whole-frame `.copy()`/reconstruct, per-candidate re-upload, and the repo's signature win
   shape: a Python-level loop calling an already-`njit` kernel once per iteration (fuse the WHOLE
   per-iteration body into one `njit(parallel=True)` + `prange` call instead).
   **Before calling anything "already optimal", you MUST grep it for `@njit`, `parallel=True`/`prange`,
   `cuda.jit`, `cupy`, and `KernelTuningCache`/`get_or_tune`** — a REJECT is only valid if one of those is
   confirmed present AND covers that path, or a documented bench-attempt-rejected note tested THIS lever.
   Never claim a speedup number you did not measure; propose a concrete bench plan instead.
6. **Memory** — frames can be 100+ GB. Whole-frame copies, unbounded caches, pickling large arrays,
   eager format conversion without a byte-size gate.
7. **Concurrency** — unlocked module-level caches, thread-local leaks on mid-block raise, thread-locals not
   crossing the joblib worker boundary, shared mutable state across parallel fits.
8. **Test gaps** — for every behaviour you could not find a test for, propose a NAMED test
   (`test_<failure_mode>`), say what it asserts and why a regression would slip past today's suite.
   Prefer mutation-resistant assertions on real values over `assert x is not None`. Flag any test that
   asserts on `inspect.getsource()` text rather than behaviour.
9. **Code quality / critique** — dead code, misleading names, docstrings that contradict the code,
   duplicated logic that should be one primitive, files near/over the 1000-LOC budget, comments carrying
   audit/process metadata (against this repo's comment rules).

## Ground rules

- **READ-ONLY.** Do not edit, fix, or reformat any source file. Your only write is your own cluster doc.
- **Zero hallucination.** Every finding needs `file:line` and a quoted or precisely described code fact.
  If you are not sure, say "unverified" and say what would settle it. A wrong finding costs more than a
  missing one — the last wave's own retro recorded four separate "took a number without checking what it
  counts" mistakes.
- **Report EVERY finding, including Low/P3.** Do not filter by perceived importance, do not summarise a
  category away, do not drop something because it looks pre-existing.
- Do NOT run the full test suite. Targeted reads, greps, and at most a couple of tiny targeted checks.
- Severity: **P0** (wrong results / data loss shipped), **P1** (real bug or live gate failure),
  **P2** (correctness gap / meaningful inefficiency), **P3** (quality / nit). Tag each finding
  `<CLUSTER>-<n>`.

## Output format (write to `audits/mrmr_audit_2026-09-14/<your-cluster>.md`)

```
# <cluster> — mrmr_audit_2026-09-14

## Scope
<files reviewed, with LOC>

## Findings

### <CLUSTER>-1 — <one-line summary>  [P1]
**Where:** `path/file.py:120-134`
**What:** <the code fact>
**Why it is wrong / costly:** <mechanism, and the regime where it bites>
**Fix:** <concrete proposal>
**Test:** `test_<failure_mode>` — <what it asserts>

### ...

## Proposed tests (beyond the per-finding ones)
<named tests for untested invariants>

## Prior-wave findings touching this cluster
<ID, whether it still holds in current source>

## Verified-clean
<what you checked and found genuinely fine — so the next wave does not re-audit it blind>
```

Only `.md` files go in this directory — no `.py`, no scratch scripts.
