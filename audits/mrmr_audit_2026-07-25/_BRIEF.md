# mrmr_audit_2026-07-25 — audit brief (shared by all cluster agents)

New round of the MRMR-module audit. The prior wave (`audits/mrmr_audit_2026-07-22/`) fixed 82 confirmed
P0/P1 findings plus many P2/P3; since then several sessions have landed further changes (a `group_aware_mi`
fix wave — FE-engineered features, `fe_max_steps=0` gating, and the raw-redundancy no-harm REVERT now all
consult group-aware relevance; a repo-wide audit-metadata comment cleanup across ~178 filters files;
baseline-debt broad-except logging waves; game-theory valuation branches). This round re-audits with fresh
eyes against the CURRENT code at git `d8091a138`.

## What to check (ALL aspects, for every file in your assigned cluster)

1. **Correctness bugs** — logic errors, off-by-one, wrong default, silent wrong-result, dtype/overflow
   (e.g. narrow `quantization_dtype` truncating widened bin codes), NaN/inf handling, empty/degenerate
   inputs, ragged-shape misalignment, seed/determinism, sklearn contract (`clone`/`get_params`/`set_params`/
   `__getstate__`/`__setstate__`), pickle round-trip.
2. **CPU/GPU parity** — any kernel with a CPU twin and a cupy/numba-cuda twin: do they produce
   bit-identical (or selection-equivalent) results? Fallback paths on device error? Residency/H2D/D2H.
3. **Concurrency / thread-safety** — unlocked module-level caches, thread-local leaks on mid-block raise,
   joblib worker republish of thread-locals, shared mutable state across parallel fits.
4. **Performance** — hot loops doing wasted per-call work, un-hoisted dispatch decisions, re-computation
   across iterations, `.copy()`/reconstruct of large frames, missing njit/prange/vectorization
   opportunities on measured hotspots. (Flag as a proposal with a concrete bench plan, do NOT assume a
   speedup without measurement.)
5. **Memory** — whole-frame copies, unbounded caches, pickling large arrays, per-candidate re-upload.
6. **Security / API** — injection surfaces (there should be none: no SQL/HTTP/eval/exec/subprocess on
   untrusted input), env-var handling, deserialization of untrusted state.
7. **Test coverage** — modules/functions with ZERO test references, gaps where a specific bug path is
   untested, fuzz/combo gaps, biz_value gaps for ML tricks.
8. **House conventions** (`CLAUDE.md`) — mypy-cleanliness, comment style (NO leftover finding-ID/date-stamp/
   audit-filename metadata in comments — the prior cleanup may have missed some), file-over-1k-LOC, monolith
   split AST-name-resolution hazards, `param: T = None` implicit-Optional, `--` in prose.

## Severity & output

- Severity: **P0** (crash / data corruption / silent wrong selection for a normal caller), **P1** (real
  bug hit under a plausible non-default config, or a genuine correctness/parity gap), **P2** (latent/fragile,
  edge-case, or quality), **P3/Low** (cosmetic, doc, dead-store, house-convention).
- The user has committed to fixing **EVERY** finding including P3/Low, so surface everything — but every
  finding MUST be concrete and verifiable: exact `file:line`, a precise mechanism, and (for a bug) a
  reproduction or a failure scenario. No hand-wave hypotheses; if you can't state the exact trigger, mark
  it a Proposal, not a finding.
- Write your findings to `audits/mrmr_audit_2026-07-25/<your_cluster_slug>.md` using this table format:

```
# <Cluster name> — audit (2026-07-25)

<1-paragraph scope: which files, what they do.>

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| SLUG-1 | P1 | bug | `file.py:123` | one-line claim | concrete inputs -> wrong output |

## Non-findings / confirmed-clean angles
- <what you checked and found clean, so the next auditor doesn't re-check>

## Proposals (perf / refactor / test — not bugs)
1. <proposal with a concrete bench/test plan>
```

- Use an ID prefix = your cluster slug in SCREAMING_SNAKE (e.g. `FIT_IMPL-1`, `GPU_RESIDENT-3`).
- Do NOT modify any production code or test — you are read-only. Your ONLY write is your findings `.md`.
- Verify each claim against the ACTUAL current source (read the real lines; don't trust prior-audit line
  numbers — the files have been re-split/edited since). Run `mypy`/`ruff`/`grep` read-only as needed.
