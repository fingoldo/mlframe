# Wave 4 audit 2026-11-02: GitHub Actions injection and CI privilege

Scope: `.github/workflows` (no local composite actions exist), read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| CI5-1 | Low-Medium | `codecov-full.yml:144-145, 180, 207-216` with the producer at `ci.yml:430-434` | RESOLVED |
| CI5-2 | Info | `codecov-full.yml:94-95` (numba-coverage run lookup has no event filter) | RESOLVED |
| CI5-3 | Info | `ci.yml:551-559` (test-durations commit job) | RESOLVED |
| CI5-4 | Info | `macos-abort-probe.yml:146` (`ref: inputs.ref || github.ref`) | RESOLVED |

CI5-1: `gh run list --branch master` also returns a fork's `pull_request` run whose head branch is called `master`, and `ci.yml` uploads coverage artefacts on PR runs, so `codecov-full` (which holds `CODECOV_TOKEN`) could combine and upload a fork's coverage. No code execution (SQLite data read by `coverage`), but the badge could be forged. Fixed by filtering the lookups to `--event push` and by gating the artefact upload on a non-PR event.
No `pull_request_target`, no `actions/github-script`, no `${{ github.event.* }}` or `github.head_ref` inside a `run:` block anywhere. Inputs reach shells through `env:`. Every workflow has top-level or job-level `permissions:`.
