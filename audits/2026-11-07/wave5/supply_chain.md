# Supply chain audit, wave 5 (mlframe, 2026-11-07, HEAD 753d61baf)

Read-only. SUP-01..15 (audits/2026-10-31) and W4-SUP-01/02 (audits/2026-11-06/wave4) not re-reported; owner decisions on W4-SUP-01 (keep ruff tag pin) and W4-SUP-02 (pre-commit floating) not re-opened. Live PyPI/GitHub checks unavailable.

## Findings

| ID | Severity | File:line | Failure scenario | Disposition |
|---|---|---|---|---|
| W5-SUP-01 | Low | .github/workflows/release.yml:43; ci.yml:241,495,747,960; deep-nightly.yml:81,171; dep-floors.yml:73,116; fs-benchmark-nightly.yml:75; macos-abort-probe.yml:184 (`astral-sh/setup-uv@<sha>` with no `version:` input); pyproject.toml:182 `[tool.uv]` has no `required-version` | The action is SHA-pinned, but with no `version` it downloads the latest uv release at run time. In release.yml (id-token: write job, :25-27) that floating binary runs `uv build` (:67) and produces the artifact that is attested and uploaded to PyPI. A compromised or regressed uv release changes the published wheel with no repo diff. Exploit requires an upstream uv compromise (not verifiable here). | RESOLVED-needed: set `version: "<exact>"` on every setup-uv step (at least release.yml:43) or `required-version = "==X.Y.Z"` in `[tool.uv]`; dependabot cannot bump it, so pair with a meta test. |
| W5-SUP-02 | Info | audits/2026-11-06/wave4/_TRACKER.md:16 vs .pre-commit-config.yaml:126,200,459 | Tracker row for W4-SUP-01 says "Now all three ruff-pre-commit revs are the commit SHA of v0.16.10" and cites a passing `test_ruff_precommit_rev_is_sha_pinned.py`; HEAD still has `rev: v0.16.10` at all three lines. Either the row is stale text from a reverted change or that test is not guarding the file. A reader trusts a pin that does not exist. Owner decision (keep the tag) is not re-opened. | DOC: rewrite the row to match HEAD; check whether the named test exists and what it asserts. |

## Checked and safe

- All workflow `uses:` refs are 40-char SHAs (grep for non-SHA refs: none).
- Remote pre-commit repos other than ruff pinned by SHA (.pre-commit-config.yaml:56,86,175,186,221,267,671).
- requirements-dev.txt:65 py-ci-shared git URL at full SHA; no `--index-url`/`--extra-index-url` in requirements-dev.txt or pyproject.toml (pyproject:284 is a comment).
- CI pyutilz git installs pinned to full SHA 6b512f2b... (ci.yml:764,911; dep-floors.yml:123); local-checkout installs elsewhere. Floating `branch = "master"` at pyproject.toml:178 is SUP-01.
- release.yml: top-level `contents: read` (:18-19), twine pinned `==6.2.0` (:70), Trusted Publishing via SHA-pinned pypa/gh-action-pypi-publish (:109), no stored PyPI token.
- Tool installs pinned exactly: codecov-full.yml:75, dep-floors.yml:89, docs.yml:45, pydoclint.yml:45, pandas/polars-matrix packaging==26.0.
- `pip install --upgrade pip` (ci.yml:763,910 and matrix jobs) runs in read-only, secret-free smoke/matrix steps; same class as SUP-05/SUP-07.
- dependabot-auto-merge.yml unchanged from SUP-03 (FUTURE); `pull_request` trigger, actor-gated (:36).
- uv.lock audited by pip-audit (ci.yml:680-681) and kept in sync (ci.yml:937+).
