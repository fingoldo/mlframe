# pip-audit of the locked dependencies, 2026-10-08

The advisory job `pip-audit` in `lint (advisory)` never audited anything: with no requirements file it resolved `.`, and
that failed on `pyutilz>=1.0.0` (git-sourced, not on PyPI) before pip-audit ran. The shared workflow now has
`pip-audit-uv-lock` (see py-ci-shared `lint-advisory.yml`), which exports `uv.lock` and audits the exact versions with
`--no-deps --disable-pip`. Run locally against this repository's lock it audited 366 dependencies and found 31
advisories in 10 packages.

| Package | Locked | Advisories | Fixed in | Status |
|---|---|---|---|---|
| tornado | 6.5.8 | GHSA-chx6-46f5-w4vp, GHSA-c2m8-h5v5-343r, GHSA-3hv7-mjh2-fv65 | 6.5.9 | RESOLVED: floor `tornado>=6.5.10`, lock 6.5.10 |
| pymongo | 4.18.0 | CVE-2026-88029, CVE-2026-96747, CVE-2026-96749, one more | 4.18.1 / 4.18.2 | RESOLVED: floor `pymongo>=4.18.3`, lock 4.18.3 |
| virtualenv | 21.7.8 | four advisories | 21.7.11 to 21.7.13 | RESOLVED: floor in `dev`, lock 21.14.5 |
| werkzeug | 3.1.8 | one advisory | 3.1.9 | RESOLVED: floor `werkzeug>=3.1.9` in `viz`, lock 3.1.9 |
| multidict | 6.7.1 | CVE-2026-104874 | 6.9.1 | RESOLVED for Python 3.10+: floor with marker (6.9 needs 3.10), lock 6.9.1 |
| urllib3 | 2.7.0 | PYSEC-2026-4175/4176/4177 | 2.8.0 | RESOLVED for Python 3.10+: floor with marker, lock 2.8.0 |
| cryptography | 48.0.1 | PYSEC-2026-3552/3553/3554 | 49.0.0, 50.0.0 | OPEN: the `mlflow` extra requires `cryptography>=48.0.1,<49`, so a floor of 50 makes the extras unsatisfiable. Needs an upstream mlflow release without the cap, or dropping the extra's cap |
| setuptools | 81.0.0 | PYSEC-2026-3447 | 83.0.0 | OPEN: torch caps `setuptools<82`; follows from the torch decision |
| torch | 2.10.0 | PYSEC-2025-194, PYSEC-2026-139 | 2.13.0 | OPEN: the floor is a compatibility decision (CUDA 12 builds, supported Python range), not a mechanical bump |
| pytorch-lightning | 2.6.5 | PYSEC-2026-3967 | 2.6.6 | OPEN: `lightning>=2.6.6` is already the floor; the standalone `pytorch-lightning` entry is held at 2.6.5 by something in the lock and needs a targeted look |

The four OPEN rows are the only advisories left in the lock (7 advisories in 4 packages).
