# pip-audit of the locked dependencies, 2026-10-08

The advisory job `pip-audit` in `lint (advisory)` never audited anything: with no requirements file it resolved `.`, and
that failed on `pyutilz>=1.0.0` (git-sourced, not on PyPI) before pip-audit ran. The shared workflow now has
`pip-audit-uv-lock` (py-ci-shared v1.22.2), which exports `uv.lock` and audits the exact versions with
`--no-deps --disable-pip`. Run locally against this repository's lock it audited 366 dependencies and found 31
advisories in 10 packages.

| Package | Locked | Advisories | Fixed in | Disposition |
|---|---|---|---|---|
| tornado | 6.5.8 | GHSA-chx6-46f5-w4vp, GHSA-c2m8-h5v5-343r, GHSA-3hv7-mjh2-fv65 | 6.5.9 | RESOLVED: floor `tornado>=6.5.10`, lock 6.5.10 |
| pymongo | 4.18.0 | CVE-2026-88029, CVE-2026-96747, CVE-2026-96749, one more | 4.18.1 / 4.18.2 | RESOLVED: floor `pymongo>=4.18.3`, lock 4.18.3 |
| virtualenv | 21.7.8 | four advisories | 21.7.11 to 21.7.13 | RESOLVED: floor in `dev`, lock 21.14.5 |
| werkzeug | 3.1.8 | one advisory | 3.1.9 | RESOLVED: floor `werkzeug>=3.1.9` in `viz`, lock 3.1.9 |
| multidict | 6.7.1 | CVE-2026-104874 | 6.9.1 | RESOLVED for Python 3.10+: floor with marker (6.9 needs 3.10), lock 6.9.1 |
| urllib3 | 2.7.0 | PYSEC-2026-4175/4176/4177 | 2.8.0 | RESOLVED for Python 3.10+: floor with marker, lock 2.8.0 |
| cryptography | 48.0.1 | PYSEC-2026-3552/3553/3554 | 49.0.0, 50.0.0 | RESOLVED for Python 3.10+: the `<49` cap was this project's own copy of an older mlflow cap; mlflow 3.17 allows `<51`, so the floor is `cryptography>=50.0.0,<51` (lock 50.0.2). Python 3.9 keeps `>=48.0.1,<49` because the mlflow releases that run there still cap it |
| pytorch-lightning | 2.6.5 | PYSEC-2026-3967 (remote code execution through checkpoint hyperparameters) | 2.6.6 | RESOLVED for Python 3.10+: `lightning` requires `pytorch-lightning` without a bound and the lock had kept 2.6.5; explicit floor `pytorch-lightning>=2.6.6`, lock 2.6.6 |
| torch | 2.10.0 | PYSEC-2025-194 (`torch.jit.script` memory corruption, local), PYSEC-2026-139 (pt2 loading handler, local) | 2.13.0 for the first; none for the second | REJECTED (accepted risk), see below |
| setuptools | 81.0.0 | PYSEC-2026-3447 (`MANIFEST.in` exclude rules when building an sdist) | 83.0.0 | REJECTED (accepted risk), see below |

## Why torch and setuptools stay

torch 2.10.0 is the last release whose wheels use CUDA 12 (`nvidia-cublas-cu12`). From 2.11 the wheels require
`cuda-toolkit` 13, and 2.11 and 2.12 still cap `setuptools<82`; only 2.13 lifts that cap. The project keeps torch on a CUDA 12 build
on purpose, because a CUDA 13 torch next to `cupy-cuda12x` in one process crashes natively (see the comment in the `neural` extra).
Measured: adding `torch>=2.13` for Python 3.10+ makes the lock unsatisfiable, because torch 2.14 and `cupy-cuda12x[ctk]` need different
`cuda-toolkit` majors. So the floor cannot be raised without moving the whole GPU stack (cupy, numba, the CUDA extras) to CUDA 13.

Exposure: both torch advisories need local code execution or an attacker-controlled model or script. The only `torch.jit.script` use is
`training/neural/_ranker_losses.py`, which compiles the project's own kernel, and models are loaded through the restricted loader.
PYSEC-2026-139 has no patched release at all. The setuptools advisory concerns building a source distribution, a build-time step
run on trusted sources, not anything mlframe does at run time.

Revisit when the GPU stack moves to CUDA 13: then `torch>=2.13`, no `setuptools<82` cap, and both rows close.
