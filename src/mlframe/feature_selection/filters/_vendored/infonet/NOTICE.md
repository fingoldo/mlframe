# NOTICE — vendored InfoNet code

This directory (`_vendored/infonet/`) contains code adapted from:

- **Upstream repository:** https://github.com/datou30/InfoNet
- **Paper:** Hu et al., "InfoNet: Neural Estimation of Mutual Information without Test-Time
  Optimization", ICML 2024.
- **Vendored commit:** `3808a6866a43b6627f74b4f5fa217c8602963169` (upstream `main` branch,
  verified via `gh api repos/datou30/InfoNet/commits` on 2026-07-24).

## License status (USABILITY_B-9, mrmr_audit_2026-07-22)

As of the vendored commit above, the upstream repository does **not** contain a `LICENSE` file
and GitHub's license-detection API (`gh api repos/datou30/InfoNet/license`) returns no result.
No explicit open-source license terms are published by the upstream authors, so no `LICENSE`
file is vendored alongside this code — fabricating one would misrepresent terms the upstream
authors never granted.

**Anyone redistributing mlframe as a library with this vendored code included should verify the
current license status with the upstream authors directly (or exclude `_vendored/infonet/` from
the redistributed package) before doing so.** This module is used internally as an optional,
lazily-imported MI estimator (`_neural_mi.py`); it is not required for any other mlframe
functionality.
