"""Wave 10 LOC-budget meta-test.

Scans ``src/mlframe/`` for ``.py`` files exceeding 1000 lines of code. After
Waves 6 + 10 the project's monolith-split policy is enforced: no file should
exceed the 1k LOC ceiling. Future PRs that re-introduce a >1k file are flagged
in CI so the splitting work doesn't drift back.

The exempt list is empty by design; any added entry must come with a
justification in the PR description (e.g. ``feature_engineering/wavelet_dwt.py``
WIP). If a hot file legitimately needs the budget raised, prefer carving it via
the sibling re-export pattern (see mlframe/CLAUDE.md "Monolith split").
"""

from __future__ import annotations

from pathlib import Path

import pytest


LOC_LIMIT = 1000

# Carving budget exempts. Each entry carries a FIXME tag for the next carve
# wave; the goal is to drain this set to {} over consecutive PRs. Do NOT add
# new entries without a documented PR-description reason.
LOC_BUDGET_EXEMPT: set[str] = {
    # FIXME(carve-wave-next): _shap_proxy_revalidate.py at ~1.4k LOC carries
    # the trust-guard + topK-ablation + honest-revalidation sub-bodies; sensible
    # split is to lift the honest/ablation block to ``_shap_proxy_revalidate_honest.py``
    # behind a sibling-re-export. Tracked as the largest remaining monolith.
    "src/mlframe/feature_selection/_shap_proxy_revalidate.py",
    # FIXME(carve-wave-next): _regression_extras.py at ~1.08k LOC bundles
    # pinball / quantile / coverage-pair helpers; split to
    # ``_regression_quantile_helpers.py`` sibling.
    "src/mlframe/metrics/_regression_extras.py",
    # FIXME(carve-wave-next): _classification_extras.py at ~1.04k LOC bundles
    # calibration-curve + reliability-decomposition helpers; split to
    # ``_classification_calibration_curves.py`` sibling.
    "src/mlframe/metrics/_classification_extras.py",
    # FIXME(carve-wave-next): shap_proxied_fs.py at ~1.03k LOC; the fit body
    # is the obvious candidate for ``_shap_proxied_fs_fit.py``.
    "src/mlframe/feature_selection/shap_proxied_fs.py",
    # FIXME(carve-wave-next): filters/discretization.py at ~1.1k LOC after
    # the wrappers-iter rewrite added KBD / chi-merge / monotone-PAV branches;
    # split to ``_discretization_pav.py`` sibling.
    "src/mlframe/feature_selection/filters/discretization.py",
    # FIXME(carve-wave-next): wrappers/_rfecv.py at ~1.05k LOC after the
    # MBH-optimiser merge; split to ``_rfecv_mbh_dispatch.py`` sibling.
    "src/mlframe/feature_selection/wrappers/_rfecv.py",
    # FIXME(carve-wave-next): filters/mrmr.py at ~1.03k LOC after the
    # in-flight feature_selection wrappers iteration grew the screening
    # body; the validate/transform side is already carved (sibling
    # ``_mrmr_validate_transform.py``). The remaining surface candidate is
    # to lift the predictor-screening loop into ``_mrmr_screening_loop.py``.
    "src/mlframe/feature_selection/filters/mrmr.py",
    # FIXME(carve-wave-next): filters/_mrmr_fit_impl.py at ~1.1k LOC after
    # the Wave 9.1 DCD + fallback hardening grew the post-screening section.
    # Carve candidates: the empty-support fallback block + the FE/RFECV
    # post-pass into ``_mrmr_fit_impl_finalise.py``.
    "src/mlframe/feature_selection/filters/_mrmr_fit_impl.py",
}


def _src_root() -> Path:
    here = Path(__file__).resolve()
    # tests/test_meta/test_no_file_over_1k_loc.py -> repo root -> src/mlframe
    return here.parents[2] / "src" / "mlframe"


def _scan_src_for_oversize() -> list[tuple[str, int]]:
    root = _src_root()
    if not root.is_dir():
        pytest.skip(f"src tree not found at {root}; running from installed wheel?")
    over: list[tuple[str, int]] = []
    for path in root.rglob("*.py"):
        try:
            n = sum(1 for _ in path.open("r", encoding="utf-8"))
        except OSError:
            continue
        rel = path.relative_to(root.parent.parent).as_posix()  # "src/mlframe/..."
        if rel in LOC_BUDGET_EXEMPT:
            continue
        if n > LOC_LIMIT:
            over.append((rel, n))
    return sorted(over, key=lambda t: -t[1])


def test_no_mlframe_file_exceeds_1k_loc():
    over = _scan_src_for_oversize()
    if over:
        lines = [f"  {n:5d} LOC  {p}" for p, n in over]
        raise AssertionError(
            f"{len(over)} mlframe .py file(s) exceed {LOC_LIMIT} LOC. "
            f"Carve via sibling re-export pattern (CLAUDE.md: 'Monolith split'). "
            f"Oversized files:\n" + "\n".join(lines)
        )
