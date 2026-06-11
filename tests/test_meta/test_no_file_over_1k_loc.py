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
    # FIXME(carve-wave-next): filters/mrmr/_mrmr_class.py at ~3.7k LOC -- the irreducible
    # ``MRMR`` estimator class body after the mrmr subpackage split (class moved verbatim;
    # the package ``__init__.py`` facade re-exports it + runs the method bindings). Carve
    # candidate if it must shrink: lift the predictor-screening loop and the FE-flag
    # plumbing block off the class body into sibling helper functions bound the same way as
    # ``_fit_impl`` / ``_run_fe_step`` already are; the validate/transform/fit/fe-step/
    # partial-fit/provenance method bodies already live in sibling modules.
    "src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py",
    # FIXME(carve-wave-next): filters/_mrmr_fit_impl.py at ~1.1k LOC after
    # the Wave 9.1 DCD + fallback hardening grew the post-screening section.
    # Carve candidates: the empty-support fallback block + the FE/RFECV
    # post-pass into ``_mrmr_fit_impl_finalise.py``.
    "src/mlframe/feature_selection/filters/_mrmr_fit_impl.py",
    # FIXME(carve-wave-next): filters/_mrmr_fe_step/_step_core.py at ~1.53k LOC --
    # the irreducible single-function body of ``_run_fe_step`` after the
    # _mrmr_fe_step subpackage split. The two small operand-pool helpers
    # (``_non_numeric_column_indices`` / ``_synergy_bootstrap_can_supply_pool``)
    # already live in the sibling ``_helpers.py``; only the one giant FE-step
    # orchestration function remains over budget (mirrors ``_pairs_core.py``).
    # Carve candidate if it must shrink: lift the per-candidate scoring /
    # quantile-discretization materialise block into a ``_step_score.py`` helper.
    "src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py",
    # FIXME(carve-wave-next): training/core/_phase_train_one_target_body.py
    # at ~1.02k LOC after the recurrent-ensemble integration + composite-
    # discovery wiring. Sibling carve candidates: the recurrent rerun block
    # and the composite-post tail into per-phase helpers.
    "src/mlframe/training/core/_phase_train_one_target_body.py",
    # FIXME(carve-wave-next): filters/_feature_engineering_pairs/_pairs_core.py at
    # ~1.59k LOC -- the irreducible single-function body of ``check_prospective_fe_pairs``
    # after the _feature_engineering_pairs subpackage split. The supporting kernels /
    # gates / dispatch / chunking already live in sibling submodules (each well under
    # 1k); only the one giant orchestration function remains over budget. Carve candidate
    # if it must shrink: lift the per-pair candidate-scoring + external-validation block
    # into a ``_pairs_score.py`` helper invoked from the pair loop.
    "src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py",
    # FIXME(carve-wave-next): training/neural/recurrent.py at ~1.01k LOC after
    # the F-44 bf16-mixed auto-promote + F-46 fused-AdamW + F-47 cuDNN
    # persistent-RNN + F-48 nested-tensor + F-51 share_memory_() + F-53
    # lengths.cpu() non-sync sequence landed. Sensible carve: lift the
    # ``RecurrentDataset`` + collate helpers into
    # ``recurrent_dataset_helpers.py`` sibling; keep the LightningModule
    # in the parent facade.
    "src/mlframe/training/neural/recurrent.py",
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
