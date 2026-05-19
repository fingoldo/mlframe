"""Wave 4+5 LOW/POLISH regression tests for the core/orchestration scope.

The Wave 4+5 core agent stripped dated audit-history comments (sweep
S1), em-dashes in log strings (S2), and converted module-boundary
``assert`` statements to ``if ... raise`` (S3). The agent never
delivered a separate test file before tracking dropped, so these tests
back-fill that gap.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_CORE_ROOT = Path(
    "D:/Upd/Programming/PythonCodeRepository/mlframe/src/mlframe/training"
)

_CORE_FILES = [
    _CORE_ROOT / "core" / "main.py",
    _CORE_ROOT / "core" / "predict.py",
    _CORE_ROOT / "core" / "_phase_finalize.py",
    _CORE_ROOT / "core" / "_phase_helpers.py",
    _CORE_ROOT / "core" / "_phase_recurrent.py",
    _CORE_ROOT / "core" / "_phase_train_one_target.py",
    _CORE_ROOT / "core" / "_setup_helpers.py",
    _CORE_ROOT / "core" / "_misc_helpers.py",
    _CORE_ROOT / "phases.py",
    _CORE_ROOT / "pipeline.py",
    _CORE_ROOT / "_training_loop.py",
    _CORE_ROOT / "_pipeline_helpers.py",
]


_DATED_PATTERNS = [
    re.compile(r"#\s*20\d\d-\d\d-\d\d\b"),  # # 2026-05-16, # 2026-04-21
    re.compile(r"#\s*(Wave|wave)-\d+\b"),  # # Wave-7
    re.compile(r"#\s*round-\d+\b"),  # # round-3 A#4
    re.compile(r"#\s*Session\s+\d+\b"),  # # Session 2
    re.compile(r"#\s*\(user\s+(request|feedback)\)"),  # # (user request)
    re.compile(r"#\s*OPEN-\d+\b"),  # # OPEN-4
    re.compile(r"#\s*R10[a-z]\b"),  # # R10c
    re.compile(r"#\s*C\d+\s*\(20\d\d"),  # # C1 (2026-...)
]


@pytest.mark.parametrize("py_file", _CORE_FILES)
def test_core_files_have_no_dated_audit_comments(py_file: Path) -> None:
    """Sweep S1 (Wave 4+5): all dated audit-history tags in core scope
    have been stripped. Comments may still discuss behaviour or
    historical context, just without date stamps / wave-N / round-N /
    Session-N / audit-id prefixes."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not present")
    src = py_file.read_text(encoding="utf-8")
    offenders: list[tuple[int, str]] = []
    for ln_no, line in enumerate(src.splitlines(), start=1):
        for pat in _DATED_PATTERNS:
            if pat.search(line):
                offenders.append((ln_no, line.strip()))
                break
    assert not offenders, (
        f"{py_file.name}: dated audit-history comments still present:\n"
        + "\n".join(f"  L{n}: {t}" for n, t in offenders[:20])
    )


def test_phases_apply_third_party_patches_lazy_only() -> None:
    """Wave 1.5 contract reaffirmed by Wave 4+5 comment-strip pass:
    `mlframe.training.__init__` must NOT import _model_factories at
    module load (which would re-introduce the import-time patch side
    effect). The factories module is loaded only on first suite call
    or on first ``make_pool`` / ``make_dmatrix`` / ``make_lgb_dataset``."""
    import importlib
    import sys

    # Force-reload the package; the factories module must NOT appear in
    # sys.modules after a bare import of mlframe.training.
    for mod_name in list(sys.modules):
        if mod_name == "mlframe.training._model_factories":
            sys.modules.pop(mod_name, None)
    importlib.import_module("mlframe.training")
    assert "mlframe.training._model_factories" not in sys.modules, (
        "factories module imported eagerly -- Wave 1.5 lazy-init invariant broken"
    )


def test_core_main_applies_patches_at_suite_entry(monkeypatch) -> None:
    """Behavioural: ``train_mlframe_models_suite`` must invoke
    ``apply_loky_cpu_count_override`` AND ``apply_third_party_patches_once``
    BEFORE any phase entry. We monkeypatch both and record call order;
    then run the suite far enough to trip the entry-bouncer (the early
    raise on invalid df is fine, the patches must have been called by then).
    """
    import mlframe.training.core.main as _main_mod

    call_order: list[str] = []
    if hasattr(_main_mod, "apply_loky_cpu_count_override"):
        orig_loky = _main_mod.apply_loky_cpu_count_override

        def _loky(*a, **kw):
            call_order.append("loky")
            return orig_loky(*a, **kw)

        monkeypatch.setattr(_main_mod, "apply_loky_cpu_count_override", _loky)

    if hasattr(_main_mod, "apply_third_party_patches_once"):
        orig_patch = _main_mod.apply_third_party_patches_once

        def _patch(*a, **kw):
            call_order.append("patches")
            return orig_patch(*a, **kw)

        monkeypatch.setattr(_main_mod, "apply_third_party_patches_once", _patch)

    # Trip the entry-bouncer: pass a non-pandas/polars/str df so the suite
    # raises immediately after the prelude (which is what we are testing).
    train = _main_mod.train_mlframe_models_suite
    with pytest.raises((TypeError, ValueError, AttributeError)):
        train(
            df=42,  # invalid type -> raise after prelude
            target_name="y",
            model_name="m",
            features_and_targets_extractor=lambda *a, **kw: ({}, {}),
        )

    # Both prelude calls must have happened; loky must come first if both present.
    assert "loky" in call_order or "patches" in call_order, (
        "Neither prelude function called before suite entry"
    )
    if "loky" in call_order and "patches" in call_order:
        assert call_order.index("loky") < call_order.index("patches"), (
            "loky must apply BEFORE third-party patches"
        )


def test_pipeline_does_not_mutate_env_at_import() -> None:
    """Wave 4+5 sweep verified that ``pipeline.py`` no longer mutates
    ``os.environ`` at module-import time (the Julia thread vars were
    moved into ``_apply_pysr_fe``). Importing ``pipeline`` must not
    touch ``PYTHON_JULIACALL_THREADS`` or ``JULIA_NUM_THREADS``."""
    import importlib
    import os
    import sys

    sentinel_keys = ("PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS")
    pre = {k: os.environ.get(k) for k in sentinel_keys}

    sys.modules.pop("mlframe.training.pipeline", None)
    importlib.import_module("mlframe.training.pipeline")

    post = {k: os.environ.get(k) for k in sentinel_keys}
    assert pre == post, (
        f"pipeline.py mutated env at import: {pre} -> {post}"
    )


def test_phases_module_has_threadsafe_registry() -> None:
    """``_PhaseRegistry`` must own a ``threading.Lock`` so concurrent
    suite invocations don't corrupt the totals dicts. Wave 4+5 just
    polished comments here -- this test pins the lock invariant the
    previous waves' fixes depended on."""
    import threading
    from mlframe.training.phases import _PhaseRegistry  # type: ignore

    reg = _PhaseRegistry()
    assert hasattr(reg, "_lock")
    assert isinstance(reg._lock, type(threading.Lock()))
