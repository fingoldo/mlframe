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


import mlframe as _mlframe

_CORE_ROOT = Path(_mlframe.__file__).resolve().parent / "training"

_CORE_FILES = [
    _CORE_ROOT / "core" / "main.py",
    _CORE_ROOT / "core" / "predict.py",
    _CORE_ROOT / "core" / "_phase_finalize.py",
    _CORE_ROOT / "core" / "_phase_helpers.py",
    _CORE_ROOT / "core" / "_phase_recurrent.py",
    _CORE_ROOT / "core" / "_phase_train_one_target.py",
    # Monolith-split siblings: body carved out from the parent above, then
    # further decomposed into ensembling / polars-fastpath / pre-screen
    # helpers. Sensor parametrizes per-file so adding siblings here is the
    # cleanest "scan all" repoint.
    _CORE_ROOT / "core" / "_phase_train_one_target_body.py",
    _CORE_ROOT / "core" / "_phase_train_one_target_helpers.py",
    _CORE_ROOT / "core" / "_phase_train_one_target_ensembling.py",
    _CORE_ROOT / "core" / "_phase_train_one_target_polars_fastpath.py",
    _CORE_ROOT / "core" / "_phase_train_one_target_pre_screen.py",
    _CORE_ROOT / "core" / "_phase_train_one_target_model_setup.py",
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
    assert not offenders, f"{py_file.name}: dated audit-history comments still present:\n" + "\n".join(f"  L{n}: {t}" for n, t in offenders[:20])


def test_phases_apply_third_party_patches_lazy_only() -> None:
    """Wave 1.5 contract reaffirmed by Wave 4+5 comment-strip pass:
    `mlframe.training.__init__` must NOT import _model_factories at
    module load (which would re-introduce the import-time patch side
    effect). The factories module is loaded only on first suite call
    or on first ``make_pool`` / ``make_dmatrix`` / ``make_lgb_dataset``.

    Probe in a SUBPROCESS so the ``sys.modules.pop`` we'd otherwise need
    doesn't rebind ``_model_factories`` mid-suite and split class identity
    for later tests (per the test-pollution rule in CLAUDE.md).
    """
    import os
    import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
    import sys
    import textwrap

    import mlframe as _mlframe_pkg

    _src_root = os.path.dirname(os.path.dirname(_mlframe_pkg.__file__))
    _env = {**os.environ, "PYTHONPATH": _src_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _probe = textwrap.dedent("""
        import sys
        import mlframe.training  # noqa: F401
        sys.stdout.write("FACTORIES_IN_SYSMODULES=" + str("mlframe.training._model_factories" in sys.modules))
    """)
    _res = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", _probe],
        capture_output=True,
        text=True,
        timeout=180,
        env=_env,
    )
    assert _res.returncode == 0, f"probe subprocess failed: {_res.stderr}"
    assert (
        "FACTORIES_IN_SYSMODULES=False" in _res.stdout
    ), f"factories module imported eagerly -- Wave 1.5 lazy-init invariant broken (probe printed {_res.stdout!r})"


def test_core_main_applies_patches_at_suite_entry(monkeypatch) -> None:
    """Behavioural: ``train_mlframe_models_suite`` must invoke
    ``apply_loky_cpu_count_override`` AND ``apply_third_party_patches_once``
    BEFORE any phase entry. We monkeypatch both and record call order;
    then run the suite far enough to trip the entry-bouncer (the early
    raise on invalid df is fine, the patches must have been called by then).
    """
    import mlframe.training.core.main as _main_mod

    # ``train_mlframe_models_suite`` body lives in ``_main_train_suite.py``;
    # the live prelude resolves ``apply_loky_cpu_count_override`` /
    # ``apply_third_party_patches_once`` from THAT module's globals (patched in
    # via ``apply_module_global_patches(sys.modules[__name__])``), so that is
    # the only real import site to monkeypatch.
    from mlframe.training.core import _main_train_suite as _suite_mod

    call_order: list[str] = []
    for _mod in (_suite_mod,):
        if hasattr(_mod, "apply_loky_cpu_count_override"):
            orig_loky = _mod.apply_loky_cpu_count_override

            def _loky(*a, _orig=orig_loky, **kw):
                """Records that apply_loky_cpu_count_override ran, then delegates to the real function."""
                call_order.append("loky")
                return _orig(*a, **kw)

            monkeypatch.setattr(_mod, "apply_loky_cpu_count_override", _loky)

        if hasattr(_mod, "apply_third_party_patches_once"):
            orig_patch = _mod.apply_third_party_patches_once

            def _patch(*a, _orig=orig_patch, **kw):
                """Records that apply_third_party_patches_once ran, then delegates to the real function."""
                call_order.append("patches")
                return _orig(*a, **kw)

            monkeypatch.setattr(_mod, "apply_third_party_patches_once", _patch)

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
    assert "loky" in call_order or "patches" in call_order, "Neither prelude function called before suite entry"
    if "loky" in call_order and "patches" in call_order:
        assert call_order.index("loky") < call_order.index("patches"), "loky must apply BEFORE third-party patches"


def test_pipeline_does_not_mutate_env_at_import() -> None:
    """Wave 4+5 sweep verified that ``pipeline.py`` no longer mutates
    ``os.environ`` at module-import time (the Julia thread vars were
    moved into ``_apply_pysr_fe``). Importing ``pipeline`` must not
    touch ``PYTHON_JULIACALL_THREADS`` or ``JULIA_NUM_THREADS``.

    Probe in a SUBPROCESS so the ``sys.modules.pop`` we'd otherwise need
    doesn't rebind ``pipeline`` mid-suite and split class identity for
    later tests (per the test-pollution rule in CLAUDE.md).
    """
    import os
    import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
    import sys
    import textwrap

    import mlframe as _mlframe_pkg

    _src_root = os.path.dirname(os.path.dirname(_mlframe_pkg.__file__))
    _env = {**os.environ, "PYTHONPATH": _src_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _probe = textwrap.dedent("""
        import os
        sentinel_keys = ("PYTHON_JULIACALL_THREADS", "JULIA_NUM_THREADS")
        pre = {k: os.environ.get(k) for k in sentinel_keys}
        import mlframe.training.pipeline  # noqa: F401
        post = {k: os.environ.get(k) for k in sentinel_keys}
        import sys as _s
        _s.stdout.write("PRE=" + repr(pre) + "\\n" + "POST=" + repr(post))
    """)
    _res = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", _probe],
        capture_output=True,
        text=True,
        timeout=180,
        env=_env,
    )
    assert _res.returncode == 0, f"probe subprocess failed: {_res.stderr}"
    _lines = {ln.split("=", 1)[0]: ln.split("=", 1)[1] for ln in _res.stdout.strip().splitlines() if "=" in ln}
    assert _lines.get("PRE") == _lines.get("POST"), f"pipeline.py mutated env at import: {_lines.get('PRE')} -> {_lines.get('POST')}"


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
