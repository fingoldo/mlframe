"""Wave 71 (2026-05-21): non-ASCII chars reaching print() / logger crash on
Windows cp866 (Russian CMD default) console with UnicodeEncodeError.

Per memory rule feedback_windows_encoding, non-ASCII Python print output on
Windows crashes. Audit found 2 em-dash (U+2014) findings:

  1. P0: feature_selection/_benchmarks/bench_mrmr_threading_vs_loky.py:114
     print(...) with em-dash, fires unconditionally on every invocation;
     also was using `{n}` without f-string prefix (separate cosmetic).
     Replaced em-dash with ASCII `--`, added missing f-prefix.

  2. P1: training/pipeline.py:535
     logger.warning(...) with em-dash inside fallback branch; logger
     handler uses console encoding by default, so emits to a cp866-active
     CMD would crash. Replaced em-dash with `--`.

Combined check: zero non-ASCII chars now appear inside print() / logger
arguments in production paths.
"""

from __future__ import annotations

from pathlib import Path


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file. A flat module that became a subpackage
    (``X.py`` -> ``X/__init__.py`` + submodules) is read as the package
    __init__ plus every submodule so structural source pins still match."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


def test_bench_mrmr_no_em_dash_in_print() -> None:
    src = _read("feature_selection/_benchmarks/bench_mrmr_threading_vs_loky.py")
    # The em-dash U+2014 must be gone from the print line at :114.
    assert "(legacy default — may break" not in src
    # Replaced with ASCII double-hyphen.
    assert "(legacy default -- may break" in src
    # And the f-string prefix is now in place.
    assert 'print(f"\\n--- run 3: backend=loky' in src


def test_pipeline_to_pandas_fallback_no_em_dash() -> None:
    src = _read("training/pipeline.py")
    assert "bare .to_pandas() — wide-frame" not in src
    assert "bare .to_pandas() -- wide-frame" in src


def test_no_non_ascii_in_print_or_logger_arguments() -> None:
    """Forensic-style: enumerate every U+2013 / U+2014 / U+2192 / U+00B1
    character in src/mlframe/ and assert none lands inside a print() or
    logger.* argument context (heuristic: check the enclosing line)."""
    import re

    forbidden_in_io = ["—", "–", "→", "←", "✓", "✗", "±", "≥", "≤", "…"]
    io_pattern = re.compile(r"\b(?:print|logger\.\w+|logging\.\w+|sys\.stdout\.write|sys\.stderr\.write|tqdmu)\s*\(")

    leaks: list = []
    for py in MLFRAME_ROOT.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:  # nosec B112 -- best-effort skip of one iteration on a non-fatal error; the test's own assertions are unaffected
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if not any(ch in line for ch in forbidden_in_io):
                continue
            if io_pattern.search(line):
                leaks.append(f"{py.relative_to(MLFRAME_ROOT)}:{line_no}: {line.strip()[:120]}")

    assert not leaks, (
        "Wave 71: non-ASCII char(s) found inside print()/logger argument(s); "
        "Windows cp866 console will crash on UnicodeEncodeError. Sites:\n  " + "\n  ".join(leaks)
    )
