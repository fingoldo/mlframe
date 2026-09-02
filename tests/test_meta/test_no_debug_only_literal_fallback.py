"""Meta-test: an exception handler that SUBSTITUTES a value must not be silent about it.

Distilled from the 2026-09-01 audit's `xcut_swallowed_failures` cluster -- sixteen findings, one shape. A
handler catches an exception, returns or assigns a literal, and logs at DEBUG (which production logging does not
emit) or not at all. The substituted value is then indistinguishable from a real result, and in every one of
those sixteen cases it was not neutral:

* ``_max_err = 0.0`` on a failed max-error computation is the BEST possible max error, so it made
  ``_max_err > 5 * y_std`` unconditionally False and switched a collapse sensor off -- in exactly the situations
  (shape mismatch, object-dtype predictions) that produce the failure.
* ``return True`` from a failed VRAM probe is the value that ALLOWS the upload, removing OOM protection
  precisely when the device is too unhealthy to answer.
* ``return -np.inf`` for a failed hinge solve is the value that GUARANTEES rejection, so a driver fault
  discarded a good breakpoint.
* ``0.0`` for a failed fingerprint statistic reads as "uncorrelated" / "no cardinality" -- both legal values --
  so the cache keyed against a description of a different dataset rather than admitting ignorance.

The test to apply at the call site: **if this value were wrong, would anything downstream notice?** When the
answer is no, the failure has to be audible. The sanctioned forms are a WARNING (throttled via
``mlframe.utils.log_throttle.log_throttle`` when the site is hot), a NaN/None that reads as "unknown" rather
than as a measurement, or re-raising.

Deliberately NOT flagged: a handler narrowed to ``ImportError`` (an optional dependency genuinely absent is a
permanent, expected condition, and its substitution is the intended answer), and handlers that already reach
``warning`` / ``error`` / ``exception`` / ``log_throttle``.

Baseline-diffed: pre-existing sites are grandfathered and only NEW ones fail. Refresh with
``--refresh-debug-only-literal-fallback-baseline`` after reviewing a finding.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_BASELINE_PATH = Path(__file__).resolve().parent / "_debug_only_literal_fallback_baseline.json"

_AUDIBLE = frozenset({"warning", "error", "exception", "critical", "warn"})
_THROTTLE = frozenset({"log_throttle"})
# `None` is the one substitution that reads as "no answer" rather than as an answer, so it is not a finding.
_NEUTRAL_LITERALS = (None,)


def _refresh_requested() -> bool:
    """True if ``--refresh-debug-only-literal-fallback-baseline`` was passed on the pytest command line."""
    return "--refresh-debug-only-literal-fallback-baseline" in sys.argv


def _is_import_error_only(handler: ast.ExceptHandler) -> bool:
    """True when the handler catches ImportError / ModuleNotFoundError and nothing broader."""
    if handler.type is None:
        return False
    names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
    return bool(names) and names <= {"ImportError", "ModuleNotFoundError"}


def _is_audible(handler: ast.ExceptHandler) -> bool:
    """True when the handler logs at warning or above, or re-raises."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in _AUDIBLE:
            return True
        if isinstance(fn, ast.Name) and fn.id in _THROTTLE:
            return True
    return False


def _substituted_literal(handler: ast.ExceptHandler) -> str | None:
    """A short description of the non-neutral literal this handler substitutes, or None."""

    def _describe(value: ast.AST) -> str | None:
        """Name the literal when it is one worth flagging."""
        if isinstance(value, ast.Constant):
            if value.value in _NEUTRAL_LITERALS:
                return None
            if isinstance(value.value, (bool, int, float, str)):
                return repr(value.value)
            return None
        # `-np.inf` / `np.inf` -- the classic "guarantees the comparison's outcome" substitution.
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            inner = value.operand
            if isinstance(inner, ast.Attribute) and inner.attr in {"inf", "Inf", "infty"}:
                return "-inf"
        if isinstance(value, ast.Attribute) and value.attr in {"inf", "Inf", "infty"}:
            return "inf"
        return None

    for node in ast.walk(handler):
        if isinstance(node, ast.Return) and node.value is not None:
            desc = _describe(node.value)
            if desc is not None:
                return f"return {desc}"
        if isinstance(node, ast.Assign) and node.targets:
            desc = _describe(node.value)
            if desc is not None:
                tgt = node.targets[0]
                name = tgt.id if isinstance(tgt, ast.Name) else "<target>"
                return f"{name} = {desc}"
    return None


def _silent_substitutions(tree: ast.Module) -> list:
    """``[(lineno, description), ...]`` for handlers substituting a non-neutral literal without being audible."""
    out: list = []
    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        if _is_import_error_only(handler) or _is_audible(handler):
            continue
        desc = _substituted_literal(handler)
        if desc is not None:
            out.append((handler.lineno, desc))
    return out


def _build_offending_set() -> set:
    """``{"relpath:lineno:description", ...}`` for every silent literal substitution under ``src/mlframe``."""
    out: set = set()
    for py in _SRC_DIR.rglob("*.py"):
        if "__pycache__" in py.parts or "_benchmarks" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_SRC_DIR).as_posix()
        for lineno, desc in _silent_substitutions(tree):
            out.add(f"{rel}:{lineno}:{desc}")
    return out


def test_no_new_debug_only_literal_fallback():
    """No new exception handler substitutes a non-neutral literal without saying so above debug level."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"debug-only-literal-fallback baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    assert not added, (
        "New silent literal substitution(s) in an exception handler:\n  "
        + "\n  ".join(added)
        + "\n\nAsk of the substituted value: if it were wrong, would anything downstream notice? If not, the "
        "failure has to be audible. Note that a value can be non-neutral in a way that DISABLES the very check "
        "it feeds -- 0.0 for a max error, True for a permission guard, -inf for a score being minimised -- which "
        "is how a safety mechanism switches itself off in exactly the conditions that trip it.\n"
        "Use one of:\n"
        "  * logger.warning(...), or log_throttle(logger, key, logging.WARNING, ...) on a hot path;\n"
        "  * a NaN/None that reads as 'unknown' rather than as a measurement;\n"
        "  * re-raise.\n"
        "An `except ImportError` handler is exempt: a genuinely absent optional dependency is permanent and its "
        "substitution is the intended answer. Refresh with --refresh-debug-only-literal-fallback-baseline."
    )
