"""H3 — meta-test that no production code uses ``except:`` (bare) or
``except BaseException:``.

Bare ``except:`` swallows EVERYTHING — ``KeyboardInterrupt`` (so the
user can't Ctrl-C), ``SystemExit`` (forks misbehave), and
``MemoryError`` (debugger gets confused). It also masks bugs in the
try-block by catching them as if they were expected. The narrow form
``except Exception:`` is the safe equivalent in nearly every case.

Catches:
  - ``except:``                     → must be ``except Exception:`` (or specific)
  - ``except BaseException:``       → same (or KeyboardInterrupt / SystemExit individually if intended)

Skips ``except Exception:`` and any narrower exception type — those
are intentional.

Snapshot-style; first run captures any existing offenders. Future
commits adding new bare excepts fail.
"""

from __future__ import annotations

import ast
import orjson
import sys
from pathlib import Path

import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_bare_except_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore")


def _refresh_requested() -> bool:
    """True if ``--refresh-bare-except-baseline`` was passed on the pytest command line."""
    return "--refresh-bare-except-baseline" in sys.argv


def _is_bare_except(handler: ast.ExceptHandler) -> bool:
    """``handler.type`` is None for bare ``except:``; or a Name
    ``BaseException`` for the equivalent dangerous form.

    EXCEPTION: ``except BaseException as e: ... raise`` (re-raise) is
    legitimate — phase-tracking context managers, request-scope cleanup,
    and similar patterns that audit EVERY exit path. We detect a bare
    re-raise inside the handler and allow that case.
    """
    if handler.type is None:
        # Bare ``except:`` — never legitimate; swallows everything
        # including KeyboardInterrupt before the (possible) re-raise.
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException":
        for sub in ast.walk(handler):
            if isinstance(sub, ast.Raise) and sub.exc is None:
                return False  # bare ``raise`` → re-raises, legitimate
        return True
    return False


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every bare/BaseException ``except`` clause under ``src/mlframe``."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if _is_bare_except(handler):
                    out.add(f"{rel}:{handler.lineno}")
    return out


_VERBOSE_GATED_BASELINE_PATH = Path(__file__).resolve().parent / "_verbose_gated_except_baseline.json"


def _refresh_verbose_gated_requested() -> bool:
    """True if ``--refresh-verbose-gated-except-baseline`` was passed on the pytest command line."""
    return "--refresh-verbose-gated-except-baseline" in sys.argv


def _references_verbose(node: ast.AST) -> bool:
    """True if ``node`` (an ``If.test`` expression) references any name containing "verbose"."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and "verbose" in sub.id.lower():
            return True
        if isinstance(sub, ast.Attribute) and "verbose" in sub.attr.lower():
            return True
    return False


def _is_log_call(node: ast.AST) -> bool:
    """True if ``node`` is (or contains) a call to a ``logger.*``/``logging.*``/``warnings.warn`` function."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name) and base.id in ("logger", "logging", "warnings", "log"):
            return True
        if isinstance(base, ast.Attribute) and base.attr in ("logger",):
            return True
    return False


def _handler_has_unconditional_log(handler: ast.ExceptHandler) -> bool:
    """True if ``handler``'s body contains a log/warn call NOT nested inside a verbose-gated ``if``.

    Walks at statement granularity (not via generic ``ast.walk``, which has no parent links) so a log
    call inside a verbose-gated ``If`` can be excluded from counting as unconditional.
    """
    def _stmt_has_unconditional_log(stmts: list) -> bool:
        """True if any statement in ``stmts`` is (or recursively contains) an unconditional log call."""
        for s in stmts:
            if isinstance(s, ast.Expr) and _is_log_call(s.value):
                return True
            if isinstance(s, ast.If):
                if _references_verbose(s.test):
                    continue  # log calls inside here are gated -- don't count them as unconditional
                if _stmt_has_unconditional_log(s.body) or _stmt_has_unconditional_log(s.orelse):
                    return True
            # Other compound statements (Try/For/While) -- recurse into their bodies unconditionally
            # (a log call inside a nested Try/For within the handler still counts as unconditional
            # unless it's itself behind a verbose check).
            elif isinstance(s, (ast.Try,)):
                if _stmt_has_unconditional_log(s.body):
                    return True
            elif hasattr(s, "body") and isinstance(getattr(s, "body", None), list):
                if _stmt_has_unconditional_log(s.body):
                    return True
        return False

    return _stmt_has_unconditional_log(handler.body)


def _handler_is_effectively_silent(handler: ast.ExceptHandler) -> bool:
    """True if a broad ``except Exception:`` handler swallows the exception with no unconditional log
    (either the body is empty/pass/continue-only, or its only log call is gated behind a verbose check)."""
    if handler.type is None:
        return False  # bare except is handled by test_no_new_bare_except_clauses, not this check
    if not (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
        return False  # narrower exception types are intentional, not this finding's scope
    # A handler that re-raises is never silent.
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise):
            return False
    return not _handler_has_unconditional_log(handler)


def _build_verbose_gated_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every broad ``except Exception:`` handler that swallows silently or logs
    only behind a ``verbose`` gate -- X_EDGE_CASES_BEST_PRACTICES / usability_a / cat_interaction /
    fe_step / orth_basis / screen_confirm findings (mrmr_audit_2026-07-22) all independently found this
    pattern masking real bugs at the library's own ``verbose=0`` default."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if _handler_is_effectively_silent(handler):
                    out.add(f"{rel}:{handler.lineno}")
    return out


def test_no_new_verbose_gated_or_silent_except_exception():
    """No new broad ``except Exception:`` handler that swallows silently or logs only behind a
    ``verbose`` gate, beyond the frozen baseline.

    mrmr_audit_2026-07-22 meta-test proposal #5 (recurred 7+ times across CAT_INTERACTION_A-3/B-3,
    FE_STEP_A-2, ORTH_BASIS_A-2, SCREEN_CONFIRM_B-5, USABILITY_A-10, X_EDGE_CASES_BEST_PRACTICES-1):
    a handler that logs ONLY when ``verbose`` is truthy is completely invisible at the library's own
    ``verbose=0`` default, functionally equivalent to a silent swallow for every default-config caller.
    """
    current = _build_verbose_gated_offending_set()

    if _refresh_verbose_gated_requested() or not _VERBOSE_GATED_BASELINE_PATH.exists():
        _VERBOSE_GATED_BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"verbose-gated-except baseline refreshed at {_VERBOSE_GATED_BASELINE_PATH.name} ({len(current)} site(s))")

    baseline = set(orjson.loads(_VERBOSE_GATED_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_verbose_gated_or_silent_except_exception] {len(fixed)} site(s) "
            f"DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-verbose-gated-except-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new ``except Exception:`` handler(s) that swallow silently or log only "
            f"behind a ``verbose`` gate. Log unconditionally (even at ``logger.debug`` level) or "
            f"re-raise:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )


def test_no_new_bare_except_clauses():
    """No new bare/BaseException ``except`` clause beyond the frozen baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"bare-except baseline refreshed at {_BASELINE_PATH.name} ({len(current)} bare clauses)")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_bare_except_clauses] {len(fixed)} site(s) "
            f"DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-bare-except-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new bare ``except:`` (or ``except BaseException:``) "
            f"clause(s). Replace with ``except Exception:`` or a narrower "
            f"specific exception type — bare-except swallows "
            f"KeyboardInterrupt/SystemExit and masks real bugs:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
