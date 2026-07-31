"""H3 (partial) — meta-test that no ``except Exception:`` handler swallows the exception silently
or logs only behind a ``verbose`` gate.

The bare-``except:``/``except BaseException:`` half of this check (formerly
``test_no_new_bare_except_clauses``) now runs via the shared ``pyutilz.dev.code_audit``
``bare_except`` scanner in ``test_code_audit_baseline.py`` instead of a local copy.
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

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore")


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
    """True if ``node`` is (or contains) a call to a ``logger.*``/``logging.*``/``warnings.warn``
    function, to ``mlframe.utils.log_throttle.log_throttle`` (a plain function, not a
    logger-attribute call, that unconditionally logs its message subject to a per-key rate limit),
    or to a chained ``logging.getLogger(...).<level>(...)`` call (a fresh logger acquired inline
    rather than through a module-level ``logger`` name)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "log_throttle":
        return True
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name) and base.id in ("logger", "logger_", "logging", "warnings", "log"):
            return True
        if isinstance(base, ast.Attribute) and base.attr in ("logger",):
            return True
        # Chained call: `logging.getLogger(...).debug(...)` / `_logging.getLogger(__name__).warning(...)`.
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) and base.func.attr == "getLogger":
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
                    # The `if verbose:` arm is gated, but an `else:` arm that ALSO logs means every
                    # path -- verbose or not -- logs something (e.g. `if verbose: logger.warning(...)
                    # else: logger.debug(...)`), so the handler is not silent at verbose=0.
                    if _stmt_has_unconditional_log(s.orelse):
                        return True
                    continue
                if _stmt_has_unconditional_log(s.body) or _stmt_has_unconditional_log(s.orelse):
                    return True
            # Other compound statements (Try/For/While) -- recurse into their bodies unconditionally
            # (a log call inside a nested Try/For within the handler still counts as unconditional
            # unless it's itself behind a verbose check). A nested Try's own except handlers are
            # also part of the outer handler's effective body (its log calls still run
            # unconditionally when the nested try fails) -- not just the try body.
            elif isinstance(s, (ast.Try,)):
                if _stmt_has_unconditional_log(s.body):
                    return True
                for h in s.handlers:
                    if _stmt_has_unconditional_log(h.body):
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
        return False  # bare except is handled by the shared bare_except scanner, not this check
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
