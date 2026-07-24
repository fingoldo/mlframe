"""Meta-test: no module/function docstring in the MRMR filters package falsely claims a mechanism is
NOT wired into ``MRMR.fit`` by default / is opt-in-only, when the corresponding ``fe_*_enable``
constructor parameter's actual default is ``True``.

mrmr_audit_2026-07-22 meta-test proposal #4: this exact stale-docstring pattern recurred 6+ times
(GPU_INFRA_C-5, GPU_INFRA_C-6, MI_GREEDY_RECIPES-6, ORTH_BASIS_A-5, USABILITY_A / usability_b cross-refs,
X_EFFICIENCY_ARCHITECTURE-4) and in one case (X_EFFICIENCY_ARCHITECTURE-4,
``_orthogonal_univariate_fe/__init__.py``) plausibly caused an entire 13-file/5542-LOC subpackage to be
skipped by every one of 26 independent audit-cluster agents in the SAME wave, because a docstring
confidently declaring "opt-in only, not production" is exactly the signal that makes an auditor
reasonably deprioritize claiming that file as in-scope.

Detection: regex-scan every module/function docstring in ``src/mlframe/feature_selection/filters/**``
for a "not wired" / "opt-in only" style phrase; for each hit, look up ``MRMR.__init__``'s corresponding
``fe_<family>_enable``-style constructor parameter (matched by the family-name overlap between the
docstring's own module/function name and the parameter name) and flag it only when that parameter's
own AST default literal is ``True``. Snapshot-style (baseline-diff), matching the established
``test_no_bare_except.py`` idiom.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_FILTERS_DIR = MLFRAME_DIR / "feature_selection" / "filters"
_MRMR_CLASS_PATH = _FILTERS_DIR / "mrmr" / "_mrmr_class.py"
_BASELINE_PATH = Path(__file__).resolve().parent / "_stale_not_wired_docstring_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "_benchmarks", "_vendored")

# The exact phrasing this bug class recurred under, per the audit reports' own citations.
_NOT_WIRED_RE = re.compile(
    r"not (currently |yet )?wired (in)?to\s+MRMR\.?fit" r"|not (enabled|active) by default" r"|benchmark[- ]script[- ]only" r"|not currently routed",
    re.IGNORECASE,
)


def _refresh_requested() -> bool:
    """True if ``--refresh-stale-not-wired-docstring-baseline`` was passed on the pytest command line."""
    return "--refresh-stale-not-wired-docstring-baseline" in sys.argv


def _true_default_enable_params() -> set[str]:
    """Every ``fe_*_enable``-style ``MRMR.__init__`` constructor parameter whose AST default is the
    literal ``True`` -- i.e. every mechanism that IS default-on in production."""
    tree = parsed_ast(_MRMR_CLASS_PATH)
    if tree is None:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
            continue
        args = node.args
        defaults = args.defaults
        # Positional/keyword-with-default args: defaults align to the LAST len(defaults) of args.args.
        params = args.args[len(args.args) - len(defaults) :] if defaults else []
        for param, default in zip(params, defaults):
            if isinstance(default, ast.Constant) and default.value is True:
                out.add(param.arg)
        # kwonly args each carry their own default slot (None where no default).
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and isinstance(default, ast.Constant) and default.value is True:
                out.add(param.arg)
        break  # first __init__ found is the real constructor
    return out


def _docstring_family_tokens(text: str) -> set[str]:
    """Cheap tokenization of a docstring's own words that could name an ``fe_<family>_enable`` flag --
    used only to narrow which true-default params are plausibly the SAME mechanism being described,
    not to prove it (a human still resolves the final call; false positives are expected and reviewed
    via the baseline diff, not auto-suppressed)."""
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()))


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every module/function docstring under ``filters/`` matching the stale
    "not wired" phrasing whose file/function name token-overlaps a true-default-on ``fe_*_enable`` param."""
    true_default_params = _true_default_enable_params()
    if not true_default_params:
        return set()
    out: set[str] = set()
    for py in _FILTERS_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        stem_tokens = _docstring_family_tokens(py.stem)
        module_doc = ast.get_docstring(tree, clean=False) or ""
        if _NOT_WIRED_RE.search(module_doc):
            doc_tokens = _docstring_family_tokens(module_doc)
            if any(_param_matches(p, stem_tokens | doc_tokens) for p in true_default_params):
                out.add(f"{rel}:1")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn_doc = ast.get_docstring(node, clean=False) or ""
            if not _NOT_WIRED_RE.search(fn_doc):
                continue
            fn_tokens = _docstring_family_tokens(node.name) | _docstring_family_tokens(fn_doc)
            if any(_param_matches(p, stem_tokens | fn_tokens) for p in true_default_params):
                out.add(f"{rel}:{node.lineno}")
    return out


def _param_matches(param_name: str, tokens: set[str]) -> bool:
    """True if ``param_name`` (e.g. ``fe_univariate_basis_enable``) shares a family-name token with the
    docstring/filename tokens (strips the ``fe_``/``_enable`` scaffolding, e.g. -> {"univariate","basis"})."""
    core = param_name.removeprefix("fe_").removesuffix("_enable")
    core_tokens = {t for t in core.split("_") if len(t) > 3}
    return bool(core_tokens & tokens)


def test_no_new_stale_not_wired_docstrings():
    """No new module/function docstring under ``filters/`` claims a default-on mechanism is opt-in/not
    wired into ``MRMR.fit``, beyond the frozen baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"stale-not-wired-docstring baseline refreshed at {_BASELINE_PATH.name} ({len(current)} site(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_stale_not_wired_docstrings] {len(fixed)} site(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-stale-not-wired-docstring-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new docstring(s) claim a mechanism is opt-in/not-wired-into-MRMR.fit while "
            f"MRMR.__init__ has a matching fe_*_enable parameter defaulting to True (default-on in "
            f"production). Correct the docstring to state the real default:\n  "
            + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
