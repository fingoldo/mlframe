"""Meta-test: no test sets ``NUMBA_DISABLE_CUDA`` / ``NUMBA_DISABLE_JIT`` and expects it to be undone later.

numba reads these two variables ONCE into an internal ``numba.core.config`` cache the first time
anything touches ``numba.cuda.is_available()`` (or numba config generally) in the process, and never
re-checks the live env var afterward -- confirmed directly: set the var, probe
``cuda.is_available()``, unset the var in the SAME process, probe again -- still returns the cached
value. This is fundamentally different from ``CUDA_VISIBLE_DEVICES`` (which mlframe's own
``gpu_globally_disabled()`` re-reads live via ``os.environ.get(...)`` on every call, no caching): a
"capture prior value + restore after the module/test" fixture -- whether hand-rolled or via pytest's
own ``monkeypatch.setenv`` -- puts the ENV VAR back correctly but cannot undo numba's own cache, so
it stays poisoned for the rest of the pytest-xdist worker process regardless, silently breaking every
LATER test's real/mocked ``cuda.is_available()`` expectation.

Real incident: ``tests/feature_selection/fe/adaptive/test_fe_rung_schedule.py`` set
``NUMBA_DISABLE_CUDA`` via exactly this restore-fixture pattern; the GPU-dispatch failure cluster it
caused reappeared on a later CI run despite the "fix". ``tests/feature_selection/mrmr/core/
test_mrmr_smooth_interaction_underselection.py`` carried the identical live bug, found and fixed in
the same session that added this meta-test (see git history for both files).

The only correct fix is to NOT set the variable in-process at all: rely on ``CUDA_VISIBLE_DEVICES``
(which has no caching problem) if the goal is "force CPU", or set the var in a subprocess's ``env``
dict (a fresh interpreter has no stale cache to poison) if the test genuinely needs
``NUMBA_DISABLE_CUDA``/``NUMBA_DISABLE_JIT`` itself. A subprocess ``env`` dict write is NOT flagged
here (the detector only looks at ``os.environ``/``monkeypatch.setenv`` mutation of the real process
env, not writes to an arbitrary local dict later passed to ``subprocess.run(env=...)``).

Baseline-diff (not zero-tolerance): run with ``--refresh-numba-config-env-baseline`` after reviewing
a new finding is either a genuine subprocess-only write the detector's heuristic couldn't tell apart,
or a deliberate exception with a documented reason.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_numba_config_env_mutation_baseline.json"

_FLAGGED_VARS = frozenset({"NUMBA_DISABLE_CUDA", "NUMBA_DISABLE_JIT"})
_MUTATING_ENVIRON_METHODS = frozenset({"setdefault", "update", "pop", "clear", "popitem"})


def _refresh_requested() -> bool:
    """True if ``--refresh-numba-config-env-baseline`` was passed on the pytest command line."""
    return "--refresh-numba-config-env-baseline" in sys.argv


def _is_environ(node: ast.AST) -> bool:
    """True for ``os.environ`` / a bare ``environ`` reference (both are the same process-wide mapping)."""
    if isinstance(node, ast.Attribute) and node.attr == "environ":
        return True
    return isinstance(node, ast.Name) and node.id == "environ"


def _const_str(node: ast.AST) -> "str | None":
    """The literal string value of a ``Constant`` node, or ``None`` if it isn't one."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _dict_literal_has_flagged_key(node: ast.AST) -> bool:
    """True if a dict literal (as passed to ``os.environ.update({...})``) has a flagged key."""
    if not isinstance(node, ast.Dict):
        return False
    return any(_const_str(k) in _FLAGGED_VARS for k in node.keys)


def _numba_config_env_mutations(tree: ast.Module) -> list[tuple[int, str]]:
    """``[(lineno, what), ...]`` for every real-process mutation of a flagged numba config var.

    Unlike the general module-env-mutation meta-test, this walks the WHOLE tree (not just import-time
    code) -- the caching footgun applies whether the mutation happens at module scope or inside a
    fixture/test function, since both run in the same live process.
    """
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _is_environ(t.value) and _const_str(t.slice) in _FLAGGED_VARS:
                    out.append((node.lineno, f"os.environ[{_const_str(t.slice)!r}] = ..."))
        elif isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            if f.attr in _MUTATING_ENVIRON_METHODS and _is_environ(f.value):
                if node.args and _const_str(node.args[0]) in _FLAGGED_VARS:
                    out.append((node.lineno, f"os.environ.{f.attr}({_const_str(node.args[0])!r}, ...)"))
                elif node.args and _dict_literal_has_flagged_key(node.args[0]):
                    out.append((node.lineno, f"os.environ.{f.attr}({{...}})"))
            elif f.attr == "putenv" and node.args and _const_str(node.args[0]) in _FLAGGED_VARS:
                out.append((node.lineno, f"os.putenv({_const_str(node.args[0])!r}, ...)"))
            elif f.attr == "setenv" and len(node.args) >= 1 and _const_str(node.args[0]) in _FLAGGED_VARS:
                # monkeypatch.setenv(...) -- pytest restores the ENV VAR at teardown, but that does
                # nothing for numba's own in-process cache, so this is just as unsafe as a bare
                # os.environ write for these two specific variables.
                out.append((node.lineno, f"monkeypatch.setenv({_const_str(node.args[0])!r}, ...)"))
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:what", ...}`` for every flagged numba-config env mutation under ``tests/``."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for lineno, what in _numba_config_env_mutations(tree):
            out.add(f"{rel}:{lineno}:{what}")
    return out


def test_no_new_numba_config_env_mutation():
    """No test module gains a new real-process mutation of NUMBA_DISABLE_CUDA/NUMBA_DISABLE_JIT."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"numba-config-env-mutation baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_numba_config_env_mutation] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-numba-config-env-baseline\n"
        )

    assert not added, (
        f"{len(added)} new real-process mutation(s) of NUMBA_DISABLE_CUDA/NUMBA_DISABLE_JIT. numba caches "
        "these into its internal config the first time anything touches numba.cuda.is_available() and never "
        "re-reads them -- restoring the env var afterward (via a hand-rolled fixture OR monkeypatch.setenv) "
        "does NOT undo numba's own cache, so it stays poisoned for every later test in this pytest-xdist "
        "worker. Use CUDA_VISIBLE_DEVICES instead (no caching problem, re-read live) if the goal is forcing "
        "CPU, or set it in a subprocess env dict (fresh interpreter, no stale cache) if the var itself is "
        "genuinely required. See tests/feature_selection/fe/adaptive/test_fe_rung_schedule.py's docstring for "
        "the full incident writeup. If this is a genuine subprocess-only write the detector misclassified, "
        "re-run with --refresh-numba-config-env-baseline after review.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
import os

os.environ["NUMBA_DISABLE_CUDA"] = "1"
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.update({"NUMBA_DISABLE_CUDA": "1", "OTHER": "2"})
os.putenv("NUMBA_DISABLE_CUDA", "1")

def test_uses_monkeypatch(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_CUDA", "1")

def safe_subprocess_env():
    env = dict(os.environ)
    env["NUMBA_DISABLE_CUDA"] = "1"
    return env

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MLFRAME_DISABLE_HNSW", "1")
'''


def test_detector_flags_all_real_mutation_shapes_and_ignores_subprocess_dict():
    """The scan flags all five real os.environ/monkeypatch mutation shapes and ignores the local-dict write."""
    found = _numba_config_env_mutations(ast.parse(_DETECTOR_SAMPLE))
    assert sorted(ln for ln, _ in found) == [4, 5, 6, 7, 10], found
