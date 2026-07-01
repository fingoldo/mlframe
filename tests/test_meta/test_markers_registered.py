"""Regression sensor: every pytest marker actually used in tests/**.py
must be registered in pyproject.toml ``[tool.pytest.ini_options].markers``
OR in tests/conftest.py ``pytest_configure`` via ``addinivalue_line``.

Under ``--strict-markers`` (active per pyproject), unregistered marker
references trigger collection-time errors. This sensor scans every
``@pytest.mark.<name>`` and ``pytestmark = pytest.mark.<name>`` (incl.
list/tuple forms) reference under ``tests/`` and asserts the marker
name appears in the union of pyproject + conftest registered markers.

Sensor for S27: ``no_xdist_parallel`` was referenced in
``tests/training/test_biz_val_pysr_fe_upgrade.py`` docstring (and not
registered) — under ``--strict-markers`` this is at minimum a
documentation footgun (copy-paste of the docstring snippet into actual
code would explode at collection). This test catches every such drift.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# Built-in pytest markers that don't need explicit registration.
_BUILTIN_MARKERS = frozenset({
    "skip", "skipif", "xfail", "parametrize", "usefixtures", "filterwarnings",
    "tryfirst", "trylast", "order", "timeout",
})

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CONFTEST = _TESTS_DIR / "conftest.py"


def _parse_pyproject_markers() -> set[str]:
    """Parse pyproject.toml ``markers = [...]`` list under
    ``[tool.pytest.ini_options]``. Each entry shape is ``"name: descr"``.
    Returns the set of registered marker names.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(_PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    raw = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", []) or []
    out: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        name = entry.split(":", 1)[0].strip()
        if name:
            out.add(name)
    return out


def _parse_conftest_markers() -> set[str]:
    """Parse tests/conftest.py for ``config.addinivalue_line("markers",
    "<name>: ...")`` calls. Returns marker names registered dynamically.
    """
    text = _CONFTEST.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "addinivalue_line"
        ):
            continue
        # call signature: config.addinivalue_line("markers", "<name>: descr")
        if len(node.args) < 2:
            continue
        first = node.args[0]
        second = node.args[1]
        if not (isinstance(first, ast.Constant) and first.value == "markers"):
            continue
        if not (isinstance(second, ast.Constant) and isinstance(second.value, str)):
            continue
        name = second.value.split(":", 1)[0].strip()
        if name:
            out.add(name)
    return out


# Regex for ``@pytest.mark.<name>`` (decorator form, ignoring args after).
# Use a non-greedy character class so we capture the identifier only.
_DECORATOR_MARK_RE = re.compile(r"@pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
# Regex for ``pytest.mark.<name>`` inside pytestmark assignments or list
# literals. Same shape but no leading ``@``.
_INLINE_MARK_RE = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")


def _scan_used_markers() -> dict[str, list[str]]:
    """Walk every .py file under ``tests/`` (excluding __pycache__ and our
    own meta-test file) and collect marker references.

    Returns {marker_name: [list of files where seen]}. Scans whole-file
    text (including docstrings) because a docstring that advertises
    ``@pytest.mark.X`` for an unregistered ``X`` is a foot-gun: future
    maintainers paste the snippet into a real decorator and trip
    collection. The S27 incident was exactly this shape — the marker
    name only appeared in a docstring but was still a real bug class.
    """
    seen: dict[str, list[str]] = {}
    for path in _TESTS_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        names = set(_DECORATOR_MARK_RE.findall(text))
        names |= set(_INLINE_MARK_RE.findall(text))
        for n in names:
            seen.setdefault(n, []).append(str(path.relative_to(_REPO_ROOT)))
    return seen


def test_every_used_marker_is_registered():
    pyproject_markers = _parse_pyproject_markers()
    conftest_markers = _parse_conftest_markers()
    registered = pyproject_markers | conftest_markers | _BUILTIN_MARKERS
    used = _scan_used_markers()

    unregistered: dict[str, list[str]] = {
        name: paths for name, paths in used.items() if name not in registered
    }
    assert not unregistered, (
        f"Unregistered pytest markers found (collection under --strict-markers "
        f"would error or warn):\n"
        + "\n".join(
            f"  {name}: seen in {paths}" for name, paths in sorted(unregistered.items())
        )
        + f"\n\nRegister in pyproject.toml ``[tool.pytest.ini_options].markers`` "
        f"or in tests/conftest.py ``pytest_configure``."
    )


def test_pyproject_marker_parser_picks_up_known_entries():
    """Smoke: parser sees the known registered names. Guards against a
    pyproject schema rename breaking the marker test silently.
    """
    pyproject_markers = _parse_pyproject_markers()
    for required in ("slow", "fast", "gpu", "no_xdist"):
        assert required in pyproject_markers, (
            f"Marker {required!r} missing from pyproject.toml registration; "
            f"either pyproject was edited or the parser is broken."
        )


def test_conftest_marker_parser_picks_up_known_entries():
    """Smoke: parser sees conftest-registered names.

    ``slow_only`` / ``no_xdist`` are pyproject-registered
    (see ``[tool.pytest.ini_options].markers`` in pyproject.toml) per the
    D2-P2-#18 dedup; only conftest-private markers (``fuzz``,
    ``biz_transformer`` -- both gated by ``--run-*`` option flags) belong
    here. Pyproject-side registration is checked by the sibling
    ``test_pyproject_marker_parser_picks_up_known_entries``.
    """
    conftest_markers = _parse_conftest_markers()
    for required in ("fuzz", "biz_transformer"):
        assert required in conftest_markers, (
            f"Marker {required!r} missing from conftest.py addinivalue_line; "
            f"either conftest was edited or the parser is broken."
        )
