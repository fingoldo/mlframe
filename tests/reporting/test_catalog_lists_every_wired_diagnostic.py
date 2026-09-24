"""Every diagnostic the dispatchers record must be listed in the catalogue that describes the suite's output.

``catalog.py`` promised this check by name long before it existed, and the list had in the meantime lost
``risk_coverage``. The recorded names are collected structurally - every ``_record(charts, "<name>", ...)`` call
in the reporting packages, found by parsing them - so a diagnostic added anywhere cannot go unlisted.
"""

import ast
from pathlib import Path

import mlframe
from mlframe.reporting import catalog

_SRC = Path(mlframe.__file__).resolve().parent
_PACKAGES = (_SRC / "reporting", _SRC / "training" / "reporting")


def _recorded_names() -> set:
    names = set()
    for pkg in _PACKAGES:
        for path in pkg.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("_record", "_record_path", "_record_skipped")
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    names.add(node.args[1].value)
    return names


def test_catalog_lists_every_wired_diagnostic():
    recorded = _recorded_names()
    assert len(recorded) >= 20, f"only {len(recorded)} recorded diagnostics found - the scan lost its subject"
    listed = {name for name, _ in catalog._STANDALONE_DIAGNOSTICS}
    missing = sorted(recorded - listed)
    assert missing == [], f"diagnostics the dispatchers record but the catalogue does not list: {missing}"


def test_every_catalogue_row_has_a_description():
    assert catalog._STANDALONE_DIAGNOSTICS
    for name, desc in catalog._STANDALONE_DIAGNOSTICS:
        assert name and len(desc) > 20, f"{name!r} needs a real description"
