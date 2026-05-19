"""Regenerate every ``_*_baseline.json`` in this directory.

JSON does not allow inline comments, so the canonical regen instructions live
in ``BASELINES_README.md`` and this script. Run from the repository root:

    python tests/test_meta/regen_baselines.py

The actual regeneration logic for each baseline lives inside the corresponding
meta-test module (``test_public_annotations``, ``test_no_bare_except``, ...)
under a ``regenerate_baseline`` helper.  This entry-point simply iterates over
the known set and delegates.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Mapping: baseline filename -> (test module, regenerator attribute name).
# Each meta-test module is responsible for owning its regenerator; if a module
# omits one we surface a clear warning rather than silently doing nothing.
_BASELINES: dict[str, tuple[str, str]] = {
    "_annotation_baseline.json": ("test_public_annotations", "regenerate_baseline"),
    "_bare_except_baseline.json": ("test_no_bare_except", "regenerate_baseline"),
    "_console_unicode_baseline.json": ("test_no_unicode_in_console_output", "regenerate_baseline"),
    "_debt_baseline.json": ("test_deferred_drift", "regenerate_baseline"),
    "_docstring_baseline.json": ("test_public_docstrings", "regenerate_baseline"),
    "_logger_lazy_baseline.json": ("test_logger_lazy_formatting", "regenerate_baseline"),
    "_mutable_defaults_baseline.json": ("test_no_mutable_defaults", "regenerate_baseline"),
    "_resource_handle_baseline.json": ("test_resource_handle_safety", "regenerate_baseline"),
}


def _regen_one(baseline: str, module_name: str, attr: str) -> bool:
    full = f"tests.test_meta.{module_name}"
    try:
        mod = importlib.import_module(full)
    except Exception as exc:
        print(f"[skip] {baseline}: cannot import {full}: {exc}", file=sys.stderr)
        return False
    fn = getattr(mod, attr, None)
    if fn is None:
        print(f"[skip] {baseline}: {full} has no '{attr}' helper", file=sys.stderr)
        return False
    fn(_HERE / baseline)
    print(f"[ok]   {baseline} regenerated via {full}.{attr}")
    return True


def main() -> int:
    ok = 0
    for baseline, (mod, attr) in _BASELINES.items():
        if _regen_one(baseline, mod, attr):
            ok += 1
    print(f"\nRegenerated {ok}/{len(_BASELINES)} baselines.")
    return 0 if ok == len(_BASELINES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
