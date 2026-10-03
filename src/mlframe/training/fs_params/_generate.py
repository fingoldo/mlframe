"""Regenerate (or verify) the strict-parameter model modules in this package from the selector constructors.

python -m mlframe.training.fs_params._generate            # rewrite every generated module
python -m mlframe.training.fs_params._generate --check    # exit 1 if any module on disk differs from what would be generated
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .._param_model_spec import LIGHT_MODULES, render_spec  # noqa: F401 -- LIGHT_MODULES kept importable from here
from ._spec import SPECS, SelectorSpec

PACKAGE_DIR = Path(__file__).parent


def render(spec: SelectorSpec) -> str:
    """Source of the module for one selector."""
    return render_spec(spec, "python -m mlframe.training.fs_params._generate")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write or check every generated module; ``--check`` returns 1 and names the stale files."""
    from .._param_model_spec import run_generator

    return run_generator(SPECS, PACKAGE_DIR, "mlframe.training.fs_params._generate", argv)


if __name__ == "__main__":
    raise SystemExit(main())
