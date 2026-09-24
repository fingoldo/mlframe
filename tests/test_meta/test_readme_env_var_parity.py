"""Meta-test: every environment variable production code reads via
``os.environ.get(...)``/``os.getenv(...)`` is documented in
docs/ENVIRONMENT_VARIABLES.md.

Uses the baseline/grandfather variant of the shared ``py_ci_shared.readme_env_var_parity``
check: mlframe has no "## Environment variables" section yet, so every var currently
read is grandfathered on first run -- only a NEW undocumented var (introduced after this
baseline was captured) fails. Documenting the grandfathered vars is a separate, deliberate
improvement this check doesn't demand up front.

The table itself lives in docs/ENVIRONMENT_VARIABLES.md (moved out of README.md, which only
keeps a one-line pointer) -- the shared checker's ``readme_path`` param accepts any markdown
file with a matching heading, not literally README.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from py_ci_shared.readme_env_var_parity import assert_no_new_undocumented_env_vars

import mlframe

_SCRIPT = Path(mlframe.__file__).resolve().parent.parent.parent / "scripts" / "gen_environment_variables_doc.py"
_spec = importlib.util.spec_from_file_location("gen_environment_variables_doc", _SCRIPT)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
README_PATH = Path(mlframe.__file__).resolve().parent.parent.parent / "docs" / "ENVIRONMENT_VARIABLES.md"
_BASELINE_PATH = Path(__file__).resolve().parent / "_readme_env_var_baseline.json"

# Mirrors this repo's pytest addopts (--ignore=legacy --ignore=benchmarks --ignore=profiling).
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "benchmarks", "profiling", "explore")


def _production_py_files() -> list[Path]:
    """Every production ``.py`` file under mlframe/, outside tests/legacy/benchmarks/profiling."""
    return [py for py in MLFRAME_DIR.rglob("*.py") if not any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS)]


def test_no_new_undocumented_env_vars():
    """Fail only on an env var read by production code that isn't in the grandfathered baseline.

    The scan also sees names bound to string constants and reads through the package's env helpers, so a switch that
    moves behind ``env_int`` / ``env_flag`` stays covered.
    """
    assert_no_new_undocumented_env_vars(
        files=_production_py_files(),
        readme_path=README_PATH,
        baseline_path=_BASELINE_PATH,
        reader_funcs=_gen.READER_FUNCS,
    )


def test_environment_variables_doc_is_current():
    """docs/ENVIRONMENT_VARIABLES.md is exactly what scripts/gen_environment_variables_doc.py generates."""
    assert README_PATH.read_text(encoding="utf-8") == _gen.render_markdown(), "run python scripts/gen_environment_variables_doc.py"
