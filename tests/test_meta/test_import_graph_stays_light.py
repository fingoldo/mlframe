"""Importing the reporting or preprocessing package must not drag in the feature-selection filters stack.

`reporting.charts.fuzzy_membership` imported `feature_engineering.fuzzy_features` at module scope (which imports
`feature_selection.drop_near_noise_univariate_auc`), and `preprocessing.auto_transform_select` imported
`feature_selection.filters` directly. Both therefore paid the filters stack - cupy, the numba.cuda probe and the
typed-dict warm-up - on every import: measured `import mlframe.reporting` 22.2 s and `import mlframe.preprocessing`
19.6-21.2 s, against `import mlframe.metrics` 5.3 s, and the same cost landed in every loky worker.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PROBE = (
    "import sys; import {mod}; "
    "leaked = sorted(m for m in sys.modules if m.startswith('mlframe.feature_selection.filters')); "
    "print('LEAKED', leaked)"
)


def _leaked_modules(mod: str) -> list[str]:
    out = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("LEAKED"))
    return eval(line[len("LEAKED ") :])  # noqa: S307 - our own literal list, printed two lines above


@pytest.mark.parametrize("mod", ["mlframe.reporting", "mlframe.preprocessing"])
def test_the_filters_stack_is_not_imported(mod):
    leaked = _leaked_modules(mod)
    assert not leaked, f"{mod} pulled in {leaked}; import those lazily inside the functions that need them"


def test_the_numba_typed_dict_warmup_does_not_run_at_import():
    """The warm-up belongs to the first `screen_predictors` call, not to every process that imports the module."""
    probe = (
        "import mlframe.feature_selection.filters._screen_predictors as sp; "
        "import mlframe.feature_selection.filters._numba_warmup as nw; print('DONE', nw._warmup_done)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "DONE False" in out.stdout, "importing the module ran the multi-second warm-up"
