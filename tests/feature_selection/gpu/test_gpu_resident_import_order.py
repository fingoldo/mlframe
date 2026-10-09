"""The names `_gpu_resident_fe` re-exports resolve whichever carved module a process imports first.

`_gpu_resident_select` imports `_gpu_resident_discretize` at its tail, which back-imports `_gpu_resident_fe`, whose own tail copies names from `_gpu_resident_select` while that is
only partly initialised. Importing the select module first used to leave `gpu_discretize_codes_host` and friends out of `_gpu_resident_fe`, so a test (or a script) that happened
to import it first failed with an ImportError that depended on nothing but import order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import mlframe

_SRC = str(Path(mlframe.__file__).resolve().parents[1])
_FIRST = [
    "_gpu_resident_select",
    "_gpu_resident_select_kernels",
    "_gpu_resident_discretize",
    "_gpu_resident_materialise",
    "_gpu_resident_basis",
    "_gpu_resident_pair_mi",
    "_gpu_resident_k_chunk_ktc",
    "_gpu_resident_fe",
]
_NAMES = ["gpu_discretize_codes_host", "gpu_materialise_discretize_codes_host", "_radix_select_interior_edges", "_transpose_to_cm", "_get_fused_gen_cm_kernel"]


@pytest.mark.parametrize("first", _FIRST)
def test_reexports_resolve_in_a_fresh_process(first):
    """Import `first`, then every re-exported name from `_gpu_resident_fe`."""
    code = (
        "import importlib, sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        f"importlib.import_module('mlframe.feature_selection.filters.{first}')\n"
        f"from mlframe.feature_selection.filters._gpu_resident_fe import {', '.join(_NAMES)}\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert result.returncode == 0 and result.stdout.strip().endswith("ok"), result.stderr[-600:]


def test_an_unknown_name_still_raises_attribute_error():
    """The late lookup does not turn typos into silent successes."""
    import mlframe.feature_selection.filters._gpu_resident_fe as fe

    with pytest.raises(AttributeError):
        fe.this_name_does_not_exist  # noqa: B018
