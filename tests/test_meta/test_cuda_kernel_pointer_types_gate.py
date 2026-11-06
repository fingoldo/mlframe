"""The shared ``cuda_kernel_pointer_types`` gate over the whole package: no untemplated ``double*`` kernel parameter, no kernel-type probe off the launched buffer.

``test_gpu_kernel_pointer_dtype.py`` is the older, mlframe-only form of this check with a JSON baseline of reasons. The gate keeps each reason next to the kernel as ``// dtype-ok: <reason>``,
which is where the next person editing the kernel reads it. Once py-ci-shared releases the gate and this repository pins that release, the JSON baseline and the older test go.
"""

from __future__ import annotations

from pathlib import Path

import pytest

gate = pytest.importorskip("py_ci_shared.cuda_kernel_pointer_types", reason="the installed py-ci-shared predates the cuda_kernel_pointer_types gate; test_gpu_kernel_pointer_dtype.py covers this meanwhile")

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_kernel_takes_an_untemplated_double_pointer_without_a_review_marker():
    """Every ``double*`` kernel parameter is templated on ``{REAL}`` or carries ``// dtype-ok: <reason>``; no type probe names an array that is cast before the launch."""
    gate.assert_cuda_kernel_pointer_types(SRC, min_files=500)
