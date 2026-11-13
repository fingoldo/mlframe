"""The shared ``cuda_kernel_pointer_types`` gate over the whole package: no untemplated ``double*`` kernel parameter, no kernel-type probe off the launched buffer.

The reason a kernel stays on a bare pointer type lives beside the kernel as ``// dtype-ok: <reason>``, where the next person editing it reads it; the older mlframe-only form of this check with a JSON baseline was retired.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared import cuda_kernel_pointer_types as gate

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_kernel_takes_an_untemplated_double_pointer_without_a_review_marker():
    """Every ``double*`` kernel parameter is templated on ``{REAL}`` or carries ``// dtype-ok: <reason>``; no type probe names an array that is cast before the launch."""
    gate.assert_cuda_kernel_pointer_types(SRC, min_files=500)
