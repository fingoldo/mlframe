"""Remove from ``tests/test_meta/_gpu_kernel_pointer_dtype_baseline.json`` the kernels that no longer declare a bare ``double*`` pointer.

The baseline only shrinks: this never adds an entry (a new kernel is templated on ``{REAL}``/``{ACC}``, or reviewed by hand with a reason). Usage: python scripts/shrink_kernel_dtype_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_meta import test_gpu_kernel_pointer_dtype as T  # noqa: E402


def main() -> int:
    """Rewrite the baseline without the fixed kernels; the count of removed entries is the exit message."""
    current = T._double_pointer_kernels()
    raw = orjson.loads(T.BASELINE.read_bytes())
    out, removed = {}, 0
    for rel, kernels in raw.items():
        keep = {k: r for k, r in kernels.items() if k in current.get(rel, [])}
        removed += len(kernels) - len(keep)
        if keep:
            out[rel] = keep
    T.BASELINE.write_bytes(orjson.dumps(out, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")
    sys.stdout.write(f"removed {removed} fixed kernel(s); {sum(len(v) for v in out.values())} remain\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
