"""The shared unary op-code table: one name per code for the numba ladder (``_numba_utils_usability``) and the CUDA stack kernel (``_unary_stack_cuda``).

The numeric codes are a contract between those two implementations and their callers, so they are defined once here and every other site refers to the name. The
resident operand build (``_gpu_resident_operand_build``) keeps its own numbering, which is a separate space and is not covered by this table.
"""

from __future__ import annotations

UNARY_OP_CODES: dict = {
    "identity": 0,
    "neg": 1,
    "abs": 2,
    "sqr": 3,
    "reciproc": 4,
    "sqrt": 5,
    "log": 6,
    "sin": 7,
    "sign": 8,
    "rint": 9,
    "qubed": 10,
    "invsquared": 11,
    "invqubed": 12,
    "cbrt": 13,
    "invcbrt": 14,
    "invsqrt": 15,
    "exp": 16,
}

__all__ = ["UNARY_OP_CODES"]
