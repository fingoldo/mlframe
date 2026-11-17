"""The serial usability kernels no longer raise ZeroDivisionError on an exact zero and agree with the parallel ones (tracker 2026-11-02 wave 12, NJ12-1)."""

import numpy as np

from mlframe.feature_selection.filters import _numba_utils_usability as nu
from mlframe.feature_selection.filters import _usability_njit_pool as pool


def test_serial_and_parallel_agree_when_a_column_holds_an_exact_zero():
    """Unary code 4 is 1/v; the serial kernels used Python's error model and raised on v == 0 while the parallel ones returned inf."""
    args = list(pool._make_usability_inputs({"n_rows": 2_000, "n_combos": 20}))
    args[0] = args[0].copy()
    args[0][::50] = 0.0
    args[6] = np.full_like(args[6], 4)  # every combo applies the reciprocal to the first operand
    serial = nu._pair_combo_mi_njit(*args)
    parallel = nu._pair_combo_mi_njit_parallel(*args)
    np.testing.assert_allclose(serial, parallel, atol=1e-12)


def test_inverse_power_unaries_give_the_registry_ceiling_at_an_exact_zero():
    """Codes 4, 11, 12, 14, 15 returned inf at 0.0 (or -0.0) where the registry's safe power returns its 1e9 ceiling, so the pool ranked combos on a different column than the one it keeps."""
    from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

    funcs = create_unary_transformations(preset="medium")
    names = {4: "reciproc", 11: "invsquared", 12: "invqubed", 14: "invcbrt", 15: "invsqrt"}
    for code, name in names.items():
        for zero in (0.0, -0.0):
            assert nu._apply_unary(zero, code, 1.0) == float(funcs[name](np.array([zero]))[0]), (code, name)
        assert nu._apply_unary(2.0, code, 1.0) == float(funcs[name](np.array([2.0]))[0]), (code, name)


def test_cupy_twin_inverse_power_unaries_give_the_registry_ceiling_at_zero():
    """The vectorised twin of the inverse-power unaries (driven with numpy standing in for cupy) matches the registry at zeros and elsewhere."""
    from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

    funcs = create_unary_transformations(preset="medium")
    x = np.array([0.0, -0.0, 2.0, 0.5])
    for code, name in {4: "reciproc", 11: "invsquared", 12: "invqubed", 14: "invcbrt", 15: "invsqrt"}.items():
        np.testing.assert_allclose(pool._GPU_UNARY_OPS[code](np, x, 1.0), funcs[name](x), rtol=1e-12, err_msg=name)
