"""Sensor for W5 PySR sympy-mappings semantic fix.

`safe_log` / `safe_sqrt` / `harmonic_mean` / `xlogy` at train time run inside Julia and return
NaN for out-of-domain inputs. The legacy sympy mappings used ``sp.log(sp.Abs(x) + 1e-9)`` style
substitutes that returned finite values at predict time, so a model trained on Julia's NaN-out
semantics saw a DIFFERENT function at predict time on negative / zero inputs. Closes A2 P2.

The Piecewise replacements faithfully replicate the Julia branches: in-domain -> finite math;
out-of-domain -> NaN (or the natural extension for safe_sqrt, which uses the absolute value).
"""

from __future__ import annotations

import math


def _mappings():
    """Helper: Mappings."""
    from mlframe.feature_engineering.pysr_operators import _make_extra_sympy_mappings

    return _make_extra_sympy_mappings()


def test_safe_log_returns_nan_on_nonpositive():
    """Safe log returns nan on nonpositive."""
    import sympy as sp

    m = _mappings()
    expr = m["safe_log"](sp.Symbol("x"))
    # x = -1 -> NaN
    assert sp.simplify(expr.subs(sp.Symbol("x"), -1.0)) == sp.nan
    # x = 0 -> NaN
    assert sp.simplify(expr.subs(sp.Symbol("x"), 0.0)) == sp.nan
    # x = 2 -> log(2)
    v = float(expr.subs(sp.Symbol("x"), 2.0))
    assert abs(v - math.log(2.0)) < 1e-12


def test_xlogy_returns_nan_on_nonpositive_y():
    """Xlogy returns nan on nonpositive y."""
    import sympy as sp

    m = _mappings()
    x, y = sp.symbols("x y")
    expr = m["xlogy"](x, y)
    assert sp.simplify(expr.subs({x: 3.0, y: -1.0})) == sp.nan
    assert sp.simplify(expr.subs({x: 3.0, y: 0.0})) == sp.nan
    assert abs(float(expr.subs({x: 3.0, y: 4.0})) - 3.0 * math.log(4.0)) < 1e-12


def test_harmonic_mean_returns_nan_on_nonpositive_sum():
    """Harmonic mean returns nan on nonpositive sum."""
    import sympy as sp

    m = _mappings()
    x, y = sp.symbols("x y")
    expr = m["harmonic_mean"](x, y)
    # x + y = 0 (degenerate) -> NaN (and crucially NOT the legacy 2xy/0 -> zoo silently)
    assert sp.simplify(expr.subs({x: 1.0, y: -1.0})) == sp.nan
    # x + y < 0 -> NaN
    assert sp.simplify(expr.subs({x: -2.0, y: -3.0})) == sp.nan
    # in-domain
    v = float(expr.subs({x: 1.0, y: 3.0}))
    assert abs(v - 2.0 * 1.0 * 3.0 / 4.0) < 1e-12


def test_safe_sqrt_uses_abs_for_negative():
    """Safe sqrt uses abs for negative."""
    import sympy as sp

    m = _mappings()
    x = sp.Symbol("x")
    expr = m["safe_sqrt"](x)
    # non-negative -> sqrt(x)
    assert abs(float(expr.subs(x, 9.0)) - 3.0) < 1e-12
    # negative -> sqrt(-x); deliberate behaviour: matches Julia's safe_sqrt(x)=sqrt(abs(x)) train-time.
    assert abs(float(expr.subs(x, -4.0)) - 2.0) < 1e-12
