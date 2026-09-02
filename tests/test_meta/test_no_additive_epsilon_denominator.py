"""Meta-test: a division must not be guarded by ADDING a small constant to its denominator.

Distilled from the 2026-09-01 audit, where this exact shape produced eight separate findings across four
subsystems -- FEATURE_ENGINEERING-6 / -10 / -15, FS_FILTERS_MRMR-8, XCUT_NUMERICAL_STABILITY-5, METRICS-15,
REMAINING_SUBSYSTEMS and the `+1e-12` pads in `anchor.py`. Every one had the same two properties, and neither is
obvious at the call site:

* **The pad is only harmless when the denominator's natural scale is far ABOVE it.** That is an assumption about
  the data, not about the code, and it fails silently in ordinary regimes -- an exponentially-decayed weighted
  variance (``0.5 ** (1/half_life)`` compounding every row), a band energy in SQUARED input units on log-returns
  at 1e-3 amplitude, the range of a large-offset near-constant window, a per-category variance of 1e-6 whose
  square is 1e-12. Measured consequences from that audit: a slope reported as exactly 0.0 against a true 1.0; a
  spectral ratio wrong by 237x purely from rescaling the same signal; a skew of 2.0 read as 0.25, which flips a
  basis-routing branch.
* **It does not guard the division, it REPLACES the answer.** `x / (d + eps)` returns a plausible finite number
  for every input, so there is no NaN, no warning, and nothing downstream can tell a real ratio from a padded
  one.

The sanctioned form is an explicit degeneracy branch: ``x / d if d > <threshold> else <documented value>``, or
``np.where(d > 0, x / np.where(d > 0, d, 1), fill)`` for the vectorised case, with the threshold RELATIVE to the
denominator's own scale when that scale can shrink. The degenerate value is then a deliberate choice a reader
can see, rather than an arbitrary bias.

Detector: an ``ast.Div`` whose right operand is ``<anything> + <float literal below 1e-3>`` (either operand
order), or a ``Name`` divided in that shape where the name was bound to such a sum in the same function. Small
literals only, so ``x / (n + 1)`` and similar honest arithmetic are not flagged.

Baseline-diffed, like the sibling meta-tests: pre-existing sites are grandfathered and only NEW ones fail.
Refresh with ``--refresh-additive-epsilon-baseline`` after reviewing a finding.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_BASELINE_PATH = Path(__file__).resolve().parent / "_additive_epsilon_baseline.json"

# Only genuinely epsilon-sized constants. `x / (n + 1)` (a Laplace count smoothing) and `x / (d + 0.5)` are
# ordinary arithmetic, not a degeneracy pad, and must not be flagged.
_EPSILON_MAX = 1e-3


def _refresh_requested() -> bool:
    """True if ``--refresh-additive-epsilon-baseline`` was passed on the pytest command line."""
    return "--refresh-additive-epsilon-baseline" in sys.argv


def _is_epsilon_constant(node: ast.AST) -> bool:
    """True for a positive float literal small enough to be a degeneracy pad rather than real arithmetic."""
    return isinstance(node, ast.Constant) and isinstance(node.value, float) and 0.0 < node.value <= _EPSILON_MAX


def _is_epsilon_sum(node: ast.AST) -> bool:
    """True for ``<expr> + <epsilon>`` or ``<epsilon> + <expr>``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    return _is_epsilon_constant(node.right) or _is_epsilon_constant(node.left)


def _epsilon_padded_names(func: ast.AST) -> set:
    """Names bound in ``func`` to an epsilon-padded sum -- ``denom = var + 1e-12`` then ``x / denom``."""
    names: set = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name) and _is_epsilon_sum(node.value):
            names.add(node.targets[0].id)
    return names


def _padded_divisions(tree: ast.Module) -> list:
    """``[(lineno, context), ...]`` for every division whose denominator carries an additive epsilon."""
    out: list = []
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] or [tree]
    for scope in scopes:
        padded = _epsilon_padded_names(scope)
        scope_name = getattr(scope, "name", "<module>")
        for node in ast.walk(scope):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            rhs = node.right
            if _is_epsilon_sum(rhs) or (isinstance(rhs, ast.Name) and rhs.id in padded):
                out.append((node.lineno, scope_name))
    return out


def _build_offending_set() -> set:
    """``{"relpath:lineno:scope", ...}`` for every epsilon-padded division under ``src/mlframe``."""
    out: set = set()
    for py in _SRC_DIR.rglob("*.py"):
        # `_benchmarks/` deliberately preserves superseded formulations for A/B comparison, so a pad there is
        # the point rather than a defect.
        if "__pycache__" in py.parts or "_benchmarks" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_SRC_DIR).as_posix()
        for lineno, scope in _padded_divisions(tree):
            out.add(f"{rel}:{lineno}:{scope}")
    return out


def test_no_new_additive_epsilon_denominator():
    """No new division guarded by adding an epsilon to its denominator."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"additive-epsilon baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    assert not added, (
        "New additive-epsilon denominator(s):\n  "
        + "\n  ".join(added)
        + "\n\nAdding a constant to a denominator does not guard the division -- it replaces the answer with a "
        "plausible-looking wrong one for every input in the regime where the pad is comparable to the true "
        "denominator, with no NaN and no warning. Use an explicit degeneracy branch instead:\n"
        "    x / d if d > threshold else <documented degenerate value>\n"
        "    np.where(d > 0, x / np.where(d > 0, d, 1.0), fill)\n"
        "and make the threshold RELATIVE to the denominator's own scale when that scale can shrink (a decayed "
        "weighted variance, a squared-units energy, the range of a large-offset column).\n"
        "If this really is ordinary arithmetic rather than a pad, refresh the baseline with "
        "--refresh-additive-epsilon-baseline."
    )
