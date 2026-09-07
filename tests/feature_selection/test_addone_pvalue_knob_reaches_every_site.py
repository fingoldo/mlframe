"""`MLFRAME_MRMR_ADDONE_PVALUE` must reach every permutation p-value in the package, not one of five.

`_perm_pvalue` is the canonical helper and carries two later additions: the full-budget denominator
correction and this env opt-out, documented in docs/ENVIRONMENT_VARIABLES.md. Four other sites hardcoded
`(1 + count) / (1 + n)` inline and consulted neither.

An operator setting the var to reproduce legacy selection therefore got a run mixing two conventions:
measured with the var set, the canonical helper returns 0.0 for 0 exceedances in 50 while an inline site
still returned 1/51 = 0.0196. The significance gates in `permutation.py` and `_cmi_perm_stop.py` then
disagreed about the same feature at any alpha between those two values. With the var unset all five agree,
which is why the drift was latent.

One of the four also shadowed the canonical name with a different signature, so a grep for `_perm_pvalue`
found it and read as though it were the same function.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.permutation import _perm_pvalue, perm_pvalues

LEGACY_INLINE = 1.0 / 51.0  # what the hardcoded (1 + 0) / (1 + 50) returned


@pytest.fixture
def plain_rate(monkeypatch):
    """Select the legacy plain-rate estimator, the setting the drift was invisible without."""
    monkeypatch.setenv("MLFRAME_MRMR_ADDONE_PVALUE", "0")


@pytest.fixture
def add_one(monkeypatch):
    """The default add-one estimator."""
    monkeypatch.delenv("MLFRAME_MRMR_ADDONE_PVALUE", raising=False)


def test_the_scalar_helper_honours_the_opt_out(plain_rate):
    """The measurement from the finding: 0.0 rather than 1/51."""
    assert _perm_pvalue(0, 50) == 0.0


def test_the_vectorised_helper_agrees_with_the_scalar_one(plain_rate):
    """The array site scores many features at once and must not answer differently."""
    counts = np.array([0, 1, 7, 50])
    assert np.allclose(perm_pvalues(counts, 50), [_perm_pvalue(int(c), 50) for c in counts])


def test_the_vectorised_helper_agrees_with_the_scalar_one_by_default(add_one):
    """Both conventions, not just the opted-out one."""
    counts = np.array([0, 1, 7, 50])
    assert np.allclose(perm_pvalues(counts, 50), [_perm_pvalue(int(c), 50) for c in counts])
    assert perm_pvalues(np.array([0]), 50)[0] == pytest.approx(LEGACY_INLINE)


def test_the_opt_out_actually_changes_the_answer(plain_rate, monkeypatch):
    """Guards the fixture: if the knob stopped being read, every test here would pass for nothing."""
    opted_out = _perm_pvalue(0, 50)
    monkeypatch.delenv("MLFRAME_MRMR_ADDONE_PVALUE")
    assert _perm_pvalue(0, 50) != opted_out


@pytest.mark.parametrize(
    "module_name,call",
    [
        ("mlframe.feature_selection.filters._cmi_perm_stop", "cmi_permutation_stop"),
        ("mlframe.feature_selection.filters._conditional_permutation", "conditional_permutation_test"),
        ("mlframe.feature_selection.filters.estimators", None),
        ("mlframe.feature_selection.structure_discovery", None),
    ],
)
def test_no_site_still_hardcodes_the_add_one_form(module_name: str, call):
    """The four sites must route through the helper rather than inline the formula.

    Checked on the parse tree: an inline `(1 + count) / (1 + n)` is a shape, not a substring, and this must
    not be answerable by reformatting. `call` is unused here beyond naming the entry point each module
    exposes, kept so the table reads as a list of call sites rather than of files.
    """
    import ast
    import importlib

    module = importlib.import_module(module_name)
    tree = ast.parse(open(module.__file__, encoding="utf-8").read())

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        num, den = node.left, node.right
        if not (isinstance(num, ast.BinOp) and isinstance(num.op, ast.Add)):
            continue
        if not (isinstance(den, ast.BinOp) and isinstance(den.op, ast.Add)):
            continue
        rendered = ast.unparse(node)
        # An add-one p-value adds a literal 1 on both sides.
        if "1" in rendered and ("exceed" in rendered or "fail" in rendered or "perm" in rendered.lower()):
            offenders.append((node.lineno, rendered[:90]))
    assert not offenders, f"{module_name} still inlines an add-one p-value: {offenders}"


def test_the_canonical_name_is_not_shadowed():
    """structure_discovery defined its own `_perm_pvalue` with a different signature, so a grep misread it."""
    import inspect

    from mlframe.feature_selection import structure_discovery

    local = getattr(structure_discovery, "_perm_pvalue", None)
    # No local definition is a pass -- nothing shadows the canonical helper then. Written as one
    # unconditional assertion so the "it vanished entirely" case cannot skip the check instead of failing.
    params = None if local is None else list(inspect.signature(local).parameters)
    assert params is None or params[:2] == ["nfailed", "nchecked"], (
        f"structure_discovery._perm_pvalue takes {params}, shadowing the canonical helper with a different meaning"
    )
