"""The frozen cpx36 baseline must differ from production in the ONE thing it was frozen to pin.

`_cpx36_baseline/fisher_weighted_residual_old.py` exists to pin a single change -- batched versus
per-perturbation predict -- and `test_batched_predict_bit_identical` asserts `np.array_equal` over ALL
output columns. Production later gained an unrelated fix: `band_y_mean` is seeded with the global
`y_t.mean()` instead of zeros, because "0.0 misleadingly reads as a genuinely low-residual band rather than
'no data'". The frozen copy kept `np.zeros`.

Any fold whose weighted residuals tie enough to collapse a quantile boundary leaves a band empty, and
`fishres_band_y_mean` then differs for every query row assigned to it -- so the identity test would have
failed, blaming the batching, for a statistic it never guarded. Today's fixture happens to produce no empty
band, which is why the drift was invisible rather than harmless.

The band loop is replayed directly here so the divergence is shown on the arithmetic rather than hoped for
through a model fit.
"""

from __future__ import annotations

import numpy as np


def _band_means(weighted_train: np.ndarray, y_t: np.ndarray, n_bands: int, seed_with_global_mean: bool) -> np.ndarray:
    """The band loop from both modules, parameterised on the initialiser that differs between them."""
    quantiles = np.quantile(weighted_train, np.linspace(0.0, 1.0, n_bands + 1))
    if seed_with_global_mean:
        band_y_mean = np.full(n_bands, float(y_t.mean()), dtype=np.float32)
    else:
        band_y_mean = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        if b == 0:
            mask = weighted_train <= quantiles[b + 1]
        elif b == n_bands - 1:
            mask = weighted_train > quantiles[b]
        else:
            mask = (weighted_train > quantiles[b]) & (weighted_train <= quantiles[b + 1])
        if mask.sum() > 0:
            band_y_mean[b] = float(y_t[mask].mean())
    return band_y_mean


def _tie_heavy():
    """Weighted residuals with enough ties to collapse three of five quantile boundaries."""
    weighted = np.array([0.0] * 8 + [5.0], dtype=np.float64)
    y = np.array([1.0] * 8 + [9.0], dtype=np.float64)
    return weighted, y, 5


def test_an_empty_band_makes_the_two_initialisers_disagree():
    """The premise: on a tie-heavy fold the choice of seed is visible in the output, not internal."""
    weighted, y, n_bands = _tie_heavy()
    zeros_seeded = _band_means(weighted, y, n_bands, seed_with_global_mean=False)
    mean_seeded = _band_means(weighted, y, n_bands, seed_with_global_mean=True)
    assert not np.array_equal(zeros_seeded, mean_seeded), "the fixture no longer produces an empty band"
    assert np.allclose(zeros_seeded, [1.0, 0.0, 0.0, 0.0, 9.0])
    assert np.allclose(mean_seeded, [1.0, 1.8889, 1.8889, 1.8889, 9.0], atol=1e-4)


def _band_seed_expression(module) -> str:
    """The right-hand side of the module's `band_y_mean = ...` initialiser, rendered from its parse tree.

    Compared on the AST rather than by searching the source text for a substring: the two files are a frozen
    copy and its original, so the question really is a source-level one, but reformatting or a reworded
    comment must not answer it.

    The file is read from disk rather than through ``inspect.getsource``, which the behavioural-test gate
    forbids outright -- and reading it is the more direct expression of a question that is about two FILES.
    """
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "band_y_mean" for t in node.targets):
            if isinstance(node.value, ast.Call):  # the initialiser, not the per-band overwrite
                return ast.unparse(node.value)
    raise AssertionError(f"{module.__name__} has no band_y_mean initialiser; this test has lost its subject")


def test_the_frozen_baseline_seeds_bands_the_way_production_does():
    """Both modules must agree on this statistic, so the identity test can only fail on the batching."""
    from mlframe.feature_engineering._benchmarks._cpx36_baseline import fisher_weighted_residual_old as old
    from mlframe.feature_engineering.transformer import fisher_weighted_residual as new

    production = _band_seed_expression(new)
    frozen = _band_seed_expression(old)
    assert "mean()" in production, f"production no longer seeds bands from the global mean ({production})"
    assert frozen == production, (
        f"the frozen cpx36 baseline seeds empty bands as {frozen!r} while production uses {production!r}, so "
        "the batching identity test would fail on fishres_band_y_mean for any tie-heavy fold -- a statistic "
        "that baseline was never frozen to pin"
    )

def test_a_band_that_has_rows_is_unaffected_by_the_seed():
    """The fix must only touch bands with no data; every populated band is overwritten either way."""
    rng = np.random.default_rng(0)
    weighted = rng.normal(size=200)
    y = rng.normal(size=200)
    assert np.allclose(
        _band_means(weighted, y, 5, seed_with_global_mean=False),
        _band_means(weighted, y, 5, seed_with_global_mean=True),
    )
