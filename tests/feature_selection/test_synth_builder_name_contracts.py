"""Name-vs-behaviour contracts for the shared biz_val synthetic builders.

Regression cover for three defects found in
``tests.feature_selection._synth.biz_val_synth``:

1. ``_build_xor_redundant`` promised XOR and delivered a plain quadratic --
   its body was byte-identical to ``_build_redundant_multi``
   (``y = sign(x1^2 + 0.6*x2^2 - median)``, zero synergy). Nine call sites in
   two MRMR biz_val files, including one paired with the synergy-aware ``cmim``
   scorer, therefore claimed to exercise synergy while feeding a quadratic. It
   is now ``_build_redundant_quadratic``. ``TestXorNamedBuildersReallyBuildXor``
   makes the class of defect non-recurring: ANY builder in the module whose name
   says "xor" must produce near-zero MARGINAL mutual information per operand and
   a clearly non-zero JOINT -- the defining property of an XOR-like target.

2. ``signal_recovery_count`` built its regex from an UNESCAPED caller-supplied
   prefix, so a prefix carrying regex metacharacters was interpreted instead of
   matched literally.

3. The same function swallowed every exception from ``get_feature_names_out()``
   under a bare ``except`` and silently fell back to the raw-index overlap --
   the metric its own docstring calls UNDER-counting -- so a broken selector was
   indistinguishable from a merely compact one.

Mutual information is measured with the production estimators
(``mlframe.feature_selection.filters.info_theory``), never a local re-roll.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.feature_selection._synth import biz_val_synth as _synth
from tests.feature_selection._synth.biz_val_synth import (
    _quantile_bin_local,
    signal_recovery_count,
)


def _xor_named_builders() -> list:
    """Collect every callable defined in the synth module whose name mentions XOR."""
    return [(nm, obj) for nm, obj in vars(_synth).items() if "xor" in nm.lower() and callable(obj) and getattr(obj, "__module__", "") == _synth.__name__]


def _call_builder(fn):
    """Invoke a synth builder with a generous ``n`` and return ``(X_2d, y, operand_indices)``."""
    kwargs = {}
    params = inspect.signature(fn).parameters
    if "n" in params:
        kwargs["n"] = 4000
    out = fn(**kwargs)
    assert len(out) == 3, f"{fn.__name__} must return (X, y, signal_indices) to be checkable for the XOR property; got {len(out)} items"
    X, y, sig = out
    X_arr = np.asarray(getattr(X, "to_numpy", lambda: X)(), dtype=np.float64)
    y_arr = np.asarray(getattr(y, "to_numpy", lambda: y)()).astype(np.int64)
    return X_arr, y_arr, [int(i) for i in sig]


def _mi_of(cols_bin: list, y_bin: np.ndarray, nbins: int) -> float:
    """Plug-in ``I(cols; y)`` via the production ``info_theory.mi`` on quantile-binned columns."""
    from mlframe.feature_selection.filters.info_theory import mi

    factors = np.column_stack([*cols_bin, y_bin]).astype(np.int32)
    y_idx = factors.shape[1] - 1
    factors_nbins = np.array([nbins] * len(cols_bin) + [int(y_bin.max()) + 1], dtype=np.int64)
    return float(
        mi(
            factors,
            np.arange(len(cols_bin), dtype=np.int64),
            np.array([y_idx], dtype=np.int64),
            factors_nbins,
        )
    )


def _cmi_of(target_col: int, others: list, cols_bin: list, y_bin: np.ndarray, nbins: int) -> float:
    """``I(x_target; y | x_others)`` via the production ``info_theory.conditional_mi``."""
    from mlframe.feature_selection.filters.info_theory import conditional_mi

    factors = np.column_stack([*cols_bin, y_bin]).astype(np.int32)
    y_idx = factors.shape[1] - 1
    factors_nbins = np.array([nbins] * len(cols_bin) + [int(y_bin.max()) + 1], dtype=np.int64)
    return float(
        conditional_mi(
            factors_data=factors,
            x=np.array([target_col], dtype=np.int64),
            y=np.array([y_idx], dtype=np.int64),
            z=np.array(sorted(others), dtype=np.int64),
            var_is_nominal=None,
            factors_nbins=factors_nbins,
            entropy_cache=None,
            can_use_x_cache=False,
            can_use_y_cache=False,
        )
    )


class TestXorNamedBuildersReallyBuildXor:
    """A builder named "xor" must produce a target whose information about the
    operands lives ONLY in their joint: near-zero marginal MI per operand,
    clearly non-zero joint MI, and a much larger conditional MI once the
    co-operands are conditioned on. This is exactly the property the old
    ``_build_xor_redundant`` lacked while advertising it in its name."""

    def test_at_least_one_xor_named_builder_exists(self):
        """The guard is not vacuous: the module still exposes an XOR-named builder to check."""
        names = [nm for nm, _ in _xor_named_builders()]
        assert names, "no XOR-named builder found in the synth module -- this guard would silently pass forever"

    @pytest.mark.parametrize("name", [nm for nm, _ in _xor_named_builders()])
    def test_marginal_mi_near_zero_joint_mi_large(self, name):
        """Each operand alone carries ~no MI about y, while the operands jointly carry a lot."""
        fn = dict(_xor_named_builders())[name]
        X, y, sig = _call_builder(fn)
        nbins = 4
        cols_bin = [_quantile_bin_local(X[:, i], nbins=nbins) for i in sig]
        marginals = [_mi_of([cols_bin[k]], y, nbins) for k in range(len(sig))]
        joint = _mi_of(cols_bin, y, nbins)
        assert max(marginals) < 0.02, f"{name}: operand marginal MIs {marginals} are not XOR-like (a real XOR hides ALL signal from the marginals); joint={joint:.4f}"
        assert joint > 0.15, f"{name}: joint MI over operands {sig} is only {joint:.4f} -- the name promises the signal lives in the joint; marginals={marginals}"

    @pytest.mark.parametrize("name", [nm for nm, _ in _xor_named_builders()])
    def test_conditional_mi_reveals_each_operand(self, name):
        """Conditioning on the co-operands turns each near-invisible operand into a strongly informative one."""
        fn = dict(_xor_named_builders())[name]
        X, y, sig = _call_builder(fn)
        nbins = 4
        cols_bin = [_quantile_bin_local(X[:, i], nbins=nbins) for i in sig]
        for k in range(len(sig)):
            others = [j for j in range(len(sig)) if j != k]
            marginal = _mi_of([cols_bin[k]], y, nbins)
            cmi = _cmi_of(k, others, cols_bin, y, nbins)
            assert cmi > marginal + 0.1, f"{name}: operand #{sig[k]} has I(x;y)={marginal:.4f} and I(x;y|rest)={cmi:.4f} -- synergy absent, so the name overstates the data"


class TestSignalRecoveryCountEscapesPrefix:
    """A caller-supplied prefix is DATA, not a pattern: metacharacters in it must
    match literally. Before the fix, ``prefix="x."`` compiled to "x followed by
    any character" and credited columns it never referenced."""

    def test_metacharacter_prefix_matches_literally(self):
        """prefix="x." credits only names literally containing "x.<digits>", never "x<digit><digits>".

        Unescaped, ``"x." + r"(\\d+)"`` compiles to "x, any char, digits", so
        ``"x79"`` wrongly yielded index 9 and ``"x93"`` index 3 -- both counted
        as recovered signal columns that the selector never referenced.
        """

        class _Sel:
            """Selector stub returning names that only an UNESCAPED "x." prefix would match."""

            def get_feature_names_out(self):
                """Return "x79"/"x93", which contain no literal "x." at all."""
                return ["x79", "x93"]

        assert signal_recovery_count(_Sel(), [9, 3], prefix="x.") == 0, "an unescaped prefix regex let 'x.' match 'x79'/'x93'; metacharacters in a caller string must be literal"

    def test_literal_metacharacter_prefix_is_still_found(self):
        """The escaped prefix still matches names that DO contain it literally."""

        class _Sel:
            """Selector stub whose names literally carry the "x." prefix."""

            def get_feature_names_out(self):
                """Return names containing the literal prefix "x." before the index."""
                return ["x.7", "add(x.9,noise)"]

        assert signal_recovery_count(_Sel(), [7, 9], prefix="x.") == 2

    def test_uncompilable_prefix_does_not_raise(self):
        """A prefix that is an INVALID regex ("x(") is matched literally instead of failing to compile."""

        class _Sel:
            """Selector stub whose names carry the literal, un-compilable prefix "x(" ."""

            def get_feature_names_out(self):
                """Return a name containing the literal "x(" prefix."""
                return ["x(3)"]

        assert signal_recovery_count(_Sel(), [3], prefix="x(") == 1


class TestSignalRecoveryCountDoesNotSwallowErrors:
    """A raising ``get_feature_names_out`` must surface, not degrade silently to
    the raw-index overlap the docstring itself calls UNDER-counting."""

    def test_raising_get_feature_names_out_propagates(self):
        """The selector's exception reaches the caller instead of being masked by a bare except."""

        class _Broken:
            """Selector stub whose get_feature_names_out raises, as an unfitted estimator would."""

            support_ = np.array([0, 1, 2])

            def get_feature_names_out(self):
                """Raise the way an unfitted sklearn selector does."""
                raise RuntimeError("selector not fitted")

        with pytest.raises(RuntimeError, match="not fitted"):
            signal_recovery_count(_Broken(), [0, 1, 2])

    def test_absent_method_still_falls_back(self):
        """The DOCUMENTED fallback (no get_feature_names_out at all) is preserved."""

        class _NoNames:
            """Selector stub exposing only raw support indices."""

            support_ = np.array([0, 1, 5])

        assert signal_recovery_count(_NoNames(), [0, 1, 2]) == 2


class TestRenamedBuilderIsNotXor:
    """The renamed ``_build_redundant_quadratic`` says what it is: a quadratic
    target over a near-duplicate candidate pool. Pinned so nobody re-labels it
    "xor" without also changing the data."""

    def test_redundant_quadratic_has_visible_marginals_and_duplicates(self):
        """x1's marginal MI with y is clearly non-zero (so NOT XOR-like), and the x_dup_* columns are near-copies of x1."""
        X, y = _synth._build_redundant_quadratic(seed=1, n=4000)
        cols_bin = [_quantile_bin_local(X["x1"].to_numpy(), nbins=4)]
        marginal = _mi_of(cols_bin, y.to_numpy().astype(np.int64), 4)
        assert marginal > 0.05, f"x1's marginal MI is {marginal:.4f}; a quadratic target must be visible marginally (an XOR one would not be)"
        corr = float(np.corrcoef(X["x1"], X["x_dup_a"])[0, 1])
        assert corr > 0.95, f"the fixture's redundancy (what the importing tests actually assert on) collapsed: corr(x1, x_dup_a)={corr:.4f}"

    def test_matches_build_redundant_multi_exactly(self):
        """The delegation is output-identical to ``_build_redundant_multi``, which it used to duplicate byte for byte."""
        Xa, ya = _synth._build_redundant_quadratic(seed=7, n=500)
        Xb, yb = _synth._build_redundant_multi(seed=7, n=500)
        assert list(Xa.columns) == list(Xb.columns)
        assert np.array_equal(Xa.to_numpy(), Xb.to_numpy())
        assert np.array_equal(ya.to_numpy(), yb.to_numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
