"""A raw operand a lossy composite cannot replace must survive the raw-redundancy drop in FULL mode.

The drop sweep treats a selected engineered composite as subsuming the raw columns it was built from. That is
right when the composite really does carry the operand's information, and wrong when the composite is LOSSY
with respect to it. An additive mixture like ``add(add(neg(sig0),neg(sig1)),add(neg(sig2),neg(sig4)))``
preserves the sum and destroys the individual contributions -- which is precisely what a downstream model
needs when the signals enter ``y`` with different coefficients.

`_fe_raw_redundancy_drop` already had the right instrument for this: a permutation-floored partial
rank-correlation leg (`raw_retains_linear_signal_given_children`) that keeps a raw retaining private linear
signal given its children. It ran only in simple mode. In full mode the operands were dropped, selection
collapsed to two features, and downstream AUC fell well below the raw-signal baseline.

These tests pin the OUTCOME (recovered accuracy, and a still-compact selection) rather than the mechanism, so
a future re-tuning of the leg is free to change how it decides as long as the result holds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Measured on this fixture: five raw signals score 0.9648, the fixed selection 0.9649, and the pre-fix
# selection 0.8969. The floor sits between the broken and the fixed value with room for CV noise, so it
# fails loudly on a regression without tracking the exact number.
_MIN_DOWNSTREAM_AUC = 0.94
# Pre-fix the selection collapsed to 2 features; disabling the drop sweep entirely gives 25. A correct fix
# sits near the true signal count, so this bounds BOTH failure directions: collapse and no-op.
_MAX_SELECTED = 12


def _fixture():
    """Five signals with DIFFERENT coefficients plus fifteen pure-noise columns."""
    rng = np.random.default_rng(200)
    n = 2500
    sig = {f"sig{k}": rng.standard_normal(n) for k in range(5)}
    # Distinct coefficients are the whole point: a sum-style composite cannot reconstruct them.
    y_lin = sum(sig[f"sig{k}"] * (0.8 - 0.1 * k) for k in range(5)) + 0.5 * rng.standard_normal(n)
    y = pd.Series((y_lin > 0).astype(np.int64))
    noise = {f"noise{k}": rng.standard_normal(n) for k in range(15)}
    return pd.DataFrame({**sig, **noise}), y


def _downstream_auc(frame: pd.DataFrame, y: pd.Series) -> float:
    """5-fold ROC-AUC of a logistic model on the selected columns."""
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=2000), frame, y, cv=5, scoring="roc_auc")))


@pytest.fixture(scope="module")
def _fitted():
    """One default-mode fit, shared by the assertions below (the fit dominates runtime)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MRMR(verbose=0, fe_keep_linearly_usable_raw_operands=True).fit(X, y)
    return X, y, model


def test_downstream_accuracy_matches_the_raw_signal_baseline(_fitted):
    """The selection must not cost accuracy against simply using the five raw signals."""
    X, y, model = _fitted
    baseline = _downstream_auc(X[[f"sig{k}" for k in range(5)]], y)
    got = _downstream_auc(pd.DataFrame(model.transform(X)), y)
    assert got >= _MIN_DOWNSTREAM_AUC, (
        f"downstream AUC {got:.4f} against a five-raw-signal baseline of {baseline:.4f}: the raw operands were "
        "dropped in favour of a composite that cannot reconstruct their individual contributions"
    )


def test_the_selection_stays_compact(_fitted):
    """Keeping usable operands must not degenerate into keeping everything."""
    _X, _y, model = _fitted
    selected = list(model.get_feature_names_out())
    assert 0 < len(selected) <= _MAX_SELECTED, f"expected a compact selection, got {len(selected)}: {selected}"


def test_the_default_keeps_fewer_raw_operands_than_the_opt_in(_fitted):
    """The shipped default is unchanged: in full mode the raw operands are still dropped.

    Discriminates on the SELECTION, not on a downstream-AUC threshold. An AUC floor looked like it separated
    the two behaviours locally (0.8969 default vs 0.9649 opt-in) and did not travel: on CI the default scored
    0.9459, above the floor, so the assertion inverted and failed for a reason that had nothing to do with the
    contract. How many raw signal columns survive the drop sweep IS the contract, and it does not depend on a
    logistic model's fold split.

    The keep leg is opt-in rather than default because CI measured its cost across the suite: subsumed raws
    kept alongside the composite that captures them (I4b), the canonical BUG1 fixtures re-adding a fully
    subsumed `a`, and -- worst -- the group-aware leak guard resurrecting a demoted `x_leak`. Those are all
    correct drops the leg undoes.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y, opt_in_model = _fitted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_model = MRMR(verbose=0).fit(X, y)

    def _raw_signals(model) -> set:
        """The bare `sigN` columns that survived, ignoring engineered composites built from them."""
        return {name for name in model.get_feature_names_out() if name in {f"sig{k}" for k in range(5)}}

    default_raws = _raw_signals(default_model)
    opt_in_raws = _raw_signals(opt_in_model)
    assert len(default_raws) < len(opt_in_raws), (
        f"the default kept {sorted(default_raws)} and the opt-in kept {sorted(opt_in_raws)} -- the keep leg is "
        "supposed to be what preserves raw operands, so either it became the default or this fixture no longer "
        "separates the two behaviours and needs rebuilding"
    )
