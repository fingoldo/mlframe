"""biz_value / verdict-pin tests for the qual-23 FE pure-form retention lever.

LEVER (qual-23): the relevance floor ``min_resid_corr`` in ``_adds_nonlinear_value`` inside
``retain_usable_pure_forms`` (``filters/_fe_pure_form_retention.py``). qual-23 exposed it (and ``min_resid_frac``)
as a tunable kwarg and benched lowering 0.08 -> 0.05; verdict was REJECT (dead knob at tractable scale: the trap
pre-check returns ``[]`` before the floor runs whenever the MI greedy already holds the pure pair form).

These tests pin BOTH the wiring (the kwarg actually controls the gate) AND the verdict (a lower floor is a no-op when
a pure pair recipe already survives). They run on a controlled stub mrmr (no full MRMR fit) so each is sub-second.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms


def _single_pair_data(n=2500, seed=0):
    """y depends on (a,b) via the non-separable joint a**2/b; c,d are pure noise."""
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    c = rng.random(n)
    d = rng.random(n)
    y = 0.2 * a**2 / b + 0.01 * rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d}), np.asarray(y, dtype=np.float64)


class _Recipe:
    """Minimal recipe stub carrying only the ``src_names`` the trap pre-check + covered-pair logic read."""

    def __init__(self, name, src_names):
        self.name = name
        self.src_names = tuple(src_names)


def _stub(df, recipes=()):
    """Helper that stub."""
    class _Stub:
        """Groups tests covering Stub."""
        feature_names_in_ = list(df.columns)
        _engineered_recipes_ = list(recipes)
        _engineered_features_ = [getattr(r, "name", "") for r in recipes]
        random_seed = 0

    return _Stub()


def test_biz_val_pure_form_retention_min_resid_corr_kwarg_threads():
    """The ``min_resid_corr`` kwarg must actually control the relevance gate. On a single-pair joint target with NO
    pre-existing pure pair recipe (so the trap pre-check admits the pass), a punitive floor (corr=0.99) rejects the
    (a,b) joint form that a permissive floor (corr=0.0) admits. If the kwarg were not threaded into ``_adds_nonlinear_value``
    the two calls would return the SAME set -- a dead-wire regression. This pins the wiring qual-23 added."""
    df, y = _single_pair_data(seed=0)

    permissive = retain_usable_pure_forms(_stub(df), df, y, seed=0, min_resid_corr=0.0)
    punitive = retain_usable_pure_forms(_stub(df), df, y, seed=0, min_resid_corr=0.99)

    def _has_ab(extra):
        """Has ab."""
        return any(frozenset(getattr(r, "src_names", ()) or ()) == frozenset({"a", "b"}) for r, _ in extra)

    assert _has_ab(permissive), "permissive corr floor (0.0) must admit the genuine a**2/b joint form"
    assert not _has_ab(punitive), "punitive corr floor (0.99) must reject every form -> kwarg is wired into the gate"


def test_biz_val_pure_form_retention_lower_corr_is_noop_when_greedy_keeps_pure_pair():
    """qual-23 verdict pin. When a PURE (<=2-operand) pair recipe for (a,b) already survives -- the common case the
    MI greedy produces -- the trap pre-check (``has_cross_mix OR not has_pure_pair``) returns ``[]`` BEFORE the
    ``min_resid_corr`` floor runs, so lowering the floor from 0.08 to 0.05 (or to 0.0) changes nothing: the pure form
    is already covered, there is nothing trapped to recover. This is exactly why the qual-23 default flip was REJECTED
    as a dead knob; a future "just lower the floor" must not silently slip through."""
    df, y = _single_pair_data(seed=1)
    pure_pair_present = [_Recipe("div(sqr(a),identity(b))", ("a", "b"))]

    old = retain_usable_pure_forms(_stub(df, pure_pair_present), df, y, seed=1, min_resid_corr=0.08)
    new = retain_usable_pure_forms(_stub(df, pure_pair_present), df, y, seed=1, min_resid_corr=0.05)
    very_low = retain_usable_pure_forms(_stub(df, pure_pair_present), df, y, seed=1, min_resid_corr=0.0)

    assert old == [], "with the pure (a,b) pair already present the trap pre-check must short-circuit to []"
    assert new == old, "lowering corr 0.08 -> 0.05 must be a no-op when the greedy already holds the pure pair"
    assert very_low == old, "even corr=0.0 is a no-op -- the floor never runs (trap pre-check gates the pass)"
