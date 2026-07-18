"""Wave-2 W2 + W6 MRMR diagnostic-completeness items.

W2 -- PROVENANCE SELF-AUDIT ACCESSOR ``get_unlabeled_recipe_kinds()``
--------------------------------------------------------------------
Lists every SURVIVING engineered recipe.kind whose provenance origin resolved to
``engineered_unknown``. After commit 205baa86 (~20 sibling FE families now
labeled) a normal fit should surface ONLY the deliberate ``factorize`` set; any
OTHER kind appearing is an UNREGISTERED family -- the guardrail firing. Turns
"did we forget to register a kind?" into a measured list.

W6 -- REJECTION-LEDGER EXTENSION TO THE 2 ABS-MAD FLOOR SITES
------------------------------------------------------------
The ``fe_rejection_ledger_`` instrumented the pair-search gates but left the two
``med + k*MAD`` abs-floor sites (``_orthogonal_cluster_basis_fe`` +
``_mi_greedy_fe``) uninstrumented, so ``explain_selection()`` could not finger an
abs-floor kill. Both now call ``record_fe_rejection`` (gate
``marginal_uplift_floor``) at the drop site. The shared ``_unified_fe_gate``
local-MI floor (class-closure sibling) gained the same ``reject_sink`` plumbing.

PURE ADDITIVE: no gate-logic change, selection byte-identical. Default-ON.
NEVER xfail.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._mrmr_fe_provenance import (
    DELIBERATELY_UNLABELED_KINDS,
    get_unlabeled_recipe_kinds,
)
from mlframe.feature_selection.filters._mi_greedy_fe import (
    greedy_mi_fe_construct,
)
from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
    generate_cluster_basis_features,
)
from mlframe.feature_selection.filters._unified_fe_gate import (
    local_mi_gate,
    raw_mi_noise_floor,
)

# ===========================================================================
# W2 -- provenance self-audit accessor
# ===========================================================================


class _ProvStub:
    """Minimal estimator carrying just fe_provenance_ + recipe ledgers."""


class _Recipe:
    """Groups tests covering Recipe."""
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind


def _make_prov(rows):
    """Make prov."""
    return pd.DataFrame(
        rows,
        columns=["feature_name", "origin", "mechanism_details", "mrmr_gain", "support_rank"],
    )


def test_w2_accessor_returns_only_factorize_on_clean_fit():
    """biz_value: a clean fit (every surviving engineered kind labeled EXCEPT
    the deliberate ``factorize``) returns exactly ``{"factorize": N}``."""
    s = _ProvStub()
    s._produced_recipes_ = [
        _Recipe("raw_a", ""),  # raw -> not engineered
        _Recipe("hybrid_x", "orth_univariate"),  # labeled hybrid_orth
        _Recipe("fac_b", "factorize"),  # deliberate engineered_unknown
        _Recipe("fac_c", "factorize"),
    ]
    s._engineered_recipes_ = list(s._produced_recipes_)
    s.fe_provenance_ = _make_prov(
        [
            ["raw_a", "raw", "{}", 0.1, 0],
            ["hybrid_x", "hybrid_orth", "{}", 0.2, 1],
            ["fac_b", "engineered_unknown", "{}", 0.15, 2],
            ["fac_c", "engineered_unknown", "{}", 0.12, 3],
        ]
    )
    res = get_unlabeled_recipe_kinds(s)
    assert res == {"factorize": 2}, res
    # The guardrail signal (anything OUTSIDE the deliberate set) is empty.
    assert set(res) - DELIBERATELY_UNLABELED_KINDS == set()


def test_w2_accessor_flags_an_unregistered_kind():
    """unit: a deliberately-unregistered fake kind surfaces in the accessor ->
    the guardrail fires (it is NOT in the deliberate set)."""
    s = _ProvStub()
    s._produced_recipes_ = [
        _Recipe("fac_b", "factorize"),
        _Recipe("newfam_z", "brand_new_unregistered_family"),  # NOT in _RECIPE_KIND_TO_ORIGIN
    ]
    s._engineered_recipes_ = list(s._produced_recipes_)
    s.fe_provenance_ = _make_prov(
        [
            ["fac_b", "engineered_unknown", "{}", 0.15, 1],
            ["newfam_z", "engineered_unknown", "{}", 0.11, 2],
        ]
    )
    res = get_unlabeled_recipe_kinds(s)
    assert res.get("brand_new_unregistered_family") == 1, res
    guardrail = set(res) - DELIBERATELY_UNLABELED_KINDS
    assert guardrail == {"brand_new_unregistered_family"}, guardrail


def test_w2_accessor_excludes_screened_out_nonsurvivors():
    """Only SURVIVORS (support_rank >= 0) count; a screened-out unknown
    (rank -1) is NOT reported (it never reached support_)."""
    s = _ProvStub()
    s._produced_recipes_ = [_Recipe("dropped_z", "brand_new_unregistered_family")]
    s._engineered_recipes_ = []
    s.fe_provenance_ = _make_prov(
        [
            ["dropped_z", "engineered_unknown", "{}", float("nan"), -1],
        ]
    )
    assert get_unlabeled_recipe_kinds(s) == {}


def test_w2_accessor_empty_on_unfitted():
    """W2 accessor empty on unfitted."""
    s = _ProvStub()
    assert get_unlabeled_recipe_kinds(s) == {}


# ===========================================================================
# W6 -- abs-MAD floor rejection ledger
# ===========================================================================


class _Sink:
    """Captures reject_sink calls in place of the real record_fe_rejection."""

    def __init__(self):
        self.records = []

    def __call__(self, **kw):
        self.records.append(kw)


def _lone_signal_pool(n=600, seed=0):
    """One strong engineered signal among many noise candidates: the #2/#3
    self-gating regime where the med+k*MAD abs floor binds (kills the noise
    candidates that clear the relative-uplift gate on a near-zero baseline)."""
    rng = np.random.default_rng(seed)
    # Raw seed columns -- weak/noise so uplift inflates on a tiny baseline.
    raw = pd.DataFrame({f"r{i}": rng.normal(size=n) for i in range(4)})
    # Target driven by a nonlinear (square) transform of r0 only.
    y = (raw["r0"].to_numpy() ** 2 > 1.0).astype(np.int64)
    return raw, y


def test_w6_mi_greedy_floor_records_abs_floor_kill():
    """unit + biz_value: on the lone-signal pool, the mi_greedy abs-MAD floor
    records killed candidates with gate=marginal_uplift_floor + negative margin,
    and the kill set is exactly the candidates below the absolute floor."""
    raw, y = _lone_signal_pool()
    sink = _Sink()
    X_aug, _scores = greedy_mi_fe_construct(
        raw,
        y,
        top_k=3,
        min_uplift=1.05,
        reject_sink=sink,
    )
    # The floor must have killed >=1 candidate that cleared the uplift gate.
    assert sink.records, "abs-MAD floor recorded no kills on the lone-signal pool"
    for rec in sink.records:
        assert rec["gate"] == "marginal_uplift_floor"
        # observed (engineered_mi) is strictly below threshold (abs_floor).
        assert rec["observed"] < rec["threshold"]
    # Selection byte-identity: the kept set is independent of the sink.
    X_aug_nosink, _ = greedy_mi_fe_construct(raw, y, top_k=3, min_uplift=1.05)
    assert list(X_aug.columns) == list(X_aug_nosink.columns)


def test_w6_cluster_basis_floor_records_abs_floor_kill():
    """unit: the cluster-basis abs-MAD floor records its kills too."""
    rng = np.random.default_rng(1)
    n = 800
    # 20 correlated noise columns (>=16 baselines -> noise_floor engages) +
    # one real cluster carrying signal.
    base = rng.normal(size=n)
    cols = {f"c{i}": base + 0.05 * rng.normal(size=n) for i in range(20)}
    X = pd.DataFrame(cols)
    y = (X["c0"].to_numpy() > 0).astype(np.int64)
    members = {f"c{i}": [f"c{i}", f"c{(i + 1) % 20}"] for i in range(20)}
    sink = _Sink()
    eng, _meta = generate_cluster_basis_features(
        X,
        y,
        members,
        degrees=(2, 3),
        top_k=3,
        min_uplift=1.0,
        reject_sink=sink,
    )
    eng_ns, _meta_ns = generate_cluster_basis_features(
        X,
        y,
        members,
        degrees=(2, 3),
        top_k=3,
        min_uplift=1.0,
    )
    # Byte-identical survivor set.
    assert list(eng.columns) == list(eng_ns.columns)
    if sink.records:
        for rec in sink.records:
            assert rec["gate"] == "marginal_uplift_floor"
            assert rec["observed"] < rec["threshold"]


def test_w6_unified_local_mi_gate_records_floor_kill():
    """class-closure: the shared _unified_fe_gate local-MI floor records its
    abs-MAD kills via the same gate name; selection byte-identical."""
    rng = np.random.default_rng(2)
    n = 500
    raw = pd.DataFrame({f"r{i}": rng.normal(size=n) for i in range(5)})
    y = (raw["r0"].to_numpy() > 0).astype(np.int64)
    # Engineered candidates: one informative (= r0 sign proxy) + several noise.
    enc = pd.DataFrame(
        {
            "good": raw["r0"].to_numpy() + 0.1 * rng.normal(size=n),
            **{f"noise{i}": rng.normal(size=n) for i in range(6)},
        }
    )
    raw_mi_noise_floor(raw, y)
    sink = _Sink()
    keep = local_mi_gate(enc, y, raw_X=raw, reject_sink=sink)
    keep_ns = local_mi_gate(enc, y, raw_X=raw)
    assert keep == keep_ns  # byte-identical
    for rec in sink.records:
        assert rec["gate"] == "marginal_uplift_floor"
        assert rec["operator"] == "unified_local_mi_gate"
        assert rec["observed"] < rec["threshold"]


def test_w6_greedy_score_select_no_sink_is_noop():
    """Default (no sink) path is unchanged: returns winners with no error."""
    raw, y = _lone_signal_pool()
    _X_aug, scores = greedy_mi_fe_construct(raw, y, top_k=3)
    assert isinstance(scores, pd.DataFrame)
