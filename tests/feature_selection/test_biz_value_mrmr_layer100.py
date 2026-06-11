"""Layer 100 biz_value: UNIFY scorer-selection under the Param-Oracle.

Layer 68 (``_orthogonal_scorer_auto_fe``) is the EXPENSIVE per-column
bootstrap-LCB bake-off over all scorers; Layer 76
(``_orthogonal_meta_scorer_fe``) is the CHEAP fingerprint -> rule cascade.
Layer 98 built the Param-Oracle (``mlframe.utils._param_oracle``), a
fingerprint -> recommend cache that LEARNS from history. Layer 100 wires
the scorer choice through ``OracleScorerSelector`` so the two prior
mechanisms unify:

* cold-start = L76 ``predict_best_scorer`` rules (reused, not reimplemented);
* the L68 bake-off becomes the "benchmark mode" that POPULATES the oracle;
* once a fingerprint bucket has confident history the learned-best scorer
  wins over the cold-start prior.

Contracts pinned
----------------
* ``TestColdStartMatchesL76``: empty oracle -> ``recommend_scorer`` equals
  L76 ``predict_best_scorer`` on linear (plug_in) / quadratic (hsic) /
  redundant (cmim) fixtures.
* ``TestBenchmarkPopulatesOracle``: ``benchmark_all_scorers`` records every
  scorer's quality for the dataset fingerprint.
* ``TestLearnedBeatsColdStart``: after observing CMIM wins on a redundant
  fingerprint where cold-start would pick something else, ``recommend_scorer``
  returns the learned winner.
* ``TestStatOnlyPersistence``: the oracle store holds no raw arrays.
* ``TestMrmrAutoOracleEndToEnd``: MRMR(default_scorer="auto_oracle") fits a
  redundant fixture, selects a sensible scorer, AUC competitive with the
  explicit-best scorer.
* ``TestL68L76StillWork``: the existing auto + meta paths are unbroken.
* ``TestFingerprintReuse``: the selector reuses Param-Oracle's
  ``default_fingerprint`` (not a duplicate fingerprinter).
* ``TestPickleAndClone``: selector pickles + MRMR(auto_oracle) clones.

NEVER xfail.
"""
from __future__ import annotations

import orjson
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)
FIXED_TS = "2026-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Isolate the on-disk oracle store under tmp_path for every test so the
# real ``~/.pyutilz/param_oracle`` store is never touched and tests do not
# cross-contaminate via shared learned history.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_oracle_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    yield


def _import_selector():
    from mlframe.feature_selection.filters._oracle_scorer_select import (
        ORACLE_FN_NAME,
        ORACLE_SCORER_NAMES,
        OracleScorerSelector,
    )
    return ORACLE_FN_NAME, ORACLE_SCORER_NAMES, OracleScorerSelector


# ---------------------------------------------------------------------------
# Fixtures mirror the L76 roster so the cold-start path can be cross-checked
# against the L76 cascade on identical data.
# ---------------------------------------------------------------------------


def _build_linear_monotone(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    y = ((1.2 * x1 + 0.8 * x2 + 0.5 * x3 + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_quadratic(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor_redundant(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x_dup_a": x_dup_a, "x_dup_b": x_dup_b, "x_dup_c": x_dup_c,
        "x2": x2,
        "noise_0": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Contract 1: cold-start == L76 rules
# ---------------------------------------------------------------------------


class TestColdStartMatchesL76:
    """With an empty oracle, ``recommend_scorer`` must equal the L76
    ``predict_best_scorer`` rule cascade on the SAME (X, y) -- proving the
    cold-start path reuses L76 rather than inventing its own rules."""

    @pytest.mark.parametrize("builder,expected", [
        (_build_linear_monotone, "plug_in"),
        (_build_quadratic, "hsic"),
        (_build_xor_redundant, "cmim"),
    ])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cold_start_equals_l76(self, builder, expected, seed):
        from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import (
            fingerprint_signal, predict_best_scorer,
        )
        _, _, OracleScorerSelector = _import_selector()
        X, y = builder(seed)
        l76 = predict_best_scorer(fingerprint_signal(X, y.to_numpy()))
        sel = OracleScorerSelector(store_path="cold.parquet")
        rec = sel.recommend_scorer(X, y)
        assert rec == l76, (
            f"seed={seed}: cold-start {rec!r} != L76 {l76!r}"
        )
        assert rec == expected, (
            f"seed={seed}: expected {expected!r}, got {rec!r}"
        )


# ---------------------------------------------------------------------------
# Contract 2: benchmark_all_scorers populates the oracle
# ---------------------------------------------------------------------------


class TestBenchmarkPopulatesOracle:

    def test_benchmark_records_all_scorers(self):
        _, ORACLE_SCORER_NAMES, OracleScorerSelector = _import_selector()
        X, y = _build_xor_redundant(seed=7)
        sel = OracleScorerSelector(store_path="bench.parquet")
        assert sel.oracle.store.read_rows() == []  # cold store
        qualities = sel.benchmark_all_scorers(
            X, y, degrees=(2,), basis="hermite", n_boot=3, ts=FIXED_TS,
        )
        # Every scorer in the pool got a recorded quality.
        assert set(qualities.keys()) == set(ORACLE_SCORER_NAMES)
        rows = sel.oracle.store.read_rows()
        recorded = {orjson.loads(r["param_combo_json"])["scorer"] for r in rows}
        assert recorded == set(ORACLE_SCORER_NAMES), (
            f"benchmark recorded {recorded}, expected {set(ORACLE_SCORER_NAMES)}"
        )
        # The recorded objective carries the quality metric.
        for r in rows:
            obj = orjson.loads(r["objective_json"])
            assert "quality" in obj


# ---------------------------------------------------------------------------
# Contract 3: learned beats cold-start
# ---------------------------------------------------------------------------


class TestLearnedBeatsColdStart:
    """After enough observations that CMIM wins on a fingerprint where the
    cold-start cascade would pick something else, ``recommend_scorer`` must
    return the learned winner -- the posterior overrides the prior."""

    def test_learned_overrides_cold_start(self):
        from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import (
            fingerprint_signal, predict_best_scorer,
        )
        _, _, OracleScorerSelector = _import_selector()
        # Linear fixture: cold-start picks plug_in.
        X, y = _build_linear_monotone(seed=13)
        cold = predict_best_scorer(fingerprint_signal(X, y.to_numpy()))
        assert cold == "plug_in"

        sel = OracleScorerSelector(store_path="learn.parquet", min_observations=3)
        assert sel.recommend_scorer(X, y) == "plug_in"  # cold-start prior

        # Observe that CMIM wins big on this fingerprint (and plug_in loses)
        # enough times to clear the confidence gate.
        for _ in range(4):
            sel.observe_scorer(X, "cmim", 0.99, y=y, ts=FIXED_TS)
            sel.observe_scorer(X, "plug_in", 0.10, y=y, ts=FIXED_TS)

        learned = sel.recommend_scorer(X, y)
        assert learned == "cmim", (
            f"learned-best should override cold-start plug_in -> cmim, "
            f"got {learned!r}"
        )

    def test_below_confidence_gate_stays_cold_start(self):
        _, _, OracleScorerSelector = _import_selector()
        X, y = _build_linear_monotone(seed=13)
        sel = OracleScorerSelector(store_path="gate.parquet", min_observations=5)
        # Only 2 observations -- below the gate -> still cold-start.
        for _ in range(2):
            sel.observe_scorer(X, "cmim", 0.99, y=y, ts=FIXED_TS)
        assert sel.recommend_scorer(X, y) == "plug_in", (
            "below min_observations the learned scorer must NOT win"
        )


# ---------------------------------------------------------------------------
# Contract 4: stat-only persistence (no raw arrays on disk)
# ---------------------------------------------------------------------------


class TestStatOnlyPersistence:

    def test_store_has_no_raw_arrays(self):
        _, _, OracleScorerSelector = _import_selector()
        X, y = _build_xor_redundant(seed=42)
        sel = OracleScorerSelector(store_path="stat.parquet")
        sel.benchmark_all_scorers(
            X, y, degrees=(2,), basis="hermite", n_boot=3, ts=FIXED_TS,
        )
        rows = sel.oracle.store.read_rows()
        assert rows, "expected recorded rows"
        for r in rows:
            for col, val in r.items():
                assert not isinstance(val, (list, tuple, dict, np.ndarray)), (
                    f"non-scalar persisted in {col}: {type(val)}"
                )
            fp_bucket = orjson.loads(r["fp_bucket_json"])
            for v in fp_bucket.values():
                assert isinstance(v, (int, float, str)), (
                    f"non-scalar in fp_bucket: {v!r}"
                )
        store_bytes = os.path.getsize(sel.oracle.store._path)
        assert store_bytes < X.to_numpy().nbytes, (
            "store larger than raw array -> likely leaking data"
        )


# ---------------------------------------------------------------------------
# Contract 5: MRMR auto_oracle end-to-end
# ---------------------------------------------------------------------------


class TestMrmrAutoOracleEndToEnd:
    """MRMR(fe_hybrid_orth_enable=True, default_scorer="auto_oracle") on a
    redundant fixture selects a sensible scorer and appends engineered
    columns. AUC of the downstream LogReg is competitive with the
    explicit-best scorer (cmim on the redundant fixture)."""

    def test_auto_oracle_appends_and_validates(self):
        X, y = _build_xor_redundant(seed=7)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_default_scorer="auto_oracle",
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, (
            "auto_oracle should append engineered columns on the redundant "
            "fixture"
        )

    def test_auto_oracle_auc_competitive_with_explicit_best(self):
        aucs_oracle, aucs_explicit = [], []
        for s in SEEDS:
            X, y = _build_xor_redundant(s)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )

            def _fit_auc(scorer):
                m = _make_mrmr(
                    fe_hybrid_orth_enable=True,
                    fe_hybrid_orth_default_scorer=scorer,
                    fe_hybrid_orth_degrees=(2,),
                    fe_hybrid_orth_basis="hermite",
                    fe_hybrid_orth_top_k=3,
                    random_seed=s,
                ).fit(X_tr, y_tr)
                added = list(getattr(m, "hybrid_orth_features_", []) or [])
                from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
                    generate_univariate_basis_features,
                )
                if added:
                    eng_tr = generate_univariate_basis_features(
                        X_tr, degrees=(2,), basis="hermite",
                    )
                    eng_te = generate_univariate_basis_features(
                        X_te, degrees=(2,), basis="hermite",
                    )
                    have = [c for c in added if c in eng_tr.columns]
                    X_tr_a = pd.concat([X_tr, eng_tr[have]], axis=1) if have else X_tr
                    X_te_a = pd.concat([X_te, eng_te[have]], axis=1) if have else X_te
                else:
                    X_tr_a, X_te_a = X_tr, X_te
                lr = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
                    X_tr_a, y_tr,
                )
                return roc_auc_score(
                    y_te, lr.predict_proba(X_te_a)[:, 1],
                )

            aucs_oracle.append(_fit_auc("auto_oracle"))
            aucs_explicit.append(_fit_auc("cmim"))
        oracle_mean = float(np.mean(aucs_oracle))
        explicit_mean = float(np.mean(aucs_explicit))
        assert oracle_mean >= explicit_mean - 0.02, (
            f"auto_oracle AUC ({oracle_mean:.4f}) lags explicit-best cmim "
            f"({explicit_mean:.4f}) by more than 0.02.\n"
            f"oracle={aucs_oracle}\nexplicit={aucs_explicit}"
        )


# ---------------------------------------------------------------------------
# Contract 6: L68 (auto) + L76 (meta) paths still work
# ---------------------------------------------------------------------------


class TestL68L76StillWork:

    def test_l68_auto_path_unbroken(self):
        X, y = _build_quadratic(seed=7)
        m = _make_mrmr(
            fe_hybrid_orth_auto_scorer_enable=True,
            fe_hybrid_orth_auto_scorer_n_boot=3,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        # L68 path runs without error; engineered columns may or may not be
        # appended depending on the gate, but the attribute must exist.
        assert hasattr(m, "hybrid_orth_features_")

    def test_l76_meta_path_unbroken(self):
        X, y = _build_quadratic(seed=7)
        m = _make_mrmr(
            fe_hybrid_orth_meta_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        chosen = getattr(m, "hybrid_orth_meta_chosen_scorer_", None)
        assert chosen == "hsic", (
            f"L76 meta path should dispatch quadratic to hsic, got {chosen!r}"
        )

    def test_l68_auto_scorer_value_unbroken(self):
        # The "auto" default-scorer value (L68 bake-off) must still route.
        X, y = _build_quadratic(seed=7)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_default_scorer="auto",
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        assert hasattr(m, "hybrid_orth_features_")


# ---------------------------------------------------------------------------
# Contract 7: fingerprint reuse (not a duplicate fingerprinter)
# ---------------------------------------------------------------------------


class TestFingerprintReuse:

    def test_selector_reuses_param_oracle_default_fingerprint(self):
        from mlframe.utils._param_oracle import default_fingerprint
        _, _, OracleScorerSelector = _import_selector()
        X, y = _build_linear_monotone(seed=1)
        sel = OracleScorerSelector(store_path="fp.parquet")
        fp_sel = sel.fingerprint(X, y)
        fp_oracle = default_fingerprint((X, y), {})
        assert fp_sel == fp_oracle, (
            "selector must reuse Param-Oracle's default_fingerprint, not a "
            "duplicate fingerprinter"
        )
        # The bucketed key the selector observes under matches the oracle's.
        from mlframe.utils._param_oracle import bucketize_fingerprint
        assert bucketize_fingerprint(fp_sel) == bucketize_fingerprint(fp_oracle)


# ---------------------------------------------------------------------------
# Contract 8: pickle / clone
# ---------------------------------------------------------------------------


class TestPickleAndClone:

    def test_selector_pickle_roundtrip(self):
        _, _, OracleScorerSelector = _import_selector()
        X, y = _build_linear_monotone(seed=1)
        sel = OracleScorerSelector(store_path="pk.parquet", min_observations=4)
        # Record so there is learned state to reconnect to after unpickle.
        for _ in range(4):
            sel.observe_scorer(X, "cmim", 0.99, y=y, ts=FIXED_TS)
        blob = pickle.dumps(sel)
        sel2 = pickle.loads(blob)
        assert sel2.store_path == sel.store_path
        assert sel2.min_observations == sel.min_observations
        assert sel2.scorer_names == sel.scorer_names
        # The unpickled selector reconnects to the SAME on-disk store and
        # therefore returns the learned scorer.
        assert sel2.recommend_scorer(X, y) == "cmim"

    def test_mrmr_auto_oracle_clone_preserves_param(self):
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_default_scorer="auto_oracle",
        )
        m2 = clone(m)
        assert m2.fe_hybrid_orth_default_scorer == "auto_oracle"

    def test_mrmr_auto_oracle_pickle_roundtrip(self):
        X, y = _build_xor_redundant(seed=42)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_default_scorer="auto_oracle",
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=2,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)
        assert list(m2.feature_names_in_) == list(m.feature_names_in_)
        before = list(getattr(m, "hybrid_orth_features_", []) or [])
        after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert before == after


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
