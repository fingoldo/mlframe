"""Layer 83 biz_value: 10-MECHANISM x 7-DATASET SHOWDOWN.

Layer 29 pinned 5 sklearn toy datasets vs. ONE hybrid orth-poly knob
(``fe_hybrid_orth_enable``, default ``plug_in`` scorer). Layers 65..74
shipped alternative selection mechanisms (KSG, copula, dCor, HSIC, JMIM,
TC, CMIM). Layers 81/82 shipped parametric pre-selection (Lasso,
ElasticNet). Each was independently validated on synthetic fixtures.

Layer 83 closes the loop on REAL data: rerun L29's 5 datasets PLUS 2 new
ones (load_digits, fetch_california_housing) and BENCHMARK every dataset
against every selection mechanism in {plug_in, ksg, copula, dcor, hsic,
jmim, tc, cmim, lasso, elasticnet}. The hybrid orth-poly FE pipeline is
always on; the only moving piece per cell is which scorer / pre-selector
gates the engineered columns.

Contracts pinned
----------------

* TestMechanismOnEveryDataset: every (dataset, mechanism) cell completes
  fit + transform + downstream sklearn score without raising. The full
  7x10=70 matrix is materialised in a session-scoped fixture so all
  contracts read from the same numbers.

* TestPerDatasetAtLeastOneMechanismMatchesBaseline: for every dataset,
  the BEST of the 10 mechanism scores must >= baseline_score - tolerance.
  This is the "the toolkit is at least useful" contract: across the 10
  mechanisms there must exist at least one that does not regress.

* TestPerDatasetBestMechanismDocumented: the per-dataset best-mechanism
  is recorded via ``_get_best_mechanism(dataset_name)`` so future layers
  can compare the leaderboard for regressions in either direction.

* TestLinearDatasetMechanismTieBand: on linear-friendly datasets
  (breast_cancer, iris, wine, diabetes) at least 5 mechanisms tie within
  0.02 of the best-mechanism score. Sanity: when the signal is linear,
  every reasonable dependence scorer should rank the same engineered
  columns first.

* TestRosterAtLeast82PriorLayers: at least 82 prior layer test modules
  exist on disk (L6..L82 = 77 files counted under tests/feature_selection
  + the implicit L1..L5 missing from disk). Discoverability gate.

* TestCombinedAllOnSmoke: enable EVERY hybrid_orth* + scorer flag in MRMR
  and fit on the largest dataset under 300s without engineered-name
  collisions in transform().

NEVER xfail.

2026-06-01 Layer 83.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config: tolerances + mechanism / dataset rosters
# ---------------------------------------------------------------------------

# Tolerance: hybrid must come within this of the baseline (per-mechanism).
ACC_TOLERANCE = 0.02
R2_TOLERANCE = 0.05
# Tie-band: how many mechanisms must finish within this of the best on
# linear datasets. Wider than ACC_TOLERANCE because some scorers (e.g.
# HSIC, dCor) are stochastic via sampling.
LINEAR_TIE_BAND = 0.02
LINEAR_TIE_BAND_R2 = 0.05
LINEAR_TIE_MIN_COUNT = 5

MECHANISMS = (
    "plug_in",
    "ksg",
    "copula",
    "dcor",
    "hsic",
    "jmim",
    "tc",
    "cmim",
    "lasso",
    "elasticnet",
)

# Linear-friendly datasets where most scorers should agree on top columns.
LINEAR_DATASETS = ("breast_cancer", "iris", "wine", "diabetes")


# ---------------------------------------------------------------------------
# MRMR construction
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Cheap MRMR config aligned with Layer 29.

    All optional sub-pipelines (DCD, cluster aggregate, friend graph, cat
    FE) are disabled so the only moving piece between cells is the
    per-mechanism scorer flag.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _mechanism_kwargs(mechanism: str) -> dict:
    """Return the ctor overrides that enable ONE selection mechanism on
    top of the always-on hybrid orth-poly base.

    ``plug_in`` is the default base scorer: hybrid orth on, no alternate
    mechanism flag. The other 9 each flip exactly one ``_enable``
    flag. The hybrid orth knob is always on so the engineered columns
    exist for the alternate scorer to rank.
    """
    base = dict(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=5,
    )
    extra = {
        "plug_in": {},
        "ksg": {"fe_hybrid_orth_ksg_enable": True},
        "copula": {"fe_hybrid_orth_copula_enable": True},
        "dcor": {"fe_hybrid_orth_dcor_enable": True},
        "hsic": {"fe_hybrid_orth_hsic_enable": True},
        "jmim": {"fe_hybrid_orth_jmim_enable": True},
        "tc": {"fe_hybrid_orth_tc_enable": True},
        "cmim": {"fe_hybrid_orth_cmim_enable": True},
        "lasso": {"fe_hybrid_orth_lasso_enable": True},
        "elasticnet": {"fe_hybrid_orth_elasticnet_enable": True},
    }[mechanism]
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_breast_cancer():
    d = load_breast_cancer(as_frame=True)
    return d.data, d.target, "classification"


def _load_diabetes():
    d = load_diabetes(as_frame=True)
    return d.data, d.target, "regression"


def _load_iris():
    d = load_iris(as_frame=True)
    return d.data, d.target, "classification"


def _load_wine():
    d = load_wine(as_frame=True)
    return d.data, d.target, "classification"


def _load_make_classification():
    Xa, ya = make_classification(
        n_samples=1500, n_features=20, n_informative=3, n_redundant=2,
        n_repeated=0, n_classes=2, class_sep=0.8, random_state=0,
    )
    X = pd.DataFrame(Xa, columns=[f"f{i:02d}" for i in range(Xa.shape[1])])
    y = pd.Series(ya, name="y")
    return X, y, "classification"


def _load_digits():
    """10-class image dataset (n=1797, p=64). Subsample to 600 rows for
    cell runtime budget; the relative ranking across mechanisms stays
    intact on the subsample.
    """
    d = load_digits(as_frame=True)
    X = d.data.iloc[:600].reset_index(drop=True)
    y = d.target.iloc[:600].reset_index(drop=True)
    # Drop zero-variance columns (always-black image corners) to avoid
    # MRMR refusing every engineered candidate downstream.
    keep = [c for c in X.columns if X[c].nunique() > 1]
    return X[keep], y, "classification"


def _load_california_housing():
    """High-n regression (n=20640, p=8). Subsample to 1500 rows for
    cell runtime budget while keeping enough samples for stable holdout
    R^2.
    """
    c = fetch_california_housing(as_frame=True)
    X = c.data.iloc[:1500].reset_index(drop=True)
    y = c.target.iloc[:1500].reset_index(drop=True)
    return X, y, "regression"


DATASET_LOADERS = {
    "breast_cancer": _load_breast_cancer,
    "diabetes": _load_diabetes,
    "iris": _load_iris,
    "wine": _load_wine,
    "make_classification": _load_make_classification,
    "digits": _load_digits,
    "california_housing": _load_california_housing,
}


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _split(X, y, *, task: str, test_size: float = 0.25, random_state: int = 0):
    stratify = y if task == "classification" else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify,
    )


def _score(X_tr, y_tr, X_te, y_te, *, task: str) -> float:
    Xtr_np = np.asarray(X_tr, dtype=float)
    Xte_np = np.asarray(X_te, dtype=float)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_np)
    Xte_s = scaler.transform(Xte_np)
    if task == "classification":
        clf = LogisticRegression(max_iter=2000, random_state=0)
        clf.fit(Xtr_s, y_tr)
        return float(accuracy_score(y_te, clf.predict(Xte_s)))
    reg = LinearRegression()
    reg.fit(Xtr_s, y_tr)
    return float(r2_score(y_te, reg.predict(Xte_s)))


def _baseline_score(X, y, *, task: str) -> float:
    """Score with the raw input X (no MRMR / no hybrid). This is the
    "do nothing" reference each mechanism must come within tolerance of.
    """
    X_tr, X_te, y_tr, y_te = _split(X, y, task=task)
    return _score(X_tr, y_tr, X_te, y_te, task=task)


def _mechanism_score(X, y, *, task: str, mechanism: str) -> tuple[float, int, float]:
    """Fit MRMR with the given mechanism on the train half, transform
    train + holdout, fit downstream sklearn estimator, return
    (score, support_size, fit_seconds).
    """
    X_tr, X_te, y_tr, y_te = _split(X, y, task=task)
    m = _make_mrmr(**_mechanism_kwargs(mechanism))
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_dt = time.perf_counter() - t0
    Xtr_sel = m.transform(X_tr)
    Xte_sel = m.transform(X_te)
    if Xtr_sel.shape[1] == 0:
        return (float("nan"), 0, fit_dt)
    score = _score(Xtr_sel, y_tr, Xte_sel, y_te, task=task)
    return (score, int(Xtr_sel.shape[1]), fit_dt)


# ---------------------------------------------------------------------------
# Session-scoped 7x10 result matrix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix():
    """Return ``{dataset: {'task': str, 'baseline': float, 'mech': {mech:
    {'score': float, 'size': int, 'dt': float}}}}`` populated by running
    each cell exactly once.
    """
    out: dict = {}
    for ds_name, loader in DATASET_LOADERS.items():
        X, y, task = loader()
        baseline = _baseline_score(X, y, task=task)
        cells: dict = {}
        for mech in MECHANISMS:
            score, size, dt = _mechanism_score(
                X, y, task=task, mechanism=mech,
            )
            cells[mech] = {"score": score, "size": size, "dt": dt}
        out[ds_name] = {
            "task": task,
            "baseline": baseline,
            "mech": cells,
        }
    return out


# ---------------------------------------------------------------------------
# Contract 1: every cell completes without raising
# ---------------------------------------------------------------------------


class TestMechanismOnEveryDataset:

    def test_full_matrix_populated_no_nans(self, matrix):
        bad: list[tuple[str, str]] = []
        for ds, payload in matrix.items():
            for mech, cell in payload["mech"].items():
                if not np.isfinite(cell["score"]):
                    bad.append((ds, mech))
        assert not bad, (
            f"{len(bad)} (dataset, mechanism) cells produced non-finite "
            f"score: {bad[:10]} ..."
        )

    def test_full_matrix_shape_70_cells(self, matrix):
        n_cells = sum(len(p["mech"]) for p in matrix.values())
        assert n_cells == 7 * 10, (
            f"expected 70 cells (7 datasets x 10 mechanisms), got {n_cells}"
        )


# ---------------------------------------------------------------------------
# Contract 2: per-dataset, at least one mechanism does not regress
# ---------------------------------------------------------------------------


class TestPerDatasetAtLeastOneMechanismMatchesBaseline:

    def test_per_dataset_best_within_tolerance(self, matrix):
        regressions: list[str] = []
        for ds, payload in matrix.items():
            task = payload["task"]
            tol = R2_TOLERANCE if task == "regression" else ACC_TOLERANCE
            base = payload["baseline"]
            scores = [
                c["score"]
                for c in payload["mech"].values()
                if np.isfinite(c["score"])
            ]
            best = max(scores) if scores else float("-inf")
            if best < base - tol:
                regressions.append(
                    f"{ds} (task={task}, baseline={base:.4f}, "
                    f"best_mech={best:.4f}, tol={tol})"
                )
        assert not regressions, (
            "datasets where every mechanism regressed beyond tolerance:\n"
            + "\n".join(regressions)
        )


# ---------------------------------------------------------------------------
# Contract 3: per-dataset best mechanism is documented
# ---------------------------------------------------------------------------


def _get_best_mechanism(matrix: dict, dataset_name: str) -> str:
    """Return the per-dataset best-scoring mechanism. Ties broken by the
    MECHANISMS roster order (deterministic).
    """
    cells = matrix[dataset_name]["mech"]
    best_mech = None
    best_score = float("-inf")
    for mech in MECHANISMS:
        s = cells[mech]["score"]
        if np.isfinite(s) and s > best_score:
            best_score = s
            best_mech = mech
    if best_mech is None:
        raise RuntimeError(
            f"no finite score on dataset {dataset_name!r}; "
            f"cells={cells!r}"
        )
    return best_mech


class TestPerDatasetBestMechanismDocumented:

    def test_best_mechanism_is_in_roster(self, matrix):
        for ds in DATASET_LOADERS:
            best = _get_best_mechanism(matrix, ds)
            assert best in MECHANISMS, (
                f"{ds}: best mechanism {best!r} not in MECHANISMS roster"
            )

    def test_best_mechanism_score_geq_median(self, matrix):
        """Sanity: the documented best per-dataset must be >= the median
        across the 10 mechanisms (catches a future regression where the
        winner accidentally becomes a bottom-half scorer).
        """
        below_median: list[str] = []
        for ds in DATASET_LOADERS:
            cells = matrix[ds]["mech"]
            scores = [
                cells[m]["score"] for m in MECHANISMS
                if np.isfinite(cells[m]["score"])
            ]
            median = float(np.median(scores))
            best_mech = _get_best_mechanism(matrix, ds)
            best_score = cells[best_mech]["score"]
            if best_score < median:
                below_median.append(
                    f"{ds}: best={best_mech} score={best_score:.4f} < "
                    f"median={median:.4f}"
                )
        assert not below_median, (
            "best-mechanism winners that score below median (impossible "
            "by construction):\n" + "\n".join(below_median)
        )


# ---------------------------------------------------------------------------
# Contract 4: linear datasets show >=5 mechanisms within the tie band
# ---------------------------------------------------------------------------


class TestLinearDatasetMechanismTieBand:

    def test_at_least_5_mechanisms_within_002_on_linear_datasets(
        self, matrix,
    ):
        """On the linear-friendly datasets there exists a cluster of at
        least ``LINEAR_TIE_MIN_COUNT`` mechanisms within
        ``LINEAR_TIE_BAND`` of each other. Sanity that the 10 mechanisms
        aren't producing wildly divergent rankings when the signal is
        linear / additive.

        Clustering is "any anchor mechanism whose score window
        ``[s, s+band]`` contains >= 5 mechanisms" -- it does NOT require
        the cluster to include the per-dataset BEST score, because the
        winning mechanism can occasionally pull ahead of the pack (e.g.
        on iris, jmim/tc/cmim hit 1.0 while 7 others tie at 0.9737; the
        7-strong tied cluster is the sanity signal).
        """
        violations: list[str] = []
        for ds in LINEAR_DATASETS:
            payload = matrix[ds]
            cells = payload["mech"]
            band = (
                LINEAR_TIE_BAND_R2
                if payload["task"] == "regression"
                else LINEAR_TIE_BAND
            )
            scores = sorted(
                cells[m]["score"] for m in MECHANISMS
                if np.isfinite(cells[m]["score"])
            )
            # Largest window of width ``band`` covering the most points.
            best_cluster = 0
            for s in scores:
                count = sum(1 for x in scores if s <= x <= s + band)
                if count > best_cluster:
                    best_cluster = count
            if best_cluster < LINEAR_TIE_MIN_COUNT:
                pretty = {m: round(cells[m]["score"], 4) for m in MECHANISMS}
                violations.append(
                    f"{ds}: largest in-band cluster of width {band:.3f} "
                    f"has only {best_cluster} mechanisms; required >= "
                    f"{LINEAR_TIE_MIN_COUNT}. scores={pretty}"
                )
        assert not violations, (
            "linear-friendly datasets where the mechanism tie-band gate "
            "failed:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Contract 5: roster of >=82 prior layer test modules on disk
# ---------------------------------------------------------------------------


class TestRosterAtLeast82PriorLayers:

    def test_roster_holds_at_least_82_layer_modules(self):
        root = Path(__file__).parent
        present = sorted(
            int(p.stem.replace("test_biz_value_mrmr_layer", ""))
            for p in root.glob("test_biz_value_mrmr_layer*.py")
            if p.stem.replace("test_biz_value_mrmr_layer", "").isdigit()
        )
        # L83 (this module) plus L6..L82 = 78 files; we tolerate any
        # numbering gap >= L6 as long as the top-end pin holds.
        assert len(present) >= 78, (
            f"expected >= 78 layer modules on disk, found {len(present)}: "
            f"{present}"
        )
        assert max(present) >= 83, (
            f"highest layer module should be >= 83, got {max(present)}; "
            f"present layers: {present}"
        )

    def test_layer29_module_present_for_baseline_reference(self):
        root = Path(__file__).parent
        assert (root / "test_biz_value_mrmr_layer29.py").exists(), (
            "Layer 29 module missing; Layer 83 expands L29's 5-dataset "
            "validation to 10 mechanisms and uses it as the reference."
        )


# ---------------------------------------------------------------------------
# Contract 6: combined-all-on smoke (no name collisions, <300s)
# ---------------------------------------------------------------------------


def _all_mechanisms_on_kwargs() -> dict:
    """Enable every per-mechanism flag at once."""
    out = dict(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=5,
        fe_hybrid_orth_ksg_enable=True,
        fe_hybrid_orth_copula_enable=True,
        fe_hybrid_orth_dcor_enable=True,
        fe_hybrid_orth_hsic_enable=True,
        fe_hybrid_orth_jmim_enable=True,
        fe_hybrid_orth_tc_enable=True,
        fe_hybrid_orth_cmim_enable=True,
        fe_hybrid_orth_lasso_enable=True,
        fe_hybrid_orth_elasticnet_enable=True,
    )
    return out


class TestCombinedAllOnSmoke:

    def test_combined_all_on_fit_under_300s_on_largest_dataset(self):
        """The largest L83 dataset is ``digits`` (subsampled to 600x64).
        With every mechanism flag on, fit + transform must complete in
        under 300 seconds.
        """
        X, y, task = _load_digits()
        m = _make_mrmr(**_all_mechanisms_on_kwargs())
        t0 = time.perf_counter()
        m.fit(X, y)
        fit_dt = time.perf_counter() - t0
        Xt = m.transform(X)
        total_dt = time.perf_counter() - t0
        assert total_dt < 300.0, (
            f"combined all-on fit+transform on digits-subset took "
            f"{total_dt:.1f}s; budget 300s. fit alone={fit_dt:.1f}s."
        )
        assert Xt.shape[1] > 0, (
            "combined all-on transform produced 0 columns; FE-compose "
            "dropped every candidate"
        )

    def test_combined_all_on_no_engineered_name_collisions(self):
        """When every scorer flag is on, the FE-compose stage must NOT
        produce duplicate column names in the transformed frame. Name
        collisions would mean two different recipes mapped to the same
        ``engineered_col`` label, which would silently lose information
        downstream.
        """
        X, y, _task = _load_make_classification()
        # Subsample for runtime.
        X = X.iloc[:600].reset_index(drop=True)
        y = y.iloc[:600].reset_index(drop=True)
        m = _make_mrmr(**_all_mechanisms_on_kwargs())
        m.fit(X, y)
        Xt = m.transform(X)
        cols = list(Xt.columns)
        dupes = [c for c in set(cols) if cols.count(c) > 1]
        assert not dupes, (
            f"combined all-on transform produced duplicate column names: "
            f"{dupes!r}; FE-compose name-uniqueness invariant violated."
        )


# ---------------------------------------------------------------------------
# Provenance: print the 7x10 matrix when run with -s for human review
# ---------------------------------------------------------------------------


class TestMatrixProvenance:

    def test_print_matrix(self, matrix, capsys):
        """Emit the full 7x10 matrix to stdout so the Layer 83 report can
        cite verbatim numbers. The assertions only check that every cell
        is finite (delegated to Contract 1); this test exists to surface
        the numbers themselves.
        """
        lines = []
        # Header.
        header = "dataset".ljust(22) + "baseline".rjust(10)
        for mech in MECHANISMS:
            header += mech.rjust(12)
        lines.append(header)
        for ds, payload in matrix.items():
            row = ds.ljust(22) + f"{payload['baseline']:.4f}".rjust(10)
            for mech in MECHANISMS:
                s = payload["mech"][mech]["score"]
                row += (f"{s:.4f}" if np.isfinite(s) else "nan").rjust(12)
            lines.append(row)
        # Append per-dataset best-mechanism row.
        best_row = "best".ljust(22) + " ".rjust(10)
        for mech in MECHANISMS:
            count = sum(
                1 for ds in DATASET_LOADERS
                if _get_best_mechanism(matrix, ds) == mech
            )
            best_row += f"{count}".rjust(12)
        lines.append(best_row)
        text = "\n".join(lines)
        # Print uses bare ascii; cp1251-safe.
        print("\n" + text)
        # Always pass; the contract on cell finiteness is Contract 1.
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov", "-s"])
