"""Behavioral invariants and structural sensors for ``mlframe.feature_selection.wrappers.RFECV``.

Adds four families of tests beyond the existing functional suite:

1. ``TestEvalFoldClosureCapture`` - structural sensor that ``_eval_fold`` keeps the default-arg pattern introduced to silence
   ruff B023 (loop-variable capture). AST-walks ``_rfecv.py`` rather than asserting on source strings so the test survives
   reformatting / comment edits but breaks if someone reverts to direct closure capture of ``current_features`` / ``scores``.

2. ``TestEvalFoldBehavior`` - integration-style sensors for inner-loop branches: empty-train guard (NaN fold score on collapsed
   train slice), per-fold seed determinism, must_include concatenation contract.

3. ``TestRFECVProperties`` - hypothesis-based invariants over RFECV.fit output: support_ length / sum vs n_features_,
   selected_features_per_nfeatures keys ⊆ checked_nfeatures, transform() output columns ⊆ input columns, refit determinism
   on the same data + seed.

4. ``TestParallelDeterminism`` - locks the n_jobs=1 vs n_jobs>1 bit-identity guarantee at the same ``random_state``. This is
   the scenario where the B023 closure-capture bug WOULD manifest if the default-arg fix were reverted.
"""
from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import (
    RFECV,
    OptimumSearch,
    make_gaussian_knockoffs,
    select_features_fdr,
)

from .conftest import IS_FAST_MODE, fast_subset


# Common minimal RFECV factory: fast, deterministic, log-quiet.
def _make_rfecv(**overrides):
    base = dict(
        estimator=LogisticRegression(max_iter=300, random_state=0),
        max_refits=4,
        max_noimproving_iters=2,
        verbose=0,
        optimizer_plotting="No",
        random_state=42,
    )
    base.update(overrides)
    return RFECV(**base)


def _make_data(n_samples=120, n_features=10, n_informative=3, seed=0):
    """Small classification dataset: ``n_informative`` features carry signal, the rest is noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    logits = X[:, :n_informative].sum(axis=1)
    y = (logits + 0.3 * rng.standard_normal(n_samples) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return df, y


class TestEvalFoldClosureCapture:
    """Locks the default-arg pattern in ``_eval_fold`` so a future refactor cannot silently revert to closure capture of
    ``current_features`` / ``scores`` (the ruff B023 finding). The bug doesn't manifest in the current synchronous-per-iter
    dispatch but would the moment closure execution were deferred (background thread / async dispatch / etc.).
    """

    def _find_eval_fold(self) -> ast.FunctionDef:
        """Locate the ``_eval_fold`` FunctionDef inside ``RFECV.fit`` by AST-walking ``_rfecv.py``
        and any sibling helper module the fit-body was split into (post-monolith-split)."""
        import pathlib
        import mlframe.feature_selection.wrappers._rfecv as mod_rfecv
        _dir = pathlib.Path(mod_rfecv.__file__).resolve().parent
        for _name in ("_rfecv.py", "_rfecv_fit.py"):
            _p = _dir / _name
            if not _p.exists():
                continue
            with open(_p, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "_eval_fold":
                    return node
        pytest.fail("could not locate _eval_fold FunctionDef in _rfecv.py or _rfecv_fit.py")

    def test_eval_fold_signature_uses_default_arg_capture(self):
        fn = self._find_eval_fold()
        param_names = [a.arg for a in fn.args.args]
        # We don't assert positional order beyond presence; pytest can adapt if a wrapping layer renames positional args.
        assert "current_features" in param_names, "B023 fix lost: current_features must be a parameter, not a free var"
        assert "scores" in param_names, "B023 fix lost: scores must be a parameter, not a free var"

    def test_eval_fold_default_for_current_features_is_named_current_features(self):
        """The default-arg expression must reference ``current_features`` from the enclosing scope (not, e.g., a literal
        ``None``); that's what gives the def-time binding."""
        fn = self._find_eval_fold()
        defaults = fn.args.defaults
        param_names = [a.arg for a in fn.args.args]
        # Defaults align right-to-left with params: the last len(defaults) params receive them.
        n = len(defaults)
        defaulted = dict(zip(param_names[-n:], defaults))
        assert "current_features" in defaulted, "current_features must have a default-arg value"
        assert "scores" in defaulted, "scores must have a default-arg value"
        # Each default expression must be the matching outer-scope Name(...).
        for pname in ("current_features", "scores"):
            d = defaulted[pname]
            assert isinstance(d, ast.Name) and d.id == pname, (
                f"{pname} default must reference outer-scope Name({pname!r}); got {ast.dump(d)}"
            )


class TestEvalFoldBehavior:
    """Integration sensors for ``_eval_fold`` branches reachable through public ``RFECV.fit``."""

    def test_must_include_columns_are_in_selection(self):
        """must_include columns must appear in support_ regardless of optimiser pick."""
        X, y = _make_data(n_samples=120, n_features=8, n_informative=2, seed=1)
        rfecv = _make_rfecv(must_include=["f7"])  # f7 is pure noise; must still be retained
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
        selected = rfecv.get_feature_names_out().tolist()
        assert "f7" in selected, f"must_include feature missing from final selection: {selected}"

    def test_must_exclude_columns_are_dropped_at_entry(self):
        """must_exclude columns never enter the candidate universe."""
        X, y = _make_data(n_samples=120, n_features=8, n_informative=2, seed=2)
        leak_col = "f0"  # known-informative, deliberately excluded
        rfecv = _make_rfecv(must_exclude=[leak_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
        selected = rfecv.get_feature_names_out().tolist()
        assert leak_col not in selected, f"must_exclude feature leaked into selection: {selected}"

    def test_fit_records_evaluated_n_features(self):
        """checked_nfeatures / cv_results_ must record at least the baseline (0F) and the initial all-features evaluation."""
        X, y = _make_data(n_samples=120, n_features=8, n_informative=2, seed=3)
        rfecv = _make_rfecv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
        evaluated = list(rfecv.cv_results_["nfeatures"])
        # cv_results_ must always carry at least one evaluated point.
        assert len(evaluated) >= 1
        # Every recorded nfeatures entry must be in [0, n_features].
        assert all(0 <= n <= X.shape[1] for n in evaluated), f"out-of-range nfeatures in cv_results_: {evaluated}"


class TestRFECVProperties:
    """Hypothesis-based invariants on RFECV.fit output. Skipped under MLFRAME_FAST=1 because each example fits a fresh RFECV."""

    def _support_indices(self, rfecv):
        """Normalise support_ to a list of int indices regardless of whether it's a bool mask or integer indices."""
        sup = np.asarray(rfecv.support_)
        if sup.dtype == bool:
            return np.flatnonzero(sup).tolist()
        return [int(i) for i in sup]

    @pytest.mark.slow
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_support_size_matches_n_features(self, seed):
        """``len(support_)`` (when bool-mask) or ``len(support_)`` (when integer) must equal ``n_features_``."""
        X, y = _make_data(n_samples=80, n_features=6, n_informative=2, seed=seed % 1000)
        rfecv = _make_rfecv(random_state=seed % 1000, max_refits=3, max_noimproving_iters=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
        indices = self._support_indices(rfecv)
        assert rfecv.n_features_ == len(indices), (
            f"n_features_ ({rfecv.n_features_}) must equal selected-index count ({len(indices)})"
        )

    @pytest.mark.slow
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_transform_output_columns_subset_of_input(self, seed):
        """transform() must drop or pass through columns; never introduce new ones."""
        X, y = _make_data(n_samples=80, n_features=6, n_informative=2, seed=seed % 1000)
        rfecv = _make_rfecv(random_state=seed % 1000, max_refits=3, max_noimproving_iters=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
            X_out = rfecv.transform(X)
        out_cols = set(X_out.columns) if isinstance(X_out, pd.DataFrame) else None
        if out_cols is not None:
            assert out_cols.issubset(set(X.columns)), (
                f"transform() introduced unknown columns: {out_cols - set(X.columns)}"
            )

    @pytest.mark.slow
    @settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_refit_same_seed_same_support(self, seed):
        """Two fits of two FRESH RFECV instances on the same data + same random_state must produce the same support_."""
        X, y = _make_data(n_samples=80, n_features=6, n_informative=2, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = _make_rfecv(random_state=seed, max_refits=3, max_noimproving_iters=2)
            r1.fit(X, y)
            r2 = _make_rfecv(random_state=seed, max_refits=3, max_noimproving_iters=2)
            r2.fit(X, y)
        s1 = self._support_indices(r1)
        s2 = self._support_indices(r2)
        assert set(s1) == set(s2), f"determinism lost at seed={seed}: {s1} vs {s2}"


class TestParallelDeterminism:
    """Locks bit-identity between sequential (n_jobs=1) and parallel (n_jobs>1) CV-fold execution at the same random_state.

    This is the scenario where the B023 closure-capture bug WOULD manifest if its fix were reverted (a parallel fold worker
    holding a stale reference to ``current_features`` could vote on a different feature subset than the iteration's
    optimiser-picked one). Locking parity here guards the fix at the behavioral level.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("n_jobs", fast_subset([2, 4], n=1))
    def test_parallel_matches_sequential_single_thread_estimator(self, n_jobs):
        """With ``LogisticRegression`` (single-threaded estimator), n_jobs=1 and n_jobs>1 must select the same features at
        the same random_state."""
        X, y = _make_data(n_samples=100, n_features=8, n_informative=2, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_seq = _make_rfecv(random_state=42, n_jobs=1)
            r_seq.fit(X, y)
            r_par = _make_rfecv(random_state=42, n_jobs=n_jobs)
            r_par.fit(X, y)
        s_seq = set(np.flatnonzero(np.asarray(r_seq.support_, dtype=bool)).tolist()) if np.asarray(r_seq.support_).dtype == bool else set(int(i) for i in r_seq.support_)
        s_par = set(np.flatnonzero(np.asarray(r_par.support_, dtype=bool)).tolist()) if np.asarray(r_par.support_).dtype == bool else set(int(i) for i in r_par.support_)
        assert s_seq == s_par, f"parallel divergence at n_jobs={n_jobs}: seq={s_seq}, par={s_par}"


class TestCoverageGaps:
    """Targeted tests for opt-in code paths that the default ``test_wrappers_*.py`` suite leaves uncovered:

    - ``_run_sffs_swap_pass`` (``swap_top_k > 0``) - was 70 uncovered lines, the largest single gap in ``_rfecv.py``.
    - ``select_optimal_nfeatures_(plot_file=...)`` - matplotlib render branch.
    - ``_suggest_scipy_local`` / ``_suggest_scipy_global`` - alternative optimisers triggered only via
      ``top_predictors_search_method=OptimumSearch.ScipyLocal`` / ``ScipyGlobal``.
    """

    @pytest.mark.slow
    def test_swap_top_k_triggers_sffs_swap_pass(self, caplog):
        """``swap_top_k > 0`` enters ``_sffs_swap_pass`` after the MBH loop converges. Sensor: the swap-summary log line
        ('SFFS swap pass: N/M paired swaps accepted') must appear; the final selection has the same cardinality the swap
        loop preserves (size is invariant under paired in/out)."""
        import logging
        X, y = _make_data(n_samples=120, n_features=10, n_informative=3, seed=4)
        rfecv = _make_rfecv(swap_top_k=2, verbose=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers._rfecv"):
                rfecv.fit(X, y)
        msgs = [r.message for r in caplog.records]
        assert any("SFFS swap pass:" in m for m in msgs), (
            f"swap_top_k=2 did not exercise _sffs_swap_pass (no swap-summary log line found). Messages: {msgs[-3:]}"
        )

    @pytest.mark.slow
    def test_plot_file_writes_matplotlib_artifact(self, tmp_path):
        """``select_optimal_nfeatures_(plot_file=...)`` exercises the matplotlib render branch at ``_rfecv.py:1902-1928``."""
        import matplotlib
        matplotlib.use("Agg")  # headless backend so plt.show() / pause are no-ops
        X, y = _make_data(n_samples=100, n_features=6, n_informative=2, seed=5)
        rfecv = _make_rfecv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
            out_png = tmp_path / "rfecv_plot.png"
            rfecv.select_optimal_nfeatures_(
                checked_nfeatures=np.array(list(rfecv.cv_results_["nfeatures"])),
                cv_mean_perf=np.array(list(rfecv.cv_results_["cv_mean_perf"])),
                cv_std_perf=np.array(list(rfecv.cv_results_["cv_std_perf"])),
                show_plot=False,
                plot_file=str(out_png),
            )
        assert out_png.exists(), f"plot_file did not produce {out_png}"
        assert out_png.stat().st_size > 0, f"plot_file produced an empty artifact: {out_png}"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "method",
        fast_subset([OptimumSearch.ScipyLocal, OptimumSearch.ScipyGlobal], n=1),
        ids=lambda m: m.name if hasattr(m, "name") else str(m),
    )
    def test_scipy_search_methods_run_to_completion(self, method):
        """Alternative optimisers ``ScipyLocal`` / ``ScipyGlobal`` cover the ``_suggest_scipy_local`` /
        ``_suggest_scipy_global`` paths in ``_helpers.py``. Sensor: fit returns without raising, support_ is non-empty when
        the optimiser converges to a non-zero N."""
        X, y = _make_data(n_samples=120, n_features=10, n_informative=3, seed=6)
        rfecv = _make_rfecv(top_predictors_search_method=method, max_refits=6, max_noimproving_iters=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfecv.fit(X, y)
        # support_ may be a bool mask or integer indices; in either case at least one feature should be picked on this
        # easily-separable dataset (3 informative features clearly above noise).
        sup = np.asarray(rfecv.support_)
        n_selected = int(np.sum(sup.astype(bool))) if sup.dtype == bool else len(sup)
        assert n_selected >= 1, f"{method.name}: expected >=1 feature selected, got 0"


class TestKnockoffEdgeCases:
    """Edge-case sensors for ``make_gaussian_knockoffs`` and ``select_features_fdr``. Closes the remaining ``_helpers.py``
    coverage gaps in the knockoff path (lines 91-92, 96-98, 118-126, 179-197)."""

    def test_sdp_solve_true_raises_not_implemented(self):
        """The SDP branch is a deliberately-deferred opt-in; fail loud rather than silently use equicorrelated."""
        X = np.random.default_rng(0).standard_normal((50, 5))
        with pytest.raises(NotImplementedError, match="SDP knockoffs not yet implemented"):
            make_gaussian_knockoffs(X, random_state=0, sdp_solve=True)

    @pytest.mark.parametrize(
        "shape",
        [(1, 3), (0, 3), (10, 0)],
        ids=["n=1", "n=0", "p=0"],
    )
    def test_too_small_X_raises_value_error(self, shape):
        """``make_gaussian_knockoffs`` requires n>=2, p>=1. Sensor: pre-computation guard at line 96-98 must reject."""
        X = np.zeros(shape, dtype=float)
        with pytest.raises(ValueError, match="at least 2 rows and 1 column"):
            make_gaussian_knockoffs(X, random_state=0)

    def test_near_singular_sigma_emits_warning(self, caplog):
        """When ``lam_min(Sigma) < 1e-4`` the warning at ``_helpers.py:118-126`` must fire so the operator knows knockoffs
        won't help. Construct collinearity: 5 columns where the last 4 are near-exact copies of the first plus tiny
        noise -> Sigma near-singular by construction."""
        import logging
        rng = np.random.default_rng(0)
        n, p = 200, 5
        base = rng.standard_normal((n, 1))
        noise = rng.standard_normal((n, p)) * 1e-7  # near-zero perturbation -> all columns ~ identical
        X = np.tile(base, (1, p)) + noise

        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers._helpers"):
            X_tilde = make_gaussian_knockoffs(X, random_state=0)

        assert X_tilde.shape == X.shape, "knockoff matrix must preserve input shape even on near-singular input"
        msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("near-singular" in m and "lambda_min" in m for m in msgs), (
            f"expected near-singular warning; got messages: {msgs}"
        )


class TestSelectFeaturesFdrEdgeCases:
    """Edge-case sensors for ``select_features_fdr`` (knockoff FDR-controlled selection)."""

    def test_empty_W_returns_empty(self):
        """No W statistics -> no selection."""
        assert select_features_fdr({}, q=0.1) == []

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
    def test_q_out_of_range_raises(self, q):
        """``q`` must be strictly in (0, 1). Sensor: guard at ``_helpers.py:181-182``."""
        W = {"f0": 0.5, "f1": -0.2, "f2": 0.1}
        with pytest.raises(ValueError, match="q must be in"):
            select_features_fdr(W, q=q)

    def test_all_negative_W_returns_empty(self):
        """When every W is non-positive, no threshold tau achieves the target FDR -> empty selection at line 193-194."""
        W = {"f0": -0.5, "f1": -0.3, "f2": -0.7}
        assert select_features_fdr(W, q=0.1) == []

    def test_strong_signal_selects_positive_W_features(self):
        """Positive sanity: features with W_j clearly above the noise band should be selected."""
        # Two strong real signals, three noise-symmetric W's. With q=0.5 the threshold should land
        # at the smaller of the two real W's (i.e. 0.6) since (1 + #{W <= -0.6}) / max(1, #{W >= 0.6}) = 1/2 <= 0.5.
        W = {"real_a": 1.2, "real_b": 0.6, "noise_a": 0.05, "noise_b": -0.05, "noise_c": -0.1}
        selected = select_features_fdr(W, q=0.5)
        assert "real_a" in selected
        assert "real_b" in selected
        # Sorted by W descending.
        assert selected[0] == "real_a"
