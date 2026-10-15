"""Behavioural tests for orthogonal, hybrid and polynomial feature-engineering recipes."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from tests.feature_selection.regression._mrmr_regression_helpers import _ORTH_SCORER_ZOO_MODULES


def test_regression_apply_unary_sign_preserves_nan():
    """Pre-fix: _apply_unary(nan, code=8, ...) returned -1.0. Post-fix: it returns nan, matching np.sign(nan)
    and the cupy twin _gpu_apply_unary's cp.sign(nan)."""
    from mlframe.feature_selection.filters._usability_njit_pool import _apply_unary

    result = _apply_unary(float("nan"), 8, 0.0)
    assert np.isnan(result), f"sign(nan) must stay nan (matches np.sign), got {result}"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-1 (P0): the cupy_kernel polynom-pair optimizer dispatch never checked gpu_globally_disabled(),
# so a plain MRMR() on a cupy-capable host launched real CUDA kernels for every hermite/orth pair-FE search
# even under MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES="".
# ---------------------------------------------------------------------------


def _same_result(a, b) -> bool:
    """Test double method _same_result."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_result(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same_result(u, v) for u, v in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_regression_numba_polynom_optimizer_seed_zero_not_substituted():
    """Pre-fix: seed=0 silently became seed=1 internally, so seed=0 and seed=1 produced identical searches.
    Post-fix: seed=0 is its own reproducible stream, distinct from seed=1, in both entry points."""
    from numpy.polynomial.hermite_e import hermeval

    from mlframe.feature_selection.filters import _numba_polynom_optimizer as npo

    rng = np.random.default_rng(0)
    n = 1500
    z_a = rng.standard_normal(n)
    z_b = rng.standard_normal(n)
    y = ((z_a * z_b) > 0).astype(np.int64)
    eval_kwargs = dict(
        eval_func=hermeval, bf_names=("mul", "add", "sub", "div"), z_a=z_a, z_b=z_b,
        discrete_target=True, y_njit=y, plugin_n_bins=20, l2_penalty=0.05,
    )

    def single(seed):
        """Test double method single."""
        return npo.run_numba_kernel_search(
            ca_size=3, cb_size=3, coef_range=(-1.0, 1.0), n_trials=60, seed=seed,
            direction_only=False, warm_start_seeds=None, eval_kwargs=eval_kwargs,
        )

    assert _same_result(single(0), single(0)), "seed=0 must be reproducible"
    assert not _same_result(single(0), single(1)), "seed=0 must not alias seed=1"

    X_arr = rng.standard_normal((n, 3))
    y_pairs = ((X_arr[:, 0] * X_arr[:, 1]) > 0).astype(np.int64)
    pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)

    def multi(seed):
        """Test double method multi."""
        return npo.optimize_all_pairs_numba_kernel(
            X_arr, y_pairs, pairs, ca_size=3, cb_size=3, coef_range=(-1.0, 1.0), n_trials=60, seed=seed,
        )

    assert _same_result(multi(0), multi(0)), "seed=0 must be reproducible"
    assert not _same_result(multi(0), multi(1)), "seed=0 must not alias seed=1"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-4 (P1): the GPU-resident additive-fusion OLS-R separability margin divided by the
# SCORING-SUBSAMPLE row count (n_sc) instead of the true full fit row count (n_rows), making the GPU
# threshold systematically stricter than the CPU sibling's at n above the subsample cap.
# ---------------------------------------------------------------------------


def test_regression_polynom_optimizer_5th_element_contract_documented():
    """Post-fix: both docstrings explicitly document that the 5th tuple element differs between backends,
    so a future reader/caller cannot assume it means the same thing."""
    from mlframe.feature_selection.filters._cupy_polynom_optimizer import run_cupy_kernel_search
    from mlframe.feature_selection.filters._numba_polynom_optimizer import run_numba_kernel_search

    # Asserts the CONTRACT is documented, not that an audit finding-ID survives in the text: comment-style
    # forbids process metadata in source, so pinning the ID makes the convention and this test contradict.
    for fn in (run_cupy_kernel_search, run_numba_kernel_search):
        doc = (fn.__doc__ or "").lower()
        assert "5th" in doc or "fifth" in doc, f"{fn.__name__} no longer documents the 5th tuple element"
        assert "differ" in doc or "not the same" in doc, f"{fn.__name__} no longer says the element differs by backend"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-11 (P2): both _hinge_detect_gpu_resident.py and _fe_additive_fusion_gpu_resident.py derived
# the subsample stride via floor division (n // max_rows), which yields stride==1 (no thinning at all) for
# n strictly between max_rows and 2*max_rows -- the documented "caps at <=max_rows" claim was only true
# once n reached ~2*max_rows. Fixed via ceiling division.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ORTH_BASIS_A-8 (P2): scan_integer_lattice_pairs (prototype) constructed a fresh np.random.default_rng
# inside the innermost per-candidate loop, so every candidate was tested against the IDENTICAL n_perm
# shuffles of y (same draws every time, since the generator restarts at the same seed each call).
# ---------------------------------------------------------------------------


def test_regression_orth_extra_basis_fe_generate_importable_directly_first():
    """Pre-fix: importing _orth_extra_basis_fe_generate BEFORE _orth_extra_basis_fe (e.g. in a fresh
    subprocess where nothing has touched this package yet) would raise ImportError on a partially
    initialized module. Post-fix: a lazy module __getattr__ breaks the load-order dependency."""
    import subprocess
    import sys

    code = (
        "from mlframe.feature_selection.filters._orthogonal_univariate_fe import "
        "_orth_extra_basis_fe_generate as g; "
        "assert callable(g.generate_extra_basis_features); "
        "from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import "
        "generate_extra_basis_features; "
        "assert callable(generate_extra_basis_features); "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-1 (P1): hybrid_orth_mi_adaptive_arity_fe_with_recipes's arity-2/3/4 branches built
# EngineeredRecipe / build_orth_quadruplet_cross_recipe WITHOUT preprocess_params_i/j/k/l -- the B-17
# replay-drift bug (frozen fit-time basis-preprocess params never persisted) was still open for arity>=2,
# which is Layer 78's entire point. The B-17 fix commit only touched this file's arity==1 branch.
# ---------------------------------------------------------------------------


def test_regression_adaptive_arity_pair_recipe_freezes_preprocess_params():
    """Pre-fix: an arity-2 recipe's extra dict had no preprocess_params_i/j keys at all (raw EngineeredRecipe
    construction). Post-fix: build_orth_pair_cross_recipe is used and both keys are populated (non-None for
    a basis with real preprocess params, e.g. zscore mean/std)."""
    import pandas as pd

    from mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe import (
        hybrid_orth_mi_adaptive_arity_fe_with_recipes,
    )

    rng = np.random.default_rng(1)
    n = 3000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x4 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    y = ((x1 * x2) + 0.02 * rng.standard_normal(n) > 0).astype(int)

    _X_aug, _uni, _adaptive, recipes = hybrid_orth_mi_adaptive_arity_fe_with_recipes(
        X,
        y,
        cols=["x1", "x2", "x3", "x4"],
        max_arity=4,
        max_degree=1,
        basis="hermite",
        seed_k=4,
    )
    pair_recipes = [r for r in recipes if r.kind == "orth_pair_cross"]
    assert pair_recipes, "expected at least one arity-2 recipe on a pure 2-way XOR signal"
    for r in pair_recipes:
        assert "preprocess_params_i" in r.extra, f"recipe {r.name} missing preprocess_params_i key entirely"
        assert "preprocess_params_j" in r.extra, f"recipe {r.name} missing preprocess_params_j key entirely"


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-3 (P2): all 11 univariate *_with_recipes builders wrapped the B-17-fixed
# _evaluate_basis_column(..., return_params=True) recompute in a bare except with zero logging.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modname,funcname",
    [
        ("_orthogonal_cmim_fe", "hybrid_orth_mi_cmim_fe_with_recipes"),
        ("_orthogonal_jmim_fe", "hybrid_orth_mi_jmim_fe_with_recipes"),
        ("_orthogonal_three_gate_mi_fe", "hybrid_orth_mi_three_gate_fe_with_recipes"),
        ("_orthogonal_bootstrap_mi_fe", "hybrid_orth_mi_bootstrap_fe_with_recipes"),
        ("_orthogonal_adaptive_arity_fe", "hybrid_orth_mi_adaptive_arity_fe_with_recipes"),
        ("_orthogonal_adaptive_degree_fe", "hybrid_orth_mi_adaptive_degree_fe_with_recipes"),
        ("_orthogonal_total_correlation_fe", "hybrid_orth_mi_tc_fe_with_recipes"),
        ("_orthogonal_ksg_mi_fe", "hybrid_orth_mi_ksg_fe_with_recipes"),
        ("_orthogonal_copula_mi_fe", "hybrid_orth_mi_copula_fe_with_recipes"),
        ("_orthogonal_dcor_fe", "hybrid_orth_mi_dcor_fe_with_recipes"),
        ("_orthogonal_hsic_fe", "hybrid_orth_mi_hsic_fe_with_recipes"),
    ],
)
def test_regression_scorer_zoo_preprocess_params_except_is_logged(modname, funcname, monkeypatch, caplog):
    """Pre-fix: `except Exception: _pp = None` with zero logging. Post-fix: a forced freeze failure is logged at debug."""
    import importlib
    import logging

    from mlframe.feature_selection.filters import _orthogonal_univariate_fe as uni

    scorer_mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    # Patch the binding the scorer actually calls. Some scorers import _evaluate_basis_column by name, so patching only the
    # univariate module is order-dependent: it takes effect only if the scorer was first imported inside this patch window.
    owner = scorer_mod if hasattr(scorer_mod, "_evaluate_basis_column") else uni
    real_evaluate = owner._evaluate_basis_column

    def _fail_only_when_freezing(*args, **kwargs):
        """Test double method _fail_only_when_freezing."""
        if kwargs.get("return_params"):
            raise RuntimeError("forced freeze failure")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(owner, "_evaluate_basis_column", _fail_only_when_freezing)
    fn = getattr(scorer_mod, funcname)

    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.standard_normal(n) * 3 + 10
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n)})
    signal = (x1 - 10.0) ** 2 + 0.6 * (x2**2)
    y = ((signal + 0.05 * rng.standard_normal(n)) > float(np.median(signal))).astype(int)
    # Default gates reject everything on this fixture for copula and ksg; relax them so the freeze path runs.
    relaxed = modname in ("_orthogonal_copula_mi_fe", "_orthogonal_ksg_mi_fe")
    extra_kwargs = {"min_uplift": 0.5, "min_abs_mi_frac": 0.0} if relaxed else {}

    caplog.set_level(logging.DEBUG)
    result = fn(X, y, cols=None, **extra_kwargs)
    assert result[-1], f"{modname}.{funcname} emitted no recipes; the freeze path was never reached"
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("failed to freeze fit-time basis preprocess_params" in m for m in messages), f"{modname}: freeze failure not logged"


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-4 (P2): _route_basis's except Exception: return "hermite" fallback had zero logging.
# ---------------------------------------------------------------------------


def test_regression_route_basis_exception_is_logged():
    """Post-fix: _route_basis's fallback is AUDIBLE (warning, not debug) before it freezes 'hermite'.

    Asserted structurally rather than by searching the source for a message, which is what the previous form
    did and why it broke: rewording the message to say what the fallback actually costs (a train/serve skew
    on that leg) left the test failing while the behaviour it guards got strictly better. The branch is
    defensive and not reachable end-to-end -- every leg of a winning recipe is a seed_sources member, so the
    cached lookup covers it -- so a caplog test cannot drive it without contriving internal state. What
    matters, and what this pins, is the LEVEL and the fallback value, neither of which prose can drift.
    """
    import ast
    import inspect

    from mlframe.feature_selection.filters import _orthogonal_adaptive_arity_fe as mod

    tree = ast.parse(inspect.getsource(mod))
    route = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_route_basis"),
        None,
    )
    assert route is not None, "_route_basis is gone; the routing fallback it guards needs a new home for this check"

    handlers = [n for n in ast.walk(route) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "_route_basis no longer has an except handler, so the routing failure is unguarded"

    levels = {
        n.func.attr
        for h in handlers
        for n in ast.walk(h)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "logger"
    }
    assert "warning" in levels, (
        f"_route_basis's fallback logs at {sorted(levels) or 'no level at all'}. It must warn: the frozen basis "
        f"is replayed by transform(), so recording 'hermite' for a column the fit routed elsewhere is a "
        f"train/serve skew, and debug is not emitted in production."
    )
    assert "debug" not in levels, "the routing fallback was downgraded back to debug, where production will not see it"

    returns = {n.value.value for n in ast.walk(route) if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)}
    assert "hermite" in returns, "the documented fallback basis is no longer returned"


# ---------------------------------------------------------------------------
# ORTH_SCORING_A-5 (P2): KSG/HSIC's near-zero-baseline uplift special-case hard-set uplift=0.0 whenever
# engineered_mi was below the FIXED _ABS_MI_FLOOR=1e-3, which could hard-reject a candidate the real
# DYNAMIC MAD-based abs_floor (gate 2) would have accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname", ["_orthogonal_ksg_mi_fe", "_orthogonal_hsic_fe"])
def test_regression_near_zero_baseline_uplift_defers_to_dynamic_floor(modname):
    """Pre-fix: uplift = 0.0 if emi < _ABS_MI_FLOOR else inf (could hard-reject a candidate the dynamic
    floor would accept). Post-fix: uplift is always inf for a near-zero baseline, deferring entirely to
    gate 2's dynamic floor."""
    import importlib

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    score = mod.score_features_by_ksg_mi_uplift if modname == "_orthogonal_ksg_mi_fe" else mod.score_features_by_hsic_uplift
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    raw_X = pd.DataFrame({"a": np.full(n, 3.0, dtype=np.float32)})  # constant source: zero baseline MI
    eng_X = pd.DataFrame({"a__h1": np.full(n, 0.5, dtype=np.float32)})  # constant engineered column: MI ~ 0
    row = score(raw_X, eng_X, y).iloc[0]
    # The engineered MI sits below the old fixed _ABS_MI_FLOOR, the band where the pre-fix code hard-set uplift=0.0.
    assert row["engineered_mi"] < mod._ABS_MI_FLOOR
    assert np.isinf(row["uplift"]), f"near-zero baseline must defer to gate 2 with uplift=inf; got {row['uplift']}"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-1 (P1): _orth_auto_scorer_fe.py's _score_plug_in coerced non-integer y via a bare
# `y_arr.astype(np.int64)` -- TRUNCATES, does not densify -- the exact B-18 bug class, reintroduced
# because this function was carved into a new sibling file BEFORE the B-18 fix pass patched its parent.
# ---------------------------------------------------------------------------


def test_regression_auto_scorer_recipe_freezes_preprocess_params():
    """Pre-fix: build_orth_univariate_recipe was called with no preprocess_params kwarg at all, so replay refit
    the z-score axis on a row slice. Post-fix: replay on a slice matches the fit-time engineered values."""
    from mlframe.feature_selection.filters._orth_auto_scorer_fe import hybrid_orth_mi_auto_scorer_fe_with_recipes

    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.standard_normal(n) * 3 + 10
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_0": rng.standard_normal(n)})
    signal = (x1 - 10.0) ** 2 + 0.6 * (x2**2)
    y = ((signal + 0.05 * rng.standard_normal(n)) > float(np.median(signal))).astype(int)

    X_aug, _scores, recipes = hybrid_orth_mi_auto_scorer_fe_with_recipes(X, y)
    assert recipes, "the auto-scorer emitted no recipes on this fixture; cannot exercise the replay contract"
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    X_slice = X.iloc[:50].copy()
    for r in recipes:
        assert r.extra["preprocess_params"] is not None, f"recipe {r.name!r} did not freeze its fit-time preprocess_params"
        replayed = np.asarray(apply_recipe(r, X_slice), dtype=np.float64)
        fit_time = X_aug[r.name].to_numpy(dtype=np.float64)[:50]
        assert np.allclose(replayed, fit_time, rtol=1e-5, atol=1e-6), f"recipe {r.name!r} replay refit the basis on the slice"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-3 (P1): _oracle_scorer_select.py's _ROWS_CACHE get-check-evict-write sequence had no
# lock, empirically reproducing crashes (dictionary changed size during iteration / KeyError) under
# concurrent access.
# ---------------------------------------------------------------------------


def test_regression_gradient_seeder_exceptions_logged_unconditionally(monkeypatch, caplog):
    """Post-fix: both failure paths log at verbose=0 - a non-coercible X at debug, a seeder crash at warning."""
    import logging
    import types

    from mlframe.feature_selection.filters import _gradient_interaction_seeder as mod

    def _boom(*_a, **_k):
        """Test double method _boom."""
        raise RuntimeError("forced seeder failure")

    monkeypatch.setattr(mod, "_route_gradient_seeder", lambda n_rows, n_pool: (True, "analytic", 200))
    fake = types.SimpleNamespace(fe_gradient_interaction_enable=True, feature_names_in_=np.array(["a", "b", "c"]))
    call = dict(
        self=fake, num_fs_steps=0, data=np.zeros((200, 3), dtype=np.int64), cols=["a", "b", "c"],
        target_indices=np.array([2]), categorical_vars=[], numeric_vars_to_consider={0, 1}, non_numeric_idx=set(), verbose=0,
    )
    caplog.set_level(logging.DEBUG, logger=mod.logger.name)

    caplog.clear()
    out = mod.propose_gradient_interaction_pairs(X_continuous="not numeric", **call)
    assert out == ({0, 1}, set())
    coercion = [r for r in caplog.records if "not array-coercible" in r.getMessage()]
    assert coercion, "non-coercible X was swallowed without any log at verbose=0"
    assert coercion[0].levelno == logging.DEBUG

    monkeypatch.setattr(mod, "rank_gradient_interaction_pairs", _boom)
    caplog.clear()
    out = mod.propose_gradient_interaction_pairs(X_continuous=np.random.default_rng(0).standard_normal((200, 3)), **call)
    assert out == ({0, 1}, set())
    failed = [r for r in caplog.records if "gradient-interaction seeder failed" in r.getMessage()]
    assert failed, "seeder failure was swallowed without any log at verbose=0"
    assert failed[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-4 (P1): _parse_engineered_name split on the FIRST "__", breaking for any source column
# whose own name contains "__" (e.g. this codebase's own orth-basis naming convention "{col}__{code}{deg}").
# ---------------------------------------------------------------------------


def test_regression_parse_engineered_name_handles_double_underscore_source():
    """Pre-fix: source 'a__b' with emitted name 'a__b__digit_1' parsed to None (recipe skipped).
    Post-fix: rsplit-based parsing correctly recovers src='a__b', code='digit_extract', arg=1."""
    from mlframe.feature_selection.filters._numeric_decompose_fe import _parse_engineered_name

    assert _parse_engineered_name("a__b__digit_1") == ("digit_extract", "a__b", 1)
    assert _parse_engineered_name("a__b__round_0p1") == ("numeric_rounding", "a__b", 0.1)
    assert _parse_engineered_name("plain__digit_2") == ("digit_extract", "plain", 2)


# ---------------------------------------------------------------------------
# CAT_INTERACTION_B-5 (P1): the lagged_diff family had no entity/group-scoping parameter at all, computing
# a global time-sorted diff that silently mixes rows across different entities on panel data.
# ---------------------------------------------------------------------------


def test_regression_raws_linearly_explain_y_rejects_high_p_overfit_noise():
    """Pre-fix: p=1900,n=2000 pure noise gave in-sample R^2=0.936, crossing the 0.92 threshold with ZERO
    real signal. Post-fix: held-out (K-fold CV) R^2 stays near/below 0 for pure noise regardless of p."""
    import pandas as pd

    from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y

    rng = np.random.default_rng(0)
    n, p = 2000, 1900
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = rng.standard_normal(n)
    assert raws_linearly_explain_y(X, y) is False, "high-p pure noise must NOT clear the linear-explains-y gate"


def test_regression_raws_linearly_explain_y_still_fires_on_real_signal():
    """A genuine strong linear relationship must still clear the gate (the fix must not make it
    permanently unreachable)."""
    import pandas as pd

    from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y

    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] * 2.0 + X["x1"] * 0.5 + rng.standard_normal(n) * 0.01
    assert raws_linearly_explain_y(X, y) is True, "a genuine strong linear relationship must still clear the gate"


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-2 (P1): four module-level bounded-FIFO memo caches used the unlocked
# `if len(cache) > N: cache.pop(next(iter(cache)))` eviction idiom with no lock, the same race class
# fixed elsewhere in this codebase for concurrent MRMR.fit() calls.
# ---------------------------------------------------------------------------


def _nested_composite_anchor_kwargs(n=1500, seed=7, n_unreferenced=200):
    """Fused ``sub(div(sqr(a),neg(b)),mul(log(c),sin(d)))`` engineered survivor over raws a..d, with y built from
    per-raw above-median indicators so each of a..d is signal bearing, plus raw names no recipe consumes."""
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe, build_unary_binary_recipe

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "a": rng.uniform(0.1, 10.0, n).astype(np.float32),
        "b": rng.uniform(0.5, 5.0, n).astype(np.float32),
        "c": rng.uniform(0.1, 10.0, n).astype(np.float32),
        "d": rng.uniform(0.0, 6.28, n).astype(np.float32),
    })
    kw = dict(unary_preset="minimal", binary_preset="minimal", quantization_nbins=None, quantization_method=None, quantization_dtype=np.float32)
    parent_a = build_unary_binary_recipe(name="div(sqr(a),neg(b))", src_a_name="a", src_b_name="b", unary_a_name="sqr", unary_b_name="neg", binary_name="div", **kw)
    parent_b = build_unary_binary_recipe(name="mul(log(c),sin(d))", src_a_name="c", src_b_name="d", unary_a_name="log", unary_b_name="sin", binary_name="mul", **kw)
    va = apply_recipe(parent_a, df)
    vb = apply_recipe(parent_b, df)
    fused_name = "sub(div(sqr(a),neg(b)),mul(log(c),sin(d)))"
    fused = build_unary_binary_recipe(
        name=fused_name, src_a_name="div(sqr(a),neg(b))", src_b_name="mul(log(c),sin(d))", unary_a_name="identity", unary_b_name="identity",
        binary_name="sub", fit_values_for_edges=(va - vb), nested_parent_a=parent_a, nested_parent_b=parent_b, **kw,
    )
    above = [(df[c].to_numpy() > np.median(df[c].to_numpy())).astype(np.int32) for c in ("a", "b", "c", "d")]
    y = (sum(above) >= 2).astype(np.int32)

    def _qbin(v, k=10):
        """Quantile-bin a continuous column into k int codes, as the fit path hands binned columns to the anchor builder."""
        edges = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(edges, v, side="right").astype(np.int32)

    eng = np.asarray(apply_recipe(fused, df), dtype=np.float32)
    data = np.column_stack([_qbin(df[c].to_numpy(np.float32)) for c in ("a", "b", "c", "d")] + [_qbin(eng)]).astype(np.int32)
    raw_names = ["a", "b", "c", "d"] + [f"n{i}" for i in range(n_unreferenced)]
    return dict(
        data=data, cols=["a", "b", "c", "d", fused_name], sel=[0, 1, 2, 3, 4], raw_name_set=set(raw_names),
        y_binned=y, y_continuous=None, engineered_continuous=None, replayable_eng_names=None,
        recipes={fused_name: fused}, raw_X=df, seed=0, verbose=0, n_rows=n, gate_resident=False,
    )


def test_regression_raw_redundancy_anchors_iterates_bounded_raw_set():
    """Post-fix: the inner loop iterates the bounded union of actually-referenced parents (_all_relevant_raws), not the
    caller's full raw_name_set. Output pin: the clean sub-expression anchors land on exactly the four referenced raws
    and on none of the unreferenced raw names carried in raw_name_set."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors

    ctx = build_raw_redundancy_anchors(**_nested_composite_anchor_kwargs())
    assert ctx.early_return is None
    assert {rn for rn, _ei in ctx.clean_subexpr_bin} == {"a", "b", "c", "d"}
    assert {ei for _rn, ei in ctx.clean_subexpr_bin} == {4}


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-3 (P1): _Y_DENSE_MEMO's read-check-evict-write sequence had no lock -- two
# MRMR.fit() calls on DIFFERENT instances in different threads of the same process could race
# next(iter(_Y_DENSE_MEMO)), raising RuntimeError: dictionary changed size during iteration.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fit_fn_name", ["_rbf_fit", "_sigmoid_fit"])
def test_regression_bases_fit_scrubs_nan_inf(fit_fn_name):
    """Regression: _rbf_fit / _sigmoid_fit did not guard NaN/Inf on the raw column before np.quantile/np.std, unlike _fourier_fit / _pade_fit."""
    from mlframe.feature_selection.filters import bases as bases_mod

    fit_fn = getattr(bases_mod, fit_fn_name)
    x = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0, -np.inf, 8.0, 9.0, 10.0])
    z, params = fit_fn(x)
    assert np.all(np.isfinite(z))
    for v in params.values():
        arr_v = np.asarray(v, dtype=np.float64)
        assert np.all(np.isfinite(arr_v)), f"{fit_fn_name} produced non-finite params: {params}"


# ---------------------------------------------------------------------------
# USABILITY_A-8 (P1): estimators.py's module docstring documented an MRMR(estimator=...) API that does
# not exist.
# ---------------------------------------------------------------------------


def test_regression_origin_from_rosters_accepts_prebuilt_sets():
    """Regression: _origin_from_rosters rebuilt all 13 roster-attr memberships from scratch, per name, instead of building them once before the per-name
    loop.
    """
    from mlframe.feature_selection.filters._mrmr_fe_provenance import (
        _build_roster_membership_sets,
        _origin_from_rosters,
    )

    class _FakeMRMR:
        """Minimal MRMR stand-in: carries just the attributes the code under test reads, nothing else."""

        mi_greedy_features_ = ["eng_col_a"]
        hybrid_orth_features_: list = []

    fake = _FakeMRMR()
    roster_sets = _build_roster_membership_sets(fake)
    assert _origin_from_rosters("eng_col_a", fake, roster_sets=roster_sets) == "mi_greedy"
    assert _origin_from_rosters("unknown_col", fake, roster_sets=roster_sets) == "engineered_unknown"
    # Omitting roster_sets still works (backward-compatible fallback path).
    assert _origin_from_rosters("eng_col_a", fake) == "mi_greedy"


# ---------------------------------------------------------------------------
# USABILITY_B-7 (P2, test_gap): _apply_sis_screen's polars-specific branches had zero test coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname,funcname", _ORTH_SCORER_ZOO_MODULES)
def test_regression_orth_scorer_zoo_recipe_freezes_preprocess_params(modname, funcname):
    """Slice replay of each scorer-zoo module's emitted recipe(s) must match the fit-time engineered
    value bit-for-bit -- pins the B-17 fix (frozen preprocess_params) across all 13 modules that
    commit f067e0d44 patched but did not individually test."""
    import importlib

    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    mod = importlib.import_module(f"mlframe.feature_selection.filters.{modname}")
    fn = getattr(mod, funcname)

    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.standard_normal(n) * 3 + 10  # mean=10, std=3 -- a small head slice differs materially
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n)})
    signal = (x1 - 10.0) ** 2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)

    # copula_mi's default uplift/MI-floor gate rejects everything on this fixture; relax it so the
    # test actually exercises the slice-replay contract instead of skipping.
    extra_kwargs = {"min_uplift": 0.5, "min_abs_mi_frac": 0.0} if modname == "_orthogonal_copula_mi_fe" else {}
    result = fn(X, y, cols=None, **extra_kwargs)
    # Result arity varies across the scorer zoo (some return an extra intermediate-scores frame),
    # but recipes is always the LAST element and X_aug is always the FIRST.
    X_aug, recipes = result[0], result[-1]
    assert recipes, f"{modname}.{funcname} emitted no recipes on this fixture; cannot exercise the slice-replay contract."

    for r in recipes:
        if "preprocess_params" not in r.extra:
            continue  # not every recipe kind in this family freezes basis-preprocess params (e.g. a pure interaction term)
        assert r.extra["preprocess_params"] is not None, f"{modname}: recipe {r.name!r} has a preprocess_params key but it is None -- B-17 regression."
        X_slice = X.iloc[:50].copy()
        fit_time_vals = X_aug[r.name].to_numpy()[:50]
        replayed_vals = apply_recipe(r, X_slice)
        # rtol/atol loose enough to tolerate ordinary FP-reorder noise (~1e-7, seen on the adaptive-degree
        # module) but tight enough to catch a genuine B-17 regression (slice-refit drift is orders of
        # magnitude larger -- the whole point of picking a slice with a materially different mean/std).
        assert np.allclose(replayed_vals, fit_time_vals, rtol=1e-5, atol=1e-6), (
            f"{modname}: recipe {r.name!r} slice replay diverged from fit-time values "
            f"(max abs diff={float(np.max(np.abs(np.asarray(replayed_vals) - fit_time_vals)))}) -- "
            f"the frozen preprocess_params are not being honoured at replay time."
        )


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-2 (P1): the Layer-27 noise-aware floor silently defaulted to 0.0 (a
# full no-op) whenever fewer than 4 scored engineered columns reached the scan -- realistically
# whenever at most 1 raw numeric source column is present (the common default-2-degrees case).
# ---------------------------------------------------------------------------


def test_regression_hybrid_orth_noise_floor_active_with_one_raw_column():
    """Post-fix: with a SINGLE weak raw numeric column (baseline_mi ~ 0), a purely noise-driven
    engineered column must NOT be admitted -- pre-fix, both the legacy and noise-aware floors
    collapsed to ~0 in this exact regime, admitting noise."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import hybrid_orth_mi_fe

    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)  # pure noise, no relationship to y
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"x1": x1})

    X_aug, _scores = hybrid_orth_mi_fe(X, y, degrees=(2, 3), basis="hermite")
    # A weak/irrelevant sole raw column must not spuriously admit a noise-driven engineered basis column.
    assert X_aug.shape[1] == X.shape[1], f"expected no engineered columns admitted on pure noise; got {list(X_aug.columns)}"


def test_regression_noise_floor_computed_without_size_gate(monkeypatch):
    """Post-fix: the MAD noise floors are computed from the scored candidates even when fewer than four exist.
    Injected scores (three candidates) put two weak engineered columns inside the noise band, so the gate must reject
    them and keep only the clear winner. Pre-fix, both floors collapsed to 0.0 below size 4 and all three were admitted."""
    from mlframe.feature_selection.filters import _fe_gpu_strict, _gpu_resident_fe
    from mlframe.feature_selection.filters import _orthogonal_univariate_fe as mod

    monkeypatch.setattr(_gpu_resident_fe, "fe_gpu_resident_basis_mi_enabled", lambda: False)
    monkeypatch.setattr(_fe_gpu_strict, "fe_gpu_strict_enabled", lambda **_kw: False)
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({"x1": rng.standard_normal(n).astype(np.float32)})
    names = ["x1__He2", "x1__He3", "x1__He4"]
    engineered = pd.DataFrame({nm: rng.standard_normal(n).astype(np.float32) for nm in names})
    scores = pd.DataFrame({
        "engineered_col": names,
        "source_col": ["x1", "x1", "x1"],
        "baseline_mi": [0.10, 0.12, 0.10],
        "engineered_mi": [0.30, 0.31, 0.60],
        "uplift": [3.0, 2.6, 6.0],
    })
    monkeypatch.setattr(mod, "generate_univariate_basis_features", lambda *a, **k: engineered)
    monkeypatch.setattr(mod, "score_features_by_mi_uplift", lambda *a, **k: scores.copy())

    X_aug, _scores = mod.hybrid_orth_mi_fe(X, y, degrees=(2, 3, 4), basis="hermite")
    assert list(X_aug.columns) == ["x1", "x1__He4"]


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-3 (P2): compute_class_weights had no upper bound on n_classes before
# allocating np.bincount(y_arr, minlength=n_classes) -- a caller-controlled unbounded allocation.
# ---------------------------------------------------------------------------


def test_regression_orth_univariate_docstring_no_longer_claims_not_wired():
    """X_EFFICIENCY_ARCHITECTURE-4: the module docstring must no longer falsely claim the univariate
    orth-basis constructor is opt-in-only / not wired into MRMR.fit by default."""
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as mod

    doc = mod.__doc__ or ""
    assert "NOT wired into MRMR.fit by default" not in doc


def test_regression_orth_univariate_bails_to_host_when_aux_pool_active(monkeypatch):
    """X_EFFICIENCY_ARCHITECTURE-3: with a semi-supervised unlabeled pool installed, hybrid_orth_mi_fe must not take the
    GPU-resident path (which ignores the pool), even when the GPU flags are forced on. The GPU twin is replaced by a
    recorder, so this runs on CPU regardless of CUDA availability."""
    from mlframe.feature_selection.filters import _fe_gpu_strict, _gpu_resident_fe
    from mlframe.feature_selection.filters import _orthogonal_univariate_fe as mod
    from mlframe.feature_selection.filters._semi_supervised_fe import unlabeled_pool_active

    monkeypatch.setattr(_gpu_resident_fe, "fe_gpu_resident_basis_mi_enabled", lambda: True)
    monkeypatch.setattr(_gpu_resident_fe, "_cuda_present", lambda: True)
    monkeypatch.setattr(_fe_gpu_strict, "fe_gpu_strict_enabled", lambda **_kw: True)
    gpu_calls = []

    def _gpu_recorder(X, cols, degrees, basis, y, nbins):
        """Record that the GPU-resident twin was taken and return the no-candidates result."""
        gpu_calls.append(cols)
        return None, [], pd.DataFrame(columns=["engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift"])

    monkeypatch.setattr(mod, "_gpu_build_and_score_univariate", _gpu_recorder)
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({"x1": rng.standard_normal(n).astype(np.float32)})
    y = (X["x1"].to_numpy() > 0).astype(np.int32)
    pool = {"x1": rng.standard_normal(2000).astype(np.float32)}

    mod.hybrid_orth_mi_fe(X, y, degrees=(2, 3), basis="hermite")
    assert len(gpu_calls) == 1, "control: with no pool installed the forced GPU flags must take the GPU-resident twin"

    gpu_calls.clear()
    with unlabeled_pool_active(pool):
        _X_aug, scores = mod.hybrid_orth_mi_fe(X, y, degrees=(2, 3), basis="hermite")
    assert gpu_calls == [], "GPU-resident path must not run while an aux pool is active"
    assert "x1__He2" in set(scores["engineered_col"]), "host path must score the candidates"


def test_regression_build_raw_redundancy_anchors_identifies_consumer():
    """FE_REDUNDANCY_SYNERGY-15 fix (mrmr_audit_2026-07-22, carried over from 2026-07-20's audit): direct
    unit test for ``build_raw_redundancy_anchors`` -- it had zero direct test anywhere across 3 audit
    generations, only ever exercised transitively through ``drop_redundant_raw_operands``'s end-to-end
    fixtures, which never actually asserted anything about the anchor-building phase's own contract."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors

    rng = np.random.default_rng(0)
    n = 200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y_cont = a + b + 0.01 * rng.standard_normal(n)

    def _bin(v, nbins=10):
        """Test helper: bin."""
        edges = np.quantile(v, np.linspace(0, 1, nbins + 1)[1:-1])
        return np.searchsorted(edges, v).astype(np.int64)

    cols = ["a", "b", "sum(a,b)"]
    data = np.column_stack([_bin(a), _bin(b), _bin(a + b)])
    y_binned = _bin(y_cont)

    out = build_raw_redundancy_anchors(
        data=data,
        cols=cols,
        sel=[0, 1, 2],
        raw_name_set={"a", "b"},
        y_binned=y_binned,
        y_continuous=None,
        engineered_continuous=None,
        replayable_eng_names=None,
        recipes=None,
        raw_X=None,
        seed=0,
        verbose=0,
        n_rows=n,
        gate_resident=False,
    )

    assert out.early_return is None
    assert out.eng_idx == [2]
    assert set(out.raw_sel_idx) == {0, 1}
    assert set(out.eng_consumers.get("a", [])) == {2}
    assert set(out.eng_consumers.get("b", [])) == {2}
    assert out.raw_is_signal_bearing("a") is True
    assert out.raw_is_signal_bearing("b") is True


def test_regression_build_raw_redundancy_anchors_no_engineered_survivors_early_returns():
    """FE_REDUNDANCY_SYNERGY-15 fix (mrmr_audit_2026-07-22): the degenerate "no engineered survivor"
    guard returns ``(sel, [])`` immediately without building any anchor state."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors

    n = 50
    cols = ["a"]
    data = np.zeros((n, 1), dtype=np.int64)
    sel = [0]

    out = build_raw_redundancy_anchors(
        data=data,
        cols=cols,
        sel=sel,
        raw_name_set={"a"},
        y_binned=np.zeros(n, dtype=np.int64),
        y_continuous=None,
        engineered_continuous=None,
        replayable_eng_names=None,
        recipes=None,
        raw_X=None,
        seed=0,
        verbose=0,
        n_rows=n,
        gate_resident=False,
    )

    assert out.early_return == (sel, [])
