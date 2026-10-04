"""Wave-28 sensors: ``is True``/``is False`` confusion fixes (5 sites).

Three P0 + 2 P1 sites where the pre-fix shape used identity-comparison
(``is True``/``is False``/``is not True``/``is not False``) on a value
that can legitimately be ``np.bool_(True)``/``np.bool_(False)`` from
config dicts. The numpy bool fails identity checks against Python's
``True``/``False`` singletons -> code paths got mis-routed silently
or raised cryptic TypeError / ValueError.

P0 sites:

#1 feature_engineering/transformer/random_features.py:_resolve_use_gpu
   ``use_gpu is False``/``is True`` raised
   ``ValueError("must be True, False, or 'auto'")`` when called with
   ``np.bool_(...)`` from a config dict. Fix: handle the string
   ``"auto"`` first via isinstance, then ``bool(use_gpu)``.

#2 feature_engineering/transformer/row_attention.py:_select_stage4_backend
   Same shape as #1 with ``gpu_stage4``. Fix: same.

#3 feature_engineering/transformer/random_features.py:compute_rff_features
   ``if standardize is not True and standardize is not False:`` strict
   type-guard rejected ``np.bool_`` and ``int(1)``. Fix:
   ``isinstance(standardize, (bool, np.bool_))``.

P1 sites:

#4 training/train_eval.py:_select  -- ``idx is False`` matched only
   Python False; ``numpy.False_`` slipped past the guard. Fix: explicit
   isinstance check.

#5 feature_selection/filters/screen.py:449 -- ``use_simple_mode is False``
   rejected ``np.bool_(False)`` from config, silently forcing the
   ``len < n`` branch. Fix: ``not use_simple_mode``.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---- #1 _resolve_use_gpu ------------------------------------------------


def test_resolve_use_gpu_accepts_numpy_bool_false():
    """Pre-fix np.bool_(False) raised ValueError; post-fix it's accepted."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    out = _resolve_use_gpu(use_gpu=np.bool_(False), work=100, threshold=None)
    assert out is False or out == False  # noqa: E712


def test_resolve_use_gpu_accepts_python_bool_false():
    """Resolve use gpu accepts python bool false."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    assert _resolve_use_gpu(use_gpu=False, work=100, threshold=None) == False  # noqa: E712


def test_resolve_use_gpu_string_auto_still_works():
    """Resolve use gpu string auto still works."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    # Without GPU available, "auto" returns False.
    out = _resolve_use_gpu(use_gpu="auto", work=0, threshold=None)
    assert isinstance(out, bool)


def test_resolve_use_gpu_invalid_string_raises():
    """Resolve use gpu invalid string raises."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    with pytest.raises(ValueError, match="must be True, False, or 'auto'"):
        _resolve_use_gpu(use_gpu="invalid", work=0, threshold=None)


# ---- #2 _select_stage4_backend source guard ------------------------------


def test_row_attention_stage4_backend_handles_numpy_bool():
    """np.bool_ flags select the CPU njit backend exactly like Python bools, and an unknown string is rejected."""
    from mlframe.feature_engineering.transformer._kernels_njit import row_attention_stage4_njit
    from mlframe.feature_engineering.transformer.row_attention import _select_stage4_backend

    assert _select_stage4_backend(gpu_stage4=np.bool_(False), n_queries_hint=1000) is row_attention_stage4_njit
    assert _select_stage4_backend(gpu_stage4=False, n_queries_hint=1000) is row_attention_stage4_njit
    with pytest.raises(ValueError, match="gpu_stage4 must be True, False, or 'auto'"):
        _select_stage4_backend(gpu_stage4="sometimes", n_queries_hint=1000)


# ---- #3 standardize isinstance guard -------------------------------------


def test_compute_rff_features_accepts_numpy_bool_standardize():
    """np.bool_ standardize behaves like the Python bool; a non-bool (int) is still rejected with TypeError."""
    from mlframe.feature_engineering.transformer.random_features import compute_rff_features

    X = np.random.default_rng(0).normal(size=(40, 3)).astype(np.float32)
    kwargs = dict(seed=1, n_features=8, use_gpu=False)
    plain = compute_rff_features(X, standardize=True, **kwargs).to_numpy()
    numpy_flag = compute_rff_features(X, standardize=np.bool_(True), **kwargs).to_numpy()
    np.testing.assert_array_equal(plain, numpy_flag)
    unstandardized = compute_rff_features(X, standardize=np.bool_(False), **kwargs).to_numpy()
    assert not np.allclose(plain, unstandardized)
    with pytest.raises(TypeError, match="standardize must be bool"):
        compute_rff_features(X, standardize=1, **kwargs)  # type: ignore[arg-type]


# ---- #4 _select np.False_ guard -----------------------------------------


def test_train_eval_select_handles_numpy_bool_false(monkeypatch):
    """A False-like train index (None, False, np.False_, empty) selects no target rows, while a real index selects exactly those rows.

    Pre-fix ``idx is False`` matched only Python False, so np.False_ indexed the target into an empty 2-D slice instead of selecting nothing.
    """
    from mlframe.training import trainer
    from mlframe.training.configs import TargetTypes
    from mlframe.training.targets import _train_eval_select_target as select_mod

    class _StopAfterSelect(Exception):
        """Raised by the stub that replaces the training-config call, ending select_target right after the index selection."""

    seen = {}

    def _capture_train_target(target_type, cur_target_name, model_name, composite_names, train_t, target):
        """Record the selected train target and hand the model name back unchanged."""
        seen["train_t"] = train_t
        return model_name

    def _stop(*args, **kwargs):
        """Stop select_target once the index selection has run."""
        raise _StopAfterSelect

    monkeypatch.setattr(select_mod, "_select_regression_target", _capture_train_target)
    monkeypatch.setattr(trainer, "configure_training_params", _stop)
    target = np.arange(10, dtype=np.float64) * 1.5

    def _run(idx):
        """Run select_target with the given train index and return the train target it selected."""
        seen.clear()
        with pytest.raises(_StopAfterSelect):
            select_mod.select_target("m", target, TargetTypes.REGRESSION, df=None, train_idx=idx)
        return seen["train_t"]

    for no_rows in (None, False, np.False_, [], np.array([], dtype=np.int64)):
        assert _run(no_rows) is None, repr(no_rows)
    np.testing.assert_array_equal(_run(np.array([1, 3, 5])), target[[1, 3, 5]])
    mask = np.zeros(10, dtype=bool)
    mask[[0, 9]] = True
    np.testing.assert_array_equal(_run(mask), target[[0, 9]])


# ---- #5 screen.py use_simple_mode ---------------------------------------


def test_screen_use_simple_mode_uses_not_not_is_false(monkeypatch):
    """np.bool_ use_simple_mode routes candidate scoring exactly like the Python bool: the worker pool runs unless simple mode has every candidate cached.

    Pre-fix ``use_simple_mode is False`` rejected np.bool_(False), silently forcing the serial branch.
    """
    from types import SimpleNamespace

    from mlframe.feature_selection.filters import _confirm_predictor as cp

    monkeypatch.setattr(cp, "_EVALUATE_CANDIDATES_POOL_ENABLED", True)
    monkeypatch.setattr(cp, "should_skip_candidate", lambda **kwargs: (False, 0))
    monkeypatch.setattr(cp, "_prefill_cond_MIs_gpu", lambda **kwargs: None)
    monkeypatch.setattr(cp, "handle_best_candidate", lambda **kwargs: (kwargs["best_gain"], kwargs["best_candidate"], False))
    serial_calls = []

    def _serial_candidate(**kwargs):
        """Record the serial evaluation of one candidate."""
        serial_calls.append(kwargs["cand_idx"])
        return 0.0, []

    monkeypatch.setattr(cp, "evaluate_candidate", _serial_candidate)

    def _score(use_simple_mode, n_cached, n_possible):
        """Score four candidates and return (pool dispatches, serial evaluations)."""
        pool_calls = []

        def _pool(generator):
            """Record a worker-pool dispatch without running the workers."""
            pool_calls.append(1)
            return []

        serial_calls.clear()
        ctx = SimpleNamespace(
            candidates=[(i,) for i in range(4)],
            interactions_order=1,
            only_unknown_interactions=False,
            failed_candidates=set(),
            added_candidates=set(),
            selected_vars=[],
            selected_interactions_vars=[],
            engineered_lineage=None,
            reduce_gain_on_subelement_chosen=False,
            n_workers=2,
            use_simple_mode=use_simple_mode,
            cached_MIs={i: 0.0 for i in range(n_cached)},
            num_possible_candidates=n_possible,
            cached_cond_MIs={},
            cached_jmim_MIs={},
            jmim_hit_counter=np.zeros(1, dtype=np.int64),
            entropy_cache={},
            workers_pool=_pool,
            y=None,
            factors_data=None,
            factors_nbins=None,
            factors_names=["a", "b", "c", "d"],
            classes_y=None,
            classes_y_safe=None,
            freqs_y=None,
            freqs_y_safe=None,
            use_gpu=False,
            partial_gains={},
            baseline_npermutations=0,
            mrmr_relevance_algo="fleuret",
            mrmr_redundancy_algo="fleuret",
            max_veteranes_interactions_order=1,
            cached_confident_MIs={},
            max_runtime_mins=0,
            start_time=0.0,
            min_relevance_gain=0.0,
            verbose=0,
            ndigits=4,
            random_seed=0,
        )
        cp.score_candidates(ctx, 0.0, None, np.zeros(4))
        return len(pool_calls), len(serial_calls)

    for off in (False, np.bool_(False)):
        assert _score(off, n_cached=4, n_possible=4) == (1, 0), repr(off)
    for on in (True, np.bool_(True)):
        assert _score(on, n_cached=4, n_possible=4) == (0, 4), repr(on)
        assert _score(on, n_cached=1, n_possible=4) == (1, 0), repr(on)
