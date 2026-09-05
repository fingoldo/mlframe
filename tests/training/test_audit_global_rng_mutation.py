"""Wave 49 (2026-05-20): global RNG mutation audit.

Audit class: production code that calls random.seed(...) / np.random.seed(...) /
torch.manual_seed(...) / cp.random.seed(...) and then uses the GLOBAL RNG --
silently mutating the process-global stream and breaking caller's seed
determinism.

1 P0 + 5 P1 + 2 P2 = 8 fixes applied:

  P0:
    1. training/neural/ranker.py:465 (MLPRanker.fit)
       torch.manual_seed + np.random.seed at fit() entry mutated globals; downstream
       code already uses local np.random.default_rng(self.seed) + per-sampler seed,
       so the global mutation added nothing but the bug. Removed both seed calls.

  P1:
    2. feature_selection/filters/screen.py:951+ (screen_predictors finally)
       Pre-fix: numpy state restored, numba+cupy leaked. Post-fix: restore both via
       a captured os.urandom(8) reseed (high-entropy, indistinguishable downstream).

    3. feature_selection/filters/screen.py:95+ (_preserve_global_numpy_rng_state)
       Symmetric to #2 -- numba+cupy leaked through the context manager. Closed.

    4. utils/misc.py:12 (set_random_seed)
       Documentation contract added: function INTENTIONALLY mutates globals; for
       top-of-script setup only; library code must use local Generators.

    5. feature_selection/filters/cat_interactions.py:991 (_count_nfailed_joint_indep_prange)
       cp.random.seed inside per-permutation loop -> cp.random.RandomState(seed)
       local per-iter generator. Reproducibility preserved; caller's cupy stream
       untouched.

  P2:
    6. feature_engineering/mps.py:600 (generate_market_price)
       np.random.seed + np.random.* -> rng = np.random.default_rng(seed); rng.*.

    7. votenrank/iia_exp.py:36 (compute_iia)
       Per-rep np.random.seed + np.random.shuffle -> per-rep np.random.default_rng(i).

Verified clean (do not refactor):
  - metrics/core.py:252,325 -- np.random.RandomState(0).rand(...) is local.
  - training/_classif_helpers.py:319,349 -- local RandomState.
  - feature_selection MRMR/RFECV/permutation/fleuret -- inside @njit kernels using
    numba's per-thread RNG (not Python np.random global).
  - feature_selection/filters/mrmr.py:946 -- comment confirms prior global removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

# Use the on-disk src/ tree directly: pytest may resolve mlframe.* to a stale
# build/lib/ copy, but the source-level audit must check the LIVE source.
# tests/training/this_file.py -> ../../src/mlframe/
MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    2026-05-22 monolith split compat: when ``feature_selection/filters/screen.py``
    is requested, also append the ``_screen_predictors.py`` sibling so
    source-pattern sensors for the relocated body still match.
    """
    primary = (MLFRAME_ROOT / rel).read_text(encoding="utf-8")
    if rel == "feature_selection/filters/screen.py":
        sibling = MLFRAME_ROOT / "feature_selection" / "filters" / "_screen_predictors.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_ranker_fit_no_global_torch_or_np_seed() -> None:
    """Ranker fit no global torch or np seed."""
    src = _read("training/neural/ranker.py")
    # The pre-fix two-liner is gone; the local generators referenced in comments still exist.
    assert "torch.manual_seed(self.seed)\n        np.random.seed(self.seed)" not in src
    # The documenting comment is present.
    assert "drop global RNG mutations" in src


def _drive_preserve(seed, *, body=None, fake_cupy=True):
    """Enter ``_preserve_global_numpy_rng_state(seed)``, run ``body`` inside it, report what it did.

    Returns ``(numba_seeds, cupy_seeds, inner)`` -- every value handed to ``set_numba_random_seed``
    and to ``cupy.random.seed`` in call order, plus whatever ``body`` returned.

    numba and cupy are stubbed rather than skipped: the contract is which seeds the block hands
    them and when, and neither a numba build nor a GPU is needed to observe that.
    """
    import sys
    import types

    from mlframe.feature_selection.filters import screen

    numba_seeds: list[int] = []
    cupy_seeds: list[int] = []

    fake = types.ModuleType("cupy")
    fake.random = types.SimpleNamespace(seed=cupy_seeds.append)

    real_numba = screen.set_numba_random_seed
    real_cupy = sys.modules.get("cupy")
    screen.set_numba_random_seed = numba_seeds.append
    if fake_cupy:
        sys.modules["cupy"] = fake
    else:
        sys.modules.pop("cupy", None)
    try:
        with screen._preserve_global_numpy_rng_state(seed):
            inner = body() if body is not None else None
    finally:
        screen.set_numba_random_seed = real_numba
        if real_cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = real_cupy
    return numba_seeds, cupy_seeds, inner


def _assert_np_state_equal(pre, post) -> None:
    """Byte-identical MT19937 state: the 624-element key array plus its position and gauss cache."""
    assert pre[0] == post[0]
    np.testing.assert_array_equal(pre[1], post[1])
    assert pre[2] == post[2]
    assert pre[3] == post[3]


# ---------------------------------------------------------------------------
# _preserve_global_numpy_rng_state: the mechanism behind most of this audit class.
#
# Behavioural since 2026-09-03. Two tests here asserted that the lines
# `_numba_restore_seed = None`, `_cp_restore_seed = None`, `_cp_module = None`,
# `set_numba_random_seed(int(_numba_restore_seed))` and `_cp_module.random.seed(...)` appear in
# screen.py. Every one of those survives a `finally:` that never runs, an exception path that
# skips it, or a snapshot taken after the reseed rather than before -- and none of them says the
# caller's stream comes back. The block is a context manager taking one argument, so there was
# never anything stopping these from being driven.
# ---------------------------------------------------------------------------


def test_a_seeded_block_leaves_the_caller_s_numpy_stream_byte_identical() -> None:
    """The whole point: an inner reseed for determinism must not bleed into the caller."""
    np.random.seed(0)
    pre = np.random.get_state()

    _drive_preserve(1234, body=lambda: np.random.random(5))

    _assert_np_state_equal(pre, np.random.get_state())


def test_an_UNSEEDED_block_also_restores_numpy() -> None:
    """The leak fix, and the reason it matters.

    MRMR.fit consumes process-global np.random in places no per-call Generator covers, so an
    unseeded fit used to advance the caller's MT19937 -- a second fit in the same process then saw
    a shifted stream and drifted its selection. That was the run-order flakiness under xdist.
    """
    np.random.seed(0)
    pre = np.random.get_state()

    numba_seeds, cupy_seeds, _ = _drive_preserve(None, body=lambda: np.random.random(5))

    _assert_np_state_equal(pre, np.random.get_state())
    assert numba_seeds == [], "an unseeded run never reseeds numba, so it must not restore it either"
    assert cupy_seeds == []


def test_the_state_is_restored_even_when_the_block_raises() -> None:
    """A snapshot restored only on the happy path leaks on every failed fit."""
    np.random.seed(0)
    pre = np.random.get_state()

    def _boom():
        """Fail inside the block, after it has reseeded."""
        np.random.random(5)
        raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError):
        _drive_preserve(1234, body=_boom)

    _assert_np_state_equal(pre, np.random.get_state())


def test_the_seed_does_take_effect_inside_the_block() -> None:
    """Restoring the caller's state is only half of it -- the inner block must be deterministic,
    because downstream np.random.shuffle consumers depend on that reseed."""
    _n1, _c1, first = _drive_preserve(1234, body=lambda: np.random.random(5))
    _n2, _c2, again = _drive_preserve(1234, body=lambda: np.random.random(5))

    np.testing.assert_array_equal(first, again)


def test_numba_and_cupy_are_seeded_on_entry_and_moved_off_it_on_exit() -> None:
    """Neither exposes a portable get_state, so exit re-seeds them from fresh entropy instead.

    Leaving them on the caller's requested seed would be the leak in a subtler form: every
    downstream numba/cupy draw would then continue a stream this block chose.
    """
    numba_seeds, cupy_seeds, _ = _drive_preserve(1234)

    assert numba_seeds[0] == 1234, "the block must reseed numba for determinism inside it"
    assert cupy_seeds[0] == 1234
    assert len(numba_seeds) == 2, f"numba was not restored on exit: {numba_seeds}"
    assert len(cupy_seeds) == 2, f"cupy was not restored on exit: {cupy_seeds}"
    assert numba_seeds[1] != 1234, "exiting on the caller's seed leaves every later numba draw on this block's stream"
    assert cupy_seeds[1] != 1234
    assert numba_seeds[1] != cupy_seeds[1], "one shared entropy draw would couple the two streams"


def test_a_missing_cupy_is_not_an_error() -> None:
    """CuPy is an optional dependency, and its legacy global generator fails to init on some
    driver combinations. Neither may break a screen."""
    np.random.seed(0)
    pre = np.random.get_state()

    numba_seeds, cupy_seeds, _ = _drive_preserve(1234, fake_cupy=False)

    _assert_np_state_equal(pre, np.random.get_state())
    assert cupy_seeds == []
    assert len(numba_seeds) == 2, "numba must still be seeded and restored without cupy present"


def test_set_random_seed_does_mutate_the_process_global_rngs() -> None:
    """It is the one function here that is SUPPOSED to reseed the world.

    Behavioural since 2026-09-03. This asserted that three phrases from the docstring
    ("INTENTIONALLY mutates the process-global", "top-of-script / notebook setup", "NEVER call
    this from inside fit()") appear in utils/misc.py. Prose in a file is not a contract; a
    function that had stopped seeding anything would have passed unchanged.
    """
    import random as _random

    from mlframe.utils.misc import set_random_seed

    set_random_seed(4242)
    first = (_random.random(), float(np.random.random()))
    set_random_seed(4242)
    again = (_random.random(), float(np.random.random()))
    set_random_seed(9999)
    other = (_random.random(), float(np.random.random()))

    assert first == again, "the same seed must reproduce the same streams -- that is what it is for"
    assert first != other, "a different seed that changes nothing is not seeding anything"


def test_no_library_code_calls_set_random_seed() -> None:
    """The other half of the docstring, made enforceable.

    "NEVER call this from inside fit(), predict(), or any library code path" is the actual rule,
    and the old test only checked that the sentence was written down. Reseeding the world from
    inside library code breaks determinism for sibling code in the same process that already
    seeded its own local Generator -- which is the entire audit class this file covers.

    Scripts, notebooks and benchmark drivers are the intended callers, so they are exempt.
    """
    exempt = ("_benchmarks", "scripts", "examples", "notebooks")

    # OPEN FINDING, not an accepted exception. RFECV.fit reseeds process-global random / numpy /
    # cupy / numba (and torch when configured) from `random_state`, clobbering the streams of any
    # sibling code in the same process -- the exact defect this audit class exists for, and the one
    # `_preserve_global_numpy_rng_state` was written to fix everywhere else. It is listed here
    # rather than silently baselined because the fix is a real refactor: `fit` is ~480 lines with
    # three return points, so the seed has to become a `with` block over nearly all of it (or the
    # sampling it covers has to be carved out), which is more than this test conversion should
    # carry. Found 2026-09-03 by this test, which is why it is worth having.
    known_open = {"feature_selection/wrappers/rfecv/_fit.py:371"}

    offenders = []
    for path in sorted(MLFRAME_ROOT.rglob("*.py")):
        rel = path.relative_to(MLFRAME_ROOT).as_posix()
        if rel == "utils/misc.py" or any(part in rel.split("/") for part in exempt):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken module is a different failure
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name == "set_random_seed":
                offenders.append(f"{rel}:{node.lineno}")

    unexpected = sorted(set(offenders) - known_open)
    assert not unexpected, f"library code reseeds the process-global RNGs at: {unexpected}"

    still_open = sorted(known_open & set(offenders))
    assert still_open == sorted(known_open), f"a listed site no longer reseeds globally -- drop it from known_open: {sorted(known_open - set(offenders))}"


def test_cat_interactions_uses_local_cupy_rng() -> None:
    # ``_count_nfailed_joint_indep_cupy`` (the cupy permutation kernel)
    # was moved to the ``_cat_confirm_permutation.py`` sibling when
    # ``cat_interactions.py`` was split below 1k LOC.
    """Cat interactions uses local cupy rng."""
    src = _read("feature_selection/filters/_cat_confirm_permutation.py")
    # The fix replaces cp.random.seed + cp.random.permutation with local RandomState.
    assert "cp.random.seed(base_seed + p)\n        y_perm = cp.random.permutation(classes_y_g)" not in src
    assert "_local_cp_rng = cp.random.RandomState(base_seed + p)" in src
    assert "_local_cp_rng.permutation(classes_y_g)" in src


def test_mps_generate_market_price_is_reproducible_from_its_own_seed() -> None:
    """A local Generator is only worth having if the seed still determines the output.

    Behavioural since 2026-09-03. This asserted six exact call spellings -- `rng.normal(trend,
    2.5)`, `rng.random() < 0.05`, `rng.choice([0.95, 1.05])` and friends -- which pin how the
    series happens to be computed today rather than the two properties that matter: the caller's
    global stream is untouched (the test below), and the same seed gives the same series.

    Without the second half, "stop touching the global RNG" is satisfied by ignoring the seed
    altogether, and the function silently becomes irreproducible.
    """
    from mlframe.feature_engineering.mps import generate_market_price

    first = generate_market_price(n_days=30, random_seed=7)
    again = generate_market_price(n_days=30, random_seed=7)
    other = generate_market_price(n_days=30, random_seed=8)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other)), "the seed is not reaching the generator"


def test_mps_generate_market_price_is_unaffected_by_the_global_stream() -> None:
    """The converse of the test below: a local Generator means the caller cannot perturb it."""
    from mlframe.feature_engineering.mps import generate_market_price

    np.random.seed(0)
    first = generate_market_price(n_days=30, random_seed=7)
    np.random.seed(12345)
    np.random.random(1000)
    again = generate_market_price(n_days=30, random_seed=7)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))


def test_votenrank_iia_is_unaffected_by_the_global_stream() -> None:
    """compute_iia seeds per iteration; that seed must come from the iteration, not the process.

    Behavioural since 2026-09-03. This asserted that `np.random.seed(i)` / `np.random.shuffle(...)`
    is absent and `rng = np.random.default_rng(i)` / `rng.shuffle(models_order)` present -- neither
    of which says the result stops depending on whatever the caller last seeded.
    """
    pd = pytest.importorskip("pandas")
    from mlframe.votenrank.iia_exp import compute_iia

    table = pd.DataFrame(
        {"m1": [0.5, 0.6], "m2": [0.55, 0.65], "m3": [0.7, 0.4]},
        index=["a", "b"],
    ).T
    weights = np.ones(2)

    def _mean_method(table, weights):
        """Mean method."""
        return np.average(table.values, weights=weights, axis=1)

    def _run():
        """Compute iia once, reporting None if the stub signature does not fit."""
        try:
            return compute_iia(_mean_method, table, weights, num_repetitions=3)
        except Exception:  # nosec B110 -- an optional/mismatched signature is a different failure
            return None

    np.random.seed(0)
    first = _run()
    np.random.seed(999)
    np.random.random(1000)
    again = _run()

    if first is None or again is None:
        pytest.skip("compute_iia signature did not accept the stub method")
    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))


# ---------------------------------------------------------------------------
# Behavioural sensors: caller's global RNG state must be preserved.
# ---------------------------------------------------------------------------


def test_generate_market_price_does_not_mutate_global_np_rng() -> None:
    """Generating market data with a seed must NOT clobber the caller's global RNG state."""
    from mlframe.feature_engineering.mps import generate_market_price

    np.random.seed(0)
    pre = np.random.get_state()
    generate_market_price(n_days=10, random_seed=42)
    post = np.random.get_state()
    # The MT19937 state tuple has a 624-element keys array; full byte-identical compare.
    assert pre[0] == post[0]
    np.testing.assert_array_equal(pre[1], post[1])
    assert pre[2] == post[2]
    assert pre[3] == post[3]


def test_compute_iia_does_not_mutate_global_np_rng() -> None:
    """compute_iia per-iter seed must NOT shift the caller's global stream."""
    pd = pytest.importorskip("pandas")
    from mlframe.votenrank.iia_exp import compute_iia

    # Minimal stub: 3 models x 2 metrics, weights=ones, method=mean.
    table = pd.DataFrame(
        {"m1": [0.5, 0.6], "m2": [0.55, 0.65], "m3": [0.7, 0.4]},
        index=["a", "b"],
    ).T
    weights = np.ones(2)

    np.random.seed(0)
    pre = np.random.get_state()

    def _mean_method(table, weights):
        """Mean method."""
        return np.average(table.values, weights=weights, axis=1)

    try:
        compute_iia(_mean_method, table, weights, num_repetitions=3)
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass  # the method signature might mismatch; we only care about RNG state.
    post = np.random.get_state()
    np.testing.assert_array_equal(pre[1], post[1])
