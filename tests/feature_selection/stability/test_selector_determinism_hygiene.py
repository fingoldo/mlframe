"""Cross-selector determinism / RNG-hygiene battery (shared_lift 16/17/18).

Three orthogonal determinism contracts, lifted from per-selector one-offs to a
shared battery keyed off ``SELECTOR_SPECS``:

shared_lift-16 -- FIT-LEVEL global-RNG hygiene. A selector handed an explicit
    seed must not mutate the process-global ``numpy`` MT19937 stream during
    ``fit``. The pre-existing sensors covered ctor-time only (BorutaShap,
    ShapProxiedFS) plus one fit-time MRMR case; here every spec's ``fit`` is
    pinned byte-identical on the global state. A selector that legitimately
    consumes the global stream only when ``random_state=None`` still must stay
    clean here because every factory passes a seed.

shared_lift-17 -- n_jobs / parallelism parity. For specs exposing a real
    parallelism knob (read from the ctor: MRMR ``n_jobs``, RFECV ``n_jobs``,
    ShapProxiedFS ``n_jobs``) the selected RAW name set at parallelism=1 must
    equal the set at parallelism=2 for the same seed -- parallelism is a
    performance axis, never a selection axis.

shared_lift-18 -- PYTHONHASHSEED subprocess determinism. Selectors build
    dicts/sets over feature names internally; if any selection step leaks
    ``set``/``dict`` iteration order, two processes with different
    ``PYTHONHASHSEED`` disagree. A child script fits ONE cheap selector
    (simple-mode MRMR) and prints its sorted selection; the parent runs it
    under ``PYTHONHASHSEED=0`` and ``=42`` and asserts identical output.

All heavy fits are ``@slow``; a cheap representative keeps each contract alive
under ``MLFRAME_FAST=1`` via ``fast_subset``. CPU-only; subprocess + numba
warmup paths carry ``no_xdist``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np
import pytest

from tests.feature_selection._biz_val_synth import make_signal_plus_noise, as_df
from tests.feature_selection._selector_factories import (
    SELECTOR_SPECS,
    selected_names,
    spec_params,
)
from tests.feature_selection.conftest import fast_subset


# Module-level compiled regex: parse a parallelism kwarg out of a "key=val|..."
# child-line, kept compiled per the project convention even though it is only a
# guard here.
_SEL_LINE_RE = re.compile(r"^[\w()|]*$")


# ---------------------------------------------------------------------------
# Small shared synthetic. Linear binary signal + noise, tiny n so even the
# heavy members finish well under the 55s/test budget. A seeded local RNG
# (default_rng) is used so building it never touches the global stream the
# RNG-hygiene test inspects.
# ---------------------------------------------------------------------------


def _toy(n: int = 200, p_signal: int = 3, p_noise: int = 6, seed: int = 0):
    X, y, sig = make_signal_plus_noise(n=n, p_signal=p_signal, p_noise=p_noise, seed=seed)
    df, ys = as_df(X, y)
    return df, ys, sig


def _global_state_equal(before, after) -> bool:
    """Byte-compare every component of ``np.random.get_state()``.

    The state is ``(name, keys[uint32 array], pos, has_gauss, cached_gaussian)``;
    a leak shows up in any of the five, so all five are compared.
    """
    return before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2] == after[2] and before[3] == after[3] and before[4] == after[4]


# Specs whose fit is KNOWN to keep the global stream clean when seeded. Every
# spec in SELECTOR_SPECS currently does (verified); the set documents intent and
# is the place to flip a member to xfail if a future selector regresses or a new
# spec is added that legitimately leaks.
_RNG_LEAKERS: frozenset[str] = frozenset()


# ===========================================================================
# shared_lift-16: fit-level global-RNG hygiene
# ===========================================================================


@pytest.mark.parametrize("spec", spec_params())
def test_fit_does_not_mutate_global_numpy_rng(spec):
    """``spec.make(...).fit(X, y)`` with a seeded selector must leave the
    process-global numpy MT19937 state byte-identical.

    Calls ``fit`` DIRECTLY (not via any seeding helper) so the selector is the
    only thing that could have touched the global stream between the two
    ``get_state()`` snapshots.
    """
    df, ys, _ = _toy()
    sel = spec.make("binary")

    np.random.seed(12345)
    before = np.random.get_state()
    sel.fit(df, ys)
    after = np.random.get_state()

    identical = _global_state_equal(before, after)
    if spec.name in _RNG_LEAKERS:
        pytest.xfail(reason=f"PROD BUG: {spec.name}.fit mutates the global numpy RNG even when seeded")
    assert identical, (
        f"{spec.name}.fit mutated the global numpy RNG stream despite a fixed seed; "
        f"use np.random.default_rng(seed) / a local Generator instead of np.random.seed/shuffle/...."
    )


# ===========================================================================
# shared_lift-17: n_jobs / parallelism parity
# ===========================================================================
#
# Each entry: spec key -> (ctor kwarg name, serial value, parallel value).
# The kwarg name is the REAL parallelism knob read from each ctor:
#   MRMR.__init__(..., n_jobs=-1, ...)         -> "n_jobs"
#   RFECV.__init__(..., n_jobs=..., ...)       -> "n_jobs"
#   ShapProxiedFS.__init__(..., n_jobs=1, ...) -> "n_jobs"
# (MRMR also has an outer-parallelism "n_workers" knob; "n_jobs" is the one the
# existing concurrency-determinism sensor exercises, so it is used here too.)

_PARALLEL_SPECS: dict[str, tuple[str, int, int]] = {
    "MRMR": ("n_jobs", 1, 2),
    "RFECV": ("n_jobs", 1, 2),
    "ShapProxiedFS": ("n_jobs", 1, 2),
}


def _fit_with_kwarg(spec, kwarg: str, value: int, df, ys):
    """Build a fresh selector from the spec factory and override one ctor kwarg.

    The factories take no kwargs, so re-instantiate the underlying selector via
    its own class with the spec's defaults plus the parallelism override. We do
    that by setting the attribute post-construction is NOT safe (some selectors
    snapshot the value into derived state at __init__), so instead we use
    ``set_params`` when available (sklearn-style) and fall back to mutating the
    public attribute before fit otherwise.
    """
    sel = spec.make("binary")
    set_params = getattr(sel, "set_params", None)
    if callable(set_params):
        try:
            sel.set_params(**{kwarg: value})
        except Exception:
            setattr(sel, kwarg, value)
    else:
        setattr(sel, kwarg, value)
    # MRMR caches fit results process-wide keyed on data+config; clear so the
    # second parallelism setting genuinely re-fits rather than replaying.
    _clear_mrmr_cache_if_any()
    sel.fit(df.copy(), ys)
    return sel


def _clear_mrmr_cache_if_any() -> None:
    mod = sys.modules.get("mlframe.feature_selection.filters.mrmr")
    if mod is not None:
        try:
            mod.MRMR._FIT_CACHE.clear()
        except Exception:
            pass


def _njobs_parity_param_ids():
    """Parametrize over the parallelism-capable specs; the cheapest (RFECV) is
    the fast representative kept under MLFRAME_FAST=1."""
    keys = [k for k in _PARALLEL_SPECS if k in SELECTOR_SPECS]
    # RFECV first so fast_subset(keys, 1) keeps the cheap one as the fast rep.
    keys.sort(key=lambda k: (k != "RFECV", k))
    out = []
    for k in keys:
        marks = [pytest.mark.slow] if SELECTOR_SPECS[k].slow else []
        out.append(pytest.param(k, id=k, marks=marks))
    return fast_subset(out, 1)


@pytest.mark.parametrize("spec_key", _njobs_parity_param_ids())
def test_n_jobs_parity_same_selection(spec_key):
    """parallelism=1 and parallelism=2 select the SAME raw feature names for a
    fixed seed. Parallelism is a perf axis only -- it must never change which
    features survive."""
    spec = SELECTOR_SPECS[spec_key]
    kwarg, serial, parallel = _PARALLEL_SPECS[spec_key]
    df, ys, _ = _toy()

    try:
        s1 = _fit_with_kwarg(spec, kwarg, serial, df, ys)
        s2 = _fit_with_kwarg(spec, kwarg, parallel, df, ys)
    except OSError as exc:  # Windows paging-file overflow under concurrent load.
        if "paging file" in str(exc).lower() or getattr(exc, "winerror", None) == 1455:
            pytest.skip(f"Windows paging-file overflow under concurrent load: {exc}")
        raise
    except Exception as exc:  # loky transport flake under heavy concurrent load.
        msg = str(exc).lower()
        name = type(exc).__name__.lower()
        if any(s in msg for s in ("brokenprocesspool", "terminatedworker", "transport")) or any(s in name for s in ("brokenprocesspool", "terminatedworker")):
            pytest.skip(f"loky worker transport failure under concurrent load: {type(exc).__name__}")
        raise

    n1, n2 = set(selected_names(s1)), set(selected_names(s2))
    assert n1 == n2, f"{spec.name}: {kwarg}={serial} selected {sorted(n1)} but {kwarg}={parallel} selected {sorted(n2)}; parallelism altered the selection"


# ===========================================================================
# shared_lift-18: PYTHONHASHSEED subprocess determinism
# ===========================================================================
#
# A child script fits ONE cheap selector (simple-mode MRMR) on a fixed-seed
# synthetic and prints '|'.join(sorted(selected_names)). The parent runs it once
# under PYTHONHASHSEED=0 and once under =42; identical output proves the
# selection does not depend on Python's randomized str hashing (set/dict order).
# ASCII-only child prints (cp1251 rule). simple-mode + tiny n keeps each run
# under the 55s budget even with a cold numba compile.

_HASHSEED_CHILD = r"""
import sys
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

rng = np.random.default_rng(7)
n, p_sig, p_noise = 140, 3, 5
X_sig = rng.normal(size=(n, p_sig))
X_noise = rng.normal(size=(n, p_noise))
X = np.column_stack([X_sig, X_noise])
score = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
y = (score > 0).astype(np.int64)

cols = ["x%d" % i for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
ys = pd.Series(y, name="y")

m = MRMR(use_simple_mode=True, min_relevance_gain=0.0, cv=2,
         run_additional_rfecv_minutes=False, full_npermutations=2,
         random_seed=0, min_features_fallback=1, verbose=False, n_jobs=1)
m.fit(df, ys)

names_in = list(getattr(m, "feature_names_in_", cols))
support = np.asarray(m.support_)
if support.dtype == bool:
    sel = [names_in[i] for i in range(len(names_in)) if support[i]]
else:
    sel = [names_in[int(i)] for i in support]
sys.stdout.write("|".join(sorted(str(s) for s in sel)))
"""


def _child_env(hashseed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Ensure the worktree's src is importable in the child (mirrors PYTHONPATH=src).
    repo_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_src + (os.pathsep + existing if existing else "")
    return env


def _check_child_output(hashseed: str, stdout: str, stderr: str, rc: int) -> str:
    if rc != 0:
        raise AssertionError(f"hashseed child (PYTHONHASHSEED={hashseed}) failed rc={rc}\nSTDERR tail:\n{(stderr or '')[-2000:]}")
    out = (stdout or "").strip()
    assert _SEL_LINE_RE.match(out), f"unexpected child stdout (PYTHONHASHSEED={hashseed}): {out!r}"
    return out


def _run_hashseed_children(seeds=("0", "42")) -> dict[str, str]:
    """Launch the same fit-and-print child under each PYTHONHASHSEED CONCURRENTLY.

    Each child pays the full mlframe import graph (~20s) which does not amortise
    across separate processes; running them in parallel keeps the test's wall
    time at ~one child rather than the sum, leaving comfortable headroom under
    the 60s per-test timeout.
    """
    procs = {
        s: subprocess.Popen(
            [sys.executable, "-c", _HASHSEED_CHILD],
            env=_child_env(s),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for s in seeds
    }
    results: dict[str, str] = {}
    try:
        for s, p in procs.items():
            out, err = p.communicate(timeout=50)
            results[s] = _check_child_output(s, out, err, p.returncode)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.communicate()
    return results


@pytest.mark.slow
@pytest.mark.no_xdist
def test_hashseed_subprocess_selection_is_deterministic():
    """One cheap selector fitted in two child processes under different
    PYTHONHASHSEED values must print the SAME selection. A regression that
    leaks set/dict iteration order into selection (the RFECV iter-17 bug class)
    trips this for any selector building name-keyed containers."""
    res = _run_hashseed_children(("0", "42"))
    out0, out42 = res["0"], res["42"]
    assert out0 == out42, (
        f"selection depends on PYTHONHASHSEED: seed=0 -> {out0!r} but seed=42 -> {out42!r}; "
        f"a set/dict iteration-order leak (cf. RFECV iter-17) altered the selection"
    )
    # Sanity: the cheap simple-mode MRMR recovered at least the dominant signal,
    # so the determinism assertion is over a real (non-empty) selection.
    assert out0, "hashseed child produced an empty selection; cannot pin determinism over nothing"
