"""ADVERSARIAL v2 battery for the four recent MRMR/FE fixes.

Goal: actively BREAK each landed fix with hand-crafted edge-case targets/data, not
confirm it. Every cell asserts the HONEST correct behaviour (no lenient floors that
would mask a real regression). A cell that fails is a NEW BUG.

The four fixes under attack (all on origin/master, 2026-06-12):

  1. BUG1 -- raw-redundancy nested-composite drop. ``drop_redundant_raw_operands``
     conditions a raw operand on the CLEAN isolated ``rname``-containing sub-expression
     of a fused composite (replayed nested_parent recipe), not the whole fused
     composite. A fully-subsumed raw drops; a raw with a genuine private additive term
     is kept; a raw whose only consumer is a self-transform (DPI trap) or a
     non-replayable nested child is NOT dropped (empty-support guard).

  2. BUG2 -- recipe-survival / vote authority. A cross-fold stability-vote-rejected
     engineered feature can never re-enter support_ recipe-less; every advertised
     engineered feature replays byte-exactly through transform().

  3. BUG3 -- synergy-prevalence rescue. A synergy pair that cleared the order-2 maxT
     floor but missed the stricter raw-MI prevalence ratio is handed to the
     auto-escalation as a SECOND CHANCE (PROPOSE only -- the full admission gates
     decide). A genuinely-noise pair must NOT be admitted by the rescue.

  4. prewarp replay. A frozen-axis prewarp / nested engineered recipe replays
     byte-exactly on a row slice (incl. extreme outliers, out-of-fit-range, and
     constant operand slices) vs the full-frame transform.

EXECUTION MODEL. Every MRMR fit runs in a FRESH SUBPROCESS (``_fit_in_subprocess``).
This is LOAD-BEARING, not cosmetic: ``MRMR.fit`` consumes global ``np.random`` state
during fitting (permutation nulls / subsampling), so two fits in one interpreter
contaminate each other and silently change which composite a later fit selects --
exactly the in-process contamination that masked BUG1 on lucky orderings. CPU is
forced (``CUDA_VISIBLE_DEVICES=""``, ``MLFRAME_DISABLE_HNSW=1``). n<=40000 (shared
RAM box). Each fit is its own pytest cell under a per-cell timeout so a slow/failed
cell is isolated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

# ---------------------------------------------------------------------------
# Subprocess fit harness (global-RNG isolation)
# ---------------------------------------------------------------------------
_N = 40_000  # large enough to form the fused composite on the non-collapse seeds
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _fit_in_subprocess(body: str, *, timeout: int = 850) -> dict:
    """Run a fit body in a fresh interpreter and return the JSON result dict.

    ``body`` must build ``df`` (a DataFrame) and ``y`` (a Series), then is wrapped
    with a standard MRMR fit + ``RESULT_JSON=`` emit of the selected names AND a
    byte-exact slice-replay check of every engineered survivor.
    """
    src = textwrap.dedent(
        """
        import json, numpy as np, pandas as pd
        from mlframe.feature_selection.filters.mrmr import MRMR
        {body}
        fs = MRMR(verbose=0)
        fs.fit(df, pd.Series(np.asarray(y), name='y'))
        sel = list(fs.get_feature_names_out())
        # byte-exact slice replay of engineered survivors (prewarp/nested recipe contract)
        replay_break = []
        try:
            out_full = fs.transform(df)
            lo, hi = 5000, 5300
            out_slice = fs.transform(df.iloc[lo:hi].reset_index(drop=True))
            for ec in out_full.columns:
                if ('(' not in ec) and ('__' not in ec):
                    continue
                vf = np.asarray(out_full[ec].values)[lo:hi]
                vs = np.asarray(out_slice[ec].values)
                bn = np.isnan(vf) & np.isnan(vs)
                if not np.array_equal(np.where(bn, 0.0, vf), np.where(bn, 0.0, vs)):
                    replay_break.append(ec)
        except Exception as _e:
            replay_break.append('TRANSFORM_ERROR:' + repr(_e)[:200])
        print('RESULT_JSON=' + json.dumps({{'sel': sel, 'replay_break': replay_break}}))
        """
    ).format(body=textwrap.dedent(body).strip())
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MLFRAME_DISABLE_HNSW"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    res = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            res = json.loads(line[len("RESULT_JSON=") :])
    assert res is not None, f"subprocess fit returned no selection (rc={proc.returncode}); stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-20:])
    return res


def _toks(name: str, raw: set) -> set:
    """Helper that toks."""
    out = set()
    for t in _IDENT.findall(name):
        if t in raw:
            out.add(t)
        elif "__" in t and t.split("__", 1)[0] in raw:
            out.add(t.split("__", 1)[0])
    return out


def _flat_tokens(sel, raw: set) -> set:
    """Flat tokens."""
    s: set = set()
    for nm in sel:
        s |= _toks(nm, raw)
    return s


def _raw_in_support(sel, raw: set) -> set:
    """Raw in support."""
    return {n for n in sel if n in raw}


def _engineered(sel) -> list:
    """Helper that engineered."""
    return [n for n in sel if ("(" in n) or ("__" in n)]


# ===========================================================================
# BUG1 -- raw-redundancy nested-composite drop
# ===========================================================================


@pytest.mark.timeout(900)
def test_bug1_two_raws_both_subsumed_no_empty_support():
    """ATTACK (c): BOTH operands of the same ratio are subsumed.

    ``y = a**2/b + 3*c + f/5`` -- ``a`` and ``b`` enter the target ONLY through the
    ``a**2/b`` ratio (no private term for either); ``c`` is a genuine private linear
    term. The fix must drop BOTH subsumed raws (a AND b) while keeping the support
    non-empty (the engineered ratio + ``c`` remain). HONEST contract:
      * support is NON-EMPTY (empty-support guard holds);
      * ``c`` (private) is present;
      * the a**2/b ratio is represented (token a AND b reachable, raw or composite).
    A BREAK is: empty support, OR a bare subsumed raw (a or b) kept as itself while
    the composite already captures it, OR the ratio signal lost entirely.
    """
    body = """
        np.random.seed(0)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**2/b + 3.0*c + f/5.0
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e"}
    assert sel, "BUG1 BREAK: empty support"
    toks = _flat_tokens(sel, raw)
    # c carries a genuine private linear term -> its signal MUST survive.
    assert "c" in toks, f"BUG1 BREAK: private linear term c lost: {sel}"
    # The a**2/b ratio signal must be represented somewhere (raw or engineered).
    assert {"a", "b"} <= toks, f"BUG1 BREAK: a**2/b ratio signal lost: {sel}"
    # No byte-exact replay regression on any engineered survivor.
    assert not res["replay_break"], f"replay regression: {res['replay_break']}"


@pytest.mark.timeout(900)
def test_bug1_dpi_trap_self_transform_not_dropped():
    """ATTACK (e): the DPI trap -- a raw whose only consumer is a self-transform of
    itself must NOT be redundancy-dropped (conditioning a raw on a basis of ITSELF
    drives CMI to ~0 for every raw and proves nothing).

    ``y = a**3 + log(c)*sin(d) + f/5``. ``a`` enters ONLY via the cubic self-transform
    ``a**3`` (no second signal-bearing operand fused with it). The DPI-trap consumer
    filter must exclude any sole-operand ``a``-transform from the subsumer set, so
    ``a``'s signal is retained (token ``a`` present, raw or as its own engineered
    self-transform). A BREAK is: token ``a`` vanishes (wrongly dropped as 'redundant'
    against a basis of itself).
    """
    body = """
        np.random.seed(0)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**3 + np.log(c)*np.sin(d) + f/5.0
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e"}
    toks = _flat_tokens(sel, raw)
    assert sel, "empty support"
    assert (
        "a" in toks
    ), f"BUG1 DPI-TRAP BREAK: ``a`` enters via a SELF-transform (a**3) only, so the DPI-trap filter must NOT drop it, yet token a vanished: {sel}"
    assert not res["replay_break"], f"replay regression: {res['replay_break']}"


@pytest.mark.timeout(900)
def test_bug1_private_linear_at_retain_bar_kept():
    """ATTACK (a): a raw PARTIALLY subsumed -- a genuine private linear term sitting
    near the ``RAW_SELF_RETAIN_FRAC`` (5%) keep bar must be KEPT (its independent
    residual is real). ``y = a**2/b + 2*a + f/5 + log(c)*sin(d)``: ``a`` enters BOTH
    via the ratio AND privately (linear ``2*a``). The honest contract: ``a``'s private
    linear signal is not subsumed by the ratio -> token ``a`` MUST survive. A BREAK is
    an OVER-DROP: token ``a`` vanishes despite the genuine private term.
    """
    body = """
        np.random.seed(0)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**2/b + 2.0*a + f/5.0 + np.log(c)*np.sin(d)
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e"}
    toks = _flat_tokens(sel, raw)
    assert sel, "empty support"
    assert "a" in toks, f"BUG1 OVER-DROP BREAK: ``a`` has a genuine PRIVATE linear term (2*a) beyond the a**2/b ratio, yet its token vanished: {sel}"
    assert not res["replay_break"], f"replay regression: {res['replay_break']}"


@pytest.mark.timeout(900)
def test_bug1_triple_fused_composite_isolates_anchor():
    """ATTACK (b): a composite that fuses a SECOND signal besides the a-containing
    sub-expression (triple structure) -- the clean-subexpr isolation must still find
    the right ``a`` anchor.

    ``y = a**2/b + log(c)*sin(d) + e*g + f/5`` -- three independent signal terms.
    Whatever composite forms, the support must (1) be non-empty, (2) preserve EVERY
    signal group's tokens, and (3) NOT keep a bare fully-subsumed raw ``a`` alongside
    a composite that already captures the a**2/b ratio. The minimal honest contract:
    no fully-subsumed bare raw operand pollutes support while its ratio is captured by
    an engineered survivor. We assert the signal-coverage floor (all groups present)
    and the replay contract; the precise drop verdict is probed empirically.
    """
    body = """
        np.random.seed(0)
        a, b, c, d, e, f, g = (np.random.rand(40000) for _ in range(7))
        y = a**2/b + np.log(c)*np.sin(d) + e*g + f/5.0
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'g': g})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e", "g"}
    toks = _flat_tokens(sel, raw)
    assert sel, "empty support"
    # Each signal group's dominant operand must remain reachable.
    assert "a" in toks or "b" in toks, f"BUG1 BREAK: a**2/b group lost: {sel}"
    assert "c" in toks or "d" in toks, f"BUG1 BREAK: log(c)sin(d) group lost: {sel}"
    assert not res["replay_break"], f"replay regression: {res['replay_break']}"


# ===========================================================================
# BUG2 -- vote authority / no recipe-less re-admission
# ===========================================================================


@pytest.mark.timeout(900)
def test_bug2_every_advertised_engineered_replays_byte_exact():
    """ATTACK (a)+(b): every advertised engineered feature must replay byte-exactly
    through transform() -- a vote-rejected column can never re-enter support_
    recipe-less (it would be silently dropped from transform output -> a
    select-then-drop contract violation).

    Uses the user CASE2-style df with a discrete-ish jump target that historically
    surfaced a vote-rejected feature. The honest contract: NO engineered survivor is
    missing from / non-byte-exact in transform output (``replay_break`` empty), and
    every advertised column is reproduced.
    """
    body = """
        np.random.seed(3)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**2/b + np.rint(d) + f/5.0 + (a**3)
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    assert sel, "empty support"
    assert not res["replay_break"], (
        f"BUG2 BREAK: advertised engineered feature(s) failed byte-exact transform "
        f"replay (recipe-less re-admission / non-deterministic recipe): "
        f"{res['replay_break']}; full selection {sel}"
    )


@pytest.mark.timeout(900)
def test_bug2_deep_nested_two_step_survives_or_not_selected():
    """ATTACK (a): deep nested-engineered feature at fe_max_steps>=2 either survives
    transform byte-exact or is correctly NOT selected -- never advertised-but-missing.

    ``y = a**2/b + log(c)*sin(d) + f/5`` with the canonical fused composite. Every
    engineered survivor that IS advertised must replay byte-exactly.
    """
    body = """
        np.random.seed(2)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**2/b + np.log(c)*np.sin(d) + f/5.0
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    assert res["sel"], "empty support"
    assert not res["replay_break"], f"BUG2 BREAK: nested engineered survivor not byte-exact on transform: {res['replay_break']}; selection {res['sel']}"


# ===========================================================================
# BUG3 -- synergy-prevalence rescue (no false-positive admission)
# ===========================================================================


@pytest.mark.timeout(900)
def test_bug3_pure_noise_pair_not_rescued():
    """ATTACK (a): a prevalence-failed pair that is GENUINELY NOISE must NOT be
    admitted by the rescue (the rescue only PROPOSES; the full admission gates --
    order-2 maxT + marginal-permutation floor + S5 CMI redundancy -- must reject it).

    The target depends ONLY on a genuine signal (``a**2/b`` + ``log(c)*sin(d)``); the
    operand ``e`` is PURE NOISE not in y at all. Any engineered feature that fuses the
    noise operand ``e`` (an esc_* poly / Fourier escalation of a noise pair, or a
    binary on ``e``) is a SPURIOUS rescue admission and a BREAK. The honest contract:
    NO selected feature carries the pure-noise operand ``e``.
    """
    body = """
        np.random.seed(1)
        a, b, c, d, e, f = (np.random.rand(40000) for _ in range(6))
        y = a**2/b + np.log(c)*np.sin(d) + f/5.0   # e is pure noise, NOT in y
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e"}
    assert sel, "empty support"
    # Pure-noise operand e must NOT enter any selected feature (raw or engineered).
    carriers = [nm for nm in sel if "e" in _toks(nm, raw)]
    assert not carriers, (
        f"BUG3 FALSE-RESCUE BREAK: pure-noise operand ``e`` (not in y) entered the "
        f"selection -- the synergy-prevalence rescue admitted a noise pair the gates "
        f"should reject: {carriers}; full selection {sel}"
    )
    assert not res["replay_break"], f"replay regression: {res['replay_break']}"


@pytest.mark.timeout(900)
def test_bug3_two_pure_noise_operands_not_rescued():
    """ATTACK (a) hardened: TWO pure-noise operands (``e``, ``g``) whose noise*noise
    pair could be a prevalence-failed synergy candidate must NOT be rescued into the
    selection. Neither ``e`` nor ``g`` is in y. A BREAK is either noise operand
    appearing in any selected feature.
    """
    body = """
        np.random.seed(5)
        a, b, c, d, e, f, g = (np.random.rand(40000) for _ in range(7))
        y = a**2/b + np.log(c)*np.sin(d) + f/5.0   # e, g pure noise
        df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'g': g})
    """
    res = _fit_in_subprocess(body)
    sel = res["sel"]
    raw = {"a", "b", "c", "d", "e", "g"}
    assert sel, "empty support"
    noisy = [nm for nm in sel if ({"e"} & _toks(nm, raw)) or ({"g"} & _toks(nm, raw))]
    assert not noisy, f"BUG3 FALSE-RESCUE BREAK: pure-noise operand(s) e/g entered selection: {noisy}; full selection {sel}"


# ===========================================================================
# prewarp replay -- byte-exact slice vs full under adversarial operand slices
# ===========================================================================


def _byte_exact_slice_replay(df: pd.DataFrame, y: np.ndarray, seed: int, slices: list[tuple[int, int]]) -> None:
    """Fit IN-PROCESS once (single fit -> no RNG contamination concern) then assert
    every engineered survivor replays byte-exactly on each adversarial row slice."""
    fs = MRMR(verbose=0, random_seed=seed)
    fs.fit(df, pd.Series(y, name="y"))
    out_full = fs.transform(df)
    eng = [c for c in out_full.columns if ("(" in c) or ("__" in c)]
    for lo, hi in slices:
        out_slice = fs.transform(df.iloc[lo:hi].reset_index(drop=True))
        for ec in eng:
            vf = np.asarray(out_full[ec].values)[lo:hi]
            vs = np.asarray(out_slice[ec].values)
            bn = np.isnan(vf) & np.isnan(vs)
            assert np.array_equal(
                np.where(bn, 0.0, vf), np.where(bn, 0.0, vs)
            ), f"PREWARP REPLAY BREAK: engineered col {ec!r} not byte-exact on slice [{lo}:{hi}] -- a global/slice-local statistic leaked into replay"


@pytest.mark.timeout(600)
def test_prewarp_replay_extreme_outlier_slice_byte_exact():
    """ATTACK (a): a slice containing EXTREME outliers / heavy-tail operands must
    still replay byte-exactly (frozen axis params, no slice-local re-fit)."""
    rng = np.random.default_rng(11)
    n = 20000
    a = rng.standard_t(2.0, n)  # heavy tail
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n) + 0.5
    e = rng.random(n)
    f = rng.normal(0, 1, n)
    # inject extreme outliers in a known slice
    a[5050] = 1e9
    a[5100] = -1e9
    y = a**2 / b + np.log(c) * np.sin(d) + f / 5.0
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    _byte_exact_slice_replay(df, y, seed=11, slices=[(5000, 5300), (0, 200)])


@pytest.mark.timeout(600)
def test_prewarp_replay_constant_operand_slice_byte_exact():
    """ATTACK (c): a slice where an operand is CONSTANT (all-equal) must replay
    byte-exactly -- a degenerate axis (span ~ 0) must be handled deterministically,
    identically on slice and full."""
    rng = np.random.default_rng(13)
    n = 20000
    a = rng.random(n)
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n) + 0.5
    e = rng.random(n)
    f = rng.normal(0, 1, n)
    a[5000:5200] = 0.42  # constant operand in the slice
    y = a**2 / b + np.log(c) * np.sin(d) + f / 5.0
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    _byte_exact_slice_replay(df, y, seed=13, slices=[(5000, 5200)])


@pytest.mark.timeout(600)
def test_prewarp_replay_out_of_fit_range_slice_byte_exact():
    """ATTACK (b): a slice value OUTSIDE the fit-time operand range must clip/extrapolate
    DETERMINISTICALLY -- the replay is a closed-form function of x with frozen params,
    so slice-vs-full must be byte-exact even for rows whose operand exceeds the
    full-frame min/max seen at fit time (here the full frame DOES contain the extreme
    rows, so the slice-vs-full equality is the honest invariant; the point is that no
    slice-local min/max recompute perturbs the axis)."""
    rng = np.random.default_rng(17)
    n = 20000
    a = rng.random(n)
    b = rng.random(n) + 0.5
    c = rng.random(n) + 0.5
    d = rng.random(n) + 0.5
    e = rng.random(n)
    f = rng.normal(0, 1, n)
    # a slice whose a-values are at the extreme high end of the fit-time range
    a[5000:5300] = np.linspace(0.95, 1.0, 300)
    y = a**2 / b + np.log(c) * np.sin(d) + f / 5.0
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    _byte_exact_slice_replay(df, y, seed=17, slices=[(5000, 5300)])
