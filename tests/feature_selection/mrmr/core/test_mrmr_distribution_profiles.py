"""Create / keep / drop verification of MRMR under REALISTIC distribution profiles.

The sibling ``test_mrmr_create_keep_drop.py`` suite drew every operand from a clean
uniform marginal. Real features are skewed / heavy-tailed / outlier-ridden. MI is
monotone-invariant, so the GENUINE signal each formula encodes is the SAME under
any marginal shape -- which makes varied marginals a robustness probe: if the
selector recovers ``a**2/b`` (or ``log(c)*sin(d)``) under ``uniform`` but LOSES it
under ``heavy_tailed`` / ``with_outliers``, that is a robustness regression to flag.

This module reuses the existing suite's TOLERANT operand-token matcher (a kept
signal is satisfied by ANY selected column whose operand-token set covers the
required raw operands -- including warped-basis surrogates like ``a__He2`` and
engineered recipes like ``div(sqr(a),neg(b))``) and drives a REPRESENTATIVE set of
formulas through the five distribution PROFILES at n<=30000.

Operand domains are DECLARATIVE: each formula tags its operands ``any`` /
``positive`` (a log/sqrt argument) / ``divisor`` (a denominator). The distribution
sampler honours those tags under every profile (see ``_synthetic_distributions``),
so every formula stays well defined (no log-of-negative, no div-by-~0) regardless
of marginal shape -- the data infra guarantees it, not per-formula hacks.

VERDICT REGISTRY (``ROBUSTNESS``): each (formula, profile, n) cell whose genuine
signal is NOT recovered carries a verified disposition:
  * ``shared_uniform`` -- the SAME non-pass already documented for the uniform
    baseline in the sibling suite (e.g. the class-3 divisor residual / raw-operand
    retention); NOT introduced by the distribution, so not a robustness bug.
  * ``robustness`` -- recovered under uniform but LOST under this profile: a
    genuine distribution-robustness gap, documented with the observed selection.
Every cell is recorded in a JSON ledger for the report. Seeded + deterministic;
no n>30000 (RAM-contended box).

Run with MLFRAME_DISABLE_HNSW=1 (harness-set); each fit is seeded.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

import mlframe
from mlframe.feature_selection.filters.mrmr import MRMR
from tests.conftest import perf_time_budget
from tests.feature_selection import _synthetic_distributions as sd

# Reuse the battle-tested tolerant matcher from the uniform suite (no duplication).
from tests.feature_selection.mrmr.core.test_mrmr_create_keep_drop import _artifact_path, _covers, _operand_tokens

SEED = 42
# perf_time_budget (2026-08-18): a raw 360s cap isn't enough once the shared 2-vCPU CI runners are
# themselves contended (observed: a Timeout(>360.0s) failure on a run where the whole ~57-job matrix
# ran several times its normal wall-clock). Widen under detected xdist/host contention via the same
# mechanism already used for other wall-clock-sensitive assertions in this suite. Evaluated once at
# collection time (module import), which is when @pytest.mark.timeout consumes it.
FIT_TIMEOUT = int(perf_time_budget(360))
_PROGRESS = _artifact_path("distros_progress.txt")
_LEDGER = []


def _checkpoint(msg: str) -> None:
    """Helper that checkpoint."""
    try:
        with open(_PROGRESS, "a", encoding="utf-8") as fh:
            fh.write(msg.rstrip("\n") + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Representative formula registry with DECLARATIVE operand domains.
# ---------------------------------------------------------------------------
# Each spec:
#   domains : {operand: domain}  (drives the distribution sampler)
#   target  : (data_dict) -> y   (uses ONLY the operands in ``domains``)
#   keep    : list of signals; each signal is a list of operand-sets, satisfied if
#             ANY set is covered by some selected column (tolerant any-of).
#   drop    : set of pure-noise operands that must be absent.
# Formulas span the difficulty axes the uniform suite probes, but are sampled under
# every profile here.

FORMULAS = {}


def _reg(name, domains, target, keep, drop, family):
    """Helper that reg."""
    FORMULAS[name] = dict(domains=domains, target=target, keep=keep, drop=drop, family=family)


# --- single weak ratio (divisor operand) -----------------------------------
_reg(
    "ratio_sqr",
    {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "e": sd.DOMAIN_ANY, "zn": sd.DOMAIN_ANY},
    lambda d: 0.30 * d["a"] ** 2 / d["b"] + 0.01 * d["e"],
    # The (a,b) ratio. We require BOTH operands (no lone-a fallback): on this single
    # weak-ratio formula a lone ``a`` next to a noise feature is NOT recovery of the
    # a**2/b interaction -- demanding {a,b} lets the mixed-marginal collapse (where
    # only a noise feature survives) surface as the genuine signal-loss it is.
    keep=[[{"a", "b"}]],
    drop={"zn"},
    family="weak-scaled",
)

# --- log*sin product (positive operand) ------------------------------------
_reg(
    "log_sin_product",
    {"c": sd.DOMAIN_POSITIVE, "d": sd.DOMAIN_ANY, "e1": sd.DOMAIN_ANY, "e2": sd.DOMAIN_ANY, "zn": sd.DOMAIN_ANY},
    lambda d: 0.40 * np.log(d["c"]) * np.sin(d["d"]) + 0.02 * d["e1"] + 0.02 * d["e2"],
    keep=[[{"c", "d"}]],
    drop={"zn"},
    family="weak-scaled",
)

# --- two genuine pairs, strong (the F2-strong shape) -----------------------
_reg(
    "two_pairs_strong",
    {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE, "d": sd.DOMAIN_ANY, "e": sd.DOMAIN_ANY},
    lambda d: d["a"] ** 2 / d["b"] + 3.0 * np.log(d["c"]) * np.sin(d["d"]) + 0.3 * d["e"],
    # both terms must be recovered; the (a,b) term may surface as the divisor
    # residual (a-image) or the joint ratio -- tolerant any-of.
    keep=[[{"a", "b"}, {"a"}], [{"c", "d"}]],
    drop=set(),  # e carries 0.3 weight; not a pure-zero noise here -> not asserted-drop
    family="two-pairs",
)

# --- product square with competing decoys (redundancy gate) ----------------
_reg(
    "product_square_decoys",
    {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_ANY, "dd": sd.DOMAIN_ANY, "e": sd.DOMAIN_ANY},
    lambda d: (d["a"] ** 2) * d["b"] + 0.20 * d["dd"],
    keep=[[{"a", "b"}, {"a"}, {"b"}], [{"dd"}]],
    drop={"e"},
    family="competing-correlated",
)

# --- additive two-term, both positive operands -----------------------------
_reg(
    "additive_two_term",
    {"a": sd.DOMAIN_POSITIVE, "c": sd.DOMAIN_POSITIVE, "e": sd.DOMAIN_ANY, "zn": sd.DOMAIN_ANY},
    lambda d: np.log(d["a"] + 1.0) + np.sqrt(d["c"]) + 0.02 * d["e"],
    keep=[[{"a"}], [{"c"}]],
    drop={"zn"},
    family="additive",
)


# ---------------------------------------------------------------------------
# SIGNAL-LOSS verdict registry (the hard robustness gate).
# ---------------------------------------------------------------------------
# A cell listed here is a documented distribution-robustness gap: a genuine signal (a keep) that is lost under that profile, with the measured reason. Cells absent from the registry must recover
# every genuine signal, and a new loss fails loudly. The registry is empty: the residual rescue (``fe_residual_rescue_enable``) judges the pool on the residual of the selected design in the
# target's own scale and recovers the weak-in-rank-space terms (a**2/b under outliers, mixed marginals and pareto tails; the weak linear 0.20*dd side term) that the rank-MI greedy dropped.
SIGNAL_LOSS = {}

# NOISE-ADMISSION registry: cells where a PURE-noise column (``zn``, or ``e`` in product_square_decoys; neither enters the target) is admitted
# next to the fully recovered signal. It is empty on purpose: the weak operands e / e1 / e2 of the other formulas are real terms of the
# noise-free target (coefficient 0.01-0.02), so admitting them is correct and they are not asserted-drop. Any admission of a pure-noise
# column fails loudly as a leak, and a documented one must carry its measured reason here.
NOISE_ADMISSION = {}


def _signal_loss_reason(formula, profile, n):
    """Signal loss reason."""
    return SIGNAL_LOSS.get((formula, profile, n))


def _noise_admission_reason(formula, profile, n):
    """Noise admission reason."""
    return NOISE_ADMISSION.get((formula, profile, n))


# ---------------------------------------------------------------------------
# Fit + evaluate
# ---------------------------------------------------------------------------


def _union_tokens(selected, df_cols):
    """Union of raw-operand tokens across ALL selected columns.

    A genuine multi-operand signal ``{c, d}`` is RECOVERED whenever both operands
    are present in the selection -- whether as one joint engineered column
    ``mul(log(c),sin(d))`` (operand-set {c,d}) OR as two separate columns (raw
    ``c`` and raw ``d``, each contributing its own token). Both give the downstream
    model access to the full interaction support, so crediting the union is the
    honest, distribution-invariant recovery contract (the same one the sibling
    suite uses for the ``MS_sin_phase_weak`` two-single-operand case)."""
    toks = set()
    for nm in selected:
        toks |= _operand_tokens(nm, df_cols)
    return toks


def _eval_keep(selected, df_cols, keep):
    """Return list of (label, covered) and the missing-signal failures.

    A signal (list of operand-sets, any-of) is covered iff EITHER some single
    selected column's operand tokens already cover one of the sets (a joint
    feature) OR the union of all selected columns' tokens covers it (both operands
    present individually). The union path credits genuinely-recovered signals that
    the library chose to represent as separate raw/engineered columns."""
    union = _union_tokens(selected, df_cols)
    results, failures = [], []
    for sig in keep:
        ok = any(_covers(selected, df_cols, want) or set(want) <= union for want in sig)
        label = " | ".join("+".join(sorted(w)) for w in sig)
        results.append((label, ok))
        if not ok:
            failures.append(f"missing keep[{label}]")
    return results, failures


def _eval_drop(selected, drop):
    """Eval drop."""
    results, failures = [], []
    for col in sorted(drop):
        admitted = col in selected
        results.append((col, admitted))
        if admitted:
            failures.append(f"admitted noise[{col}]")
    return results, failures


def _fit_profile(formula, profile, n):
    """Fit profile."""
    spec = FORMULAS[formula]
    data = sd.sample_operands(seed=SEED, n=n, domains=spec["domains"], profile=profile)
    y = spec["target"](data)
    assert np.all(np.isfinite(y)), f"{formula}/{profile}: target not finite (domain bug)"
    df = pd.DataFrame({k: data[k] for k in spec["domains"]})
    fs = MRMR(verbose=0, random_state=SEED)
    fs.fit(df, pd.Series(y, name="y"))
    selected = list(fs.get_feature_names_out())
    df_cols = set(df.columns)
    keep_res, keep_fail = _eval_keep(selected, df_cols, spec["keep"])
    drop_res, drop_fail = _eval_drop(selected, spec["drop"])
    _LEDGER.append(
        dict(
            formula=formula,
            profile=profile,
            n=int(n),
            selected=selected,
            keep=[{"want": l, "covered": c} for l, c in keep_res],
            drop=[{"col": c, "admitted": a} for c, a in drop_res],
            signal_lost=keep_fail,
            noise_admitted=drop_fail,
        )
    )
    return selected, keep_fail, drop_fail


# ---------------------------------------------------------------------------
# Parametrization: every formula x every profile at n=20000 (RAM-tight, <=30k).
# ---------------------------------------------------------------------------
_PROFILES = list(sd.available_profiles())
_BROAD_N = 20000


def _assert_gate(formula, profile, n, selected, keep_fail, drop_fail):
    """The shared verdict gate.

    HARD gate (the robustness question): a missing keep == a GENUINE signal lost.
    It must either be a documented SIGNAL_LOSS cell (xfail with the verified
    distribution-robustness reason) or fail loudly as a regression.

    SOFT residual: a noise admission on a tiny-weight operand is the documented
    default-preset marginal-uplift residual (also fires under the uniform baseline);
    it is recorded but, when documented in NOISE_ADMISSION, does NOT fail the gate.
    An UNDOCUMENTED noise admission still fails -- so a NEW noise leak is caught.
    """
    if keep_fail:
        reason = _signal_loss_reason(formula, profile, n)
        if reason is not None:
            pytest.xfail(f"{formula}/{profile} n={n}: SIGNAL-LOSS (documented robustness gap): {reason}; observed={keep_fail}; selected={selected}")
        pytest.fail(
            f"{formula}/{profile} n={n}: GENUINE SIGNAL LOST: {'; '.join(keep_fail)}  selected={selected}",
            pytrace=False,
        )
    if drop_fail:
        reason = _noise_admission_reason(formula, profile, n)
        if reason is not None:
            pytest.xfail(f"{formula}/{profile} n={n}: noise-admission residual (signal recovered): {reason}; observed={drop_fail}; selected={selected}")
        pytest.fail(
            f"{formula}/{profile} n={n}: NOISE ADMITTED (signal recovered): {'; '.join(drop_fail)}  selected={selected}",
            pytrace=False,
        )


@pytest.mark.timeout(FIT_TIMEOUT)
@pytest.mark.parametrize("formula", sorted(FORMULAS.keys()))
@pytest.mark.parametrize("profile", _PROFILES)
def test_create_keep_drop_under_profile(profile, formula):
    """Create keep drop under profile."""
    _checkpoint(f"PROFILE start {formula}/{profile} n={_BROAD_N}")
    selected, keep_fail, drop_fail = _fit_profile(formula, profile, _BROAD_N)
    _checkpoint(f"PROFILE done  {formula}/{profile} n={_BROAD_N} sel={selected} signal_lost={keep_fail} noise={drop_fail}")
    _assert_gate(formula, profile, _BROAD_N, selected, keep_fail, drop_fail)


# A small n-sweep for the divisor-ratio formula -- the one whose recoverability the
# probe showed to be profile-fragile -- to confirm the fragility is not just a
# single-n artifact. n in {10000, 30000}, both <= 30000.
_NSWEEP_NS = [10000, 30000]


@pytest.mark.timeout(FIT_TIMEOUT)
@pytest.mark.parametrize("profile", ["uniform", "heavy_tailed", "with_outliers"])
@pytest.mark.parametrize("n", _NSWEEP_NS)
def test_ratio_sqr_nsweep_under_profile(profile, n):
    """Ratio sqr nsweep under profile."""
    _checkpoint(f"NSWEEP start ratio_sqr/{profile} n={n}")
    selected, keep_fail, drop_fail = _fit_profile("ratio_sqr", profile, n)
    _checkpoint(f"NSWEEP done  ratio_sqr/{profile} n={n} sel={selected} signal_lost={keep_fail} noise={drop_fail}")
    _assert_gate("ratio_sqr", profile, n, selected, keep_fail, drop_fail)


# ---------------------------------------------------------------------------
# Fresh-process RECOVERY sensor for the documented xfail registries.
# ---------------------------------------------------------------------------
# The in-suite gate uses IMPERATIVE ``pytest.xfail`` (not strict-xfail) on the
# SIGNAL_LOSS / NOISE_ADMISSION cells of BOTH this file and the sibling
# ``test_mrmr_create_keep_drop.py``: those cells are PROCESS-STATE-DEPENDENT --
# they recover the genuine signal in a fresh process but lose it after the
# accumulated in-suite global-RNG / dispatcher / JIT state shifts a weak
# divisor ratio just under the screen floor (see the registry reasons, e.g. the
# ``ratio_sqr / with_outliers / 30000`` "IN-SUITE BOUNDARY flip" note). Strict
# xfail would therefore FLAKE on XPASS whenever a cell happens to recover.
#
# But a runtime xfail can never XPASS, so a documented gap that gets genuinely
# FIXED would silently rot (the cell would recover yet the xfail keeps passing).
# This sensor closes that hole WITHOUT making the registries strict: it re-runs a
# few representative documented cells -- one per SUBPROCESS so each starts from
# clean process state -- and asserts the genuine keep-signal RECOVERS in
# isolation. A real regression (a cell that fails to recover EVEN when fresh) is
# a true signal loss and trips this loudly; an in-suite-only loss stays a
# tolerated, documented process-state artifact. Kept to 4 small cells (2 from
# each suite's registry families) so it stays fast.

# (kind, formula, profile_or_None, n, label). ``profile=None`` => sibling
# create_keep_drop cell. Each is a documented in-suite SIGNAL_LOSS / noise /
# protect cell that is KNOWN to recover the genuine keep-signal in isolation.
_RECOVERY_SENSOR_CELLS = [
    ("distro", "ratio_sqr", "uniform", 10000, "noise-admission residual; (a,b) recovers fresh"),
    ("distro", "ratio_sqr", "with_outliers", 30000, "SIGNAL_LOSS in-suite; (a,b) recovers fresh (the documented boundary-flip cell)"),
    ("distro", "two_pairs_strong", "mixed", 20000, "SIGNAL_LOSS in-suite; both (a,b)+(c,d) recover fresh"),
    ("ckd", "ws1_easy_ratio_sqr", None, 1000, "create_keep_drop small-n protect cell; (a,b) recovers fresh"),
]


def _recovery_subprocess_src(kind: str, formula: str, profile, n: int) -> str:
    """Source for a one-cell fit run in a CLEAN child process. Exits 0 when the
    genuine keep-signal is recovered, 3 when it is lost (a real regression)."""
    if kind == "distro":
        body = (
            "from tests.feature_selection.mrmr.core import test_mrmr_distribution_profiles as M\n"
            f"sel, keep_fail, drop_fail = M._fit_profile({formula!r}, {profile!r}, {int(n)})\n"
        )
    else:
        body = (
            "from tests.feature_selection.mrmr.core import test_mrmr_create_keep_drop as M\n"
            f"spec, sel, failures = M._fit_and_eval({formula!r}, {int(n)}, "
            f"fe_max_steps=M.FORMULAS[{formula!r}].get('fe_max_steps', 1))\n"
            "keep_fail = [f for f in failures if f.get('kind') == 'missing_signal']\n"
        )
    return (
        "import os, sys, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "os.environ.setdefault('MLFRAME_DISABLE_HNSW', '1')\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')\n" + body + "print('KEEP_FAIL=' + repr(keep_fail))\n"
        "print('SELECTED=' + repr(sel))\n"
        "sys.exit(0 if not keep_fail else 3)\n"
    )


@pytest.mark.timeout(FIT_TIMEOUT)
@pytest.mark.parametrize(
    "kind,formula,profile,n,label",
    _RECOVERY_SENSOR_CELLS,
    ids=[f"{c[0]}:{c[1]}:{c[2]}:{c[3]}" for c in _RECOVERY_SENSOR_CELLS],
)
def test_documented_xfail_cell_recovers_in_fresh_process(kind, formula, profile, n, label):
    """A documented in-suite xfail cell MUST recover its genuine keep-signal
    when re-run in a clean child process. This is what makes the imperative
    (non-strict) xfail registries non-rotting: an in-suite-only loss is a
    tolerated process-state artifact, but a loss that persists even fresh is a
    real regression and fails here. See ``label`` for the per-cell datum."""
    import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)

    src = _recovery_subprocess_src(kind, formula, profile, n)
    env = dict(os.environ)
    env.setdefault("MLFRAME_DISABLE_HNSW", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    # This machine's global editable-install .pth can be stale; propagate the src/ dir this
    # process actually imported mlframe from so the subprocess resolves the same package.
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(pathlib.Path(mlframe.__file__).resolve().parents[1]), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        env=env,
        timeout=FIT_TIMEOUT - 5,
    )
    if proc.returncode == 3:
        pytest.fail(
            f"REGRESSION: documented cell {kind}:{formula}:{profile}:{n} ({label}) "
            f"failed to recover the genuine keep-signal even in a FRESH process -- "
            f"this is a real signal loss, not a process-state artifact.\n"
            f"stdout:\n{proc.stdout}\nstderr tail:\n{proc.stderr[-2000:]}",
            pytrace=False,
        )
    assert (
        proc.returncode == 0
    ), f"recovery subprocess for {kind}:{formula}:{profile}:{n} errored (rc={proc.returncode}); stdout:\n{proc.stdout}\nstderr tail:\n{proc.stderr[-2000:]}"


@pytest.fixture(scope="session", autouse=True)
def _dump_profile_ledger():
    """Dump profile ledger."""
    yield
    ledger_path = _artifact_path("distros_profile_ledger.json")
    try:
        import orjson

        with open(ledger_path, "wb") as fh:
            fh.write(orjson.dumps(_LEDGER, option=orjson.OPT_INDENT_2))
    except Exception as exc:
        try:
            import json
            import warnings

            warnings.warn(f"distros ledger orjson dump failed ({exc!r}); using json fallback", stacklevel=2)
            with open(ledger_path, "w", encoding="utf-8") as fh:
                json.dump(_LEDGER, fh, indent=2)
        except Exception as exc2:
            import warnings

            warnings.warn(f"distros ledger dump failed entirely: {exc2!r}", stacklevel=2)
