"""Hard cases for the four-pillar bench, with the contract each one is judged by.

The pillars (F2 across distributions, pure noise, 3-way XOR among noise, conjunction with proxies) each had a single representative fit on one seed. The cases here stress what those
leave out: monotone and exact decoys, leverage, offsets, duplicates, discrete and string columns, NaN patterns, label noise, rare conjunctions, mixed operand types, and paired null
controls. Every generator is deterministic given its seed. ``CASES`` lists them with their kind (``f2``, ``null`` or ``cover``), their contract arguments and whether the verdict is a
gate or only reported; ``four_pillars_contracts`` turns a seed panel into a Clopper-Pearson verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtr


def _sig(a, b, c, d):
    """The F2 compound ``a**2 / b + log(c) * sin(d)``."""
    return a**2 / b + np.log(np.abs(c) + 1e-9) * np.sin(d)


def _base(r, n):
    """Clean F2 operands: ``a, b ~ U(.5, 2)``, ``c ~ U(.5, 4)``, ``d ~ U(0, 2 pi)``, ``e`` noise."""
    return {"a": r.uniform(0.5, 2, n), "b": r.uniform(0.5, 2, n), "c": r.uniform(0.5, 4, n), "d": r.uniform(0, 2 * np.pi, n), "e": r.normal(size=n)}


def f2_monotone_decoys(seed, n=3000):
    """F2 with monotone transforms and a rank copy of the signal columns as decoys; name tokens cannot tell an operand from its transform."""
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["q0"] = np.exp(o["a"]); X["q1"] = np.log(o["b"]); X["q2"] = o["c"] ** 3
    X["q3"] = np.sin(o["d"]) + 1e-3 * r.normal(size=n); X["q4"] = pd.Series(o["a"]).rank().to_numpy()
    return X, y, truth


def f2_leverage_tiny_b(seed, n=3000):
    """F2 with a lognormal divisor reaching 1e-2 and Student-t(1.5) target noise: leverage in both the feature and the target."""
    r = np.random.default_rng(seed)
    a = r.normal(1.0, 0.3, n); b = r.lognormal(-1.0, 1.0, n)
    c = r.uniform(.5, 5, n); d = r.uniform(0, 2 * np.pi, n); e = r.standard_t(2, n)
    truth = _sig(a, b, c, d); y = truth + 0.5 * r.standard_t(1.5, n)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth


def f2_nan_mnar(seed, n=3000):
    """F2 with values missing not at random in the signal columns."""
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X.loc[r.random(n) < 0.05, "c"] = np.nan
    X.loc[r.random(n) < 0.3 / (1 + np.exp(-2 * (y - np.median(y)))), "a"] = np.nan
    X.loc[o["e"] > 1.5, "b"] = np.nan
    return X, y, truth


def f2_scale_and_range(seed, n=3000):
    """F2 with operands at very different offsets and scales (a tiny range, a huge offset)."""
    r = np.random.default_rng(seed); o = _base(r, n)
    a, b, c = 1e3 * o["a"], 1e6 * o["b"], 1e-4 * o["c"]
    d = o["d"].astype(np.float32).astype(np.float64)
    truth = _sig(a, b, c, d); y = truth + 0.2 * r.normal(size=n)
    e = 1e9 + 1e-3 * o["e"]
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth


def f2_dupes_collinear(seed, n=3000):
    """F2 with exact and near duplicates, rescaled copies and a half-signal proxy."""
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["a_dup"] = o["a"].copy(); X["a_jit"] = o["a"] + 1e-6 * r.normal(size=n)
    X["c_x1000"] = 1000 * o["c"]; X["d_deg"] = np.degrees(o["d"])
    X["ab_ratio"] = o["a"] / o["b"]; X["e_dup"] = o["e"].copy()
    return X, y, truth


def f2_wide_small_n(seed, n=600, p=150):
    """F2 among 150 noise columns at small n: a power case, reported rather than gated."""
    r = np.random.default_rng(seed); o = _base(r, n); o.pop("e")
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    Z = r.normal(size=(n, 5)) @ r.normal(size=(5, p)) * 0.7 + r.normal(size=(n, p))
    X = pd.concat([pd.DataFrame(o), pd.DataFrame(Z, columns=[f"n{j}" for j in range(p)])], axis=1)
    return X[r.permutation(X.columns)], y, truth


def f2_discrete_mixed(seed, n=3000):
    """F2 whose operands are quantised or few-level, carrying less information."""
    r = np.random.default_rng(seed)
    a = r.integers(1, 9, n).astype(float); b = r.gamma(3.0, 0.5, n) + 0.1
    c = r.integers(1, 7, n).astype(float); d = np.round(r.uniform(0, 360, n)) * np.pi / 180
    e = r.integers(0, 5, n).astype(float)
    truth = _sig(a, b, c, d); y = truth + 0.3 * r.normal(size=n)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e)), y, truth


def f2_class_targets(seed, n=3000, k=2, flip=0.10, rare=False):
    """F2 compound as a binary, multiclass or rare-class target with label flips."""
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); lat = truth + r.logistic(size=n)
    qs = [0.95] if rare else list(np.linspace(0, 1, k + 1)[1:-1] if k > 2 else [0.5])
    y = np.searchsorted(np.quantile(lat, qs), lat)
    m = r.random(n) < flip; y[m] = r.integers(0, k, m.sum())
    return pd.DataFrame(o), y.astype(np.int64), truth


def _ar1(r, n, rho):
    """AR(1) series of length n with coefficient phi."""
    x = np.empty(n); x[0] = r.normal(); eps = r.normal(size=n) * np.sqrt(1 - rho ** 2)
    for i in range(1, n): x[i] = rho * x[i - 1] + eps[i]
    return x

def f2_time_autocorr_walk(seed, n=3000):
    """F2 where the operands are autocorrelated random walks and the evaluation is chronological."""
    r = np.random.default_rng(seed)

    def u():
        """A fresh uniform-marginal AR(1) path."""
        return ndtr(_ar1(r, n, 0.98))

    a, b, c, d = .5 + 1.5 * u(), .5 + 1.5 * u(), .5 + 3.5 * u(), 2 * np.pi * u()
    truth = _sig(a, b, c, d); y = truth + 0.3 * _ar1(r, n, 0.9)
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=np.cumsum(r.normal(size=n)))), y, truth


def f2_id_columns(seed, n=3000):
    """F2 with row-id, user-id, group and constant columns among the distractors."""
    r = np.random.default_rng(seed); o = _base(r, n)
    truth = _sig(o["a"], o["b"], o["c"], o["d"]); y = truth + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["rid"] = np.arange(n); X["uid"] = r.permutation(n)
    X["grp"] = r.integers(0, 300, n); X["const"] = 1.0
    return X, y, truth


def noise_factor_blocks(seed, n=2000, p=30, k=3, rho=0.9, binary=True):
    """Pure noise from correlated factor blocks plus product and ratio columns: pair structure exists but is independent of y."""
    r = np.random.default_rng(seed); g = r.normal(size=(n, k)); blk = np.arange(p) % k
    X = pd.DataFrame(np.sqrt(rho) * g[:, blk] + np.sqrt(1 - rho) * r.normal(size=(n, p)),
                     columns=[f"x{j}" for j in range(p)])
    X["x_prod01"] = X.x0 * X.x1; X["x_ratio23"] = X.x2 / (np.abs(X.x3) + 0.1)
    y = (r.random(n) > 0.5).astype(np.int64) if binary else r.normal(size=n)
    return X, y


def noise_wide_pn(seed, n=300, p=600):
    """Pure noise with p much larger than n and mixed column types."""
    r = np.random.default_rng(seed); cols = {}
    for j in range(p):
        t = j % 4
        cols[f"x{j}"] = (r.normal(size=n) if t == 0 else r.lognormal(0, 1.5, n) if t == 1
                         else r.integers(0, 2, n).astype(float) if t == 2 else r.integers(0, 5, n).astype(float))
    return pd.DataFrame(cols), (r.random(n) > 0.5).astype(np.int64)


def noise_tied_targets(seed, kind="zero_infl", n=None):
    """Pure-noise frames whose target is heavily tied, near-constant or rare."""
    r = np.random.default_rng(seed); n = n or (5000 if kind == "rare" else 2000)
    X = pd.DataFrame({f"x{j}": (r.normal(size=n) if j % 2 else r.integers(0, 6, n).astype(float)) for j in range(10)})
    if kind == "zero_infl": y = np.where(r.random(n) < 0.03, r.lognormal(0, 1, n), 0.0)
    elif kind == "ordinal": y = np.searchsorted([0.70, 0.95], r.random(n))
    else: y = (r.random(n) < 0.006).astype(np.int64)
    return X, y


def noise_highcard_strings(seed, n=2000):
    """Pure noise with high-cardinality string and categorical columns and an id."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"s{L}": pd.Categorical(r.integers(0, L, n).astype(str)) for L in (1000, 200, 50)})
    X["id"] = r.permutation(n); X["num"] = r.normal(size=n)
    return X, (r.random(n) > 0.5).astype(np.int64)


def noise_missing_dupes_constants(seed, n=2000):
    """Pure noise with NaN patterns, duplicated, constant and all-NaN columns."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 8)), columns=[f"x{j}" for j in range(8)])
    X["dup"] = X.x0; X["neg"] = -X.x1; X["jit"] = X.x2 + 1e-9 * r.normal(size=n)
    X["const"] = 7.0; X["allnan"] = np.nan; X["near_const"] = (r.random(n) < 0.002).astype(float)
    X.loc[r.random(n) < 0.3, "x3"] = np.nan
    X.loc[X.x4 > 0.5, "x4"] = np.nan
    X.loc[X.x5.isna() | (X.x6 > 1.0), "x7"] = np.nan
    return X, (r.random(n) > 0.5).astype(np.int64)


def noise_ar1_stationary(seed, n=2000, p=8, rho=0.9):
    """Autocorrelated stationary noise: an iid permutation null is anti-conservative here (report only)."""
    r = np.random.default_rng(seed)
    def ar(m):
        x = np.empty(n); x[0] = r.normal(); e = r.normal(size=n) * np.sqrt(1 - rho ** 2)
        for i in range(1, n): x[i] = rho * x[i - 1] + e[i]
        return x
    return pd.DataFrame({f"x{j}": ar(0) for j in range(p)}), ar(0)


def noise_random_walks(seed, n=2000, p=8):
    """Independent random walks (spurious regression, report only)."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(np.cumsum(r.normal(size=(n, p)), axis=0), columns=[f"w{j}" for j in range(p)])
    return X, np.cumsum(r.normal(size=n))


def noise_heavy_tail_ties(seed, n=2000, p=8):
    """Pure noise with heavy-tailed columns and tied target values."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{j}": (r.lognormal(0, 3, n) if j % 2 else r.standard_cauchy(n)) for j in range(p)})
    return X, np.round(r.standard_cauchy(n))


def noise_permuted_f2(seed, n=2000):
    """The F2 frame with its target permuted: the paired negative control."""
    r = np.random.default_rng(seed); o = _base(r, n)
    y = _sig(o["a"], o["b"], o["c"], o["d"]) + 0.2 * r.normal(size=n)
    X = pd.DataFrame(o)
    X["q0"] = np.exp(o["a"]); X["q1"] = np.log(o["b"]); X["a_dup"] = o["a"].copy()
    X["ab_ratio"] = o["a"] / o["b"]; X["sqa_over_b"] = o["a"] ** 2 / o["b"]
    return X, r.permutation(y)


def noise_float32_offset_quantised(seed, n=2000):
    """Pure noise stored as offset float32 columns, quantised."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"big32": (1e9 + r.normal(size=n)).astype(np.float32),
                      "ts_ns": (1_700_000_000_000_000_000 + np.sort(r.integers(0, 10 ** 12, n))).astype(np.int64),
                      "flag": r.random(n) > 0.5, "half": r.normal(size=n).astype(np.float16),
                      "ok": r.normal(size=n)})
    return X, r.normal(size=n)


def _frame(cols, p_noise, rng, n):
    """Dict of signal columns plus ``p_noise`` standard-normal noise columns as a frame."""
    d = dict(cols)
    for i in range(p_noise):
        d[f"noise_{i}"] = rng.standard_normal(n)
    return pd.DataFrame(d)


def g_parity_k(seed, k=3, n=3000, p_noise=100, flip=0.0, continuous=False):
    """Parity of k binary or sign-split operands (zero marginal signal), with label flips and noise columns."""
    rng = np.random.default_rng(seed)
    if continuous:
        Z = rng.uniform(-1, 1, (n, k)); par = (Z > 0).sum(1) % 2
    else:
        Z = rng.integers(0, 2, (n, k)).astype(float); par = Z.sum(1).astype(int) % 2
    y = np.where(rng.random(n) < flip, 1 - par, par)
    X = _frame({f"x{i+1}": Z[:, i] for i in range(k)}, p_noise, rng, n)
    return X, pd.Series(y)


def g_xor_mixed_types(seed, n=3000, p_noise=50):
    """3-way XOR over a continuous, a binary and a few-level categorical operand."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n); b = rng.integers(0, 2, n); c = rng.integers(0, 4, n)
    y = ((a > 0).astype(int) + b + (c % 2)) % 2
    cols = {"x1": a, "x2": b.astype(float), "x3": pd.Categorical(c.astype(str))}
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_xor_unbalanced(seed, n=3000, q=0.15, p_noise=50):
    """3-way XOR of Bernoulli(q) operands: unequal class balance and a real marginal."""
    rng = np.random.default_rng(seed)
    Z = (rng.random((n, 3)) < q).astype(float)
    y = Z.sum(1).astype(int) % 2
    return _frame({f"x{i+1}": Z[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)


def g_xor_nan(seed, n=3000, nan_rate=0.1, p_noise=50):
    """3-way XOR whose operands carry MCAR NaNs."""
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)).astype(float)
    y = Z.sum(1).astype(int) % 2
    Zm = np.where(rng.random((n, 3)) < nan_rate, np.nan, Z)
    return _frame({f"x{i+1}": Zm[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)


def g_xor_weak_marginal(seed, n=3000, beta=0.5, p_noise=50):
    """3-way XOR where one operand also has a weak marginal effect."""
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)).astype(float)
    par = Z.sum(1) % 2
    logit = 4.0 * (par - 0.5) + beta * (Z[:, 0] - 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return _frame({f"x{i+1}": Z[:, i] for i in range(3)}, p_noise, rng, n), pd.Series(y)


def g_two_xor_triples(seed, n=3000, p_noise=50, overlap=False):
    """Two XOR triples selected by a coin per row, disjoint or sharing one operand."""
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 6)).astype(float)
    t1 = Z[:, :3].sum(1) % 2
    t2 = (Z[:, [0, 3, 4]] if overlap else Z[:, 3:]).sum(1) % 2
    sel = rng.random(n) < 0.5
    y = np.where(sel, t1, t2).astype(int)
    cols = {f"x{i+1}": Z[:, i] for i in range(6)}
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_gated_parity(seed, n=4000, p_noise=50):
    """Parity of three operands that holds only when a fourth column is 1."""
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, 2, (n, 3)); g = rng.integers(0, 2, n)
    y = np.where(g == 1, Z.sum(1) % 2, rng.integers(0, 2, n))
    cols = {"g": g.astype(float), **{f"x{i+1}": Z[:, i].astype(float) for i in range(3)}}
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_parity_monotone_twins(seed, n=3000, p_noise=50):
    """Parity of three operands that also appear as exp and cube monotone twins."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, (n, 3))
    y = ((Z > 0).sum(1) % 2).astype(int)
    cols = {f"x{i+1}": Z[:, i] for i in range(3)}
    cols.update({f"x{i+1}_exp": np.exp(2 * Z[:, i]) for i in range(3)})
    cols.update({f"x{i+1}_cube": Z[:, i] ** 3 for i in range(3)})
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_conj_exact_partial_decoys(seed, n=3000, p_noise=50, prev=0.5):
    """Conjunction of A, B, C with exact, monotone and partial-AND decoys."""
    rng = np.random.default_rng(seed)
    A, B, C = (rng.random((3, n)) < prev).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 2.5 * A + 3.0 * A * B * C)))).astype(int)
    cols = {"A": A, "B": B, "C": C, "A_copy": 10.0 * A + 3, "A_exp": np.exp(A), "AB": A & B, "AC": A & C, "BC": B & C}
    return _frame({k: np.asarray(v, float) for k, v in cols.items()}, p_noise, rng, n), pd.Series(y)


def g_rare_conj(seed, k=3, prev=0.25, n=3000, flip=0.02, p_noise=50):
    """Rare conjunction of k Bernoulli(prev) operands with a small flip rate."""
    rng = np.random.default_rng(seed)
    Z = (rng.random((n, k)) < prev).astype(float)
    y = Z.prod(1).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    return _frame({f"x{i+1}": Z[:, i] for i in range(k)}, p_noise, rng, n), pd.Series(y)


def g_and_or_mixture(seed, n=3000, p_noise=50, flip=0.03):
    """(A and B) or (C and D) with label flips."""
    rng = np.random.default_rng(seed)
    A, B, C, D = rng.integers(0, 2, (4, n))
    y = ((A & B) | (C & D)).astype(int)
    y = np.where(rng.random(n) < flip, 1 - y, y)
    cols = {k: v.astype(float) for k, v in zip("ABCD", (A, B, C, D))}
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_many_level_cat_xor(seed, n=3000, L=20, p_noise=50):
    """XOR of a many-level categorical's half-set indicator with a binary column."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, L, n); b = rng.integers(0, 2, n)
    S = np.random.default_rng(99).permutation(L)[: L // 2]
    y = (np.isin(cat, S).astype(int) + b) % 2
    cols = {"cat": pd.Categorical([f"c{v}" for v in cat]), "b": b.astype(float)}
    return _frame(cols, p_noise, rng, n), pd.Series(y)


def g_regression_xor_conj(seed, n=3000, p_noise=50, heavy=True):
    """Regression target: heavy-tailed sign-parity times capped magnitude plus a conjunction offset."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_t(3, (n, 3)) if heavy else rng.normal(size=(n, 3))
    B = rng.integers(0, 2, (2, n))
    y = np.sign(Z).prod(1) * np.minimum(np.abs(Z).prod(1), 5) + 0.5 * rng.normal(size=n)
    cols = {"x1": Z[:, 0], "x2": Z[:, 1], "x3": Z[:, 2], "A": B[0].astype(float), "C": B[1].astype(float)}
    return _frame(cols, p_noise, rng, n), pd.Series(y + 2.0 * B[0] * B[1])


def g_null_control(seed, n=3000, p_noise=50):
    """The parity frame with a permuted target: the null control of the same shape."""
    X, y = g_parity_k(seed, 3, n, p_noise)
    return X, pd.Series(np.random.default_rng(seed + 7).permutation(y.to_numpy()))


@dataclass(frozen=True)
class Case:
    """One bench case: a generator, the kind of contract that judges it, and the panel it runs over.

    ``kind`` is ``f2`` (generator returns ``(X, y, truth)``, judged by ``f2_verdict``), ``null`` (generator returns ``(X, y)``, nothing may be selected) or ``cover`` (generator
    returns ``(X, y)``, judged by ``cover_verdict`` token coverage). ``report_only`` cases print their panel verdict but never fail the bench: their gates are proposals not yet
    calibrated against real runs, or the failure is a documented methodology limit.
    """

    name: str
    pillar: int
    kind: str
    gen: Callable
    gen_kwargs: dict = field(default_factory=dict)
    contract: dict = field(default_factory=dict)
    seeds: int = 10
    power_gate: float = 0.74
    fp_gate: float = 0.26
    report_only: bool = True
    note: str = ""


_XOR3 = [["x1", "x2", "x3"]]
_NOISE = ("noise_",)

CASES: list = [
    Case("f2_monotone_decoys", 1, "f2", f2_monotone_decoys, contract={"noise_cols": ("e",), "max_cols": 3}),
    Case("f2_leverage_tiny_b", 1, "f2", f2_leverage_tiny_b, contract={"noise_cols": ("e",), "min_rho": 0.80}, seeds=20, power_gate=0.70, note="tail noise caps attainable rho"),
    Case("f2_scale_and_range", 1, "f2", f2_scale_and_range, contract={"noise_cols": ("e",)}),
    Case("f2_dupes_collinear", 1, "f2", f2_dupes_collinear, contract={"noise_cols": ("e", "e_dup"), "max_cols": 3}),
    Case("f2_discrete_mixed", 1, "f2", f2_discrete_mixed, contract={"noise_cols": ("e",), "min_rho": 0.85}),
    Case("f2_id_columns", 1, "f2", f2_id_columns, contract={"noise_cols": ("e", "rid", "uid", "grp", "const")}),
    Case("noise_factor_blocks", 2, "null", noise_factor_blocks, seeds=20, fp_gate=0.14),
    Case("noise_wide_pn", 2, "null", noise_wide_pn, seeds=20, fp_gate=0.14),
    Case("noise_tied_targets_zero_infl", 2, "null", noise_tied_targets, gen_kwargs={"kind": "zero_infl"}, seeds=20, fp_gate=0.14),
    Case("noise_highcard_strings", 2, "null", noise_highcard_strings, seeds=20, fp_gate=0.14),
    Case("noise_missing_dupes_constants", 2, "null", noise_missing_dupes_constants, seeds=20, fp_gate=0.14),
    Case("noise_heavy_tail_ties", 2, "null", noise_heavy_tail_ties, seeds=20, fp_gate=0.14),
    Case("noise_permuted_f2", 2, "null", noise_permuted_f2, seeds=20, fp_gate=0.14, note="paired negative control of the F2 frame"),
    Case("noise_float32_offset_quantised", 2, "null", noise_float32_offset_quantised, seeds=20, fp_gate=0.14),
    Case("noise_ar1_stationary", 2, "null", noise_ar1_stationary, seeds=30, fp_gate=0.10, note="iid permutation null is anti-conservative on autocorrelated data"),
    Case("noise_random_walks", 2, "null", noise_random_walks, seeds=30, fp_gate=0.10, note="spurious regression of independent random walks"),
    Case("null_control_parity_frame", 2, "null", g_null_control, seeds=20, fp_gate=0.14, note="same frame shape as the XOR cases, permuted target"),
    Case("parity3_continuous", 3, "cover", g_parity_k, {"k": 3, "continuous": True, "p_noise": 100}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("parity3_binary", 3, "cover", g_parity_k, {"k": 3, "p_noise": 100}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("parity4_binary", 3, "cover", g_parity_k, {"k": 4, "n": 4000, "p_noise": 100}, {"required": [["x1", "x2", "x3", "x4"]], "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("parity3_label_noise_0.1", 3, "cover", g_parity_k, {"k": 3, "flip": 0.1, "p_noise": 100}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("parity3_label_noise_0.2", 3, "cover", g_parity_k, {"k": 3, "flip": 0.2, "p_noise": 100}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.70),
    Case("xor3_mixed_types", 3, "cover", g_xor_mixed_types, contract={"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("xor3_unbalanced_q0.3", 3, "cover", g_xor_unbalanced, {"q": 0.3}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("xor3_unbalanced_q0.15", 3, "cover", g_xor_unbalanced, {"q": 0.15}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("xor3_nan_0.1", 3, "cover", g_xor_nan, {"nan_rate": 0.1}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("xor3_nan_0.3", 3, "cover", g_xor_nan, {"nan_rate": 0.3}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.70),
    Case("xor3_weak_marginal_0.5", 3, "cover", g_xor_weak_marginal, {"beta": 0.5}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("xor3_weak_marginal_1.5", 3, "cover", g_xor_weak_marginal, {"beta": 1.5}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
    Case("two_xor_triples", 3, "cover", g_two_xor_triples, contract={"required": [["x1", "x2", "x3"], ["x4", "x5", "x6"]], "forbidden_prefixes": _NOISE, "max_cols": 2}, seeds=20, power_gate=0.70),
    Case("two_xor_triples_sharing_x1", 3, "cover", g_two_xor_triples, {"overlap": True, "n": 4000}, {"required": [["x1", "x2", "x3"], ["x1", "x4", "x5"]], "forbidden_prefixes": _NOISE, "max_cols": 2}, seeds=20, power_gate=0.60),
    Case("gated_parity", 3, "cover", g_gated_parity, contract={"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 2}, seeds=20, power_gate=0.80),
    Case("parity_monotone_twins", 3, "cover", g_parity_monotone_twins, contract={"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 1, "normalize": ("_exp", "_cube")}, seeds=20, power_gate=0.80),
    Case("conj_exact_partial_decoys", 4, "cover", g_conj_exact_partial_decoys, contract={"required": [["A", "B", "C"]], "forbidden_prefixes": _NOISE, "forbidden_exact": ("AB", "AC", "BC"), "max_cols": 2, "normalize": ("_copy", "_exp")}, seeds=20, power_gate=0.80),
    Case("rare_conj_k3_prev0.25", 4, "cover", g_rare_conj, {"k": 3, "prev": 0.25, "n": 6000, "flip": 0.002}, {"required": _XOR3, "forbidden_prefixes": _NOISE, "max_cols": 2}, seeds=20, power_gate=0.70),
    Case("and_or_mixture", 4, "cover", g_and_or_mixture, contract={"required": [["A"], ["B"], ["C"], ["D"]], "forbidden_prefixes": _NOISE, "max_cols": 3}, seeds=20, power_gate=0.80),
    Case("many_level_cat_xor", 4, "cover", g_many_level_cat_xor, contract={"required": [["cat", "b"]], "forbidden_prefixes": _NOISE, "max_cols": 1}, seeds=20, power_gate=0.80),
]
