"""Synthetic panels for the feature-transform study (research B4, 2026-11-24).

make_panel builds n rows x p columns with a Gaussian-copula correlation structure:
* k informative NONLINEAR columns, each with a known link g in {log, square, recip, sqrt, abs, sigmoid}, placed so that the marginal fits the link
  (lognormal for log, Gaussian for square/abs, exponential or Poisson counts for sqrt, shifted lognormal for recip, scaled Gaussian or Student-t3 for sigmoid);
* n_lin informative LINEAR columns (identity link; one of them lognormal, one heavy-tailed t3 when asked: a transform must NOT be credited to them);
* for every informative column one or two PARTNER noise columns correlated with it at rho in {0.3, 0.6, 0.9} (a transformed feature correlated with an untransformed
  one: the credit must go to the right column);
* the remaining columns are noise: blocks of 5 with within-block correlation 0.5, marginals a mixture of Gaussian / lognormal / Student-t3 (heavy tail) /
  Poisson and 5-level ordinal (discrete) / exponential / uniform.
y = sum_k g_k(x_k) standardised, plus Gaussian noise so that the signal share of var(y) is `r2`.
Returns (X float64 (n, p), y, info) where info lists per column: role ('nl', 'lin', 'partner', 'noise'), link, marginal, partner_of, rho.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy import stats

LINKS = ("log", "square", "recip", "sqrt", "abs", "sigmoid")
# marginal that goes with each link
LINK_MARGINAL = {"log": "lognormal", "square": "gauss", "recip": "poslognormal", "sqrt": "exp", "abs": "gauss", "sigmoid": "gauss2"}
NOISE_MARGINALS = ("gauss", "lognormal", "t3", "poisson", "ordinal", "exp", "uniform")
NOISE_PROBS = (0.34, 0.14, 0.14, 0.10, 0.10, 0.10, 0.08)


def _marginal(z, kind):
    """Map a standard normal latent z (any shape) to the requested marginal through the Gaussian copula."""
    u = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
    if kind == "gauss":
        return z
    if kind == "gauss2":
        return 2.0 * z
    if kind == "lognormal":
        return np.exp(0.8 * z)
    if kind == "poslognormal":
        return 0.5 + np.exp(0.6 * z)
    if kind == "exp":
        return -np.log1p(-u)
    if kind == "t3":
        return stats.t.ppf(u, 3)
    if kind == "t3pos":
        return np.abs(stats.t.ppf(u, 3)) + 0.1
    if kind == "poisson":
        return stats.poisson.ppf(u, 3.0).astype(np.float64)
    if kind == "poisson4":
        return stats.poisson.ppf(u, 4.0).astype(np.float64)
    if kind == "ordinal":
        return (1 + np.digitize(z, [-1.0, -0.3, 0.3, 1.0])).astype(np.float64)
    if kind == "uniform":
        return u
    raise ValueError(kind)


def _link(g, x):
    if g == "log":
        return np.log(x)
    if g == "square":
        return x ** 2
    if g == "recip":
        return 2.0 / x
    if g == "sqrt":
        return np.sqrt(x)
    if g == "abs":
        return np.abs(x)
    if g == "sigmoid":
        return 2.0 * np.tanh(x)  # x already 2*z for gauss2, so this is tanh-shaped over the bulk
    if g == "id":
        return x
    raise ValueError(g)


def make_panel(n, p, seed, k_nl=4, n_lin=1, links=None, r2=0.7, partners=True, heavy_informative=False, discrete_informative=False,
               rho_levels=(0.3, 0.6, 0.9), noise_block=5, noise_rho=0.5, lin_marginals=("gauss",)):
    r = np.random.default_rng(seed)
    if links is None:
        links = [LINKS[(seed + i) % len(LINKS)] for i in range(k_nl)]
        # make sure the same link does not repeat when k_nl <= 6
        links = list(r.permutation(LINKS)[:k_nl]) if k_nl <= len(LINKS) else [LINKS[i % len(LINKS)] for i in range(k_nl)]
    links = list(links)
    k_nl = len(links)
    Z = r.standard_normal((n, p))
    role = ["noise"] * p
    link = ["-"] * p
    marg = ["gauss"] * p
    partner_of = [-1] * p
    rho_of = [0.0] * p
    col = 0
    inf_cols = []
    # informative nonlinear columns
    for _i, g in enumerate(links):
        m = LINK_MARGINAL[g]
        if heavy_informative and g == "sigmoid":
            m = "t3sig"
        if discrete_informative and g == "sqrt":
            m = "poisson4"
        role[col], link[col], marg[col] = "nl", g, m
        inf_cols.append(col)
        col += 1
    for i in range(n_lin):
        m = lin_marginals[i % len(lin_marginals)]
        role[col], link[col], marg[col] = "lin", "id", m
        inf_cols.append(col)
        col += 1
    # partners (noise columns correlated with an informative column)
    if partners:
        for ci, c in enumerate(inf_cols):
            for t in range(2 if len(inf_cols) * 3 + 2 * len(inf_cols) <= p else 1):
                if col >= p:
                    break
                rho = float(rho_levels[(ci + t) % len(rho_levels)])
                Z[:, col] = rho * Z[:, c] + np.sqrt(1 - rho * rho) * Z[:, col]
                role[col], partner_of[col], rho_of[col] = "partner", c, rho
                # partner marginal: untransformed Gaussian or lognormal (so a log/sqrt transform of it is a tempting wrong credit)
                marg[col] = ("lognormal", "gauss", "exp")[(ci + t) % 3]
                col += 1
    # remaining noise columns: blocks of `noise_block` with within-block correlation
    j = col
    while j < p:
        e = min(noise_block, p - j)
        common = r.standard_normal(n)
        for q in range(j, j + e):
            Z[:, q] = np.sqrt(noise_rho) * common + np.sqrt(1 - noise_rho) * Z[:, q]
        j += e
    for c in range(col, p):
        if role[c] == "noise":
            marg[c] = str(r.choice(NOISE_MARGINALS, p=NOISE_PROBS))
    X = np.empty((n, p))
    for c in range(p):
        m = marg[c]
        if m == "t3sig":
            X[:, c] = 2.0 * _marginal(Z[:, c], "t3")
        else:
            X[:, c] = _marginal(Z[:, c], m)
    sig = np.zeros(n)
    for c in inf_cols:
        gv = _link(link[c], X[:, c])
        sd = gv.std()
        sig += (gv - gv.mean()) / (sd if sd > 0 else 1.0)
    sig_sd = sig.std()
    noise_sd = sig_sd * np.sqrt((1 - r2) / r2)
    y = sig + noise_sd * r.standard_normal(n)
    info = {"role": role, "link": link, "marginal": marg, "partner_of": partner_of, "rho": rho_of, "inf_cols": inf_cols, "nl_cols": [c for c in inf_cols if role[c] == "nl"]}
    return X, y, info


def make_null_panel(n, p, seed, n_lin=0, noise_block=5, noise_rho=0.5):
    """Mostly-noise panel for the family-wise error study: y = noise (n_lin=0) or y = n_lin LINEAR informative columns plus noise; every column has a
    random marginal from NOISE_MARGINALS (heavy-tailed and discrete ones included), noise blocks of 5 correlated at 0.5."""
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, p))
    j = 0
    while j < p:
        e = min(noise_block, p - j)
        common = r.standard_normal(n)
        for q in range(j, j + e):
            Z[:, q] = np.sqrt(noise_rho) * common + np.sqrt(1 - noise_rho) * Z[:, q]
        j += e
    marg = [str(r.choice(NOISE_MARGINALS, p=NOISE_PROBS)) for _ in range(p)]
    X = np.empty((n, p))
    for c in range(p):
        X[:, c] = _marginal(Z[:, c], marg[c])
    sig = np.zeros(n)
    lin_cols = list(r.choice(p, n_lin, replace=False)) if n_lin else []
    for c in lin_cols:
        v = X[:, c]
        sig += (v - v.mean()) / (v.std() + 1e-12)
    if n_lin:
        y = sig + sig.std() * np.sqrt(0.3 / 0.7) * r.standard_normal(n)
    else:
        y = r.standard_normal(n)
    block = np.arange(p) // noise_block
    mates = sorted({int(c) for c in range(p) for l in lin_cols if block[c] == block[l] and c != l and noise_rho > 0})
    return X, y, {"marginal": marg, "lin_cols": [int(c) for c in lin_cols], "mates": mates}
