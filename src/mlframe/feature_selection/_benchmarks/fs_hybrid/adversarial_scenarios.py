"""Adversarial Phase-0 scenarios: the beds where this repo's own flagship selectors are documented to LOSE.

``scenarios.py`` grew organically around cases MRMR handles well (linear + redundant clusters + a single
multiplicative interaction). ``docs/MRMR_RESEARCH.md`` scores MRMR **L (miss)** on 2-way XOR (row 6),
3-way XOR / parity (row 7) and group-additive sums (row 11), and ``filters/_mrmr_tree_rescue.py`` records
MRMR collapsing to under four features on madelon (downstream lgbm 0.6885 against 0.87 on all features).
A scenario list that omits those rows is rigged by omission, so they live here as first-class beds.

Every generator is a PURE FUNCTION of ``(name, seed)``:

* RNG streams are addressed by NAME through :func:`stream_for` (``blake2b`` entropy, never the builtin
  ``hash()`` -- ``PYTHONHASHSEED``-dependent, and this repository has been bitten by it twice). Inserting
  a scenario, or a new stream inside one, cannot shift another one's draws.
* Every emitted column is standardised to unit variance, and the pre-standardisation scale is recorded in
  ``truth["pre_std_scale"]``. In an additively generated model the marginal variance grows with depth in
  the generative order, so sorting features by raw variance partially recovers the informative set
  (varsortability -- Reisach, Seiler & Drton, NeurIPS 2021). Standardising removes that leak, which is
  what makes the ``VarianceSortArm`` control meaningful.

``truth`` keeps the exact shape of ``scenarios.py`` (``base`` / ``relevant`` / ``noise`` /
``interaction_operands`` / ``quadratic_operands``) and adds:

  expected_to_break  - tuple of arm names this bed is designed to defeat (pre-registration requirement)
  pre_std_scale      - name -> standard deviation before standardisation
  metric             - the primary metric for this bed when it is NOT recall ("fdr", "n_selected")
  notes              - free-form string describing the planted structure

Scenario names are the registry keys of :data:`ADVERSARIAL_SCENARIOS`; ``scenarios.make`` dispatches over
the merged mapping.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Sequence, Tuple, cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# deterministic, name-addressed RNG
# ---------------------------------------------------------------------------


def stable_name_hash(name: str) -> int:
    """Return a stable 64-bit integer for ``name`` (``blake2b``; NEVER the builtin ``hash()``)."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def stream_for(root_seed: int, *path: str) -> np.random.Generator:
    """Return the generator for stream ``path`` under ``root_seed``, addressed by name rather than position."""
    entropy = [int(root_seed), *(stable_name_hash(part) for part in path)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# shared assembly
# ---------------------------------------------------------------------------

_TRUTH_LIST_KEYS = ("base", "relevant", "noise", "interaction_operands", "quadratic_operands")


def _finalize(
    name: str,
    seed: int,
    cols: Dict[str, np.ndarray],
    y: np.ndarray,
    truth: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Shuffle column order, standardise every column to unit variance and complete the ``truth`` dict."""
    frame = pd.DataFrame(cols)
    order = list(frame.columns)
    stream_for(seed, name, "column_order").shuffle(order)
    frame = frame[order]

    scales: Dict[str, float] = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        sd = float(values.std())
        scales[column] = sd
        if sd > 0.0:
            frame[column] = (values - values.mean()) / sd
        else:  # pragma: no cover - a constant column would be a generator bug, not a data property
            frame[column] = values - values.mean()

    for key in _TRUTH_LIST_KEYS:
        truth.setdefault(key, [])
    truth["pre_std_scale"] = scales
    truth.setdefault("metric", "recall")
    truth.setdefault("notes", "")
    if not truth.get("expected_to_break"):
        raise ValueError(f"scenario {name!r} must declare a non-empty expected_to_break")
    return frame, pd.Series(np.asarray(y, dtype=int), name="target"), truth


def _noise_block(rng: np.random.Generator, n: int, count: int, cols: Dict[str, np.ndarray]) -> List[str]:
    """Append ``count`` iid standard-normal probe columns to ``cols`` and return their names."""
    names = []
    for i in range(count):
        key = f"noise_{i}"
        cols[key] = rng.standard_normal(n)
        names.append(key)
    return names


def _bernoulli(rng: np.random.Generator, probability: np.ndarray) -> np.ndarray:
    """Draw a binary label vector from per-row success probabilities."""
    return (rng.random(probability.shape[0]) < probability).astype(int)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)


def _min_rows_for_minority(n: int, prevalence: float, min_minority: int = 5000) -> int:
    """Return a row count large enough that the minority class holds at least ``min_minority`` rows."""
    rate = min(prevalence, 1.0 - prevalence)
    return max(n, int(np.ceil(min_minority / max(rate, 1e-9))))


# ---------------------------------------------------------------------------
# 1. group_additive -- MRMR_RESEARCH.md row 11 (MRMR "L", everyone else W/T)
# ---------------------------------------------------------------------------


def group_additive(seed: int = 0, n: int = 5000, n_group: int = 10, n_noise: int = 40) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """``y`` driven by the SUM of ~10 individually-weak features; no member is marginally strong."""
    name = "group_additive"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    group = rng_x.standard_normal((n, n_group))
    logit = 0.30 * group.sum(axis=1)  # a member alone barely moves y (AUC ~0.57); the sum of ten does (~0.72)
    y = _bernoulli(rng_y, _sigmoid(logit))

    cols: Dict[str, np.ndarray] = {f"grp_{i}": group[:, i] for i in range(n_group)}
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    base = [f"grp_{i}" for i in range(n_group)]
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": noise,
        "expected_to_break": ("mrmr", "univariate_ht"),
        "notes": "MRMR_RESEARCH.md row 11: only the sum is informative; greedy marginal-MI ranking drops members below probes.",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 2. compensable_pair -- MRMR_RESEARCH.md row 13 neighbourhood (ShapProxied uniquely best)
# ---------------------------------------------------------------------------


def compensable_pair(seed: int = 0, n: int = 5000, n_noise: int = 30) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """A near-collinear pair whose DIFFERENCE drives ``y``; either half alone is nearly useless.

    ``a = u + e_a`` and ``b = u + e_b`` share a dominant latent ``u`` that does not enter the target at all,
    so their marginal association with ``y`` is small while ``a - b`` carries the whole signal. A redundancy
    penalty sees a highly correlated pair and keeps exactly one -- destroying the signal.
    """
    name = "compensable_pair"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    latent = rng_x.standard_normal(n)
    delta_a = 0.20 * rng_x.standard_normal(n)
    delta_b = 0.20 * rng_x.standard_normal(n)
    a = latent + delta_a
    b = latent + delta_b
    decoy = latent + 0.20 * rng_x.standard_normal(n)  # same cluster, carries no delta -> no value at all
    y = _bernoulli(rng_y, _sigmoid(6.0 * (a - b)))

    cols: Dict[str, np.ndarray] = {"comp_a": a, "comp_b": b, "cluster_decoy": decoy}
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    truth: Dict[str, object] = {
        "base": ["comp_a", "comp_b"],
        "relevant": ["comp_a", "comp_b"],
        "noise": noise + ["cluster_decoy"],
        "interaction_operands": ["comp_a", "comp_b"],
        "expected_to_break": ("mrmr", "group_aware_mrmr", "cluster_aggregate", "univariate_ht", "knockoffs"),
        "notes": "The pair is jointly decisive and individually weak; de-duplicating the correlated cluster erases the signal.",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 3. fdr_under_budget -- MRMR_RESEARCH.md row 15 (knockoffs uniquely best); metric is FDR
# ---------------------------------------------------------------------------


def fdr_under_budget(
    seed: int = 0, n: int = 5000, n_informative: int = 20, n_noise: int = 180, nominal_fdr: float = 0.10
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Graded-relevance mixture scored by FALSE DISCOVERY RATE against a declared budget, not by recall."""
    name = "fdr_under_budget"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    informative = rng_x.standard_normal((n, n_informative))
    # Geometrically graded coefficients: the weakest members sit under any practical detection threshold,
    # so an arm can only gain recall by spending false discoveries -- which is exactly what FDR prices.
    coefficients = np.geomspace(0.60, 0.08, n_informative)
    y = _bernoulli(rng_y, _sigmoid(informative @ coefficients))

    cols: Dict[str, np.ndarray] = {f"inf_{i}": informative[:, i] for i in range(n_informative)}
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    base = [f"inf_{i}" for i in range(n_informative)]
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": noise,
        "metric": "fdr",
        "nominal_fdr": nominal_fdr,
        "coefficients": {f"inf_{i}": float(coefficients[i]) for i in range(n_informative)},
        "expected_to_break": ("rfecv", "mrmr", "boruta"),
        "notes": "MRMR_RESEARCH.md row 15. Primary metric is FDR at the declared budget; recall is secondary.",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 4-5. pure parity beds -- MRMR_RESEARCH.md rows 6 and 7 (MRMR "L" on both)
# ---------------------------------------------------------------------------


def _parity(operands: np.ndarray) -> np.ndarray:
    """Return the XOR of the operand signs as a {0, 1} vector."""
    return np.asarray((np.sum(operands > 0.0, axis=1) % 2).astype(int))


def _make_xor(name: str, seed: int, n: int, order: int, n_noise: int, flip: float) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Build a pure ``order``-way parity bed: EVERY operand has exactly zero marginal MI with ``y``."""
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    operands = rng_x.standard_normal((n, order))
    y = _parity(operands)
    flips = rng_y.random(n) < flip
    y = np.where(flips, 1 - y, y)

    cols: Dict[str, np.ndarray] = {f"xor_{i}": operands[:, i] for i in range(order)}
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    base = [f"xor_{i}" for i in range(order)]
    # Greedy forward selection scores one candidate at a time, so no single operand ever shows a gain.
    breaks = (
        ("mrmr", "univariate_ht", "knockoffs", "forward_select")
        if order == 2
        else ("mrmr", "rfecv", "shap_proxied", "boruta", "knockoffs", "univariate_ht", "forward_select")
    )
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": noise,
        "interaction_operands": list(base),
        "label_flip_rate": flip,
        "expected_to_break": breaks,
        "notes": f"Pure {order}-way synergy (MRMR_RESEARCH.md row {6 if order == 2 else 7}); zero marginal MI on every operand by symmetry.",
    }
    return _finalize(name, seed, cols, y, truth)


def xor2(seed: int = 0, n: int = 5000, n_noise: int = 30) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Pure 2-way XOR: both operands carry zero marginal MI, the pair carries nearly a full bit."""
    return _make_xor("xor2", seed, n, 2, n_noise, 0.05)


def xor3(seed: int = 0, n: int = 5000, n_noise: int = 30) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Pure 3-way parity: every operand AND every operand PAIR carries zero MI; only the triple is informative."""
    return _make_xor("xor3", seed, n, 3, n_noise, 0.05)


def xor3_plus_marginal_decoy(seed: int = 0, n: int = 5000, n_noise: int = 30) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """3-way parity plus a decoy with HIGH marginal MI and no conditional value beyond the operands.

    The decoy is a noisy reflection of the parity itself, so it is fully screened off by the three operands
    (zero conditional MI given them) while dominating every marginal ranking. One dataset therefore measures
    both failure modes at once: marginal-greedy selection stops at the decoy, and synergy detection is needed
    to reach the operands the decoy is a lossy copy of.
    """
    name = "xor3_plus_marginal_decoy"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    rng_d = stream_for(seed, name, "decoy")
    operands = rng_x.standard_normal((n, 3))
    parity = _parity(operands)
    y = np.where(rng_y.random(n) < 0.05, 1 - parity, parity)
    decoy = (2.0 * parity - 1.0) + 1.1 * rng_d.standard_normal(n)  # strong marginal, redundant given operands

    cols: Dict[str, np.ndarray] = {f"xor_{i}": operands[:, i] for i in range(3)}
    cols["marginal_decoy"] = decoy
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    base = [f"xor_{i}" for i in range(3)]
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": noise,
        "interaction_operands": list(base),
        "decoy": ["marginal_decoy"],
        "label_flip_rate": 0.05,
        "expected_to_break": ("mrmr", "univariate_ht", "forward_select", "knockoffs"),
        "notes": "Decoy reflects the parity (high marginal MI, screened off by the operands); the operands have zero marginal MI.",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 6. probe_flood_p1000 -- NIPS 2003 designed-probe spirit
# ---------------------------------------------------------------------------


def probe_flood_p1000(seed: int = 0, n: int = 5000, n_informative: int = 8, n_probes: int = 992) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """A handful of informative features hidden among ~1000 probes drawn to MATCH the informative marginals.

    Probes are not plain N(0,1): each is drawn with a random scale and a random light tail, in the spirit of
    the NIPS 2003 feature-selection challenge designed probes, so a variance or moment screen cannot separate
    probes from signal.
    """
    name = "probe_flood_p1000"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    rng_p = stream_for(seed, name, "probes")
    informative = rng_x.standard_normal((n, n_informative))
    coefficients = np.linspace(0.9, 0.35, n_informative)
    y = _bernoulli(rng_y, _sigmoid(informative @ coefficients))

    cols: Dict[str, np.ndarray] = {f"inf_{i}": informative[:, i] for i in range(n_informative)}
    probe_names = []
    for i in range(n_probes):
        scale = float(rng_p.uniform(0.5, 2.0))
        draw = rng_p.standard_normal(n) if rng_p.random() < 0.5 else rng_p.standard_t(df=8, size=n)
        key = f"probe_{i}"
        cols[key] = scale * draw
        probe_names.append(key)
    base = [f"inf_{i}" for i in range(n_informative)]
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": probe_names,
        "expected_to_break": ("rfecv", "boruta", "shap_proxied"),
        "notes": "MRMR_RESEARCH.md row 14 at p=1000 with MATCHED probes; RFECV's one-SE band keeps hundreds of them (madelon 251/500).",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 7. null_p{10,100,1000} -- the negative control; runs FIRST and gates leaderboard entry
# ---------------------------------------------------------------------------


def _make_null(name: str, seed: int, n: int, p: int) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Build a bed where ``y`` is independent of ``X`` by construction; the correct answer is the empty set."""
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    rows = _min_rows_for_minority(n, 0.5)
    cols: Dict[str, np.ndarray] = {f"noise_{i}": rng_x.standard_normal(rows) for i in range(p)}
    y = (rng_y.random(rows) < 0.5).astype(int)  # drawn from an independent stream: no dependence can exist
    truth: Dict[str, object] = {
        "base": [],
        "relevant": [],
        "noise": list(cols),
        "metric": "n_selected",
        "gate": True,
        "expected_to_break": ("rfecv", "boruta", "zero_importance_pruning", "variance_sort"),
        "notes": (
            "Negative control. Report E[|selected|] and P(|selected| > 0) per arm. This bed RUNS FIRST and gates "
            "leaderboard entry: an arm selecting far above its nominal rate on pure noise is disqualified regardless "
            "of how it ranks elsewhere."
        ),
    }
    return _finalize(name, seed, cols, y, truth)


def null_p10(seed: int = 0, n: int = 5000) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Negative control at p=10."""
    return _make_null("null_p10", seed, n, 10)


def null_p100(seed: int = 0, n: int = 5000) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Negative control at p=100."""
    return _make_null("null_p100", seed, n, 100)


def null_p1000(seed: int = 0, n: int = 5000) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """Negative control at p=1000."""
    return _make_null("null_p1000", seed, n, 1000)


# ---------------------------------------------------------------------------
# 8. latent_replicates_private_delta -- the direct counter-scenario to cluster-aggregate
# ---------------------------------------------------------------------------


def latent_replicates_private_delta(
    seed: int = 0, n: int = 5000, n_replicates: int = 5, noise_sd: float = 0.7, distinct_sd: float = 0.6, n_noise: int = 20
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """``k`` noisy replicates of a latent factor, each carrying a PRIVATE delta that also drives ``y``.

    De-duplicating the cluster (keeping a medoid, or replacing it with a mean/PCA aggregate) destroys the
    private deltas and therefore destroys information: the correct answer is to keep the cluster WHOLE. This
    is the ported shape of ``make_latent_reflections(distinct_sd > 0)`` (case S4) from the biz-value suite,
    rebuilt here on name-addressed streams with standardised columns.
    """
    name = "latent_replicates_private_delta"
    rng_x = stream_for(seed, name, "features")
    rng_d = stream_for(seed, name, "deltas")
    rng_y = stream_for(seed, name, "target")
    latent = rng_x.standard_normal(n)
    deltas = distinct_sd * rng_d.standard_normal((n, n_replicates))
    replicates = latent[:, None] + noise_sd * rng_x.standard_normal((n, n_replicates)) + deltas
    independent = rng_x.standard_normal(n)
    # The deltas enter y with ALTERNATING, UNEQUAL weights. An equally-weighted sum would survive a mean
    # aggregate (the mean is proportional to it), so the bed would not actually punish de-duplication.
    delta_weights = np.linspace(1.6, 0.7, n_replicates) * np.where(np.arange(n_replicates) % 2 == 0, 1.0, -1.0)
    logit = 1.2 * latent + 0.4 * independent + deltas @ delta_weights
    y = _bernoulli(rng_y, _sigmoid(logit))

    cols: Dict[str, np.ndarray] = {f"refl_{i}": replicates[:, i] for i in range(n_replicates)}
    cols["indep"] = independent
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    group = [f"refl_{i}" for i in range(n_replicates)]
    truth: Dict[str, object] = {
        "base": group + ["indep"],
        "relevant": group + ["indep"],
        "noise": noise,
        "jointly_necessary_group": group,
        "delta_weights": {name: float(delta_weights[i]) for i, name in enumerate(group)},
        "must_keep_whole": True,
        "expected_to_break": ("mrmr", "group_aware_mrmr", "cluster_aggregate", "shap_proxied"),
        "notes": "Redundancy that is JOINTLY NECESSARY: aggregating the cluster averages the private deltas away.",
    }
    return _finalize(name, seed, cols, y, truth)


# ---------------------------------------------------------------------------
# 9. linear_gaussian_lowdim_n200 -- diagnostic: a t-test should beat a binned-MI selector at small n
# ---------------------------------------------------------------------------


def linear_gaussian_lowdim_n200(
    seed: int = 0, n: int = 200, n_informative: int = 5, n_noise: int = 15
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """5 informative, 15 noise, Gaussian, n=200: parametric structure that binning throws away.

    A plain univariate t-test filter exploits the Gaussian/linear form and should beat a binned-MI selector
    at this sample size. If MRMR wins here too, either the implementation is remarkable or the bed is broken
    -- that diagnostic value is the point of including it.
    """
    name = "linear_gaussian_lowdim_n200"
    rng_x = stream_for(seed, name, "features")
    rng_y = stream_for(seed, name, "target")
    informative = rng_x.standard_normal((n, n_informative))
    coefficients = np.linspace(0.9, 0.5, n_informative)
    y = _bernoulli(rng_y, _sigmoid(informative @ coefficients))

    cols: Dict[str, np.ndarray] = {f"inf_{i}": informative[:, i] for i in range(n_informative)}
    noise = _noise_block(stream_for(seed, name, "noise"), n, n_noise, cols)
    base = [f"inf_{i}" for i in range(n_informative)]
    truth: Dict[str, object] = {
        "base": base,
        "relevant": list(base),
        "noise": noise,
        "coefficients": {f"inf_{i}": float(coefficients[i]) for i in range(n_informative)},
        "expected_to_break": ("mrmr", "boruta"),
        "notes": "Small-n parametric bed: equal-frequency binning discards the Gaussian structure a t-test uses.",
    }
    return _finalize(name, seed, cols, y, truth)


ScenarioFn = Callable[..., Tuple[pd.DataFrame, pd.Series, Dict[str, object]]]

#: Beds that RUN FIRST and gate leaderboard entry (see ``_make_null``).
GATE_SCENARIOS: Tuple[str, ...] = ("null_p10", "null_p100", "null_p1000")

ADVERSARIAL_SCENARIOS: Dict[str, ScenarioFn] = {
    "null_p10": null_p10,
    "null_p100": null_p100,
    "null_p1000": null_p1000,
    "group_additive": group_additive,
    "compensable_pair": compensable_pair,
    "fdr_under_budget": fdr_under_budget,
    "xor2": xor2,
    "xor3": xor3,
    "xor3_plus_marginal_decoy": xor3_plus_marginal_decoy,
    "probe_flood_p1000": probe_flood_p1000,
    "latent_replicates_private_delta": latent_replicates_private_delta,
    "linear_gaussian_lowdim_n200": linear_gaussian_lowdim_n200,
}


def expected_to_break_index(names: Sequence[str] = ()) -> Dict[str, Tuple[str, ...]]:
    """Return ``arm -> scenarios it is expected to break``, for the pre-registration coverage meta-test."""
    index: Dict[str, List[str]] = {}
    for scenario in names or tuple(ADVERSARIAL_SCENARIOS):
        _, _, truth = ADVERSARIAL_SCENARIOS[scenario](0)
        arms = cast(Tuple[str, ...], truth["expected_to_break"])
        for arm in arms:
            index.setdefault(str(arm), []).append(scenario)
    return {arm: tuple(scenarios) for arm, scenarios in index.items()}


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    for scenario_name, builder in ADVERSARIAL_SCENARIOS.items():
        frame_out, target_out, truth_out = builder(0)
        print(
            f"{scenario_name:32s} shape={frame_out.shape} pos={float(target_out.mean()):.3f} "
            f"base={len(truth_out['base'])} noise={len(truth_out['noise'])} "  # type: ignore[arg-type]
            f"metric={truth_out['metric']} breaks={truth_out['expected_to_break']}"
        )
