# `mlframe.data.datasets` — synthetic data that keeps its own answer key

Most synthetic-data helpers hand back `(X, y)` and throw away the only thing that made them worth
generating. `sklearn.make_classification` shuffles its columns, so it cannot tell you which ones were
informative even in principle. This package exists because a feature-selection benchmark needs the
answer key, and needs it to be defensible.

```python
from mlframe.data.datasets import scenarios
from mlframe.data.datasets.generator import generate

dataset = generate(scenarios.get("mb_spouse_collider").build(seed=0))
frame, target, truth = dataset.as_tuple()

truth.target_sets["markov_blanket"]     # the pre-registered answer key
truth.true_prob                          # the exact law the labels were drawn from
dataset.calibration["bayes_auc"]         # the achievable ceiling, computed not estimated
```

## What makes it different from a generator that returns a tuple

**Truth comes from the graph, not from a list.** A flat `informative=[...]` is undefined on the beds that
matter. Parity operands have zero marginal mutual information. A mediator is screened off by the thing it
mediates. A descendant of the target is highly predictive with zero causal effect. Five redundant copies
are each individually sufficient and jointly carry one bit. So the package derives three answer keys from
the declared edges and makes a scenario say which one it is scored against — on the collider bed the
choice decides the winner outright, and leaving it implicit would publish a preference as a finding.

**The ceiling is exact.** Every corruption must state how it transforms `true_prob`; the generator refuses
one that cannot. That invariant is what makes the achievable Brier exactly `mean p(1-p)`, the log-loss
exactly `mean H(p)`, and the AUC the exact pairwise probability under the realised law — no simulation, no
sampling noise, no convergence question.

**Difficulty is calibrated, not chosen.** A coefficient of 0.8 means something completely different under
a logistic link than under a parity gate, so a suite that pins coefficients is comparing links rather than
methods. Scenarios declare an achievable AUC and the link scale is bisected against the FINAL
probabilities — after the prevalence shift and after any corruption, because a bed declaring "0.80
achievable" and then flipping 5% of its labels ships data whose real ceiling is 0.76.

**Streams are addressed by name.** `stream_for(root_seed, *path)` hashes the path with blake2b, never with
the built-in `hash()` and never by position. Inserting a column into the middle of a spec leaves every
other column bit-identical, which is what makes the scenario library editable at all;
`SeedSequence.spawn(n)` quietly destroys that property.

**Columns are standardised for a reason.** In an additively generated SCM, variance grows with depth in
topological order, so sorting columns by variance recovers the causal order without ever looking at the
target (Reisach, Seiler & Drton, NeurIPS 2021). Unit variance closes that channel, the pre-standardisation
scale is kept in the truth record, and `tests/data/test_datasets_varsortability.py` checks every registered
bed that it stayed closed.

## Layout

| module | what lives there |
|---|---|
| `spec.py` | the declarative spec: features, copulas, latents, links, noise, missingness, targets |
| `ground_truth.py` | `FeatureRole`, the three target sets, the `Ceiling` record |
| `_rng.py` | name-addressed streams |
| `_columns.py` | marginal families, outlier injection, quantisation, standardisation |
| `_copula.py` | Gaussian / Student-t / Clayton dependence, and the tail-dependence diagnostic |
| `_latent.py` | latent factors and their reflections |
| `_links.py` | additive, interaction, parity, tail-gate and named nonlinear terms; heteroscedasticity |
| `_noise.py` | corruptions, each with its `true_prob` update |
| `_missing.py` | MCAR / MAR / MNAR masking, applied after the link |
| `_target.py` | prevalence, calibration, the exact Bayes ceiling |
| `_oracle.py` | the ceiling and the reference MI bundles |
| `_oracle_crosscheck.py` | the oracle's exact MI, confirmed against third-party `dit` on the exact joint law |
| `generator.py` | the orchestrator |
| `scenarios/` | the named bed library and its lock file |

## The scenario library

Beds are grouped by what they attack, not by how they were built:

| family | beds | what they isolate |
|---|---|---|
| `null` | 2 | false-discovery discipline where nothing is relevant |
| `linear` | 2 | the control, and the small-sample case |
| `redundant` | 2 | importance splitting across copies; jointly necessary members |
| `interactions` | 2 | zero marginal association; finding the wrong thing confidently |
| `tails` | 3 | non-monotone joint structure, and tail dependence isolated from rank correlation |
| `causal` | 6 | a blanket member invisible until a collider is conditioned on; M-bias, confounding, an instrument, proxy attenuation |
| `reference` | 4 | published formulas nobody here chose |
| `marginals` | 4 | heavy tails, contamination, point masses, quantisation |
| `mixed_types` | 3 | cardinality bias, the identifier trap, power-law levels |
| `observation` | 4 | missingness mechanisms, rare classes, the two halves of drift |
| `targets` | 3 | multiclass, ordinal and count, sharing one structure so only the target varies |
| `corrupted` | 3 | label noise: the only place the declared ceiling and the reachable one diverge |
| `economics` | 2 | columns that cost different amounts, and four correlation levels in one bed |
| `structure` | 2 | a sign that flips between subgroups, and rows that are not independent |

Every scenario declares `expected_to_break` — which arms it is designed to defeat — **before** the run.
Those declarations are scored afterwards, so declaring one costs something. Roughly half hold, which is
the useful answer: a suite whose predictions always held would be tuned to its author's beliefs, and one
whose predictions never held would not be measuring what it thinks.

`REGISTRY.lock.json` hashes every bed's structure. Adding a bed after seeing results is allowed and is
often the right response to a surprise, but it bumps the lock, shows up in the diff, and is reported as a
post-hoc addition rather than blending in.

## Writing a scenario

```python
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

def my_bed(n_samples: int = 5000, seed: int = 0) -> DatasetSpec:
    """Return a bed where two columns matter and thirty do not."""
    return DatasetSpec(
        name="my_bed",
        n_samples=n_samples,
        root_seed=seed,
        features=tuple(FeatureSpec(name=f"s{i}") for i in range(2)) + tuple(FeatureSpec(name=f"n{i:03d}") for i in range(30)),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.35,
                link=LinkSpec(kind="logistic", coefficients={"s0": 1.0, "s1": 0.7}),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(EdgeSpec(source="s0", target="y"), EdgeSpec(source="s1", target="y")),
    )
```

Three rules the meta-tests enforce, each because it was got wrong at least once:

1. **Probes share the informative columns' marginal.** Otherwise the bed is solvable without looking at
   the target, and the varsortability tripwire will say so.
2. **Name what the bed should defeat, and mean it.** A bed expecting nothing to break cannot produce a
   negative result, and every arm must be named by at least two beds before its prediction counts.
3. **Check that the bed tests what its name says.** Two builders in this repository were named after
   parity while computing a product, which leaves both operands marginally informative — the opposite of
   what parity is for. A third fired a "joint tail" gate on the upper tail only, leaving a marginal
   correlation of +0.51 that every univariate filter found immediately. All three looked right in
   review; the measurement is what caught them.

## Why the test suite's own generators were not redirected here

`tests/feature_selection/_synth/` holds the generators the biz_value tests share, and forty-nine test
files import them. The migration plan for this package called for pointing those shims at the promoted
production scenarios once the library existed, on the assumption that the two would produce equivalent
data. They do not, and the difference is deliberate on both sides.

Measured on the closest pair, `make_3way_xor` against `parity_spec`:

| | rows | columns | calibrated | column order |
|---|---|---|---|---|
| test shim | 2 000 | 10 | no | declaration order |
| this package | 6 000 | 33 | to an achievable AUC of 0.85 | shuffled per seed |

Every one of those differences is a property this package was built to have. Calibrating to a ceiling is
what makes difficulty comparable across link types; shuffling the columns is what stops a stable-sort tie
break from inheriting the answer key. The shims have none of them because they were written for a
different job: a fast, fixed fixture a biz_value test can pin a numeric threshold against.

Redirecting the shims would therefore change the data under forty-nine test files whose thresholds were
measured on the old data — silently, since the tests would still run. The two sets of generators answer
different questions and stay separate. What was worth taking from the migration was done: the generators
whose NAMES were wrong got fixed in place (two builders named after parity computed a product), and the
guard that catches that class of error, `test_synth_builder_name_contracts.py`, lives in the test suite
where those builders do.
