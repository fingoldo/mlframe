"""Two shapes that make a reasonable procedure wrong: a sign that flips, and rows that are not independent.

**Simpson's reversal.** A column whose effect on the target is positive in one subgroup and negative in
the other, in equal measure. Its marginal association with the target is zero by construction -- the two
halves cancel -- so every filter in the roster ranks it with the probes, and every one of them is wrong:
inside either group it is one of the strongest columns there is. Measured on this bed: marginal
correlation -0.003, and +0.51 / -0.51 within the two groups.

This is a different blindness from parity. A parity operand carries no information alone and needs its
partners; here the column carries a great deal of information alone, and the aggregate destroys it. A
method that conditions on the group finds it immediately, which makes the bed a direct test of whether a
method conditions at all.

**Grouped observations.** The target carries a per-group offset that no feature explains, so two rows of
the same group are correlated for a reason outside the model. That breaks the assumption every split in
this benchmark rests on. A random split puts rows of one group on both sides of it, the model learns the
group's offset from the training half and is rewarded for it on the holdout, and the estimate comes out
optimistic -- not by a little, and not in a way any metric computed on that split can reveal.

The bed does not fix that. It exposes it: the group column is in the frame, the offset is real, and the
truth says both. A protocol that splits by row will report a number it should not, and a protocol that
splits by group will report a smaller and honest one. The difference between those two numbers is the
thing worth measuring, and until this bed existed the suite had no way to produce it.
"""

from __future__ import annotations

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import BasisTerm, CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["simpson_reversal_spec", "grouped_rows_spec", "N_GROUPS"]

#: Distinct groups in the grouped bed. Few enough that every group has many rows -- a per-group offset
#: estimated from three rows is noise, not an offset -- and many enough that the split has something to
#: get wrong.
N_GROUPS = 40


def simpson_reversal_spec(n_noise: int = 28, n_samples: int = 6000, ceiling: float = 0.80, seed: int = 0) -> DatasetSpec:
    """Return the sign-reversal bed: a strong column with zero marginal association.

    The group indicator is a fair coin, so the two subgroups are the same size and the effects cancel
    exactly rather than approximately. An unbalanced version would leave a residual marginal association
    and the bed would be testing attenuation rather than reversal.
    """
    return DatasetSpec(
        name="simpson_sign_reversal",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_reversing"),
            FeatureSpec(name="group", family="bernoulli", params={"p": 0.5}),
            FeatureSpec(name="x_plain"),
            *probes(n_noise),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.5,
                # The reversing column enters ONLY through its interaction with the group, so its own
                # coefficient is absent rather than zero: a zero coefficient and no coefficient generate
                # the same data, and the absence says which the bed meant.
                link=LinkSpec(kind="linear", coefficients={"x_plain": 0.5}, interactions=(("x_reversing", "group"),), interaction_weights=(1.5,)),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=(
            EdgeSpec(source="x_reversing", target="y"),
            EdgeSpec(source="group", target="y"),
            EdgeSpec(source="x_plain", target="y"),
        ),
        provenance={"family": "structure", "purpose": "a strong column whose marginal association is zero because its sign flips between subgroups"},
    )


def grouped_rows_spec(n_noise: int = 28, n_samples: int = 8000, ceiling: float = 0.80, seed: int = 0, n_groups: int = N_GROUPS) -> DatasetSpec:
    """Return the bed whose rows are not independent: a per-group offset no feature explains.

    The offset is drawn once per group and shared by every row in it, which is what makes two rows of the
    same group correlated beyond anything the features say. Every estimate this suite computes assumes a
    split separates independent rows; here it does not, and a row-wise split will be optimistic by an
    amount no metric computed on that same split can detect.

    The group column is left IN the frame on purpose. Hiding it would make the bed a test of whether a
    method can detect unobserved clustering, which is a much harder and rarer problem; leaving it in makes
    the bed a test of whether the protocol uses the information it already has.
    """
    weights = {f"s{i}": float(1.1 * (0.8**i)) for i in range(4)}
    return DatasetSpec(
        name=f"grouped_rows_{n_groups}",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            *(FeatureSpec(name=name) for name in weights),
            FeatureSpec(name="group_id", family="categorical", params={"n_levels": float(n_groups)}, dtype="int", standardize=False),
            *probes(n_noise),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(
                    kind="logistic",
                    coefficients=weights,
                    # The offset is comparable to the strongest feature's contribution. Smaller and the
                    # leak would be real but unmeasurable; larger and the bed would be about the group
                    # column alone rather than about the split.
                    basis_terms=(BasisTerm(kind="group_effect", columns=("group_id",), weight=1.0),),
                ),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=(*(EdgeSpec(source=column, target="y") for column in weights), EdgeSpec(source="group_id", target="y")),
        provenance={"family": "structure", "purpose": "rows of one group share an offset no feature explains, so a row-wise split is optimistic"},
    )
