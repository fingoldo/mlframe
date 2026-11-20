"""A regional link gate is recorded on every column the link reads, not dropped (audit 2026-11-20, finding 04-F4)."""
from mlframe.data.datasets._scm import build_ground_truth, link_regions
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, GateSpec, LinkSpec, TargetSpec


def _spec(region):
    """Two features, one target whose link carries the given region."""
    return DatasetSpec(
        name="regional",
        n_samples=200,
        root_seed=0,
        features=(FeatureSpec(name="x1"), FeatureSpec(name="x2")),
        targets=(TargetSpec(name="y", prevalence=0.5, link=LinkSpec(coefficients={"x1": 1.0, "x2": 0.5}, region=region), calibrate_to=CeilingTarget(metric="auc", value=0.8)),),
        edges=(EdgeSpec(source="x1", target="y"), EdgeSpec(source="x2", target="y")),
    )


def test_region_is_recorded_on_each_column_the_link_reads():
    """Without the helper the caller passed no regions, so every FeatureTruth.region stayed None."""
    gate = GateSpec(column="x1", low=0.0)
    spec = _spec(gate)
    truth = build_ground_truth(spec, target_name="y", regions=link_regions(spec.targets[0].link))
    assert truth.features["x1"].region == gate
    assert truth.features["x2"].region == gate


def test_link_without_region_gives_no_gates():
    """A link with no region keeps the old behaviour: no per-column gates."""
    assert link_regions(_spec(None).targets[0].link) is None
