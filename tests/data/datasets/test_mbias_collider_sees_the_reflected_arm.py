"""The M-bias collider is recognised through its reflected latent arm (audit 2026-11-20, finding 04-F5)."""
from mlframe.data.datasets._scm import build_ground_truth, graph_from_spec
from mlframe.data.datasets.ground_truth import FeatureRole
from mlframe.data.datasets.scenarios._causal_extra import m_bias_spec


def test_latent_reflection_is_an_edge_of_the_graph():
    """u_right generates y_parent, so the graph must carry u_right -> y_parent; without it the collider's second arm is invisible."""
    graph = graph_from_spec(m_bias_spec())
    assert any(edge.source == "u_right" and edge.target == "y_parent" for edge in graph.edges)


def test_m_bias_collider_gets_its_role():
    """m_collider has two latent parents, one of which reaches the target through its reflection, so it is the M-bias collider."""
    truth = build_ground_truth(m_bias_spec(), target_name="y")
    assert truth.features["m_collider"].role is FeatureRole.M_COLLIDER
