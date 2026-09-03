"""Tests for ``mlframe.utils.experiments`` -- A/B experiment/route DB lookups.

``_validate_and_join_fields`` is the security-relevant piece: every DB-calling function routes its
caller-supplied ``fields`` through it before f-string-interpolating the result into a SQL SELECT, so an
unknown/unsafe field name must never reach the query string. DB calls (``safe_execute`` /
``safe_execute_values``) are monkeypatched -- no real database is needed to test the allowlisting,
row-unpacking, or query-shape logic.
"""

from __future__ import annotations

import pytest

from mlframe.utils.experiments import (
    _validate_and_join_fields,
    create_experiment,
    get_experiment_routes,
    get_experiments,
    read_experiment,
    read_route,
    update_routes_audiences,
)


class TestValidateAndJoinFields:
    """Groups tests covering _validate_and_join_fields."""

    def test_accepts_comma_separated_string(self):
        """Accepts comma separated string."""
        out = _validate_and_join_fields("id,name", frozenset({"id", "name", "started_at"}))
        assert out == "id,name"

    def test_accepts_sequence_of_strings(self):
        """Accepts sequence of strings."""
        out = _validate_and_join_fields(["id", "name"], frozenset({"id", "name"}))
        assert out == "id,name"

    def test_strips_whitespace_around_fields(self):
        """Strips whitespace around fields."""
        out = _validate_and_join_fields(" id , name ", frozenset({"id", "name"}))
        assert out == "id,name"

    def test_empty_string_raises(self):
        """Empty string raises."""
        with pytest.raises(ValueError):
            _validate_and_join_fields("", frozenset({"id"}))

    def test_whitespace_only_string_raises(self):
        """Whitespace only string raises."""
        with pytest.raises(ValueError):
            _validate_and_join_fields("   ", frozenset({"id"}))

    def test_empty_sequence_raises(self):
        """Empty sequence raises."""
        with pytest.raises(ValueError):
            _validate_and_join_fields([], frozenset({"id"}))

    def test_unknown_field_raises(self):
        """Unknown field raises."""
        with pytest.raises(ValueError, match="Unknown or unsafe field"):
            _validate_and_join_fields("id,secret_column", frozenset({"id", "name"}))

    def test_sql_injection_attempt_raises(self):
        """A field crafted to break out of the allowlisted column list (e.g. a subquery or comment
        injection) must be rejected outright, not partially sanitized."""
        with pytest.raises(ValueError, match="Unknown or unsafe field"):
            _validate_and_join_fields("id; drop table experiments;--", frozenset({"id", "name"}))

    def test_sql_injection_via_sequence_element_raises(self):
        """Sql injection via sequence element raises."""
        with pytest.raises(ValueError, match="Unknown or unsafe field"):
            _validate_and_join_fields(["id", "name) union select password from users --"], frozenset({"id", "name"}))


class TestGetExperiments:
    """Groups tests covering get_experiments."""

    def test_rejects_unsafe_fields_before_any_db_call(self, monkeypatch):
        """An unsafe field must raise before ``safe_execute`` is ever invoked (never reaches the DB layer)."""
        called = []
        monkeypatch.setattr("mlframe.utils.experiments.safe_execute", lambda *a, **k: called.append((a, k)))
        with pytest.raises(ValueError):
            get_experiments("acme", fields="id,drop_table")
        assert not called, "safe_execute must not be called when field validation fails"

    def test_builds_expected_query_and_params(self, monkeypatch):
        """Builds expected query and params."""
        captured = {}

        def _fake_safe_execute(query, params):
            """Capture the query/params the caller passed and return a fixed row set."""
            captured["query"] = query
            captured["params"] = params
            return [("exp-1", "spring_sale", "2026-01-01", None)]

        monkeypatch.setattr("mlframe.utils.experiments.safe_execute", _fake_safe_execute)
        out = get_experiments("acme")
        assert out == [("exp-1", "spring_sale", "2026-01-01", None)]
        assert captured["params"] == ("acme",)
        assert "select id,name,started_at,finished_at from experiments" in captured["query"]
        assert "%s" in captured["query"]  # product_name is parameterized, not interpolated
        assert "acme" not in captured["query"]  # the value itself must never appear in the query string

    def test_custom_fields_sequence_is_joined_and_validated(self, monkeypatch):
        """Custom fields sequence is joined and validated."""
        captured = {}
        monkeypatch.setattr("mlframe.utils.experiments.safe_execute", lambda q, p: (captured.__setitem__("query", q), [])[1])
        get_experiments("acme", fields=["id", "product_id"])
        assert "select id,product_id from experiments" in captured["query"]


class TestGetExperimentRoutes:
    """Groups tests covering get_experiment_routes."""

    def test_rejects_unsafe_fields_before_any_db_call(self, monkeypatch):
        """An unsafe field must raise before ``safe_execute`` is ever invoked."""
        called = []
        monkeypatch.setattr("mlframe.utils.experiments.safe_execute", lambda *a, **k: called.append((a, k)))
        with pytest.raises(ValueError):
            get_experiment_routes("exp-1", fields="id,drop_table")
        assert not called

    def test_builds_expected_query_and_params(self, monkeypatch):
        """Builds expected query and params."""
        captured = {}

        def _fake_safe_execute(query, params):
            """Capture the query/params and return a fixed route row set."""
            captured["query"] = query
            captured["params"] = params
            return [("r1", "control", "adults", "static")]

        monkeypatch.setattr("mlframe.utils.experiments.safe_execute", _fake_safe_execute)
        out = get_experiment_routes("exp-1")
        assert out == [("r1", "control", "adults", "static")]
        assert captured["params"] == ("exp-1",)
        assert "select id,name,audience,type from experiments_routes" in captured["query"]


class TestReadExperiment:
    """Groups tests covering read_experiment."""

    def test_unpacks_the_first_four_fields(self):
        """Unpacks the first four fields."""
        row = ("exp-1", "spring_sale", "2026-01-01", None)
        assert read_experiment(row) == ("exp-1", "spring_sale", "2026-01-01", None)

    def test_ignores_extra_trailing_fields(self):
        """A row with extra columns beyond the first four (e.g. product_id) is still unpacked correctly."""
        row = ("exp-1", "spring_sale", "2026-01-01", "2026-02-01", "product-9", "2026-01-01")
        assert read_experiment(row) == ("exp-1", "spring_sale", "2026-01-01", "2026-02-01")


class TestReadRoute:
    """Groups tests covering read_route."""

    def test_unpacks_and_normalizes_audience_as_set(self):
        """Unpacks and normalizes audience as set."""
        row = ("r1", "control", ["adults", "us"], "static")
        route_id, route_name, audience, route_type = read_route(row)
        assert (route_id, route_name, route_type) == ("r1", "control", "static")
        assert audience == {"adults", "us"}

    def test_null_audience_normalizes_to_empty_set(self):
        """Null audience normalizes to empty set."""
        row = ("r1", "control", None, "static")
        _id, _name, audience, _type = read_route(row)
        assert audience == set()

    def test_ignores_extra_trailing_fields(self):
        """Ignores extra trailing fields."""
        row = ("r1", "control", ["adults"], "static", "exp-1", "2026-01-01")
        _id, _name, audience, route_type = read_route(row)
        assert route_type == "static"
        assert audience == {"adults"}


class TestUpdateRoutesAudiences:
    """Groups tests covering update_routes_audiences."""

    def test_forwards_records_to_safe_execute_values(self, monkeypatch):
        """Forwards records to safe execute values."""
        captured = {}

        def _fake_safe_execute_values(query, records):
            """Capture the query/records passed through."""
            captured["query"] = query
            captured["records"] = records

        monkeypatch.setattr("mlframe.utils.experiments.safe_execute_values", _fake_safe_execute_values)
        records = [("r1", ["adults"]), ("r2", ["kids"])]
        update_routes_audiences(records)
        assert captured["records"] == records
        assert "update experiments_routes set audience" in captured["query"]


class TestCreateExperiment:
    """Groups tests covering create_experiment."""

    def test_is_a_documented_noop_stub(self):
        """create_experiment is an intentional stub -- must not raise and must return None."""
        assert create_experiment("product-1", variants=["a", "b"]) is None
