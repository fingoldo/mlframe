
from typing import Sequence, Union
import numpy as np
from pyutilz.db  import connect_to_db,select,showcase_table,explain_table,safe_execute,safe_execute_values


# Whitelists for SQL field injection protection.
_ALLOWED_EXPERIMENT_FIELDS = frozenset({
    "id", "name", "started_at", "finished_at", "product_id", "created_at", "updated_at",
})
_ALLOWED_ROUTE_FIELDS = frozenset({
    "id", "name", "audience", "type", "experiment_id", "created_at", "updated_at",
})


def _validate_and_join_fields(fields: Union[str, Sequence[str]], allowed: frozenset) -> str:
    """Validate ``fields`` against an allowlist and return a comma-joined SQL fragment.

    Accepts either a comma-separated string or a Sequence of column names.
    Raises ValueError on any unknown/unsafe field.
    """
    if isinstance(fields, str):
        parts = [f.strip() for f in fields.split(",") if f.strip()]
    else:
        parts = [str(f).strip() for f in fields]
    if not parts:
        raise ValueError("fields must be non-empty")
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unknown or unsafe field: {p!r}")
    return ",".join(parts)


def create_experiment(product_id: str, variants: list) -> str:
    # if there are active experiments currently, quit
    pass

def get_experiments(product_name: str, fields: Union[str, Sequence[str]] = "id,name,started_at,finished_at") -> list:
    # if there are active experiments currently, get them
    safe_fields = _validate_and_join_fields(fields, _ALLOWED_EXPERIMENT_FIELDS)
    return safe_execute(
        f"select {safe_fields} from experiments where started_at is not null and finished_at is null and product_id=(select id from products where name =%s) limit 1",
        (product_name,),
    )


def get_experiment_routes(
    experiment_id: str, fields: Union[str, Sequence[str]] = "id,name,audience,type"
) -> list:
    # if there are active experiments currently, get them
    safe_fields = _validate_and_join_fields(fields, _ALLOWED_ROUTE_FIELDS)
    routes = safe_execute(
        f"select {safe_fields} from experiments_routes where experiment_id =%s",
        (experiment_id,),
    )
    return routes

def read_experiment(experiment) -> tuple:
    experiment_id,experiment_name,experiment_started_at,experiment_finished_at, *_ = experiment

    return experiment_id,experiment_name,experiment_started_at,experiment_finished_at
    
def read_route(route) -> tuple:
    route_id, route_name, route_audience, route_type, *_ = route
    if route_audience is None:
        route_audience = []
    route_audience = set(route_audience)

    return route_id, route_name, route_audience, route_type

def update_routes_audiences(records) -> None:

    safe_execute_values(
        "with data (experiment_id,audience) as (VALUES %s) update experiments_routes set audience=data.audience::jsonb from data where id=data.experiment_id::uuid",
        records,
    )