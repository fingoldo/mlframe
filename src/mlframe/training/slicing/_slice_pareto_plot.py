"""Pareto-frontier artifact for slice-stable early stopping.

When ``SliceStableESConfig.pareto_plot_enabled=True`` and a training run finished with a
populated ``slice_aggregate_history`` / ``slice_shard_score_history`` on the callback, this
module emits:

  - a per-iteration scatter ``(mean_t, std_t)`` coloured by iteration index;
  - the Pareto frontier overlaid as a connected line + larger markers;
  - the selected ``best_iter`` highlighted with a red X + numeric annotation;
  - alternative selections under different ``pareto_risk_quantile`` knobs, labelled so the
    operator can see what the trade-off knob would have picked without re-running.

Output paths land in ``ctx.metadata["slice_stable_es"][target_name][model_name]`` via
``register_pareto_artifact``. Backends + formats are taken from the config (default:
``plotly[html] + matplotlib[png]``).

We don't go through ``FigureSpec`` here because the multi-overlay layout (frontier line +
annotation + alternative markers on the same axes) doesn't map cleanly onto the existing
panel-spec types; direct backend calls with WARN-once-on-failure semantics keep the rest
of the training run safe from any rendering hiccup.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np

from .._cv_aggregation import compute_pareto_frontier, select_from_pareto

logger = logging.getLogger(__name__)


def generate_pareto_artifact(
    ctx: Any,
    target_name: str,
    model_name: str,
    callback: Any,
    *,
    config: Any,
    output_dir: str,
    monitor_metric_name: str = "metric",
    direction: str = "min",
) -> dict[str, Any] | None:
    """Render the Pareto-frontier plot for one (target, model) and register paths in metadata.

    Returns the per-model meta dict that was attached (also accessible via
    ``ctx.metadata['slice_stable_es'][target_name][model_name]``) so the caller can chain
    additional diagnostic fields. Returns ``None`` when no plot was emitted (disabled / short
    run / no shard history).
    """
    if not getattr(config, "pareto_plot_enabled", True):
        return None

    shard_history = getattr(callback, "slice_shard_score_history", None) or []
    agg_history = getattr(callback, "slice_aggregate_history", None) or []
    n_iters = len(shard_history)
    min_iters = int(getattr(config, "pareto_plot_min_iterations", 10))
    if n_iters < min_iters:
        logger.info(
            "slice-stable Pareto plot skipped for target=%s model=%s: n_iters=%d < min=%d",
            target_name, model_name, n_iters, min_iters,
        )
        return None

    means = np.array([float(np.mean(s)) for s in shard_history], dtype=float)
    stds = np.array([float(np.std(s, ddof=1)) if len(s) > 1 else 0.0 for s in shard_history], dtype=float)
    pareto_idx = compute_pareto_frontier(list(zip(means.tolist(), stds.tolist())),
                                          mean_direction=direction)  # type: ignore[arg-type]

    best_iter = getattr(callback, "best_iter", None)
    if best_iter is None or best_iter < 0 or best_iter >= n_iters:
        best_iter = int(np.argmin(means) if direction == "min" else np.argmax(means))

    # Alternative selections per risk-quantile knob.
    alt_quantiles = list(getattr(config, "pareto_plot_show_alt_quantiles", [0.5, 0.7, 0.9, 0.95]))
    alt_picks: dict[str, int] = {}
    if pareto_idx:
        for q in alt_quantiles:
            try:
                pick = select_from_pareto(
                    pareto_idx, means.tolist(), stds.tolist(), shard_history,
                    risk_quantile=float(q), direction=direction,  # type: ignore[arg-type]
                )
                alt_picks[f"{q}"] = int(pick)
            except Exception as exc:  # robust to numeric edge cases
                logger.debug("Pareto alt-quantile %.2f failed: %s", q, exc)

    # Build output paths.
    os.makedirs(output_dir, exist_ok=True)
    base_stem = os.path.join(output_dir, f"slice_stable_es_pareto.{_sanitize(target_name)}.{_sanitize(model_name)}")
    backends = list(getattr(config, "pareto_plot_backends", ["plotly", "matplotlib"]))
    formats_by_backend = dict(getattr(config, "pareto_plot_formats", {"plotly": ["html"], "matplotlib": ["png"]}))

    saved_paths: dict[str, str] = {}
    title = f"Slice-stable ES Pareto frontier - target={target_name} model={model_name} K={len(shard_history[0])}"

    for backend in backends:
        for fmt in formats_by_backend.get(backend, []):
            out_path = f"{base_stem}.{backend}.{fmt}"
            try:
                if backend == "matplotlib":
                    _render_matplotlib(out_path, means, stds, pareto_idx, best_iter, alt_picks,
                                        title=title, monitor_metric_name=monitor_metric_name,
                                        direction=direction)
                elif backend == "plotly":
                    _render_plotly(out_path, means, stds, pareto_idx, best_iter, alt_picks,
                                    title=title, monitor_metric_name=monitor_metric_name,
                                    direction=direction)
                else:
                    logger.warning("Unknown Pareto plot backend %r; skipping", backend)
                    continue
                saved_paths[backend] = out_path
            except Exception as exc:
                logger.warning(
                    "slice-stable Pareto plot %s/%s failed for target=%s model=%s: %s",
                    backend, fmt, target_name, model_name, exc,
                )

    persist_history = bool(getattr(config, "pareto_persist_shard_history", False))
    shard_history_path: str | None = None
    if persist_history:
        try:
            shard_history_path = _persist_shard_history(
                base_stem, shard_history, monitor_metric_name=monitor_metric_name,
            )
        except Exception as exc:
            logger.warning("slice-stable shard_history dump failed: %s", exc)

    meta = {
        "best_iter": int(best_iter),
        "n_iters": int(n_iters),
        "pareto_frontier_size": len(pareto_idx),
        "pareto_frontier_indices": list(map(int, pareto_idx)),
        "alternative_selections": alt_picks,
        "pareto_plot_paths": saved_paths,
        "shard_history_path": shard_history_path,
    }
    if agg_history and 0 <= int(best_iter) < len(agg_history):
        meta["slice_aggregate_at_best"] = float(agg_history[int(best_iter)])

    # Register on ctx.metadata, mirroring the fairness_report pattern.
    md = getattr(ctx, "metadata", None)
    if isinstance(md, dict):
        md.setdefault("slice_stable_es", {}).setdefault(target_name, {})[model_name] = meta
    return meta


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def _render_matplotlib(
    path: str, means: np.ndarray, stds: np.ndarray, pareto_idx: list[int],
    best_iter: int, alt_picks: dict[str, int], *,
    title: str, monitor_metric_name: str, direction: str,
) -> None:
    """matplotlib backend. Uses Agg to stay headless."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    iters = np.arange(len(means))
    sc = ax.scatter(means, stds, c=iters, cmap="viridis", s=18, alpha=0.65,
                    edgecolors="none", label="all iterations")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("iteration")

    if pareto_idx:
        pf_idx = np.asarray(pareto_idx, dtype=int)
        ax.plot(means[pf_idx], stds[pf_idx], "-", color="black", linewidth=1.0, alpha=0.5,
                zorder=2, label="Pareto frontier")
        ax.scatter(means[pf_idx], stds[pf_idx], s=60, facecolors="none", edgecolors="black",
                    linewidths=1.0, zorder=3, label="non-dominated")

    if 0 <= int(best_iter) < len(means):
        ax.scatter([means[best_iter]], [stds[best_iter]], marker="X", s=180, c="crimson",
                    zorder=4, label=f"best_iter={best_iter}")
        ax.annotate(
            f"iter={best_iter}\nmean={means[best_iter]:.4f}\nstd={stds[best_iter]:.4f}",
            xy=(means[best_iter], stds[best_iter]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="crimson",
            bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.75, pad=2.0),
        )

    for q_str, idx in alt_picks.items():
        if 0 <= int(idx) < len(means) and int(idx) != int(best_iter):
            ax.scatter([means[idx]], [stds[idx]], marker="o", s=70, facecolors="none",
                        edgecolors="orange", linewidths=1.5, zorder=3)
            ax.annotate(f"q={q_str}", xy=(means[idx], stds[idx]),
                        xytext=(6, -10), textcoords="offset points",
                        fontsize=7, color="orange")

    direction_arrow = "->" if direction == "min" else "<-"
    ax.set_xlabel(f"mean across shards ({monitor_metric_name}, lower is better {direction_arrow})"
                  if direction == "min"
                  else f"mean across shards ({monitor_metric_name}, higher is better)")
    ax.set_ylabel("std across shards (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    # tmp suffix preserves the real file extension so matplotlib can infer the backend format.
    tmp_path = f"{path}.tmp{os.path.splitext(path)[1]}"
    fig.savefig(tmp_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp_path, path)


def _render_plotly(
    path: str, means: np.ndarray, stds: np.ndarray, pareto_idx: list[int],
    best_iter: int, alt_picks: dict[str, int], *,
    title: str, monitor_metric_name: str, direction: str,
) -> None:
    """plotly backend (HTML output for interactivity)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    iters = np.arange(len(means))
    fig.add_trace(go.Scatter(
        x=means, y=stds, mode="markers",
        marker=dict(color=iters.tolist(), colorscale="Viridis", size=7,
                    showscale=True, colorbar=dict(title="iteration"),
                    line=dict(width=0)),
        text=[f"iter={i}" for i in iters], hovertemplate="%{text}<br>mean=%{x:.4f}<br>std=%{y:.4f}<extra></extra>",
        name="all iterations",
    ))
    if pareto_idx:
        pf_idx = np.asarray(pareto_idx, dtype=int)
        fig.add_trace(go.Scatter(
            x=means[pf_idx], y=stds[pf_idx], mode="lines+markers",
            line=dict(color="rgba(0,0,0,0.5)", width=1.2),
            marker=dict(color="rgba(0,0,0,0)", size=14, line=dict(color="black", width=1.2)),
            name="Pareto frontier",
            hovertemplate="iter=%{customdata}<br>mean=%{x:.4f}<br>std=%{y:.4f}<extra></extra>",
            customdata=pf_idx.tolist(),
        ))
    if 0 <= int(best_iter) < len(means):
        fig.add_trace(go.Scatter(
            x=[means[best_iter]], y=[stds[best_iter]], mode="markers+text",
            marker=dict(symbol="x", color="crimson", size=18, line=dict(width=2)),
            text=[f"best_iter={best_iter}"],
            textposition="top right", textfont=dict(color="crimson"),
            name=f"selected best_iter={best_iter}",
        ))
    if alt_picks:
        alt_idx = [i for i in alt_picks.values() if 0 <= int(i) < len(means) and int(i) != int(best_iter)]
        if alt_idx:
            alt_labels = [k for k, v in alt_picks.items() if v in alt_idx]
            fig.add_trace(go.Scatter(
                x=[means[i] for i in alt_idx],
                y=[stds[i] for i in alt_idx],
                mode="markers+text",
                marker=dict(symbol="circle-open", color="orange", size=12, line=dict(width=2)),
                text=[f"q={lbl}" for lbl in alt_labels],
                textposition="bottom right", textfont=dict(color="orange", size=10),
                name="alt risk-quantiles",
            ))

    direction_arrow = "lower is better" if direction == "min" else "higher is better"
    fig.update_layout(
        title=title,
        xaxis_title=f"mean across shards ({monitor_metric_name}, {direction_arrow})",
        yaxis_title="std across shards (lower is better)",
        template="plotly_white",
        height=600, width=950,
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.7)"),
    )
    tmp_path = f"{path}.tmp{os.path.splitext(path)[1]}"
    fig.write_html(tmp_path, include_plotlyjs="cdn")
    os.replace(tmp_path, path)


def _persist_shard_history(
    base_stem: str, shard_history: Sequence[Sequence[float]], *,
    monitor_metric_name: str,
) -> str:
    """Dump the full ``(iter, shard_idx, score)`` table to parquet next to the plot."""
    import pandas as pd

    records = []
    for it, scores in enumerate(shard_history):
        for sh, v in enumerate(scores):
            records.append({"iter": it, "shard_idx": sh, "score": float(v),
                            "metric_name": monitor_metric_name})
    df = pd.DataFrame.from_records(records)
    path = f"{base_stem}.shard_history.parquet"
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path
