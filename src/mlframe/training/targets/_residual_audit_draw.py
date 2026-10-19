"""Stages of ``plot_residual_diagnostics``."""

from __future__ import annotations

# --- imports (managed) ---
import math
import numpy as np
# --- end imports ---


def _draw_residual_axes(ax_hist, plot_resid, audit, ax_resid_vs_pred, plot_pred):
    """Stage of ``plot_residual_diagnostics``."""
    if ax_hist is not None:
        n_bins = max(20, min(80, int(math.sqrt(plot_resid.size))))
        ax_hist.hist(plot_resid, bins=n_bins, alpha=0.6, color="steelblue", edgecolor="white", linewidth=0.4, density=True)
        # Overlay fitted Normal density (the baseline assumption). Guard
        # against constant-residual inputs: when min == max, linspace
        # collapses to a single point and the normal PDF is meaningless.
        # The audit.std > 0 check covered most cases but ``residuals.std``
        # uses ddof=1 on a SAMPLED subset, so the plot-time residual array
        # can still be all-identical even when the population std is nonzero.
        _r_min = float(plot_resid.min())
        _r_max = float(plot_resid.max())
        if audit.std > 0 and _r_max > _r_min:
            x_grid = np.linspace(_r_min, _r_max, 200)
            normal_pdf = 1 / (audit.std * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x_grid - audit.mean) / audit.std) ** 2)
            ax_hist.plot(x_grid, normal_pdf, color="red", linestyle="--", linewidth=1.4, label=f"Normal(mu={audit.mean:.2g}, sigma={audit.std:.2g})")
            ax_hist.legend(loc="best", fontsize=8, framealpha=0.7)
        ax_hist.axvline(0, color="green", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_hist.set_xlabel("Residual (y_true - y_pred)")
        ax_hist.set_ylabel("Density")
        # Residual hypothesis + suggested loss now live on the
        # histogram title (was previously appended to the scatter title,
        # crowding it). Self-contained: skew/kurt -> hypothesis -> suggested
        # loss reads top-to-bottom on the same panel that visualises the
        # residual distribution.
        _suggested = audit.suggested_loss.split("(")[0].strip() if audit.suggested_loss else ""
        _hyp_line = f"hypothesis: {audit.hypothesis}"
        if _suggested:
            _hyp_line += f" (suggested: {_suggested})"
        ax_hist.set_title(f"Residuals (skew={audit.skew:+.2f}, excess_kurt={audit.excess_kurt:+.2f})\n" f"{_hyp_line}")
        ax_hist.grid(True, alpha=0.3)

    # Residuals vs predicted (heteroscedasticity panel)
    if ax_resid_vs_pred is not None:
        ax_resid_vs_pred.scatter(plot_pred, plot_resid, alpha=0.3, s=10, color="steelblue")
        ax_resid_vs_pred.axhline(0, color="green", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_resid_vs_pred.set_xlabel("Predicted (y_hat)")
        ax_resid_vs_pred.set_ylabel("Residual")
        het_marker = "(!) heteroscedastic" if audit.hetero_significant else "homoscedastic"
        ax_resid_vs_pred.set_title(f"Residuals vs predicted ({het_marker})\n" f"spearman(|resid|, y_hat) = {audit.hetero_spearman:+.3f}")
        ax_resid_vs_pred.grid(True, alpha=0.3)
