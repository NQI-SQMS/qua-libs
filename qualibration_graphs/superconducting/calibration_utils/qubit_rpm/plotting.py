"""Plotting utilities for the qubit RPM (Rabi Population Measurement) experiment."""
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from calibration_utils.qubit_rpm.analysis import _cosine, CHI2_THRESHOLD


def plot_rpm(
    ds_g: xr.Dataset,
    ds_e: xr.Dataset,
    qubits,
    fit_results: dict,
) -> Figure:
    """Plot both RPM sweeps ('g' and 'e') with fits, errors, and chi² annotation."""
    grid = QubitGrid(ds_g, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        q_name = qubit["qubit"]
        _plot_single(
            ax,
            ds_g.sel(qubit=q_name),
            ds_e.sel(qubit=q_name),
            fit_results.get(q_name),
        )
    grid.fig.suptitle("Qubit RPM — Thermal Population Measurement")
    grid.fig.set_size_inches(10, 6)
    grid.fig.tight_layout()
    return grid.fig


def _plot_single(ax, ds_g, ds_e, fit_params):
    x = ds_g.amp_factor.values.astype(float)

    sig = "state" if "state" in ds_g.data_vars else "I"
    scale = 1.0 if sig == "state" else 1e3
    unit_str = "" if sig == "state" else " (mV)"

    y_g = getattr(ds_g, sig).values.astype(float) * scale
    y_e = getattr(ds_e, sig).values.astype(float) * scale

    # ── Raw data ──────────────────────────────────────────────────────────────
    ax.plot(x, y_g, "o", ms=3, color="C0", alpha=0.6, label="from |g⟩")
    ax.plot(x, y_e, "s", ms=3, color="C1", alpha=0.6, label="from thermal")

    # ── Fitted curves ─────────────────────────────────────────────────────────
    if fit_params is not None and fit_params.amp_g > 0 and np.isfinite(fit_params.freq_g):
        x_dense = np.linspace(x[0], x[-1], 500)

        y_g_fit = _cosine(x_dense, fit_params.amp_g, fit_params.freq_g,
                          fit_params.phi_g, fit_params.offset_g) * scale
        ax.plot(x_dense, y_g_fit, "-", lw=1.5, color="C0", label="g fit")

        if np.isfinite(fit_params.amp_e) and np.isfinite(fit_params.freq_e):
            y_e_fit = _cosine(x_dense, fit_params.amp_e, fit_params.freq_e,
                              fit_params.phi_e, fit_params.offset_e) * scale
            ax.plot(x_dense, y_e_fit, "-", lw=1.5, color="C1", label="e fit")

    # ── Annotation box ────────────────────────────────────────────────────────
    if fit_params is not None:
        lines = _build_annotation(fit_params)
        color = "lightyellow" if fit_params.success else "#ffe0e0"
        ax.text(
            0.98, 0.05,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="gray", alpha=0.9),
        )

    ax.set_xlabel("EF amplitude scale factor")
    ax.set_ylabel(f"State population{unit_str}")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.05, 1.15)


def _build_annotation(fp) -> list:
    """Build the list of annotation strings for one qubit panel."""
    lines = []

    # Thermal population
    p = fp.thermal_population
    p_err = fp.thermal_population_err
    if np.isfinite(p):
        if np.isfinite(p_err):
            lines.append(f"P_th = ({100*p:.2f} ± {100*p_err:.2f}) %")
        else:
            lines.append(f"P_th = {100*p:.2f} %")
    else:
        lines.append("P_th = N/A")

    # Effective temperature
    t = fp.effective_temperature_mk
    t_err = fp.effective_temperature_mk_err
    if np.isfinite(t):
        if np.isfinite(t_err):
            lines.append(f"T_eff = ({t:.1f} ± {t_err:.1f}) mK")
        else:
            lines.append(f"T_eff = {t:.1f} mK")
    else:
        lines.append("T_eff = N/A")

    # chi² quality indicators
    c2g = fp.chi2_g
    if np.isfinite(c2g):
        flag = "" if c2g <= CHI2_THRESHOLD else " ✗"
        lines.append(f"χ²_g = {c2g:.3f}{flag}")
    c2e = fp.chi2_e
    if np.isfinite(c2e):
        lines.append(f"χ²_e = {c2e:.3f} (info)")

    # Overall status
    lines.append("● PASS" if fp.success else "● FAIL")

    return lines
