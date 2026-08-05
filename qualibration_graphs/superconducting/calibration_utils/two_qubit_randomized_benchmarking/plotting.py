from typing import List, Literal
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from uncertainties import ufloat

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from calibration_utils.two_qubit_randomized_benchmarking.plot_tools import (
    QubitPairGrid,
    grid_pair_names,
)
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair

from calibration_utils.common_utils.fitting_tools import power_law

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(
    ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset, include_raw: bool = True
) -> List[Figure]:
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    figs = []

    # Plot raw data
    if include_raw:
        fig = plot_raw_data(ds, qubit_pairs)
        figs.append(fig)

    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    # grid = QubitPairGrid(ds, [q.grid_location for q in qubit_pairs])
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qp, fits.sel(qubit_pair=qp["qubit"]))

    fig = grid.fig
    fig.suptitle("Two qubit randomized benchmarking", fontsize=16)
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    figs.append(fig)

    return figs


def plot_raw_data(ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair]) -> Figure:
    colors = {"00": "C0", "01": "C1", "10": "C2", "11": "C3"}

    ds_mean = ds.mean("nb_of_sequences")
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp_dict in grid_iter(grid):
        # for ax, qp in zip(axes, qps):
        qp = qp_dict["qubit"]
        dqp = ds.sel(qubit_pair=qp)
        mqp = ds_mean.sel(qubit_pair=qp)

        x = dqp.depths.values  # (D,)
        for ms in ds.measured_state.values:
            # Y = dqp.state.sel(measured_state=ms).values  # (Nseq, D)
            # # add shift depending on measured state for better visibility
            # x_shift = [-0.2, 0.1, 0.1, 0.2]
            # X = np.tile(x, (Y.shape[1], 1)) + x_shift[int(ms, 2)]
            # ax.scatter(
            #     X.ravel(),
            #     Y.ravel(),
            #     s=12,
            #     alpha=0.3,
            #     marker=".",
            #     color=colors[ms],
            #     label=f"$|{ms}\\rangle$",
            # )

            y_mean = mqp.state.sel(measured_state=ms).values  # (D,)
            ax.plot(x, y_mean, ".-", lw=2, color=colors[ms])

            # violin plot of all data points at each depth
            y = dqp.state.sel(measured_state=ms).values
            # workaround for weird violinplot behavior
            try:
                parts = ax.violinplot(
                    y,
                    positions=x,
                    widths=0.8,
                    showmeans=True,
                    showmedians=False,
                    showextrema=True,
                )
            except ValueError:
                print("Violin plot failed, trying transpose")
                parts = ax.violinplot(
                    y.T,
                    positions=x,
                    widths=0.8,
                    showmeans=True,
                    showmedians=False,
                    showextrema=True,
                )

            # bodies
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[ms])
                pc.set_edgecolor(colors[ms])
                pc.set_alpha(0.2)

            # lines: 'cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'
            for k in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
                if k in parts:
                    parts[k].set_color(colors[ms])
                    parts[k].set_linewidth(1)
                    parts[k].set_alpha(0.8)

        legend_handles = [
            Patch(facecolor=color, edgecolor=color, alpha=0.25, label=rf"$|{ms}\rangle$")
            for ms, color in colors.items()
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=2)

        ax.axhline(0.25, color="gray", linewidth=4, alpha=0.2)
        ax.set_title(f"qubit_pair = {qp}")
        ax.set_ylim(-0.05, 1.05)

    # ylabel only on left column
    for grid_row in grid.axes:
        grid_row[0].set_ylabel(r"State probability P($|00\rangle$)")
    # xlabel only on bottom row
    for last_row_ax in grid.all_axes[-1]:
        last_row_ax.set_xlabel("Clifford depth")

    # one legend
    # handles, labels = grid.axes[0][0].get_legend_handles_labels()
    # grid.fig.legend(handles, labels, loc="upper right", ncol=4)
    grid.fig.suptitle("Raw data", fontsize=16)
    # grid.fig.suptitle("Raw data", x=0.25, horizontalalignment="left", fontsize=16)

    grid.fig.set_size_inches(10, 8)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(
    ax: Axes, ds: xr.Dataset, qubit_pair: dict[str, str], fit: xr.Dataset = None, show_legend: bool = True
):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    ax.errorbar(
        fit.depths,
        fit.data_mean,
        yerr=fit.data_sem,
        fmt=".",
        capsize=2,
        elinewidth=0.5,
    )
    ax.axhline(0.25, color="gray", linewidth=4, alpha=0.2)
    ax.set_title(qubit_pair["qubit"], pad=22)
    ax.set_xlabel("Clifford depth")
    ax.set_ylabel(r"State probability P($|00\rangle$)")
    ax.set_ylim(-0.05, 1.05)

    # Fitted decay
    try:
        if "fit_data" not in fit:
            return
        max_depths = np.log(0.5) / np.log(fit.fit_data.sel(param="p").values)
        max_depths = max(max_depths, fit.depths.max().item() * 1.05)
        fit_depths = np.linspace(0, max_depths, 100)
        fitted = power_law(
            fit_depths,
            fit.fit_data.sel(param="p").values,
            fit.fit_data.sel(param="A").values,
            fit.fit_data.sel(param="B").values,
        )
        ax.plot(fit_depths, fitted, "r--")
        if show_legend:
            text = []
            # Annotate with coherence limit
            text.append(f"Coherence limit = {fit.coherence_limit.values:.3e}\n")
            # Annotate with EPC and fidelity
            epc = ufloat(fit.error_per_clifford.values, fit.error_per_clifford_sem.values)
            text.append(f"2Q RB fidelity = {100 * (1 - epc):.3f}%")
            text.append(f"EPC = {epc:.3e}")
            if "epg_eval_method" in fit.keys() and fit.epg_eval_method != "N/A":
                epg = ufloat(fit.error_per_gate.values, fit.error_per_gate_sem.values)
                text.append(f"EPG (2Q gate) = {epg:.3e} [{fit.epg_eval_method.values}]")
                text.append(f"EPG/coherence limit = {epg.nominal_value / fit.coherence_limit.values:.2%}")
            ax.plot([], label="\n".join(text))
            # Remove the handle from the legend box.
            ax.legend(handlelength=0)
    except Exception as e:
        print(f"Could not plot fit for {qubit_pair['qubit']}: {e}")


def plot_grid(ds: xr.Dataset, qubit_pairs: List[AnyTransmonPair], fits: xr.Dataset, grid_dims=("p1", "p2")):
    p1, p2 = grid_dims
    figs = []

    for qp in qubit_pairs:
        qp_name = qp.name
        qp_dict = {"qubit": qp_name}
        fit_grid = fits.sel(qubit_pair=qp_name)
        p1_vals = fit_grid.coords[p1].values
        p2_vals = fit_grid.coords[p2].values
        p1_len = len(p1_vals)
        p2_len = len(p2_vals)

        fig, axs = plt.subplots(
            p1_len,
            p2_len,
            sharex=True,
            sharey=True,
            figsize=(2 * p2_len, 1.5 * p1_len),
            constrained_layout=True,
        )
        fig.supylabel(fit_grid.coords[p1].long_name)
        fig.supxlabel(fit_grid.coords[p2].long_name)
        fig.suptitle(f"Parameter scan 2QRB for Qubit Pair {qp_name}")
        axs = np.atleast_2d(axs)

        for i, p1v in enumerate(p1_vals):
            for j, p2v in enumerate(p2_vals):
                fit_ij = fit_grid.sel(**{p1: p1v, p2: p2v})
                plot_individual_data_with_fit(axs[i, j], ds, qp_dict, fit=fit_ij, show_legend=False)
                # remove title, xlabel, ylabel for cleaner look
                axs[i, j].set_title("")
                if i == p1_len - 1:
                    axs[i, j].set_xlabel(f"{p2}={p2v:.3f}")
                else:
                    axs[i, j].set_xlabel("")
                if j == 0:
                    axs[i, j].set_ylabel(f"{p1}={p1v:.3f}")
                else:
                    axs[i, j].set_ylabel("")

        height = fig.get_figheight()
        top_in = 1.2
        bottom_in = 0.4
        top = 1 - top_in / height
        bottom = bottom_in / height
        fig.suptitle(f"Parameter scan 2QRB for Qubit Pair {qp_name}")
        fig.tight_layout(rect=[0, bottom, 1, top])
        figs.append(fig)

    return figs


def plot_data_with_best(da, best_definition: Literal["min", "max"], col="qubit_pair", marker="x", real_values=None):
    # check if log scale is doable
    if (da <= 0).any():
        use_log = False
    else:
        use_log = True

    # plot data
    max_ncols = min(3, da[col].values.size)
    if da.ndim == 2:
        if use_log:
            g = da.plot(col=col, col_wrap=max_ncols, yscale="log")
        else:
            g = da.plot(col=col, col_wrap=max_ncols)
    else:
        if use_log:
            g = da.plot(col=col, col_wrap=max_ncols, norm=LogNorm())
        else:
            g = da.plot(col=col, col_wrap=max_ncols)

    col_vals = da[col].values
    axes = np.ravel(g.axes)

    for ax, col_val in zip(axes, col_vals):
        dai = da.sel({col: col_val})

        vals = np.asarray(dai.values)
        if best_definition == "max":
            k = np.nanargmax(vals)
        elif best_definition == "min":
            k = np.nanargmin(vals)
        v0 = vals.flat[k]

        formatting = dict(marker=marker, markersize=6, mew=2, linestyle="None", color="r")
        if dai.ndim == 1:
            dim = dai.dims[0]
            x0 = np.asarray(dai[dim].values)[k]
            text = f"{best_definition} {da.name}={v0:.3g}\ninput: ({dim}={x0:.3g})"
            if real_values is not None:
                x0_real = real_values[dim][col_val][k]
                text += f"\nreal: ({dim}={x0_real})"
            ax.plot(x0, v0, label=text, **formatting)
        elif dai.ndim == 2:
            ydim, xdim = dai.dims
            iy, ix = np.unravel_index(k, vals.shape)
            x0 = np.asarray(dai[xdim].values)[ix]
            y0 = np.asarray(dai[ydim].values)[iy]
            text = f"{best_definition} {da.name}={v0:.3g}\ninput: ({ydim}={y0:.3g}, {xdim}={x0:.3g})"
            if real_values is not None:
                y0_real = real_values[ydim][col_val][iy]
                x0_real = real_values[xdim][col_val][ix]
                text += f"\nreal: ({ydim}={y0_real}, {xdim}={x0_real})"
            ax.plot(x0, y0, label=text, **formatting)

        ax.legend(loc="best", fontsize=8, handlelength=1)

    g.fig.set_size_inches(10, 8)
    g.fig.tight_layout()

    return g.fig
