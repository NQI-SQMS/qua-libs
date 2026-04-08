from typing import List

import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter


def plot_distances_vs_length(
    ds_fit: xr.Dataset, qubits: List, fit_results: dict
) -> Figure:
    """Plot D_ge, D_ef, D_gf and min-Distance vs cumulative readout length with optimum marked."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        q = qubit["qubit"]
        r = fit_results[q]
        ds_q = ds_fit.sel(qubit=q)
        lengths = ds_q.readout_length_ns.values

        ax.plot(lengths, 1e3 * ds_q.Dge.values, "--", color="C1", lw=1.5, label="D_ge")
        ax.plot(lengths, 1e3 * ds_q.Def.values, "--", color="C2", lw=1.5, label="D_ef")
        ax.plot(lengths, 1e3 * ds_q.Dgf.values, "--", color="C3", lw=1.5, label="D_gf")
        ax.plot(lengths, 1e3 * ds_q.Distance.values, color="C0", lw=2, label="min distance")

        opt = r["optimal_readout_length_ns"]
        opt_dist_mv = 1e3 * r["optimal_distance"]
        ax.axvline(opt, color="k", ls="--", lw=1.5,
                   label=f"opt = {opt} ns  ({opt_dist_mv:.1f} mV)")

        ax.set_xlabel("Readout length [ns]")
        ax.set_ylabel("IQ distance [mV]")
        ax.set_title(q)
        ax.legend(fontsize=7)

    grid.fig.suptitle("GEF readout length optimisation")
    grid.fig.set_size_inches(12, 6)
    grid.fig.tight_layout()
    return grid.fig
