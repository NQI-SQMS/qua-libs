from typing import List
import xarray as xr
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_fidelity_vs_length(
    ds_fit: xr.Dataset, qubits: List[AnyTransmon], fit_results: dict
):
    """Plot readout fidelity vs cumulative readout length for each qubit."""
    grid = QubitGrid(ds_fit, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        q = qubit["qubit"]
        r = fit_results[q]

        lengths = ds_fit.sel(qubit=q).readout_length_ns.values
        fidelities = ds_fit.sel(qubit=q).fidelity.values

        # BUG FIX: two_state_discriminator returns fidelity already in percent [0, 100].
        # Previously this was multiplied by 100 again, producing values like 9000% instead of 90%.
        ax.plot(lengths, fidelities, "b-", linewidth=1.5, label="Fidelity")
        ax.axvline(
            r["optimal_readout_length_ns"],
            color="r",
            linestyle="--",
            label=f"Opt. {r['optimal_readout_length_ns']} ns\n({r['optimal_fidelity']:.1f}%)",
        )
        ax.set_xlabel("Readout length [ns]")
        ax.set_ylabel("Fidelity [%]")
        ax.set_title(q)
        ax.legend(fontsize=8)

    grid.fig.suptitle("Readout length optimization")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
