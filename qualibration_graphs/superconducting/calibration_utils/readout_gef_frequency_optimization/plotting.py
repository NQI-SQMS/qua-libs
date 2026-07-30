from typing import List, Optional

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_distances_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset) -> Figure:
    """Plot GEF discrimination fidelity vs readout frequency with the optimum marked."""
    qubit_map = {q.name: q for q in qubits}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_distance_with_fit(
            ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), qubit_map.get(qubit["qubit"])
        )
    grid.fig.suptitle("GEF readout frequency optimisation — fidelity")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_distance_with_fit(
    ax: Axes,
    ds: xr.Dataset,
    qubit: dict[str, str],
    fit: xr.Dataset = None,
    qubit_obj: Optional[AnyTransmon] = None,
):
    q_name = qubit["qubit"]
    ds_q = ds.sel(qubit=q_name)
    freqs_mhz = ds_q.coords["frequency"].values / 1e6

    ax.plot(freqs_mhz, 100 * ds_q.fidelity_gef.values, color="C0", lw=2, label="GEF fidelity")
    if fit is not None and "optimal_detuning" in fit.coords:
        opt = float(fit.coords["optimal_detuning"])
        opt_fid = float(ds_q.fidelity_gef.sel(frequency=opt, method="nearest").values)
        ax.axvline(opt / 1e6, color="r", ls="--", lw=1.5,
                   label=f"opt {opt/1e6:.3f} MHz ({opt_fid*100:.1f}%)")
    ax.set_xlabel("Detuning from current GEF frequency [MHz]")
    ax.set_ylabel("GEF fidelity [%]")
    ax.set_title(q_name)
    ax.legend(fontsize=8)

    if qubit_obj is not None:
        gef_shift = qubit_obj.resonator.GEF_frequency_shift or 0
        center_ghz = (qubit_obj.resonator.RF_frequency + gef_shift) / 1e9
        ax2 = ax.secondary_xaxis(
            "top",
            functions=(
                lambda x, c=center_ghz: x / 1e3 + c,
                lambda x, c=center_ghz: (x - c) * 1e3,
            ),
        )
        ax2.set_xlabel("RF frequency [GHz]")
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
