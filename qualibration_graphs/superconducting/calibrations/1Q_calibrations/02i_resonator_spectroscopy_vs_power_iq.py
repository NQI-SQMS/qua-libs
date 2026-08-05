"""Resonator spectroscopy versus readout power — with per-power I/Q circle overlay.

Exploratory variant of ``1Q_05_resonator_spectroscopy_vs_power.py``. The QUA program and
analysis are identical; the only addition is a third figure (``iq_circles``) that overlays
one raw I/Q circle per readout power on a single axes per qubit. It imports from the
self-contained ``resonator_spectroscopy_vs_power_iq`` utils so that the production module
and node are left untouched.
"""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.core.utils.node.record_state_update import record_state_update, update_machine_attribute
from quam_config import Quam
from calibration_utils.resonator_spectroscopy_vs_power_iq import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_raw_data_amp_linear,
    plot_iq_circles_vs_power,
    plot_iq_circle_centers_vs_power,
    plot_dip_traces_vs_power,
    plot_normalized_complex_response,
    plot_quality_factors_vs_power,
)
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates
from calibration_utils.common_utils.plotting_tools import patch_fig_info


# %% {Readout-power helper — 1 dB grid, bypassing quam_builder's 3 dB validation}
# The installed quam_builder ``set_output_power`` still steps ``full_scale_power_dbm`` in
# 3 dB and validates the result against a 3 dB grid (np.arange(-11, 17, 3)). The MW-FEM
# hardware actually supports a 1 dB grid, so a state whose readout full-scale sits on the
# 1 dB grid (e.g. 0 dBm) makes ``set_output_power`` raise:
#     ValueError: Expected full_scale_power_dbm ... in steps of 3 dB, got 3.
# We therefore BYPASS it and set ``full_scale_power_dbm`` + the readout-operation amplitude
# directly, on the 1 dB grid — exactly the same approach already used by
# 08b_qubit_spectroscopy_vs_power.py. ``update_state`` below already works on the 1 dB grid.
_FULL_SCALE_DBM_MIN, _FULL_SCALE_DBM_MAX = -11, 16


def _on_grid_full_scale_dbm(power_in_dbm: float) -> int:
    """Smallest 1 dB-grid full-scale power >= ``power_in_dbm``, clamped to the allowed range."""
    return int(min(max(int(np.ceil(power_in_dbm)), _FULL_SCALE_DBM_MIN), _FULL_SCALE_DBM_MAX))


def set_readout_power_1db_grid(resonator, power_in_dbm: float, max_amplitude: float, operation: str = "readout"):
    """Drop-in replacement for ``resonator.set_output_power`` that uses the 1 dB grid.

    Mirrors ``set_output_power`` exactly (it sets ``operations[operation].amplitude`` and
    ``opx_output.full_scale_power_dbm`` and returns the same dict), but chooses the full-scale
    power on the 1 dB grid and writes it directly, so quam_builder's 3 dB-grid validation is
    never hit. Resonators sharing a readout line share the same port object, so writing the
    full-scale here behaves identically to the original call.
    """
    fs = _on_grid_full_scale_dbm(power_in_dbm - 20.0 * np.log10(max_amplitude))
    amp = calculate_voltage_scaling_factor(fixed_power_dBm=fs, target_power_dBm=power_in_dbm)
    resonator.operations[operation].amplitude = amp
    resonator.opx_output.full_scale_power_dbm = fs
    return {"full_scale_power_dbm": fs, "amplitude": amp}


# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS READOUT POWER (with I/Q circle overlay)
This sequence involves measuring the resonator by sending a readout pulse and
demodulating the signals to extract the 'I' and 'Q' quadratures for all resonators
simultaneously. This is done across various readout frequencies and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by
noting the resonator frequency splitting. This information can then be used to adjust
the readout amplitude, choosing a readout amplitude value just before the observed
frequency splitting.

In addition to the standard heatmaps, this node draws one raw I/Q circle per readout
power on a single axes per qubit (colour-coded by power [dBm]), so that the growth and
punch-out deformation of the resonator response is visible at a glance.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a_resonator_spectroscopy.py).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The readout frequency at the optimal readout power: qubit.resonator.f_01 & qubit.resonator.RF_frequency
    - The readout power: qubit.resonator.set_output_power()
    - The readout frequency for the optimal readout power.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="05b_resonator_spectroscopy_vs_power_iq",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine = Quam.load()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2", "q3"]
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Update the readout power to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_resonators"] = []
    set_power_amp_dict = None
    for i, qubit in enumerate(qubits):
        with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            # BYPASS quam_builder's 3 dB-grid set_output_power (see set_readout_power_1db_grid above):
            # set full_scale_power_dbm + readout amplitude directly on the 1 dB grid.
            set_power_amp_dict = set_readout_power_1db_grid(
                resonator,
                power_in_dbm=node.parameters.max_power_dbm,
                max_amplitude=node.parameters.max_amp,
            )
            node.namespace["tracked_resonators"].append(resonator)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
    amp_min = calculate_voltage_scaling_factor(node.parameters.max_power_dbm, node.parameters.min_power_dbm)
    amps = np.geomspace(amp_min, 1, node.parameters.num_power_points)
    # Provenance (improvement #2, ported from Soon 02b): record the actual full-scale power,
    # the base readout amplitude and the amplitude-scaling vector so that the chosen readout
    # power/amplitude is unambiguous in the saved data (the rich power/amp plots show "what to choose").
    if set_power_amp_dict is not None:
        node.results["readout_power_provenance"] = {
            "set_power_dbm": set_power_amp_dict["full_scale_power_dbm"],
            "set_amp": set_power_amp_dict["amplitude"],
            "amp_scaling": amps.tolist(),
        }
    power_dbm = np.linspace(
        node.parameters.min_power_dbm,
        node.parameters.max_power_dbm,
        node.parameters.num_power_points,
    )
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        "power": xr.DataArray(power_dbm, attrs={"long_name": "readout power", "units": "dBm"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        # For instance, here 'I' is a python list containing two QUA fixed variables.
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
                save(n, n_st)
                with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # QUA for_ loop for sweeping the readout amplitude
                        # with for_(*from_array(a, amps)):
                        with for_each_(a, amps):
                            # readout the resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                            # wait for the resonator to deplete
                            rr.wait(rr.depletion_time * u.ns)
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data,
    storing it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # load_from_id rebuilds node.parameters from the SAVED run; preserve the user's re-fit
    # knobs (offset / plot) so re-analysing loaded data with changed knobs takes effect.
    _refit_keep = {k: getattr(node.parameters, k) for k in (
        "load_data_id",
        "power_below_punchout_db",
        "num_iq_circles_to_plot",
    )}
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    for _k, _v in _refit_keep.items():
        setattr(node.parameters, _k, _v)
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary."""
    # TODO: requires manual setting of the readout power since the analysis isn't robust enough...
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_amp_linear = plot_raw_data_amp_linear(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_iq_circles = plot_iq_circles_vs_power(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        num_circles=node.parameters.num_iq_circles_to_plot,
    )
    fig_iq_centers = plot_iq_circle_centers_vs_power(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    fig_dip_traces = plot_dip_traces_vs_power(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        num_circles=node.parameters.num_iq_circles_to_plot,
        normalize=True,
    )
    fig_dip_traces_raw = plot_dip_traces_vs_power(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        num_circles=node.parameters.num_iq_circles_to_plot,
        normalize=False,
    )
    fig_normalized_complex = plot_normalized_complex_response(
        node.results["ds_raw"],
        node.namespace["qubits"],
        num_circles=node.parameters.num_iq_circles_to_plot,
    )
    fig_quality_factors = plot_quality_factors_vs_power(
        node.results["ds_raw"],
        node.namespace["qubits"],
    )
    patch_fig_info(node)  # stamp node/run provenance onto every figure (ported from Soon 02b)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "power_dbm": fig_raw_fit,
        "amplitude_linear": fig_amp_linear,
        "iq_circles": fig_iq_circles,
        "iq_circle_centers": fig_iq_centers,
        "dip_traces": fig_dip_traces,
        "dip_traces_raw": fig_dip_traces_raw,
        "normalized_complex": fig_normalized_complex,
        "quality_factors": fig_quality_factors,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Apply the fitted readout power per shared readout line (multiplexed-safe).

    Resonators on the same readout line share one ``full_scale_power_dbm`` (a referenced
    port). The analysis already chose one common full-scale per line (sized for the loudest
    tone, on the 1 dB grid) and a per-resonator amplitude. Here we:
      * change the shared full-scale once per line and record it manually (it is a port
        value that ``record_state_updates()`` would otherwise skip and silently persist,
        see 1Q_17_xyz_delay);
      * set each resonator's amplitude on that line - successful ones to their optimal
        power, the others (not measured / failed this run) rescaled to PRESERVE their
        current output power under the new full-scale (amplitude is auto-recorded);
      * update f_01 / RF_frequency for the successful resonators.
    """
    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    fit = node.results["fit_results"]

    # Group ALL machine resonators by their shared full-scale port reference (= readout line).
    lines: dict[str, list] = {}
    for q in node.machine.qubits.values():
        ref = q.resonator.opx_output.get_reference(attr="full_scale_power_dbm")
        lines.setdefault(ref, []).append(q)

    with node.record_state_updates():
        for ref, members in lines.items():
            # Successful, measured members define this line's common full-scale.
            succ = [q for q in members if q.name in fit and fit[q.name]["success"]]
            if not succ:
                continue
            new_fs = int(fit[succ[0].name]["target_full_scale_power_dbm"])
            old_fs = int(succ[0].resonator.opx_output.full_scale_power_dbm)

            # Shared full-scale: record once (revert-then-prompt). The live value is left at
            # old_fs so node.save() does not silently persist it; Accept applies new_fs.
            if new_fs != old_fs:
                attr_key = update_machine_attribute(node.machine, ref, old_fs)
                record_state_update(node, ref, str(attr_key), old_fs, new_fs)

            # Per-resonator amplitude on this line.
            for q in members:
                if q.name in fit and fit[q.name]["success"]:
                    q.resonator.operations["readout"].amplitude = float(fit[q.name]["target_amplitude"])
                    q.resonator.f_01 += fit[q.name]["frequency_shift"]
                    q.resonator.RF_frequency += fit[q.name]["frequency_shift"]
                elif new_fs != old_fs:
                    # Not measured / failed: preserve current output power under the new full-scale.
                    old_amp = q.resonator.operations["readout"].amplitude
                    q.resonator.operations["readout"].amplitude = float(old_amp * 10 ** ((old_fs - new_fs) / 20))

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
