"""Qubit spectroscopy versus flux with a detuned (re-centered) frequency window.

This is a variant of ``09_qubit_spectroscopy_vs_flux`` whose only behavioural
difference is the XY-drive frequency window:

- Node 09 sweeps the qubit drive symmetrically around the bare GE frequency
  (detuning 0). Because the transmon frequency only bends DOWNWARD with flux,
  the upper half of that window is structurally empty noise.
- Node 09b centers the window ``detuning_in_mhz`` BELOW the GE frequency (a
  user-supplied value, since the flux-arch curvature is not yet known at this
  early stage -- this node is what measures it), so a much narrower span lands
  the full arch in the middle of the map.

When the requested detuning would push the intermediate frequency past the
+-400 MHz limit, the node temporarily shifts each affected qubit's upconverter
(LO) by the window center, plays the re-referenced IF, and reverts the LO/RF
state *before* analysis -- so the fit and state update always reference the
true bare GE frequency.

It uses a dedicated, cloned utils package (``qubit_spectroscopy_vs_flux_b``) so
it can be run side-by-side with node 09 for a clean before/after comparison.
"""

# %% {Imports}
import math
import warnings
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
from quam_config import Quam
from calibration_utils.qubit_spectroscopy_vs_flux_b import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qualibration_libs.core import tracked_updates
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        QUBIT SPECTROSCOPY VERSUS FLUX (detuned / re-centered window)
This sequence does a qubit spectroscopy for several flux biases to exhibit the qubit frequency
versus flux response, but -- unlike node 09 -- it centers the XY-drive sweep `detuning_in_mhz`
below the bare GE frequency so the downward flux arch fills the map instead of the upper half
being wasted empty noise.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having specified the desired flux point (qubit.z.flux_point).

State update:
    - The qubit 0->1 frequency at the set flux point: qubit.f_01 & qubit.xy.RF_frequency
    - The flux bias corresponding to the set flux point: q.z.independent_offset or q.z.joint_offset.
"""


node = QualibrationNode[Parameters, Quam](
    name="09b_qubit_spectroscopy_vs_flux",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)



# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Check if the qubits have a z-line attached
    if any(q.z is None for q in qubits):
        warnings.warn("Found qubits without a flux line. Skipping")

    operation = node.parameters.operation  # The qubit operation to play
    n_avg = node.parameters.num_shots
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
    # Qubit detuning sweep. Unlike node 09 (symmetric around the GE frequency), the window is
    # centered `detuning_in_mhz` BELOW the GE frequency so the downward flux arch is centered.
    # `dfs` stays referenced to the bare GE frequency (detuning 0 == GE), so the analysis below
    # is unchanged: it reconstructs the true RF as detuning + qubit.xy.RF_frequency.
    center = node.parameters.detuning_in_mhz * u.MHz
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-center - span / 2, -center + span / 2 + step / 2, step, dtype=np.int32)
    # Flux bias sweep in V
    flux_span = node.parameters.flux_offset_span_in_v * u.V
    num = node.parameters.num_flux_points
    dcs = np.linspace(-flux_span / 2, +flux_span / 2, num)
    # Buffer time for flux to wait (in unit of clock cycle)
    settle_t = node.parameters.settle_time_in_ns // 4
    buffer_t = node.parameters.buffer_time_in_ns // 4

    # --- LO/IF guard ---------------------------------------------------------
    # A detuned window can push the intermediate frequency past the +-400 MHz limit.
    # When the lowest sweep point would, shift that qubit's upconverter (LO) by the
    # window center, play the re-referenced IF (df + IF - if_update), and track the
    # change so it can be reverted before analysis (see analyse_data / save_results).
    detuning_hz = int((dfs.min() + dfs.max()) / 2)  # window center (= -center)
    if_update = []
    tracked_qubits = []
    for i, q in enumerate(qubits):
        if (q.xy.intermediate_frequency + int(dfs.min())) < -400e6:
            warnings.warn(
                f"{q.name}: detuned window pushes the IF below -400 MHz; shifting the "
                f"upconverter (LO) by {detuning_hz / 1e6:.1f} MHz for this scan (reverted after)."
            )
            if_update.append(detuning_hz)
            # Track the LO and RF changes so they revert later (keeps IF unchanged).
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                lo_frequency = q_upd.xy.opx_output.upconverter_frequency + detuning_hz
                if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                if (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                print(f"Updating {q_upd.name} LO to {lo_frequency}")
                q_upd.xy.opx_output.upconverter_frequency = lo_frequency
                q_upd.xy.RF_frequency += detuning_hz
                tracked_qubits.append(q_upd)
        elif (q.xy.intermediate_frequency + int(dfs.max())) > 400e6:
            warnings.warn(
                f"{q.name}: detuned window pushes the IF above +400 MHz. Reduce the span or use a "
                f"smaller/positive detuning_in_mhz. Proceeding without an LO shift (may clip)."
            )
            if_update.append(0)
        else:
            if_update.append(0)
    node.namespace["if_update"] = if_update
    node.namespace["tracked_qubits"] = tracked_qubits

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "flux bias", "units": "V"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency
        dc = declare(fixed)  # QUA variable for the flux dc level

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # Update each qubit's frequency. Done per-qubit (so multiplexed runs are
                    # correct, unlike node 09 which only updated the last qubit) and subtracting
                    # the per-qubit LO shift so the played RF is df above/below the GE frequency.
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency - if_update[i])
                    with for_(*from_array(dc, dcs)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            # Wait for the qubits to decay to the ground state
                            qubit.reset_qubit_thermal()
                            # Flux sweeping for a qubit
                            duration = (
                                operation_len * u.ns
                                if operation_len is not None
                                else qubit.xy.operations[operation].length * u.ns
                            ) // 4
                        align()

                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            # Bring the qubit to the desired point during the saturation pulse
                            qubit.z.play(
                                "const",
                                amplitude_scale=dc / qubit.z.operations["const"].amplitude,
                                duration=duration + settle_t + buffer_t,
                            )
                            # Wait for the qubit to settle (in units of clock cycle)
                            qubit.xy.wait(settle_t)
                            # Apply saturation pulse to all qubits
                            qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration,
                            )
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

            # Measure sequentially
            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    # load_from_id rebuilds node.parameters from the SAVED run, which would revert any
    # re-fit / analysis knob the user changed to re-analyse loaded data. Snapshot the
    # user's current values for those knobs (+ load_data_id) and restore them after load.
    _refit_keep = {k: getattr(node.parameters, k) for k in (
        "load_data_id",
        "input_line_impedance_in_ohm",
        "line_attenuation_in_db",
        "target_flux",
    )}
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    for _k, _v in _refit_keep.items():
        setattr(node.parameters, _k, _v)
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)
    # ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    # ds_processed.IQ_abs.plot()


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary."""
    # The LO/RF shift (if any) was only needed during acquisition. Revert it now so the
    # fit and state update reference the true bare GE frequency (detuning axis is GE-referenced).
    for q_upd in node.namespace.get("tracked_qubits", []):
        try:
            q_upd.revert_changes()
        except Exception:  # noqa: BLE001
            pass
    node.namespace["tracked_qubits"] = []

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
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    half_span = node.parameters.flux_offset_span_in_v / 2
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_results = node.results["fit_results"][q.name]
            idle_offset = fit_results.get("idle_offset", float("nan"))
            qubit_frequency = fit_results.get("qubit_frequency", float("nan"))
            if math.isnan(idle_offset) or abs(idle_offset) > half_span:
                node.log(
                    f"Skipping state update for {q.name}: "
                    f"idle_offset={idle_offset} V is out of scan range "
                    f"(+-{half_span} V)."
                )
            elif math.isnan(qubit_frequency):
                node.log(
                    f"Skipping state update for {q.name}: qubit_frequency is NaN."
                )
            else:
                if q.z.flux_point == "independent":
                    q.z.independent_offset = idle_offset
                elif q.z.flux_point == "joint":
                    q.z.joint_offset += idle_offset
                q.xy.RF_frequency = qubit_frequency
                q.f_01 = qubit_frequency
            for key in [
                "upper_sweet_spot_flux",
                "upper_sweet_spot_frequency",
                "lower_sweet_spot_flux",
                "lower_sweet_spot_frequency",
            ]:
                val = fit_results.get(key, float("nan"))
                if not math.isnan(val):
                    q.extras[key] = val


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    # Safety net: revert any LO/RF shift that survived (e.g. the simulate path, where
    # analyse_data is skipped). On the normal path analyse_data already cleared the list.
    for q_upd in node.namespace.get("tracked_qubits", []):
        try:
            q_upd.revert_changes()
        except Exception:  # noqa: BLE001
            pass
    node.save()
