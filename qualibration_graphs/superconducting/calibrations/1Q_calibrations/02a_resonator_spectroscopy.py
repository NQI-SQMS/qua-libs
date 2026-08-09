# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam, TemporaryCalibrationData
from qualibration_libs.core import tracked_updates
from calibration_utils.resonator_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude_with_fit,
    plot_raw_phase,
    plot_circle_fit_result,
)
from calibration_utils.error_codes import (
    ResonatorSpectroscopyErrorCode,
    ResonatorSpectroscopyCorrectiveAction,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        1D RESONATOR SPECTROSCOPY
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all the active qubits.
The data is then post-processed to determine the resonator resonance frequency.
This frequency is used to update the readout frequency in the state.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration, and the resonators depletion time.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The readout frequency: qubit.resonator.f_01 & qubit.resonator.RF_frequency
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02a_resonator_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    # Temporarily set the readout power if readout_power_dbm is specified.
    # The change is recorded for reversion at the end of update_state.
    node.namespace["tracked_resonators"] = []
    if node.parameters.readout_power_dbm is not None:
        for qubit in qubits:
            with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
                resonator.set_locked_output_power(power_in_dbm=node.parameters.readout_power_dbm)
                node.namespace["tracked_resonators"].append(resonator)
        node.log(
            f"Resonator spectroscopy: temporarily set readout power to "
            f"{node.parameters.readout_power_dbm:.1f} dBm (max_amp={node.parameters.max_amp})"
        )

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the resonator frequencies for all resonators
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to deplete
                        rr.wait(rr.depletion_time // 4)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
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
    fig_raw_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["qubits"])
    fig_fit_amplitude = plot_raw_amplitude_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "phase": fig_raw_phase,
        "amplitude": fig_fit_amplitude,
    }

    if node.parameters.run_circle_fit:
        fig_circle = plot_circle_fit_result(
            node.results["ds_raw"],
            node.namespace["qubits"],
            node.results["fit_results"],
            circle_fit_results=node.namespace.get("circle_fit_results"),
        )
        plt.show()
        node.results["figures"]["circle_fit"] = fig_circle


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful.

    On failure (NO_DIP_FOUND): adds the resonator's RF frequency to the blacklist
    in temp_calibration so that upstream broad spectroscopy can avoid it.
    """
    # Revert any temporary readout power change made in create_qua_program
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    machine = node.machine  # this is your Quam instance

    # Ensure the container exists
    if machine.temp_calibration is None:
        machine.temp_calibration = {}

    def _ensure_temp_calibration(machine, qubit_name: str):
        """Return the TemporaryCalibrationData for qubit_name, creating it if absent."""
        if machine.temp_calibration is None:
            machine.temp_calibration = {}
        if qubit_name not in machine.temp_calibration:
            machine.temp_calibration[qubit_name] = TemporaryCalibrationData()
        temp_data = machine.temp_calibration[qubit_name]
        # Backward-compatibility: add field if an older state.json omitted it
        if not hasattr(temp_data, "blacklisted_resonator_frequencies"):
            object.__setattr__(temp_data, "blacklisted_resonator_frequencies", None)
        return temp_data

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            fit_result = node.results["fit_results"][q.name]
            error_code = ResonatorSpectroscopyErrorCode(
                fit_result.get("error_code", int(ResonatorSpectroscopyErrorCode.SUCCESS))
            )

            if node.outcomes[q.name] == "failed":
                if error_code == ResonatorSpectroscopyErrorCode.NO_DIP_FOUND:
                    temp_data = _ensure_temp_calibration(machine, q.name)

                    # Blacklist the candidate frequency that just failed
                    candidate_freq = float(q.resonator.RF_frequency)
                    if temp_data.blacklisted_resonator_frequencies is None:
                        temp_data.blacklisted_resonator_frequencies = []
                    if candidate_freq not in temp_data.blacklisted_resonator_frequencies:
                        temp_data.blacklisted_resonator_frequencies.append(candidate_freq)
                    fit_result["corrective_action"] = int(ResonatorSpectroscopyCorrectiveAction.BLACKLIST_FREQUENCY)
                    fit_result["action_magnitude"] = candidate_freq
                    node.log(
                        f"[Adaptive] {q.name}: No resonator dip found. "
                        f"Blacklisted RF frequency {candidate_freq / 1e9:.6f} GHz."
                    )

                    # Restore the user's original resonator frequency so that the
                    # next broad spectroscopy sweep is centred on the original guess.
                    if temp_data.initial_resonator_f01 is not None:
                        q.resonator.f_01 = temp_data.initial_resonator_f01
                        q.resonator.RF_frequency = temp_data.initial_resonator_RF_frequency
                        node.log(
                            f"[Adaptive] {q.name}: Restored initial resonator frequency "
                            f"{temp_data.initial_resonator_RF_frequency / 1e9:.6f} GHz "
                            f"for next broad spectroscopy sweep."
                        )
                continue

            results = fit_result

            # --- Standard resonator updates ---
            freq = float(results["frequency"])
            q.resonator.f_01 = freq
            q.resonator.RF_frequency = freq

            # Permanently save the readout power/amplitude if requested.
            # (tracked_resonator.revert_changes() above undid the temporary change;
            # we re-apply it here outside the tracked context so it persists.)
            if (
                node.parameters.save_readout_amplitude
                and node.parameters.readout_power_dbm is not None
            ):
                q.resonator.set_locked_output_power(power_in_dbm=node.parameters.readout_power_dbm)

            # Store circle-fit Q parameters when run_circle_fit=True and fit succeeded
            if node.parameters.run_circle_fit and np.isfinite(
                results.get("Q_loaded", float("nan"))
            ):
                q.resonator.Q_loaded       = float(results["Q_loaded"])
                q.resonator.Q_internal     = float(results["Q_internal"])
                q.resonator.Q_external     = float(results["Q_external"])
                q.resonator.kappa_Hz       = float(results["kappa_Hz"])
                q.resonator.phi0_circlefit = float(results["phi0_circlefit"])

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()