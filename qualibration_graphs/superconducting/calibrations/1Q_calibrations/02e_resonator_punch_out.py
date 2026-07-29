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
from calibration_utils.resonator_punch_out import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.error_codes import (
    ResonatorPunchOutErrorCode,
    ResonatorPunchOutCorrectiveAction,
)
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Helper functions}
def _ensure_temp_calibration_fields(machine, qubit_name: str) -> TemporaryCalibrationData:
    """
    Ensure temp_calibration data has all required fields.

    This is needed for backward compatibility when loading old state.json files
    that don't have newly added fields. Adds missing fields directly to the
    existing object using object.__setattr__ to bypass QUAM's property system.

    Args:
        machine: The QUAM machine object
        qubit_name: Name of the qubit

    Returns:
        TemporaryCalibrationData object with all fields properly initialized
    """
    # Initialize if not present
    if qubit_name not in machine.temp_calibration:
        machine.temp_calibration[qubit_name] = TemporaryCalibrationData()
        return machine.temp_calibration[qubit_name]

    temp_data = machine.temp_calibration[qubit_name]

    # Define all expected fields with their default values
    expected_fields = {
        'parameters': None,
        'adaptive_frequency_span_mhz': None,
        'adaptive_power_shift_dbm': None,
        'adaptive_num_shots': None,
        'last_updated': None,
        'notes': None,
    }

    # Add any missing fields directly using object.__setattr__
    # This bypasses QUAM's property system and adds the field to the instance
    for field_name, default_value in expected_fields.items():
        if not hasattr(temp_data, field_name):
            object.__setattr__(temp_data, field_name, default_value)

    return temp_data


# %% {Node initialisation}
description = """
        RESONATOR PUNCH-OUT SPECTROSCOPY
This sequence characterizes the resonator response as a function of readout power
in order to detect power-induced shifts of the resonator frequency (punch-out).
A readout pulse is applied and the demodulated 'I' and 'Q' quadratures are acquired
for all resonators simultaneously while sweeping the readout frequency at a small
number of readout power levels.

For each power level, the resonator frequency is extracted directly from the
measured response. By comparing the resonator frequency at low and high readout
power, the presence of a power-induced frequency shift is detected. Based on this
analysis, an optimal readout power is selected that avoids resonator punch-out while
maintaining sufficient signal strength.

Prerequisites:
    - Having calibrated the resonator frequency at low power
      (e.g., node 02a_resonator_spectroscopy.py).
    - Having specified the desired flux point, if relevant (qubit.z.flux_point).

State update:
    - The readout power is set to the selected optimal value using
      qubit.resonator.set_output_power().
    - The readout frequency is set to the measured resonator frequency at the
      optimal readout power:
          * qubit.resonator.f_01
          * qubit.resonator.RF_frequency
    - Adaptive parameters in temp_calibration (when use_adaptive_span=True):
      * On power too high (no shift + at bare frequency): decreases power span by 10 dB
      * On success: resets adaptive_power_shift_dbm
"""



# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02e_resonator_punch_out",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


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

    # Use adaptive power shift if available (from previous failed calibrations)
    min_power_dbm = node.parameters.min_power_dbm
    max_power_dbm = node.parameters.max_power_dbm

    if node.parameters.use_adaptive_span and node.machine.temp_calibration is not None:
        # Collect adaptive power shifts from all active qubits and apply the
        # most conservative (most negative) shift so the sweep stays valid for
        # every qubit in the batch.
        shifts = []
        for qubit in qubits:
            temp_data = _ensure_temp_calibration_fields(node.machine, qubit.name)
            if temp_data.adaptive_power_shift_dbm is not None:
                shifts.append(temp_data.adaptive_power_shift_dbm)

        if shifts:
            power_shift = min(shifts)  # most negative = most conservative
            candidate_min = min_power_dbm + power_shift
            if candidate_min >= max_power_dbm:
                # Stale or invalid adaptive shift (e.g. positive value from old code,
                # or shift so large that the range collapses).  Reset and use the
                # parameter-specified range.
                node.log(
                    f"WARNING: adaptive power shift ({power_shift:.1f} dBm) would "
                    f"collapse the sweep range to [{candidate_min:.1f}, {max_power_dbm:.1f}] dBm. "
                    f"Resetting adaptive_power_shift_dbm to None for all qubits."
                )
                for qubit in qubits:
                    temp_data = _ensure_temp_calibration_fields(node.machine, qubit.name)
                    temp_data.adaptive_power_shift_dbm = None
            else:
                min_power_dbm = candidate_min
                node.log(
                    f"Using adaptive power shift: {power_shift:.1f} dBm "
                    f"(from {[q.name for q in qubits]})\n"
                    f"  Min power: {min_power_dbm:.1f} dBm\n"
                    f"  Max power: {max_power_dbm:.1f} dBm"
                )

    # Update the readout power to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_resonators"] = []
    for i, qubit in enumerate(qubits):
        with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.set_locked_output_power(power_in_dbm=max_power_dbm)
            node.namespace["tracked_resonators"].append(resonator)

    # The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
    amp_min = calculate_voltage_scaling_factor(max_power_dbm, min_power_dbm)
    amps = np.geomspace(amp_min, 1, node.parameters.num_power_points)
    power_dbm = np.linspace(
        min_power_dbm,
        max_power_dbm,
        node.parameters.num_power_points,
    )
    # Frequency sweep starting at the bare resonator frequency and going to higher
    # frequencies (punch-out shifts the resonance UP at high power for this device).
    # df_bare is the detuning of frequency_bare from the current LO/RF reference.
    # When broad-spec has just run, RF_frequency == frequency_bare and df_bare == 0.
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    left_offset = int(round(node.parameters.sweep_left_offset_mhz * u.MHz))
    df_bare = int(round(np.mean([
        q.resonator.frequency_bare - q.resonator.RF_frequency
        for q in qubits
    ])))
    # Shift the sweep window left so that frequencies below the bare resonator
    # (e.g. the dispersive-shifted low-power resonance) are included.
    dfs = np.arange(df_bare - left_offset, df_bare - left_offset + span, step)

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
                            rr.wait(10 * rr.depletion_time // 4)
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
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    # Ensure temp_calibration exists
    if node.machine.temp_calibration is None:
        node.machine.temp_calibration = {}

    # Constants for adaptive power adjustment
    POWER_DECREASE_DBM = -10.0  # Decrease power when no shift detected (power too high)
    MIN_POWER_DBM = -100.0      # Minimum allowed power
    FREQUENCY_TOLERANCE_HZ = 1e6  # 1 MHz tolerance for checking if at bare frequency

    # Update the state
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            # Ensure all temp_calibration fields exist
            temp_data = _ensure_temp_calibration_fields(node.machine, q.name)

            # Get fit results for this qubit
            fit_result = node.results["fit_results"][q.name]
            freq_shift = fit_result["frequency_shift"]
            resonator_frequency = fit_result["resonator_frequency"]
            bare_frequency = q.resonator.frequency_bare
            error_code = ResonatorPunchOutErrorCode(fit_result.get("error_code", 0))


            # Check if power is too high (no/small positive shift and at bare frequency)
            # freq_shift = freq_high - freq_low, so positive means resonator shifted UP at high power
            no_shift = freq_shift < node.parameters.frequency_shift_threshold_in_hz
            at_bare_frequency = abs(resonator_frequency - bare_frequency) < FREQUENCY_TOLERANCE_HZ
            power_too_high = no_shift and at_bare_frequency

            if node.outcomes[q.name] == "failed":
                # Handle failed calibration with adaptive power adjustment
                if node.parameters.use_adaptive_span and power_too_high:
                    # Get current power shift
                    current_power_shift = temp_data.adaptive_power_shift_dbm or 0.0
                    new_power_shift = current_power_shift + POWER_DECREASE_DBM

                    # Check if we can decrease power further
                    if node.parameters.min_power_dbm + new_power_shift >= MIN_POWER_DBM:
                        temp_data.adaptive_power_shift_dbm = new_power_shift

                        # Update error code to indicate at bare frequency
                        node.results["fit_results"][q.name]["error_code"] = int(ResonatorPunchOutErrorCode.AT_BARE_FREQUENCY)
                        node.results["fit_results"][q.name]["corrective_action"] = int(ResonatorPunchOutCorrectiveAction.DECREASE_POWER_SPAN)
                        node.results["fit_results"][q.name]["action_magnitude"] = float(POWER_DECREASE_DBM)

                        node.log(
                            f"[{q.name}] ERROR CODE: AT_BARE_FREQUENCY ({ResonatorPunchOutErrorCode.AT_BARE_FREQUENCY})\n"
                            f"  CORRECTIVE ACTION: DECREASE_POWER_SPAN ({POWER_DECREASE_DBM} dBm)\n"
                            f"  Bare frequency:       {bare_frequency / 1e9:.6f} GHz\n"
                            f"  Measured frequency:   {resonator_frequency / 1e9:.6f} GHz\n"
                            f"  Frequency shift:      {freq_shift / 1e6:.3f} MHz\n"
                            f"  Current power shift:  {current_power_shift:.1f} dBm\n"
                            f"  New power shift:      {new_power_shift:.1f} dBm\n"
                            f"  Next min power:       {node.parameters.min_power_dbm + new_power_shift:.1f} dBm\n"
                            f"  Next max power:       {node.parameters.max_power_dbm + new_power_shift:.1f} dBm"
                        )
                    else:
                        node.results["fit_results"][q.name]["corrective_action"] = int(ResonatorPunchOutCorrectiveAction.NONE)
                        node.log(
                            f"[{q.name}] ERROR CODE: AT_BARE_FREQUENCY ({ResonatorPunchOutErrorCode.AT_BARE_FREQUENCY})\n"
                            f"  CORRECTIVE ACTION: NONE (minimum power limit reached: {MIN_POWER_DBM} dBm)"
                        )
                else:
                    node.results["fit_results"][q.name]["corrective_action"] = int(ResonatorPunchOutCorrectiveAction.NONE)
                    node.log(
                        f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                        f"  CORRECTIVE ACTION: NONE (adaptive adjustments {'disabled' if not node.parameters.use_adaptive_span else 'not applicable'})"
                    )
                continue

            # -----------------------------
            # Successful calibration: reset adaptive parameters and update state
            # -----------------------------
            temp_data.adaptive_frequency_span_mhz = None
            temp_data.adaptive_power_shift_dbm = None
            temp_data.adaptive_num_shots = None

            # Update fit results with success corrective action
            node.results["fit_results"][q.name]["corrective_action"] = int(ResonatorPunchOutCorrectiveAction.RESET_ADAPTIVE_PARAMS)
            node.results["fit_results"][q.name]["action_magnitude"] = 0.0

            # Update the readout power
            q.resonator.set_locked_output_power(power_in_dbm=fit_result["optimal_power"])
            # Set the resonator frequency directly to the measured low-power resonance.
            # This is more robust than incrementing by freq_shift, which can accumulate errors.
            freq_low_abs = fit_result["freq_low_abs"]
            q.resonator.f_01 = freq_low_abs
            q.resonator.RF_frequency = freq_low_abs

            node.log(
                f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                f"  CORRECTIVE ACTION: RESET_ADAPTIVE_PARAMS\n"
                f"  Updated state:\n"
                f"    Optimal power:          {fit_result['optimal_power']:.2f} dBm\n"
                f"    Low-power frequency:    {freq_low_abs / 1e9:.6f} GHz\n"
                f"    Frequency shift:        {freq_shift / 1e6:.3f} MHz"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()