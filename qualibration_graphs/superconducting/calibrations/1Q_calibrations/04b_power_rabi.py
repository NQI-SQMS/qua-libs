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
from quam_config import Quam
from calibration_utils.power_rabi import (
    Parameters,
    get_number_of_pulses,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.error_codes import PowerRabiErrorCode, PowerRabiCorrectiveAction
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from quam_config.instrument_limits import instrument_limits


# %% {Description}
description = """
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having set the qubit gates duration (qubit.xy.operations["x180"].length).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit pulse amplitude corresponding to the specified operation (x180, x90...)
    (qubit.xy.operations[operation].amplitude).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="04b_power_rabi",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # node.parameters.max_number_pulses_per_sweep = 100
    # node.parameters.min_amp_factor = 0.8
    # node.parameters.max_amp_factor = 1.2
    # node.parameters.amp_factor_step = 0.01
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

    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    # Number of applied Rabi pulses sweep
    N_pi_vec = get_number_of_pulses(node.parameters)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_pulses": xr.DataArray(N_pi_vec, attrs={"long_name": "number of pulses"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    # Apply operation_length_in_ns override before config generation (modifies QUAM in-memory)
    if node.parameters.operation_length_in_ns is not None:
        for qubit in qubits:
            qubit.xy.operations[operation].length = node.parameters.operation_length_in_ns

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        npi = declare(int)  # QUA variable for the number of qubit pulses

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, N_pi_vec)):
                    with for_(*from_array(a, amps)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()

                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            # Loop for error amplification (perform many qubit pulses)
                            count = declare(int)  # QUA variable for counting the qubit pulses
                            with for_(count, 0, count < npi, count + 1):
                                qubit.xy.play(operation, amplitude_scale=a)
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                            if node.parameters.use_state_discrimination:
                                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                                save(state[i], state_st[i])
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                I_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amps)).buffer(len(N_pi_vec)).average().save(f"state{i + 1}")


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
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful.

    In adaptive mode (use_adaptive=True):
    - NO_OSCILLATION: adds the qubit's RF frequency to the blacklist in temp_calibration.
      If duration adaptation was active, restores the original pulse length.
    - TOO_MANY_PERIODS: scales the base amplitude down (new = old / num_periods).
    - TOO_FEW_PERIODS: three-stage escalation:
        Stage 1 – scales amplitude up while hardware headroom remains.
        Stage 2 – increases Octave RF upconversion gain in small dB steps
                   (≤ 3 dB/step, up to 20 dB max) once amplitude is maxed.
        Stage 3 – increases pulse duration (new_len = old_len / num_periods,
                   rounded to 4 ns) once both amplitude and Octave gain are maxed.
    - SUCCESS: updates to the fitted optimal amplitude; if duration adaptation was
      active, keeps the adapted length and clears the temp fields.
    """
    # Maximum Octave upconversion gain (dB).  At this value the RF chain is
    # fully open and further power can only be gained by increasing pulse duration.
    _MAX_OCTAVE_GAIN_DB = 20.0
    # Maximum per-step Octave gain increase (dB).  Caps the dB delta derived from
    # num_periods to avoid large jumps between iterations.
    _MAX_OCTAVE_GAIN_STEP_DB = 3.0
    # Minimum allowed pulse length (must be a multiple of 4 ns for QUA).
    _MIN_PULSE_LENGTH_NS = 16

    def _ensure_temp_calibration(machine, qubit_name: str):
        """Return the TemporaryCalibrationData for qubit_name, creating it if absent."""
        from quam_config.my_quam import TemporaryCalibrationData
        if machine.temp_calibration is None:
            machine.temp_calibration = {}
        if qubit_name not in machine.temp_calibration:
            machine.temp_calibration[qubit_name] = TemporaryCalibrationData()
        temp_data = machine.temp_calibration[qubit_name]
        # Backward-compatibility: add fields if an older state.json omitted them
        for field in ("blacklisted_qubit_points", "adaptive_x180_length_ns", "initial_x180_length_ns"):
            if not hasattr(temp_data, field):
                object.__setattr__(temp_data, field, None)
        return temp_data

    def _get_rf_frequency(qubit) -> float:
        """Return the best available RF frequency estimate for the qubit XY channel."""
        try:
            return float(qubit.xy.LO_frequency + qubit.xy.intermediate_frequency)
        except AttributeError:
            return float(qubit.xy.intermediate_frequency)

    def _get_octave_gain(qubit) -> float:
        """Return the Octave upconversion gain in dB, or -inf if not applicable."""
        try:
            return float(qubit.xy.frequency_converter_up.gain)
        except AttributeError:
            return float("-inf")

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            fit_result = node.results["fit_results"][q.name]
            error_code = PowerRabiErrorCode(
                fit_result.get("error_code", int(PowerRabiErrorCode.SUCCESS))
            )
            operation = q.xy.operations[node.parameters.operation]
            limits = instrument_limits(q.xy)

            # ── Failed calibration ──────────────────────────────────────────────
            if node.outcomes[q.name] == "failed":
                if node.parameters.use_adaptive and error_code == PowerRabiErrorCode.NO_OSCILLATION:
                    temp_data = _ensure_temp_calibration(node.machine, q.name)

                    # Restore original pulse length if duration adaptation was active
                    if temp_data.initial_x180_length_ns is not None:
                        original_len = int(temp_data.initial_x180_length_ns)
                        operation.length = original_len
                        if node.parameters.operation == "x180":
                            try:
                                q.xy.operations["x90"].length = original_len
                            except ValueError:
                                pass  # x90.length is a reference to x180.length; updates automatically
                        temp_data.adaptive_x180_length_ns = None
                        temp_data.initial_x180_length_ns = None
                        node.log(
                            f"[Adaptive] {q.name}: No oscillation after duration adaptation. "
                            f"Restored original pulse length: {original_len} ns."
                        )

                    # Blacklist the (qubit frequency, drive power) pair so upstream
                    # nodes can avoid this specific 2D point in future spectroscopy runs.
                    rf_freq = _get_rf_frequency(q)
                    power_dbm = q.xy.get_output_power("saturation")
                    if temp_data.blacklisted_qubit_points is None:
                        temp_data.blacklisted_qubit_points = []
                    if [rf_freq, power_dbm] not in temp_data.blacklisted_qubit_points:
                        temp_data.blacklisted_qubit_points.append([rf_freq, power_dbm])
                    fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.BLACKLIST_FREQUENCY)
                    fit_result["action_magnitude"] = rf_freq
                    node.log(
                        f"[Adaptive] {q.name}: No oscillation detected. "
                        f"Blacklisted RF frequency {rf_freq / 1e9:.6f} GHz "
                        f"at drive power {power_dbm:.1f} dBm."
                    )
                continue

            # ── Adaptive: rescale if period count is off ─────────────────────────
            if node.parameters.use_adaptive and error_code in (
                PowerRabiErrorCode.TOO_MANY_PERIODS,
                PowerRabiErrorCode.TOO_FEW_PERIODS,
            ):
                num_periods = fit_result.get("num_periods", 1.0)
                if num_periods > 0 and np.isfinite(num_periods):
                    current_amp = operation.amplitude

                    # ── TOO_FEW_PERIODS: three-stage escalation
                    #    Stage 1 – scale amplitude up while hardware headroom remains.
                    #    Stage 2 – increase Octave RF gain in small dB steps once the
                    #               amplitude sweep saturates the hardware limit.
                    #    Stage 3 – increase pulse duration once both amplitude and
                    #               Octave gain are at their maximum.
                    if error_code == PowerRabiErrorCode.TOO_FEW_PERIODS:
                        amplitude_maxed = (
                            current_amp * node.parameters.max_amp_factor
                            >= limits.max_x180_wf_amplitude
                        )
                        current_gain = _get_octave_gain(q)
                        gain_maxed = current_gain >= _MAX_OCTAVE_GAIN_DB

                        if amplitude_maxed and gain_maxed:
                            # Stage 3: Switch to duration adaptation.
                            # num_periods ∝ duration → new_len = old_len / num_periods
                            temp_data = _ensure_temp_calibration(node.machine, q.name)

                            # Save original length on the first duration-adaptation step
                            if temp_data.initial_x180_length_ns is None:
                                temp_data.initial_x180_length_ns = float(operation.length)

                            current_len = float(operation.length)
                            # Round to a multiple of 4 ns (one QUA clock cycle)
                            new_len = int(round(current_len / num_periods / 4) * 4)
                            new_len = max(new_len, _MIN_PULSE_LENGTH_NS)
                            operation.length = new_len
                            if node.parameters.operation == "x180":
                                try:
                                    q.xy.operations["x90"].length = new_len
                                except ValueError:
                                    pass  # x90.length is a reference to x180.length; updates automatically

                            temp_data.adaptive_x180_length_ns = float(new_len)
                            fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.INCREASE_DURATION)
                            fit_result["action_magnitude"] = float(new_len)
                            node.log(
                                f"[Adaptive] {q.name}: TOO_FEW_PERIODS ({num_periods:.2f} periods). "
                                f"Amplitude maxed ({current_amp * node.parameters.max_amp_factor:.4f} V "
                                f">= {limits.max_x180_wf_amplitude:.3f} V) and Octave gain maxed "
                                f"({current_gain:.0f} dB). "
                                f"Increasing pulse duration: {current_len:.0f} ns → {new_len} ns."
                            )

                        elif amplitude_maxed:
                            # Stage 2: Amplitude is at the hardware limit; increase
                            # Octave RF upconversion gain in small dB steps.
                            # dB equivalent of the amplitude scale factor, capped per step.
                            gain_delta_db = min(
                                20.0 * np.log10(1.0 / num_periods),
                                _MAX_OCTAVE_GAIN_STEP_DB,
                            )
                            new_gain = min(current_gain + gain_delta_db, _MAX_OCTAVE_GAIN_DB)
                            try:
                                q.xy.frequency_converter_up.gain = new_gain
                            except AttributeError:
                                pass  # No Octave connected; no-op
                            fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.INCREASE_OCTAVE_GAIN)
                            fit_result["action_magnitude"] = new_gain
                            node.log(
                                f"[Adaptive] {q.name}: TOO_FEW_PERIODS ({num_periods:.2f} periods). "
                                f"Amplitude maxed ({current_amp * node.parameters.max_amp_factor:.4f} V "
                                f">= {limits.max_x180_wf_amplitude:.3f} V). "
                                f"Increasing Octave gain: {current_gain:.1f} dB → {new_gain:.1f} dB."
                            )

                        else:
                            # Stage 1: Amplitude headroom remains – scale amplitude up.
                            # Clip to max_x180_wf_amplitude / max_amp_factor so the
                            # next sweep's peak (base × max_amp_factor) stays within
                            # the hardware safe limit.
                            max_safe_base_amp = (
                                limits.max_x180_wf_amplitude / node.parameters.max_amp_factor
                            )
                            new_amp = float(
                                np.clip(current_amp / num_periods, 0.0, max_safe_base_amp)
                            )
                            operation.amplitude = new_amp
                            if node.parameters.operation == "x180":
                                q.xy.operations["x90"].amplitude = new_amp / 2
                            fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.INCREASE_AMPLITUDE)
                            fit_result["action_magnitude"] = new_amp
                            node.log(
                                f"[Adaptive] {q.name}: TOO_FEW_PERIODS ({num_periods:.2f} periods). "
                                f"Rescaling {node.parameters.operation} amplitude: "
                                f"{1e3 * current_amp:.2f} mV → {1e3 * new_amp:.2f} mV "
                                f"(limit {1e3 * max_safe_base_amp:.2f} mV = "
                                f"{1e3 * limits.max_x180_wf_amplitude:.0f} mV / {node.parameters.max_amp_factor})."
                            )

                    # ── TOO_MANY_PERIODS: always scale amplitude down
                    else:
                        new_amp = float(
                            np.clip(current_amp / num_periods, 0.0, limits.max_x180_wf_amplitude)
                        )
                        operation.amplitude = new_amp
                        if node.parameters.operation == "x180":
                            q.xy.operations["x90"].amplitude = new_amp / 2
                        fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.REDUCE_AMPLITUDE)
                        fit_result["action_magnitude"] = new_amp
                        node.log(
                            f"[Adaptive] {q.name}: TOO_MANY_PERIODS ({num_periods:.2f} periods). "
                            f"Rescaling {node.parameters.operation} amplitude: "
                            f"{1e3 * current_amp:.2f} mV → {1e3 * new_amp:.2f} mV."
                        )
                else:
                    # Degenerate case (num_periods NaN / zero): fall back to normal update
                    safe_amp = float(np.clip(fit_result["opt_amp"], 0.0, limits.max_x180_wf_amplitude))
                    operation.amplitude = safe_amp
                    if node.parameters.operation == "x180":
                        q.xy.operations["x90"].amplitude = safe_amp / 2

            # ── Normal update (non-adaptive, or adaptive SUCCESS) ────────────────
            else:
                # When the fitted pi amplitude exceeds the hardware DAC limit,
                # amplitude (Stage 1) is already saturated.  In adaptive mode,
                # escalate identically to the amplitude-maxed TOO_FEW_PERIODS path:
                #   Stage 2 – increase Octave gain.
                #   Stage 3 – increase pulse duration once gain is also maxed.
                # The outcome is forced to "failed" so the calibration loop retries.
                if node.parameters.use_adaptive and fit_result["opt_amp"] > limits.max_x180_wf_amplitude:
                    # Clamp amplitude to the hardware maximum.
                    operation.amplitude = float(limits.max_x180_wf_amplitude)
                    if node.parameters.operation == "x180":
                        try:
                            q.xy.operations["x90"].amplitude = float(limits.max_x180_wf_amplitude) / 2
                        except ValueError:
                            pass  # x90.amplitude is a reference; updates automatically

                    ratio = fit_result["opt_amp"] / limits.max_x180_wf_amplitude  # > 1.0
                    current_gain = _get_octave_gain(q)
                    gain_maxed = current_gain >= _MAX_OCTAVE_GAIN_DB

                    if not gain_maxed:
                        # Stage 2: increase Octave gain so the same DAC amplitude
                        # produces more RF power, reducing the required waveform value.
                        gain_delta_db = min(20.0 * np.log10(ratio), _MAX_OCTAVE_GAIN_STEP_DB)
                        new_gain = min(current_gain + gain_delta_db, _MAX_OCTAVE_GAIN_DB)
                        try:
                            q.xy.frequency_converter_up.gain = new_gain
                        except AttributeError:
                            pass  # No Octave; no-op
                        fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.INCREASE_OCTAVE_GAIN)
                        fit_result["action_magnitude"] = new_gain
                        node.log(
                            f"[Adaptive] {q.name}: opt_amp ({1e3 * fit_result['opt_amp']:.2f} mV) "
                            f"exceeds hardware limit ({1e3 * limits.max_x180_wf_amplitude:.0f} mV). "
                            f"Amplitude maxed. Increasing Octave gain: "
                            f"{current_gain:.1f} dB → {new_gain:.1f} dB."
                        )
                    else:
                        # Stage 3: both amplitude and Octave gain are maxed.
                        # Increase pulse duration (pi-rotation angle ∝ amplitude × duration,
                        # so multiplying duration by `ratio` lowers the required amplitude
                        # by the same factor while keeping ΩT = π constant).
                        temp_data = _ensure_temp_calibration(node.machine, q.name)
                        if temp_data.initial_x180_length_ns is None:
                            temp_data.initial_x180_length_ns = float(operation.length)
                        current_len = float(operation.length)
                        new_len = int(round(current_len * ratio / 4) * 4)
                        new_len = max(new_len, _MIN_PULSE_LENGTH_NS)
                        operation.length = new_len
                        if node.parameters.operation == "x180":
                            try:
                                q.xy.operations["x90"].length = new_len
                            except ValueError:
                                pass  # x90.length is a reference; updates automatically
                        temp_data.adaptive_x180_length_ns = float(new_len)
                        fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.INCREASE_DURATION)
                        fit_result["action_magnitude"] = float(new_len)
                        node.log(
                            f"[Adaptive] {q.name}: opt_amp ({1e3 * fit_result['opt_amp']:.2f} mV) "
                            f"exceeds hardware limit ({1e3 * limits.max_x180_wf_amplitude:.0f} mV). "
                            f"Amplitude and Octave gain maxed. "
                            f"Increasing pulse duration: {current_len:.0f} ns → {new_len} ns."
                        )

                    # Force the bringup loop to retry: the pi pulse is not yet
                    # calibrated to the correct amplitude.
                    node.outcomes[q.name] = "failed"

                else:
                    # Amplitude within hardware limits: apply fitted value directly.
                    safe_amp = float(np.clip(fit_result["opt_amp"], 0.0, limits.max_x180_wf_amplitude))
                    operation.amplitude = safe_amp
                    if node.parameters.operation == "x180":
                        q.xy.operations["x90"].amplitude = safe_amp / 2
                    # Save the pulse length used in this run
                    pulse_len = fit_result.get("pulse_length_ns", float("nan"))
                    if np.isfinite(pulse_len):
                        operation.length = int(pulse_len)
                    if node.parameters.use_adaptive:
                        # If duration adaptation was active, keep the adapted length and
                        # clear the temp fields so future runs start fresh.
                        temp_data = _ensure_temp_calibration(node.machine, q.name)
                        if temp_data.adaptive_x180_length_ns is not None:
                            node.log(
                                f"[Adaptive] {q.name}: SUCCESS after duration adaptation. "
                                f"Keeping adapted pulse length: {operation.length} ns. "
                                f"Clearing adaptive length fields."
                            )
                            temp_data.adaptive_x180_length_ns = None
                            temp_data.initial_x180_length_ns = None
                        fit_result["corrective_action"] = int(PowerRabiCorrectiveAction.NONE)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
