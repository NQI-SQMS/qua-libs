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

from calibration_utils.qubit_spectroscopy_vs_power import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_gradient_score_diagnostics,
    plot_iq_pca_diagnostics,
)
from calibration_utils.error_codes import (
    QubitSpectroscopyErrorCode,
    QubitSpectroscopyCorrectiveAction,
)

from qualibration_libs.parameters import get_qubits
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
        'selected_power_dbm': None,
        'selected_octave_gain_db': None,
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
        QUBIT SPECTROSCOPY VS DRIVE POWER
This sequence involves probing the qubit transition by applying an XY drive while sweeping the drive power and
intermediate frequency around the expected qubit transition for all active qubits.
The qubit response is measured via the readout resonator, and the demodulated I/Q signals are post-processed to extract
the qubit spectroscopy signal as a function of frequency and drive power.

The resulting 2D spectroscopy map is analyzed to identify the qubit transition frequency, assess power broadening
and saturation effects, and select an appropriate drive power for subsequent calibrations.
A rough estimate of the qubit frequency at the selected drive power is extracted and used to update the qubit state.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the XY control line (node 01a_mixer_calibration.py).
    - Having calibrated the readout chain, including time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the readout resonator frequency (node 02_resonator_spectroscopy.py).
    - Having initialized the QUAM state parameters for the qubit XY pulse frequency span, power sweep range,
      and pulse duration.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit transition frequency at the selected power:
      qubit.xy.f_01 & qubit.xy.RF_frequency
    - The selected spectroscopy drive power: qubit.xy.spectroscopy_power
    - Adaptive parameters in temp_calibration (when use_adaptive_span=True):
      * On no peak: expands frequency span AND increases power
      * On success: resets all adaptive parameters (adaptive_frequency_span_mhz,
        adaptive_power_shift_dbm, adaptive_num_shots)
"""


node = QualibrationNode[Parameters, Quam](
    name="03c_qubit_spectroscopy_vs_power",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.frequency_span_in_mhz = 200
    # node.parameters.frequency_step_in_mhz = 1
    # node.parameters.num_power_points = 10
    # node.parameters.num_shots = 100
    # node.parameters.min_power_dbm = -80
    # node.parameters.max_power_dbm = 0
    # node.parameters.operation = "saturation"
    # node.parameters.operation_len_in_ns = 200_000
    # node.parameters.max_amplitude_opx = 0.3
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)

    qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots
    node.namespace["qubits"] = qubits

    node.namespace["tracked_qubits"] = []
    for qubit in qubits:
        with tracked_updates(qubit.xy, auto_revert=False) as xy:
            xy.set_output_power(
                power_in_dbm=node.parameters.max_power_dbm,
                max_amplitude=node.parameters.max_amplitude_opx,
                operation=node.parameters.operation,
            )
            node.namespace["tracked_qubits"].append(xy)

    # Use adaptive frequency span/step if available (from previous failed calibrations)
    # Otherwise fall back to the default parameters
    frequency_span_mhz = node.parameters.frequency_span_in_mhz
    frequency_step_mhz = node.parameters.frequency_step_in_mhz

    if (
        node.parameters.use_adaptive_span
        and node.machine.temp_calibration is not None
        and len(qubits) == 1
    ):
        # Only use adaptive span/step for single-qubit calibrations
        qubit = qubits[0]
        temp_data = _ensure_temp_calibration_fields(node.machine, qubit.name)

        if temp_data.adaptive_frequency_span_mhz is not None:
            frequency_span_mhz = temp_data.adaptive_frequency_span_mhz
            node.log(
                f"[{qubit.name}] Using adaptive frequency span: {frequency_span_mhz:.1f} MHz"
            )

    span = frequency_span_mhz * u.MHz
    step = frequency_step_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Use adaptive power shift if available (from previous over-saturation detection)
    min_power_dbm = node.parameters.min_power_dbm
    max_power_dbm = node.parameters.max_power_dbm

    if (
        node.parameters.use_adaptive_span
        and node.machine.temp_calibration is not None
        and len(qubits) == 1
    ):
        # Only use adaptive power shift for single-qubit calibrations
        qubit = qubits[0]
        temp_data = _ensure_temp_calibration_fields(node.machine, qubit.name)

        power_shift = temp_data.adaptive_power_shift_dbm
        if power_shift is not None:
            min_power_dbm += power_shift
            max_power_dbm += power_shift
            node.log(
                f"[{qubit.name}] Using adaptive power shift: {power_shift:.1f} dBm\n"
                f"  Min power: {min_power_dbm:.1f} dBm\n"
                f"  Max power: {max_power_dbm:.1f} dBm"
            )

    powers_dbm = np.linspace(
        min_power_dbm,
        max_power_dbm,
        node.parameters.num_power_points,
    )

    amps = np.geomspace(node.parameters.min_amplitude_opx, 1.0, node.parameters.num_power_points)

    node.namespace["powers_dbm"] = powers_dbm
    node.namespace["amps"] = amps

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names(), dims="qubit"),
        "detuning": xr.DataArray(dfs, dims="detuning", attrs={"units": "Hz"}),
        "power": xr.DataArray(powers_dbm, dims="power", attrs={"units": "dBm"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

        df = declare(int)
        a = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # OUTER LOOP: frequency
                with for_(*from_array(df, dfs)):
                    for qubit in multiplexed_qubits.values():
                        qubit.xy.update_frequency(
                            df + qubit.xy.intermediate_frequency
                        )

                    # INNER LOOP: amplitude (power)
                    with for_each_(a, amps):
                        for qubit in multiplexed_qubits.values():
                            duration = (
                                node.parameters.operation_len_in_ns
                                if node.parameters.operation_len_in_ns is not None
                                else qubit.xy.operations[node.parameters.operation].length
                            )
                            qubit.xy.play(
                                node.parameters.operation,
                                duration=duration // 4,
                                amplitude_scale=a,
                            )

                        align()

                        for qi, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure(
                                "readout",
                                qua_vars=(I[qi], Q[qi]),
                            )
                            qubit.resonator.wait(
                                node.machine.depletion_time * u.ns
                            )
                            align()
                            save(I[qi], I_st[qi])
                            save(Q[qi], Q_st[qi])

                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                (
                    I_st[i]
                    .buffer(len(amps))   # INNER loop (power)
                    .buffer(len(dfs))    # OUTER loop (detuning)
                    .average()
                    .save(f"I{i+1}")
                )
                (
                    Q_st[i]
                    .buffer(len(amps))
                    .buffer(len(dfs))
                    .average()
                    .save(f"Q{i+1}")
                )



# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])

        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )

        node.log(job.execution_report())
    
    node.results["ds_raw"] = dataset

# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
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
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        signal_source=node.parameters.signal_source,
    )
    plt.show()

    fig_pca = plot_iq_pca_diagnostics(
        node.results["ds_fit"],
        node.namespace["qubits"],
    )
    plt.show()

    fig_grad = plot_gradient_score_diagnostics(
        node.results["ds_fit"],
        node.namespace["qubits"],
        signal_source=node.parameters.signal_source,
    )
    plt.show()

    node.results["figures"] = {
        "spectroscopy_vs_power": fig,
        "iq_pca_diagnostics": fig_pca,
        "gradient_score_diagnostics": fig_grad,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """
    Update qubit state based on spectroscopy vs power analysis:
      - XY Octave output power
      - Saturation amplitude
      - Qubit frequency
      - Adaptive frequency span for failed calibrations
    """

    for tracked_qubit in node.namespace.get("tracked_qubits", []):
        tracked_qubit.revert_changes()

    # Ensure temp_calibration exists
    if node.machine.temp_calibration is None:
        node.machine.temp_calibration = {}

    MAX_FREQUENCY_SPAN_MHZ = 800.0
    FREQUENCY_SPAN_EXPANSION_FACTOR = 1.5
    POWER_INCREASE_DBM = 10.0    # Increase power when no/weak peak
    POWER_DECREASE_DBM = -10.0   # Decrease power when over-saturated
    MIN_POWER_DBM = -100.0       # Minimum allowed power
    MAX_POWER_DBM = 40.0         # Maximum allowed power

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            # Ensure all fields exist (for backward compatibility with old state files)
            # This will create the object if it doesn't exist, or replace it with a proper
            # instance if fields are missing
            temp_data = _ensure_temp_calibration_fields(node.machine, q.name)

            if node.outcomes[q.name] == "failed":
                # -----------------------------
                # Handle failed calibration
                # -----------------------------
                # Get error code
                error_code = QubitSpectroscopyErrorCode(node.results["fit_results"][q.name].get("error_code", 0))

                if node.parameters.use_adaptive_span:
                    # Get current adaptive parameters
                    current_span = node.parameters.frequency_span_in_mhz
                    if temp_data.adaptive_frequency_span_mhz is not None:
                        current_span = temp_data.adaptive_frequency_span_mhz

                    # Get current power shift
                    current_power_shift = temp_data.adaptive_power_shift_dbm or 0.0

                    # No peak at all → increase frequency span AND power
                    new_span = min(current_span * FREQUENCY_SPAN_EXPANSION_FACTOR, MAX_FREQUENCY_SPAN_MHZ)
                    new_power_shift = current_power_shift + POWER_INCREASE_DBM

                    # Update frequency span
                    if new_span < MAX_FREQUENCY_SPAN_MHZ:
                        temp_data.adaptive_frequency_span_mhz = new_span
                        span_msg = f"  Current span: {current_span:.1f} MHz\n  New span:     {new_span:.1f} MHz"
                    else:
                        span_msg = f"  Frequency span: {MAX_FREQUENCY_SPAN_MHZ:.1f} MHz (max reached)"

                    # Update power shift
                    if node.parameters.max_power_dbm + new_power_shift <= MAX_POWER_DBM:
                        temp_data.adaptive_power_shift_dbm = new_power_shift
                        power_msg = f"  Current power shift: {current_power_shift:.1f} dBm\n  New power shift:     {new_power_shift:.1f} dBm"
                    else:
                        power_msg = f"  Power shift: {current_power_shift:.1f} dBm (max reached)"

                    # Update fit results with corrective actions
                    expansion_percent = int((FREQUENCY_SPAN_EXPANSION_FACTOR - 1) * 100)
                    node.results["fit_results"][q.name]["corrective_action"] = int(QubitSpectroscopyCorrectiveAction.EXPAND_FREQUENCY_SPAN)
                    node.results["fit_results"][q.name]["action_magnitude"] = float(expansion_percent)

                    node.log(
                        f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                        f"  CORRECTIVE ACTION: EXPAND_FREQUENCY_SPAN + INCREASE_POWER\n"
                        f"{span_msg}\n"
                        f"{power_msg}"
                    )
                else:
                    node.log(
                        f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                        f"  No adaptive adjustments (disabled by parameter)"
                    )

                continue

            # -----------------------------
            # Check for over-saturation (even if calibration succeeded)
            # -----------------------------
            is_over_saturated = node.results["fit_results"][q.name].get("over_saturated", False)
            error_code = QubitSpectroscopyErrorCode(node.results["fit_results"][q.name].get("error_code", 0))

            if is_over_saturated and node.parameters.use_adaptive_span:
                # Get current power shift (fields already initialized at top of loop)
                current_shift = temp_data.adaptive_power_shift_dbm or 0.0

                # Apply additional power reduction
                new_shift = current_shift + POWER_DECREASE_DBM

                # Check if we can reduce further
                if node.parameters.min_power_dbm + new_shift >= MIN_POWER_DBM:
                    temp_data.adaptive_power_shift_dbm = new_shift

                    # Update fit results with corrective action
                    node.results["fit_results"][q.name]["corrective_action"] = int(QubitSpectroscopyCorrectiveAction.DECREASE_POWER)
                    node.results["fit_results"][q.name]["action_magnitude"] = float(POWER_DECREASE_DBM)

                    node.log(
                        f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                        f"  CORRECTIVE ACTION: DECREASE_POWER ({POWER_DECREASE_DBM} dBm)\n"
                        f"  Current power shift: {current_shift:.1f} dBm\n"
                        f"  New power shift:     {new_shift:.1f} dBm\n"
                        f"  Next min power:      {node.parameters.min_power_dbm + new_shift:.1f} dBm\n"
                        f"  Next max power:      {node.parameters.max_power_dbm + new_shift:.1f} dBm"
                    )
                else:
                    node.results["fit_results"][q.name]["corrective_action"] = int(QubitSpectroscopyCorrectiveAction.NONE)
                    node.log(
                        f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                        f"  CORRECTIVE ACTION: NONE (minimum power limit reached: {MIN_POWER_DBM} dBm)"
                    )

                # Don't update state parameters if over-saturated - retry with lower power
                continue

            # -----------------------------
            # Successful calibration: reset adaptive parameters
            # -----------------------------
            # (fields already initialized at top of loop)
            temp_data.adaptive_frequency_span_mhz = None
            temp_data.adaptive_power_shift_dbm = None
            temp_data.adaptive_num_shots = None

            # Update fit results with success corrective action
            node.results["fit_results"][q.name]["corrective_action"] = int(QubitSpectroscopyCorrectiveAction.RESET_ADAPTIVE_PARAMS)
            node.results["fit_results"][q.name]["action_magnitude"] = 0.0

            # -----------------------------
            # Selected power (dBm)
            # -----------------------------
            selected_power_dbm = node.results["fit_results"][q.name]["selected_power"]

            # -----------------------------
            # Selected qubit frequency (Hz)
            # -----------------------------
            selected_freq = node.results["fit_results"][q.name]["rough_qubit_frequency"]

            # -----------------------------
            # Update XY output power
            # -----------------------------
            # x180: set to the broadening-fit power (short pi pulse for time Rabi).
            # saturation: set to the spectroscopy selected power (with buffer),
            #             using the SAME Octave gain as x180 so the gain never
            #             needs changing between nodes.
            x180_power_dbm = node.results["fit_results"][q.name].get("x180_power_dbm", float("nan"))

            if np.isfinite(x180_power_dbm):
                # Step 1: set x180 → determines the Octave/FEM gain.
                # The extrapolated x180 power may exceed the hardware ceiling
                # (e.g. 10 dBm on MW-FEM channels).  Fall back to max_power_dbm
                # if set_output_power rejects the value.
                try:
                    new_power_settings = q.xy.set_output_power(
                        power_in_dbm=x180_power_dbm,
                        max_amplitude=node.parameters.max_amplitude_opx,
                        operation="x180",
                    )
                except ValueError:
                    node.log(
                        f"[{q.name}] x180 power {x180_power_dbm:.1f} dBm exceeds hardware "
                        f"limit; clamping to max_power_dbm = {node.parameters.max_power_dbm:.1f} dBm"
                    )
                    x180_power_dbm = node.parameters.max_power_dbm
                    new_power_settings = q.xy.set_output_power(
                        power_in_dbm=x180_power_dbm,
                        max_amplitude=node.parameters.max_amplitude_opx,
                        operation="x180",
                    )
                # Step 2: set saturation at selected_power using the same
                # gain/full_scale_power already written to the channel by step 1.
                if hasattr(q.xy, "frequency_converter_up"):
                    # IQ + Octave (OPX+): lock the Octave gain
                    q.xy.set_output_power(
                        power_in_dbm=selected_power_dbm,
                        gain=q.xy.frequency_converter_up.gain,
                        operation=node.parameters.operation,
                    )
                else:
                    # MW-FEM (OPX1000): lock the full_scale_power_dbm
                    q.xy.set_output_power(
                        power_in_dbm=selected_power_dbm,
                        full_scale_power_dbm=q.xy.opx_output.full_scale_power_dbm,
                        operation=node.parameters.operation,
                    )
                power_source = "x180 from broadening fit; saturation at selected power"
                drive_power_dbm = x180_power_dbm  # used for logging / temp_data
            else:
                # Fallback: both at selected power.
                new_power_settings = q.xy.set_output_power(
                    power_in_dbm=selected_power_dbm,
                    max_amplitude=node.parameters.max_amplitude_opx,
                    operation=node.parameters.operation,
                )
                q.xy.set_output_power(
                    power_in_dbm=selected_power_dbm,
                    max_amplitude=node.parameters.max_amplitude_opx,
                    operation="x180",
                )
                power_source = "selected power (fallback)"
                drive_power_dbm = selected_power_dbm

            # Save selected power and the Octave gain used to reach it so that
            # downstream nodes (e.g. power_rabi) know the full RF chain state.
            temp_data.selected_power_dbm = drive_power_dbm
            temp_data.selected_octave_gain_db = float(new_power_settings.get("gain", 0.0))

            # -----------------------------
            # Update qubit frequency
            # -----------------------------
            q.xy.RF_frequency = selected_freq
            q.f_01 = selected_freq

            # -----------------------------
            # Save IQ rotation angle (integration weight angle)
            # -----------------------------
            iw_angle = node.results["fit_results"][q.name].get("iw_angle", float("nan"))
            if np.isfinite(iw_angle):
                prev_angle = q.resonator.operations["readout"].integration_weights_angle
                q.resonator.operations["readout"].integration_weights_angle = (
                    (prev_angle + iw_angle) % (2 * np.pi)
                )

            # -----------------------------
            # Logging
            # -----------------------------
            x180_pwr = node.results["fit_results"][q.name].get("x180_power_dbm", float("nan"))
            x180_str = f"{x180_pwr:.1f} dBm" if np.isfinite(x180_pwr) else "N/A"
            node.log(
                f"[{q.name}] ERROR CODE: {error_code.name} ({error_code.value})\n"
                f"  CORRECTIVE ACTION: RESET_ADAPTIVE_PARAMS\n"
                f"  Updated state:\n"
                f"    Drive power       = {drive_power_dbm:.2f} dBm  ({power_source})\n"
                f"    Octave/FEM power  = {new_power_settings.get('gain', new_power_settings.get('full_scale_power_dbm', float('nan'))):.2f}"
                f"  {'dB gain' if 'gain' in new_power_settings else 'dBm full-scale'}  (saved to temp_calibration)\n"
                f"    Pulse amplitude   = {new_power_settings['amplitude']:.4f}\n"
                f"    Qubit frequency   = {selected_freq / 1e9:.6f} GHz\n"
                f"    x180/sat power    = {x180_str}"
            )




# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
