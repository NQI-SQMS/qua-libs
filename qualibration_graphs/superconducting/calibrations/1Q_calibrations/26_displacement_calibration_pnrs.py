# %% {Imports}
import logging
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
from calibration_utils.shared import apply_confusion_matrix_correction, _get_cavity_mode
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from calibration_utils.power_lock import set_locked_output_power
from qualibration_libs.core import tracked_updates
from quam_config import Quam
from calibration_utils.displacement_calibration_pnrs import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_spectrum_at_power,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

logger = logging.getLogger(__name__)


# %% {Description}
description = """
        DISPLACEMENT CALIBRATION — PHOTON NUMBER RESOLVED SPECTROSCOPY METHOD (28)

Sweeps the cavity displacement pulse amplitude A (outer loop) and the qubit
ge spectroscopy frequency (inner loop).  For each amplitude A a spectrum is
acquired; N Gaussian peaks are fitted to extract the photon-number distribution
P(n).  The mean photon number n̄ = Σ n·P(n) is computed for every A, and the
calibration curve n̄ = k·A² is fitted to obtain:

    A₁ph = 1/√k  — displacement amplitude_scale that deposits 1 photon on average

The dispersive shift chi = -(PNRS peak spacing) [Hz] is also extracted from the spectra.

Physical model:
    ω_q(n) = ω_q + chi·n   (each photon shifts the qubit by chi; chi < 0)
    P(n)   = exp(-n̄) · n̄ⁿ / n!   (coherent state: Poisson distribution)

Prerequisites:
    - Calibrated selective_x180 pulse (node 14b).
    - Calibrated cavity drive frequency (node 27, recommended).
    - Displacement pulse operation present on cavity_mode_drive.

State updates:
    - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude  (= A₁ph)
    - cavity_transmon_pairs["{qubit}_{mode}"].chi
    - cavity_transmon_pairs["{qubit}_{mode}"].displacement_k  (= k)
"""

node = QualibrationNode[Parameters, Quam](
    name="26_displacement_calibration_pnrs",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.amp_points = 21
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    step_hz = int(node.parameters.frequency_step_in_mhz * u.MHz)
    left_hz = int(node.parameters.left_span_mhz * u.MHz)
    right_hz = int(node.parameters.right_offset_mhz * u.MHz)
    dfs = np.arange(-left_hz, right_hz, step_hz)

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for active cavity cooling (used only if requested)
    mode_name = node.parameters.mode_name
    sideband_drive = None
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}") and getattr(pair, "sideband_drive", None) is not None:
            sideband_drive = pair.sideband_drive
            break
    node.namespace["sideband_drive"] = sideband_drive

    # Set cavity drive to max power (tracked so it can be reverted after the experiment)
    node.namespace["tracked_drives"] = []
    with tracked_updates(cavity_mode.cavity_mode_drive, auto_revert=False, dont_assign_to_none=True) as cav_drive:
        set_locked_output_power(cav_drive, power_in_dbm=node.parameters.max_power_dbm, operation="displacement")
        node.namespace["tracked_drives"].append(cav_drive)

    # Amplitude prefactors: logarithmically spaced from min_power to max_power
    if node.parameters.min_power_dbm >= node.parameters.max_power_dbm:
        raise ValueError(
            f"min_power_dbm ({node.parameters.min_power_dbm}) must be less than "
            f"max_power_dbm ({node.parameters.max_power_dbm}). "
            "min_power_dbm is the lowest power (most negative dBm)."
        )
    amp_min = calculate_voltage_scaling_factor(node.parameters.max_power_dbm, node.parameters.min_power_dbm)
    amps = np.geomspace(amp_min, 1.0, node.parameters.num_power_points)
    power_dbm = np.linspace(node.parameters.min_power_dbm, node.parameters.max_power_dbm, node.parameters.num_power_points)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "power": xr.DataArray(
            power_dbm, attrs={"long_name": "displacement power", "units": "dBm"}
        ),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        a = declare(fixed)
        df = declare(int)
        n_st = declare_stream()

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    with for_(*from_array(df, dfs)):
                        for i, qubit in multiplexed_qubits.items():
                            cavity_mode.reset(
                                node.parameters.cavity_reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                                sideband_drive=sideband_drive,
                                qubit_thermalization_time=qubit.thermalization_time,
                                fock_n=node.parameters.cavity_active_cooling_fock_n,
                                sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            )
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )

                            # Apply displacement
                            dur_ns = node.parameters.displacement_pulse_duration_ns
                            if dur_ns is not None:
                                cavity_mode.cavity_mode_drive.play(
                                    "displacement", duration=dur_ns // 4, amplitude_scale=a
                                )
                            else:
                                cavity_mode.cavity_mode_drive.play(
                                    "displacement", amplitude_scale=a
                                )

                            # Qubit spectroscopy with selective pulse
                            align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                            qubit.xy.update_frequency(
                                df + qubit.xy.intermediate_frequency
                            )
                            qubit.xy.play("selective_x180")

                            # Measure
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                            if node.parameters.use_state_discrimination:
                                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                                save(state[i], state_st[i])
                            qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(
                        f"state{i + 1}"
                    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["cavity_mode"] = _get_cavity_mode(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    figures = {}
    figures["displacement_calibration_pnrs"] = plot_raw_data_with_fit(
        node.results["ds_fit"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()

    if node.parameters.selected_power_dbm is not None:
        figures["pns_spectrum_slice"] = plot_spectrum_at_power(
            node.results["ds_fit"],
            node.namespace["qubits"],
            selected_power_dbm=node.parameters.selected_power_dbm,
            fit_results=node.results["fit_results"],
            mode_name=node.parameters.mode_name,
        )
        plt.show()

    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    mode_name = node.parameters.mode_name

    # Read the base amplitude set by set_output_power (= max_amp) before reverting
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)

    # Revert the set_output_power change so the QUAM state is clean before recording updates
    for tracked_drive in node.namespace.get("tracked_drives", []):
        tracked_drive.revert_changes()

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            amp_one_scale = res["amp_for_one_photon"]  # scale factor relative to base_amp
            chi_hz = res["chi_hz"]
            k = res["k"]

            # 1-photon voltage; then scale up to fill DAC headroom
            cal_amplitude_1ph = base_amp * amp_one_scale
            MAX_VOLTAGE = 0.5
            AMP_SCALE_LIMIT = 1.9
            alpha_max = MAX_VOLTAGE / (cal_amplitude_1ph * AMP_SCALE_LIMIT)
            cal_amplitude = cal_amplitude_1ph * alpha_max  # ≈ 0.263 V

            cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = float(cal_amplitude)

            # Update CavityTransmonPair (create on-the-fly if missing)
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None:
                if pair_key not in pairs:
                    from quam_builder.architecture.superconducting.qubit_pair import CavityTransmonPair
                    pairs[pair_key] = CavityTransmonPair(
                        qubit_name=qubit.name, cavity_mode_name=mode_name
                    )
                    logger.info(f"Created CavityTransmonPair '{pair_key}' on the fly.")
                pairs[pair_key].chi = float(chi_hz)
                pairs[pair_key].displacement_k = float(k)
                if hasattr(pairs[pair_key], "displacement_alpha_max"):
                    pairs[pair_key].displacement_alpha_max = float(alpha_max)
            else:
                logger.warning(
                    f"machine has no cavity_transmon_pairs field — "
                    f"chi/k for '{pair_key}' not persisted to QUAM. "
                    "Check that SrfQuam is the active QUAM class."
                )

            logger.info(
                f"Displacement (PNRS) calibration: alpha_max={alpha_max:.3f}, "
                f"stored amplitude={cal_amplitude:.6f} V  "
                f"(amplitude_scale=1 -> {alpha_max:.2f} photons)"
            )

            break  # one cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
