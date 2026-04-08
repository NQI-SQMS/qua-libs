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
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.displacement_calibration_vacuum import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_vacuum_calibration,
)

logger = logging.getLogger(__name__)


# %% {Node initialisation}
description = """
        DISPLACEMENT VACUUM-POPULATION CALIBRATION (35)

Calibrates the unit displacement amplitude by sweeping the cavity displacement
amplitude and measuring the vacuum-state population with a selective qubit π-pulse.

Sequence (per displacement amplitude scale a):
  1. Reset cavity (thermal or active sideband cooling) and qubit.
  2. Apply displacement pulse at amplitude_scale = a.
  3. Apply selective_x180 (or x180) on qubit — flips qubit only when cavity is in |0⟩.
  4. Measure qubit state.
  5. Apply D(-a) to return cavity toward vacuum (if active_reset = True).

The measured signal:
    P_e(a) = amplitude · exp(-(a / A_1ph)²) + offset

where A_1ph = sigma is the displacement amplitude_scale that produces exactly 1 photon
on average (n̄ = 1 for a coherent state).

Parameters:
  - mode_name:       Cavity mode to calibrate ('alice' or 'bob').
  - qubit_pulse:     'selective_x180' (spectrally selective, recommended) or 'x180'.
  - amp_min/max:     Amplitude sweep range (amp_min=0 for half-Gaussian).
  - active_reset:    Play D(-a) after measurement to speed up cavity thermalization.

State updates:
  - cavity_mode.cavity_mode_drive.operations["displacement"].amplitude
    (set to base_amp × sigma, so amplitude_scale=1 → 1 photon)
"""

node = QualibrationNode[Parameters, Quam](
    name="06_displacement_calibration_vacuum",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.qubit_pulse = "selective_x180"  # or "x180"
    # node.parameters.amp_min = 0.0
    # node.parameters.amp_max = 2.0
    # node.parameters.amp_points = 51
    # node.parameters.active_reset = True
    # node.parameters.num_shots = 1000
    pass


node.machine = Quam.load()


def _get_cavity_mode(node):
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the QUA program for the vacuum-population displacement calibration."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for active cavity cooling (used if cavity_reset_type='active_sideband')
    mode_name = node.parameters.mode_name
    sideband_drive = None
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}") and getattr(pair, "sideband_drive", None) is not None:
            sideband_drive = pair.sideband_drive
            break
    node.namespace["sideband_drive"] = sideband_drive

    # Amplitude sweep
    amp_array = np.linspace(
        node.parameters.amp_min,
        node.parameters.amp_max,
        node.parameters.amp_points,
    )
    node.namespace["amp_array"] = amp_array

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp": xr.DataArray(
            amp_array,
            attrs={"long_name": "displacement amplitude scale", "units": "a.u."},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)
        a_neg = declare(fixed)

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amp_array)):
                    assign(a_neg, -a)

                    # --- Reset cavity and qubit BEFORE displacement ---
                    sideband_drive = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # --- Displace cavity ---
                    # align() with no args synchronises all QUA elements including
                    # cavity_mode_drive, which is separate from the qubit elements.
                    align()
                    cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a)

                    # --- Probe vacuum population ---
                    # selective_x180 flips the qubit only when cavity is in |0⟩.
                    # x180 is a broadband π-pulse (use for comparison / if selective not calibrated).
                    for i, qubit in multiplexed_qubits.items():
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.play(node.parameters.qubit_pulse)

                    # --- Measure ---
                    for i, qubit in multiplexed_qubits.items():
                        align(qubit.xy.name, qubit.resonator.name)
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    # --- Active reset: D(-a) returns cavity toward vacuum ---
                    if node.parameters.active_reset:
                        align()
                        cavity_mode.cavity_mode_drive.play("displacement", amplitude_scale=a_neg)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amp_array)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amp_array)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amp_array)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    _, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)
    fig = plot_vacuum_calibration(
        node.results["ds_raw"],
        node.results["fit_results"],
        mode_name=node.parameters.mode_name,
        qubit_pulse=node.parameters.qubit_pulse,
        normalize_plot=node.parameters.normalize_plot,
        base_amplitude=base_amp,
    )
    plt.show()
    node.results["figures"] = {"vacuum_calibration": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]
    base_amp = float(cavity_mode.cavity_mode_drive.operations["displacement"].amplitude)

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            sigma = res["sigma"]
            cal_amplitude = base_amp * sigma
            cavity_mode.cavity_mode_drive.operations["displacement"].amplitude = float(cal_amplitude)

            # Write displacement_k to CavityTransmonPair for downstream nodes
            mode_name = node.parameters.mode_name
            pair_key = f"{qubit.name}_{mode_name}"
            pairs = getattr(node.machine, "cavity_transmon_pairs", None)
            if pairs is not None and pair_key in pairs:
                k_fit = 1.0 / (sigma ** 2)  # n̄ = k·A² → k = 1/sigma²
                if hasattr(pairs[pair_key], "displacement_k"):
                    pairs[pair_key].displacement_k = float(k_fit)

            break  # one cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
