# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibration_libs.parameters import (
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.shared import apply_confusion_matrix_correction, _get_cavity_mode
from calibration_utils.cavity_coherent_T2 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cavity_coherent_T2.parameters import Parameters

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit for amplitude_scale

# %% {Node initialisation}
description = """
        CAVITY MODE T2 RAMSEY (COHERENT STATE)
Measures the T2 Ramsey coherence time of a cavity mode (e.g. alice or bob) by
creating a coherent state and performing a displacement-based Ramsey experiment.

The artificial detuning is encoded directly in the cavity drive frequency: both
the forward and reverse displacements are played at (cavity_IF + ramsey_detuning_hz).
During the wait τ the cavity evolves at its natural frequency, so the reverse
displacement sees a phase offset of 2π × detuning × τ.  This produces Ramsey fringes
without an explicit frame rotation.

Sequence:
  1. Reset cavity and qubit
  2. Shift cavity drive to (cavity_IF + ramsey_detuning_hz)
  3. Displace cavity → |α⟩
  4. Wait variable time tau
  5. Reverse displace (same amplitude, opposite sign)
  6. Reset cavity drive frequency
  7. π pulse on qubit  (non-selective x180)
  8. Measure qubit state

P(|e⟩) vs tau follows a decaying sinusoid with time constant T2ramsey.

Prerequisites:
    - Calibrated displacement amplitude (node 21 or equivalent).

Parameters:
    - mode_name:             'alice' or 'bob'
    - displacement_alpha:    Coherent state amplitude |α|
    - ramsey_detuning_hz:    Artificial detuning for Ramsey fringes [Hz]
    - idle_time_*:           Range of wait times

State update:
    - cavity_mode.T2ramsey  (seconds)
"""



node = QualibrationNode[Parameters, Quam](
    name="27_cavity_coherent_T2",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.displacement_alpha = 1.0
    # node.parameters.ramsey_detuning_hz = 1000.0
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)  # in clock cycles (4 ns)

    cavity_mode = _get_cavity_mode(node)

    # Resolve sideband_drive and displacement_alpha_max from the CavityTransmonPair QuAM state
    mode_name = node.parameters.mode_name
    sideband_drive = None
    alpha_max = 1.0
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}"):
            if getattr(pair, "sideband_drive", None) is not None:
                sideband_drive = pair.sideband_drive
            if getattr(pair, "displacement_alpha_max", None) is not None:
                alpha_max = float(pair.displacement_alpha_max)
            break
    node.namespace["sideband_drive"] = sideband_drive
    amplitude_scale = node.parameters.displacement_alpha / alpha_max
    if abs(amplitude_scale) > _AMP_MAX:
        raise ValueError(
            f"displacement_alpha={node.parameters.displacement_alpha} / "
            f"alpha_max={alpha_max} = {amplitude_scale:.4f} exceeds the QUA "
            f"hardware limit ±{_AMP_MAX:.6f}.  Reduce displacement_alpha."
        )
    node.namespace["amplitude_scale"] = amplitude_scale
    node.log(
        f"Displacement: alpha={node.parameters.displacement_alpha}, "
        f"alpha_max={alpha_max}, amplitude_scale={amplitude_scale:.4f}"
    )

    cavity_IF = int(cavity_mode.cavity_mode_drive.intermediate_frequency)
    # Phase increment per clock cycle (4 ns) for the Ramsey frame rotation: turns/cc
    detuning_turns_per_cc = node.parameters.ramsey_detuning_hz * 1e-9 * 4
    node.log(f"Ramsey detuning: {node.parameters.ramsey_detuning_hz:.0f} Hz  (cavity_IF {cavity_IF}, {detuning_turns_per_cc:.6f} turns/cc)")

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        phase = declare(fixed)
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

                with for_each_(t, idle_times):
                    # -- 1. Reset cavity and qubit -----------------------------
                    sideband_drive = node.namespace["sideband_drive"]
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

                    # -- 2. Displace cavity → |α⟩ (at natural cavity_IF) ------
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=node.namespace["amplitude_scale"],
                    )

                    # -- 3. Wait tau + apply Ramsey phase via frame rotation ---
                    # frame_rotation_2pi gives a deterministic phase = detuning × t
                    # for every shot at the same t, avoiding the phase drift that
                    # update_frequency accumulates across shots.
                    align()
                    assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))
                    with strict_timing_():
                        cavity_mode.cavity_mode_drive.wait(t)
                        frame_rotation_2pi(phase, cavity_mode.cavity_mode_drive.name)

                    # -- 4. Reverse displacement (with Ramsey phase applied) ---
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=-node.namespace["amplitude_scale"],
                    )
                    reset_frame(cavity_mode.cavity_mode_drive.name)

                    # -- 5. π pulse on qubit -----------------------------------
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play(node.parameters.qubit_probe_operation)

                    # -- 7. Measure --------------------------------------------
                    align()
                    for i, qubit in multiplexed_qubits.items():
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
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


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
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    # process_raw_dataset must run first: it creates the 'state' DataArray from state1/state2/...
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if res["success"] else "failed")
        for q_name, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_coherent_T2": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            T2ramsey_ns = node.results["fit_results"][q.name]["T2ramsey_ns"]
            T2ramsey_s = float(T2ramsey_ns) * 1e-9
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "T2ramsey"):
                    mode.T2ramsey = T2ramsey_s
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
