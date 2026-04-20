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
    CommonNodeParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibrate.parameters import RunnableParameters
from qualibrate import NodeParameters
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_mode_T2 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        CAVITY MODE T2 RAMSEY
Measures the T2 Ramsey coherence time of a cavity mode (e.g. alice or bob) by
creating a Fock-state superposition and performing a Ramsey experiment:

  1. Wait thermalization time (2x T1)
  2. pi/2_ge  ->  (|g> + |e>) / sqrt(2)
  3. pi_ef    ->  (|g> + |f>) / sqrt(2)  [on ef frequency]
  4. f0g1 pi  ->  (|g,0> + |g,1>) / sqrt(2)  [Fock superposition in cavity]
  5. Wait variable time tau
  6. Frame rotation by (detuning_hz * 1e-9 * 4 * tau) turns  [imprint phase]
  7. f0g1 pi  ->  retrieve photon if present
  8. pi_ef + pi/2_ge  ->  project back to ge basis
  9. Measure qubit state

Population(|e>) vs tau follows a decaying sinusoid with time constant T2ramsey.

Prerequisites:
    - Calibrated f0g1 pi-pulse (node 22 or 24).

Parameters:
    - mode_name:         'alice' or 'bob'
    - ramsey_detuning_hz: artificial detuning for Ramsey fringes [Hz]
    - idle_time_*:       range of wait times

State update:
    - cavity_mode.T2ramsey  (seconds)
"""


class _ModeParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to measure: attribute name on the Cavity object."""
    ramsey_detuning_hz: float = 1000.0
    """Artificial detuning applied via frame rotation [Hz]."""
    use_state_discrimination: bool = True
    """True -> measure qubit state (recommended). False -> measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    _ModeParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass


node = QualibrationNode[Parameters, Quam](
    name="29_cavity_mode_T2",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.ramsey_detuning_hz = 1000.0
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
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)  # in clock cycles (4 ns)

    cavity_mode = _get_cavity_mode(node)

    # Phase increment per clock cycle for the frame rotation:
    # detuning_hz * 1e-9 * 4 ns/cc = turns per clock cycle
    detuning_turns_per_cc = node.parameters.ramsey_detuning_hz * 1e-9 * 4

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
                    for i, qubit in multiplexed_qubits.items():
                        # ── 1. Reset ─────────────────────────────────────────
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                        # ── 2. pi/2_ge: |g> -> (|g>+|e>)/sqrt(2) ────────────
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x90")

                        # ── 3. pi_ef: |e> -> |f> ─────────────────────────────
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency - qubit.anharmonicity
                        )
                        qubit.xy.play("x180")

                        # ── 4. f0g1 pi: |f,0> -> |g,1> (photon in cavity) ────
                        align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play("f0g1_pi")

                        # ── 5. Wait tau ───────────────────────────────────────
                        align(cavity_mode.cavity_mode_drive.name, qubit.resonator.name)
                        qubit.resonator.wait(t)

                        # ── 6. Frame rotation (artificial detuning) ───────────
                        assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))
                        frame_rotation_2pi(phase, cavity_mode.cavity_mode_drive.name)

                        # ── 7. f0g1 pi: retrieve photon ───────────────────────
                        align(qubit.resonator.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play("f0g1_pi")
                        reset_frame(cavity_mode.cavity_mode_drive.name)

                        # ── 8. Unwind: pi_ef + pi/2_ge ────────────────────────
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.play("x180")    # ef: |f> -> |e>
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x90")     # ge pi/2: project to |g>/|e>

                        # ── 9. Measure ────────────────────────────────────────
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
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
    node.results["figures"] = {"cavity_mode_T2": fig}


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
