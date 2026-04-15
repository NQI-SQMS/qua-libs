# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.T1 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        CAVITY MODE T1 MEASUREMENT
Measures the photon lifetime T1 of a cavity mode (e.g. alice or bob) by:
  1. Wait thermalization time (2× T1) — qubit in |g⟩
  2. Prepare qubit in |f⟩ (π_ge + π_ef)
  3. Apply calibrated f0g1 π-pulse: |f,0⟩ → |g,1⟩  (photon created in cavity)
  4. Wait variable time τ
  5. Re-prepare |f⟩ (π_ge + π_ef) and apply f0g1 π-pulse again (photon retrieval)
  6. Back-swap with π_ef
  7. Measure qubit state:
       |e⟩  → photon was still present (retrieved)
       |g⟩  → photon had decayed

Population(|e⟩) vs τ follows an exponential decay with time constant T1_cavity.

Prerequisites:
    - Calibrated f0g1 π-pulse (node 22).

Parameters:
    - mode_name:   'alice' or 'bob' (attribute name on the Cavity object)
    - idle_time_*: range of wait times

State update:
    - cavity_mode.T1  (seconds)
"""


class _ModeParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to measure: attribute name on the Cavity object (e.g. 'alice' or 'bob')."""
    use_state_discrimination: bool = True
    """True → measure qubit state (recommended). False → measure raw I/Q."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    _ModeParameters,
    IdleTimeNodeParameters,
    QubitsExperimentNodeParameters,
):
    pass


node = QualibrationNode[Parameters, Quam](
    name="28_cavity_mode_T1",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
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

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
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
                        # ── 1. Reset qubit ──────────────────────────────────────
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                        # ── 2. Prepare |f⟩ ─────────────────────────────────────
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")                      # |g⟩ → |e⟩
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency - qubit.anharmonicity
                        )
                        qubit.xy.play("x180")                      # |e⟩ → |f⟩

                        # ── 3. Create photon: f0g1 π-pulse |f,0⟩ → |g,1⟩ ──────
                        align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play("f0g1_pi")           # qubit → |g⟩, cavity ← 1 photon

                        # ── 4. Wait τ ───────────────────────────────────────────
                        align(cavity_mode.cavity_mode_drive.name, qubit.resonator.name)
                        qubit.resonator.wait(t)

                        # ── 5. Retrieve photon: prepare |f⟩ again, apply f0g1 π ─
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")                      # |g⟩ → |e⟩
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency - qubit.anharmonicity
                        )
                        qubit.xy.play("x180")                      # |e⟩ → |f⟩
                        align(qubit.xy.name, cavity_mode.cavity_mode_drive.name)
                        cavity_mode.cavity_mode_drive.play("f0g1_pi")           # if photon present: |f,1⟩→|g,0⟩

                        # ── 6. Back-swap: π_ef → qubit in |e⟩ if photon retrieved
                        align(cavity_mode.cavity_mode_drive.name, qubit.xy.name)
                        qubit.xy.play("x180")                      # ef drive: |f⟩→|e⟩

                        # ── 7. Measure ──────────────────────────────────────────
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                            wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)

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

    log_fitted_results(node.results["ds_fit"], log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if fit_result["success"] else "failed")
        for q_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {"cavity_mode_T1": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            T1_s = float(node.results["ds_fit"].sel(qubit=q.name).tau.values) * 1e-9
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "T1"):
                    mode.T1 = T1_s
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
