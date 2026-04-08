# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.f0g1_time_rabi import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        F0G1 TIME RABI (DURATION SWEEP)
Sweeps the f0g1 sideband drive duration while the qubit is prepared in |f>.
A Rabi-like oscillation is observed in the qubit state population, from which
the pi-pulse duration is extracted (first minimum of the fitted sinusoid).

Sequence:
  1. Wait thermalization time (2x T1)
  2. pi_ge  ->  |e>
  3. pi_ef  ->  |f>
  4. Play f0g1 pulse with varying duration
  5. pi_ef  (back-swap)
  6. Measure qubit state

Prerequisites:
    - Calibrated f0g1 sideband frequency (node 21).
    - Rough f0g1 pi-pulse amplitude calibrated (node 22).

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].sideband_drive.operations[operation].length  ->  pi-pulse duration [ns].
"""

node = QualibrationNode[Parameters, Quam](
    name="05_f0g1_time_rabi",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.mode_name = "alice"
    pass


node.machine = Quam.load()


def _get_sideband_drive(node):
    """Return the sideband_drive channel for the cavity_transmon_pair whose
    cavity_mode_name matches node.parameters.mode_name."""
    mode_name = node.parameters.mode_name
    for pair in node.machine.cavity_transmon_pairs.values():
        if pair.cavity_mode_name == mode_name:
            return pair.sideband_drive
    raise KeyError(f"No cavity_transmon_pair with cavity_mode_name='{mode_name}'")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    durations_ns = np.arange(
        node.parameters.min_duration_ns,
        node.parameters.max_duration_ns,
        node.parameters.duration_step_ns,
    )
    durations_cc = (durations_ns // 4).astype(int)

    sideband_drive = _get_sideband_drive(node)
    op = node.parameters.operation

    # Cavity thermalization in clock cycles (computed once outside QUA loops).
    if node.parameters.cavity_thermalization_time_ns is not None:
        therm_clk = int(min(max(node.parameters.cavity_thermalization_time_ns // 4, 4), 2_500_000_000))
    else:
        cav_mode = next(
            (getattr(cav, node.parameters.mode_name, None)
             for cav in node.machine.cavities.values()
             if getattr(cav, node.parameters.mode_name, None) is not None),
            None,
        )
        therm_clk = int(min(max(
            cav_mode.T1 * cav_mode.thermalization_time_factor * 1e9 / 4, 4
        ), 2_500_000_000)) if cav_mode is not None else 4

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration_cc": xr.DataArray(
            durations_cc, attrs={"long_name": "f0g1 pulse duration", "units": "clock cycles"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        n_st = declare_stream()

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        else:
            I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(t, durations_cc)):
                    for i, qubit in multiplexed_qubits.items():
                        # Prepare |f>: pi_ge then pi_ef
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + qubit.anharmonicity
                        )
                        qubit.xy.play("EF_x180")

                        # f0g1 drive with swept duration
                        align(qubit.xy.name, sideband_drive.name)
                        sideband_drive.play(op, duration=t)

                        # Back-swap (ef tone still set)
                        align(sideband_drive.name, qubit.xy.name)
                        qubit.xy.play("EF_x180")

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        # Thermalise cavity and qubit after each point.
                        sideband_drive.wait(therm_clk)
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(durations_cc)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(durations_cc)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(durations_cc)).average().save(f"Q{i + 1}")


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
    node.results["figures"] = {"f0g1_time_rabi": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    sideband_drive = _get_sideband_drive(node)
    op = node.parameters.operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            pi_duration_ns = node.results["fit_results"][q.name]["pi_duration_ns"]
            sideband_drive.operations[op].length = int(round(pi_duration_ns))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
