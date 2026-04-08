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
from calibration_utils.f0g1_spectroscopy import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    update_state,
)

# %% {Node initialisation}
description = """
        F0G1 SPECTROSCOPY
Sweeps the f0g1 sideband drive frequency while the qubit is prepared in |f⟩.
When the sideband drive is resonant, the |f,0⟩ ↔ |g,1⟩ transition is driven;
the qubit is left in |g⟩ and the back-swap π_ef leaves it in |g⟩ → DIP in
state measurement.

Sequence:
  1. Wait thermalization time (2× T1)
  2. π_ge  →  |e⟩
  3. π_ef  →  |f⟩
  4. Sweep f0g1 IF;  play saturation pulse on f0g1 channel
  5. π_ef  (back-swap: |f⟩ → |e⟩ if no photon created; |g⟩ unchanged)
  6. Measure qubit state

Prerequisites:
    - Calibrated ge and ef transitions (nodes 04b, 13).

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].sideband_drive.RF_frequency  →  sideband resonance frequency.
"""

node = QualibrationNode[Parameters, Quam](
    name="04_f0g1_spectroscopy",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.mode_name = "alice"
    # node.parameters.frequency_span_in_mhz = 50
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
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    sideband_drive = _get_sideband_drive(node)
    op = node.parameters.operation
    op_len = node.parameters.operation_len_in_ns

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
        if cav_mode is not None:
            therm_clk = int(min(max(
                cav_mode.T1 * cav_mode.thermalization_time_factor * 1e9 / 4, 4
            ), 2_500_000_000))
        else:
            therm_clk = 4

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "f0g1 detuning", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        f = declare(int)
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

                with for_(*from_array(f, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Prepare |f⟩: π_ge then π_ef
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + qubit.anharmonicity
                        )
                        qubit.xy.play("EF_x180")

                        # Sweep f0g1 sideband frequency and drive
                        sideband_drive.update_frequency(
                            sideband_drive.intermediate_frequency + f
                        )
                        align(qubit.xy.name, sideband_drive.name)
                        if op_len is not None:
                            sideband_drive.play(
                                op,
                                amplitude_scale=node.parameters.operation_amplitude_factor,
                                duration=op_len >> 2,
                            )
                        else:
                            sideband_drive.play(
                                op,
                                amplitude_scale=node.parameters.operation_amplitude_factor,
                            )

                        # Back-swap: π_ef (ef tone still set from above)
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
                    state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


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
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"f0g1_spectroscopy": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state_node(node: QualibrationNode[Parameters, Quam]):
    with node.record_state_updates():
        fit_params = {k: type("FP", (), v)() for k, v in node.results["fit_results"].items()}
        update_state(node, fit_params)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
