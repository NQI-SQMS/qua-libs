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
from calibration_utils.time_rabi_ef import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        EF TIME RABI
This sequence prepares the qubit in |e⟩ via a ge x180 pulse, then plays the EF drive pulse
with a variable duration at the e→f transition frequency, and applies a final ge x180 before
readout for improved readout fidelity.

The result is a Rabi oscillation in the I quadrature from which the EF π-pulse duration is
extracted.

Prerequisites:
    - Having calibrated the ge x180 pulse (nodes 03a, 04b/04c).
    - Having found the EF transition frequency (node 12).
    - Having a defined EF drive operation (e.g., EF_x180) in the QUAM state.

State update:
    - The EF pi-pulse duration: qubit.xy.operations[ef_x180_operation].length
"""

node = QualibrationNode[Parameters, Quam](
    name="04d_time_rabi_ef",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.min_duration_ns = 16
    # node.parameters.max_duration_ns = 2000
    # node.parameters.duration_step_ns = 4
    # node.parameters.ef_x180_operation = "EF_x180"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the EF time Rabi QUA program and register sweep axes."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    ef_op = node.parameters.ef_x180_operation
    op_amp_factor = node.parameters.operation_amplitude_factor

    # Duration sweep in clock cycles (4 ns each); enforce multiples of 4 ns
    min_cc = max(4, (node.parameters.min_duration_ns // 4))
    max_cc = max(min_cc + 1, (node.parameters.max_duration_ns // 4))
    step_cc = max(1, (node.parameters.duration_step_ns // 4))
    durations_cc = np.arange(min_cc, max_cc, step_cc, dtype=int)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration_cc": xr.DataArray(
            durations_cc,
            attrs={"long_name": "EF pulse duration", "units": "clock cycles"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable: EF pulse duration in clock cycles

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, durations_cc)):
                    # Wait twice the thermalization time for proper |f> state reset
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(2 * qubit.thermalization_time * u.ns)
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # Step 1: prepare |e⟩ via ge pi pulse
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")
                        # Step 2: set IF to EF transition and play EF pulse with variable duration
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + qubit.anharmonicity)
                        qubit.xy.play(ef_op, duration=t, amplitude_scale=op_amp_factor)
                        # Step 3: reset to ge frequency and apply ge pi pulse for readout fidelity
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")
                    align()

                    # Readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(durations_cc)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(durations_cc)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(durations_cc)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
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
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit EF Rabi oscillations and extract the EF π-pulse duration."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the EF time Rabi oscillations with fitted curves."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
    )
    plt.show()
    node.results["figures"] = {"time_rabi_ef": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the EF pi-pulse duration if the fit was successful."""
    ef_op = node.parameters.ef_x180_operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] != "successful":
                continue
            pi_dur = node.results["fit_results"][q.name]["pi_duration_ns"]
            if np.isfinite(pi_dur) and pi_dur > 0:
                pi_dur_int = int(pi_dur)
                q.xy.operations[ef_op].length = pi_dur_int
                node.log(f"[{q.name}] Updated {ef_op} duration: {pi_dur_int} ns")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
