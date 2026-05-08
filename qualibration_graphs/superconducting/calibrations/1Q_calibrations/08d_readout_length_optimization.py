# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict
import xarray as xr
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.readout_length_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_fidelity_vs_length,
)


# %% {Node initialisation}
description = """
        READOUT LENGTH OPTIMIZATION

Finds the optimal readout pulse duration by maximising the g/e state discrimination fidelity.

Uses accumulated demodulation: within a single readout pulse the IQ signal is accumulated
in chunks of `division_length_in_cc` clock cycles (= 4 ns each). For each shot both the
ground state (after thermalization) and the excited state (after x180) are measured. The
two-state discriminator is applied at each cumulative length to compute fidelity vs time.
The readout pulse length is then updated to the length that gives the highest fidelity.

Note: integration weight names ("rotated_cos", "rotated_sin", "rotated_minus_sin") can be
adjusted via parameters if your readout operation uses different names.

Prerequisites:
    - Calibrated readout frequency and power (nodes 08a, 08b).
    - Calibrated x180 pulse (node 04b or 04c).
    - Calibrated IQ rotation angle (node 07_iq_blobs).

State update:
    - qubit.resonator.operations[readout_operation].length (in ns)
"""

node = QualibrationNode[Parameters, Quam](
    name="08d_readout_length_optimization",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 10000
    # node.parameters.division_length_in_cc = 4
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    division_length = node.parameters.division_length_in_ns // 4  # convert ns → clock cycles
    readout_op = node.parameters.readout_operation
    cos_w = node.parameters.cos_weight_name
    sin_w = node.parameters.sin_weight_name
    minus_sin_w = node.parameters.minus_sin_weight_name

    # Apply the requested readout length to all qubits before building the config
    for qubit in qubits:
        qubit.resonator.operations[readout_op].length = node.parameters.max_readout_length_in_ns

    # Compute number of divisions from the (now-updated) readout pulse length
    first_qubit = list(qubits)[0]
    readout_length_ns = first_qubit.resonator.operations[readout_op].length
    number_of_divisions = int(readout_length_ns // (4 * division_length))
    node.namespace["number_of_divisions"] = number_of_divisions
    node.namespace["division_length_ns"] = 4 * division_length

    # No sweep axis — all lengths obtained from one set of shots per state
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
    }
    node.namespace["n_avg"] = n_avg

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        ind = declare(int)

        # Declare per-qubit accumulated demod arrays and streams
        II_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        IQ_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QI_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QQ_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        II_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        IQ_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QI_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QQ_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        I_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Q_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        I_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Q_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Ig_st = [declare_stream() for _ in range(num_qubits)]
        Qg_st = [declare_stream() for _ in range(num_qubits)]
        Ie_st = [declare_stream() for _ in range(num_qubits)]
        Qe_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # --- Reset ---
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()

                # --- Readout (|g> state) ---
                for i, qubit in multiplexed_qubits.items():
                    measure(
                        readout_op,
                        qubit.resonator.name,
                        None,
                        demod.accumulated(cos_w, II_g[i], division_length, "out1"),
                        demod.accumulated(sin_w, IQ_g[i], division_length, "out2"),
                        demod.accumulated(minus_sin_w, QI_g[i], division_length, "out1"),
                        demod.accumulated(cos_w, QQ_g[i], division_length, "out2"),
                    )
                    with for_(ind, 0, ind < number_of_divisions, ind + 1):
                        assign(I_g[i][ind], II_g[i][ind] + IQ_g[i][ind])
                        save(I_g[i][ind], Ig_st[i])
                        assign(Q_g[i][ind], QQ_g[i][ind] + QI_g[i][ind])
                        save(Q_g[i][ind], Qg_st[i])
                    qubit.resonator.wait(qubit.thermalization_time // 4)

                align()

                # --- Qubit drive (|e> state preparation) ---
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()

                # --- Readout (|e> state) ---
                for i, qubit in multiplexed_qubits.items():
                    measure(
                        readout_op,
                        qubit.resonator.name,
                        None,
                        demod.accumulated(cos_w, II_e[i], division_length, "out1"),
                        demod.accumulated(sin_w, IQ_e[i], division_length, "out2"),
                        demod.accumulated(minus_sin_w, QI_e[i], division_length, "out1"),
                        demod.accumulated(cos_w, QQ_e[i], division_length, "out2"),
                    )
                    with for_(ind, 0, ind < number_of_divisions, ind + 1):
                        assign(I_e[i][ind], II_e[i][ind] + IQ_e[i][ind])
                        save(I_e[i][ind], Ie_st[i])
                        assign(Q_e[i][ind], QQ_e[i][ind] + QI_e[i][ind])
                        save(Q_e[i][ind], Qe_st[i])
                    qubit.resonator.wait(qubit.thermalization_time // 4)

                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ig_st[i].buffer(number_of_divisions).save_all(f"Ig{i + 1}")
                Qg_st[i].buffer(number_of_divisions).save_all(f"Qg{i + 1}")
                Ie_st[i].buffer(number_of_divisions).save_all(f"Ie{i + 1}")
                Qe_st[i].buffer(number_of_divisions).save_all(f"Qe{i + 1}")


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
    qubits = node.namespace["qubits"]
    n_avg = node.namespace["n_avg"]
    n_div = node.namespace["number_of_divisions"]

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # XarrayDataFetcher cannot handle save_all (2-D) streams — poll n directly
        import time as _time
        result_handles = job.result_handles
        n_handle = result_handles.get("n")
        t_start = _time.time()
        while result_handles.is_processing():
            n_fetched = n_handle.fetch_all()
            count = int(np.atleast_1d(n_fetched)[-1]) if n_fetched is not None else 0
            progress_counter(count, n_avg, start_time=t_start)
            _time.sleep(0.1)
        # Wait for full streaming before the session closes
        result_handles.wait_for_all_values()
        node.log(job.execution_report())

    def _fetch_plain(handle):
        """Fetch all data and unwrap structured dtype from save_all streams."""
        data = handle.fetch_all()
        if data is not None and data.dtype.names and "value" in data.dtype.names:
            data = data["value"]
        return np.atleast_2d(data).astype(float)

    # Build per-qubit arrays from the raw results (all data already streamed)
    ds_dict = {}
    for i, q in enumerate(qubits):
        Ig = _fetch_plain(result_handles.get(f"Ig{i + 1}"))
        Qg = _fetch_plain(result_handles.get(f"Qg{i + 1}"))
        Ie = _fetch_plain(result_handles.get(f"Ie{i + 1}"))
        Qe = _fetch_plain(result_handles.get(f"Qe{i + 1}"))
        ds_dict[f"Ig_{q.name}"] = xr.DataArray(Ig, dims=["n_runs", "division"])
        ds_dict[f"Qg_{q.name}"] = xr.DataArray(Qg, dims=["n_runs", "division"])
        ds_dict[f"Ie_{q.name}"] = xr.DataArray(Ie, dims=["n_runs", "division"])
        ds_dict[f"Qe_{q.name}"] = xr.DataArray(Qe, dims=["n_runs", "division"])

    node.results["ds_raw"] = xr.Dataset(ds_dict)


# %% {Load_historical_data}
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
        q: ("successful" if r["success"] else "failed")
        for q, r in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_fidelity_vs_length(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results["fit_results"],
    )
    plt.show()
    node.results["figures"] = {"fidelity_vs_length": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    readout_op = node.parameters.readout_operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            optimal_length_ns = node.results["fit_results"][q.name]["optimal_readout_length_ns"]
            q.resonator.operations[readout_op].length = optimal_length_ns


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
