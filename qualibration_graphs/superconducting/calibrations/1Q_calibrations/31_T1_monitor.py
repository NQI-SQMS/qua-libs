# %% {Imports}
import time
import matplotlib.pyplot as plt
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.T1_monitor import (
    Parameters,
    T1MonitorResult,
    process_raw_dataset,
    fit_single_dataset,
    log_monitor_summary,
    build_dataset,
    plot_T1_monitor,
)


# %% {Node initialisation}
description = """
        T1 MONITOR
Repeatedly measures the qubit T1 relaxation time and records T1 vs elapsed time.
Each iteration runs the same sequence as the standard T1 node:
  x180 → wait(t) → measure
and fits an exponential decay to extract T1.

Useful for monitoring T1 fluctuations over minutes to hours.

Prerequisites:
    - Calibrated x180 pulse (nodes 03/04).
    - Calibrated readout (nodes 02a/02b/02c).

State update: none (monitoring only).
"""

node = QualibrationNode[Parameters, Quam](
    name="31_T1_monitor",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.n_iter = 200
    # node.parameters.num_shots = 200
    # node.parameters.min_wait_time_in_ns = 16
    # node.parameters.max_wait_time_in_ns = 150_000
    # node.parameters.wait_time_num_points = 71
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the QUA program for a single T1 sweep (identical to node 05_T1)."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)
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
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.wait(t)

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                            wait(qubit.resonator.depletion_time // 4, qubit.resonator.name)

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
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Run the T1 program n_iter times and accumulate T1 vs elapsed time."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    n_iter = node.parameters.n_iter
    use_sd = node.parameters.use_state_discrimination

    # Initialise accumulators
    monitor_results = {
        q.name: T1MonitorResult(qubit=q.name, t_min=[], T1_us=[], T1_error_us=[])
        for q in qubits
    }
    raw_list: list = []

    t0 = time.time()

    for iteration in range(n_iter):
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(node.namespace["qua_program"])
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                pass  # wait for completion

        elapsed_min = (time.time() - t0) / 60.0

        ds = process_raw_dataset(dataset, use_sd, qubits)
        fit = fit_single_dataset(ds, use_sd)
        raw_list.append(ds.assign_coords(iteration=iteration))

        for q_name, res in fit.items():
            monitor_results[q_name].t_min.append(elapsed_min)
            monitor_results[q_name].T1_us.append(res["T1_us"])
            monitor_results[q_name].T1_error_us.append(res["T1_error_us"])

        node.log(
            f"Iter {iteration + 1}/{n_iter}  |  t = {elapsed_min:.1f} min  |  "
            + "  ".join(
                f"{q}: T1 = {fit[q]['T1_us']:.1f} µs" if fit[q]["success"] else f"{q}: FAIL"
                for q in fit
            )
        )

    node.results["monitor_results"] = {
        q: {
            "t_min": r.t_min,
            "T1_us": r.T1_us,
            "T1_error_us": r.T1_error_us,
        }
        for q, r in monitor_results.items()
    }
    node.namespace["monitor_results"] = monitor_results
    node.namespace["raw_list"] = raw_list


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)

    # Reconstruct T1MonitorResult objects.
    # Prefer the xarray dataset (new format) over the plain dict (legacy).
    monitor_results = {}
    ds = node.results.get("ds")
    if ds is not None:
        for q in ds.qubit.values:
            monitor_results[str(q)] = T1MonitorResult(
                qubit=str(q),
                t_min=ds.sel(qubit=q).elapsed_min.values.tolist(),
                T1_us=ds.sel(qubit=q).T1_us.values.tolist(),
                T1_error_us=ds.sel(qubit=q).T1_error_us.values.tolist(),
            )
    else:
        for q, d in node.results.get("monitor_results", {}).items():
            monitor_results[q] = T1MonitorResult(
                qubit=q,
                t_min=d["t_min"],
                T1_us=d["T1_us"],
                T1_error_us=d["T1_error_us"],
            )
    node.namespace["monitor_results"] = monitor_results


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Build the results dataset and log a summary of the T1 monitor run."""
    monitor_results = node.namespace["monitor_results"]
    node.results["ds"] = build_dataset(monitor_results)
    log_monitor_summary(monitor_results, log_callable=node.log)

    # Concatenate per-iteration raw I/Q datasets and save for archival
    if "raw_list" in node.namespace and node.namespace["raw_list"]:
        node.results["ds_raw"] = xr.concat(node.namespace["raw_list"], dim="iteration")


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot T1 vs elapsed time for all qubits."""
    fig = plot_T1_monitor(
        node.namespace["monitor_results"],
        node.namespace["qubits"],
    )
    plt.show()
    node.results["figures"] = {"T1_monitor": fig}


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
