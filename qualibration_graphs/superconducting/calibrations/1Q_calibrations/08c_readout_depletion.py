# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.readout_depletion import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)


# %% {Node initialisation}
description = """
        READOUT DEPLETION MEASUREMENT

Measures how long the resonator takes to deplete photons after a readout pulse.

Sequence (repeated n_shots times, sweeping tau):
  1. First readout (excites resonator photons, result discarded)
  2. Wait tau on resonator (photons decay)
  3. Ramsey on qubit: x90 → wait(ramsey_idle_time) → x90
  4. Second readout (measures qubit state)

When tau is short, residual photons AC-Stark shift the qubit during the Ramsey
idle time, changing the excited-state population. As tau grows the photons
deplete and the Ramsey outcome stabilises. Fitting the exponential decay gives
the resonator depletion time constant.

Prerequisites:
    - Calibrated readout parameters (nodes 02a, 02b).
    - Calibrated x90 pulse (node 04b_power_rabi or 04c_time_rabi).
    - Calibrated IQ blobs / rotation angle (node 07_iq_blobs).

State update:
    - qubit.resonator.depletion_time (in ns)
"""

node = QualibrationNode[Parameters, Quam](
    name="08c_readout_depletion",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.min_wait_time_in_ns = 16
    # node.parameters.max_wait_time_in_ns = 4000
    # node.parameters.wait_time_num_points = 100
    # node.parameters.ramsey_idle_time_in_ns = 200
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)  # in clock cycles (4 ns each)
    ramsey_idle_cc = node.parameters.ramsey_idle_time_in_ns // 4  # clock cycles

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "wait time after readout", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        # Temporary variables for the first (discarded) readout
        I_discard = [declare(fixed) for _ in range(num_qubits)]
        Q_discard = [declare(fixed) for _ in range(num_qubits)]
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
                    # --- Reset qubits to ground state ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()

                    # --- First readout: excites resonator photons ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I_discard[i], Q_discard[i]))
                        # Wait t clock cycles for resonator photons to decay
                        qubit.resonator.wait(t)

                    align()

                    # --- Ramsey sequence: sensitive to residual photon-induced dephasing ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x90")
                        if ramsey_idle_cc >= 4:
                            qubit.xy.wait(ramsey_idle_cc)
                        qubit.xy.play("x90")

                    align()

                    # --- Second readout: measure qubit state ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)

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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["ds_fit"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if r["success"] else "failed")
        for q, r in node.results["fit_results"].items()
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
    node.results["figures"] = {"depletion": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            depletion_ns = node.results["fit_results"][q.name]["depletion_time_ns"]
            # Store 3× the time constant to ensure >99.5% depletion
            q.resonator.depletion_time = int(round(3 * depletion_ns))


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
