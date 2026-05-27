# %% {Imports}
import time as _time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.readout_gef_length_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_distances_vs_length,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node initialisation}
description = """
        GEF READOUT LENGTH OPTIMISATION
Finds the optimal readout pulse duration for three-level (g/e/f) state discrimination
by maximising min(D_ge, D_ef, D_gf) as a function of cumulative integration time.

Uses accumulated demodulation: within a single readout pulse the IQ signal is averaged
in chunks of `division_length_in_ns` nanoseconds. For each shot all three states are
measured (idle → |g⟩, x180 → |e⟩, x180+EF_x180 → |f⟩). The averaged IQ centroids
at each cumulative length are used to compute the three pairwise distances; the
worst-case metric min(D_ge, D_ef, D_gf) is maximised to find the optimum.

The readout operates at the GEF-optimised frequency (qubit.resonator.GEF_frequency_shift).

Prerequisites:
    - GEF readout frequency calibrated (node 14).
    - GEF readout power calibrated (node 14b).
    - ge and EF π-pulses calibrated (nodes 04b, 13).
    - Integration weight names match parameters (default: iw1/iw2/iw3).

State update:
    - qubit.resonator.operations["readout"].length  =  optimal readout length [ns]
"""

node = QualibrationNode[Parameters, Quam](
    name="14c_readout_gef_length_optimization",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    division_length = node.parameters.division_length_in_ns // 4  # clock cycles
    readout_op = node.parameters.readout_operation
    cos_w = node.parameters.cos_weight_name
    sin_w = node.parameters.sin_weight_name
    minus_sin_w = node.parameters.minus_sin_weight_name

    # Temporarily extend the readout pulse to the requested maximum length
    for qubit in qubits:
        qubit.resonator.operations[readout_op].length = node.parameters.max_readout_length_in_ns

    first_qubit = list(qubits)[0]
    readout_length_ns = first_qubit.resonator.operations[readout_op].length
    number_of_divisions = int(readout_length_ns // (4 * division_length))
    node.namespace["number_of_divisions"] = number_of_divisions
    node.namespace["division_length_ns"] = 4 * division_length
    node.namespace["n_avg"] = n_avg

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        ind = declare(int)

        # Per-qubit QUA arrays for accumulated demod (g, e, f)
        II_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        IQ_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QI_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QQ_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]

        II_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        IQ_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QI_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QQ_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]

        II_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        IQ_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QI_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        QQ_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]

        I_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Q_g = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        I_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Q_e = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        I_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]
        Q_f = [declare(fixed, size=number_of_divisions) for _ in range(num_qubits)]

        Ig_st = [declare_stream() for _ in range(num_qubits)]
        Qg_st = [declare_stream() for _ in range(num_qubits)]
        Ie_st = [declare_stream() for _ in range(num_qubits)]
        Qe_st = [declare_stream() for _ in range(num_qubits)]
        If_st = [declare_stream() for _ in range(num_qubits)]
        Qf_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
                # Fix readout at GEF-optimised frequency for the whole program
                if qubit.resonator.GEF_frequency_shift is None:
                    qubit.resonator.GEF_frequency_shift = 0
                qubit.resonator.update_frequency(
                    qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift
                )
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── |g⟩ measurement ──────────────────────────────────────────
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(2 * qubit.thermalization_time // 4)
                align()
                for i, qubit in multiplexed_qubits.items():
                    measure(
                        readout_op,
                        qubit.resonator.name,
                        None,
                        demod.accumulated(cos_w,       II_g[i], division_length, "out1"),
                        demod.accumulated(sin_w,       IQ_g[i], division_length, "out2"),
                        demod.accumulated(minus_sin_w, QI_g[i], division_length, "out1"),
                        demod.accumulated(cos_w,       QQ_g[i], division_length, "out2"),
                    )
                    with for_(ind, 0, ind < number_of_divisions, ind + 1):
                        assign(I_g[i][ind], II_g[i][ind] + IQ_g[i][ind])
                        save(I_g[i][ind], Ig_st[i])
                        assign(Q_g[i][ind], QQ_g[i][ind] + QI_g[i][ind])
                        save(Q_g[i][ind], Qg_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                align()

                # ── |e⟩ measurement ──────────────────────────────────────────
                for i, qubit in multiplexed_qubits.items():
                    qubit.wait(2 * qubit.thermalization_time // 4)
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
                    measure(
                        readout_op,
                        qubit.resonator.name,
                        None,
                        demod.accumulated(cos_w,       II_e[i], division_length, "out1"),
                        demod.accumulated(sin_w,       IQ_e[i], division_length, "out2"),
                        demod.accumulated(minus_sin_w, QI_e[i], division_length, "out1"),
                        demod.accumulated(cos_w,       QQ_e[i], division_length, "out2"),
                    )
                    with for_(ind, 0, ind < number_of_divisions, ind + 1):
                        assign(I_e[i][ind], II_e[i][ind] + IQ_e[i][ind])
                        save(I_e[i][ind], Ie_st[i])
                        assign(Q_e[i][ind], QQ_e[i][ind] + QI_e[i][ind])
                        save(Q_e[i][ind], Qe_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                align()

                # ── |f⟩ measurement ──────────────────────────────────────────
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.wait(2 * qubit.thermalization_time // 4)
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency + qubit.anharmonicity)
                    qubit.xy.play("EF_x180")
                    update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                    qubit.align()
                    measure(
                        readout_op,
                        qubit.resonator.name,
                        None,
                        demod.accumulated(cos_w,       II_f[i], division_length, "out1"),
                        demod.accumulated(sin_w,       IQ_f[i], division_length, "out2"),
                        demod.accumulated(minus_sin_w, QI_f[i], division_length, "out1"),
                        demod.accumulated(cos_w,       QQ_f[i], division_length, "out2"),
                    )
                    with for_(ind, 0, ind < number_of_divisions, ind + 1):
                        assign(I_f[i][ind], II_f[i][ind] + IQ_f[i][ind])
                        save(I_f[i][ind], If_st[i])
                        assign(Q_f[i][ind], QQ_f[i][ind] + QI_f[i][ind])
                        save(Q_f[i][ind], Qf_st[i])
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Average over shots; each stream produces a 1-D array of length n_divisions
                Ig_st[i].buffer(number_of_divisions).average().save(f"Ig{i + 1}")
                Qg_st[i].buffer(number_of_divisions).average().save(f"Qg{i + 1}")
                Ie_st[i].buffer(number_of_divisions).average().save(f"Ie{i + 1}")
                Qe_st[i].buffer(number_of_divisions).average().save(f"Qe{i + 1}")
                If_st[i].buffer(number_of_divisions).average().save(f"If{i + 1}")
                Qf_st[i].buffer(number_of_divisions).average().save(f"Qf{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    n_avg = node.namespace["n_avg"]
    n_div = node.namespace["number_of_divisions"]
    div_ns = node.namespace["division_length_ns"]

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        result_handles = job.result_handles
        n_handle = result_handles.get("n")
        t_start = _time.time()
        while result_handles.is_processing():
            n_fetched = n_handle.fetch_all()
            count = int(np.atleast_1d(n_fetched)[-1]) if n_fetched is not None else 0
            progress_counter(count, n_avg, start_time=t_start)
            _time.sleep(0.5)
        result_handles.wait_for_all_values()
        node.log(job.execution_report())

    # Build raw dataset: per-qubit 1-D arrays (n_divisions,) — averaged over shots
    ds_dict = {}
    for i, q in enumerate(qubits):
        for prefix in ("Ig", "Qg", "Ie", "Qe", "If", "Qf"):
            data = result_handles.get(f"{prefix}{i + 1}").fetch_all()
            if data is not None and hasattr(data, "dtype") and data.dtype.names and "value" in data.dtype.names:
                data = data["value"]
            ds_dict[f"{prefix}_{q.name}"] = xr.DataArray(
                np.atleast_1d(data).astype(float), dims=["division"]
            )

    node.results["ds_raw"] = xr.Dataset(ds_dict)


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
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
    fig = plot_distances_vs_length(
        node.results["ds_fit"], node.namespace["qubits"], node.results["fit_results"]
    )
    plt.show()
    node.results["figures"] = {"distances_vs_length": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    readout_op = node.parameters.readout_operation
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.resonator.operations[readout_op].length = node.results["fit_results"][q.name][
                "optimal_readout_length_ns"
            ]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
