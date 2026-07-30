# %% {Imports}
import dataclasses
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.readout_gef_power_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_distances_with_fit,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Node initialisation}
description = """
        GEF READOUT POWER OPTIMISATION
Sweeps the readout pulse amplitude while preparing the qubit successively in |g⟩, |e⟩,
and |f⟩ at the current GEF readout frequency.  For each amplitude the three IQ centroids
are measured and the pairwise distances D_ge, D_ef, D_gf are computed.  The discrimination
metric is min(D_ge, D_ef, D_gf) — the worst-case separation.  The amplitude that maximises
this metric is selected as the new readout amplitude.

Measurement flow:
    1. Sweep readout amplitude from min_amp_factor × A_0 to max_amp_factor × A_0.
    2. For each amplitude acquire averaged IQ for |g⟩, |e⟩, and |f⟩.
       - |g⟩: idle thermalization then readout
       - |e⟩: ge x180 then readout
       - |f⟩: ge x180 + EF_x180 then readout
    3. Compute Distance = min(D_ge, D_ef, D_gf) vs amplitude.
    4. Find the optimum and update the machine state.

Prerequisites:
    - GEF readout frequency calibrated (node 14).
    - ge and EF π-pulses calibrated (nodes 04b, 13).
    - qubit.anharmonicity set.

State update:
    - qubit.resonator.operations["readout_GEF"].amplitude  =  optimal amplitude
"""

node = QualibrationNode[Parameters, Quam](
    name="14b_readout_gef_power_optimization",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Local overrides for interactive debugging."""
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the amplitude sweep for g, e, f readout at the GEF operating frequency."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    for qubit_obj in qubits:
        if "readout_GEF" in qubit_obj.resonator.operations:
            continue
        readout_op = qubit_obj.resonator.operations["readout"]
        new_length = int(round(readout_op.length * 1.5 / 4) * 4)  # multiple of 4 ns
        qubit_obj.resonator.operations["readout_GEF"] = dataclasses.replace(
            readout_op,
            length=new_length,
            threshold=None,
            rus_exit_threshold=None,
        )

    n_avg = node.parameters.num_shots
    operation = node.parameters.operation

    amps = np.linspace(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.num_amps,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(n_avg), attrs={"long_name": "shot index"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "readout amplitude scale factor"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
        I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()
        a = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
                # Set readout to GEF operating frequency for the whole program
                if qubit.resonator.GEF_frequency_shift is None:
                    qubit.resonator.GEF_frequency_shift = 0
                qubit.resonator.update_frequency(
                    qubit.resonator.intermediate_frequency + qubit.resonator.GEF_frequency_shift
                )
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amps)):
                    # ── |g⟩ readout ──────────────────────────────────────────
                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time // 4)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]), amplitude_scale=a)
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                        save(I_g[i], I_g_st[i])
                        save(Q_g[i], Q_g_st[i])
                    align()

                    # ── |e⟩ readout ──────────────────────────────────────────
                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time // 4)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]), amplitude_scale=a)
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])
                    align()

                    # ── |f⟩ readout ──────────────────────────────────────────
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency + qubit.anharmonicity)
                        qubit.xy.play("EF_x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        qubit.align()
                        qubit.resonator.measure(operation, qua_vars=(I_f[i], Q_f[i]), amplitude_scale=a)
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                        save(I_f[i], I_f_st[i])
                        save(Q_f[i], Q_f_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].buffer(len(amps)).buffer(n_avg).save(f"Ig{i + 1}")
                Q_g_st[i].buffer(len(amps)).buffer(n_avg).save(f"Qg{i + 1}")
                I_e_st[i].buffer(len(amps)).buffer(n_avg).save(f"Ie{i + 1}")
                Q_e_st[i].buffer(len(amps)).buffer(n_avg).save(f"Qe{i + 1}")
                I_f_st[i].buffer(len(amps)).buffer(n_avg).save(f"If{i + 1}")
                Q_f_st[i].buffer(len(amps)).buffer(n_avg).save(f"Qf{i + 1}")


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
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_distances_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    node.results["figures"] = {"distances": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            node.machine.qubits[q.name].resonator.operations[node.parameters.operation].amplitude = (
                node.results["fit_results"][q.name]["optimal_amp"]
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
