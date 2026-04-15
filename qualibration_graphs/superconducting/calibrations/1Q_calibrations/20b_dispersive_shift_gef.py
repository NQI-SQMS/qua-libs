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
from calibration_utils.dispersive_shift_gef import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)

# %% {Node initialisation}
description = """
        GEF DISPERSIVE SHIFT MEASUREMENT
This node measures all three resonator frequencies by sweeping the readout
resonator frequency in three conditions:
  1. Qubit in |g⟩ (thermal / after reset)
  2. Qubit in |e⟩ (after x180 pulse)
  3. Qubit in |f⟩ (after x180 + EF_x180 pulses)

Each spectrum is fitted with a Lorentzian dip. The extracted quantities are:
  chi_ge = f_resonator(|e⟩) - f_resonator(|g⟩)
  chi_ef = f_resonator(|f⟩) - f_resonator(|e⟩)

The optimal readout frequency is set to f_resonator(|e⟩), which gives maximum
discrimination contrast between |g⟩ and |e⟩.

Prerequisites:
    - Calibrated resonator (nodes 02a/02b).
    - Calibrated x180 pulse (node 04b).
    - Calibrated EF_x180 pulse (node 13).

State updates:
    - qubit.resonator.RF_frequency → f_resonator(|e⟩).
    - qubit.chi    (if attribute exists) → chi_ge [Hz].
    - qubit.chi_ef (if attribute exists) → chi_ef [Hz].
"""

node = QualibrationNode[Parameters, Quam](
    name="20b_dispersive_shift_gef",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.qubits = ["q1"]
    pass


node.machine = Quam.load()


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

    # qubit_state: 0=|g⟩, 1=|e⟩, 2=|f⟩
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "qubit_state": xr.DataArray([0, 1, 2], attrs={"long_name": "qubit state"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "resonator detuning", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── |g⟩ sweep ──────────────────────────────────────────────
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.update_frequency(
                            qubit.resonator.intermediate_frequency + df
                        )
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

                # ── |e⟩ sweep ──────────────────────────────────────────────
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(qubit.thermalization_time * u.ns)
                        qubit.xy.play("x180")
                        qubit.resonator.update_frequency(
                            qubit.resonator.intermediate_frequency + df
                        )
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

                # ── |f⟩ sweep ──────────────────────────────────────────────
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(qubit.thermalization_time * u.ns)
                        # Prepare |e⟩
                        qubit.xy.play("x180")
                        # Switch to EF frequency and prepare |f⟩
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + int(qubit.anharmonicity)
                        )
                        qubit.xy.play("EF_x180")
                        # Reset xy to GE frequency
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.resonator.update_frequency(
                            qubit.resonator.intermediate_frequency + df
                        )
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # buffer: [qubit_state=3, detuning=len(dfs)]
                I_st[i].buffer(3, len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(3, len(dfs)).average().save(f"Q{i + 1}")


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
    import matplotlib.pyplot as plt
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        {k: type("FP", (), v)() for k, v in node.results["fit_results"].items()},
    )
    plt.show()
    node.results["figures"] = {"dispersive_shift_gef": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            res = node.results["fit_results"][q.name]
            # Set readout to f_resonator(|e⟩) for maximum |g⟩/|e⟩ contrast
            q.resonator.RF_frequency = res["f_optimal"]
            # Store chi_ge if attribute exists
            if hasattr(q, "chi"):
                q.chi = res["chi_ge_hz"]
            # Store chi_ef if attribute exists (SrfTransmon)
            if hasattr(q, "chi_ef"):
                q.chi_ef = res["chi_ef_hz"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
