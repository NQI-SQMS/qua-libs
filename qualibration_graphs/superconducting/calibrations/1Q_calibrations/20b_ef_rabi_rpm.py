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
from calibration_utils.ef_rabi_rpm import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)

# %% {Node initialisation}
description = """
        EF RABI RPM — f-STATE PREPARATION CHECK
Uses the RPM back-swap readout scheme to calibrate the EF π-pulse amplitude
while relying only on standard ge state discrimination (no GEF readout needed).

Sequence:
  1. Qubit thermalization wait
  2. ge_π  →  |e⟩
  3. ef(a) [sweep amplitude]  →  EF Rabi rotation
  4. ef_π  (back-swap: maps |f⟩→|e⟩ and |e⟩→|f⟩)
  5. ge_π  (maps |e⟩→|g⟩; |f⟩ stays as |f⟩ → detected as excited)
  6. Readout

Signal: P(a) = cos²(π·a/2) — starts at 1, minimum at a=1 (correct π pulse),
returns to 1 at a=2.  The first minimum gives the EF π-pulse amplitude.

Why use this instead of direct EF power Rabi?
- No need for GEF-optimized readout.
- Back-swap readout gives full contrast (0→1) using only ge discrimination.
- Directly validates that |f⟩ state preparation is correct end-to-end.

Prerequisites:
    - Calibrated ge transition (node 07).
    - Rough EF pulse: EF_x180 (node 13).

State update:
    - qubit.xy.operations["EF_x180"].amplitude  →  calibrated EF π-pulse amplitude.
"""

node = QualibrationNode[Parameters, Quam](
    name="20b_ef_rabi_rpm",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots
    ef_op = node.parameters.ef_operation

    amp_factors = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_factor": xr.DataArray(amp_factors, attrs={"long_name": "EF amplitude factor"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        a = declare(fixed)
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
                with for_(*from_array(a, amp_factors)):
                    for i, qubit in multiplexed_qubits.items():
                        # Thermalization
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)

                        # Prepare |e⟩ via ge_π
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")

                        # EF Rabi with swept amplitude
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + qubit.anharmonicity
                        )
                        qubit.xy.play(ef_op, amplitude_scale=a)

                        # Back-swap: ef_π maps |f⟩→|e⟩ (and |e⟩→|f⟩)
                        qubit.xy.play(ef_op)

                        # ge_π: maps remaining |e⟩→|g⟩; |f⟩ stays, reads as excited
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                        )

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amp_factors)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amp_factors)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(amp_factors)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
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
    fit_results = {q: type("FP", (), v)() for q, v in node.results["fit_results"].items()}
    fig = plot_raw_data_with_fit(
        node.results["ds_fit"],
        None,
        node.namespace["qubits"],
        fit_results,
    )
    plt.show()
    node.results["figures"] = {"ef_rabi_rpm": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    ef_op = node.parameters.ef_operation
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            q = qubit.name
            if node.outcomes[q] == "failed":
                continue
            pi_amp = node.results["fit_results"][q]["pi_amp"]
            qubit.xy.operations[ef_op].amplitude = pi_amp


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
