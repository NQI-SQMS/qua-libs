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
from calibration_utils.qubit_rpm import (
    Parameters,
    FitParameters,
    process_raw_datasets,
    fit_raw_data,
    log_fitted_results,
    plot_rpm,
)

# %% {Node initialisation}
description = """
        QUBIT RABI POPULATION MEASUREMENT (RPM)
Measures the qubit thermal population by comparing two EF amplitude sweeps:

  'g' sweep: ge_π → ef(a) → ef_π (back-swap) → ge_π → readout
             Starts deterministically from |g⟩.
             P_g(a) = cos²(π·a/2): oscillates 1→0→1, minimum at a=1.

  'e' sweep: ef(a) → ge_π → readout
             Starts from the thermal state. The |g⟩ component (1-P_th) always
             maps to |e⟩ after ge_π; the |e⟩ component (P_th) undergoes EF
             Rabi. P_e(a) = (1-P_th) + P_th·sin²(π·a/2).

Extracting the sinusoidal amplitudes A_g and A_e:
    P_th = A_e / (A_e + A_g)

The effective qubit temperature is derived from P_th and the qubit frequency.

Prerequisites:
    - Calibrated ge transition (node 07).
    - Calibrated ef pulse: EF_x180 (node 13).

State update:
    None — this is a diagnostic node.
"""

node = QualibrationNode[Parameters, Quam](
    name="20_qubit_rpm",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    pass


node.machine = Quam.load()


def _make_rpm_program(qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g: bool):
    """
    Build a QUA program for one RPM sweep direction.

    start_from_g=True  → 'g' sequence: ge_π → ef(a) → ef_π(back-swap) → ge_π → measure
    start_from_g=False → 'e' sequence: ef(a) → ge_π → measure
    """
    with program() as qua_prog:
        n = declare(int)
        a = declare(fixed)
        n_st = declare_stream()

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amp_factors)):
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)

                        if start_from_g:
                            # Prepare |e⟩ via ge_π
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.xy.play("x180")

                        # EF Rabi with swept amplitude
                        qubit.xy.update_frequency(
                            qubit.xy.intermediate_frequency + qubit.anharmonicity
                        )
                        qubit.xy.play(ef_op, amplitude_scale=a)

                        if start_from_g:
                            # Back-swap: ef_π maps |f⟩→|e⟩, then ge_π maps |e⟩→|g⟩
                            qubit.xy.play(ef_op)

                        # Return to ge basis
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        qubit.xy.play("x180")

                        # Measure
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                        save(state[i], state_st[i])
                        qubit.resonator.wait(node.machine.depletion_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(amp_factors)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amp_factors)).average().save(f"Q{i + 1}")
                state_st[i].buffer(len(amp_factors)).average().save(f"state{i + 1}")

    return qua_prog


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

    node.namespace["qua_program_g"] = _make_rpm_program(
        qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g=True
    )
    node.namespace["qua_program_e"] = _make_rpm_program(
        qubits, amp_factors, num_qubits, n_avg, ef_op, u, start_from_g=False
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    # Simulate only the 'g' program for brevity
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program_g"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()

    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # ── 'g' sweep ────────────────────────────────────────────────────────
        node.log("Running RPM 'g' sweep (start from |g⟩)…")
        job_g = qm.execute(node.namespace["qua_program_g"])
        fetcher_g = XarrayDataFetcher(job_g, node.namespace["sweep_axes"])
        for ds_g in fetcher_g:
            progress_counter(
                fetcher_g["n"],
                node.parameters.num_shots,
                start_time=fetcher_g.t_start,
            )

        # ── 'e' sweep ────────────────────────────────────────────────────────
        node.log("Running RPM 'e' sweep (start from thermal)…")
        job_e = qm.execute(node.namespace["qua_program_e"])
        fetcher_e = XarrayDataFetcher(job_e, node.namespace["sweep_axes"])
        for ds_e in fetcher_e:
            progress_counter(
                fetcher_e["n"],
                node.parameters.num_shots,
                start_time=fetcher_e.t_start,
            )
        node.log(job_e.execution_report())

    node.results["ds_raw_g"] = ds_g
    node.results["ds_raw_e"] = ds_e


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
    ds_g, ds_e = process_raw_datasets(
        node.results["ds_raw_g"], node.results["ds_raw_e"], node
    )
    node.results["ds_raw_g"] = ds_g
    node.results["ds_raw_e"] = ds_e

    fit_results = fit_raw_data(ds_g, ds_e, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # Diagnostic node: succeed if fit converged, regardless of P_th value
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fit_results = {
        q: type("FP", (), v)()
        for q, v in node.results["fit_results"].items()
    }
    fig = plot_rpm(
        node.results["ds_raw_g"],
        node.results["ds_raw_e"],
        node.namespace["qubits"],
        fit_results,
    )
    plt.show()
    node.results["figures"] = {"qubit_rpm": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    # No state update — RPM is a diagnostic measurement.
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
