# %% {Imports}
import logging
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
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_pair_components,
    _get_transition_rf,
    _fock_prep_qua,
    _ge_if_at_fock,
    _ef_if_at_fock,
)
from calibration_utils.fNgN1_qubit_ef_spectroscopy import Parameters, fit_raw_data, plot_raw_data_with_fit

# %% {Node initialisation}
description = """
        QUBIT ef SPECTROSCOPY AT FOCK |n⟩

Sweeps the qubit ef drive frequency while the cavity is prepared in Fock state |fock_level⟩.
The Kerr nonlinearity shifts the ef resonance (ef_delta_f_focka) relative to vacuum.
This node calibrates the per-Fock ef frequency shift.

Sequence:
  0. Cavity + qubit reset (thermal or active sideband).
  1. [Fock prep] For j = 0 … fock_level-1:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity |j+1⟩, qubit |g⟩.
  2. π_ge at chi_focka-shifted frequency → qubit |e⟩.
  3. Sweep ef frequency around ef_chi_focka estimate.
  4. Play saturation pulse on qubit ef.
  5. Refocusing π_ge at chi_focka-shifted frequency: maps |e⟩→|g⟩ off-resonance, leaves |f⟩ untouched on-resonance.
  6. Measure qubit state.

Prerequisites:
    - Calibrated chi_focka (nodes 26d/26e, fock_level=k).
    - Calibrated ge and ef transitions (nodes 04b, 13).
    - For fock_level > 0: calibrated sideband transitions 0…fock_level-1.

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].ef_delta_f_focka
"""


node = QualibrationNode[Parameters, Quam](
    name="26g_fNgN1_qubit_ef_spectroscopy",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.fock_level = 1
    # node.parameters.mode_name  = "alice"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    k = node.parameters.fock_level
    n_avg = node.parameters.num_shots
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)
    op = node.parameters.operation
    op_len = node.parameters.operation_len_in_ns

    ge_if_k = _ge_if_at_fock(pair, pair_qubit, k + 1)
    ef_centre_if = _ef_if_at_fock(pair, pair_qubit, k + 1)
    node.namespace["ge_if_k"] = ge_if_k
    node.namespace["ef_centre_if"] = ef_centre_if
    node.namespace["ge_base_if"] = int(pair_qubit.xy.intermediate_frequency)

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs,
            attrs={"long_name": f"ef detuning at Fock {k + 1}", "units": "Hz"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        f = declare(int)
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

                with for_(*from_array(f, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # -- Fock state preparation --------------------------
                        _fock_prep_qua(k + 1, pair, qubit, sideband_drive)

                        # -- π_ge at chi_focka-shifted frequency ------------
                        qubit.xy.update_frequency(ge_if_k)
                        qubit.xy.play("x180")

                        # -- Sweep ef around ef_chi_focka resonance --
                        qubit.xy.update_frequency(ef_centre_if + f)
                        if op_len is not None:
                            qubit.xy.play(
                                op,
                                amplitude_scale=node.parameters.operation_amplitude_factor,
                                duration=op_len >> 2,
                            )
                        else:
                            qubit.xy.play(
                                op,
                                amplitude_scale=node.parameters.operation_amplitude_factor,
                            )

                        # -- Refocusing ge π: maps |e⟩→|g⟩ off-resonance, leaves |f⟩ untouched on-resonance --
                        qubit.xy.update_frequency(ge_if_k)
                        qubit.xy.play("x180")

                        # -- Readout ----------------------------------------
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(
                                state[i],
                                Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold),
                            )
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                        # -- Cavity + qubit reset ----------------------------
                        cav_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            chi_hz=chi_hz,
                            pair=pair,
                        )
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")


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
    pair, pair_qubit, _, _ = _get_pair_components(node)
    k = node.parameters.fock_level
    ge_if_k = node.namespace.get("ge_if_k", _ge_if_at_fock(pair, pair_qubit, k + 1))
    ef_centre_if = node.namespace.get("ef_centre_if", _ef_if_at_fock(pair, pair_qubit, k + 1))
    ge_base_if = node.namespace.get("ge_base_if", int(pair_qubit.xy.intermediate_frequency))
    anharmonicity_estimate = ef_centre_if - ge_if_k

    ds = node.results["ds_raw"]
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        ds = apply_confusion_matrix_correction(ds, node.namespace["qubits"])
        node.results["ds_raw"] = ds
    ds_fit, fit_results = fit_raw_data(ds, node.parameters.frequency_span_in_mhz, node.parameters.use_state_discrimination)

    bare_anharmonicity = float(pair_qubit.anharmonicity)
    for res in fit_results.values():
        res["ef_delta_f_focka_hz"] = anharmonicity_estimate + res["frequency_hz"] - bare_anharmonicity

    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results

    log = logging.getLogger(__name__)
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        if res["success"]:
            log.info(
                f"[{q}] {status}: ef_delta_f_focka = {res['ef_delta_f_focka_hz'] * 1e-3:.1f} kHz "
                f"(ef peak detuning from center = {res['frequency_hz'] * 1e-3:.1f} kHz)"
            )
        else:
            log.info(f"[{q}] {status}: fit failed")

    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    ds = node.results.get("ds_fit", node.results.get("ds_raw"))

    pair, pair_qubit, _, _ = _get_pair_components(node)
    ge_base_if = node.namespace.get("ge_base_if", int(pair_qubit.xy.intermediate_frequency))
    ef_centre_if = node.namespace.get("ef_centre_if", _ef_if_at_fock(pair, pair_qubit, k + 1))
    rf_center_hz = pair_qubit.xy.RF_frequency - ge_base_if + ef_centre_if

    fig = plot_raw_data_with_fit(ds, node.results["fit_results"], rf_center_hz, k + 1, node.parameters.mode_name)
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_ef_spectroscopy": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, _, _, _ = _get_pair_components(node)

    with node.record_state_updates():
        for q_name, res in node.results["fit_results"].items():
            if not res["success"]:
                continue
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            pair.transitions[tr_key].ef_delta_f_focka = res["ef_delta_f_focka_hz"]
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
