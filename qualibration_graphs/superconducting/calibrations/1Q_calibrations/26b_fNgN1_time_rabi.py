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
from calibration_utils.fNgN1_time_rabi import (
    Parameters,
    FitParameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_pair_components,
    _get_transition_rf,
    _fock_prep_qua,
    _ge_if_at_fock,
    _ef_if_at_fock,
)

# %% {Node initialisation}
description = """
        SIDEBAND TIME RABI - generalised to any |n⟩ → |n+1⟩ transition

Sweeps the sideband drive duration while the qubit is in |f⟩ and the cavity in
Fock |fock_level⟩.  A Rabi-like oscillation is observed; the π-pulse duration is
extracted from the first minimum of the fitted sinusoid.

Sequence:
  0. Thermalize cavity and qubit.
  1. [Fock prep] For j = 0 … fock_level-1:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity in |j+1⟩, qubit in |g⟩.
  2. π_ge  →  |e⟩
  3. π_ef  →  |f⟩
  4. Play f{k}g{k+1} sideband pulse with swept duration.
  5. π_ef  (back-swap)
  6. Measure qubit state.

Prerequisites:
    - Calibrated ge and ef transitions (nodes 04b, 13).
    - Calibrated sideband frequency for this transition (node 26, fock_level=k).
    - For fock_level > 0: calibrated sideband pulses for transitions 0…fock_level-1.

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].sideband_drive.operations["f{k}g{k+1}_pi"].length
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].pi_flat_top_length_ns
"""


node = QualibrationNode[Parameters, Quam](
    name="26b_fNgN1_time_rabi",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.fock_level = 1
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    durations_ns = np.arange(
        node.parameters.min_duration_ns,
        node.parameters.max_duration_ns,
        node.parameters.duration_step_ns,
    )
    durations_cc = (durations_ns // 4).astype(int)

    k = node.parameters.fock_level
    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)

    # Set the sideband drive to the calibrated transition frequency
    centre_rf = _get_transition_rf(pair, sideband_drive, k)
    if_offset = int(centre_rf - sideband_drive.RF_frequency)
    target_if = int(sideband_drive.intermediate_frequency) + if_offset

    ge_if_k = _ge_if_at_fock(pair, pair_qubit, k)
    ef_if_k = _ef_if_at_fock(pair, pair_qubit, k)

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration_cc": xr.DataArray(
            durations_cc,
            attrs={"long_name": f"f{k}g{k+1} pulse duration", "units": "clock cycles"},
        ),
    }

    # Cavity thermalization override: use explicit wait if provided, otherwise delegate to cav_mode.reset.
    if node.parameters.cavity_thermalization_time_ns is not None:
        cavity_therm_clk = int(min(max(node.parameters.cavity_thermalization_time_ns // 4, 4), 2_500_000_000))
    else:
        cavity_therm_clk = None

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
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

                with for_(*from_array(t, durations_cc)):
                    for i, qubit in multiplexed_qubits.items():
                        # -- Fock state preparation --------------------------
                        _fock_prep_qua(k, pair, qubit, sideband_drive)

                        # -- Prepare qubit in |f⟩ at Fock-k-shifted frequencies -
                        qubit.xy.update_frequency(ge_if_k)
                        qubit.xy.play("x180")
                        qubit.xy.update_frequency(ef_if_k)
                        qubit.xy.play("EF_x180")

                        # -- Time Rabi sweep --------------------------------
                        sideband_drive.update_frequency(target_if)
                        align(qubit.xy.name, sideband_drive.name)
                        with strict_timing_():
                            sideband_drive.play("sideband_ramp_up")
                            sideband_drive.play("sideband_square", duration=t)
                            sideband_drive.play("sideband_ramp_down")

                        # -- Back-swap --------------------------------------
                        align(sideband_drive.name, qubit.xy.name)
                        qubit.xy.update_frequency(ef_if_k)
                        qubit.xy.play("EF_x180")

                        # -- Readout ----------------------------------------
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                        )

                        # -- Reset cavity and qubit -------------------------
                        if cavity_therm_clk is not None and node.parameters.cavity_reset_type == "thermal":
                            if not node.parameters.simulate:
                                cav_mode.wait(cavity_therm_clk)
                        else:
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
                I_st[i].buffer(len(durations_cc)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(durations_cc)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(durations_cc)).average().save(f"state{i + 1}")


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
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
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
    k = node.parameters.fock_level
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    fig.suptitle(f"Sideband time Rabi f{k}g{k+1} — {node.parameters.mode_name}")
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_time_rabi": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, _, sideband_drive, _ = _get_pair_components(node)

    with node.record_state_updates():
        for q_name, res in node.results["fit_results"].items():
            if not res["success"]:
                continue
            pi_ns = int(round(res["pi_duration_ns"]))
            # Update ramp length if explicitly provided in parameters.
            ramp_length_ns = node.parameters.ramp_length_ns
            if ramp_length_ns is not None:
                sideband_drive.operations["sideband_ramp_up"].length = ramp_length_ns
            ramp_ns = sideband_drive.operations["sideband_ramp_up"].length
            # Keep sideband_square.length in sync (used as default duration).
            sideband_drive.operations["sideband_square"].length = pi_ns
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            pair.transitions[tr_key].pi_flat_top_length_ns = pi_ns
            pair.transitions[tr_key].rabi_rate_hz = res.get("rabi_rate_hz")
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
