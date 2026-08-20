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
from calibration_utils.fNgN1_spectroscopy import (
    Parameters,
    FitParameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_pair_components,
)

# %% {Node initialisation}
description = """
        SIDEBAND SPECTROSCOPY - generalised to any |n⟩ → |n+1⟩ transition

Sweeps the sideband drive frequency while the qubit is prepared in |f⟩ and the
cavity is prepared in Fock |fock_level⟩.

When the sideband drive is resonant, the |f, n⟩ ↔ |g, n+1⟩ transition is driven,
the qubit is left in |g⟩, and the back-swap π_ef leaves it in |g⟩ → DIP in state
measurement.

Sequence:
  0. Thermalize cavity and qubit.
  1. [Fock prep] For j = 0 … fock_level-1:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity |j+1⟩, qubit back to |g⟩.
  2. π_ge  →  |e⟩
  3. π_ef  →  |f⟩
  4. Sweep f{k}g{k+1} sideband IF;  play saturation/long pulse.
  5. π_ef  (back-swap)
  6. Measure qubit state.

Prerequisites:
    - Calibrated ge and ef transitions (nodes 04b, 13).
    - For fock_level > 0: calibrated sideband pulses for transitions 0…fock_level-1
      stored in pair.transitions["f{j}g{j+1}"].pi_flat_top_length_ns.

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].RF_frequency
      where k = fock_level.
    - Also updates sideband_drive.RF_frequency when fock_level == 0.
"""


node = QualibrationNode[Parameters, Quam](
    name="07_fNgN1_spectroscopy",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centre_rf_freq(node, pair, qubit, sideband_drive, cav_mode):
    """Return the centre RF frequency for the transition being calibrated.

    When use_theoretical_frequency_estimate=True:
      - k=0: uses 2·f_ge + anharmonicity − f_cav.
      - k>0: offsets f0g1 by k × |chi|.
    When False (default): returns the RF_frequency already saved in the state.
    """
    k = node.parameters.fock_level
    stored = pair.get_transition_rf(k)

    if not node.parameters.use_theoretical_frequency_estimate:
        return stored

    # Theoretical estimate path
    if k == 0:
        if cav_mode is not None:
            return (
                2 * qubit.xy.RF_frequency
                + qubit.anharmonicity
                - cav_mode.cavity_mode_drive.RF_frequency
            )
        return stored
    # k > 0: offset f0g1 by k × |chi|
    f0g1_rf = pair.get_transition_rf(0)
    chi = pair.chi if pair.chi is not None else 0
    return f0g1_rf + k * chi


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

    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)
    op = node.parameters.operation
    op_len = node.parameters.operation_len_in_ns
    k = node.parameters.fock_level

    # Centre of frequency sweep for this transition
    centre_rf = _centre_rf_freq(node, pair, pair_qubit, sideband_drive, cav_mode)
    # Store as initial guess in transitions if not yet set
    tr_key = f"f{k}g{k+1}"
    if tr_key not in pair.transitions:
        pair.transitions[tr_key] = SidebandTransition(RF_frequency=centre_rf)
    elif pair.transitions[tr_key].RF_frequency is None:
        pair.transitions[tr_key].RF_frequency = centre_rf

    # QUA will update_frequency to: sideband_drive.intermediate_frequency + if_offset + f
    if_offset = int(centre_rf - sideband_drive.RF_frequency)
    target_if_base = int(sideband_drive.intermediate_frequency) + if_offset

    # Store centre RF on the node so process_raw_dataset can use it for the x-axis
    node.namespace["centre_rf"] = centre_rf

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    displaced_threshold = None
    if node.parameters.use_state_discrimination and node.parameters.use_displaced_threshold:
        _t = getattr(pair, "ge_iq_threshold_displaced", None)
        if _t is not None:
            displaced_threshold = float(_t)

    ge_if_k = pair.ge_if_at_fock(pair_qubit, k)
    ef_if_k = pair.ef_if_at_fock(pair_qubit, k)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": f"f{k}g{k+1} detuning", "units": "Hz"}),
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
                        # ── Fock state preparation ──────────────────────────
                        pair.fock_prep_qua(k, qubit)

                        # ── Prepare qubit in |f⟩ at Fock-k-shifted frequencies ─
                        qubit.xy.update_frequency(ge_if_k)
                        qubit.xy.play("x180")
                        qubit.xy.update_frequency(ef_if_k)
                        qubit.xy.play("EF_x180")

                        # ── Sweep sideband around this transition ───────────
                        sideband_drive.update_frequency(target_if_base + f)
                        align(qubit.xy.name, sideband_drive.name)
                        amp_scale = node.parameters.operation_amplitude_factor
                        if op == "sideband_flat_top":
                            flat_cc = (op_len >> 2) if op_len is not None else None
                            pair.play_sideband_flattop(flat_top_duration_clk=flat_cc, amplitude_scale=amp_scale)
                        elif op_len is not None:
                            sideband_drive.play(op, amplitude_scale=amp_scale, duration=op_len >> 2)
                        else:
                            sideband_drive.play(op, amplitude_scale=amp_scale)

                        # ── Back-swap ──────────────────────────────────────
                        align(sideband_drive.name, qubit.xy.name)
                        qubit.xy.update_frequency(ef_if_k)
                        qubit.xy.play("EF_x180")

                        # ── Readout ────────────────────────────────────────
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                            threshold=displaced_threshold,
                        )

                        # ── Reset cavity and qubit ─────────────────────────
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
    # process_raw_dataset reads sideband_drive.RF_frequency for the x-axis centre;
    # temporarily override it to the correct transition RF so k>0 plots correctly.
    pair, _, sideband_drive, _ = _get_pair_components(node)
    original_rf = sideband_drive.RF_frequency
    sideband_drive.RF_frequency = node.namespace.get("centre_rf", original_rf)

    try:
        node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
        if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
            node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
        node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    finally:
        sideband_drive.RF_frequency = original_rf

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
        mode_name=node.parameters.mode_name,
        transition_label=f"f{k}g{k+1}",
    )
    fig.suptitle(f"Sideband spectroscopy f{k}g{k+1} — {node.parameters.mode_name}")
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_spectroscopy": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state_node(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, _, sideband_drive, _ = _get_pair_components(node)
    fit_params = {q: type("FP", (), v)() for q, v in node.results["fit_results"].items()}

    with node.record_state_updates():
        for q_name, fp in fit_params.items():
            if not fp.success:
                continue
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            pair.transitions[tr_key].RF_frequency = fp.frequency
            # For k=0 also keep sideband_drive.RF_frequency in sync
            if k == 0:
                sideband_drive.RF_frequency = fp.frequency
            # Derive and save chi from the measured sideband shift when absent
            if k > 0 and node.parameters.update_chi_if_absent and pair.chi is None:
                f0g1_rf = pair.get_transition_rf(0)
                pair.chi = (fp.frequency - f0g1_rf) / k
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
