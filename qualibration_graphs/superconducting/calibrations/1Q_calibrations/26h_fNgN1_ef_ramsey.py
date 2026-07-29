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
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.fNgN1_ef_ramsey import Parameters
from calibration_utils.ramsey import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_pair_components,
    _fock_prep_qua,
    _ge_if_at_fock,
    _ef_if_at_fock,
)

# %% {Node initialisation}
description = """
        ef RAMSEY AT FOCK |n⟩ — Kerr refinement of ef_delta_f_focka

Performs a Ramsey experiment on the qubit ef transition while the cavity is in Fock
state |fock_level⟩.  The fitted oscillation frequency reveals the residual detuning of
the ef drive from the true ef resonance at this Fock level.  This captures the Kerr
nonlinearity that the chi-linear approximation misses at higher photon numbers.

Both signs of the artificial detuning ±δ are swept so the frequency correction is
unambiguous:
    f_obs±  =  |f_drive - f_ef_focka ± δ|
    correction  =  (f_obs+ - f_obs-) / 2   →   ef_delta_f_focka -= correction

Sequence:
  0. Cavity + qubit reset (thermal or active sideband).
  1. [Fock prep] For j = 0 … fock_level-1:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity |j+1⟩, qubit |g⟩.
  2. π_ge at delta_f_focka-shifted frequency → qubit |e⟩ with cavity |fock_level⟩.
  3. π/2 ef pulse at ef_delta_f_focka-shifted frequency.
  4. Wait τ + virtual frame rotation (±δ).
  5. π/2 ef pulse.
  6. Refocusing π_ge at chi_focka-shifted frequency: maps |e⟩→|g⟩ off-resonance, leaves |f⟩ untouched on-resonance.
  7. Measure qubit state.

Prerequisites:
    - Calibrated delta_f_focka (nodes 26e/26f, fock_level=k).
    - Calibrated ef_delta_f_focka (node 26g, fock_level=k).
    - Calibrated ef π-pulse duration (nodes 04b, 13).
    - For fock_level > 0: calibrated sideband transitions 0…fock_level-1.

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].ef_delta_f_focka
      (refined with Kerr correction)
"""


node = QualibrationNode[Parameters, Quam](
    name="26h_fNgN1_ef_ramsey",
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
    idle_times_cc = (
        np.linspace(
            node.parameters.min_wait_ns,
            node.parameters.max_wait_ns,
            node.parameters.num_wait_points,
        )
        // 4
    ).astype(int)
    idle_times_ns = idle_times_cc * 4

    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)

    ge_if_k = _ge_if_at_fock(pair, pair_qubit, k + 1)
    ef_if_k = _ef_if_at_fock(pair, pair_qubit, k + 1)
    node.namespace["ge_if_k"] = ge_if_k
    node.namespace["ef_if_k"] = ef_if_k

    # π/2 ef duration in clock cycles (half the calibrated ef pi pulse)
    ef_pi_ns = int(pair_qubit.xy.operations["EF_x180"].length)
    ef_pi2_cc = max(ef_pi_ns // 8, 4)

    detuning_hz = float(node.parameters.frequency_detuning_in_mhz) * 1e6
    detuning_factor = detuning_hz * 1e-9       # GHz = cycles per ns
    detuning_factor_neg = -detuning_factor
    detuning_signs = [-1, 1]

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            idle_times_ns,
            attrs={"long_name": f"Ramsey wait time at Fock {k}", "units": "ns"},
        ),
        "detuning_signs": xr.DataArray(
            detuning_signs,
            attrs={"long_name": "detuning sign"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        phi = declare(fixed)
        detuning_sign = declare(int)
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

                with for_each_(t, idle_times_cc):
                    with for_(*from_array(detuning_sign, detuning_signs)):
                        for i, qubit in multiplexed_qubits.items():
                            # -- Fock state preparation --------------------------
                            _fock_prep_qua(k + 1, pair, qubit, sideband_drive)

                            # -- π_ge at chi_focka-shifted frequency ------------
                            qubit.xy.update_frequency(ge_if_k)
                            qubit.xy.play("x180")

                            # -- ef Ramsey --------------------------------------
                            qubit.xy.update_frequency(ef_if_k)
                            reset_frame(qubit.xy.name)
                            qubit.xy.play("EF_x180", duration=ef_pi2_cc)

                            qubit.xy.wait(t)
                            with if_(detuning_sign == 1):
                                assign(phi, Cast.mul_fixed_by_int(detuning_factor, 4 * t))
                            with else_():
                                assign(phi, Cast.mul_fixed_by_int(detuning_factor_neg, 4 * t))
                            frame_rotation_2pi(phi, qubit.xy.name)

                            qubit.xy.play("EF_x180", duration=ef_pi2_cc)

                            # -- Refocusing ge π: maps |e⟩→|g⟩ off-resonance, leaves |f⟩ untouched on-resonance --
                            qubit.xy.update_frequency(ge_if_k)
                            qubit.xy.play("x180")

                            # -- Readout ----------------------------------------
                            align(qubit.xy.name, qubit.resonator.name)
                            qubit.readout_state(
                                state[i] if node.parameters.use_state_discrimination else None,
                                I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                                state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                            )

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
                # Innermost loop: detuning_signs; outer loop: idle_time
                I_st[i].buffer(len(detuning_signs)).buffer(len(idle_times_cc)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(detuning_signs)).buffer(len(idle_times_cc)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(detuning_signs)).buffer(len(idle_times_cc)).average().save(f"state{i + 1}")


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
    node.results["fit_results"] = {q: asdict(v) for q, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_ef_ramsey": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, pair_qubit, _, _ = _get_pair_components(node)

    with node.record_state_updates():
        for q_name, res in node.results["fit_results"].items():
            if not res["success"]:
                continue
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            current = pair.transitions[tr_key].ef_delta_f_focka
            if current is None:
                current = 0.0
            pair.transitions[tr_key].ef_delta_f_focka = current - res["freq_offset"]
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
