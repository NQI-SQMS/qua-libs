# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibration_libs.parameters import (
    get_qubits,
    get_idle_times_in_clock_cycles,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_fock1_T2 import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cavity_fock1_T2.parameters import Parameters

# %% {Node initialisation}
description = """
        CAVITY FOCK |1⟩ T2 RAMSEY (DISPLACEMENT + SNAP)
Measures the T2 Ramsey coherence time of the cavity Fock |1⟩ state using the
displacement + SNAP gate protocol for state preparation and a chi-dispersive
Ramsey on the qubit for readout.

Sequence (per wait time τ):
  1. Thermal reset of cavity and qubit.
  2. Fock |1⟩ preparation via D-SNAP₀-D:
       a. D(α₁): displace cavity  →  amplitude_scale = fock1_alpha1 / displacement_k
       b. SNAP₀(2π): two consecutive selective_x180 at bare qubit frequency
       c. D(α₂): correction displacement  →  amplitude_scale = fock1_alpha2 / displacement_k
  3. Chi-dispersive Ramsey:
       a. X90 on qubit at n=1 dressed frequency (qubit_IF − 2χ)
       b. Wait variable time τ
       c. Frame rotation by (ramsey_detuning_hz × 4ns × τ) turns  [artificial detuning]
       d. X90 on qubit at n=1 dressed frequency  [close Ramsey]
       e. Reset qubit drive to qubit_IF
  4. Measure qubit state.

Physical principle:
  The qubit is driven at its n=1 dressed frequency so that its phase accumulates
  relative to the cavity Fock |1⟩ state. As the Fock state dephases or the cavity
  photon decays, the qubit Ramsey contrast decreases. The decay envelope gives T2
  of the qubit coupled to the Fock |1⟩ state (limited by cavity T1 and any pure
  cavity dephasing).

P(|e⟩) vs τ follows a decaying sinusoid with time constant T2ramsey.

Prerequisites:
    - Calibrated displacement amplitude (node 22 or 32).
    - Calibrated χ dispersive shift (node 25) stored in CavityTransmonPair.chi.
    - Calibrated x90 pulse.

Parameters:
    - mode_name:             'alice' or 'bob'
    - fock1_alpha1:          first displacement amplitude [photons]  (default 1.0)
    - fock1_alpha2:          correction displacement amplitude [photons]  (default −0.59)
    - ramsey_detuning_hz:    artificial detuning for Ramsey fringes [Hz]
    - idle_time_*:           range of wait times

State update:
    - cavity_mode.T2ramsey  (seconds)
"""



node = QualibrationNode[Parameters, Quam](
    name="36_cavity_fock1_T2",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.fock1_alpha1 = 1.0
    # node.parameters.fock1_alpha2 = -0.59
    # node.parameters.ramsey_detuning_hz = 1000.0
    pass


node.machine = Quam.load()


def _get_cavity_mode(node):
    mode_name = node.parameters.mode_name
    for cav in node.machine.cavities.values():
        mode = getattr(cav, mode_name, None)
        if mode is not None:
            return mode
    raise KeyError(f"Cavity mode '{mode_name}' not found in machine.cavities")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)  # clock cycles (4 ns each)

    cavity_mode = _get_cavity_mode(node)
    mode_name = node.parameters.mode_name

    # ── Resolve displacement_k and χ from QuAM CavityTransmonPair ──────────────
    displacement_k = 1.0
    chi_hz = 0.0
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}"):
            if getattr(pair, "displacement_k", None) is not None:
                displacement_k = float(pair.displacement_k)
            if getattr(pair, "chi", None) is not None:
                chi_hz = float(pair.chi)
            break

    alpha1_scale = node.parameters.fock1_alpha1 / displacement_k
    alpha2_scale = node.parameters.fock1_alpha2 / displacement_k

    # Phase increment per clock cycle for the qubit frame rotation:
    # detuning_hz * 1e-9 * 4 ns/cc = turns per clock cycle
    detuning_turns_per_cc = node.parameters.ramsey_detuning_hz * 1e-9 * 4

    node.log(
        f"Fock1 prep: α₁={node.parameters.fock1_alpha1:.3f} (scale={alpha1_scale:.4f}), "
        f"α₂={node.parameters.fock1_alpha2:.3f} (scale={alpha2_scale:.4f}), "
        f"χ={chi_hz * 1e-3:.3f} kHz, "
        f"detuning={node.parameters.ramsey_detuning_hz:.1f} Hz"
    )
    node.namespace["alpha1_scale"] = alpha1_scale
    node.namespace["alpha2_scale"] = alpha2_scale
    node.namespace["chi_hz"] = chi_hz

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        phase = declare(fixed)
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

                with for_each_(t, idle_times):
                    for i, qubit in multiplexed_qubits.items():
                        qubit_IF = int(qubit.xy.intermediate_frequency)
                        n1_freq = qubit_IF - int(round(2 * chi_hz))

                        # ── 1. Thermal reset ──────────────────────────────────
                        qubit.xy.wait(qubit.thermalization_time // 4)

                        # ── 2. Fock |1⟩ preparation: D(α₁) → SNAP₀ → D(α₂) ──
                        # Step 2a: first displacement D(α₁)
                        align(qubit.xy.name, cavity_mode.cavity_mode_drive.name,
                              qubit.resonator.name)
                        cavity_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=node.namespace["alpha1_scale"],
                        )
                        align()

                        # Step 2b: SNAP₀(2π) — two selective_x180 at bare qubit IF
                        qubit.xy.update_frequency(qubit_IF)
                        with strict_timing_():
                            qubit.xy.play("selective_x180")
                            qubit.xy.play("selective_x180")
                        align()

                        # Step 2c: correction displacement D(α₂)
                        cavity_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=node.namespace["alpha2_scale"],
                        )
                        align()

                        # ── 3a. First X90 at n=1 dressed qubit frequency ──────
                        qubit.xy.update_frequency(n1_freq)
                        qubit.xy.play("x90")

                        # ── 3b. Wait τ ────────────────────────────────────────
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.wait(t)

                        # ── 3c. Frame rotation (artificial detuning) ──────────
                        assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_cc, t))
                        frame_rotation_2pi(phase, qubit.xy.name)

                        # ── 3d. Close Ramsey: second X90 at n=1 frequency ─────
                        qubit.xy.play("x90")
                        qubit.xy.update_frequency(qubit_IF)   # reset to ge freq

                        # ── 4. Measure ────────────────────────────────────────
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
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_fock1_T2": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    mode_name = node.parameters.mode_name
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            T2ramsey_ns = node.results["fit_results"][q.name]["T2ramsey_ns"]
            T2ramsey_s = float(T2ramsey_ns) * 1e-9
            for cav in node.machine.cavities.values():
                mode = getattr(cav, mode_name, None)
                if mode is not None and hasattr(mode, "T2ramsey"):
                    mode.T2ramsey = T2ramsey_s
                    break


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
