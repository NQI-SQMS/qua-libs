# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_builder.architecture.superconducting.qubit_pair.cavity_transmon_pair import SidebandTransition
from calibration_utils.shared import _get_pair_components
from calibration_utils.iq_blobs.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
)
from calibration_utils.iq_blobs.plotting import plot_iq_blobs, plot_confusion_matrices
from calibration_utils.fNgN1_ge_iq_blobs import Parameters

# %% {Node initialisation}
description = """
        ge IQ BLOBS AT FOCK |n⟩

Measures the qubit g/e IQ blobs while the Alice cavity is prepared in Fock state
|fock_level+1⟩.  Because the large dispersive shift (χ ≈ −282 MHz) shifts the
readout resonator IQ response when Alice has photons, the vacuum-calibrated
threshold is invalid.  This node calibrates the correct ge discrimination threshold
for use by node 26e (ge spectroscopy at Fock k).

Sequence (per shot):
  — |g, k+1⟩ blob —
  1. Fock prep: π_ge → π_ef → sideband_pi(f{j}g{j+1}) for j=0…k → Alice |k+1⟩, qubit |g⟩.
  2. Measure resonator → Ig, Qg.
  3. Reset cavity + thermalize qubit.

  — |e, k+1⟩ blob —
  4. Same Fock prep → Alice |k+1⟩, qubit |g⟩.
  5. π_ge at chi-shifted ge IF (accounting for (k+1)×χ shift).
  6. Measure resonator → Ie, Qe.
  7. Reset cavity + thermalize qubit.

Prerequisites:
    - Calibrated ge and ef transitions (nodes 04b, 13).
    - Calibrated sideband frequency and π-pulse for transitions 0…fock_level (nodes 26, 26b).

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].ge_iq_threshold
"""

node = QualibrationNode[Parameters, Quam](
    name="07d_fNgN1_ge_iq_blobs",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.fock_level = 0
    # node.parameters.mode_name = "alice"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    k = node.parameters.fock_level
    n_runs = node.parameters.num_shots

    pair, pair_qubit, sideband_drive, cav_mode = _get_pair_components(node)

    # ge IF when Alice has k+1 photons — used for the x180 in the |e⟩ blob
    ge_if_k1 = pair.ge_if_at_fock(pair_qubit, k + 1)

    chi_hz = float(pair.chi) if (pair is not None and getattr(pair, "chi", None) is not None) else 0.0

    # Resolve displacement amplitude scale (used when preparation_type='displacement')
    _AMP_MAX = 2.0 - 2**-16
    alpha_max = 1.0
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, _pair in pairs.items():
        if pair_key.endswith(f"_{node.parameters.mode_name}"):
            if getattr(_pair, "displacement_alpha_max", None) is not None:
                alpha_max = float(_pair.displacement_alpha_max)
            break
    amplitude_scale = node.parameters.displacement_alpha / alpha_max
    if node.parameters.preparation_type == "displacement" and abs(amplitude_scale) > _AMP_MAX:
        raise ValueError(
            f"displacement_alpha={node.parameters.displacement_alpha} / "
            f"alpha_max={alpha_max} = {amplitude_scale:.4f} exceeds the QUA "
            f"hardware limit ±{_AMP_MAX:.6f}.  Reduce displacement_alpha."
        )
    node.namespace["amplitude_scale"] = amplitude_scale

    # Cavity thermalization override
    if node.parameters.cavity_thermalization_time_ns is not None:
        cavity_therm_clk = int(
            min(max(node.parameters.cavity_thermalization_time_ns // 4, 4), 2_500_000_000)
        )
    else:
        cavity_therm_clk = None

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(
            np.linspace(1, n_runs, n_runs),
            attrs={"long_name": "number of shots"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_stream()
        I_g, I_g_st, Q_g, Q_g_st, _, _ = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)

                for i, qubit in multiplexed_qubits.items():
                    # ── |g, k+1⟩ blob ──────────────────────────────────────
                    if node.parameters.preparation_type == "sideband":
                        pair.fock_prep_qua(k + 1, qubit)
                    else:
                        cav_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=node.namespace["amplitude_scale"],
                        )

                    align(qubit.xy.name, qubit.resonator.name)
                    qubit.resonator.measure(
                        node.parameters.operation, qua_vars=(I_g[i], Q_g[i])
                    )
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

                    # Reset cavity + qubit before second Fock prep
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

                    # ── |e, k+1⟩ blob ──────────────────────────────────────
                    if node.parameters.preparation_type == "sideband":
                        pair.fock_prep_qua(k + 1, qubit)
                    else:
                        cav_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=node.namespace["amplitude_scale"],
                        )

                    # π_ge at the chi-shifted ge frequency with k+1 Alice photons
                    qubit.xy.update_frequency(ge_if_k1)
                    qubit.xy.play("x180")

                    align(qubit.xy.name, qubit.resonator.name)
                    qubit.resonator.measure(
                        node.parameters.operation, qua_vars=(I_e[i], Q_e[i])
                    )
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

                    # Reset cavity + qubit
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
                I_g_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Q_g_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                I_e_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                Q_e_st[i].buffer(n_runs).save(f"Qe{i + 1}")


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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


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
    fig_iq = plot_iq_blobs(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_confusion = plot_confusion_matrices(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    for fig in (fig_iq, fig_confusion):
        fig.suptitle(f"ge IQ blobs at Fock |{k + 1}⟩ — {node.parameters.mode_name}")
    plt.show()
    node.results["figures"] = {"iq_blobs": fig_iq, "confusion_matrix": fig_confusion}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    tr_key = f"f{k}g{k+1}"
    pair, pair_qubit, _, _ = _get_pair_components(node)
    operation = pair_qubit.resonator.operations[node.parameters.operation]

    with node.record_state_updates():
        for q_name, res in node.results["fit_results"].items():
            if not res["success"]:
                continue
            if tr_key not in pair.transitions:
                pair.transitions[tr_key] = SidebandTransition()
            # Convert from analysis voltage units to QUA demod units (same
            # conversion as in 07_iq_blobs update_state)
            pair.transitions[tr_key].ge_iq_threshold = (
                float(res["ge_threshold"]) * operation.length / 2**12
            )
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
