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
from calibration_utils.shared import _get_pair_components
from calibration_utils.fNgN1_resonator_spectroscopy import (
    Parameters,
    fit_raw_data,
    process_raw_dataset,
    log_fitted_results,
    split_fock_vacuum,
    plot_results,
)

# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY AT FOCK |n⟩  (vs vacuum reference)

Sweeps the readout resonator frequency at two cavity preparations back-to-back:
  1. Fock |fock_level+1⟩  — via the standard sideband preparation sequence.
  2. Vacuum |0⟩            — no preparation; cavity already empty after reset.

Both sweeps use the same detuning axis so the traces can be directly overlaid to
show the photon-number-dependent resonator frequency shift (qubit-mediated cross-Kerr).

Sequence per shot / frequency point:
  0. Thermalize cavity and qubit.
  1. [Fock prep] For j = 0 … fock_level:
       π_ge → π_ef → sideband_pi(f{j}g{j+1}) → cavity |j+1⟩, qubit |g⟩.
  2. Measure resonator → (I_fock, Q_fock).
  3. Reset cavity (leaves cavity in |0⟩, qubit in |g⟩).
  4. Measure resonator → (I_vac, Q_vac).   [vacuum reference, same detuning]
  5. Qubit thermalization wait.

Prerequisites:
    - Calibrated resonator spectroscopy at vacuum (node 02a).
    - Calibrated ge and ef transitions (nodes 04b, 13).
    - For fock_level > 0: calibrated sideband transitions 0…fock_level-1 (nodes 26/26b).

State update:
    - cavity_transmon_pairs["{qubit}_{mode}"].transitions["f{k}g{k+1}"].resonator_f_fock_hz
"""

node = QualibrationNode[Parameters, Quam](
    name="07i_fNgN1_resonator_spectroscopy",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Debugging / local overrides."""
    # node.parameters.fock_level = 0
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

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs,
            attrs={"long_name": f"resonator detuning at Fock {k + 1}", "units": "Hz"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        # Standard streams for Fock measurement (I{i}, Q{i})
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)

        # Additional streams for vacuum reference measurement
        I_vac = [declare(fixed) for _ in range(num_qubits)]
        Q_vac = [declare(fixed) for _ in range(num_qubits)]
        I_vac_st = [declare_stream() for _ in range(num_qubits)]
        Q_vac_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator

                        # ── State preparation ─────────────────────────────
                        if node.parameters.preparation_type == "sideband":
                            pair.fock_prep_qua(k + 1, qubit)
                        else:
                            cav_mode.cavity_mode_drive.play(
                                "displacement",
                                amplitude_scale=node.namespace["amplitude_scale"],
                            )
                        align(qubit.xy.name, rr.name)
                        rr.update_frequency(df + rr.intermediate_frequency)
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        rr.wait(rr.depletion_time // 4)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                        # ── Cavity reset (cavity → |0⟩, qubit → |g⟩) ─────
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

                        # ── Vacuum reference measurement ───────────────────
                        # Cavity is already in |0⟩ after the reset; measure
                        # at the same detuning for a direct comparison.
                        align(qubit.xy.name, rr.name)
                        rr.update_frequency(df + rr.intermediate_frequency)
                        rr.measure("readout", qua_vars=(I_vac[i], Q_vac[i]))
                        rr.wait(rr.depletion_time // 4)
                        save(I_vac[i], I_vac_st[i])
                        save(Q_vac[i], Q_vac_st[i])

                        # ── Qubit thermalization ───────────────────────────
                        qubit.xy.wait(2 * qubit.thermalization_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
                I_vac_st[i].buffer(len(dfs)).average().save(f"I_vac{i + 1}")
                Q_vac_st[i].buffer(len(dfs)).average().save(f"Q_vac{i + 1}")


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
    ds_fock_raw, ds_vac_raw = split_fock_vacuum(node.results["ds_raw"])

    ds_fock = process_raw_dataset(ds_fock_raw, node)
    ds_vac  = process_raw_dataset(ds_vac_raw,  node)

    _, fit_results_fock = fit_raw_data(ds_fock, node)
    _, fit_results_vac  = fit_raw_data(ds_vac,  node)

    node.results["ds_fock"] = ds_fock
    node.results["ds_vac"]  = ds_vac
    node.results["fit_results"]     = {q: asdict(v) for q, v in fit_results_fock.items()}
    node.results["fit_results_vac"] = {q: asdict(v) for q, v in fit_results_vac.items()}

    node.log("── Fock |{}⟩ ──".format(node.parameters.fock_level + 1))
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.log("── Vacuum |0⟩ ──")
    log_fitted_results(node.results["fit_results_vac"], log_callable=node.log)

    node.outcomes = {
        q: ("successful" if fr["success"] else "failed")
        for q, fr in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    k = node.parameters.fock_level
    fig = plot_results(
        ds_fock=node.results["ds_fock"],
        ds_vac=node.results["ds_vac"],
        fit_results_fock=node.results["fit_results"],
        fit_results_vac=node.results["fit_results_vac"],
        fock_level=k,
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {f"f{k}g{k+1}_resonator_spectroscopy": fig}


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
            pair.transitions[tr_key].resonator_f_fock_hz = res["frequency"]
            break  # single sideband drive per mode


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
