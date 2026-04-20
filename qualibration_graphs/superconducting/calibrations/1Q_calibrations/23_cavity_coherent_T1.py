# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array  # noqa: F401 (kept for consistency)
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.cavity_coherent_T1 import (
    Parameters,
    CoherentT1Fit,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_coherent_T1,
)

logger = logging.getLogger(__name__)


# %% {Node initialisation}
description = """
        CAVITY COHERENT T1 (33)

Measures the energy relaxation time T1 of a selected cavity mode by preparing
a coherent state |α⟩ and probing the vacuum-state population with a selective
qubit π-pulse.

Sequence (per wait time t):
  1. Thermalize cavity (wait ≥ 5×T1) and reset qubit.
  2. Apply displacement pulse at amplitude_scale = displacement_scale.
  3. Wait for total time t = delay_repeats × t_per_rep.
  4. Apply selective_x180 on qubit — flips qubit only when cavity is in |0⟩.
  5. Measure qubit state.

The measured signal is:
    P_e(t) = A · exp(−n̄₀ · exp(−t / T1)) + offset

where n̄₀ = displacement_scale² and T1 is the cavity photon lifetime.

Parameters:
  - mode_name:          Cavity mode to probe ('alice' or 'bob').
  - displacement_scale: Amplitude scale of the displacement pulse.
                        After node 32 calibration: scale=1 → 1 photon.
  - t_start_ns / t_end_ns / t_num_points: logarithmic time sweep range.
  - delay_repeats:      Multiply effective wait time to extend range.

State updates:
  - cavity_mode.T1  (in seconds)
"""

node = QualibrationNode[Parameters, Quam](
    name="23_cavity_coherent_T1",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.displacement_scale = 2.0
    # node.parameters.t_start_ns = 16
    # node.parameters.t_end_ns = 5_000_000
    # node.parameters.t_num_points = 51
    # node.parameters.delay_repeats = 1
    # node.parameters.num_shots = 1000
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
    """Build the QUA program for the coherent T1 measurement."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    n_avg = node.parameters.num_shots

    cavity_mode = _get_cavity_mode(node)
    node.namespace["cavity_mode"] = cavity_mode

    # Resolve sideband_drive for active cavity cooling (used only if requested)
    mode_name = node.parameters.mode_name
    sideband_drive = None
    alpha_max = 1.0
    pairs = getattr(node.machine, "cavity_transmon_pairs", {})
    for pair_key, pair in pairs.items():
        if pair_key.endswith(f"_{mode_name}"):
            if getattr(pair, "sideband_drive", None) is not None:
                sideband_drive = pair.sideband_drive
            if getattr(pair, "displacement_alpha_max", None) is not None:
                alpha_max = float(pair.displacement_alpha_max)
            break
    node.namespace["sideband_drive"] = sideband_drive
    node.namespace["alpha_max"] = alpha_max

    # ---- Time sweep (log or linear, via IdleTimeNodeParameters) --------------
    # get_idle_times_in_clock_cycles returns clock cycles for the per-repeat wait.
    # delay_repeats multiplies the effective time: total = delay_repeats × t_per_rep.
    # So min/max_wait_time_in_ns define the per-repeat range, and the full sweep
    # spans [min, delay_repeats × max] ns total.
    delay_repeats = node.parameters.delay_repeats
    t_per_rep_clk = get_idle_times_in_clock_cycles(node.parameters)  # per-repeat clk

    # Total physical time stored as the dataset coordinate.
    t_actual_ns = (t_per_rep_clk * 4 * delay_repeats).astype(int)

    node.namespace["t_per_rep_clk"] = t_per_rep_clk
    node.namespace["t_actual_ns"] = t_actual_ns

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            t_actual_ns,
            attrs={"long_name": "total wait time", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)  # per-repeat clock cycles (arbitrary array → for_each_)

        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, t_per_rep_clk.tolist()):

                    # --- Reset cavity and qubit BEFORE displacement ---
                    sideband_drive = node.namespace["sideband_drive"]
                    for i, qubit in multiplexed_qubits.items():
                        cavity_mode.reset(
                            node.parameters.cavity_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            f0g1_pulse_duration_ns=node.parameters.f0g1_pulse_duration_ns,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # --- Displace the cavity ---
                    # align() with no args includes cavity_mode_drive, which is not part of qubit.align().
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=node.parameters.displacement_scale / node.namespace["alpha_max"],
                    )

                    # --- Wait for decay (delay_repeats × t) ---
                    align()
                    for _ in range(delay_repeats):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.wait(t)

                    # --- Selective π probe ---
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("selective_x180")

                    # --- Measure ---
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(t_per_rep_clk)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(t_per_rep_clk)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(t_per_rep_clk)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.namespace["cavity_mode"] = _get_cavity_mode(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    node.results["mode_name"] = node.parameters.mode_name

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q: ("successful" if res["success"] else "failed")
        for q, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_coherent_T1(
        node.results["ds_raw"],
        node.results["fit_results"],
        mode_name=node.parameters.mode_name,
        normalize_plot=node.parameters.normalize_plot,
    )
    plt.show()
    node.results["figures"] = {"coherent_T1": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    cavity_mode = node.namespace["cavity_mode"]

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            res = node.results["fit_results"].get(qubit.name)
            if res is None or not res["success"]:
                continue

            T1_s = res["T1_ns"] * 1e-9
            cavity_mode.T1 = float(T1_s)
            break  # single cavity mode shared across all qubits in this run


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
