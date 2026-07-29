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
from calibration_utils.shared import apply_confusion_matrix_correction, _get_cavity_mode
from calibration_utils.cavity_coherent_T1 import (
    Parameters,
    CoherentT1Fit,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_coherent_T1,
)

logger = logging.getLogger(__name__)

_AMP_MAX = 2.0 - 2**-16  # QUA hardware limit


# %% {Node initialisation}
description = """
        CAVITY COHERENT T1 (23)

Measures the energy relaxation time T1 of a selected cavity mode by preparing
a coherent state |α⟩ and probing the vacuum-state population with a selective
qubit π-pulse.

Sequence (per wait time t):
  1. Thermalize cavity (wait ≥ 5×T1) and reset qubit.
  2. Apply displacement pulse: amplitude_scale = displacement_alpha / displacement_alpha_max.
  3. Wait for total time t = delay_repeats × t_per_rep.
  4. Apply selective_x180 on qubit — flips qubit only when cavity is in |0⟩.
  5. Measure qubit state.

The measured signal is:
    P_e(t) = A · exp(-|α₀|² · exp(-t / T1)) + offset

where |α₀|² = displacement_alpha² and T1 is the cavity photon lifetime.

Fitting extracts T1 and |α₀|².  The dataset is augmented with the inferred
photon-number decay |α(t)|² = -ln((P_e - offset) / A), plotted as a simple
exponential: |α₀|² · exp(-t / T1).

Parameters:
  - mode_name:           Cavity mode to probe ('alice' or 'bob').
  - displacement_alpha:  Desired coherent-state amplitude α.
                         amplitude_scale = displacement_alpha / displacement_alpha_max
                         (displacement_alpha_max read from CavityTransmonPair QuAM state).
                         After node 26/30 calibration: displacement_alpha=1 → 1 photon.
  - min/max_wait_time_in_ns / wait_time_num_points: time sweep range.
  - delay_repeats:       Multiply effective wait time to extend range.

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
    # node.parameters.displacement_alpha = 1.0
    # node.parameters.min_wait_time_in_ns = 16
    # node.parameters.max_wait_time_in_ns = 5_000_000
    # node.parameters.wait_time_num_points = 51
    # node.parameters.delay_repeats = 1
    # node.parameters.num_shots = 1000
    pass


node.machine = Quam.load()


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

    # Resolve sideband_drive and displacement_alpha_max from the QuAM state
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

    # Compute and validate the QUA amplitude_scale
    amplitude_scale = node.parameters.displacement_alpha / alpha_max
    if abs(amplitude_scale) > _AMP_MAX:
        raise ValueError(
            f"displacement_alpha={node.parameters.displacement_alpha} / "
            f"alpha_max={alpha_max} = {amplitude_scale:.4f} exceeds the QUA "
            f"hardware limit ±{_AMP_MAX:.6f}.  Reduce displacement_alpha."
        )
    node.namespace["amplitude_scale"] = amplitude_scale
    node.log(
        f"Displacement: alpha={node.parameters.displacement_alpha}, "
        f"alpha_max={alpha_max}, amplitude_scale={amplitude_scale:.4f}"
    )

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

    subtract_baseline = node.parameters.subtract_baseline

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)  # per-repeat clock cycles (arbitrary array → for_each_)

        if subtract_baseline:
            I_base = [declare(fixed) for _ in range(num_qubits)]
            Q_base = [declare(fixed) for _ in range(num_qubits)]
            I_base_st = [declare_stream() for _ in range(num_qubits)]
            Q_base_st = [declare_stream() for _ in range(num_qubits)]
            # When subtracting the baseline, state discrimination cannot be computed
            # per-shot inside QUA: the threshold must be applied to
            # (I_signal - I_baseline), but those come from separate shots and are
            # only both available after averaging.  State discrimination is therefore
            # deferred to Python (process_raw_dataset).

        if not subtract_baseline and node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_each_(t, t_per_rep_clk.tolist()):

                    # ============================================================
                    # PART 1 — BASELINE measurement (only when subtract_baseline=True)
                    # ============================================================
                    # Purpose: capture the bare readout-resonator IQ response at each
                    # wait time WITHOUT any qubit drive.  The cavity-resonator cross-Kerr
                    # coupling shifts the resonator IQ as a function of photon number,
                    # so this baseline is time-dependent and must be measured
                    # independently at every point.
                    if subtract_baseline:
                        sideband_drive = node.namespace["sideband_drive"]
                        for i, qubit in multiplexed_qubits.items():
                            cavity_mode.reset(
                                node.parameters.cavity_reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                                sideband_drive=sideband_drive,
                                qubit_thermalization_time=qubit.thermalization_time,
                                fock_n=node.parameters.cavity_active_cooling_fock_n,
                                sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            )
                            qubit.reset(
                                node.parameters.reset_type,
                                node.parameters.simulate,
                                log_callable=node.log,
                            )

                        align()
                        cavity_mode.cavity_mode_drive.play(
                            "displacement",
                            amplitude_scale=node.namespace["amplitude_scale"],
                        )

                        # Wait for the same duration as the signal sequence so the
                        # cross-Kerr environment is identical in both sub-sequences.
                        align()
                        for _ in range(delay_repeats):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.wait(t)

                        # NO selective π-pulse — qubit stays in |g⟩.
                        # Measure baseline IQ (cross-Kerr shift only, no vacuum signal).
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.readout_state(None, I=I_base[i], Q=Q_base[i], I_st=I_base_st[i], Q_st=Q_base_st[i])
                        align()

                    # ============================================================
                    # PART 2 — SIGNAL measurement (full protocol, WITH π-pulse)
                    # ============================================================
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
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                        )
                        qubit.reset(
                            node.parameters.reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                        )

                    # --- Displace the cavity ---
                    # amplitude_scale = displacement_alpha / alpha_max
                    # so the actual coherent amplitude is displacement_alpha.
                    # align() with no args includes cavity_mode_drive, which is not part of qubit.align().
                    align()
                    cavity_mode.cavity_mode_drive.play(
                        "displacement",
                        amplitude_scale=node.namespace["amplitude_scale"],
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
                        qubit.readout_state(
                            state[i] if (not subtract_baseline and node.parameters.use_state_discrimination) else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if (not subtract_baseline and node.parameters.use_state_discrimination) else None,
                        )
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(t_per_rep_clk)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(t_per_rep_clk)).average().save(f"Q{i + 1}")
                if subtract_baseline:
                    # Named Ib{i+1} / Qb{i+1} so XarrayDataFetcher groups them into
                    # dataset variables 'Ib' and 'Qb', stacked along the qubit axis.
                    I_base_st[i].buffer(len(t_per_rep_clk)).average().save(f"Ib{i + 1}")
                    Q_base_st[i].buffer(len(t_per_rep_clk)).average().save(f"Qb{i + 1}")
                elif node.parameters.use_state_discrimination:
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
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(node.results["ds_raw"], node.namespace["qubits"])
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
        node.results["ds_fit"],
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
