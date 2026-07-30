# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.time_rabi import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        TIME RABI
This sequence plays a qubit drive pulse with variable duration and measures the resonator
for different pulse durations.  The result is a Rabi oscillation in the I quadrature from
which the π-pulse duration is extracted.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the qubit drive (node 01a).
    - Having calibrated the readout (time of flight, offsets, gains).
    - Having found the qubit frequency (node 03a_qubit_spectroscopy or 03c_qubit_spectroscopy_vs_power).

State update:
    - The qubit pulse duration for the selected operation:
      qubit.xy.operations[operation].length  (in nanoseconds)
    - If drive_power_dbm is set and the fit succeeds, the amplitude override
      is kept in the state (not reverted). If the fit fails it is reverted.
"""

node = QualibrationNode[Parameters, Quam](
    name="04c_time_rabi",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]
    # node.parameters.min_duration_ns = 16
    # node.parameters.max_duration_ns = 2000
    # node.parameters.duration_step_ns = 4
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the time Rabi QUA program and register sweep axes."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    op_amp_factor = node.parameters.operation_amplitude_factor

    # Duration sweep in clock cycles (4 ns each); enforce multiples of 4 ns
    min_cc = max(4, (node.parameters.min_duration_ns // 4))
    max_cc = max(min_cc + 1, (node.parameters.max_duration_ns // 4))
    step_cc = max(1, (node.parameters.duration_step_ns // 4))
    durations_cc = np.arange(min_cc, max_cc, step_cc, dtype=int)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration_cc": xr.DataArray(
            durations_cc,
            attrs={"long_name": "pulse duration", "units": "clock cycles"},
        ),
    }

    # Optionally override XY drive power (reverted in update_state)
    node.namespace["tracked_xy"] = []
    if node.parameters.drive_power_dbm is not None:
        for qubit in qubits:
            with tracked_updates(qubit.xy, auto_revert=False, dont_assign_to_none=True) as xy:
                xy.set_locked_output_power(power_in_dbm=node.parameters.drive_power_dbm, operation=operation)
                node.namespace["tracked_xy"].append(xy)
        node.log(
            f"Time Rabi: temporarily set XY drive power to "
            f"{node.parameters.drive_power_dbm:.1f} dBm "
            f"(max_amp={node.parameters.max_amplitude_opx})"
        )

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable: pulse duration in clock cycles

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(t, durations_cc)):
                    # --- Reset ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()

                    # --- Qubit drive ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play(operation, duration=t, amplitude_scale=op_amp_factor)
                    align()

                    # --- Readout ---
                    for i, qubit in multiplexed_qubits.items():
                        qubit.readout_state(
                            state[i] if node.parameters.use_state_discrimination else None,
                            I=I[i], Q=Q[i], I_st=I_st[i], Q_st=Q_st[i],
                            state_st=state_st[i] if node.parameters.use_state_discrimination else None,
                        )
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
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
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
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit Rabi oscillations and extract the π-pulse duration."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the time Rabi oscillations with fitted curves."""
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        fit_results=node.results["fit_results"],
    )
    plt.show()
    node.results["figures"] = {"time_rabi": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the qubit pulse duration if the fit was successful.

    If drive_power_dbm was set, the amplitude override is kept for successful
    qubits and reverted only for failed ones.
    """
    tracked_xy_list = node.namespace.get("tracked_xy", [])
    # Pad with None so we can always zip against qubits
    padded = list(tracked_xy_list) + [None] * (len(node.namespace["qubits"]) - len(tracked_xy_list))

    with node.record_state_updates():
        for q, tracked_xy in zip(node.namespace["qubits"], padded):
            if node.outcomes[q.name] != "successful":
                if tracked_xy is not None:
                    tracked_xy.revert_changes()
                    node.log(f"[{q.name}] Reverted XY amplitude override (fit failed).")
                continue

            # Fit succeeded: keep amplitude override (do not revert)
            if tracked_xy is not None:
                node.log(f"[{q.name}] Keeping XY amplitude override (fit succeeded).")

            pi_dur = node.results["fit_results"][q.name]["pi_duration_ns"]
            if np.isfinite(pi_dur) and pi_dur > 0:
                # Round to nearest 4 ns (QUA clock cycle)
                pi_dur_int = max(4, (int(pi_dur) // 4) * 4)
                sigma_val  = max(4, pi_dur_int // 5)

                # ── Rabi operation length ─────────────────────────────────────────
                # Do NOT update "saturation": it is a SquarePulse used for spectroscopy
                # and must keep its original long duration (e.g. 20 000 ns).
                # Only update if the operation is a calibrated π-pulse type (x180, etc.).
                if node.parameters.operation != "saturation":
                    q.xy.operations[node.parameters.operation].length = pi_dur_int
                    node.log(
                        f"[{q.name}] Updated {node.parameters.operation}: length={pi_dur_int} ns"
                    )

                # ── x180: DragGaussian π pulse ──────────────────────────────────────
                if "x180" in q.xy.operations:
                    q.xy.operations["x180"].length = pi_dur_int
                    try:
                        q.xy.operations["x180"].sigma = sigma_val
                    except (ValueError, KeyError, AttributeError):
                        pass
                    node.log(f"[{q.name}] Updated x180: length={pi_dur_int} ns, sigma={sigma_val} ns")

                # ── x90: mirrors x180 (may be a reference; explicit update as fallback) ──
                if "x90" in q.xy.operations:
                    try:
                        q.xy.operations["x90"].length = pi_dur_int
                        q.xy.operations["x90"].sigma = sigma_val
                    except (ValueError, KeyError, AttributeError):
                        pass

                # ── EF_x180: same length and sigma as x180 ──────────────────────
                # If EF_x180.length/sigma are QUAM references to x180, they update
                # automatically and direct assignment raises ValueError — catch it.
                if "EF_x180" in q.xy.operations:
                    try:
                        q.xy.operations["EF_x180"].length = pi_dur_int
                    except (ValueError, KeyError, AttributeError):
                        pass  # reference to x180.length propagates automatically
                    try:
                        q.xy.operations["EF_x180"].sigma = sigma_val
                    except (ValueError, KeyError, AttributeError):
                        pass  # reference to x180.sigma propagates automatically
                    node.log(f"[{q.name}] Updated EF_x180: length={pi_dur_int} ns, sigma={sigma_val} ns")

                # ── selective_x180: 100× π time, sigma = length/5 ──────────────────
                # Narrow-bandwidth pulse: long duration → small bandwidth ≈ 1/(pi_dur*100)
                sel_len_int = max(4, (pi_dur_int * 100 // 4) * 4)
                sel_sigma   = max(4, sel_len_int // 5)
                if "selective_x180" in q.xy.operations:
                    q.xy.operations["selective_x180"].length = sel_len_int
                    try:
                        q.xy.operations["selective_x180"].sigma = sel_sigma
                    except (ValueError, KeyError, AttributeError):
                        pass
                    node.log(
                        f"[{q.name}] Updated selective_x180: "
                        f"length={sel_len_int} ns, sigma={sel_sigma} ns"
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
