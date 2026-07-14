# %% {Imports}
import datetime
from dataclasses import asdict
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.readout_gef_freq_amp_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_gef_fidelity_map,
    plot_iq_blobs_at_optimal,
    plot_fidelity_vs_frequency,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot


# %% {Description}
description = """
        GEF READOUT FREQUENCY × AMPLITUDE OPTIMISATION
Performs a 2D joint sweep over readout IF detuning (relative to the current GEF operating
frequency) and readout pulse amplitude.  At every grid point single-shot IQ data is acquired
for all three states (|g⟩, |e⟩, |f⟩) and the true 3-state discrimination fidelity is computed
from a full 3×3 confusion matrix.  The joint optimum is selected as the argmax of this fidelity.

This combines and supersedes running nodes 14 and 14b sequentially, which optimise frequency
and amplitude in isolation using a pairwise IQ-distance proxy rather than true fidelity.

Execution strategy (iterative):
  One QUA job is submitted per amplitude step, sweeping the full frequency range inside each
  job.  After each job, a NumPy checkpoint (.npz) is written to C:/tmp/.  Partial results
  survive a crash — re-run with load_data_id to resume from a saved dataset.

Prerequisites:
  - GEF readout frequency roughly calibrated (node 14 recommended first-pass).
  - ge and EF π pulses calibrated (nodes 04b, 13).
  - qubit.anharmonicity set.

State updates:
  - qubit.resonator.GEF_frequency_shift  += optimal GEF detuning found
  - qubit.resonator.operations[operation].amplitude  =  optimal amplitude
"""

node = QualibrationNode[Parameters, Quam](
    name="14d_readout_gef_freq_amp_optimization",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Local parameter overrides for interactive debugging."""
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 200
    # node.parameters.frequency_span_in_mhz = 2.0
    # node.parameters.min_amp_factor = 0.5
    # node.parameters.max_amp_factor = 1.5
    # node.parameters.num_amps = 11
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Pre-compute sweep arrays and build a representative QUA program for simulation."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    p = node.parameters

    span = int(p.frequency_span_in_mhz * u.MHz)
    step = int(p.frequency_step_in_mhz * u.MHz)
    if_detuning_array = np.arange(-span // 2, span // 2, step, dtype=int)
    node.namespace["if_detuning_array"] = if_detuning_array

    amp_array = np.linspace(p.min_amp_factor, p.max_amp_factor, p.num_amps)
    node.namespace["amp_array"] = amp_array

    node.log(
        f"Frequency sweep: {len(if_detuning_array)} points "
        f"({if_detuning_array[0]/1e6:.2f} … {if_detuning_array[-1]/1e6:.2f} MHz)"
    )
    node.log(
        f"Amplitude sweep: {len(amp_array)} steps "
        f"({amp_array[0]:.3f} … {amp_array[-1]:.3f})"
    )

    # Representative program (g state, first amplitude) for simulation only
    node.namespace["qua_program"] = _build_state_program(
        qubits, if_detuning_array, p.num_shots, "g", float(amp_array[0]), p
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Iterative execution: one QUA job per (state, amplitude) pair with checkpointing."""
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    if_detuning_array = node.namespace["if_detuning_array"]
    amp_array = node.namespace["amp_array"]
    p = node.parameters
    states = ["g", "e", "f"]
    n_freq = len(if_detuning_array)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path("C:/tmp") / f"qualibrate_gef_readout_opt_{ts}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    node.namespace["checkpoint_dir"] = checkpoint_dir
    node.log(f"Checkpoint directory: {checkpoint_dir}")

    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qm = qmm.open_qm(config)

    try:
        completed_steps = []

        for amp_idx, amp_factor in enumerate(amp_array):
            if p.max_readout_amplitude_v is not None:
                actual_amps_v = [
                    amp_factor * q.resonator.operations[p.operation].amplitude
                    for q in qubits
                ]
                if any(v > p.max_readout_amplitude_v for v in actual_amps_v):
                    node.log(
                        f"Skipping amp_scale={amp_factor:.4f}: "
                        f"exceeds max_readout_amplitude_v={p.max_readout_amplitude_v:.4f} V"
                    )
                    continue

            node.log(
                f"Amplitude step {amp_idx + 1}/{len(amp_array)} | "
                f"scale={amp_factor:.4f} | "
                f"amp={amp_factor * qubits[0].resonator.operations[p.operation].amplitude * 1e3:.2f} mV"
            )

            state_data = {}

            for state in states:
                qua_prog = _build_state_program(
                    qubits, if_detuning_array, p.num_shots, state, float(amp_factor), p
                )
                job = qm.execute(qua_prog)
                handles = job.result_handles

                while handles.is_processing():
                    try:
                        iteration = handles.get("iteration").fetch_all()
                        if iteration is not None:
                            progress_counter(int(iteration), n_freq)
                    except Exception:
                        pass
                    sleep(0.5)
                progress_counter(n_freq, n_freq)

                for qi in range(num_qubits):
                    raw_i = handles.get(f"I{state}{qi + 1}").fetch_all()
                    raw_q = handles.get(f"Q{state}{qi + 1}").fetch_all()
                    if raw_i is None or raw_q is None:
                        raise RuntimeError(
                            f"Failed to fetch data for state={state}, qubit_idx={qi}, "
                            f"amp_scale={amp_factor:.4f}."
                        )
                    state_data[f"I{state}{qi}"] = np.asarray(raw_i)   # (n_freq, n_shots)
                    state_data[f"Q{state}{qi}"] = np.asarray(raw_q)

            cp_path = checkpoint_dir / f"amp_step_{amp_idx:03d}_scale_{amp_factor:.4f}.npz"
            np.savez(
                cp_path,
                amp_scale=float(amp_factor),
                if_detuning_array=if_detuning_array,
                qubit_names=np.array([q.name for q in qubits]),
                **state_data,
            )
            completed_steps.append(amp_idx)
            node.log(f"  Saved checkpoint: {cp_path.name}")

    finally:
        qm.close()

    node.log(
        f"Execution complete: {len(completed_steps)}/{len(amp_array)} amplitude steps.\n"
        f"Checkpoints: {checkpoint_dir}"
    )

    node.results["ds_raw"] = _combine_checkpoints(checkpoint_dir, qubits)
    node.results["checkpoint_dir"] = str(checkpoint_dir)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Compute the 3-state fidelity map and extract optimal operating point parameters."""
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
    ds_fit = node.results["ds_fit"]
    fit_results = node.results["fit_results"]
    qubits = node.namespace["qubits"]

    fig_map = plot_gef_fidelity_map(ds_fit, qubits, fit_results)
    fig_blobs = plot_iq_blobs_at_optimal(ds_fit, qubits, fit_results)
    fig_freq = plot_fidelity_vs_frequency(ds_fit, qubits, fit_results)
    plt.show()

    node.results["figures"] = {
        "fidelity_map": fig_map,
        "iq_blobs_at_optimal": fig_blobs,
        "fidelity_vs_frequency": fig_freq,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update GEF_frequency_shift and readout amplitude from the jointly optimised operating point."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                node.log(f"Skipping state update for {q.name}: analysis failed.")
                continue

            res = node.results["fit_results"][q.name]
            operation = q.resonator.operations[node.parameters.operation]

            if q.resonator.GEF_frequency_shift is None:
                q.resonator.GEF_frequency_shift = 0.0
            q.resonator.GEF_frequency_shift += float(res["optimal_gef_detuning"])
            operation.amplitude = float(res["optimal_amplitude_v"])

            node.log(
                f"Updated {q.name}: "
                f"GEF_freq_shift += {res['optimal_gef_detuning']/1e6:.3f} MHz, "
                f"amp = {res['optimal_amplitude_v']*1e3:.2f} mV, "
                f"GEF fidelity = {res['max_fidelity']*100:.2f} %"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_state_program(qubits, if_detuning_array, n_shots, state, amp_factor, p):
    """
    Build a QUA program that sweeps the GEF readout frequency (outer loop) and
    collects n_shots single-shot IQ measurements per frequency point for one prepared state.

    The amplitude is baked in at compile time, so a new program is compiled per amplitude step.
    Stream names: I{state}{1-indexed qubit}, Q{state}{1-indexed qubit}.
    Output shape after fetch: (n_freq, n_shots).
    """
    num_qubits = len(qubits)

    with program() as qua_prog:
        n = declare(int)
        df = declare(int)
        counter = declare(int, value=0)
        counter_st = declare_stream()

        I_vars = [declare(fixed) for _ in range(num_qubits)]
        Q_vars = [declare(fixed) for _ in range(num_qubits)]
        I_streams = [declare_stream() for _ in range(num_qubits)]
        Q_streams = [declare_stream() for _ in range(num_qubits)]

        with for_(*from_array(df, if_detuning_array)):
            # Set resonator IF to GEF base frequency + current detuning step
            for qubit in qubits:
                gef_shift = qubit.resonator.GEF_frequency_shift or 0
                update_frequency(
                    qubit.resonator.name,
                    qubit.resonator.intermediate_frequency + gef_shift + df,
                )
            save(counter, counter_st)

            with for_(n, 0, n < n_shots, n + 1):
                # ── State preparation ────────────────────────────────────────
                if state == "g":
                    for qubit in qubits:
                        qubit.reset(p.reset_type, p.simulate)
                    align()

                elif state == "e":
                    for qubit in qubits:
                        qubit.reset(p.reset_type, p.simulate)
                    align()
                    for qubit in qubits:
                        qubit.xy.play("x180")
                        qubit.align()

                elif state == "f":
                    for qubit in qubits:
                        qubit.reset(p.reset_type, p.simulate)
                    align()
                    for qubit in qubits:
                        qubit.xy.play("x180")
                        qubit.align()
                        update_frequency(
                            qubit.xy.name,
                            qubit.xy.intermediate_frequency + qubit.anharmonicity,
                        )
                        qubit.xy.play(p.ef_pi_pulse)
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        qubit.align()

                # ── Readout ─────────────────────────────────────────────────
                for i, qubit in enumerate(qubits):
                    qubit.resonator.measure(
                        p.operation,
                        qua_vars=(I_vars[i], Q_vars[i]),
                        amplitude_scale=amp_factor,
                    )
                    qubit.resonator.wait(qubit.resonator.depletion_time // 4)
                    save(I_vars[i], I_streams[i])
                    save(Q_vars[i], Q_streams[i])

            assign(counter, counter + 1)

        with stream_processing():
            counter_st.save("iteration")
            for i in range(num_qubits):
                I_streams[i].buffer(n_shots).buffer(len(if_detuning_array)).save(f"I{state}{i + 1}")
                Q_streams[i].buffer(n_shots).buffer(len(if_detuning_array)).save(f"Q{state}{i + 1}")

    return qua_prog


def _combine_checkpoints(checkpoint_dir: Path, qubits) -> xr.Dataset:
    """Load all .npz checkpoint files and assemble into an xarray Dataset with dims
    (qubit, frequency, amplitude, shot)."""
    cp_files = sorted(checkpoint_dir.glob("amp_step_*.npz"))
    if not cp_files:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_dir}")

    qubit_names = [q.name for q in qubits]
    num_qubits = len(qubits)
    states = ["g", "e", "f"]

    amp_scales = []
    all_data = []

    for cp_file in cp_files:
        cp = np.load(cp_file, allow_pickle=True)
        amp_scales.append(float(cp["amp_scale"]))
        step = {}
        for state in states:
            for qi in range(num_qubits):
                step[(state, qi)] = (cp[f"I{state}{qi}"], cp[f"Q{state}{qi}"])
        all_data.append(step)
        if_detuning_array = cp["if_detuning_array"]

    amp_array = np.array(amp_scales)
    n_amp = len(amp_array)
    n_shots = all_data[0][("g", 0)][0].shape[1]

    ds_vars = {}
    for state in states:
        for qi, q_name in enumerate(qubit_names):
            # Each checkpoint entry has shape (n_freq, n_shots); stack over amplitude → (n_freq, n_amp, n_shots)
            I_all = np.stack([all_data[ai][(state, qi)][0] for ai in range(n_amp)], axis=1)
            Q_all = np.stack([all_data[ai][(state, qi)][1] for ai in range(n_amp)], axis=1)
            ds_vars.setdefault(f"I{state}", []).append(I_all)
            ds_vars.setdefault(f"Q{state}", []).append(Q_all)

    ds_data_vars = {}
    for key, arrays in ds_vars.items():
        # Stack over qubit axis → (n_qubits, n_freq, n_amp, n_shots)
        stacked = np.stack(arrays, axis=0)
        ds_data_vars[key] = xr.DataArray(stacked, dims=["qubit", "frequency", "amplitude", "shot"])

    return xr.Dataset(
        ds_data_vars,
        coords={
            "qubit": qubit_names,
            "frequency": xr.DataArray(
                if_detuning_array.astype(float),
                dims=["frequency"],
                attrs={"long_name": "GEF readout IF detuning", "units": "Hz"},
            ),
            "amplitude": xr.DataArray(
                amp_array,
                dims=["amplitude"],
                attrs={"long_name": "amplitude scale factor"},
            ),
            "shot": np.arange(n_shots),
        },
    )

# %%
