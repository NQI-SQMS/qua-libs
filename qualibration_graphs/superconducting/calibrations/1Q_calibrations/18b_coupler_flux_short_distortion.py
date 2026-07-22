"""Cryoscope calibration for coupler flux line — spectroscopy + Ramsey amplitude paths."""


# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_flux_short_distortion import (
    Parameters,
    baked_coupler_waveform,
    fit_fir_data,
    fit_raw_data,
    log_fitted_results,
    plot_cryoscope_freq,
    plot_fir_figures,
    plot_fit,
    plot_flux_response,
    plot_raw_data,
    plot_unwrapped_phase,
    process_raw_dataset,
)
from calibration_utils.coupler_flux_long_distortion_qubitspec import (
    _derive_coupler_flux_from_decouple,
    _load_coupler_spectroscopy_curve,
    _load_ramseyflux_curve_from_param,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Node_parameters}
description = """
COUPLER FLUX CRYOSCOPE (22c — spectroscopy + Ramsey amplitude paths, with optional FIR)

Same pulse sequence as 22b: Ramsey-style cryoscope sweeping coupler flux pulse duration
at a fixed amplitude, with frame rotation for phase reconstruction.

Extends 22b with a three-path cascade for computing the cryoscope flux amplitude:

  Path 1 — Coupler spectroscopy (use_spectroscopy_data=True, spectroscopy_run_id set):
      Loads the freq-vs-coupler-flux curve from a qubit-spectroscopy-vs-coupler-flux
      run (node 10) and inverts it to find the coupler flux amplitude that achieves
      the target detuning.

  Path 2 — Ramsey vs coupler flux (use_ramsey_data=True, ramsey_run_id set):
      Loads the freq-vs-coupler-flux curve from a Ramsey calibration run and performs
      the same inversion.

  Path 3 — Fixed amplitude fallback:
      Uses coupler_flux_amplitude_in_v directly.  Always available; least accurate.

Paths are attempted in order; the first successful result is used per qubit pair.

Post-processing, IIR fitting, and optional FIR analysis are identical to 22b,
using qubit_flux_short_distortion library functions.

Prerequisites:
        - Resonator spectroscopy performed.
        - Qubit gates (x90, y90) calibrated.
        - For Path 1: a completed qubit-spectroscopy-vs-coupler-flux run (e.g. node 10).
        - For Path 2: a completed Ramsey-vs-coupler-flux run.

Next steps:
        - Push IIR (and optionally FIR) filter taps into the configuration.
        - WARNING: digital filters add a global delay — recalibrate IQ blobs.
"""

node = QualibrationNode[Parameters, Quam](
    name="18b_coupler_flux_short_distortion",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)



@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""

    # Three-path amplitude cascade

    # FIR analysis
    pass


# Instantiate the QUAM class from the state file
stored_machine = Quam.load()

loaded_fractions = node.parameters.exponential_fit_time_fractions
stored_use_fir = node.parameters.use_fir
stored_update_iir = node.parameters.update_iir
stored_update_fir = node.parameters.update_fir


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for coupler flux cryoscope measurement."""
    u = unit(coerce_to_integer=True)

    # --- Resolve qubit pairs and extract measured qubits --------------------
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Select which qubit in each pair acts as the Ramsey sensor
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    # Set "qubits" for compatibility with cryoscope analysis functions
    node.namespace["qubits"] = measured_qubits

    n_avg = node.parameters.num_shots
    cryoscope_len = node.parameters.cryoscope_len

    # ------------------------------------------------------------------
    # Amplitude resolution — three-path cascade per qubit pair
    # ------------------------------------------------------------------
    use_spec = node.parameters.use_spectroscopy_data
    spec_run_id = node.parameters.spectroscopy_run_id
    use_ramsey = node.parameters.use_ramsey_data
    ramsey_run_id = node.parameters.ramsey_run_id

    branch = node.parameters.coupler_flux_branch

    amplitudes = {}
    amplitude_resolution = {}
    for qp in qubit_pairs:
        q = qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
        amp = None
        source = None
        decouple_offset = qp.coupler.decouple_offset
        detuning_hz = node.parameters.detuning_target_in_mhz * 1e6

        # Path 1: coupler spectroscopy (node 10)
        if use_spec and spec_run_id is not None:
            try:
                curve = _load_coupler_spectroscopy_curve(spec_run_id, q, qp.coupler, node)
                if curve is not None:
                    result = _derive_coupler_flux_from_decouple(
                        detuning_hz,
                        decouple_offset,
                        curve,
                        branch=branch,
                    )
                    if result is not None:
                        amp, _freq_at_decouple = result
                    if amp is not None:
                        source = "spectroscopy"
                        node.log(f"  {qp.coupler.name}: flux_amp={amp:.6f} V (from spec run #{spec_run_id})")
                    else:
                        node.log(f"  {qp.coupler.name}: detuning outside spec range, trying next path")
                else:
                    node.log(f"  {qp.coupler.name}: spec curve not found in run #{spec_run_id}, trying next path")
            except Exception as exc:
                node.log(f"  {qp.coupler.name}: spec load failed ({exc}), trying next path")

        # Path 2: Ramsey vs coupler flux
        if amp is None and use_ramsey and ramsey_run_id is not None:
            try:
                curve = _load_ramseyflux_curve_from_param(ramsey_run_id, q, qp.coupler, node)
                if curve is not None:
                    result = _derive_coupler_flux_from_decouple(
                        detuning_hz,
                        decouple_offset,
                        curve,
                        branch=branch,
                    )
                    if result is not None:
                        amp, _freq_at_decouple = result
                    if amp is not None:
                        source = "ramsey"
                        node.log(f"  {qp.coupler.name}: flux_amp={amp:.6f} V (from Ramsey run #{ramsey_run_id})")
                    else:
                        node.log(f"  {qp.coupler.name}: detuning outside Ramsey range, falling back to fixed amplitude")
                else:
                    node.log(
                        f"  {qp.coupler.name}: Ramsey curve not found in run #{ramsey_run_id}, "
                        "falling back to fixed amplitude"
                    )
            except Exception as exc:
                node.log(f"  {qp.coupler.name}: Ramsey load failed ({exc}), falling back to fixed amplitude")

        # Path 3: fixed amplitude fallback (no quad_term for couplers)
        if amp is None:
            amp = node.parameters.coupler_flux_amplitude_in_v
            source = "fixed"
            node.log(f"  {qp.coupler.name}: flux_amp={amp:.6f} V (from coupler_flux_amplitude_in_v)")

        amplitudes[qp.coupler.name] = amp
        amplitude_resolution[qp.coupler.name] = {
            "amp_v": amp,
            "source": source,
            "detuning_mhz": node.parameters.detuning_target_in_mhz,
            "branch": branch,
        }

    # Record how each coupler amplitude was resolved for traceability.
    node.results["amplitude_resolution"] = amplitude_resolution

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)
    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()

    baked_signals = {
        qp.coupler.name: baked_coupler_waveform(baked_config, amplitudes[qp.coupler.name], qp.coupler, max_length=16)
        for qp in qubit_pairs
    }

    node.namespace["baked_config"] = baked_config

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        t_left_ns = declare(int)
        t_cycles = declare(int)
        idx = declare(int)
        frame = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    with for_each_(frame, frames):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        ################################################################################################
                        # The duration argument in the play command can only produce pulses with duration multiple of  #
                        # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                        # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                        # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                        ################################################################################################
                        with if_(idx <= 16):
                            with switch_(idx):
                                for j in range(1, 17):
                                    with case_(j):
                                        align()
                                        for ii, qp in multiplexed_qubit_pairs.items():
                                            protagonist = (
                                                qp.qubit_control
                                                if node.parameters.measure_qubit == "control"
                                                else qp.qubit_target
                                            )
                                            protagonist.xy.play("x90")
                                            qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qp.coupler.name][j - 1].run()
                                            protagonist.xy.wait((cryoscope_len + 16) >> 2)
                                            protagonist.xy.frame_rotation_2pi(frame)
                                            protagonist.xy.play("x90")
                        with else_():
                            assign(t_cycles, idx >> 2)
                            assign(t_left_ns, idx - (t_cycles << 2))
                            with switch_(t_left_ns):
                                with case_(0):
                                    align()
                                    for ii, qp in multiplexed_qubit_pairs.items():
                                        protagonist = (
                                            qp.qubit_control
                                            if node.parameters.measure_qubit == "control"
                                            else qp.qubit_target
                                        )
                                        protagonist.xy.play("x90")
                                        qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                        qp.coupler.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=amplitudes[qp.coupler.name]
                                            / qp.coupler.operations["const"].amplitude,
                                        )
                                        protagonist.xy.wait((cryoscope_len + 16) // 4)
                                        protagonist.xy.frame_rotation_2pi(frame)
                                        protagonist.xy.play("x90")
                                for j in range(1, 4):
                                    with case_(j):
                                        align()
                                        for ii, qp in multiplexed_qubit_pairs.items():
                                            protagonist = (
                                                qp.qubit_control
                                                if node.parameters.measure_qubit == "control"
                                                else qp.qubit_target
                                            )
                                            protagonist.xy.play("x90")
                                            qp.coupler.wait((protagonist.xy.operations["x90"].length + 16) // 4)
                                            qp.coupler.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=amplitudes[qp.coupler.name]
                                                / qp.coupler.operations["const"].amplitude,
                                            )
                                            baked_signals[qp.coupler.name][j - 1].run()
                                            protagonist.xy.wait((cryoscope_len + 16) // 4)
                                            protagonist.xy.frame_rotation_2pi(frame)
                                            protagonist.xy.play("x90")

                        align()
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            if node.parameters.use_state_discrimination:
                                protagonist.readout_state(state[ii])
                                save(state[ii], state_st[ii])
                            else:
                                protagonist.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program, fetch the raw data and store it
    in an xarray dataset called "ds_raw".
    """
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
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

    # Rename "qubit_pair" -> "qubit" for cryoscope analysis compatibility.
    if "qubit_pair" in dataset.dims:
        qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_names)
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    ds_raw = node.results.get("ds_raw")
    if ds_raw is not None and "qubit_pair" in ds_raw.dims:
        qubit_names = [q.name for q in measured_qubits]
        ds_raw = ds_raw.rename({"qubit_pair": "qubit"})
        ds_raw = ds_raw.assign_coords(qubit=qubit_names)
    if ds_raw is not None:
        node.results = {"ds_raw": ds_raw}

    node.parameters.exponential_fit_time_fractions = loaded_fractions
    node.parameters.use_fir = stored_use_fir
    node.parameters.update_iir = stored_update_iir
    node.parameters.update_fir = stored_update_fir
    if stored_update_iir or stored_update_fir:
        node.machine = stored_machine
        print(
            f"State update enabled: IIR={stored_update_iir}, FIR={stored_update_fir}"
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data, store the fitted data in an xarray dataset "ds_fit" and
    the fitted results in the "fit_results" dictionary.
    """
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

    # If dispersion_run_id is explicitly set via ramsey path, inject it into each
    # measured qubit's extras so that qubit_flux_short_distortion.fit_raw_data uses the correct
    # dispersion snapshot for flux fitting.
    _run_id = node.parameters.ramsey_run_id if node.parameters.use_ramsey_data else None
    _injected = {}
    if _run_id is not None:
        for qp, q in zip(node.namespace["qubit_pairs"], node.namespace["measured_qubits"]):
            _key = f"{qp.coupler.name}_dispersion_load_id"
            _injected[q.name] = (q, _key, q.extras.get(_key))
            q.extras[_key] = _run_id

    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Restore original extras values
    for q, _key, _old in _injected.values():
        if _old is None:
            q.extras.pop(_key, None)
        else:
            q.extras[_key] = _old

    log_fitted_results(fit_results, log_callable=node.log)

    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
    measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
    node.outcomes = {
        qubit_pair_name: (
            "successful" if node.results["fit_results"].get(measured_qubit_name, {}).get("success", False) else "failed"
        )
        for qubit_pair_name, measured_qubit_name in zip(qubit_pair_names, measured_qubit_names)
    }

    # --- FIR analysis (optional) ---
    if node.parameters.use_fir:
        fir_results = fit_fir_data(node.results["ds_fit"], node)
        node.namespace["fir_results"] = fir_results
        node.results["fir_results"] = {
            qn: {k: v for k, v in res.items() if not str(k).startswith("fig")}
            for qn, res in fir_results.items()
        }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot all analysis stages and reference curves.

    Figure inventory (always generated):
      raw_<qname>              — raw state/I vs frame: line slices + 2D heatmap.
      unwrapped_phase          — unwrapped phase vs time for all qubits.
      cryoscope_freq_linear    — cryoscope frequency vs time (linear x-axis).
      cryoscope_freq_log       — cryoscope frequency vs time (log x-axis).
      flux_response_linear     — flux step response vs time (linear x-axis).
      flux_response_log        — flux step response vs time (log x-axis).
      iir_fitted_data          — IIR exponential fit overlay (linear + log).

    Reference curve figures (conditional on parameter flags and available run IDs):
      spectroscopy_curve       — qubit freq vs coupler flux from spectroscopy run.
      ramsey_curve             — qubit freq vs coupler flux from Ramsey run.

    Additional figures when use_fir=True and FIR analysis succeeded:
      fir_resampled_<qname>          — 1 GS/s vs 2 GS/s normalised flux.
      fir_fit_diagnostic_<qname>     — forward FIR fit 2x2 diagnostic.
      fir_inverse_diagnostic_<qname> — inverse FIR 3x2 diagnostic.
      fir_corrected_<qname>          — corrected response validation at 1 GS/s.
      fir_stem_<qname>               — FIR coefficient stem plots.
    """
    if "qubits" not in node.namespace:
        node.namespace["qubits"] = node.namespace["measured_qubits"]

    ds_raw = node.results["ds_raw"]
    ds_fit = node.results["ds_fit"]
    qubits = node.namespace["qubits"]
    fir_results = node.namespace.get("fir_results", {})
    n_q = len(qubits)

    figures = {}

    # Raw measurement data
    figures.update(plot_raw_data(ds_raw, qubits))

    # Unwrapped phase (output of process_raw_dataset)
    figures["unwrapped_phase"] = plot_unwrapped_phase(ds_fit, qubits)

    # Cryoscope frequency (linear + log)
    figures["cryoscope_freq_linear"] = plot_cryoscope_freq(ds_fit, qubits, log_scale=False)
    figures["cryoscope_freq_log"] = plot_cryoscope_freq(ds_fit, qubits, log_scale=True)

    # Flux step response (linear + log)
    figures["flux_response_linear"] = plot_flux_response(ds_fit, qubits, log_scale=False)
    figures["flux_response_log"] = plot_flux_response(ds_fit, qubits, log_scale=True)

    # IIR exponential fit overlay
    figures["iir_fitted_data"] = plot_fit(ds_fit, qubits, fit_results=node.results["fit_results"])

    # --- Reference curve: coupler spectroscopy (Path 1) ---
    if node.parameters.use_spectroscopy_data and node.parameters.spectroscopy_run_id is not None:
        qubit_pairs = node.namespace["qubit_pairs"]
        fig_spec, axes = plt.subplots(1, n_q, figsize=(5 * n_q, 4), squeeze=False)
        for ax, (q, qp) in zip(axes[0], zip(qubits, qubit_pairs)):
            curve = _load_coupler_spectroscopy_curve(node.parameters.spectroscopy_run_id, q, qp.coupler, node)
            if curve is not None:
                ax.plot(curve[0], np.array(curve[1]) / 1e9, marker=".", linestyle="-")
            ax.set_xlabel("Coupler flux (V)")
            ax.set_ylabel("Qubit frequency (GHz)")
            ax.set_title(f"{q.name} vs {qp.coupler.name}")
            ax.grid(True)
        fig_spec.suptitle(
            f"Qubit spectroscopy vs coupler flux — run #{node.parameters.spectroscopy_run_id}"
        )
        fig_spec.tight_layout()
        figures["spectroscopy_curve"] = fig_spec

    # --- Reference curve: Ramsey vs coupler flux (Path 2) ---
    if node.parameters.use_ramsey_data and node.parameters.ramsey_run_id is not None:
        qubit_pairs = node.namespace["qubit_pairs"]
        fig_ram, axes = plt.subplots(1, n_q, figsize=(5 * n_q, 4), squeeze=False)
        for ax, (q, qp) in zip(axes[0], zip(qubits, qubit_pairs)):
            curve = _load_ramseyflux_curve_from_param(node.parameters.ramsey_run_id, q, qp.coupler, node)
            if curve is not None:
                ax.plot(curve[0], np.array(curve[1]) / 1e9, marker=".", linestyle="-")
            ax.set_xlabel("Coupler flux (V)")
            ax.set_ylabel("Qubit frequency (GHz)")
            ax.set_title(f"{q.name} vs {qp.coupler.name}")
            ax.grid(True)
        fig_ram.suptitle(f"Ramsey vs coupler flux — run #{node.parameters.ramsey_run_id}")
        fig_ram.tight_layout()
        figures["ramsey_curve"] = fig_ram

    # FIR diagnostics (only when FIR analysis ran)
    if fir_results:
        figures.update(plot_fir_figures(ds_fit, qubits, fir_results))

    node.results["figures"] = figures
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update each pair's coupler exponential filter (IIR) and feedforward filter (FIR)."""
    if not (node.parameters.update_iir or node.parameters.update_fir):
        return

    qubit_pairs = node.namespace["qubit_pairs"]
    measured_qubits = node.namespace["measured_qubits"]

    with node.record_state_updates():
        for qp, measured_qubit in zip(qubit_pairs, measured_qubits):
            if node.outcomes[qp.name] == "failed":
                continue

            coupler_out = qp.coupler.opx_output
            if coupler_out.exponential_filter is None:
                coupler_out.exponential_filter = []

            if node.parameters.update_iir:
                fit = node.results["fit_results"][measured_qubit.name]
                components = fit["components"]
                a_dc = fit["a_dc"]
                A_list = [amp / a_dc for amp, _ in components]
                tau_list = [tau for _, tau in components]
                coupler_out.exponential_filter.extend(list(zip(A_list, tau_list)))
                print(f"Updated {qp.coupler.name} IIR filter: {coupler_out.exponential_filter}")

            if node.parameters.update_fir:
                fir_results = node.namespace.get("fir_results", {})
                res = fir_results.get(measured_qubit.name)
                if res is not None and res.get("success"):
                    coupler_out.feedforward_filter = res["inverse_fir"]
                    print(f"Updated {qp.coupler.name} FIR feedforward filter ({len(res['inverse_fir'])} taps)")
                else:
                    node.log(
                        f"  WARNING: FIR unavailable for {qp.coupler.name} "
                        f"(use_fir={node.parameters.use_fir}); skipping feedforward_filter update"
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()


# %%
