"""Cryoscope calibration for flux line step response — spectroscopy + Ramsey amplitude paths."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.qubit_flux_short_distortion import (
    Parameters,
    baked_waveform,
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
from calibration_utils.qubit_flux_long_distortion_qubitspec import (
    _flux_amp_from_curve,
    _load_ramsey_curve,
    _load_spectroscopy_curve,
)
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Node_parameters}
description = """
CRYOSCOPE (20e — spectroscopy + Ramsey amplitude paths, with optional FIR)

Same pulse sequence as 20d: Ramsey-style cryoscope sweeping flux pulse duration
at a fixed amplitude, with frame rotation for phase reconstruction.

Extends 20d with a three-path cascade for computing the cryoscope flux amplitude:

  Path 1 — Qubit spectroscopy vs Z-flux (use_spectroscopy_data=True, spectroscopy_run_id set):
      Loads the freq-vs-flux curve from a previous qubit spectroscopy run and inverts
      it to find the flux amplitude that achieves the target detuning.  Most accurate
      when the dispersion curve has been measured over the relevant flux range.

  Path 2 — Ramsey vs Z-flux (use_ramsey_data=True, state extras key present):
      Loads the freq-vs-flux curve from the per-qubit Ramsey calibration run ID stored
      in state at `qubit.extras["ramsey_vs_flux_calibration_load_id"]` (written by node 23),
      then performs the same inversion. Used as a fallback when spectroscopy data is
      unavailable or when Ramsey data covers the flux range better.

  Path 3 — Quadratic model fallback:
      Uses qubit.freq_vs_flux_01_quad_term to analytically compute the amplitude.
      Always available; least accurate for large detunings.

Paths are attempted in order; the first successful result is used per qubit.

Post-processing, IIR fitting, and optional FIR analysis are identical to 20d.

Prerequisites:
        - Resonator spectroscopy performed.
        - Qubit gates (x90, y90) calibrated.
        - For Path 1: a completed qubit spectroscopy vs Z-flux run (e.g. node 09).
        - For Path 2: a completed Ramsey vs Z-flux run (e.g. node 23), with
          `ramsey_vs_flux_calibration_load_id` stored in qubit extras in state.

Next steps:
        - Push IIR (and optionally FIR) filter taps into the configuration.
        - WARNING: digital filters add a global delay — recalibrate IQ blobs.
"""

node = QualibrationNode[Parameters, Quam](
    name="20_qubit_flux_short_distortion",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)



@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    pass


# Instantiate the QUAM class from the state file
stored_machine = Quam.load()

loaded_n_exponentials = node.parameters.n_exponentials
stored_use_fir = node.parameters.use_fir
stored_update_iir = node.parameters.update_iir
stored_update_fir = node.parameters.update_fir


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    cryoscope_len = node.parameters.cryoscope_len

    # ------------------------------------------------------------------
    # Amplitude resolution — three-path cascade per qubit
    # ------------------------------------------------------------------
    use_spec = node.parameters.use_spectroscopy_data
    spec_run_id = node.parameters.spectroscopy_run_id
    use_ramsey = node.parameters.use_ramsey_data
    ramsey_run_id_override = node.parameters.ramsey_run_id

    amplitudes = []
    for q in qubits:
        amp = None
        ramsey_run_id_q = (
            ramsey_run_id_override
            or (q.extras.get("ramsey_vs_flux_calibration_load_id") if hasattr(q, "extras") else None)
        )

        # Path 1: qubit spectroscopy vs Z-flux
        if use_spec and spec_run_id is not None:
            try:
                curve = _load_spectroscopy_curve(spec_run_id, q.name, q.xy.RF_frequency)
                if curve is not None:
                    amp = _flux_amp_from_curve(
                        node.parameters.detuning_target_in_mhz * 1e6,
                        q.xy.RF_frequency,
                        curve[0],
                        curve[1],
                    )
                    if amp is not None:
                        print(f"  {q.name}: flux_amp={amp:.6f} V (from spec run #{spec_run_id})")
                    else:
                        print(f"  {q.name}: detuning outside spec range, trying next path")
                else:
                    print(f"  {q.name}: spec curve not found in run #{spec_run_id}, trying next path")
            except Exception as exc:
                print(f"  {q.name}: spec load failed ({exc}), trying next path")

        # Path 2: Ramsey vs Z-flux
        if amp is None and use_ramsey and ramsey_run_id_q is not None:
            try:
                curve = _load_ramsey_curve(ramsey_run_id_q, q.name, q.xy.RF_frequency)
                if curve is not None:
                    amp = _flux_amp_from_curve(
                        node.parameters.detuning_target_in_mhz * 1e6,
                        q.xy.RF_frequency,
                        curve[0],
                        curve[1],
                    )
                    if amp is not None:
                        print(f"  {q.name}: flux_amp={amp:.6f} V (from Ramsey run #{ramsey_run_id_q})")
                    else:
                        print(f"  {q.name}: detuning outside Ramsey range, falling back to quad_term")
                else:
                    print(f"  {q.name}: Ramsey curve not found in run #{ramsey_run_id_q}, falling back to quad_term")
            except Exception as exc:
                print(f"  {q.name}: Ramsey load failed ({exc}), falling back to quad_term")

        # Path 3: quadratic model fallback
        if amp is None:
            amp = float(np.sqrt(-node.parameters.detuning_target_in_mhz * 1e6 / q.freq_vs_flux_01_quad_term))
            print(f"  {q.name}: flux_amp={amp:.6f} V (from quad_term)")

        amplitudes.append(amp)

    cryoscope_time = np.arange(1, cryoscope_len + 1, 1)
    frames = np.linspace(0, 1, node.parameters.num_frames)

    baked_config = node.machine.generate_config()
    baked_signals = {
        q.name: baked_waveform(baked_config, amplitudes[i], q, max_length=16) for i, q in enumerate(qubits)
    }

    node.namespace["baked_config"] = baked_config
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "time": xr.DataArray(cryoscope_time, attrs={"long_name": "Cryoscope pulse duration", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "Frame rotation index"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        t_left_ns = declare(int)
        t_cycles = declare(int)
        idx = declare(int)
        frame = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(idx, 1, idx <= cryoscope_len, idx + 1):
                    with for_each_(frame, frames):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
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
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 16) >> 2)
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")
                        with else_():
                            assign(t_cycles, idx >> 2)
                            assign(t_left_ns, idx - (t_cycles << 2))
                            with switch_(t_left_ns):
                                with case_(0):
                                    align()
                                    for i, qubit in multiplexed_qubits.items():
                                        qubit.xy.play("x90")
                                        qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                        qubit.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=amplitudes[i] / qubit.z.operations["const"].amplitude,
                                        )
                                        qubit.xy.wait((cryoscope_len + 16) // 4)
                                        qubit.xy.frame_rotation_2pi(frame)
                                        qubit.xy.play("x90")
                                for j in range(1, 4):
                                    with case_(j):
                                        align()
                                        for i, qubit in multiplexed_qubits.items():
                                            qubit.xy.play("x90")
                                            qubit.z.wait((qubit.xy.operations["x90"].length + 16) // 4)
                                            qubit.z.play(
                                                "const",
                                                duration=t_cycles,
                                                amplitude_scale=amplitudes[i] / qubit.z.operations["const"].amplitude,
                                            )
                                            baked_signals[qubit.name][j - 1].run()
                                            qubit.xy.wait((cryoscope_len + 16) // 4)
                                            qubit.xy.frame_rotation_2pi(frame)
                                            qubit.xy.play("x90")

                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
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
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    node.parameters.n_exponentials = loaded_n_exponentials
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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    log_fitted_results(fit_results, log_callable=node.log)

    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    node.outcomes = {
        qubit_name: ("successful" if fit_result.success else "failed") for qubit_name, fit_result in fit_results.items()
    }

    # --- FIR analysis (optional) ---
    if node.parameters.use_fir:
        fir_results = fit_fir_data(node.results["ds_fit"], node)
        node.namespace["fir_results"] = fir_results
        node.results["fir_results"] = {
            qn: {k: v for k, v in res.items() if not str(k).startswith("fig")} for qn, res in fir_results.items()
        }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot all analysis stages and dispersion reference curves.

    Figure inventory (always generated):
      raw_<qname>              — raw state/I vs frame: line slices + 2D heatmap.
      unwrapped_phase          — unwrapped phase vs time for all qubits.
      cryoscope_freq_linear    — cryoscope frequency vs time (linear x-axis).
      cryoscope_freq_log       — cryoscope frequency vs time (log x-axis).
      flux_response_linear     — flux step response vs time (linear x-axis).
      flux_response_log        — flux step response vs time (log x-axis).
      iir_fitted_data          — IIR exponential fit overlay (linear + log).

    Reference curve figures (conditional on parameter flags and available run IDs):
      spectroscopy_curve       — qubit freq vs Z-flux from spectroscopy run.
      ramsey_curve             — qubit freq vs Z-flux from Ramsey run.

    Additional figures when use_fir=True and FIR analysis succeeded:
      fir_resampled_<qname>          — 1 GS/s vs 2 GS/s normalised flux.
      fir_fit_diagnostic_<qname>     — forward FIR fit 2×2 diagnostic.
      fir_inverse_diagnostic_<qname> — inverse FIR 3×2 diagnostic.
      fir_corrected_<qname>          — corrected response validation at 1 GS/s.
      fir_stem_<qname>               — FIR coefficient stem plots.
    """
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

    # --- Reference curve: qubit spectroscopy vs Z-flux (Path 1) ---
    if node.parameters.use_spectroscopy_data and node.parameters.spectroscopy_run_id is not None:
        fig_spec, axes = plt.subplots(1, n_q, figsize=(5 * n_q, 4), squeeze=False)
        for ax, q in zip(axes[0], qubits):
            curve = _load_spectroscopy_curve(node.parameters.spectroscopy_run_id, q.name, q.xy.RF_frequency)
            if curve is not None:
                ax.plot(curve[0], np.array(curve[1]) / 1e9, marker=".", linestyle="-")
            ax.set_xlabel("Z flux (V)")
            ax.set_ylabel("Qubit frequency (GHz)")
            ax.set_title(q.name)
            ax.grid(True)
        fig_spec.suptitle(f"Qubit spectroscopy vs Z-flux — run #{node.parameters.spectroscopy_run_id}")
        fig_spec.tight_layout()
        figures["spectroscopy_curve"] = fig_spec

    # --- Reference curve: Ramsey vs Z-flux (Path 2) ---
    if node.parameters.use_ramsey_data:
        fig_ram, axes = plt.subplots(1, n_q, figsize=(5 * n_q, 4), squeeze=False)
        used_run_ids = set()
        _ramsey_id_ov = node.parameters.ramsey_run_id
        for ax, q in zip(axes[0], qubits):
            ramsey_run_id_q = (
                _ramsey_id_ov
                or (q.extras.get("ramsey_vs_flux_calibration_load_id") if hasattr(q, "extras") else None)
            )
            if ramsey_run_id_q is None:
                continue
            used_run_ids.add(int(ramsey_run_id_q))
            curve = _load_ramsey_curve(ramsey_run_id_q, q.name, q.xy.RF_frequency)
            if curve is not None:
                ax.plot(curve[0], np.array(curve[1]) / 1e9, marker=".", linestyle="-")
            ax.set_xlabel("Z flux (V)")
            ax.set_ylabel("Qubit frequency (GHz)")
            ax.set_title(q.name)
            ax.grid(True)
        if used_run_ids:
            runs_txt = ", ".join(str(rid) for rid in sorted(used_run_ids))
            fig_ram.suptitle(f"Ramsey vs Z-flux — run(s) from state: {runs_txt}")
        else:
            fig_ram.suptitle("Ramsey vs Z-flux — no run IDs found in qubit extras")
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
    """Push IIR and/or FIR filters into state when the corresponding flag is set."""
    if not (node.parameters.update_iir or node.parameters.update_fir):
        return
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            # --- IIR exponential filter ---
            if node.parameters.update_iir:
                components = node.results["fit_results"][q.name]["components"]
                a_dc = node.results["fit_results"][q.name]["a_dc"]
                A_list = [amp / a_dc for amp, _ in components]
                tau_list = [tau for _, tau in components]
                node.machine.qubits[q.name].z.opx_output.exponential_filter.extend(
                    list(zip(A_list, tau_list))
                )

            # --- FIR feedforward filter ---
            if node.parameters.update_fir:
                fir_results = node.namespace.get("fir_results", {})
                res = fir_results.get(q.name)
                if res is not None and res.get("success"):
                    node.machine.qubits[q.name].z.opx_output.feedforward_filter = res["inverse_fir"]
                else:
                    node.log(
                        f"  WARNING: FIR unavailable for {q.name} (use_fir={node.parameters.use_fir}); "
                        f"skipping feedforward_filter update"
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()
