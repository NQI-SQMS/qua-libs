"""Pi vs coupler flux calibration for long flux distortion — ramsey_vs_flux dataload variant."""


# %%
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.coupler_flux_long_distortion_qubitspec import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_fit,
    process_raw_dataset,
    _derive_coupler_flux_from_decouple,
    _load_coupler_spectroscopy_curve,
    _load_ramseyflux_curve_from_param,
    plot_spectroscopy_curve as _plot_coupler_spectroscopy_curve,
    plot_ramsey_curve as _plot_coupler_ramsey_curve,
)
from calibration_utils.qubit_flux_long_distortion_qubitspec import (
    plot_center_freqs,
    plot_flux_response,
    plot_iq_abs_heatmap,
    plot_phase_heatmap,
    _to_python,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %%
description = """
Long cryoscope (π vs coupler flux) calibration — ramsey_vs_flux dataload variant.

Same experiment as 21_coupler_flux_long_distortion, but the coupler flux amplitude
and the freq->flux conversion in analysis are both derived from a previously acquired
qubit-ramsey_vs_flux-vs-coupler-flux run (node 10), specified via `ramsey_vs_flux_run_id`, or read per pair from `qubit.extras[f"{coupler}_dispersion_load_id"]` when the parameter is None.

Workflow:
For each qubit pair, load the Ramsey-vs-coupler-flux dispersion curve from
`ramsey_vs_flux_run_id`, look up the qubit frequency at the coupler's
`decouple_offset`, and compute the absolute coupler flux that achieves the
**signed** `detuning_in_mhz` from that reference frequency.  Then sweep
XY-drive detuning vs coupler flux-pulse duration to characterise the coupler
flux-line step response.
Analysis: fit a sum of decaying exponentials; optionally write cascade coefficients
to coupler.opx_output.exponential_filter.

Prerequisites
- A valid rotation angle and threshold if using state discrimination
- Calibrated XY-Coupler delay
- A calibrated pi-pulse
- A completed node-10 (qubit ramsey_vs_flux vs coupler flux) run whose ID is provided
  as ramsey_vs_flux_run_id.

Outputs and state updates
- Results: processed dataset, fit results, and figures are saved under `node.results`.
- If `update_state=True` and fits succeed, updates `state.json` for the coupler at
  `coupler.opx_output.exponential_filter` with cascade coefficients `(A_c, tau_c)`.
REMINDER: Adding digital filters will add a global delay --> need to recalibrate IQ
blobs (rotation_angle & ge_threshold) and XY-Coupler delay.
"""

node = QualibrationNode[Parameters, Quam](
    name="17b_coupler_flux_long_distortion_qubitspec",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)



# %% {Custom_param}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""


# Instantiate machine
stored_machine = Quam.load()

# store fitting parameter and GUI flag set from GUI
loaded_n_exponentials = node.parameters.n_exponentials
stored_gui_update_flag = node.parameters.update_state_from_GUI


# %% {Create_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program for pi vs coupler flux measurement."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the measured qubits based on measure_qubit parameter
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    operation_name = node.parameters.operation
    for qubit in measured_qubits:
        if hasattr(qubit.xy.operations, operation_name):
            continue
        warnings.warn(f"Qubit {qubit.name} has no operation '{operation_name}', defaulting to 'x180'")
        operation_name = "x180"

    operation_amp_scale = node.parameters.operation_amplitude_factor or 1.0

    # Signed detuning from the qubit freq at the coupler's decouple_offset
    detuning_hz = node.parameters.detuning_in_mhz * 1e6  # signed
    span_hz = node.parameters.frequency_span_in_mhz * 1e6
    step_hz = node.parameters.frequency_step_in_mhz * 1e6

    use_spec = node.parameters.use_spectroscopy_data
    spec_run_id = node.parameters.spectroscopy_run_id
    use_ramsey = node.parameters.use_ramsey_vs_flux_data

    # --- Per-pair coupler flux derivation from decouple_offset ---
    coupler_flux_amps = []
    per_pair_info = []  # for sanity-check printout
    for qp in qubit_pairs:
        qubit = qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
        decouple_offset = qp.coupler.decouple_offset
        coupler_flux_q = None
        freq_at_decouple = None
        source_label = None
        curve_for_info = None

        # --- Try spectroscopy curve ---
        if use_spec and spec_run_id is not None:
            spec_curve = _load_coupler_spectroscopy_curve(spec_run_id, qubit, qp.coupler, node)
            if spec_curve is not None:
                result = _derive_coupler_flux_from_decouple(detuning_hz, decouple_offset, spec_curve)
                if result is not None:
                    coupler_flux_q, freq_at_decouple = result
                    source_label = f"spectroscopy #{spec_run_id}"
                    curve_for_info = spec_curve

        # --- Try Ramsey curve ---
        ramsey_run_id_q = node.parameters.ramsey_vs_flux_run_id or (
            qubit.extras.get(f"{qp.coupler.name}_dispersion_load_id")
            if hasattr(qubit, "extras")
            else None
        )
        if coupler_flux_q is None and use_ramsey and ramsey_run_id_q is not None:
            ramsey_curve = _load_ramseyflux_curve_from_param(ramsey_run_id_q, qubit, qp.coupler, node)
            if ramsey_curve is not None:
                result = _derive_coupler_flux_from_decouple(detuning_hz, decouple_offset, ramsey_curve)
                if result is not None:
                    coupler_flux_q, freq_at_decouple = result
                    source_label = f"Ramsey #{ramsey_run_id_q}"
                    curve_for_info = ramsey_curve

        # --- coupler_flux_amplitude fallback (user input) ---
        if coupler_flux_q is None:
            cfa = node.parameters.coupler_flux_amplitude_in_v
            if cfa is not None and cfa != 0:
                coupler_flux_q = cfa
                source_label = f"coupler_flux_amplitude={cfa:.4f} V (user input)"

        if coupler_flux_q is None:
            raise ValueError(
                f"Cannot derive coupler flux for {qp.name} / {qp.coupler.name}. "
                f"detuning={node.parameters.detuning_in_mhz:+.2f} MHz, "
                f"decouple_offset={decouple_offset:.4f} V. "
                f"Likely cause: target frequency is outside the loaded dispersion "
                f"curve range (check warnings above), or no curve data was loaded. "
                f"Ensure spectroscopy_run_id / ramsey_vs_flux_run_id / coupler dispersion_load_id extras is set and "
                f"the curve covers the requested detuning."
            )

        if abs(coupler_flux_q) > 0.5:
            warnings.warn(
                f"{qp.name}: derived coupler_flux={coupler_flux_q:.4f} V exceeds 0.5 V. "
                f"Verify detuning_in_mhz={node.parameters.detuning_in_mhz} is correct."
            )

        coupler_flux_amps.append(coupler_flux_q)
        per_pair_info.append({
            "qp": qp, "qubit": qubit,
            "decouple_offset": decouple_offset,
            "freq_at_decouple": freq_at_decouple,
            "coupler_flux": coupler_flux_q,
            "source": source_label,
            "curve": curve_for_info,
        })

    # Store coupler flux center and decouple offsets in namespace for analysis
    node.namespace["coupler_flux_center"] = coupler_flux_amps[0]
    node.namespace["coupler_flux_amps"] = coupler_flux_amps
    decouple_offsets = [info["decouple_offset"] for info in per_pair_info]
    node.namespace["decouple_offsets"] = decouple_offsets
    # Persist for load_data path (namespace is ephemeral, results survive save/load)
    node.results["coupler_flux_center"] = coupler_flux_amps[0]
    node.results["decouple_offsets"] = decouple_offsets

    # ------------------------------------------------------------------
    # Build dfs array: signed, centered at detuning relative to idle.
    # If freq_at_decouple differs from RF_frequency (idle), dfs must
    # account for that offset so the XY sweep is centered correctly.
    # We use the first pair's freq_at_decouple as the reference
    # (per-pair LO tracking below handles per-qubit adjustments).
    # ------------------------------------------------------------------
    ref_info = per_pair_info[0]
    ref_qubit = ref_info["qubit"]
    if ref_info["freq_at_decouple"] is not None:
        idle_to_decouple_offset_hz = ref_info["freq_at_decouple"] - ref_qubit.xy.RF_frequency
    else:
        idle_to_decouple_offset_hz = 0.0

    dfs_center_hz = detuning_hz + idle_to_decouple_offset_hz
    dfs = np.arange(
        dfs_center_hz - span_hz / 2,
        dfs_center_hz + span_hz / 2 + step_hz / 2,
        step_hz,
        dtype=np.int32,
    )

    # ------------------------------------------------------------------
    # Sanity-check printout: verify sweep intent before hardware execution
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  SANITY CHECK — Coupler flux distortion sweep summary")
    print("=" * 65)
    for i, info in enumerate(per_pair_info):
        qp = info["qp"]
        qubit = info["qubit"]
        dec_off = info["decouple_offset"]
        f_dec = info["freq_at_decouple"]
        target_freq = f_dec + detuning_hz if f_dec is not None else None
        curve = info["curve"]
        curve_fmin = float(np.nanmin(curve[1])) if curve is not None else None
        curve_fmax = float(np.nanmax(curve[1])) if curve is not None else None
        within = (
            (curve_fmin <= target_freq <= curve_fmax)
            if (target_freq is not None and curve_fmin is not None)
            else "N/A"
        )
        print(f"  Pair: {qp.name}  |  Coupler: {qp.coupler.name}")
        print(f"    decouple_offset (from state):   {dec_off:.4f} V")
        if f_dec is not None:
            print(f"    qubit freq @ decouple_offset:   {f_dec / 1e9:.6f} GHz")
        else:
            print(f"    qubit freq @ decouple_offset:   N/A (fallback path)")
        print(f"    qubit RF_frequency (idle):      {qubit.xy.RF_frequency / 1e9:.6f} GHz")
        if f_dec is not None:
            print(f"    drift (decouple - idle):        {(f_dec - qubit.xy.RF_frequency) / 1e6:+.3f} MHz")
        print(f"    signed detuning requested:      {node.parameters.detuning_in_mhz:+.2f} MHz")
        if target_freq is not None:
            print(f"    target qubit freq:              {target_freq / 1e9:.6f} GHz")
        print(f"    derived coupler_flux:           {info['coupler_flux']:.6f} V  (source: {info['source']})")
        print(f"    XY sweep dfs range:             [{dfs[0] / 1e6:.2f}, {dfs[-1] / 1e6:.2f}] MHz  (rel. to idle)")
        print(f"    XY sweep abs freq range:        "
              f"[{(dfs[0] + qubit.xy.RF_frequency) / 1e9:.6f}, "
              f"{(dfs[-1] + qubit.xy.RF_frequency) / 1e9:.6f}] GHz")
        if curve_fmin is not None:
            print(f"    curve freq range available:     [{curve_fmin / 1e9:.6f}, {curve_fmax / 1e9:.6f}] GHz")
            print(f"    target freq within curve?:      {within}")
        print()
    print("=" * 65)

    # Time sweep — linear or log scale
    if node.parameters.time_axis == "linear":
        times = np.arange(
            node.parameters.min_wait_time_in_ns // 4,
            node.parameters.duration_in_ns // 4,
            max(node.parameters.time_step_in_ns, 4) // 4,
            dtype=np.int32,
        )
    else:
        times = np.logspace(
            np.log10(max(node.parameters.min_wait_time_in_ns // 4, 1)),
            np.log10(max(node.parameters.duration_in_ns // 4, 2)),
            max(node.parameters.time_step_num, 3),
            dtype=np.int32,
        )
        times = np.unique(times)

    # ------------------------------------------------------------------
    # LO tracking: shift upconverter if dfs pushes IF outside [-400, +400] MHz
    # ------------------------------------------------------------------
    tracked_qubits = []
    if_update = []
    for i, (qubit, qp) in enumerate(zip(measured_qubits, qubit_pairs)):
        dfs_mid = int((dfs.min() + dfs.max()) / 2)
        if_lo = qubit.xy.intermediate_frequency
        if (if_lo + int(dfs.min())) < -400e6 or (if_lo + int(dfs.max())) > 400e6:
            node.parameters.reset_type = "thermal"
            warnings.warn(
                "Qubit LO has been changed to reach desired detuning, "
                "active reset will not work. Reset type changed to thermal."
            )
            if_update.append(dfs_mid)
            with tracked_updates(qubit, auto_revert=False, dont_assign_to_none=False) as q_upd:
                lo_frequency = q_upd.xy.opx_output.upconverter_frequency + dfs_mid
                if (q_upd.xy.opx_output.band == 3) and (lo_frequency < 6.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                if (q_upd.xy.opx_output.band == 2) and (lo_frequency < 4.5e9):
                    raise ValueError("Requested detuning is too large for the given MW FEM band")
                print(f"Updating {q_upd.name} LO to {lo_frequency}")
                q_upd.xy.opx_output.upconverter_frequency = lo_frequency
                q_upd.xy.RF_frequency += dfs_mid
                tracked_qubits.append(q_upd)
        else:
            if_update.append(0)
            print(f"No LO update needed for {qubit.name}")

    node.namespace["if_update"] = if_update
    node.namespace["tracked_qubits"] = tracked_qubits

    # buffer times
    buf_during_op = node.parameters.buffer_during_operation_in_ns // 4
    buf_after_op = node.parameters.buffer_after_operation_in_ns // 4

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(4 * times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    with program() as qua_prog:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]

        df = declare(int)
        t_delay = declare(int)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_each_(t_delay, times):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist_qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        # Extra wait to ensure long distortions have fully decayed between repetitions
                        wait(times.max())
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            protagonist_qubit.xy.update_frequency(df + protagonist_qubit.xy.intermediate_frequency - if_update[ii])
                            align()
                            qp.coupler.play(
                                "const",
                                amplitude_scale=coupler_flux_amps[ii] / qp.coupler.operations["const"].amplitude,
                                duration=t_delay + buf_during_op,
                            )
                            protagonist_qubit.xy.wait(t_delay)
                            protagonist_qubit.xy.play(operation_name, amplitude_scale=operation_amp_scale)
                            protagonist_qubit.wait(buf_after_op)
                            align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            protagonist_qubit = (
                                qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                            )
                            if node.parameters.use_state_discrimination:
                                protagonist_qubit.readout_state(state[ii])
                                save(state[ii], state_st[ii])
                            else:
                                protagonist_qubit.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                                save(I[ii], I_st[ii])
                                save(Q[ii], Q_st[ii])
                        align()

            if not node.parameters.multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")

    node.namespace["qua_program"] = qua_prog


# %% {Simulate_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(node.namespace["qua_program"], config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}
    plt.show()


# %% {Execute_qua_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        node.log(job.execution_report())
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
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

    if "qubit_pair" in node.results["ds_raw"].dims:
        qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_names)

    # Restore coupler_flux_center and decouple_offsets to namespace for analysis
    cfc = node.results.get("coupler_flux_center")
    if cfc is not None:
        node.namespace["coupler_flux_center"] = cfc
    dec_offs = node.results.get("decouple_offsets")
    if dec_offs is not None:
        node.namespace["decouple_offsets"] = dec_offs
    else:
        node.namespace["decouple_offsets"] = [qp.coupler.decouple_offset for qp in qubit_pairs]

    # Overwrite the loaded node parameters with the ones defined from the GUI
    node.parameters.n_exponentials = loaded_n_exponentials
    node.parameters.update_state_from_GUI = stored_gui_update_flag
    if node.parameters.update_state_from_GUI:
        node.machine = stored_machine
        node.parameters.update_state = True
        print("State update from GUI is enabled")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw dataset and fit exponential components to the flux response data."""
    ds_in = process_raw_dataset(node.results["ds_raw"], node)
    ds, fit_results = fit_raw_data(ds_in, node)

    node.results["ds_fit"] = ds
    node.results["fit_results"] = {k: _to_python(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot raw IQ, center freq, flux response, exponential fits, and dispersion reference curve(s)."""
    if "ds_fit" not in node.results:
        return
    ds = node.results["ds_fit"]
    qubits = node.namespace.get("qubits") or node.namespace.get("measured_qubits", [])
    fit_results = node.results["fit_results"]

    figures = {
        "iq_abs_linear": plot_iq_abs_heatmap(ds, qubits, log_scale=False),
        "iq_abs_log": plot_iq_abs_heatmap(ds, qubits, log_scale=True),
        "phase": plot_phase_heatmap(ds, qubits),
        "center_freq_linear": plot_center_freqs(ds, qubits, log_scale=False),
        "center_freq_log": plot_center_freqs(ds, qubits, log_scale=True),
        "flux_response_linear": plot_flux_response(ds, qubits, log_scale=False),
        "flux_response_log": plot_flux_response(ds, qubits, log_scale=True),
        "fitted_data": plot_fit(ds, qubits, fit_results),
    }
    qubit_pairs_for_plot = node.namespace.get("qubit_pairs", [])

    # Plot qubit spectroscopy vs coupler flux reference curve (path 1)
    if node.parameters.use_spectroscopy_data:
        spec_fig = _plot_coupler_spectroscopy_curve(
            node.parameters.spectroscopy_run_id, qubit_pairs_for_plot, qubits, node
        )
        if spec_fig is not None:
            figures["spectroscopy_curve"] = spec_fig

    # Plot Ramsey-vs-coupler-flux reference curve (path 2)
    if node.parameters.use_ramsey_vs_flux_data:
        ramsey_fig = _plot_coupler_ramsey_curve(
            node.parameters.ramsey_vs_flux_run_id, qubit_pairs_for_plot, qubits, node
        )
        if ramsey_fig is not None:
            figures["ramsey_curve"] = ramsey_fig

    node.results["figures"] = figures
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update IIR filter tabs on coupler if fitting was successful."""
    if not node.parameters.update_state:
        return

    qubit_pairs = node.namespace["qubit_pairs"]
    measured_qubits = node.namespace["measured_qubits"]

    for qp in qubit_pairs:
        if qp.coupler.opx_output.exponential_filter is None:
            qp.coupler.opx_output.exponential_filter = []

    with node.record_state_updates():
        for qp, measured_qubit in zip(qubit_pairs, measured_qubits):
            res = node.results["fit_results"].get(measured_qubit.name)
            if res is None:
                continue
            fit_success = res["fit_successful"]
            if not fit_success:
                continue

            best_a_dc = res["a_dc"]
            components = res["a_tau_tuple"]
            A_list = [amp / best_a_dc for amp, _ in components]
            tau_list = [tau for _, tau in components]
            qp.coupler.opx_output.exponential_filter.extend(list(zip(A_list, tau_list)))
            print(f"Updated {qp.coupler.name} filter to: {qp.coupler.opx_output.exponential_filter}")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results, revert tracked qubit changes, and persist state."""
    for qubit in node.namespace.get("tracked_qubits", []):
        try:
            qubit.revert_changes()
        except Exception:
            pass
    node.save()


# %%
