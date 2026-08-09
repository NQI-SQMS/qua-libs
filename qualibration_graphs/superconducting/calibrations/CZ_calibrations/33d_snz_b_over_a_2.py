# %% {Imports}
"""SNZ B/A ratio scan – Di Carlo Sudden Net-Zero experiment (Fig 2C).

This module implements a 2-D scan of the SNZ bipolar flux pulse over
the B/A ratio (shape of transition samples) and overall amplitude,
measuring control- and target-qubit populations to identify the optimal
net-zero operating point.
"""



from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz_v2 import resolve_cz_branch  # shared |20>/|02> branch resolver
from calibration_utils.snz_b_over_a_2 import (
    Parameters,
    decompose_t_phi_eff,
    fit_raw_data,
    log_fitted_results,
    plot_snz_raw,
    process_raw_dataset,
    snz_factory,
)
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam


# %% {Initialisation}
description = """
        SNZ t_phi_eff SCAN (Di Carlo Sudden Net-Zero)

This experiment scans the effective idle time (t_phi_eff) of an SNZ bipolar
flux pulse applied to the control qubit, together with the overall amplitude.

The effective idle time combines the integer idle samples (t_phi) with the B/A
transition-sample ratio via:

    t_phi_eff = t_phi + 2 * (1 - B/A)

The node accepts (t_phi_eff_min, t_phi_eff_max, t_phi_eff_step) and internally
decomposes each value into (t_phi, B/A) to bake the appropriate waveform:

    [padding | +A flat | +B | idle(t_phi) | -B | -A flat | padding]

The total flat duration (both halves) equals the calibrated CZ pulse length.
The overall amplitude is swept around 1.0 via QUA's *amp() operation.

For every (amplitude, t_phi_eff) point the sequence prepares |11⟩ and measures
the resulting populations of the |2>-excited qubit (g/e/f) and the other qubit (g/e).
The |2>-excited qubit is set by the cz_branch parameter (inherited from
qp.extras["cz_branch"]): branch "20" -> control is GEF-read (legacy), "02" -> target.

This is the analysis+update variant of 38: it additionally FITS the leakage
landscape (argmin of the control |f> population) and writes the result to the
cz_SNZ macro as a SEED. Because this node only measures leakage (no conditional
phase), the written point is a coarse starting estimate, NOT a pi-calibrated CZ;
run 39_2 / 39b afterwards to fix the conditional phase.

Prerequisites:
    - Calibrated single-qubit gates and readout for both qubits.
    - A calibrated CZ macro providing the nominal amplitude and pulse length.

State update (cz_SNZ macro only, written as a leakage SEED):
    - qp.macros["cz_SNZ"].flux_pulse_qubit.amplitude   (abs V at min-leakage)
    - qp.macros["cz_SNZ"].flux_pulse_qubit.t_phi_eff   (at min-leakage)
    - qp.macros["cz_SNZ"].flux_pulse_qubit.flat_length (the length used in the scan)
"""

node = QualibrationNode[Parameters, Quam](
    name="38_2_snz_b_over_a",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Local / debug overrides (ignored when run via GUI or graph)."""
    pass


# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Bake one SNZ waveform per t_phi_eff value and build the QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Resolve the |20>/|02> branch roles per pair (which qubit reaches |2> = GEF-read).
    # Both branches flux-move the CONTROL qubit, so the baked SNZ waveform on
    # qubit_control.z is branch-independent; only the readout roles swap.
    node.namespace["branch_specs"] = {
        qp.name: resolve_cz_branch(qp, None) for qp in qubit_pairs
    }

    operation = node.parameters.operation
    padding = node.parameters.padding
    n_avg = node.parameters.num_shots

    t_phi_eff_values = np.arange(
        node.parameters.t_phi_eff_min,
        node.parameters.t_phi_eff_max,
        node.parameters.t_phi_eff_step,
    )
    amplitudes = np.arange(
        1 - node.parameters.amp_range,
        1 + node.parameters.amp_range,
        node.parameters.amp_step,
    )

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "relative amplitude scale", "units": "a.u."}),
        "t_phi_eff": xr.DataArray(t_phi_eff_values, attrs={"long_name": "effective idle time", "units": "ns"}),
    }

    # ---- Bake one waveform per (qubit_pair, t_phi_eff) ----
    baked_config = node.machine.generate_config()
    baked_snz = {}

    for qp in qubit_pairs:
        A_nominal = qp.macros[operation].flux_pulse_qubit.amplitude
        if operation == "cz_SNZ":
            length = qp.macros[operation].flux_pulse_qubit.flat_length
        else:
            length = qp.macros[operation].flux_pulse_qubit.length
        baked_snz[qp.name] = []
        for j, tpe in enumerate(t_phi_eff_values):
            t_phi, ratio = decompose_t_phi_eff(tpe)
            wf = snz_factory(A_nominal, ratio, length, t_phi, padding)
            with baking(baked_config, padding_method="right") as b:
                b.add_op(f"snz_{qp.name}_{j}", qp.qubit_control.z.name, wf.tolist())
                b.play(f"snz_{qp.name}_{j}", qp.qubit_control.z.name)
            baked_snz[qp.name].append(b)

    node.namespace["baked_config"] = baked_config
    node.namespace["baked_snz"] = baked_snz

    num_tpe = len(t_phi_eff_values)

    # ---- QUA program ----
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        a = declare(fixed)
        idx = declare(int)
        n_st = declare_stream()

        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_cg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ce_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_cf_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(a, amplitudes)):
                    with for_(idx, 0, idx < num_tpe, idx + 1):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x180")
                            align()

                            with switch_(idx):
                                for j in range(num_tpe):
                                    with case_(j):
                                        baked_snz[qp.name][j].run(amp_array=[(qp.qubit_control.z.name, a)])

                            align()

                            # Branch roles: GEF-read the |2>-excited qubit (state_control streams),
                            # g-e read the other qubit (state_target stream). Branch "20" == legacy.
                            spec = node.namespace["branch_specs"][qp.name]
                            if node.parameters.use_state_discrimination:
                                spec.excited_qubit.readout_state_gef(state_c[ii])
                                spec.other_qubit.readout_state(state_t[ii])
                                with switch_(state_c[ii]):
                                    with case_(0):
                                        wait(4)
                                        save(1, state_cg_st[ii])
                                        save(0, state_ce_st[ii])
                                        save(0, state_cf_st[ii])
                                    with case_(1):
                                        wait(4)
                                        save(0, state_cg_st[ii])
                                        save(1, state_ce_st[ii])
                                        save(0, state_cf_st[ii])
                                    with default_():
                                        wait(4)
                                        save(0, state_cg_st[ii])
                                        save(0, state_ce_st[ii])
                                        save(1, state_cf_st[ii])
                                save(state_t[ii], state_t_st[ii])
                            else:
                                spec.excited_qubit.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                spec.other_qubit.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])

            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_cg_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"g_state_control{i}")
                    state_ce_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"e_state_control{i}")
                    state_cf_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"f_state_control{i}")
                    state_t_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"state_target{i}")
                else:
                    I_c_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(num_tpe).buffer(len(amplitudes)).average().save(f"Q_target{i}")


# %% {Simulate}
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


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program using the baked config and fetch raw data."""
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process the raw data and fit the leakage-minimum operating point."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fr["success"] else "failed")
        for qp_name, fr in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the leakage landscape and mark the selected min-leakage seed."""
    # optimal_amplitude from the fit is a RELATIVE scale; the plot x-axis is
    # amp_full (volts), so convert to absolute before placing the marker.
    operation = node.parameters.operation
    opt_points = {}
    for qp in node.namespace["qubit_pairs"]:
        fr = node.results.get("fit_results", {}).get(qp.name)
        if fr and fr.get("success"):
            stored_amp = qp.macros[operation].flux_pulse_qubit.amplitude
            opt_points[qp.name] = (fr["optimal_amplitude"] * stored_amp, fr["optimal_t_phi_eff"])
    fig = plot_snz_raw(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        opt_points=opt_points,
    )
    plt.show()
    node.results["figures"] = {"snz_b_over_a": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the leakage-minimum point to cz_SNZ as a SEED (amplitude + t_phi_eff + flat_length).

    This node measures leakage only (no conditional phase), so the written values
    are a COARSE starting estimate, not a pi-calibrated CZ -- run 39_2 / 39b after
    to fix the conditional phase. Pairs without a cz_SNZ macro or with a failed fit
    are skipped; an edge optimum is still written but flagged (widen the scan).
    """
    operation = node.parameters.operation
    fit_results = node.results["fit_results"]
    ds_fit = node.results["ds_fit"]
    amp_arr = ds_fit.amplitude.values
    tpe_arr = ds_fit.t_phi_eff.values
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit failed.")
                continue
            if "cz_SNZ" not in qp.macros:
                node.log(f"Skipping state update for {qp.name}: no 'cz_SNZ' macro on this pair.")
                continue
            fr = fit_results[qp.name]
            stored_amp = qp.macros[operation].flux_pulse_qubit.amplitude
            if operation == "cz_SNZ":
                scanned_length = qp.macros[operation].flux_pulse_qubit.flat_length
            else:
                scanned_length = qp.macros[operation].flux_pulse_qubit.length
            amp_abs = float(fr["optimal_amplitude"]) * float(stored_amp)
            tpe_opt = float(fr["optimal_t_phi_eff"])
            on_edge = fr["optimal_amplitude"] in (float(amp_arr[0]), float(amp_arr[-1])) or tpe_opt in (
                float(tpe_arr[0]),
                float(tpe_arr[-1]),
            )
            if on_edge:
                node.log(f"{qp.name}: seed optimum sits on a scan edge -- consider widening the range.")
            snz_pulse = qp.macros["cz_SNZ"].flux_pulse_qubit
            snz_pulse.amplitude = amp_abs
            snz_pulse.t_phi_eff = tpe_opt
            snz_pulse.flat_length = int(scanned_length)
            node.log(
                f"{qp.name}: cz_SNZ SEED written -- amplitude={amp_abs:.6f} V, "
                f"t_phi_eff={tpe_opt:.4f} ns, flat_length={int(scanned_length)} "
                f"(min leakage={fr['min_leakage']:.4f})."
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
