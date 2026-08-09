# %% {Imports}
"""SNZ conditional phase measurement.

This module combines the SNZ t_phi_eff baking approach with a conditional
phase Ramsey measurement (x90 - SNZ - frame_rotation - x90) on the target
qubit, with the control qubit prepared in either |g> or |e>.

The result is a 4-D dataset:
    (qubit_pair, amplitude, t_phi_eff, frame, control_axis)

from which the conditional phase difference is extracted by fitting the
oscillation along the frame dimension.
"""


from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz_v2 import resolve_cz_branch  # shared |20>/|02> branch resolver
from calibration_utils.snz_b_over_a import decompose_t_phi_eff, snz_factory
from calibration_utils.snz_conditional_phase import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_snz_conditional_phase,
    process_raw_dataset,
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
        SNZ CONDITIONAL PHASE MEASUREMENT

This experiment combines the SNZ t_phi_eff scan with a conditional phase
Ramsey measurement on the target qubit.  For every (amplitude, t_phi_eff)
point the sequence is:

1. Reset both qubits and frames.
2. Conditionally prepare the control qubit in |e> (x180 if control_initial=1).
3. Prepare the target qubit in a superposition (x90).
4. Apply the baked SNZ pulse (selected by t_phi_eff index, scaled by amplitude).
5. Apply a virtual frame rotation on the target qubit (phase tomography).
6. Close the Ramsey sequence (x90 on target).
7. Measure: g/e/f on control, g/e on target.

The oscillation of the target signal vs frame rotation is fitted to extract
the phase for each control state.  The conditional phase difference
(phase[ctrl=0] - phase[ctrl=1]) mod 1 is the key observable (0.5 = pi).

NOTE: "control" above means the |2>-excited / conditioning qubit set by the cz_branch
parameter (inherited from qp.extras["cz_branch"]). Branch "20" -> excited=control,
Ramsey on target (legacy); branch "02" -> excited=target, Ramsey on control. The
conditioning x180 + GEF readout always go on the excited qubit, the Ramsey + g-e
readout on the other; control_axis / control_initial track the excited qubit's state.

Prerequisites:
    - Calibrated single-qubit gates and readout for both qubits.
    - A calibrated CZ macro providing the nominal amplitude and pulse length.

State update:
    - None (exploratory scan).
"""

node = QualibrationNode[Parameters, Quam](
    name="39_snz_conditional_phase",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Local / debug overrides (ignored when run via GUI or graph)."""
    # node.parameters.load_data_id = 11260
    pass


# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Bake SNZ waveforms and build the 4-D QUA program."""

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Resolve the |20>/|02> branch roles per pair: the conditioning (x180) + GEF readout go on the
    # |2>-excited qubit, the Ramsey (x90/frame/x90) + g-e readout on the other. Both branches
    # flux-move the CONTROL qubit, so the baked SNZ waveform on qubit_control.z is unchanged.
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
    frames = np.arange(0, 1, 1 / node.parameters.num_frame_rotations)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "relative amplitude scale", "units": "a.u."}),
        "t_phi_eff": xr.DataArray(t_phi_eff_values, attrs={"long_name": "effective idle time", "units": "ns"}),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2\u03c0"}),
        "control_axis": xr.DataArray([0, 1], attrs={"long_name": "control qubit state"}),
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
        frame = declare(fixed)
        control_initial = declare(int)
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
                        with for_(*from_array(frame, frames)):
                            with for_(*from_array(control_initial, [0, 1])):
                                for ii, qp in multiplexed_qubit_pairs.items():
                                    qp.qubit_control.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                    )
                                    qp.qubit_target.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                    )
                                    qp.align()

                                    reset_frame(qp.qubit_target.xy.name)
                                    reset_frame(qp.qubit_control.xy.name)

                                    # Branch roles: conditioning + GEF on the |2>-excited qubit,
                                    # Ramsey + g-e on the other. Branch "20" == legacy (excited=control).
                                    spec = node.namespace["branch_specs"][qp.name]
                                    spec.excited_qubit.xy.play("x180", condition=control_initial == 1)
                                    spec.other_qubit.xy.play("x90")
                                    qp.align()

                                    with switch_(idx):
                                        for j in range(num_tpe):
                                            with case_(j):
                                                baked_snz[qp.name][j].run(amp_array=[(qp.qubit_control.z.name, a)])
                                    align()

                                    spec.other_qubit.xy.frame_rotation_2pi(frame)
                                    spec.other_qubit.xy.play("x90")
                                    qp.align()

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
                    (
                        state_cg_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"g_state_control{i}")
                    )
                    (
                        state_ce_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"e_state_control{i}")
                    )
                    (
                        state_cf_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"f_state_control{i}")
                    )
                    (
                        state_t_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"state_target{i}")
                    )
                else:
                    (
                        I_c_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"I_control{i}")
                    )
                    (
                        Q_c_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"Q_control{i}")
                    )
                    (
                        I_t_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"I_target{i}")
                    )
                    (
                        Q_t_st[i]
                        .buffer(2)
                        .buffer(len(frames))
                        .buffer(num_tpe)
                        .buffer(len(amplitudes))
                        .average()
                        .save(f"Q_target{i}")
                    )


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
    """Process raw data, fit oscillations, extract conditional phase."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(fit_results, log_callable=node.log)

    node.outcomes = {qp_name: ("successful" if fr.success else "failed") for qp_name, fr in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot conditional phase and leakage heatmaps with optimal point."""
    fig = plot_snz_conditional_phase(
        node.results["ds_fit"],
        node.namespace["qubit_pairs"],
        fit_results=node.results["fit_results"],
    )
    plt.show()
    node.results["figures"] = {"snz_conditional_phase": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No state update for now."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
