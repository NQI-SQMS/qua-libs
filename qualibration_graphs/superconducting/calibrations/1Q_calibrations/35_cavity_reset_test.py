# %% {Imports}
import matplotlib.pyplot as plt
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from calibration_utils.shared import (
    apply_confusion_matrix_correction,
    _get_cavity_mode,
    _get_pair,
    _get_transition_rf,
    _ef_if_at_fock,
    _ge_if_at_fock,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.cavity_reset_test import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cavity_reset_test.parameters import Parameters

# %% {Node initialisation}
description = """
        CAVITY ACTIVE RESET TEST
Characterises the effectiveness of the sideband active reset by measuring
P(0) = 1 − P(|e⟩) as a function of the reset drive flat-top duration.

Protocol (per reset-duration point):
  1. Pre-reset cavity and wait qubit thermalization.
  2. Prepare Fock |1⟩ via sideband ladder:
       ge pi → ef pi → f0g1 sideband pi
  3. Drive the sideband reset for a variable flat-top duration t.
  4. Inverse sideband readout:
       f0g1 sideband pi → ef pi → measure qubit ge
       P(|e⟩) = photon survived; P(|g⟩) = cavity reset to |0⟩
  5. Measure and save qubit state.

P(0) = 1 − P(|e⟩) vs t should rise from ≈0 to ≈1.  The duration at which
P(0) first exceeds 0.95 (t95) is reported.

Prerequisites:
  - Calibrated f0g1 sideband operations on the CavityTransmonPair
    sideband_drive (nodes 26, 26b).
  - Calibrated EF_x180 pulse.

Parameters:
  - mode_name:                    'alice' or 'bob'
  - reset_duration_start_in_ns:   start of flat-top sweep [ns]
  - reset_duration_end_in_ns:     end of flat-top sweep [ns]
  - reset_duration_step_in_ns:    step size [ns]
  - cavity_pre_reset_type:        'thermal' or 'active_sideband'

No QuAM state update (characterisation node only).
"""

node = QualibrationNode[Parameters, Quam](
    name="35_cavity_reset_test",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.reset_duration_end_in_ns = 2_000_000
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots

    # Build reset duration array in ns, then convert to clock cycles (÷4)
    t_start = node.parameters.reset_duration_start_in_ns
    t_end = node.parameters.reset_duration_end_in_ns
    t_step = node.parameters.reset_duration_step_in_ns
    reset_durations_ns = np.arange(t_start, t_end + 1, t_step, dtype=int)
    # Ensure multiples of 4 (QUA clock cycle requirement)
    reset_durations_ns = (reset_durations_ns // 4) * 4
    reset_durations_clk = reset_durations_ns // 4  # clock cycles
    node.namespace["reset_durations_ns"] = reset_durations_ns
    node.namespace["reset_durations_clk"] = reset_durations_clk

    cavity_mode = _get_cavity_mode(node)
    pair = _get_pair(node)

    sideband_drive = None
    chi_hz = 0.0
    if pair is not None:
        sideband_drive = getattr(pair, "sideband_drive", None)
        chi_hz = float(pair.chi) if getattr(pair, "chi", None) is not None else 0.0
    if sideband_drive is None:
        raise ValueError(
            "No sideband_drive found in the CavityTransmonPair. "
            "Run the f0g1 sideband calibration nodes (26 / 26b) first."
        )
    node.namespace["sideband_drive"] = sideband_drive

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "reset_duration": xr.DataArray(
            reset_durations_ns,
            attrs={"long_name": "reset flat-top duration", "units": "ns"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        t = declare(int)
        n_st = declare_stream()

        I, I_st, Q, Q_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_each_(t, reset_durations_clk.tolist()):
                    for i, qubit in multiplexed_qubits.items():
                        qubit_IF = int(qubit.xy.intermediate_frequency)

                        # Resolve f0g1 sideband parameters (cavity in |0⟩)
                        ef_if_0 = _ef_if_at_fock(pair, qubit, 0)
                        sb_rf_0 = _get_transition_rf(pair, sideband_drive, 0)
                        target_if_0 = int(sideband_drive.intermediate_frequency) + int(
                            sb_rf_0 - sideband_drive.RF_frequency
                        )
                        tr_0 = pair.transitions.get("f0g1")
                        override_ns = node.parameters.sideband_pulse_duration_ns
                        if override_ns is not None:
                            flat_top_clk_0 = override_ns // 4
                        elif tr_0 and tr_0.pi_flat_top_length_ns:
                            flat_top_clk_0 = tr_0.pi_flat_top_length_ns // 4
                        else:
                            flat_top_clk_0 = None  # use default pulse length

                        # -- 1. Pre-reset cavity and wait qubit thermalization ----
                        cavity_mode.reset(
                            node.parameters.cavity_pre_reset_type,
                            node.parameters.simulate,
                            log_callable=node.log,
                            sideband_drive=sideband_drive,
                            qubit_thermalization_time=qubit.thermalization_time,
                            fock_n=node.parameters.cavity_active_cooling_fock_n,
                            sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                            chi_hz=chi_hz,
                            pair=pair,
                        )
                        qubit.xy.wait(qubit.thermalization_time // 4)

                        # -- 2. Fock |1⟩ prep: ge pi → ef pi → f0g1 pi -----------
                        align(qubit.xy.name, sideband_drive.name, qubit.resonator.name)
                        qubit.xy.update_frequency(qubit_IF)
                        qubit.xy.play("x180")          # ge pi  →  |e,0⟩
                        qubit.xy.update_frequency(ef_if_0)
                        qubit.xy.play("EF_x180")       # ef pi  →  |f,0⟩
                        align(qubit.xy.name, sideband_drive.name)
                        sideband_drive.update_frequency(target_if_0)
                        with strict_timing_():
                            sideband_drive.play("sideband_ramp_up")
                            if flat_top_clk_0 is not None:
                                sideband_drive.play("sideband_square", duration=flat_top_clk_0)
                            else:
                                sideband_drive.play("sideband_square")
                            sideband_drive.play("sideband_ramp_down")
                        # Cavity now in |1⟩, qubit back to |g⟩
                        align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)

                        # -- 3. Apply reset drive for variable duration t ---------
                        sideband_drive.update_frequency(target_if_0)
                        with strict_timing_():
                            sideband_drive.play("sideband_ramp_up")
                            sideband_drive.play("sideband_square", duration=t)  # swept
                            sideband_drive.play("sideband_ramp_down")
                        align(sideband_drive.name, qubit.xy.name, qubit.resonator.name)

                        # -- 4. Inverse sideband readout --------------------------
                        # |g,1⟩ → |f,0⟩ → |e,0⟩  if photon survived (reset failed)
                        # |g,0⟩ unchanged            if reset succeeded
                        sideband_drive.update_frequency(target_if_0)
                        with strict_timing_():
                            sideband_drive.play("sideband_ramp_up")
                            if flat_top_clk_0 is not None:
                                sideband_drive.play("sideband_square", duration=flat_top_clk_0)
                            else:
                                sideband_drive.play("sideband_square")
                            sideband_drive.play("sideband_ramp_down")
                        align(sideband_drive.name, qubit.xy.name)
                        qubit.xy.update_frequency(ef_if_0)
                        qubit.xy.play("EF_x180")
                        qubit.xy.update_frequency(qubit_IF)

                        # -- 5. Measure -------------------------------------------
                        align(qubit.xy.name, qubit.resonator.name)
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                        if node.parameters.use_state_discrimination:
                            assign(
                                state[i],
                                Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold),
                            )
                            save(state[i], state_st[i])
                        qubit.resonator.wait(qubit.resonator.depletion_time // 4)

                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(reset_durations_ns)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(reset_durations_ns)).average().save(f"Q{i + 1}")
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(reset_durations_ns)).average().save(f"state{i + 1}")


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
                data_fetcher["n"],
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
    if node.parameters.use_state_discrimination and node.parameters.use_confusion_matrix_correction:
        node.results["ds_raw"] = apply_confusion_matrix_correction(
            node.results["ds_raw"], node.namespace["qubits"]
        )
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        q_name: ("successful" if res["success"] else "failed")
        for q_name, res in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    fig = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fit_results=node.results["fit_results"],
        mode_name=node.parameters.mode_name,
    )
    plt.show()
    node.results["figures"] = {"cavity_reset_test": fig}


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
